//! A `host` entry in `[[kv.window]]` places the live slot's KV buffers in host memory the kernels
//! address directly, and a sized `vram` entry before it is the device part. These tests count where
//! `BufferUsage::KvCache` buffers land through the backend's placement tally.
//!
//! Run: `cargo test -p infr-vulkan --test kv_host_tier -- --ignored --nocapture`

use std::sync::Arc;

use infr_core::backend::{Backend, BufferUsage};
use infr_core::budget::SpillCounts;
use infr_core::config::{Config, KvBackend, WindowEntry};
use infr_vulkan::VulkanBackend;

const MIB: usize = 1 << 20;

fn window(backend: KvBackend, bytes: usize) -> WindowEntry {
    WindowEntry {
        backend: Some(backend),
        size: Some(infr_core::SizeSpec::Bytes(bytes as u64)),
    }
}

fn backend(configure: impl FnOnce(&mut Config)) -> Option<VulkanBackend> {
    let mut cfg = Config::default();
    configure(&mut cfg);
    infr_testkit::gpu_or_skip(VulkanBackend::new_with(Arc::new(cfg)))
}

/// Allocate `n` KV buffers of `size` bytes and return where they landed.
fn place(be: &VulkanBackend, n: usize, size: usize) -> SpillCounts {
    let bufs: Vec<_> = (0..n)
        .map(|_| be.alloc(size, BufferUsage::KvCache).expect("kv alloc"))
        .collect();
    let counts = be.kv_placement().expect("a window plan keeps a tally");
    drop(bufs);
    counts
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_host_first_window_allocates_no_device_kv() {
    let Some(be) = backend(|c| c.kv.window = vec![window(KvBackend::Host, 64 * MIB)]) else {
        return;
    };
    let data: Vec<u8> = (0..4 * MIB).map(|i| (i * 31 % 251) as u8).collect();
    let bufs: Vec<_> = (0..4)
        .map(|_| {
            let b = be.alloc(data.len(), BufferUsage::KvCache).expect("alloc");
            be.upload(b.as_ref(), &data).expect("upload");
            b
        })
        .collect();
    for b in &bufs {
        let mut back = vec![0u8; data.len()];
        be.download(b.as_ref(), &mut back).expect("download");
        assert_eq!(
            back, data,
            "a host-resident buffer reads back what was written"
        );
    }
    let counts = be.kv_placement().expect("tally");
    assert_eq!(
        (counts.vram_bufs, counts.vram_bytes),
        (0, 0),
        "no KV in device memory"
    );
    assert_eq!(
        (counts.host_bufs, counts.host_bytes),
        (4, (4 * data.len()) as u64)
    );
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_vram_then_host_window_splits_at_the_device_part() {
    let Some(be) = backend(|c| {
        c.kv.window = vec![
            window(KvBackend::Vram, 8 * MIB),
            window(KvBackend::Host, 64 * MIB),
        ];
    }) else {
        return;
    };
    let counts = place(&be, 4, 4 * MIB);
    assert_eq!((counts.vram_bufs, counts.vram_bytes), (2, 8 * MIB as u64));
    assert_eq!((counts.host_bufs, counts.host_bytes), (2, 8 * MIB as u64));
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_full_host_window_refuses_the_next_buffer() {
    let Some(be) = backend(|c| c.kv.window = vec![window(KvBackend::Host, 8 * MIB)]) else {
        return;
    };
    let _held: Vec<_> = (0..2)
        .map(|_| {
            be.alloc(4 * MIB, BufferUsage::KvCache)
                .expect("fits the window")
        })
        .collect();
    let err = be
        .alloc(4 * MIB, BufferUsage::KvCache)
        .err()
        .expect("past the window's size")
        .to_string();
    assert!(err.contains("host kv window"), "{err}");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_sized_vram_window_alone_has_nowhere_to_spill() {
    let Some(be) = backend(|c| c.kv.window = vec![window(KvBackend::Vram, 8 * MIB)]) else {
        return;
    };
    let _held: Vec<_> = (0..2)
        .map(|_| {
            be.alloc(4 * MIB, BufferUsage::KvCache)
                .expect("fits the window")
        })
        .collect();
    let err = be
        .alloc(4 * MIB, BufferUsage::KvCache)
        .err()
        .expect("past the window's size, with no host window")
        .to_string();
    assert!(err.contains("no host window"), "{err}");
}

fn live(be: &VulkanBackend) -> (u64, u64) {
    let counts = be.kv_placement().expect("a window plan keeps a tally");
    (counts.live_vram_bytes, counts.live_host_bytes)
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_dropped_buffer_returns_its_bytes_to_the_host_window() {
    let Some(be) = backend(|c| c.kv.window = vec![window(KvBackend::Host, 8 * MIB)]) else {
        return;
    };
    for round in 0..3 {
        let held = be
            .alloc(8 * MIB, BufferUsage::KvCache)
            .expect("fills the window");
        assert_eq!(live(&be), (0, 8 * MIB as u64), "round {round}");
        assert!(be.alloc(MIB, BufferUsage::KvCache).is_err(), "full");
        drop(held);
        assert_eq!(live(&be), (0, 0), "round {round}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_failed_session_init_gives_the_window_back() {
    let Some(be) = backend(|c| c.kv.window = vec![window(KvBackend::Host, 8 * MIB)]) else {
        return;
    };
    let init = |be: &VulkanBackend| -> Result<Vec<_>, _> {
        (0..3)
            .map(|_| be.alloc(4 * MIB, BufferUsage::KvCache))
            .collect()
    };
    assert!(init(&be).is_err(), "the third buffer passes the window");
    assert_eq!(live(&be), (0, 0), "the buffers of the failed init are gone");
    let bufs: Vec<_> = (0..2)
        .map(|_| be.alloc(4 * MIB, BufferUsage::KvCache).expect("retry fits"))
        .collect();
    assert_eq!(live(&be), (0, 8 * MIB as u64));
    drop(bufs);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_buffer_that_misses_the_device_remainder_goes_whole_to_the_host_window() {
    let Some(be) = backend(|c| {
        c.kv.window = vec![
            window(KvBackend::Vram, 6 * MIB),
            window(KvBackend::Host, 6 * MIB),
        ];
    }) else {
        return;
    };
    let first = be
        .alloc(4 * MIB, BufferUsage::KvCache)
        .expect("in the device part");
    assert_eq!(live(&be), (4 * MIB as u64, 0));
    let second = be
        .alloc(4 * MIB, BufferUsage::KvCache)
        .expect("misses the 2 MiB device remainder, fits the host window");
    let both = (4 * MIB as u64, 4 * MIB as u64);
    assert_eq!(live(&be), both, "placed whole in host, not split");
    let err = be
        .alloc(4 * MIB, BufferUsage::KvCache)
        .err()
        .expect("4 MiB fits neither remainder");
    assert!(err.to_string().contains("host kv window"), "{err}");
    assert_eq!(live(&be), both, "the refusal holds nothing");
    drop((first, second));
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn racing_allocations_never_pass_a_window_size() {
    let Some(be) = backend(|c| {
        c.kv.window = vec![
            window(KvBackend::Vram, 8 * MIB),
            window(KvBackend::Host, 8 * MIB),
        ];
    }) else {
        return;
    };
    let held = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..12 {
            s.spawn(|| {
                if let Ok(buf) = be.alloc(4 * MIB, BufferUsage::KvCache) {
                    held.lock().unwrap().push(buf);
                }
            });
        }
    });
    let counts = be.kv_placement().expect("tally");
    let admitted = held.lock().unwrap().len() as u64;
    assert!(admitted <= 4, "{admitted} buffers pass two 8 MiB windows");
    assert_eq!(
        counts.vram_bufs + counts.host_bufs,
        admitted,
        "counted once"
    );
    assert!(counts.vram_bufs <= 2 && counts.host_bufs <= 2, "{counts:?}");
    assert!(counts.live_vram_bytes <= 8 * MIB as u64, "{counts:?}");
    assert!(counts.live_host_bytes <= 8 * MIB as u64, "{counts:?}");
    assert_eq!(
        counts.live_vram_bytes + counts.live_host_bytes,
        admitted * 4 * MIB as u64
    );
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn the_device_cap_counts_live_bytes_not_lifetime_bytes() {
    let Some(be) = backend(|c| {
        c.kv.window = vec![
            window(KvBackend::Vram, 8 * MIB),
            window(KvBackend::Host, 1 << 30),
        ];
    }) else {
        return;
    };
    for round in 0..2 {
        let bufs: Vec<_> = (0..2)
            .map(|_| be.alloc(4 * MIB, BufferUsage::KvCache).expect("alloc"))
            .collect();
        assert_eq!(live(&be), (8 * MIB as u64, 0), "round {round}");
        drop(bufs);
    }
    let counts = be.kv_placement().expect("tally");
    assert_eq!(
        (counts.vram_bufs, counts.host_bufs),
        (4, 0),
        "the second round is admitted after the first was freed"
    );
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_malformed_window_section_fails_the_kv_allocation() {
    let Some(be) = backend(|c| c.kv.window = vec![window(KvBackend::Vram, 0)]) else {
        return;
    };
    let err = be
        .alloc(MIB, BufferUsage::KvCache)
        .err()
        .expect("a malformed window is not silently ignored")
        .to_string();
    assert!(err.contains("kv window"), "{err}");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn no_window_keeps_no_placement_tally() {
    let Some(be) = backend(|_| {}) else {
        return;
    };
    let _buf = be.alloc(MIB, BufferUsage::KvCache).expect("alloc");
    assert_eq!(be.kv_placement(), None);
}
