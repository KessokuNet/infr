//! A `redb` entry leading `[[kv.tiers]]` makes the store the authoritative copy of the live
//! window, so the slot's rows are staging: every live KV buffer lands in host-visible memory and
//! none in device memory. That is the placement half of store-backed KV; the tokens and the
//! per-step store traffic are covered on CPU by `infr-llama`'s own tests.
//!
//! Run: `cargo test -p infr-vulkan --test kv_store_backed -- --ignored --nocapture`

use std::sync::Arc;

use infr_core::backend::{Backend, BufferUsage};
use infr_core::config::{Config, KvBackend, KvTier};
use infr_vulkan::VulkanBackend;

const MIB: usize = 1 << 20;

fn backend(configure: impl FnOnce(&mut Config)) -> Option<VulkanBackend> {
    let mut cfg = Config::default();
    configure(&mut cfg);
    infr_testkit::gpu_or_skip(VulkanBackend::new_with(Arc::new(cfg)))
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn a_leading_store_tier_places_the_live_kv_in_host_memory() {
    let Some(be) = backend(|c| {
        c.kv.tiers = vec![KvTier {
            backend: Some(KvBackend::Redb),
            size: Some(infr_core::SizeSpec::Bytes(64 * MIB as u64)),
            ..Default::default()
        }]
    }) else {
        return;
    };
    // The residency the engine runs under: no device share, the whole window on the host side.
    let residency = be.cfg().kv.residency().expect("a valid tier list");
    assert!(residency.store_backed);
    assert_eq!(residency.spill.and_then(|s| s.device_cap), Some(0));

    let data: Vec<u8> = (0..4 * MIB).map(|i| (i * 31 % 251) as u8).collect();
    let bufs: Vec<_> = (0..4)
        .map(|_| {
            let b = be
                .alloc(data.len(), BufferUsage::KvCache)
                .expect("kv alloc");
            be.upload(b.as_ref(), &data).expect("upload");
            b
        })
        .collect();
    let counts = be.kv_placement().expect("a spill plan keeps a tally");
    assert_eq!(
        (counts.vram_bufs, counts.vram_bytes),
        (0, 0),
        "a leading store tier puts no live KV in device memory"
    );
    assert_eq!(
        (counts.host_bufs, counts.host_bytes),
        (4, (4 * data.len()) as u64)
    );
    for b in &bufs {
        let mut back = vec![0u8; data.len()];
        be.download(b.as_ref(), &mut back).expect("download");
        assert_eq!(back, data, "a host-resident KV buffer round-trips");
    }
}
