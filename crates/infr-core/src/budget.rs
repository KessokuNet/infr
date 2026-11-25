//! Shared KV-cache / VRAM-budget arithmetic.
//!
//! Deciding *where* a KV cache or a weight bank lives is two separable things: the device half
//! (`vkCreateBuffer`, the memory type, the arena, the staging ring) and a
//! device-INDEPENDENT half — how many bytes a KV format costs, which `INFR_*` knob spells what,
//! the cumulative-cap gate the VRAM-first spill applies, and the one-shot placement banner. The
//! second half was re-derived per backend (and, inside `infr-vulkan`, per multi-device wrapper)
//! and drifted. This module hosts it ONCE.
//!
//! What is here:
//!
//! - [`parse_kv_dtype`] — the `INFR_KV_TYPE_K` / `INFR_KV_TYPE_V` name grammar (llama's
//!   `--cache-type-k` / `--cache-type-v`), the single spelling table.
//! - [`kv_fmt_bytes`] — the EXACT allocation size of a KV buffer of `elems` elements, and
//!   [`kv_bytes_per_elem`] — the same thing as a fractional per-element rate, which is what the
//!   context-fit ESTIMATES divide by. The two must agree; a test pins them against each other.
//! - [`flag_from`] / [`reserve_bytes`] — the grammar and accessor the KV knobs share (`empty`/`0`
//!   = off; the 12%-of-VRAM-floored-at-2-GiB reserve). The VALUES come from the backend's [`Config`](crate::config::Config); nothing here
//!   reads the environment (R3).
//! - [`SpillTally`] / [`SpillCounts`] / [`spill_report_line`] — the VRAM-first placement
//!   bookkeeping and its banner. The counters and the message SKELETON are shared; every noun and
//!   every device-specific clause is passed in ([`SpillNouns`]), and so is the byte formatter,
//!   because a backend's MiB rounding is user-visible text that must not move.
//!
//! What is deliberately NOT here (device facts that only look shared):
//!
//! - The placement itself. Whether VRAM has room is a device question (Vulkan's live
//!   `VK_EXT_memory_budget` heap probe minus a guard headroom, in `kv_stays_in_vram`). Only the
//!   cumulative *cap* gate ([`SpillTally::admits`]), which is pure arithmetic over the tally, is
//!   shared.
//! - The human-readable byte formatter. Vulkan prints MiB to one decimal; unifying the spellings
//!   would move user-visible text for no behavioral gain, so [`spill_report_line`] takes the
//!   formatter as a parameter instead.
//! - The per-format GATES (`kv_align_ok`, which backends have a Q8 KV kernel, TurboQuant's
//!   `head_dim % 128`). Those are capabilities, and they stay at the seam that knows the backend.

use crate::decode_spec::block_layout;
use crate::tensor::DType;
use crate::SizeSpec;
use std::sync::atomic::{AtomicU64, Ordering};

// ── KV format sizing ─────────────────────────────────────────────────────────

/// The `INFR_KV_TYPE_K` / `INFR_KV_TYPE_V` name grammar (llama.cpp's `--cache-type-k` /
/// `--cache-type-v`): a spelling → KV dtype, or `None` for unset/unrecognized, which every caller
/// treats as "fall through to the default ladder".
///
/// This is the ONLY table of KV format names. The runner turns the dtype into an allocation after
/// applying its backend/alignment gates; the placement estimator turns the same dtype into a
/// bytes-per-element rate via [`kv_bytes_per_elem`]. Both used to spell the names themselves, so a
/// format added to one and not the other was priced as f16 while being allocated as something
/// else.
pub fn parse_kv_dtype(name: &str) -> Option<DType> {
    Some(match name {
        // TurboQuant: WHT-rotated, 128-elem blocks.
        "turbo2" => DType::Turbo2,
        "turbo3" => DType::Turbo3,
        "turbo4" => DType::Turbo4,
        "q8_0" | "q8" | "Q8_0" => DType::Q8_0,
        // Mainline low-bit block quants (32-elem blocks).
        "q4_0" => DType::Q4_0,
        "q4_1" => DType::Q4_1,
        "q5_0" => DType::Q5_0,
        "q5_1" => DType::Q5_1,
        "iq4_nl" => DType::Iq4Nl,
        // Dense.
        "bf16" => DType::Bf16,
        "f32" => DType::F32,
        "f16" | "F16" => DType::F16,
        _ => return None,
    })
}

/// EXACT byte size of one KV-cache buffer holding `elems` elements in `dt` — the number the runner
/// hands `Backend::alloc`, so it is also what the VRAM budget guard sees.
///
/// Q8_0 additionally rounds up to a `u32` multiple: the GPU KV kernels address the cache as
/// `uint`s, and a 34-B-per-block cache is only 2-byte aligned at odd block counts. The other block
/// formats have even block sizes, so their products are already aligned.
///
/// Anything not a KV format falls to `elems * 4` (f32), which is what the seam has always done —
/// the KV dtype is chosen by [`parse_kv_dtype`], so the fallthrough is only ever reached for F32
/// itself.
pub fn kv_fmt_bytes(dt: DType, elems: usize) -> usize {
    match dt {
        DType::Q8_0 => (elems / 32 * 34).next_multiple_of(4),
        // TurboQuant 128-elem blocks: turbo2 = 34 B, turbo3 = 50 B, turbo4 = 66 B.
        DType::Turbo2 => elems / 128 * 34,
        DType::Turbo3 => elems / 128 * 50,
        DType::Turbo4 => elems / 128 * 66,
        // Mainline low-bit KV quants (32-elem blocks) + bf16.
        DType::Q4_0 | DType::Iq4Nl => elems / 32 * 18,
        DType::Q4_1 => elems / 32 * 20,
        DType::Q5_0 => elems / 32 * 22,
        DType::Q5_1 => elems / 32 * 24,
        DType::F16 | DType::Bf16 => elems * 2,
        _ => elems * 4, // F32
    }
}

/// KV bytes per ELEMENT in `dt`, as a fraction (e.g. Q8_0 = 34/32 = 1.0625).
///
/// The context-fit estimators need a rate, not a size: they divide a free-byte budget by
/// "bytes per token across all layers" to solve for a context length, which the exact
/// [`kv_fmt_bytes`] (with its per-buffer flooring and alignment rounding) cannot express. It is
/// nonetheless the SAME number — read straight off [`block_layout`], the one decode spec, instead
/// of a second hand-written table of ratios — and `kv_fmt_bytes_matches_the_per_elem_rate` pins
/// that the two agree on every whole block.
pub fn kv_bytes_per_elem(dt: DType) -> f64 {
    let (elems, bytes) = block_layout(dt);
    bytes as f64 / elems as f64
}

// ── grammars + accessors shared by the spill knobs ───────────────────────────

/// The boolean-flag grammar the `kv.paged` knob uses, over an explicit value: `Some(v)` with `v`
/// neither empty nor `"0"` ⇒ on; unset ⇒ off. (`INFR_KV_PAGED`.)
/// NOT the same as the `is_ok()` presence grammar, where an empty
/// value is ON — `config::env`'s `flag` reader calls this so the difference survives the
/// migration; the read sites take the resolved `bool` off their backend's `Config`.
pub fn flag_from(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| !v.is_empty() && v != "0")
}

/// VRAM headroom a VRAM-first spill keeps free below the live free-byte figure, so the consumers
/// that allocate AFTER the spilled class (per-forward activation scratch, dequant caches, BLAS
/// workspace, a paged-MoE arena) still have room. Default 12% of total VRAM floored at 2 GiB —
/// enough for a several-hundred-row prefill on a large-vocab model.
///
/// The override has no config spelling yet: the VRAM guard keeps its own fixed headroom, and no
/// knob reaches either number.
///
/// Over-reserving costs residency; under-reserving OOMs mid-forward in an allocator that is
/// infallible by contract, so the floor is deliberate.
pub fn reserve_bytes(total_vram: u64, reserve: Option<SizeSpec>) -> u64 {
    match reserve {
        Some(SizeSpec::Bytes(bytes)) => bytes,
        Some(SizeSpec::Percent(share)) => (total_vram as f64 * share) as u64,
        None => (total_vram / 100 * 12).max(2 * 1024 * 1024 * 1024),
    }
}

// ── VRAM-first spill bookkeeping + banner ────────────────────────────────────

/// Resident/spilled counters for one VRAM-first placement class (the KV cache, or the dense
/// weight banks). Bumped from `alloc` on whichever arm won, drained once by the banner.
///
/// All-zero (the `Default`) means the flag was off or nothing of that class was allocated — both
/// print nothing.
///
/// Beside those lifetime totals it keeps LIVE occupancy per memory ([`SpillTally::claim`]): bytes
/// reserved before an allocation and released when the buffer goes, which is what a tier's `size`
/// limits.
#[derive(Debug, Default)]
pub struct SpillTally {
    vram_bufs: AtomicU64,
    vram_bytes: AtomicU64,
    host_bufs: AtomicU64,
    host_bytes: AtomicU64,
    live_vram_bytes: AtomicU64,
    live_host_bytes: AtomicU64,
}

/// The memory a KV buffer's bytes occupy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMemory {
    /// Device-local memory.
    Device,
    /// Host memory the kernels address directly.
    Host,
}

/// Bytes reserved in one memory by [`SpillTally::claim`], released on drop unless [`keep`](Self::keep)
/// hands them to the buffer that now holds them.
#[derive(Debug)]
#[must_use = "an unused claim releases its bytes at once"]
pub struct KvClaim<'a> {
    tally: &'a SpillTally,
    memory: KvMemory,
    bytes: u64,
}

impl KvClaim<'_> {
    /// Keep the bytes held; the caller releases them with [`SpillTally::release`].
    pub fn keep(self) -> (KvMemory, u64) {
        let held = (self.memory, self.bytes);
        std::mem::forget(self);
        held
    }
}

impl Drop for KvClaim<'_> {
    fn drop(&mut self) {
        self.tally.release(self.memory, self.bytes);
    }
}

/// A [`SpillTally`] read at one instant — plain values, so the banner formats from a consistent
/// snapshot instead of four independent loads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpillCounts {
    pub vram_bufs: u64,
    pub vram_bytes: u64,
    pub host_bufs: u64,
    pub host_bytes: u64,
    /// Bytes of KV buffers alive in device memory now.
    pub live_vram_bytes: u64,
    /// Bytes of KV buffers alive in host memory now.
    pub live_host_bytes: u64,
}

impl SpillCounts {
    /// Buffers placed either way. `0` ⇒ nothing to report.
    pub fn total_bufs(&self) -> u64 {
        self.vram_bufs + self.host_bufs
    }
}

impl SpillTally {
    /// Record one buffer that stayed resident in device-local VRAM.
    pub fn record_vram(&self, bytes: u64) {
        self.vram_bufs.fetch_add(1, Ordering::Relaxed);
        self.vram_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record one buffer that spilled to host RAM.
    pub fn record_host(&self, bytes: u64) {
        self.host_bufs.fetch_add(1, Ordering::Relaxed);
        self.host_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Snapshot for the banner.
    pub fn counts(&self) -> SpillCounts {
        SpillCounts {
            vram_bufs: self.vram_bufs.load(Ordering::Relaxed),
            vram_bytes: self.vram_bytes.load(Ordering::Relaxed),
            host_bufs: self.host_bufs.load(Ordering::Relaxed),
            host_bytes: self.host_bytes.load(Ordering::Relaxed),
            live_vram_bytes: self.live_vram_bytes.load(Ordering::Relaxed),
            live_host_bytes: self.live_host_bytes.load(Ordering::Relaxed),
        }
    }

    fn live(&self, memory: KvMemory) -> &AtomicU64 {
        match memory {
            KvMemory::Device => &self.live_vram_bytes,
            KvMemory::Host => &self.live_host_bytes,
        }
    }

    /// Reserve `bytes` of live occupancy in `memory` BEFORE allocating them, atomically against
    /// every other claim, so racing allocations cannot both pass a limit one of them would break.
    /// `cap` is `None` for no limit. A refusal returns the bytes held at that moment and reserves
    /// nothing; the reservation is returned when the claim drops unless it is kept.
    pub fn claim(
        &self,
        memory: KvMemory,
        bytes: u64,
        cap: Option<u64>,
    ) -> Result<KvClaim<'_>, u64> {
        let live = self.live(memory);
        let mut held = live.load(Ordering::Acquire);
        loop {
            let next = held.saturating_add(bytes);
            if cap.is_some_and(|cap| next > cap) {
                return Err(held);
            }
            match live.compare_exchange_weak(held, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    return Ok(KvClaim {
                        tally: self,
                        memory,
                        bytes,
                    });
                }
                Err(now) => held = now,
            }
        }
    }

    /// Return bytes a kept claim reserved, when the buffer that held them is gone.
    pub fn release(&self, memory: KvMemory, bytes: u64) {
        let clamped = self.release_clamped(memory, bytes);
        debug_assert!(!clamped, "released more KV bytes than were held");
    }

    /// Subtract without wrapping; `true` when `bytes` exceeded what was held and the tally clamped to zero.
    fn release_clamped(&self, memory: KvMemory, bytes: u64) -> bool {
        let before = self
            .live(memory)
            .try_update(Ordering::AcqRel, Ordering::Acquire, |held| {
                Some(held.saturating_sub(bytes))
            })
            .unwrap_or_else(|held| held);
        before < bytes
    }

    /// The diagnostic CUMULATIVE-cap gate: may `bytes` more still be placed in VRAM?
    ///
    /// `cap` is `None` for the normal case (no cap — spill only when VRAM is genuinely full);
    /// `Some(0)` forces the whole class to host, which is how the whole-host path stays
    /// reproducible on a model that would otherwise fit entirely. The cap gates PLACEMENT only,
    /// never the real VRAM guard.
    ///
    /// The running total + `bytes` SATURATES: a wrapped
    /// sum is a SMALL sum, and a small sum passes a cap it should have failed — the tally would
    /// keep admitting buffers into VRAM after the cap was blown. Saturating turns the degenerate
    /// input into a refusal (spill to host), which is the safe side of this gate. `u64::MAX` bytes
    /// is not a real allocation, so no honest caller is affected either way.
    pub fn admits(&self, cap: Option<u64>, bytes: u64) -> bool {
        match cap {
            None => true,
            Some(cap) => {
                self.vram_bytes
                    .load(Ordering::Relaxed)
                    .saturating_add(bytes)
                    <= cap
            }
        }
    }
}

/// The per-class nouns of the placement banner — everything [`spill_report_line`] cannot know.
#[derive(Debug, Clone, Copy)]
pub struct SpillNouns<'a> {
    /// The config section that enabled the placement, printed as the line's tag (`kv.window`).
    pub env: &'a str,
    /// Plural noun for the counted buffers (`"KV buffers"`, `"weight banks"`).
    pub noun: &'a str,
    /// Clause closing the all-resident line, after `"none spilled; "` (`"no PCIe KV reads."`).
    pub resident_note: &'a str,
    /// Clause closing the partial-spill line, after `"in "` — the placement AND who reads it over
    /// PCIe (`"SYSTEM RAM — attention reads those K/V over PCIe ..."`). Device-specific.
    pub spill_note: &'a str,
}

/// The one-shot VRAM-first placement banner, or `None` when nothing of the class was allocated.
///
/// Two shapes: everything stayed resident (worth saying — it means no PCIe reads at all), or the
/// cache split, in which case the counts on BOTH sides are named so a partial spill is visible
/// rather than looking like a slow run. Callers gate on their own enable flag and `eprintln!` the
/// result; keeping this a pure `String` builder is what lets a test pin the text.
pub fn spill_report_line(
    c: SpillCounts,
    n: &SpillNouns<'_>,
    fmt_bytes: impl Fn(u64) -> String,
) -> Option<String> {
    let total = c.total_bufs();
    if total == 0 {
        return None;
    }
    let (env, noun) = (n.env, n.noun);
    Some(if c.host_bufs == 0 {
        format!(
            "[infr] {env}: all {total} {noun} ({}) fit in VRAM — none spilled; {}",
            fmt_bytes(c.vram_bytes),
            n.resident_note,
        )
    } else {
        format!(
            "[infr] {env}: {} of {total} {noun} ({} resident) in VRAM, the remaining {} ({}) in {}",
            c.vram_bufs,
            fmt_bytes(c.vram_bytes),
            c.host_bufs,
            fmt_bytes(c.host_bytes),
            n.spill_note,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every KV spelling the two pre-hoist tables (the runner's `parse_kv_fmt` and the
    /// estimator's `kv_bytes_per_elem`) accepted, and the dtype each produced. A name dropped here
    /// silently reverts that format to f16 at runtime.
    #[test]
    fn kv_dtype_names_match_the_inline_tables() {
        for (name, want) in [
            ("f32", DType::F32),
            ("bf16", DType::Bf16),
            ("f16", DType::F16),
            ("F16", DType::F16),
            ("q8_0", DType::Q8_0),
            ("q8", DType::Q8_0),
            ("Q8_0", DType::Q8_0),
            ("q4_0", DType::Q4_0),
            ("q4_1", DType::Q4_1),
            ("q5_0", DType::Q5_0),
            ("q5_1", DType::Q5_1),
            ("iq4_nl", DType::Iq4Nl),
            ("turbo2", DType::Turbo2),
            ("turbo3", DType::Turbo3),
            ("turbo4", DType::Turbo4),
        ] {
            assert_eq!(parse_kv_dtype(name), Some(want), "{name}");
        }
        // Unset / misspelled / a weight-only quant / case variants that were never accepted all
        // fall through to the caller's default ladder.
        for name in ["", "q4_k", "Q4_0", "F32", "BF16", "turbo", "q8_1", "f 16"] {
            assert_eq!(parse_kv_dtype(name), None, "{name}");
        }
    }

    /// The per-element rates, verbatim from the pre-hoist `kv_bytes_per_elem` table in
    /// `infr-llama`'s `seam/model.rs`. These divide the free-VRAM budget in the default-ctx clamp,
    /// so a wrong one moves the context a user gets.
    #[test]
    fn kv_bytes_per_elem_matches_the_inline_table() {
        for (dt, want) in [
            (DType::F32, 4.0),
            (DType::Bf16, 2.0),
            (DType::F16, 2.0),
            (DType::Q8_0, 34.0 / 32.0),
            (DType::Q4_0, 18.0 / 32.0),
            (DType::Iq4Nl, 18.0 / 32.0),
            (DType::Q4_1, 20.0 / 32.0),
            (DType::Q5_0, 22.0 / 32.0),
            (DType::Q5_1, 24.0 / 32.0),
            (DType::Turbo2, 34.0 / 128.0),
            (DType::Turbo3, 50.0 / 128.0),
            (DType::Turbo4, 66.0 / 128.0),
        ] {
            // Bit-exact, not approximate: both sides are the same ratio of small integers, and an
            // estimator that drifts by an ULP is an estimator that was recomputed differently.
            assert_eq!(kv_bytes_per_elem(dt), want, "{dt:?}");
        }
    }

    /// The exact allocation sizes, verbatim from the pre-hoist `kv_fmt_bytes` in `infr-llama`'s
    /// `seam/mod.rs` (and the copy in `infr-metal`'s parity suite).
    #[test]
    fn kv_fmt_bytes_matches_the_inline_table() {
        let e = 4096usize;
        for (dt, want) in [
            (DType::Q8_0, (e / 32 * 34usize).next_multiple_of(4)),
            (DType::Turbo2, e / 128 * 34),
            (DType::Turbo3, e / 128 * 50),
            (DType::Turbo4, e / 128 * 66),
            (DType::Q4_0, e / 32 * 18),
            (DType::Iq4Nl, e / 32 * 18),
            (DType::Q4_1, e / 32 * 20),
            (DType::Q5_0, e / 32 * 22),
            (DType::Q5_1, e / 32 * 24),
            (DType::F16, e * 2),
            (DType::Bf16, e * 2),
            (DType::F32, e * 4),
        ] {
            assert_eq!(kv_fmt_bytes(dt, e), want, "{dt:?}");
        }
        // The Q8_0 u32 rounding is load-bearing at an odd block count: 33 blocks x 34 B = 1122,
        // which is not a u32 multiple.
        assert_eq!(kv_fmt_bytes(DType::Q8_0, 33 * 32), 1124);
        assert_eq!(kv_fmt_bytes(DType::Q8_0, 32 * 32), 32 * 34);
        // Sub-block element counts floor to zero bytes, as they always have (the runner never
        // allocates one — every KV row is gate-checked for block alignment first).
        assert_eq!(kv_fmt_bytes(DType::Q8_0, 31), 0);
        assert_eq!(kv_fmt_bytes(DType::Turbo3, 127), 0);
    }

    /// The exact sizer and the estimator rate are two views of one number: on any whole-block
    /// count they must agree (modulo Q8_0's u32 rounding, which only ever rounds UP).
    #[test]
    fn kv_fmt_bytes_matches_the_per_elem_rate() {
        for dt in [
            DType::F32,
            DType::F16,
            DType::Bf16,
            DType::Q8_0,
            DType::Q4_0,
            DType::Q4_1,
            DType::Q5_0,
            DType::Q5_1,
            DType::Iq4Nl,
            DType::Turbo2,
            DType::Turbo3,
            DType::Turbo4,
        ] {
            for blocks in [1usize, 3, 128, 1024] {
                let elems = blocks * 128; // a multiple of both block sizes
                let exact = kv_fmt_bytes(dt, elems) as f64;
                let rate = kv_bytes_per_elem(dt) * elems as f64;
                assert!(
                    exact >= rate && exact - rate < 4.0,
                    "{dt:?} x {elems}: exact {exact} vs rate {rate}"
                );
            }
        }
    }

    /// `empty`/`0`/unset = off; anything else = on — over explicit values via [`flag_from`], so
    /// no test in this binary has to touch the process environment.
    #[test]
    fn flag_grammar() {
        assert!(!flag_from(None));
        for (raw, want) in [
            ("", false),
            ("0", false),
            ("1", true),
            ("00", true), // only the exact string "0" is off — matches the shipped grammar
            ("true", true),
            ("no", true),
        ] {
            assert_eq!(flag_from(Some(raw)), want, "raw={raw:?}");
        }
    }

    /// The same grammar as the CONFIG layer resolves it: `config::env`'s `flag` reader is
    /// `flag_from`, so a `Config` built from these spellings carries exactly the booleans above.
    #[test]
    fn kv_paged_config_field_follows_the_flag_grammar() {
        use crate::config::Config;
        for (raw, want) in [("", false), ("0", false), ("1", true), ("true", true)] {
            let key = "INFR_KV_PAGED";
            let layer =
                crate::config::env::parse(&|k| (k == key).then(|| raw.to_string())).expect(key);
            let cfg = Config::load_from_layers(&[layer]);
            assert_eq!(cfg.kv.paged, Some(want), "{key}={raw:?}");
        }
        // Unset ⇒ the default, which paging reads as on.
        assert_eq!(Config::default().kv.paged, None);
    }

    /// 12% of total, floored at 2 GiB, the override in bytes or as a share of total VRAM.
    #[test]
    fn reserve_bytes_matches_the_inline_formula() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // Below the crossover (2 GiB / 0.12 ≈ 16.67 GiB) the floor wins.
        assert_eq!(reserve_bytes(8 * GIB, None), 2 * GIB);
        assert_eq!(reserve_bytes(16 * GIB, None), 2 * GIB);
        // Above it, 12% — computed as `total / 100 * 12`, the shipped (truncating) order.
        assert_eq!(reserve_bytes(24 * GIB, None), 24 * GIB / 100 * 12);
        assert_eq!(reserve_bytes(0, None), 2 * GIB);
        // The override wins outright, including below the floor and at zero.
        let bytes = |b| Some(SizeSpec::Bytes(b));
        assert_eq!(reserve_bytes(24 * GIB, bytes(128 << 20)), 128 << 20);
        assert_eq!(reserve_bytes(24 * GIB, bytes(0)), 0);
        assert_eq!(
            reserve_bytes(24 * GIB, Some(SizeSpec::Percent(0.5))),
            12 * GIB
        );
    }

    /// A saturated cap behaves as "no cap" at the gate it feeds: no real byte count reaches it.
    #[test]
    fn a_huge_cap_admits_every_real_byte_count() {
        let t = SpillTally::default();
        t.record_vram(1 << 40);
        assert!(t.admits(Some(u64::MAX), 1 << 40));
    }

    /// The cumulative cap admits up to and including the cap, and `Some(0)` admits nothing —
    /// the whole-host reproducibility case.
    #[test]
    fn spill_tally_cap_boundaries() {
        let t = SpillTally::default();
        assert!(t.admits(None, u64::MAX));
        assert!(t.admits(Some(0), 0));
        assert!(!t.admits(Some(0), 1));
        assert!(t.admits(Some(100), 100)); // exactly the cap fits
        assert!(!t.admits(Some(100), 101));

        t.record_vram(60);
        assert!(t.admits(Some(100), 40));
        assert!(!t.admits(Some(100), 41));
        t.record_vram(40);
        assert!(t.admits(Some(100), 0));
        assert!(!t.admits(Some(100), 1));
        // Host placements never consume the VRAM cap.
        t.record_host(1 << 40);
        assert!(t.admits(Some(200), 100));

        let c = t.counts();
        assert_eq!(
            c,
            SpillCounts {
                vram_bufs: 2,
                vram_bytes: 100,
                host_bufs: 1,
                host_bytes: 1 << 40,
                ..Default::default()
            }
        );
        assert_eq!(c.total_bufs(), 3);
    }

    /// A claim holds its bytes until released, a refusal holds nothing, and a claim dropped
    /// unkept gives its bytes back, so a tier fills, frees and fills again.
    #[test]
    fn claims_gate_live_bytes_and_release_them() {
        let t = SpillTally::default();
        let first = t.claim(KvMemory::Host, 4, Some(8)).unwrap();
        let (memory, bytes) = t.claim(KvMemory::Host, 4, Some(8)).unwrap().keep();
        assert_eq!(t.counts().live_host_bytes, 8);
        assert_eq!(t.claim(KvMemory::Host, 1, Some(8)).unwrap_err(), 8);
        assert_eq!(t.counts().live_host_bytes, 8, "a refusal reserves nothing");
        assert!(t.claim(KvMemory::Device, 100, Some(8)).is_err());
        assert_eq!(t.counts().live_vram_bytes, 0, "the tiers count apart");

        drop(first);
        t.release(memory, bytes);
        assert_eq!(t.counts().live_host_bytes, 0);
        drop(
            t.claim(KvMemory::Host, 8, Some(8))
                .expect("freed bytes are reusable"),
        );
        assert_eq!(t.counts().live_host_bytes, 0);
        assert!(t.claim(KvMemory::Host, u64::MAX, None).is_ok());
    }

    /// Threads racing for a tier never hold more than its size between them.
    #[test]
    fn concurrent_claims_never_pass_the_cap() {
        let t = SpillTally::default();
        let kept = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..16 {
                s.spawn(|| {
                    if let Ok(claim) = t.claim(KvMemory::Device, 4, Some(8)) {
                        claim.keep();
                        kept.fetch_add(4, Ordering::Relaxed);
                    }
                });
            }
        });
        assert_eq!(kept.into_inner(), 8);
        assert_eq!(t.counts().live_vram_bytes, 8);
    }

    /// The cap gate's `resident + bytes` must SATURATE. A wrapped sum is a small sum, and a small
    /// sum passes a cap it should have failed — the tally would go on admitting buffers into VRAM
    /// after the cap was already blown (and in debug it panics outright inside a placement
    /// decision). Saturating refuses, which is the safe side: the buffer spills to host.
    #[test]
    fn spill_tally_cap_saturates_instead_of_wrapping() {
        let t = SpillTally::default();
        t.record_vram(1 << 40);
        // `(1<<40) + u64::MAX` wraps to `(1<<40) - 1`, which would sneak under a modest cap.
        assert!(!t.admits(Some(1 << 41), u64::MAX));
        assert!(!t.admits(Some(u64::MAX - 1), u64::MAX));
        // An uncapped tally never consults the sum at all, so it still admits anything.
        assert!(t.admits(None, u64::MAX));
        // A cap of exactly `u64::MAX` (a saturated size) still admits, since
        // the saturated sum compares equal to it.
        assert!(t.admits(Some(u64::MAX), u64::MAX));
        // Right at the boundary with no saturation involved: the honest arithmetic is unchanged.
        let u = SpillTally::default();
        u.record_vram(u64::MAX - 10);
        assert!(u.admits(Some(u64::MAX), 10));
    }

    /// A release larger than the held bytes clamps to zero instead of wrapping past every cap.
    #[test]
    fn release_larger_than_held_clamps_to_zero() {
        let t = SpillTally::default();
        let claim = t.claim(KvMemory::Host, 10, Some(100)).unwrap();
        let (memory, _) = claim.keep();
        assert!(t.release_clamped(memory, 25));
        assert_eq!(t.counts().live_host_bytes, 0);
        assert!(t.claim(KvMemory::Host, 100, Some(100)).is_ok());
    }

    /// The banner, byte-for-byte against the pre-hoist `eprintln!` (Vulkan KV).
    #[test]
    fn spill_report_line_reproduces_the_inline_banners() {
        /// Vulkan's `fmt_bytes` (MiB to one decimal).
        fn vk_fmt(b: u64) -> String {
            const KIB: u64 = 1 << 10;
            const MIB: u64 = 1 << 20;
            const GIB: u64 = 1 << 30;
            match b {
                b if b >= GIB => format!("{:.2} GiB", b as f64 / GIB as f64),
                b if b >= MIB => format!("{:.1} MiB", b as f64 / MIB as f64),
                b if b >= KIB => format!("{:.1} KiB", b as f64 / KIB as f64),
                b => format!("{b} B"),
            }
        }
        const VK_KV: SpillNouns = SpillNouns {
            env: "kv.window",
            noun: "KV buffers",
            resident_note: "no PCIe KV reads.",
            spill_note: "SYSTEM RAM — attention reads those K/V over PCIe by device address \
                         (PCIe-bound on the spilled layers). Spilled KV bytes are exempt from the \
                         VRAM budget guard.",
        };

        // Nothing allocated (or the flag off, which leaves the tally at zero) ⇒ no line.
        assert_eq!(
            spill_report_line(SpillCounts::default(), &VK_KV, vk_fmt),
            None
        );

        let all_resident = SpillCounts {
            vram_bufs: 56,
            vram_bytes: 3 << 30,
            host_bufs: 0,
            host_bytes: 0,
            ..Default::default()
        };
        assert_eq!(
            spill_report_line(all_resident, &VK_KV, vk_fmt).unwrap(),
            "[infr] kv.window: all 56 KV buffers (3.00 GiB) fit in VRAM — none spilled; \
             no PCIe KV reads."
        );

        let split = SpillCounts {
            vram_bufs: 40,
            vram_bytes: 3 << 30,
            host_bufs: 16,
            host_bytes: 1 << 30,
            ..Default::default()
        };
        assert_eq!(
            spill_report_line(split, &VK_KV, vk_fmt).unwrap(),
            "[infr] kv.window: 40 of 56 KV buffers (3.00 GiB resident) in VRAM, the \
             remaining 16 (1.00 GiB) in SYSTEM RAM — attention reads those K/V over PCIe by \
             device address (PCIe-bound on the spilled layers). Spilled KV bytes are exempt from \
             the VRAM budget guard."
        );
    }
}
