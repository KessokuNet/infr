# Plan 19: KV cache as a store, anywhere

Status: P1 shipped (`258a6e5f`, review fixes `ada1ebe4`); P2 done (the `vram` tier: config and sizing,
then its behaviour under pressure); P3 built and green on the CPU gate plus its Vulkan placement
tests, the bit-identity and decode numbers measured on the RX 580; P4
built and measured (the store-backed experiment; the cost table in `docs/perf/kv-prefix.md` says it
is not a speed play). Plan 18 unified where idle blocks
are kept; this makes the memory a slot's rows live in a tier like any other, with a `vram` tier as
the default rather than a hard-coded device buffer.

## Problem

After 18 a store can live anywhere except on the device. `[[kv.tiers]]` names `redb` and `redis`; a
live slot's rows are `SeamKv`'s device buffers, allocated `GpuOnly` by `BufferUsage::KvCache` with no
way to say otherwise; `DeviceKv` is a view over one slot rather than a store that holds many
extents. Two axes are tangled in that arrangement:

- **placement** — which store keeps idle blocks, unified in 18 behind `KvStore`;
- **residency** — where a live slot's bytes physically sit, hard-coded to device-local memory.

Left fused, three things stay impossible: an idle block cannot stay on the card, because `park_idle`
captures it out to the first tier and resets the slot; a slot cannot live in host memory
(`-nkvo`-style) for a model that barely fits; and the one escape hatch that exists, `kv.overflow`, is
a placement decision expressed as a boolean, which cannot also hold idle blocks or chain into a
storage tier.

## Constraint

Attention reads every row of the live window on every step, so residency has a bandwidth floor of
bytes per token times the window. With F16 K/V:

| model                       | KV per token | 8k working set | per step over PCIe ~25 GB/s | per step over loopback Redis ~3 GB/s |
| --------------------------- | ------------ | -------------- | --------------------------- | ------------------------------------ |
| Qwen2.5-0.5B (24L, 2 KV, 64) | 12 KB        | 100 MB         | 4 ms                        | 33 ms                                |
| 8B GQA (32L, 8 KV, 128)     | 128 KB       | 1.07 GB        | 43 ms                       | 357 ms                               |

Fair attention over 1 GbE puts the 8B row into seconds per token. So the live window's memory must
be device-addressable: either device-local VRAM, or host memory the kernels can address. A `redb` or
`redis` tier backs a restore, never a step — unless it leads the list, which is P4's experiment and
pays the numbers above per step.

## Design

**A tier names the memory.** Backends split into addressable (`vram`, `host`) and storage (`redb`,
`redis`). The first entry of `[[kv.tiers]]` is where a live slot's rows sit; storage tiers below it
receive what the addressable tier demotes. A storage backend may also be named first, and then it is
the slot's backing: the slot is host-resident and its window is synchronised with that store every
step. That is the experiment P4 measures — technically the same code path as a `host` tier, with the
store doing the reading and writing instead of a local buffer — not a configuration to recommend.

**The default is one `vram` tier, and it is implicit.** No tier list means `backend = "vram"` sized
from the device budget after weights and activations — exactly what a plain `infr run` allocates
today, so nothing changes for anyone who never writes a list. Writing `[[kv.tiers]] backend = "vram"`
with a `size` is how the same thing is said out loud and capped.

**Zero-copy is the retention path, not an optimisation of a copy.** A paged slot's rows are offsets
into buffers the pool owns: `PagedSlot::map` is `SlotMap { Arc<KvPool>, Vec<u32> }`, physical pages
carry refcounts, and `SlotMap::share_prefix` adopts another map's pages by cloning the table and
bumping the refs. So a parked conversation does not have to move at all: hand its `SlotMap` to a
resident entry keyed by the block chain it covers, keep the refs, and clear the slot's own map. A
restore that matches installs the entry with `share_prefix` and the rows are already where attention
reads them; the first write into a shared page pays one `cow_page`, a page copy rather than a prefix
copy. Nothing crosses the bus and nothing is decoded. This is P1 and it is shipped.

That makes the owner of the pages an L1 cache in VRAM, which is why it needs a budget and an LRU:
retained prefixes hold pages live slots would otherwise use. A page is retainable only when every
block it covers matched (512 rows against 128-token blocks), partial coverage falls through to the
storage tiers, and a retired entry is where the download happens — capture to the next tier, then
free the pages.

**The `vram` tier is a reservation beside the live windows.** A slot that is generating cannot have
its rows evicted underneath the kernels, so live windows are never in the tier's LRU: the tier holds
only retained idle pages. Its `size` is what `kv_fit_ctx` subtracts before dividing the remainder
among slots, and the paged pool's total is the bytes the slots' windows need plus the tier's `size`,
so a full window still leaves the tier's bytes for retention; the retained pages are capped at that
size. Retention itself copies nothing: the slot's page map moves into the resident entry. What a
retention does write is the storage tier below: `retain` captures blocks that tier lacks before the
pages can be evicted, so a cap eviction or a pool-pressure spill only drops the device copy, and a
later restore promotes the rows from the store into a live slot. A device-to-device demote
(`Backend::copy_buffer_at`, as the recurrent snapshot and the continuation seed use) exists only
once two addressable tiers do (P3); with one `vram` tier over a host-backed store a demote
necessarily crosses to host. A
configuration whose tier alone removes the requested window (an explicit `--ctx`, else one minimum
window, priced at the best KV format the session could pick) is refused at open with the tier size,
the window's bytes and what is left, not silently clamped; every open path runs the same check.

**Host residency is `backend = "host"`, the same shape.** Host-allocated KV buffers the kernels
address directly (`alloc_kv_host`, `INFR_KV_OVERFLOW`'s existing path), which is the `-nkvo`
analogue and the thing to reach for when a model leaves no room for a window. `kv.overflow` and
`kv.overflow_vram_mb` become the legacy spelling of a `vram` tier with a `size` followed by a `host`
tier, and keep working through a config migration. A tier is a cap on logical bytes, and the
allocator rounds each buffer up, so a host tier sized to the exact logical sum can still fail on
that rounding: leave a few MiB.

## Phases

- **P1** (done, `258a6e5f`, review fixes `ada1ebe4`) resident page retention: a `SlotMap` handed to
  an idle entry on park, restored through `share_prefix`, a page budget with LRU spill, and a retired
  entry captured to the store before its pages go back. Gate met: a paged resume after a park
  performs no upload (asserted with counters), produces identical rows, and the GPU park test
  resumes with no store restore.
- **P2** (done) the `vram` tier: `backend = "vram"` in `[[kv.tiers]]`, implicit as the default when
  the list is absent, a reservation beside the live windows (never in the LRU) holding retained idle
  pages with LRU eviction, and `kv_fit_ctx` priced from the tier `size`. Gate, met: the resident-page
  retention path performs no device-to-device copy, and its demote into a storage tier crosses to
  host (the "never stage through the host" clause applies once two addressable tiers exist, P3); a vram tier demotes into a `redb` tier and a
  store restore promotes back into a live slot with equal rows; an eviction leaves rows a live slot
  shares intact and the cap counts retained pages only; a vram tier that leaves less than one window
  is refused at open; and the implicit tier allocates what today's default path allocates.
- **P3** (built; hardware checks owed) the `host` tier: `backend = "host"` as first entry, or after a
  leading `vram` entry, host-visible mapped buffers with unchanged kernels, working with paging
  (the old paging/`kv.overflow` exclusion was only the pool-window rule), and the legacy `overflow`
  keys resolving onto it. Gate: output bit-identical to a vram tier; a whole host tier stores and
  resumes without a device allocation for KV. Measure: decode tokens/s at 2k and 8k on the RX 580
  and the integrated GPU, next to the table above.
  - *Done, tested on CPU:* `KvBackend::Host` with entry-naming validation; `KvCfg::residency`
    resolving the list and the legacy keys to one plan (`vram_reserve`, `KvSpill { device_cap,
    host_cap }`); the fit counts only the device part and bounds the window by `vram + host`;
    `refuse_host_tier` at open; the pool cap bounded by the tier; placement sweeps (dense
    resident, streaming, MoE expert budget) price device KV only; the storage chain skips both
    addressable entries; Vulkan places `KvCache` buffers by the plan, tallies them
    (`Backend::kv_placement`) and refuses a buffer past the host `size`. The CPU gate shows equal
    tokens, rows and KV bytes against a `vram` tier, a whole host tier parking and resuming
    through its chain, and legacy-versus-list resolution.
  - *Ran on the RX 580 (RADV):* `crates/infr-vulkan/tests/kv_host_tier.rs` (nine `#[ignore]`
    tests: no device KV for a host-first tier, the `vram` + `host` split, the size refusal, legacy
    placement equal to the list's, no tally without a plan, live occupancy released on drop and
    on a failed init, whole-buffer placement past the device remainder, racing allocations held
    to each tier's size). A tier's `size` limits LIVE bytes (claimed before the allocation,
    released when the buffer drops); the lifetime tally only feeds the report line. The fit and
    the open-time refusal walk the buffers in allocation order, and the pool cap leaves room
    for each slot's page table and private layers.
  - *Measured:* on the RX 580 with Qwen2.5-0.5B q4_k_m (q8_0 K/V), `--ctx` fixed per row so only
    the placement differs: 137.8 tok/s on a `vram` tier against 116.4 on a leading `host` tier at
    depth 2048, and 75.8 against 54.5 at depth 8192 — a 15% and a 28% cost, from reading the
    window's rows over PCIe every step. The same 64-token greedy continuation is byte-identical
    under `vram`, under `host`, and under `host` with `kv.paged = true`, which is the placement
    claim on a real GGUF. `docs/perf/kv-prefix.md` carries the table.
  - *Semantics to know:* a `vram` entry alone stays P2's retention reservation, but before a `host`
    entry it is the device part of the live windows. The legacy flag keeps its old sizing (full
    cache priced in VRAM, the spill as the clamp ladder's last rung, paged window per slot): only
    an explicit `host` entry has a host size, so only it gets the exact fit.
- **P4** (built and measured) the store-backed experiment: `[[kv.tiers]]` with `redis` (or
  `redb`) as the first entry, host-resident slot buffers, and the live window synchronised with that
  store each step — the configuration that proves placement is a property of the list rather than of
  the engine. Gate: output bit-identical to a vram tier, bytes transferred per step recorded and
  equal to the window, and the cost published. Expect it to be slow where the table above says it
  must be (a 4k window at 128 KB/token is ~0.5 GB over loopback per step); the small-model
  short-context row is the one that stays usable, and both numbers are what close the question for
  good.
  - *Built and green on the CPU gate:* `KvCfg::residency` resolves a leading storage entry to
    `store_backed` with `device_cap = Some(0)` and a host cap (the redb `size`, or unbounded for
    a redis tier, whose budget belongs to the server), so every live KV allocation already takes
    the P3 `alloc_kv_host` path — a leading storage tier therefore also implies host-resident slot
    buffers, and `kv.paged` alongside it is redundant rather than wrong. A window's extent id
    comes from a reserved range (`LIVE_WINDOW_BASE` down, one per slot, 1024 slots) and content
    block ids are mapped below it (`BLOCK_ID_FLOOR`), so the two id spaces cannot collide; the block
    index walks blocks, the mirror walks its own ids, and neither can evict the other (`RedbKv` LRU
    skips reserved live ids, `BlockCache::load` filters them, and a stored window no slot of this
    engine owns is dropped at open). A leading `redis` entry cannot have another store below it —
    the server owns its eviction, so it cannot be the bounded half of a chain — and that is a load
    error rather than a silent reassignment of the primary. `seam::mirror` holds
    the per-slot `Mirror`, `LiveWindow` (the shared `BlockCache` behind its lock) and `attach`;
    `SeamKv.mirror` is `None` by default, so an unmirrored run is unchanged. Six hooks fire after
    each forward that advances the window (DG denoise, MTP verify, chained prefill chunk, chain
    accept, per-token decode, and teardown where `cached` becomes the full window); each step
    downloads rows 0..`written_end` unlocked, takes the cache lock only for the store write and
    read-back, and writes the store's answer into the slot unlocked — the round trip is what makes
    the store authoritative, and `steps`, `bytes written`, `bytes read`, `last step bytes` are kept
    beside the prefix-cache counters (`ParallelSeam::kv_mirror_stats`). The extent's side metadata
    names the window's tokens, so `ParallelSeam` hydrates a slot from the store on checkout when
    those tokens prefix the prompt, before the block restore, and skips the re-prefill
    (`kv_hydrate_stats`). Hydration refuses models whose cache carries state the rows do not
    (`Config::has_unrewindable_state`: compressor rings, delta nets), and the engine leaves the
    window in the slot for them with a warning — a store cannot replay state it never held. It also
    refuses a stored window whose layers do not cover what a wrapped ring still attends to, one
    saved under other KV formats, and one whose pages cannot be made private without exhausting the
    pool (a miss, not a failed request); a failed upload resets the slot instead of leaving it
    claiming rows it lost. The CPU
    gate shows: leading-tier resolution and its errors naming `kv.tiers[0]`; a mirror step against
    an in-memory redb moving exactly the window's bytes each way, twice, with the counters
    accumulating; a store that answers with inverted bytes ending up in the slot; tokens
    bit-identical to a `vram` tier with `steps == prefills + fed tokens + 1` and the last step equal
    to `rows x row bytes`; and a fresh slot resuming a stored window with `n_cached == rows - 1` and
    the same continuation the first run produced.
  - *Needs hardware:* the decode-rate and bytes/step table
    (`crates/infr-llama/tests/kv_store_backed.rs`, `#[ignore]`, `INFR_TEST_MODEL` or the HF cache,
    `INFR_TEST_REDIS_URL` at an empty database) at ~2k and ~8k for `vram`, redb-on-disk and
    Valkey-over-loopback, and `crates/infr-vulkan/tests/kv_store_backed.rs` for the placement
    claim on a real device.
  - *Measured:* Qwen2.5-0.5B q4_k_m, RX 580, decode after a ~2k/~8k prompt: 141.5 tok/s on `vram`
    against 27.7 with redb on disk and 23.2 with Valkey over loopback; at ~8k, 79.4 against 7.8 and
    6.1. The window (98 MB at ~8k) moves out and back every step, so the run is 1.5 GB/s of store
    traffic, and the loopback store lands below a redb file: the experiment answers its own question
    — the window can live anywhere, and anywhere but the device is one to two orders of magnitude
    slower. `docs/perf/kv-prefix.md` carries the table, the driver and the Valkey version.
  - *Still owed:* `kv.paged` beside a leading storage tier (built, redundant, unexercised). The
    placement claim itself is checked on a device: `crates/infr-vulkan/tests/kv_store_backed.rs`
    passes on the RX 580 (no device KV allocated for a leading store tier).
  - *Semantics to know:* a failed step fails the generation, because the store is the authoritative
    copy under this placement; the store's extent holds stale tail rows after a rewind while the
    metadata keeps naming only the tokens the rows are known to match; a speculative chain writes
    rows past `cached` before they are accepted; a recurrent snapshot rides the extent only when it
    sits exactly at the window's end, and a model with unwindable state is refused outright; and the
    window counts against the store's own budget like any other extent, so a redb `size` smaller
    than the window is refused at open rather than mirrored into nothing. The mirror lives where the
    prefix store is opened (the `ParallelSeam` engine: `infr serve`, `infr multi`); one-shot
    `infr run` places the window host-side instead and warns that it opens no store.

## Non-goals

- Serving full attention from a `redb` or `redis` tier as a recommended configuration; P4's
  store-backed experiment exists to show the path works, not to make it the default.
- Multi-device stores: a tier lives on one device, and the tensor-parallel wrappers stay as they are.
- Changing what a plain `infr run` does: the implicit `vram` tier is the default, so the default
  stays device-local.

## Verification

A seam test for park → resume that counts copies and asserts zero, with rows equal to the capture
path; unit tests for the retained-page budget/LRU, for the vram tier's pinned-versus-evictable split
and its demotion into the next tier, and for the implicit tier matching today's allocation; a
SeamModel test that a vram tier shrinks the per-slot window by its `size` and that an oversized one is
refused; GPU benchmarks for P3 recorded in `docs/perf/kv-prefix.md`.
