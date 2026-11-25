# Plan 20: cold KV in a compressed format, and an eviction that keeps the trunk

Status: scoped, re-scoped 2026-11-26. The original draft was written against `[[kv.tiers]]`, which the
fork removed: `crates/infr-core/src/config/kv.rs` refuses that key and migrates an old list into
`[[kv.window]]` plus `[[kv.block]]` (`tiers_replacement`). So "the format belongs to the tier" now
means "the format belongs to a `[[kv.block]]` entry" — and to the live window that the first entry can
back. P1 (eviction) never depended on the config shape and is unchanged.

## Problem

A `[[kv.block]]` store holds `KvRows` verbatim: `RedbKv::put(id, first, rows)` writes the bytes the
live cache produced, in the row's own `k_fmt`/`v_fmt` (`DeviceKv::capture` builds them from the
slot's `k_fmt`/`v_fmt`). Three consequences, all avoidable:

- **Cold capacity is the live format.** An f16 cache cannot keep q8 cold blocks, so the touted
  2–4× of a compressed cold store needs the whole cache quantized, which taxes every step instead of
  only the restore. `docs/perf/kv-prefix.md` measures a 32k restore at 593 ms and 385 MB, about half
  of it redb decode plus the host copy: those bytes are the store's format, not the device's.
- **A format difference is a total loss.** The store's formats blob (`FORMATS_META_ID`, one per
  store) is read on `load` and adopted, and `capture` compares it to the live cache's
  `k_fmt`/`v_fmt` and, on a difference, calls `self.clear()` and starts the store over
  (`kv_store.rs:1145`). So changing the live format — or pointing a q8_0 window at an f16 block
  store — throws every block away instead of requantising it. Today there is neither a refusal nor a
  conversion; the store is simply emptied.
- **Eviction drops the wrong end.** `KvStore::contains` never stamps the budgeted store,
  `BlockCache::matched` walks the chain with `contains`, and `capture` skips blocks that are already
  present. A chain hit therefore leaves the trunk unstamped while fresh blocks carry recent stamps,
  so `take_lru` evicts the front of the chain first: the prefix every request needs is the first
  thing a squeeze throws away, and the next request misses at token 0. `Entry` also has no parent, so
  nothing can tell a leaf from a block other chains extend through.

## Design

**The format belongs to the entry.** `KvBlock` grows `type_k`/`type_v` (`f16`, `q8_0`, `q4_0`;
absent means `source` — store what the live cache holds, today's behaviour). Validation joins
`check_blocks` in `config/kv.rs`, beside `backend`/`size`/`path`/`url`: a storage-only key, exactly
like `size`, so a `[[kv.window]]` entry rejects it. The live window keeps its own `kv.type_k`/
`kv.type_v`; the window is never a cold leaf.

**Config declares intent, the blob records what is on disk, and they are reconciled at open.** The
config is the only place a user can ask for q8 on one specific entry; the per-store formats blob is
the only thing that knows what bytes the store actually holds. On open, resolve them: if the blob
exists and disagrees with the configured entry, **refuse to open** that store with an error naming
the entry, the stored pair and the configured pair — rather than today's silent `clear()`, and rather
than reinterpreting old bytes. A compressed store that quietly resets on a config typo is worse than
one that will not start. This is the one behaviour change P2 makes to an existing store, and it is
deliberate.

**Conversion happens at the store boundary.** `KvStore` grows `fn format(&self) -> (KvFmt, KvFmt)` and
`get` takes the format the caller wants; an entry whose own format already matches returns the bytes
untouched, the others convert. Rows are read through `RowLayout::spans()` into packed host bytes and
converted 32 elements at a time (q8_0: f16 scale plus 32 `i8`; q4_0 the same with nibbles) — a CPU
pass with no shader and no device work. The Vulkan planar Q8_0 side layout stays a device-side
concern; host rows are packed, so a planar slot compressing to q8 is nearly free.

**The leading entry is special and must stay honest.** Plan 19 makes the first `[[kv.block]]` the
authoritative copy of each slot's live window (`store_backed`): the window is synchronised with it
every step. Giving that entry `type_k`/`type_v` is exactly the "a Valkey tier costs half the bytes of
the window it backs" case — but it also means every hydrate pays a dequant pass over the window. So a
compressed leading entry is allowed (absent = the window's format, i.e. today), but P4 must measure
the per-step bytes *and* the dequant milliseconds it adds, and the docs must say that the resume win
shrinks as the gap between window format and entry format grows. Entries *below* the leading backer
are pure demotion targets: compression there wins on capacity for free.

**Eviction keeps the trunk.** `Entry` gains `parent: Option<u64>`, known at insert from the chain the
capture walked. `contains` stamps, `capture` re-stamps a block it finds present, `matched` touches
the chain it matched, and the LRU ranks on those stamps — so the trunk of a chain in use is the most
recently used thing in the store instead of the least. On overflow a leaf-first pass runs: an entry
no stored entry names as parent goes before anything a chain continues through, and a mid-chain block
only goes when every leaf is newer. With compression multiplying what a store holds, this policy is
what decides whether the extra capacity is worth having.

**Miss reasons become counters.** A restore that matches nothing, or stops at block k, records why
(empty chain, mid-chain, format, coverage, `max_ctx`) next to `restores`/`restored_tokens`, exposed
where `kv_store_stats()` already is. Today a miss is a missing debug line and nothing else.

## Phases

- **P1** (scoped) the eviction fix, format and config untouched: `parent` on `Entry`, stamping
  `contains` and `capture`, chain touch in `matched`, leaf-first overflow, and the miss-reason
  counters. Gate: a chain long enough to overflow a small store keeps blocks 0..k while both its tail
  and a second chain's tail are dropped, and an entry reached only through `contains` is no longer
  the next victim. P1 needs no new config key and can land on its own before any format work.
- **P2** (scoped) the per-entry format: `type_k`/`type_v` on `KvBlock` and their validation in
  `check_blocks`; `KvStore::format()` and `get`'s wanted format; requant on `put`, dequant on `get`;
  the per-store blob becomes a guard (refuse on mismatch) instead of `capture`'s clear; `Tiered`
  promotion converting instead of refusing. Gate: an f16 live window with a q8_0 block store
  round-trips rows equal to a q8_0 live cache's; a store reopened with a blob that disagrees with its
  configured `type_k`/`type_v` is refused with an error naming the entry; and a 32k restore from a
  q8_0 store moves half the bytes it moves today.
- **P3** (scoped) `q4_0` cold behind the same knob, offered only with a greedy-output comparison
  against f16 and q8_0 on a fixed prompt set, the divergence recorded, and a warning at open.
- **P4** (scoped) measures in `docs/perf/kv-prefix.md`: 32k restore bytes and milliseconds per
  format on the RX 580, capacity per GiB, the same for Redis over loopback, and the per-step hydrate
  cost of a compressed leading `[[kv.block]]` backer (bytes and dequant ms), next to the numbers
  plan 18 recorded.

## Non-goals

- Reusing a block at a different position (chunked hashing, CacheBlend-style partial recompute):
  separate plan, tracked as B242.
- Peer-to-peer block sharing and any RDMA path: single node is the target, and a cluster we do not
  own cannot be verified.
- Disaggregated prefill and decode: there is no second GPU to test the split with, and the transport
  it would need is the store we already have.
- Lossless `q4_0`: it is a capacity trade, judged by the output comparison, not by bit equality.

## Verification

A config test for each new key (every format accepted on a `[[kv.block]]` entry, an unknown format
rejected, a format on a `[[kv.window]]` entry rejected); a refusal test (blob pair ≠ configured pair
→ open error naming the entry); a conversion round-trip per format pair, comparing against rows
produced by a live cache configured with that format rather than against the source; a `Tiered`
promotion test across two formats; a store-backed test that a compressed leading entry still resumes
a slot and reports its per-step dequant; and the restore measurements for P4.

Where they apply, borrow upstream `docs/plan/15-kv-disk-store.md`'s S4 format cases: Q8_0 byte
rounding in `kv_fmt_bytes` (rows pack only at an even number of blocks per row), turbo blocks wider
than some rows, and planar capacity from `kv_rows_at`, never the buffer length.
