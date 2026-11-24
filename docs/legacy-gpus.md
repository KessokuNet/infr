# Older and lower-end Vulkan GPUs

This fork ([KessokuNet/infr](https://github.com/KessokuNet/infr)) optimizes and
backports infr to Vulkan GPUs that upstream does not target: no cooperative
matrix, no hardware-accelerated int8 dot product, and wave64-only subgroups.
The baseline is **AMD GCN4 / Polaris** (RX 470/480/570/580), a common cheap
used GPU. Changes that are not specific to these GPUs are meant to go upstream
to [kryptic-sh/infr](https://github.com/kryptic-sh/infr).

## What is different on these GPUs

| Change | What it does |
|---|---|
| Wave64 support | Wave32 kernels run as 32-lane halves of a 64-lane wave (`shaders/wave64.glsl`, spec constant `W64`), so infr starts at all on wave64-only devices. |
| First-forward submit split | Slow discrete GPUs split even the first prefill into submits of 64 dispatches, so a long prompt can't trip the driver's hang watchdog. |
| `kq_fma` prefill GEMM | An fp32 FMA GEMM for Q4_K/Q6_K (`native_gemm_kq_fma.comp`) replaces the dp4a MMQ kernels, which are emulated in ALU on GCN. |
| `kq_fma` decode GEMV | An fp32 Q4_K/Q6_K GEMV on f32 activations (`native_gemv_kq_fma.comp`) replaces `quant_q8` + the dp4a GEMV. |
| `kq_fma` verify GEMV | An fp32 multi-row Q4_K/Q6_K GEMV (`native_gemv_kq_fma_mrow.comp`) for MTP-verify batches — MR=4 rows share each superblock's weight-word load, bit-identical row-for-row to the decode GEMV. Replaces the int8 `mrow` on verify batches, whose quantized activations flip near-tie greedy tokens vs decode. |
| Float decode GEMVs | F32 weights use a multi-output GEMV (`linear_f32_mo.comp`); BF16 weights use the dedicated reduction GEMV. |

All of these switch on automatically when the device does not report int8 dot
as hardware-accelerated (`Capabilities::i8_dot_fast`), and are off on
GPUs that have it.

## Switches

| Variable | Config key | Effect |
|---|---|---|
| `INFR_KQ_FMA_TIER=0\|1` | `kernels.vulkan.kq_fma_tier` | Force the `kq_fma` prefill GEMM off/on. |
| `INFR_KQ_FMA_GEMV=0\|1` | `kernels.vulkan.kq_fma_gemv` | Force the `kq_fma` decode GEMV off/on. |
| `INFR_WAVE64_NATIVE=0\|1` | `kernels.vulkan.wave64_native` | Run the subgroup decode GEMV at its native 64-lane width instead of emulated 32-lane halves (default on; inert off wave64 hardware). |
| `INFR_WEIGHTS_AS_SSBO=0\|1` | `kernels.vulkan.weights_as_ssbo` | Bind `kq_fma` decode-GEMV weights as storage buffers instead of reading them by 64-bit device address (default off). May help GPUs without a scalar-base global load, e.g. GFX8/Polaris, where each device-address load builds its own 64-bit address. |
| `INFR_SUBMIT_DISPATCHES=N` | `device.submit_dispatches` | Dispatches per submit (`0` = never split). |

## Serving tips (8 GB cards)

- Pass `--ctx` (e.g. `--ctx 8192`). At the default 131k context, placement
  falls back to a 256-row prefill chunk, which slows prefill and breaks
  KV-prefix reuse between chat turns on Gemma's sliding-window layers.
- A larger `--ubatch` (e.g. `1024`) gives more room for prefix reuse after
  long replies, if it fits.

## Measured (RX 580 8 GB, RADV)

`infr compare --sweep` against llama.cpp `785322d` (its Vulkan backend), same GGUF,
same card, both re-run back to back. Tokens/s, `-r 3`.

### 2026-11-24

The same sweep three times over, no `infr serve` resident, median of the three. Every
`infr` cell moved by under 3% between runs (the 3B's `tg128`, 77-79 t/s, was the
widest) except llama.cpp's four-token prefill on LFM3 (214-251 t/s), so the ratios
are not sampling noise.

| Model | metric | infr | llama.cpp | ratio |
|---|---|---|---|---|
| Qwen2.5-Coder-3B Q4_K_M | pp512 | 635 | 540 | 1.18x |
| | tg128 | 79 | 78 | 1.01x |
| | tg64 @ d4096 | 59 | 68 | **0.87x** |
| | pp4 @ d4096 | 82 | 136 | **0.60x** |
| Gemma 4 E4B QAT UD-Q4_K_XL | pp512 | 454 | 415 | 1.09x |
| | tg128 | 50 | 53 | 0.94x |
| | tg64 @ d4096 | 46 | 46 | 1.00x |
| | pp4 @ d4096 | 68 | 114 | **0.60x** |
| LFM3-1.7B Q6_K | pp512 | 1068 | 937 | 1.14x |
| | tg128 | 114 | 105 | 1.08x |
| | tg64 @ d4096 | 105 | 101 | 1.04x |
| | pp4 @ d4096 | 147 | 248 | **0.59x** |

Prefill is up 19% on both models that appear in the 2026-11-05 table as well
(535 → 635 and 382 → 454), and decode holds parity or better. What has not moved is
the four-token turn at depth: `pp4@d4096` is still about 0.6x of llama.cpp on every
model measured, and it is now the only measured deficit against the reference. That
is the follow-up message shape — a long transcript plus a short new instruction — so
it is also the one that matters most to chat clients. `tg64@d4096` on the 3B lost
5 t/s against the November run while llama.cpp gained 6; the two tools cache and
batch the incremental prefill differently there and the direction flipped.

LFM3 is new here: it is the model the live `infr serve` install answers with, and the
114 t/s decode matches the server's own journal (109-114 t/s on the same card).

### 2026-11-05

| Model | metric | infr | llama.cpp | ratio |
|---|---|---|---|---|
| Qwen2.5-Coder-3B Q4_K_M | pp512 | 535 | 476 | 1.13x |
| | tg128 | 80 | 71 | 1.14x |
| | tg64 @ d4096 | 64 | 62 | 1.02x |
| | pp4 @ d4096 | 75 | 136 | **0.55x** |
| Gemma 4 E4B QAT UD-Q4_K_XL | pp512 | 382 | 367 | 1.04x |
| | tg128 | 49 | 49 | 1.00x |
| | tg64 @ d4096 | 45 | 44 | 1.02x |
| | pp4 @ d4096 | 69 | 113 | **0.61x** |

The E4B row does not load on the 2026-11-24 build without the seam fix in
`5057fa31`: that file declares `gemma4.attention.shared_kv_layers`, and the loader
asked for an `attn_k` the shared layers do not have.

Earlier campaign (same card, before the `kq_fma` tier and the seam work; no date was
recorded at the time, so read the llama.cpp columns as historical):

| Model | pp512 before | pp512 after | llama.cpp pp512 | tg128 after | llama.cpp tg128 |
|---|---|---|---|---|---|
| Qwen2.5-Coder-3B | 206 | 499 | 421 | 63.6 | 63.8 |
| Gemma 4 E2B | 285 | 616 | 602 | — | 65.7 |

Gemma 4 E4B decode at 4k context went 29.6 → 44.4 t/s across both campaigns.

MTP-verify shapes (RX 580, effective weight-stream GB/s, m = drafts+1): the verify
kernel ties the int8 `mrow` it replaces on big weights (down-proj Q4_K m=7: 39 vs 49)
and beats it on small ones (lm_head m=2: 89 vs 77), while crushing the f32-dequant
`mrow` that serves small weights (2–3.5×). The property that matters is exactness:
verify-flagged graphs score rel ~2e-7 vs the host oracle (pure f32 reassociation
noise), where the int8 path scores ~4e-3 — the noise that flipped greedy tokens
and collapsed draft acceptance.

## Status and plans

Tracked on the fork's project board. Next up: Q4_0 support in the `kq_fma`
kernels (Gemma QAT quants are Q4_0), single-pass sliding-window attention for
decode, KV shifting for rolling chat windows, and speculative decoding on
Vulkan.
