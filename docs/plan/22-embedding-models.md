# Plan 22: embedding models

Status: P1–P4 in progress (this file lands with P1). Scope fixed by cappy on 2026-11-27: text first,
then image input over the Gemma 4 `mmproj` path; land on the fork's `main`, not an upstream PR;
expose `/v1/embeddings`; CPU and Vulkan (Metal last).

## Problem

infr runs autoregressive decoders only. `Config::from_gguf` accepts a fixed arch list, every graph
runs causal attention, the head predicts vocab logits and there is no pooling step and no embeddings
route — `crates/infr-server/src/lib.rs` serves `/v1/{models,completions,chat/completions}`, and
`infr` has no `embed` subcommand. A feature-extraction model therefore cannot even be loaded.

The first target is `EmbeddingGemma 2` (`unsloth/embeddinggemma-2-GGUF`), a 270M text tower plus
Gemma 4's vision and audio encoders. A reference implementation exists in llama.cpp
(`src/models/gemma-embedding2.cpp`, arch string `gemma-embedding2`), which is what the parity test
compares against.

## The model, measured from the Q8_0 file

- `general.architecture = gemma-embedding2`, 24 layers, `n_embd` 512, FF 2048, vocab 262144,
  head_count 4, eps 1e-6, no KV sharing.
- SWA pattern 5:1 (period 6). SWA layers: head_dim 256, 2 KV heads, rope dim 256, base 1e4.
  Full layers (5/11/17/23): head_dim 512, **1** KV head, rope dim 512, base 1e6.
- `attention.causal = false`; `attention.sliding_window = 1024` with llama.cpp's
  `LLAMA_SWA_TYPE_SYMMETRIC` (`src/llama-hparams.h:553`), so an SWA layer's row `i` attends
  `|p0 - p1| <= 512` in both directions and a full layer attends the whole prompt.
- `embedding_length_per_layer_input` 512; per-layer PLE is **projection-only** (`inp_gate`,
  `proj`, `post_norm`, `layer_output_scale`) over the model-level `per_layer_model_proj`
  (BF16 `[24*512, 512]`) + `per_layer_proj_norm`; there is no `per_layer_token_embd` table.
- `pooling_type = 1` (mean), `embedding_length_out = 768` via `output_norm` then
  `output.weight` Q8_0 `[768, 512]`, and no softcap.
- Instruction prefixes are the caller's (`task: search result | query: …`); they measurably move
  the vector (0.90 cosine against the unprefixed text), so the tests keep one as a tripwire.

## Design

**The forward pass is the existing dense forward, taken whole.** No new graph, no sampling loop,
no persistent KV: tokenize, run ONE batched pass over the whole prompt with `max_new = 0`, and read
the per-position head rows from that pass's `verify` output (`[tokens * head_rows]`). The ordinary
prefill path is not it — it withholds the last prompt token for the decode loop, which a `max_new =
0` request never runs, so the batched teacher-forced (`verify`) branch is what covers every row in
one go. It is dense-only, which every embedding arch is.

**The head is the embedding projection.** `output.weight` is `[768, 512]`, not `[vocab, n_embd]`.
`Config::head_rows()` returns `n_embd_out` when the GGUF carries `embedding_length_out` and
`vocab` otherwise; the LM-head `Op::Linear`, the softcap width and the argmax width use it. Since
the projection is linear, mean-pooling the per-token projected rows equals projecting the mean of
the hidden rows, so pooling needs no graph op and runs on the host: mean over positions, then L2
normalize, then MRL truncation (`dimensions`) followed by a second normalize — the card is explicit
that a truncated vector must be renormalized.

**Masks.** Full-attention layers already have exactly the right variant: `AttnMask::Canvas { lo: 0 }`
means "every row attends `[0, kv_len)`", which is bidirectional full attention. SWA layers need the
symmetric band, which nothing expresses — add `AttnMask::Banded { half }` to `infr-core` and teach
the CPU executor the `[p1 - half, p1 + half]` clip; `half = swa_window / 2`, matching
`is_masked_swa`'s symmetric arm. Vulkan gets the same variant in P3.

**No softcap, no sampling.** The arch sets `final_softcap = 0` and refuses generation surfaces
(sampling flags, tool-call template work) with a clear message rather than producing nonsense.

**PLE without the token table.** `build_per_layer_embd` (`crates/infr-llama/src/util.rs:77`)
currently requires `per_layer_token_embd.weight` whenever `n_embd_per_layer != 0`. Reinterpret it as
a table handle and make it optional; the E2B prologue already runs the model-level projection +
RMSNorm as graph ops (`per_layer_model_proj` / `per_layer_proj_norm` uploads,
`crates/infr-llama/src/seam/runner.rs:1488`), which is the half this model has.

**Surface.** `infr embed -m <gguf> -p <text> [--dimensions N]` prints one JSON vector per input;
`POST /v1/embeddings` takes the OpenAI shape (`input`: string or array, `model`, `dimensions`,
`encoding_format`), returns `object: "list"`, per-item `index`, and `data[].embedding`. Prefixes stay
caller-side and get documented.

## Phases

- **P1 — text tower, CPU.** Arch const + `ALL` + `TRANSFORMER`; config parse (`n_embd_out`,
  `non_causal`, PLE width, dual rope, per-layer KV heads); weight loads; `AttnMask::Banded` + CPU
  executor; `head_rows()`; host mean-pool + normalize; `infr embed`.
  Closes on cosine ≥ 0.999 against `llama-embedding` on a fixed corpus and the cosine tripwire
  (paraphrase 0.93 ≫ unrelated 0.71, prefixed vs unprefixed 0.90 — a broken mask or PLE collapses
  these toward 1.0).
- **P2 — `/v1/embeddings`.** Route, request/response types, `dimensions`, `encoding_format`,
  batching, `serve` model kind, `Host`/auth gating like every other route, request-size limits.
- **P3 — Vulkan.** `AttnMask::Banded` in `infr-vulkan`'s attention path, device-side pooling if the
  round trip is measurable, `gpu-run` benchmarks versus the CPU reference vector.
- **P4 — image input.** EmbeddingGemma 2's `mmproj` is Gemma 4's vision (170M) plus audio (300M)
  encoders; vision rides the existing `serve.mmproj` plumbing from INFR-46 for image inputs to the
  embeddings route. Audio is its own task.

## Non-goals

- Generation from an embedding model (no canvas, no chat template, no sampling).
- Audio embeddings in P1–P4.
- Model conversion; the GGUF is the input format.

## Verification

`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings`,
`cargo nextest run --workspace --locked`, `RUSTDOCFLAGS="-D warnings" cargo doc --workspace
--no-deps --locked`. Parity vectors are produced by
`llama.cpp/build-embed/bin/llama-embedding -m ~/Models/embeddinggemma-2-Q8_0.gguf -ngl 0 -c 512
--embd-output-format json -p <text>`; unit tests cover geometry (per-layer head dims, KV heads, head
rows), the mask clip arithmetic, MRL truncation + renormalization, and the refusals.
