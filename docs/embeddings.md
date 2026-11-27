# Embeddings

infr serves embedding models through the same OpenAI-shaped surface as everything else:
`infr embed` on the CLI and `POST /v1/embeddings` over HTTP. The first (and currently only)
architecture is **EmbeddingGemma 2** — `general.architecture = gemma-embedding2`, a 270M text tower
with a mean-pooled 768-d head, plus a Gemma 4 `mmproj` carrying the vision (and audio) encoders.
Design and measurements: [plan/22-embedding-models.md](plan/22-embedding-models.md).

## CLI

```sh
# text
infr embed unsloth/embeddinggemma-2-GGUF:Q8_0 -p "cat"

# MRL truncation: 768/512/256/128 are the trained widths; the vector is renormalized
infr embed <model.gguf> -p "cat" --dimensions 256

# an image, through the projector beside the checkpoint (or serve.mmproj)
infr embed <model.gguf> --image photo.png

# the same forward on the GPU
infr embed <model.gguf> --dev Vulkan0 -p "cat"
```

Output is an OpenAI-shaped JSON list on stdout, one `{object, index, embedding}` entry per input,
texts first and then `--image` files in argument order. `-p` and `--image` are both repeatable.

## HTTP

```sh
curl -s localhost:8080/v1/embeddings -H 'content-type: application/json' -d '{
  "model": "embeddinggemma-2",
  "input": ["cat", {"type": "text", "text": "a dog"}],
  "dimensions": 256
}'
```

`input` takes a string, an array of strings, or an array of content parts: `{"type":"text","text":…}`
and `{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}`. Only `data:` URIs are
accepted — the server never fetches a remote URL. Token-id arrays are refused: they would skip the
model's own `<bos>`/`<eos>` framing, which changes the vector.

`dimensions` truncates to the leading entries (Matryoshka) and renormalizes; `encoding_format` is
`float` (default) or `base64` (little-endian f32 bytes). `usage.prompt_tokens` is the real token
count, image soft tokens included. At most 256 inputs and four images per request.

An embedding model skips the concurrent slot engine in `infr serve` — the route runs on the
serialised seam, one request at a time, so generation-style batching does not apply.

## Instruction prefixes (caller-side)

Text quality depends on the task prefix the model was trained with. infr does not inject one:
prepend it yourself, and pass images with no prefix. The pairs from the model card:

| task | query side | document side |
|---|---|---|
| search | `task: search result \| query: {query}` | `title: {title} \| text: {content}` |
| question answering | `task: question answering \| query: {question}` | `title: {title} \| text: {passage}` |
| fact checking | `task: fact checking \| query: {claim}` | `title: {title} \| text: {evidence}` |
| code search | `task: code retrieval \| query: {query}` | `title: {title or filename} \| text: {code}` |
| classification | `task: classification \| query: {content}` | — |
| clustering | `task: clustering \| query: {content}` | — |
| similarity | `task: sentence similarity \| query: {content}` | — |

The prefix moves the vector materially — the unprefixed text sits at ~0.90 cosine from the prefixed
one — so index and query with the same convention or the scores will be quiet nonsense.

## Projector

`serve.mmproj` names the projector; unset, the first `mmproj*.gguf` beside the checkpoint is used. A
model without one refuses images (400 on the route, an error on the CLI) rather than embedding text
only. Only F32/F16/BF16 projectors load today; Q8_0 is
[tracked](https://tasks.cappuchino.xyz/tasks/119).

## Backends

CPU and Vulkan compute the same vector to cosine ≥ 0.999; Metal is wired but unverified (no Apple
hardware here). Vulkan refuses a banded pass longer than the SWA ring (1024 tokens) instead of
reading evicted rows — run those on the CPU until the cache is sized per request
([tracked](https://tasks.cappuchino.xyz/tasks/117)).
