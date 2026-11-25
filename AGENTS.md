# Repository instructions

## Shell and local tests

- Use Git Bash on Windows when available. Do not use PowerShell.
- Keep Windows builds and tests native to Windows; do not substitute WSL for
  Windows verification.
- Run all local tests and their subprocesses headlessly, without visible console
  windows. Use hidden or no-window process creation when a runner needs it.
- Both GPUs on the Windows box may be used: the integrated GPU and the RX 7900
  XTX. Pick the device with `INFR_DEV` after checking `infr devices`, say which
  one a result came from, and run GPU workloads serially.
- Prefer cached small models. Downloading a model to test with is allowed while
  disk space lasts; check the free space and the file size first.

## Branch and delivery

- Keep all work on `main`.
- Pull the latest changes before continuing each slice of work, preserving local
  changes when integrating upstream work.
- Commit and push each completed slice to `main`.
- Run what CI runs before pushing: `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --locked -- -D warnings`,
  `cargo nextest run --workspace --locked`, and
  `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked`. The
  docs job fails on a public item's doc linking to a private one.
- Watch CI for the pushed commit, inspect its jobs, and resolve failures before
  considering the slice complete. A successful push alone is not verification. A
  push cancels the run before it, so let a run finish before pushing the next
  slice when its verdict is still owed.

## How work is run

- The main agent orchestrates. Each slice goes through a coder agent, then a
  separate reviewer agent that reads the diff without being told what to
  conclude; the orchestrator re-checks both reports against the tree before
  committing.
- While sub-agents are working, the orchestrator plans what comes next rather
  than waiting: plans under `docs/plan/`, findings as tasks on the Vikunja
  board.
- When the board's actionable items run out, have a reviewer read a part of the
  codebase no earlier pass covered and file what it finds as tasks.

## Where findings go

The tracker is Vikunja, project `Kessoku` → sub-project `infr` (id 7, instance
`https://tasks.cappuchino.xyz`, MCP server `vikunja`). File every finding, gap
and deferred piece of work there, assigned to `bot-agent`.

`docs/backlog.md` is **upstream's** file and this fork does not write to it. It
stays as inherited history: read it for context, and keep an existing `B###` id
in a task's body as the pointer back (`docs/backlog.md B245`), but new findings
do not get a `B###` and rarely get an entry. The review-leftover sections are
upstream's triage, not ours.

`docs/plan/*.md` stays in the repo — a plan records design and measurements, not
tasks — and a plan's phases are closed there as well as reflected on the board.

A task from this repository says, in its body: what is wrong or missing with the
numbers that make it worth doing, the file it lives in, the upstream `B###` or
`docs/plan/NN` pointer if one exists, and the test or measurement that closes
it. Its title is a plain imperative (“Warm the request path's pipelines at
load”), and the domain is a **label**: `kv`, `serve`, `vulkan`, `config`,
`cache`, `vram`, `sampling`, `tests`, `sync`, `tracker`, plus `decision` for
work blocked on a choice rather than on code. Relate tasks that move together
(`related`, `precedes`, `blocked`), so a thread that picks one sees the rest.

The MCP `vikunja` surface cannot create or list labels or task relations, only
attach existing ones, so the REST API is the way and the token is already in
`~/.config/vikunja/mcp-token`: `GET /api/v2/labels`, `POST /api/v2/labels`,
`POST /api/v2/tasks/{id}/labels` with `{"label_id": N}`, and
`POST /api/v2/tasks/{id}/relations` with
`{"other_task_id": N, "relation_kind": "related"}`. The v2 instance answers
`405` to `PUT` on those paths — creation is `POST`.

## What to work on

- Correctness is the top priority, and performance matters just as much: a
  change that buys one must not quietly cost the other. Both come before
  anything else in improvement work.
- Do not shy away from a large refactor when it is worth it for correctness, for
  performance, or for both. Say what it buys, and verify it: a test that can
  fail for correctness, a measurement for performance.
- New model families and planned features are in scope. Support recent models
  people will actually run; an older or less useful family gets support only
  when it is genuinely little work. The reference for a port is the llama.cpp
  checkout at `~/Projects/mxaddict/llama.cpp`.
- Streaming is a standing goal: disk to DRAM to VRAM, so that any model size
  runs on any hardware. That includes the cache eviction logic in every tier,
  which should be as fast as it can be made.
- Do not block a turn polling slow CI jobs: the CPU-goldens suite takes ~20+ min
  on GHA. Merge/push once the fast signal is green (platform `cargo test` jobs,
  clippy, fmt, plus a local run of the directly-affected tests), then check back
  on the slow jobs and resolve any failures as a follow-up slice.
