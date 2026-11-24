//! What a store-backed KV window costs per decode step: the row bytes each step moves out to the
//! store and reads back, and the decode rate that leaves, against the same model on a `vram` tier
//! and against a redb store on disk and a Valkey store over loopback.
//!
//! Runs only when a suitable GGUF is in the HF cache and a Vulkan device exists. The Valkey row
//! needs `INFR_TEST_REDIS_URL` pointing at an EMPTY database, and no server is started here: the
//! store is keyed by model geometry and by each row's own prompt tag, so rows left behind by an
//! earlier run are re-read instead of measured. Prints a table instead of asserting a threshold:
//! the cost is a property of the model's row size and of the transport, not of this test.
//!
//! Run: `INFR_TEST_REDIS_URL=redis://127.0.0.1:6379 cargo test -p infr-llama --release \
//!       --test kv_store_backed -- --ignored --nocapture`

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Prompt tokens measured, and the context each run gets. One decode step moves the whole window,
/// so the two sizes are what separates "the row copy is noise" from "the row copy is the step".
const MEASURED: [(usize, usize); 2] = [(2048, 4096), (8192, 12288)];

/// Decode steps per measurement: enough for the rate to settle, short enough to keep the run small.
const DECODE: usize = 64;

fn prompt(tokens: usize, tag: &str) -> String {
    let sentence = "The maintenance log of the Honshu line is kept in triplicate: one copy rides \
                    the train, one sits in the depot office, and one is sealed in the yardmaster's \
                    safe. ";
    let mut s = String::from("Use this reference material:\n");
    while s.len() < tokens * 4 {
        s.push_str(sentence);
    }
    s.push_str(&format!("Report the {tag} entry."));
    s
}

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
        return Some(PathBuf::from(p));
    }
    let store = infr_hub::Store::discover().ok()?;
    [
        (
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        ),
        ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
    ]
    .into_iter()
    .find_map(|(repo, file)| store.snapshot_file(repo, file))
}

/// The three placements compared: `Vram` is the plain engine, the others back the live window
/// with the `[[kv.block]]` store whose cost is being measured.
#[derive(Clone, Copy)]
enum Tier {
    Vram,
    RedbFile,
    Redis,
}

impl Tier {
    fn label(self) -> &'static str {
        match self {
            Tier::Vram => "vram",
            Tier::RedbFile => "redb-file",
            Tier::Redis => "valkey",
        }
    }
}

fn engine(tier: Tier, disk: &std::path::Path, url: Option<&str>) -> infr_llama::EngineConfig {
    let mut c = infr_llama::EngineConfig::default();
    c.kv.slots = 1;
    let entry = match tier {
        Tier::Vram => return c,
        Tier::RedbFile => infr_core::config::KvBlock {
            backend: Some(infr_core::config::KvBackend::Redb),
            size: Some(infr_core::SizeSpec::Bytes(1024 << 20)),
            path: Some(disk.to_path_buf()),
            ..Default::default()
        },
        Tier::Redis => infr_core::config::KvBlock {
            backend: Some(infr_core::config::KvBackend::Redis),
            url: url.map(str::to_string),
            ..Default::default()
        },
    };
    c.kv.block.push(entry);
    c.kv.store_backed = true;
    c
}

struct Cost {
    tok_s: f64,
    rows: usize,
    step_bytes: u64,
    steps: u64,
    written: u64,
    read: u64,
}

fn measure(
    path: &std::path::Path,
    tier: Tier,
    disk: &std::path::Path,
    url: Option<&str>,
    tokens: usize,
    ctx: usize,
) -> Cost {
    let model = infr_llama::SeamModel::load_with(path, None, Arc::new(engine(tier, disk, url)))
        .expect("load");
    let seam = infr_llama::parallel::ParallelSeam::new(
        model,
        1,
        Some(infr_core::SizeSpec::Bytes(ctx as u64)),
    )
    .expect("engine");
    let greedy = infr_llama::sampling::RequestSampling {
        temp: Some(0.0),
        ..Default::default()
    };
    let run = |p: &str| {
        let req = seam.request_ctx(greedy.clone());
        let t0 = Instant::now();
        let stats = seam
            .generate(p, DECODE, None, &req, |_| {})
            .expect("generation");
        (t0.elapsed().as_secs_f64(), stats)
    };
    run(&prompt(512, "warmup"));
    // A tag of its own per row: nothing this table prints can be a restore of another row.
    let p = prompt(tokens, &format!("{}-{tokens}", tier.label()));
    let (_, stats) = run(&p);
    let (steps, written, read, step_bytes) = seam.kv_mirror_stats();
    Cost {
        tok_s: stats.n_gen as f64 / stats.decode_secs.max(1e-9),
        rows: stats.n_prompt + stats.n_cached,
        step_bytes,
        steps,
        written,
        read,
    }
}

fn row(label: &str, ctx: usize, c: &Cost) -> String {
    format!(
        "{label:10} {ctx:6} {:7.1} tok/s {:10} B/step {:8} rows  {:>4} steps {:>10} / {:>10} B \
         out/in",
        c.tok_s, c.step_bytes, c.rows, c.steps, c.written, c.read
    )
}

#[test]
#[ignore = "requires a Vulkan GPU and a cached GGUF (INFR_TEST_MODEL overrides the HF cache)"]
fn a_store_backed_window_costs_a_step_per_tier() {
    let Some(path) = model_path() else {
        eprintln!("skip: no GGUF in the HF cache");
        return;
    };
    let dir = std::env::temp_dir().join(format!("infr-kv-remote-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let url = std::env::var("INFR_TEST_REDIS_URL").ok();
    println!(
        "\nplacement     ctx  decode rate  bytes/step  window     steps  bytes moved out/in\n\
         (bytes/step is the whole live window, moved once each way per forward)"
    );
    for (tokens, ctx) in MEASURED {
        let disk = dir.join(format!("kv-{tokens}.redb"));
        let vram = measure(&path, Tier::Vram, &disk, None, tokens, ctx);
        println!("{}", row("vram", ctx, &vram));
        let redb = measure(&path, Tier::RedbFile, &disk, None, tokens, ctx);
        println!("{}", row("redb-file", ctx, &redb));
        match url.as_deref() {
            Some(url) => {
                let redis = measure(&path, Tier::Redis, &disk, Some(url), tokens, ctx);
                println!("{}", row("valkey", ctx, &redis));
            }
            None => println!(
                "{:<10} {ctx:6} skipped (INFR_TEST_REDIS_URL is unset)",
                "valkey"
            ),
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}
