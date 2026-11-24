//! The KV placement sections: `[[kv.window]]`, `[kv.pages]`, `[[kv.block]]`, the removed keys that
//! used to say the same things, and the paging matrix.

use std::path::Path;

use super::{
    cli, env, file, Config, ConfigError, ConfigOverrides, KvBackend, KvBlock, KvPages, Paging,
    PartialConfig, WindowEntry, WindowPlan, KV_REDB_DEFAULT_BYTES,
};
use crate::{DType, SizeSpec};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn parsed(text: &str) -> (PartialConfig, Vec<String>) {
    file::parse_str(text, Path::new("infr.toml")).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn cfg_of(text: &str) -> Config {
    Config::load_from_layers(&[parsed(text).0])
}

fn file_err(text: &str) -> String {
    match file::parse_str(text, Path::new("infr.toml")) {
        Err(ConfigError::Value { path, message }) => format!("{path}: {message}"),
        other => panic!("{text}: expected a value error, got {other:?}"),
    }
}

fn bytes(n: u64) -> Option<SizeSpec> {
    Some(SizeSpec::Bytes(n))
}

fn window(backend: KvBackend, size: Option<SizeSpec>) -> WindowEntry {
    WindowEntry {
        backend: Some(backend),
        size,
    }
}

fn block(backend: KvBackend) -> KvBlock {
    KvBlock {
        backend: Some(backend),
        ..KvBlock::default()
    }
}

fn redis(url: &str) -> KvBlock {
    KvBlock {
        url: Some(url.to_string()),
        ..block(KvBackend::Redis)
    }
}

fn with_windows(entries: Vec<WindowEntry>) -> super::KvCfg {
    let mut kv = Config::default().kv;
    kv.window = entries;
    kv
}

fn with_blocks(entries: Vec<KvBlock>) -> super::KvCfg {
    let mut kv = Config::default().kv;
    kv.block = entries;
    kv
}

#[test]
fn the_three_sections_parse_and_resolve_together() {
    let (partial, warnings) = parsed(
        "[kv]\npaged = true\n\n\
         [[kv.window]]\nbackend = \"vram\"\nsize = \"4g\"\n\n\
         [[kv.window]]\nbackend = \"host\"\nsize = \"16g\"\n\n\
         [kv.pages]\nsize = \"2g\"\n\n\
         [[kv.block]]\nbackend = \"redb\"\npath = \"/var/cache/infr/kv.redb\"\nsize = \"20g\"\n\n\
         [[kv.block]]\nbackend = \"redis\"\nurl = \"redis://cache.internal:6379/0\"\n",
    );
    assert!(warnings.is_empty(), "{warnings:?}");
    let kv = Config::load_from_layers(&[partial]).kv;

    let layout = kv.layout().expect("valid");
    assert_eq!(
        layout.window,
        WindowPlan {
            device_cap: Some(4 * GIB),
            host_cap: Some(16 * GIB),
        }
    );
    assert_eq!(layout.pages, Some(2 * GIB));
    assert!(!layout.store_backed);
    assert_eq!(layout.blocks.len(), 2);
    assert_eq!(layout.blocks[0].backend, Some(KvBackend::Redb));
    assert_eq!(layout.blocks[0].bytes(), Some(20 * GIB));
    assert_eq!(
        layout.blocks[0].path.as_deref(),
        Some(Path::new("/var/cache/infr/kv.redb"))
    );
    assert_eq!(layout.blocks[1].backend, Some(KvBackend::Redis));
    assert_eq!(
        layout.blocks[1].url.as_deref(),
        Some("redis://cache.internal:6379/0")
    );
    // The pages reserve rides on the last memory the windows name.
    assert_eq!(layout.placement().host_cap, Some(18 * GIB));
}

#[test]
fn absent_sections_are_independent() {
    let layout = Config::default().kv.layout().expect("valid");
    assert_eq!(layout, super::KvLayout::default());
    assert_eq!(layout.window, WindowPlan::default());

    let pages_only = cfg_of("[kv.pages]\nsize = \"1g\"\n").kv.layout().unwrap();
    assert_eq!(pages_only.window, WindowPlan::default());
    assert_eq!(pages_only.pages, Some(GIB));
    assert!(pages_only.blocks.is_empty());

    let blocks_only = cfg_of("[[kv.block]]\nbackend = \"redb\"\n")
        .kv
        .layout()
        .unwrap();
    assert_eq!(blocks_only.window, WindowPlan::default());
    assert_eq!(blocks_only.pages, None);
    assert_eq!(blocks_only.blocks.len(), 1);
}

#[test]
fn windows_resolve_to_a_device_and_host_plan() {
    let plan = |entries| with_windows(entries).layout().expect("valid").window;
    let (vram, host) = (KvBackend::Vram, KvBackend::Host);
    assert_eq!(plan(vec![]), WindowPlan::default());
    // An unsized `vram` window is the same as none: it fills what the fit leaves.
    assert_eq!(plan(vec![window(vram, None)]), WindowPlan::default());
    assert_eq!(
        plan(vec![window(vram, bytes(4 * GIB))]),
        WindowPlan {
            device_cap: Some(4 * GIB),
            host_cap: None
        }
    );
    // A host window with no `vram` before it holds every row.
    assert_eq!(
        plan(vec![window(host, bytes(8 * GIB))]),
        WindowPlan {
            device_cap: Some(0),
            host_cap: Some(8 * GIB)
        }
    );
    assert_eq!(
        plan(vec![window(vram, None), window(host, bytes(8 * GIB))]),
        WindowPlan {
            device_cap: None,
            host_cap: Some(8 * GIB)
        }
    );
    assert_eq!(
        plan(vec![window(vram, bytes(GIB)), window(host, bytes(8 * GIB))]),
        WindowPlan {
            device_cap: Some(GIB),
            host_cap: Some(8 * GIB)
        }
    );
}

#[test]
fn window_entries_follow_the_order_and_size_rules() {
    use SizeSpec::Percent;
    let (vram, host) = (KvBackend::Vram, KvBackend::Host);
    let cases = [
        (
            vec![window(host, bytes(MIB)), window(vram, bytes(MIB))],
            "kv.window[1] (vram)",
            "comes first",
        ),
        (
            vec![window(vram, None), window(vram, None)],
            "kv.window[1] (vram)",
            "at most one",
        ),
        (
            vec![window(host, bytes(MIB)), window(host, bytes(MIB))],
            "kv.window[1] (host)",
            "at most one",
        ),
        (
            vec![
                window(vram, None),
                window(host, bytes(MIB)),
                window(host, bytes(MIB)),
            ],
            "kv.window[2] (host)",
            "at most one",
        ),
        (
            vec![window(host, None)],
            "kv.window[0] (host)",
            "needs a byte",
        ),
        (
            vec![window(host, bytes(0))],
            "kv.window[0] (host)",
            "more than zero",
        ),
        (
            vec![window(vram, bytes(0))],
            "kv.window[0] (vram)",
            "more than zero",
        ),
        (
            vec![window(vram, Some(Percent(0.5)))],
            "kv.window[0] (vram)",
            "not a share",
        ),
        (
            vec![WindowEntry::default()],
            "kv.window[0]",
            "`backend` is required",
        ),
    ];
    for (entries, entry, why) in cases {
        let err = with_windows(entries).layout().expect_err(why);
        assert!(err.contains(entry) && err.contains(why), "{err}");
    }
}

#[test]
fn a_storage_backend_in_a_window_points_at_the_block_section() {
    for backend in [KvBackend::Redb, KvBackend::Redis] {
        let err = with_windows(vec![window(backend, None)])
            .layout()
            .expect_err("a store is not a window");
        assert!(
            err.contains("kv.window[0]") && err.contains("[[kv.block]]"),
            "{err}"
        );
    }
    // Only the harness's switch places a window in a store, and it does not accept this either.
    let mut kv = with_windows(vec![window(KvBackend::Redb, None)]);
    kv.block = vec![block(KvBackend::Redb)];
    kv.store_backed = true;
    assert!(kv.layout().is_err());
}

#[test]
fn a_window_without_a_backend_is_a_load_error() {
    for text in [
        "[[kv.window]]\nsize = \"1g\"\n",
        "[[kv.window]]\nbackend = \"vram\"\n\n[[kv.window]]\nsize = \"1g\"\n",
    ] {
        let err = file_err(text);
        assert!(
            err.contains("kv.window.") && err.contains("backend"),
            "{err}"
        );
    }
    let err = file_err("[[kv.block]]\npath = \"/tmp/kv.redb\"\n");
    assert!(err.contains("kv.block.0.backend"), "{err}");

    let mut partial = PartialConfig::default();
    partial.set_path("kv.window.0.size", "1g").expect("sets");
    let kv = Config::load_from_layers(&[partial]).kv;
    assert!(kv
        .layout()
        .expect_err("no backend")
        .contains("kv.window[0]"));
}

#[test]
fn an_empty_list_entry_reaches_validation_with_its_index() {
    for (text, needle) in [
        ("[[kv.window]]\n", "kv.window.0.backend"),
        (
            "[[kv.window]]\nbackend = \"vram\"\n\n[[kv.window]]\n",
            "kv.window.1.backend",
        ),
        ("[[kv.block]]\n", "kv.block.0.backend"),
        (
            "[[kv.block]]\nbackend = \"redb\"\n\n[[kv.block]]\n",
            "kv.block.1.backend",
        ),
    ] {
        let err = file_err(text);
        assert!(err.contains(needle), "{text}: {err}");
    }
}

#[test]
fn a_window_entry_with_a_path_or_url_is_an_error_naming_entry_and_field() {
    for (text, needle) in [
        (
            "[[kv.window]]\nbackend = \"host\"\nsize = \"1g\"\npath = \"/tmp/x\"\n",
            "kv.window.0.path",
        ),
        (
            "[[kv.window]]\nbackend = \"vram\"\n\n[[kv.window]]\nbackend = \"host\"\nsize = \"1g\"\nurl = \"redis://x\"\n",
            "kv.window.1.url",
        ),
    ] {
        let err = file_err(text);
        assert!(err.starts_with(needle), "{text}: {err}");
        assert!(err.contains("[[kv.block]]"), "{err}");
    }
}

#[test]
fn a_kv_backend_name_is_validated_at_parse() {
    assert_eq!("VRAM".parse(), Ok(KvBackend::Vram));
    assert_eq!("valkey".parse(), Ok(KvBackend::Redis));
    let err = "lmcache".parse::<KvBackend>().expect_err("unknown");
    assert!(err.contains("lmcache"), "{err}");
    assert!(file::parse_str(
        "[[kv.window]]\nbackend = \"lmcache\"\n",
        Path::new("infr.toml")
    )
    .is_err());
    assert!(file::parse_str(
        "[[kv.block]]\nbackend = \"lmcache\"\n",
        Path::new("infr.toml")
    )
    .is_err());
}

#[test]
fn a_block_list_parses_from_a_config_file_and_a_typo_reaches_the_warning() {
    let (partial, warnings) = parsed(
        "[kv]\ntype_k = \"q8_0\"\n\n[[kv.block]]\nbackend = \"redb\"\nsize = \"1g\"\n\n\
         [[kv.block]]\nbackend = \"redis\"\nurl = \"redis://127.0.0.1:6379/2\"\nbogus = 1\n",
    );
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("kv.block.1.bogus"), "{}", warnings[0]);
    let blocks = partial.kv.block.expect("two blocks");
    assert_eq!(blocks[0].bytes(), Some(GIB));
    assert!(
        blocks[0].path.is_none(),
        "no path keeps the database in memory"
    );
    assert_eq!(blocks[1].url.as_deref(), Some("redis://127.0.0.1:6379/2"));
    assert_eq!(blocks[1].bytes(), None);
}

#[test]
fn block_chains_keep_the_store_rules() {
    use SizeSpec::Percent;
    let redb = || block(KvBackend::Redb);
    for ok in [
        vec![],
        vec![redb()],
        vec![redb(), redis("redis://x")],
        vec![redis("valkey://x")],
        vec![KvBlock {
            size: bytes(MIB),
            ..redb()
        }],
    ] {
        assert!(with_blocks(ok.clone()).layout().is_ok(), "{ok:?}");
    }
    let cases = [
        (
            vec![block(KvBackend::Vram)],
            "kv.block[0] (vram)",
            "[[kv.window]]",
        ),
        (
            vec![redb(), block(KvBackend::Host)],
            "kv.block[1] (host)",
            "[[kv.window]]",
        ),
        (
            vec![block(KvBackend::Redis)],
            "kv.block[0] (redis)",
            "needs a `url`",
        ),
        (
            vec![KvBlock {
                path: Some("/tmp/kv".into()),
                ..redis("redis://x")
            }],
            "kv.block[0] (redis)",
            "`path` belongs to a redb",
        ),
        (
            vec![KvBlock {
                size: bytes(GIB),
                ..redis("redis://x")
            }],
            "kv.block[0] (redis)",
            "budget belongs to the server",
        ),
        (
            vec![KvBlock {
                url: Some("redis://x".into()),
                ..redb()
            }],
            "kv.block[0] (redb)",
            "`url` belongs to a redis",
        ),
        (
            vec![KvBlock {
                size: bytes(0),
                ..redb()
            }],
            "kv.block[0] (redb)",
            "more than zero",
        ),
        (
            vec![KvBlock {
                size: Some(Percent(0.5)),
                ..redb()
            }],
            "kv.block[0] (redb)",
            "not a share",
        ),
        (
            vec![redis("redis://x"), redb()],
            "kv.block[0] (redis)",
            "cannot have another store below it",
        ),
        (
            vec![redb(), redis("redis://x"), redb()],
            "kv.block[1] (redis)",
            "cannot have another store below it",
        ),
    ];
    for (blocks, entry, why) in cases {
        let err = with_blocks(blocks).layout().expect_err(why);
        assert!(err.contains(entry) && err.contains(why), "{err}");
    }
}

#[test]
fn pages_need_a_positive_byte_size() {
    for (size, why) in [
        (bytes(0), "more than zero"),
        (Some(SizeSpec::Percent(0.5)), "not a share"),
    ] {
        let mut kv = Config::default().kv;
        kv.pages = KvPages {
            declared: true,
            size,
        };
        let err = kv.layout().expect_err(why);
        assert!(err.contains("kv.pages") && err.contains(why), "{err}");
    }
    let mut kv = Config::default().kv;
    kv.pages = KvPages {
        declared: true,
        size: None,
    };
    assert_eq!(kv.layout().unwrap().pages, None);
}

#[test]
fn a_pages_section_declares_paging_even_with_no_keys() {
    let kv = cfg_of("[kv.pages]\n").kv;
    assert!(kv.pages.declared && kv.pages.size.is_none());
    assert!(kv.declares_paging());
    let kv = cfg_of("[kv.pages]\nsize = \"2g\"\n").kv;
    assert!(kv.pages.declared);
    assert!(!Config::default().kv.declares_paging());
}

#[test]
fn pages_beside_paged_false_is_a_load_error_naming_both() {
    let kv = cfg_of("[kv]\npaged = false\n\n[kv.pages]\nsize = \"1g\"\n").kv;
    let ConfigError::Value { message, .. } = kv.check().expect_err("contradiction") else {
        panic!("a value error");
    };
    assert!(
        message.contains("[kv.pages]") && message.contains("kv.paged = false"),
        "{message}"
    );
}

#[test]
fn a_store_backed_window_comes_from_the_first_block() {
    let backed = |blocks| {
        let mut kv = with_blocks(blocks);
        kv.store_backed = true;
        kv.layout()
    };
    let memory = backed(vec![block(KvBackend::Redb)]).expect("valid");
    assert!(memory.store_backed);
    assert_eq!(
        memory.window,
        WindowPlan {
            device_cap: Some(0),
            host_cap: Some(KV_REDB_DEFAULT_BYTES)
        }
    );
    let sized = backed(vec![KvBlock {
        size: bytes(2 * GIB),
        ..block(KvBackend::Redb)
    }])
    .expect("valid");
    assert_eq!(sized.window.host_cap, Some(2 * GIB));
    let remote = backed(vec![redis("redis://127.0.0.1:6379")]).expect("valid");
    assert_eq!(remote.window.host_cap, Some(u64::MAX));

    assert!(backed(vec![]).is_err(), "a window needs a store to back it");
    assert!(backed(vec![redis("redis://x"), block(KvBackend::Redb)]).is_err());
    let mut windowed = with_blocks(vec![block(KvBackend::Redb)]);
    windowed.store_backed = true;
    windowed.window = vec![window(KvBackend::Vram, None)];
    assert!(windowed.layout().is_err(), "the store owns the window");
}

#[test]
fn store_backed_has_no_file_or_set_spelling() {
    let err = file_err("[kv]\nstore_backed = true\n");
    assert!(
        err.contains("kv.store_backed") && err.contains("harness"),
        "{err}"
    );
    let overrides = ConfigOverrides {
        sets: vec!["kv.store_backed=true".to_string()],
        ..ConfigOverrides::default()
    };
    assert!(cli::parse_reporting(&overrides).is_err());
}

#[test]
fn window_capacity_is_the_device_room_plus_the_host_part() {
    let plan = |device_cap, host_cap| WindowPlan {
        device_cap,
        host_cap,
    };
    // Unsized: the device room the fit leaves, never zero.
    assert_eq!(plan(None, None).capacity(5 * GIB), 5 * GIB);
    assert_eq!(plan(None, Some(2 * GIB)).capacity(5 * GIB), 7 * GIB);
    // Sized: the cap, but never more than the room there is.
    assert_eq!(plan(Some(GIB), Some(2 * GIB)).capacity(5 * GIB), 3 * GIB);
    assert_eq!(plan(Some(9 * GIB), None).capacity(5 * GIB), 5 * GIB);
    assert_eq!(plan(Some(0), Some(2 * GIB)).capacity(5 * GIB), 2 * GIB);
}

#[test]
fn retained_pages_extend_the_last_memory_a_plan_names() {
    let plan = |device_cap, host_cap| WindowPlan {
        device_cap,
        host_cap,
    };
    let pages = Some(GIB);
    assert_eq!(
        plan(Some(GIB), Some(2 * GIB)).with_pages(pages).host_cap,
        Some(3 * GIB)
    );
    assert_eq!(
        plan(Some(GIB), None).with_pages(pages).device_cap,
        Some(2 * GIB)
    );
    assert_eq!(plan(None, None).with_pages(pages), plan(None, None));
    assert_eq!(
        plan(Some(GIB), None).with_pages(None),
        plan(Some(GIB), None)
    );
}

// ── removed keys ─────────────────────────────────────────────────────────────

#[test]
fn every_removed_kv_key_is_a_load_error_with_its_replacement() {
    let cases = [
        (
            "[kv]\noverflow = true\n",
            "kv.overflow",
            &["[[kv.window]]", "backend = \"host\"", "unbounded"][..],
        ),
        (
            "[kv]\noverflow = true\noverflow_vram_mb = 512\n",
            "kv.overflow",
            &["backend = \"vram\"", "size = \"512m\"", "unbounded"],
        ),
        (
            "[kv]\noverflow_vram_mb = 512\n",
            "kv.overflow_vram_mb",
            &["size = \"512m\"", "[[kv.window]]"],
        ),
        (
            "[kv]\noverflow_reserve_mb = 128\n",
            "kv.overflow_reserve_mb",
            &["no replacement", "fixed constant"],
        ),
        (
            "[kv]\npool_mb = 1024\n",
            "kv.pool_mb",
            &[
                "[[kv.window]]",
                "[kv.pages]",
                "size = \"1024m\"",
                "windows plus the retained pages",
            ],
        ),
        (
            "[[kv.tiers]]\nbackend = \"redb\"\n",
            "kv.tiers",
            &["[[kv.block]]", "backend = \"redb\""],
        ),
    ];
    for (text, path, needles) in cases {
        let err = file_err(text);
        assert!(err.starts_with(path), "{path}: {err}");
        for needle in needles {
            assert!(err.contains(needle), "{path} lacks {needle:?}: {err}");
        }
    }
}

#[test]
fn an_old_tier_list_is_rewritten_into_the_new_sections() {
    let err = file_err(
        "[[kv.tiers]]\nbackend = \"vram\"\nsize = \"1g\"\n\n[[kv.tiers]]\nbackend = \"host\"\nsize = \"4g\"\n\n\
         [[kv.tiers]]\nbackend = \"redb\"\npath = \"/tmp/kv.redb\"\nsize = \"2g\"\n\n\
         [[kv.tiers]]\nbackend = \"redis\"\nurl = \"redis://x\"\n",
    );
    for needle in [
        "[[kv.window]]\nbackend = \"vram\"\nsize = \"1g\"",
        "[[kv.window]]\nbackend = \"host\"\nsize = \"4g\"",
        "[[kv.block]]\nbackend = \"redb\"\npath = \"/tmp/kv.redb\"\nsize = \"2g\"",
        "[[kv.block]]\nbackend = \"redis\"\nurl = \"redis://x\"",
    ] {
        assert!(err.contains(needle), "lacks {needle:?}: {err}");
    }

    // A lone `vram` entry only ever reserved retained pages.
    let lone = file_err("[[kv.tiers]]\nbackend = \"vram\"\nsize = \"2g\"\n");
    assert!(lone.contains("[kv.pages]\nsize = \"2g\""), "{lone}");
    assert!(!lone.contains("[[kv.window]]\nbackend"), "{lone}");

    let host = file_err("[[kv.tiers]]\nbackend = \"host\"\nsize = \"8g\"\n");
    assert!(
        host.contains("[[kv.window]]\nbackend = \"host\"\nsize = \"8g\""),
        "{host}"
    );
}

/// The TOML between `after` and the next blank-line-separated prose paragraph.
fn snippet<'a>(message: &'a str, after: &str, until: &str) -> &'a str {
    let (_, rest) = message.split_once(after).expect("snippet start");
    rest.split_once(until).map_or(rest, |(toml, _)| toml)
}

#[test]
fn a_windows_path_in_an_old_tier_list_becomes_toml_that_loads() {
    let path = r"C:\Users\me\kv\cache.redb";
    let escaped = toml::Value::String(path.to_string()).to_string();
    let text = format!("[[kv.tiers]]\nbackend = \"redb\"\npath = {escaped}\nsize = \"2g\"\n");
    let err = file_err(&text);
    let replacement = snippet(&err, "Your list becomes:\n\n", "\u{0}");
    let (partial, warnings) = parsed(replacement);
    assert!(warnings.is_empty(), "{warnings:?}");
    let blocks = partial.kv.block.expect("one block");
    assert_eq!(blocks[0].path.as_deref(), Some(Path::new(path)));
    assert_eq!(blocks[0].bytes(), Some(2 * GIB));
}

#[test]
fn a_zero_device_cap_migrates_to_the_host_window_alone() {
    for text in [
        "[kv]\noverflow_vram_mb = 0\n",
        "[kv]\noverflow = true\noverflow_vram_mb = 0\n",
    ] {
        let err = file_err(text);
        let replacement = snippet(&err, "instead:\n\n", "The old host side");
        assert!(!replacement.contains("0m"), "{replacement}");
        let (partial, _) = parsed(&replacement.replace("<cap>", "4g"));
        let layout = Config::load_from_layers(&[partial])
            .kv
            .layout()
            .expect("the emitted snippet validates");
        assert_eq!(layout.window.host_cap, Some(4 * GIB));
        assert_eq!(layout.window.device_cap, Some(0), "every row sits on host");
    }
}

#[test]
fn removed_environment_variables_are_errors_naming_the_replacement() {
    for (key, needle) in [
        ("INFR_KV_OVERFLOW", "[[kv.window]]"),
        ("INFR_KV_OVERFLOW_VRAM_MB", "[[kv.window]]"),
        ("INFR_KV_OVERFLOW_RESERVE_MB", "kv.overflow_reserve_mb"),
        ("INFR_KV_POOL_MB", "[kv.pages]"),
    ] {
        let err = env::parse(&|k| (k == key).then(|| "1".to_string()))
            .expect_err(key)
            .to_string();
        assert!(err.starts_with(key) && err.contains(needle), "{err}");
    }
}

#[test]
fn removed_set_paths_are_errors_naming_the_replacement() {
    for (set, needle) in [
        ("kv.tiers=redb", "[[kv.block]]"),
        ("kv.tiers.0.backend=redb", "[[kv.block]]"),
        ("kv.overflow=true", "[[kv.window]]"),
        ("kv.overflow_vram_mb=512", "[[kv.window]]"),
        ("kv.overflow_reserve_mb=128", "no replacement"),
        ("kv.pool_mb=64", "[kv.pages]"),
    ] {
        let overrides = ConfigOverrides {
            sets: vec![set.to_string()],
            ..ConfigOverrides::default()
        };
        let err = cli::parse_reporting(&overrides).expect_err(set).to_string();
        assert!(err.contains(needle), "{set}: {err}");
    }
    // The new keys take `--set` too.
    let overrides = ConfigOverrides {
        sets: vec![
            "kv.window.0.backend=host".to_string(),
            "kv.window.0.size=8g".to_string(),
            "kv.pages=2g".to_string(),
            "kv.block=redb".to_string(),
        ],
        ..ConfigOverrides::default()
    };
    let (layer, _) = cli::parse_reporting(&overrides).expect("sets");
    let kv = Config::load_from_layers(&[layer]).kv;
    let layout = kv.layout().expect("valid");
    assert_eq!(layout.window.host_cap, Some(8 * GIB));
    assert_eq!(layout.pages, Some(2 * GIB));
    assert_eq!(layout.blocks.len(), 1);
}

// ── the paging matrix ────────────────────────────────────────────────────────

fn turbo(side: &str, paged: Option<bool>, pages: bool) -> super::KvCfg {
    let mut kv = Config::default().kv;
    match side {
        "k" => kv.type_k = Some(DType::Turbo3),
        _ => kv.type_v = Some(DType::Turbo2),
    }
    kv.paged = paged;
    kv.pages.declared = pages;
    kv
}

#[test]
fn paging_is_on_by_default() {
    let kv = Config::default().kv;
    assert_eq!(kv.paging(), Paging::On);
    assert!(kv.check().is_ok());
    // A format that does page changes nothing.
    let mut q8 = Config::default().kv;
    q8.type_k = Some(DType::Q8_0);
    assert_eq!(q8.paging(), Paging::On);
}

#[test]
fn paged_false_is_linear_for_any_format() {
    let mut kv = Config::default().kv;
    kv.paged = Some(false);
    assert_eq!(kv.paging(), Paging::Off);
    assert!(kv.check().is_ok());
    for side in ["k", "v"] {
        let kv = turbo(side, Some(false), false);
        assert_eq!(kv.paging(), Paging::Off, "{side}");
        assert!(kv.check().is_ok(), "an explicit opt-out is not an error");
    }
}

#[test]
fn naming_a_turbo_format_alone_turns_paging_off_for_the_run() {
    for side in ["k", "v"] {
        let kv = turbo(side, None, false);
        assert_eq!(kv.paging(), Paging::OffForTurbo, "{side}");
        assert!(!kv.paging().is_on());
        assert!(
            kv.check().is_ok(),
            "the default case warns, it does not fail"
        );
    }
}

#[test]
fn asking_for_paging_beside_a_turbo_format_is_a_load_error_naming_both() {
    for (side, key) in [("k", "kv.type_k"), ("v", "kv.type_v")] {
        let explicit = turbo(side, Some(true), false);
        let ConfigError::Value { message, .. } = explicit.check().expect_err("conflict") else {
            panic!("a value error");
        };
        assert!(
            message.contains("kv.paged = true") && message.contains(key),
            "{message}"
        );

        let pages = turbo(side, None, true);
        assert_eq!(pages.paging(), Paging::On);
        let ConfigError::Value { message, .. } = pages.check().expect_err("conflict") else {
            panic!("a value error");
        };
        assert!(
            message.contains("[kv.pages]") && message.contains(key),
            "{message}"
        );
    }
}

#[test]
fn a_file_that_names_both_fails_at_the_whole_load_check() {
    let kv = cfg_of("[kv]\npaged = true\ntype_k = \"turbo3\"\n").kv;
    assert!(kv.check().is_err());
}
