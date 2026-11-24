//! The KV cache's three sections and what they resolve to.
//!
//! - `[[kv.window]]`: the live rows. The order is the placement ladder, `vram` first, at most one
//!   of each backend. Attention reads every window row on every step, so only memory the kernels
//!   address directly (`vram`, `host`) can hold one.
//! - `[kv.pages]`: the retention reserve for parked page maps, on top of the windows.
//! - `[[kv.block]]`: the 128-token prefix extents of the cross-conversation cache, a demotion
//!   chain of storage backends (`redb`, `redis`).
//!
//! [`KvCfg::layout`] folds the three into one [`KvLayout`]; [`KvCfg::paging`] is the one place that
//! decides whether the session pages.

use std::path::PathBuf;

use crate::{DType, SizeSpec};

use super::{ConfigError, KvCfg};

/// Bytes a `redb` block store keeps when its entry names no `size`.
pub const KV_REDB_DEFAULT_BYTES: u64 = 1 << 30;

/// What a `[[kv.window]]` or `[[kv.block]]` entry is backed by. `vram` and `host` are addressable
/// (a window's memory); `redb` and `redis` are storage (a block chain's).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvBackend {
    /// Device memory.
    Vram,
    /// Host memory the kernels address directly.
    Host,
    /// Local: a file, or memory when `path` is unset.
    Redb,
    /// A shared server (`valkey` is accepted as a spelling).
    Redis,
}

impl KvBackend {
    /// Whether the backend is memory the kernels address (a window), not a store.
    pub const fn is_addressable(self) -> bool {
        matches!(self, Self::Vram | Self::Host)
    }
}

impl std::str::FromStr for KvBackend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "vram" => Ok(Self::Vram),
            "host" => Ok(Self::Host),
            "redb" => Ok(Self::Redb),
            "redis" | "valkey" => Ok(Self::Redis),
            other => Err(format!(
                "unknown kv backend {other:?} (vram, host, redb, redis)"
            )),
        }
    }
}

impl std::fmt::Display for KvBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Vram => "vram",
            Self::Host => "host",
            Self::Redb => "redb",
            Self::Redis => "redis",
        })
    }
}

/// One `[[kv.window]]` entry: a memory the live rows may sit in.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WindowEntry {
    /// `vram` or `host`. `None` only while a config is being assembled; every entry has to name
    /// one, and a config that loads without one is an error.
    pub backend: Option<KvBackend>,
    /// Bytes of live rows this memory takes. Optional for `vram` (omitted = the device room the fit
    /// leaves), required for `host`.
    pub size: Option<SizeSpec>,
}

impl WindowEntry {
    /// A one-line description for logs: the backend and the cap it states.
    pub fn describe(&self) -> String {
        let backend = self
            .backend
            .map_or_else(|| "?".to_string(), |b| b.to_string());
        match self.size {
            Some(SizeSpec::Bytes(b)) => format!("{backend} {b} bytes"),
            Some(SizeSpec::Percent(p)) => format!("{backend} {:.1}%", p * 100.0),
            None => format!("{backend} unsized"),
        }
    }
}

/// `[kv.pages]`: the retention reserve for parked page maps, above the windows.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KvPages {
    /// Whether the section was written, with or without a `size`. A written section declares
    /// paging, like `kv.paged = true` does.
    pub declared: bool,
    /// Bytes of retained pages on top of the windows. `None` = the pool is just the windows.
    pub size: Option<SizeSpec>,
}

/// One `[[kv.block]]` entry: a place the cross-conversation prefix cache keeps 128-token extents.
///
/// The list is the chain: entry 0 is the fastest, an eviction from it demotes into entry 1, and so
/// on; a read that misses entry 0 falls through and promotes back up. Every entry but the last has
/// to be able to evict, which a `redis` entry cannot: the server owns that policy.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KvBlock {
    /// Where this entry keeps blocks: `redb` or `redis`. `None` only while a config is being
    /// assembled; every entry has to name one.
    pub backend: Option<KvBackend>,
    /// `redb`: bytes kept before evicting (or demoting, when an entry follows). Default 1 GiB.
    pub size: Option<SizeSpec>,
    /// `redb`: the database file. Unset keeps the database in memory.
    pub path: Option<PathBuf>,
    /// `redis`: the server URL (`redis://`, `rediss://`, `valkey://`).
    pub url: Option<String>,
}

impl KvBlock {
    /// The bytes this entry keeps, or `None` for an entry with no budget.
    pub fn bytes(&self) -> Option<u64> {
        match self.size {
            Some(SizeSpec::Bytes(b)) => Some(b),
            _ => None,
        }
    }

    /// The backend's name for logs and messages.
    fn backend_name(&self) -> String {
        self.backend
            .map_or_else(|| "?".to_string(), |b| b.to_string())
    }

    /// A one-line description for logs, with any credentials stripped from a `url`.
    pub fn describe(&self) -> String {
        let where_ = self
            .path
            .as_ref()
            .map(|p| p.display().to_string())
            .or_else(|| self.url.as_deref().map(redact_url))
            .unwrap_or_else(|| "memory".to_string());
        match self.bytes() {
            Some(b) => format!("{} {where_} ({b} bytes)", self.backend_name()),
            None => format!("{} {where_}", self.backend_name()),
        }
    }

    /// Bytes of live window this entry can back when it leads a store-backed layout: the `redb`
    /// `size`, unbounded for `redis`, whose server owns its budget.
    fn window_bytes(&self) -> u64 {
        match self.backend {
            Some(KvBackend::Redis) => u64::MAX,
            _ => self.bytes().unwrap_or(KV_REDB_DEFAULT_BYTES),
        }
    }
}

/// `scheme://user:secret@host` as `scheme://host`: a URL in a log line keeps its shape and loses
/// its credentials.
fn redact_url(url: &str) -> String {
    match url.split_once("://") {
        Some((scheme, rest)) => match rest.rsplit_once('@') {
            Some((_, host)) => format!("{scheme}://{host}"),
            None => url.to_string(),
        },
        None => url.to_string(),
    }
}

/// How the live rows split between device and host memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WindowPlan {
    /// Device bytes the live rows may take before the rest goes to host. `None` = the device room
    /// the fit leaves (an unsized or absent `vram` window), never zero.
    pub device_cap: Option<u64>,
    /// Host bytes the live rows may take. `None` = no `host` window, so nothing goes to host.
    pub host_cap: Option<u64>,
}

impl WindowPlan {
    /// Bytes of live rows the plan holds when `device_room` bytes of device memory are free for
    /// them: the device part (the room itself when unsized, never more than it) plus the host part.
    pub fn capacity(&self, device_room: u64) -> u64 {
        self.device_cap
            .map_or(device_room, |cap| cap.min(device_room))
            .saturating_add(self.host_cap.unwrap_or(0))
    }

    /// The plan the allocator places KV buffers by once `pages` bytes of retained pages ride on top
    /// of the windows: the extra bytes join the last memory the plan names, host when a `host`
    /// window exists, else the sized device part. A plan with no bound has nothing to extend.
    #[must_use]
    pub fn with_pages(mut self, pages: Option<u64>) -> Self {
        let extra = pages.unwrap_or(0);
        if let Some(host) = &mut self.host_cap {
            *host = host.saturating_add(extra);
        } else if let Some(device) = &mut self.device_cap {
            *device = device.saturating_add(extra);
        }
        self
    }
}

/// The KV cache's three sections, resolved and validated.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KvLayout {
    /// Where the live rows sit.
    pub window: WindowPlan,
    /// Bytes of retained pages above the windows; `None` = the pool is just the windows.
    pub pages: Option<u64>,
    /// The prefix store chain, fastest first. Empty = no prefix store.
    pub blocks: Vec<KvBlock>,
    /// Set when the first block is the authoritative copy of each slot's live window: the rows are
    /// host-resident and synchronised with that store every step. A measurement experiment, only
    /// ever set programmatically ([`KvCfg::store_backed`]).
    pub store_backed: bool,
}

impl KvLayout {
    /// The plan the allocator places KV buffers by: the windows plus the retention reserve.
    pub fn placement(&self) -> WindowPlan {
        self.window.with_pages(self.pages)
    }
}

/// Whether the session pages its KV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Paging {
    /// One page pool shared by the slots, where a multi-slot session on a page-capable backend has
    /// a poolable layer.
    On,
    /// Every slot owns a contiguous cache.
    Off,
    /// A TurboQuant cache is named and nothing asked for paging, so the run stays linear.
    OffForTurbo,
}

impl Paging {
    /// Whether the session builds a page pool where it can.
    pub const fn is_on(self) -> bool {
        matches!(self, Self::On)
    }
}

/// A TurboQuant KV format (no page-aware store or dequant).
const fn is_turbo(dtype: Option<DType>) -> bool {
    matches!(dtype, Some(DType::Turbo2 | DType::Turbo3 | DType::Turbo4))
}

/// The byte count of a size that has to be bytes above zero, as the entry's error names it.
fn positive_bytes(name: &str, size: SizeSpec) -> Result<u64, String> {
    match size {
        SizeSpec::Bytes(b) if b > 0 => Ok(b),
        SizeSpec::Bytes(_) => Err(format!("{name}: `size` has to be more than zero")),
        SizeSpec::Percent(_) => Err(format!("{name}: `size` is bytes, not a share of something")),
    }
}

/// Resolve `[[kv.window]]`. `vram` is valid only first, `host` first or right after a `vram`
/// entry, and neither repeats; a storage backend belongs in `[[kv.block]]`.
fn window_plan(entries: &[WindowEntry]) -> Result<WindowPlan, String> {
    let (mut vram, mut host) = (None, None);
    for (i, entry) in entries.iter().enumerate() {
        let name = format!("kv.window[{i}]");
        let Some(backend) = entry.backend else {
            return Err(format!("{name}: `backend` is required (vram or host)"));
        };
        let name = format!("{name} ({backend})");
        match backend {
            KvBackend::Redb | KvBackend::Redis => {
                return Err(format!(
                    "{name}: a {backend} store keeps blocks, not live rows; declare it under \
                     `[[kv.block]]`"
                ));
            }
            KvBackend::Vram if i != 0 => {
                return Err(format!(
                    "{name}: the vram window comes first, and there is at most one"
                ));
            }
            KvBackend::Host
                if i != 0 && !(i == 1 && entries[0].backend == Some(KvBackend::Vram)) =>
            {
                return Err(format!(
                    "{name}: the host window comes first or right after the vram window, and \
                     there is at most one"
                ));
            }
            KvBackend::Vram => {
                vram = entry
                    .size
                    .map(|size| positive_bytes(&name, size))
                    .transpose()?;
            }
            KvBackend::Host => {
                let size = entry
                    .size
                    .ok_or_else(|| format!("{name}: a host window needs a byte `size`"))?;
                host = Some(positive_bytes(&name, size)?);
            }
        }
    }
    let has_vram = entries.first().and_then(|e| e.backend) == Some(KvBackend::Vram);
    Ok(WindowPlan {
        // A host window with no vram window before it holds every row; a vram window alone, or
        // an absent one, takes whatever device room the fit leaves.
        device_cap: if host.is_some() && !has_vram {
            Some(0)
        } else {
            vram
        },
        host_cap: host,
    })
}

/// Validate `[[kv.block]]`: storage backends only, each with its own fields, and no `redis` entry
/// above another store (the server owns its eviction, so it cannot be the bounded half of a chain).
fn check_blocks(blocks: &[KvBlock]) -> Result<(), String> {
    for (i, block) in blocks.iter().enumerate() {
        let name = format!("kv.block[{i}]");
        let Some(backend) = block.backend else {
            return Err(format!("{name}: `backend` is required (redb or redis)"));
        };
        let name = format!("{name} ({backend})");
        match backend {
            KvBackend::Vram | KvBackend::Host => {
                return Err(format!(
                    "{name}: {backend} memory holds live rows, not blocks; declare it under \
                     `[[kv.window]]`"
                ));
            }
            KvBackend::Redb => {
                if block.url.is_some() {
                    return Err(format!(
                        "{name}: a redb block store keeps its blocks in `path`, and `url` \
                         belongs to a redis one"
                    ));
                }
                if let Some(size) = block.size {
                    positive_bytes(&name, size)?;
                }
            }
            KvBackend::Redis => {
                if block.path.is_some() {
                    return Err(format!(
                        "{name}: a redis block store is named by `url`, and `path` belongs to a \
                         redb one"
                    ));
                }
                if block.url.is_none() {
                    return Err(format!("{name}: a redis block store needs a `url`"));
                }
                if block.size.is_some() {
                    return Err(format!(
                        "{name}: a redis block store's budget belongs to the server, so it takes \
                         no `size`"
                    ));
                }
                if i + 1 < blocks.len() {
                    return Err(format!(
                        "{name}: a redis block store cannot have another store below it, because \
                         the server owns its budget and so it cannot be the bounded half of a \
                         chain; declare the `redb` entry first"
                    ));
                }
            }
        }
    }
    Ok(())
}

impl KvCfg {
    /// Whether the config asks for paging by name: `kv.paged = true` or a `[kv.pages]` section.
    pub fn declares_paging(&self) -> bool {
        self.paged == Some(true) || self.pages.declared
    }

    /// Whether either KV side names a TurboQuant format.
    pub fn names_turbo(&self) -> bool {
        is_turbo(self.type_k) || is_turbo(self.type_v)
    }

    /// Whether the session pages its KV. Paging is the default: only `kv.paged = false`, or a
    /// TurboQuant format named with nothing asking for paging, turns it off. Whether a given
    /// session CAN page (several slots, a page-capable backend, a poolable layer) is the seam's.
    pub fn paging(&self) -> Paging {
        match self.paged {
            Some(false) => Paging::Off,
            Some(true) => Paging::On,
            None if self.pages.declared => Paging::On,
            None if self.names_turbo() => Paging::OffForTurbo,
            None => Paging::On,
        }
    }

    /// Resolve and validate `[[kv.window]]`, `[kv.pages]` and `[[kv.block]]`.
    pub fn layout(&self) -> Result<KvLayout, String> {
        check_blocks(&self.block)?;
        let pages = self
            .pages
            .size
            .map(|size| positive_bytes("kv.pages", size))
            .transpose()?;
        let window = window_plan(&self.window)?;
        if !self.store_backed {
            return Ok(KvLayout {
                window,
                pages,
                blocks: self.block.clone(),
                store_backed: false,
            });
        }
        if !self.window.is_empty() {
            return Err(
                "a store-backed live window is placed by the first `[[kv.block]]`, so no \
                 `[[kv.window]]` may be declared beside it"
                    .to_string(),
            );
        }
        let first = self
            .block
            .first()
            .ok_or("a store-backed live window needs a `[[kv.block]]` store to back it")?;
        Ok(KvLayout {
            window: WindowPlan {
                device_cap: Some(0),
                host_cap: Some(first.window_bytes()),
            },
            pages,
            blocks: self.block.clone(),
            store_backed: true,
        })
    }

    /// The load-time checks of the whole section: the layout, and the paging matrix.
    pub fn check(&self) -> Result<(), ConfigError> {
        self.layout().map_err(|message| ConfigError::Value {
            path: "kv".to_string(),
            message,
        })?;
        if self.paged == Some(false) && self.pages.declared {
            return Err(ConfigError::Value {
                path: "kv.pages".to_string(),
                message: "`[kv.pages]` reserves retained pages, which `kv.paged = false` turns \
                          off: drop one of them"
                    .to_string(),
            });
        }
        if self.declares_paging() && self.names_turbo() {
            let side = if is_turbo(self.type_k) {
                "kv.type_k"
            } else {
                "kv.type_v"
            };
            let asked = if self.pages.declared && self.paged != Some(true) {
                "`[kv.pages]`"
            } else {
                "`kv.paged = true`"
            };
            return Err(ConfigError::Value {
                path: side.to_string(),
                message: format!(
                    "{asked} and a TurboQuant `{side}` cannot be combined: a TurboQuant cache has \
                     no page-aware store or dequant. Drop one of them, or set `kv.paged = false`"
                ),
            });
        }
        Ok(())
    }
}

/// Why a removed `kv.*` key no longer loads, and what to write instead. The TOML in each message
/// is what to paste; `value` is what the old key held, when the caller knows it.
pub(crate) fn removed_kv_key(path: &str, value: Option<&str>) -> Option<String> {
    let held = |fallback: &str| value.map_or_else(|| fallback.to_string(), str::to_string);
    // `kv.tiers.0.backend` is the removed `kv.tiers`: the first two segments name the key.
    let path = path
        .match_indices('.')
        .nth(1)
        .map_or(path, |(end, _)| &path[..end]);
    Some(match path {
        "kv.tiers" => "`kv.tiers` was removed. Storage entries become `[[kv.block]]` (same \
             `backend`, `path`, `url` and `size`), and a leading `vram` or `host` entry becomes a \
             `[[kv.window]]`; a lone `vram` entry that only reserved retained pages becomes \
             `[kv.pages]` with the same `size`."
            .to_string(),
        "kv.overflow" | "kv.overflow_vram_mb" => {
            // A zero device cap put every row on the host, which the host window alone does.
            let zero = value.and_then(crate::parse_size) == Some(SizeSpec::Bytes(0));
            let vram = match value {
                _ if zero => String::new(),
                Some(size) => {
                    format!("[[kv.window]]\nbackend = \"vram\"\nsize = \"{size}\"\n\n")
                }
                None => "[[kv.window]]\nbackend = \"vram\"\n\
                         # size = \"<N>m\"   # the device part; omit to fill what the fit leaves\n\n"
                    .to_string(),
            };
            format!(
                "`{path}` was removed. Declare the windows instead:\n\n{vram}\
                 [[kv.window]]\nbackend = \"host\"\nsize = \"<cap>\"\n\n\
                 The old host side was unbounded, so a host `size` has to be chosen."
            )
        }
        "kv.overflow_reserve_mb" => "`kv.overflow_reserve_mb` was removed with no replacement: \
             the VRAM guard's device headroom is a fixed constant, and the key never reached it. \
             Delete it."
            .to_string(),
        "kv.pool_mb" => format!(
            "`kv.pool_mb` was removed. The pool is now the windows plus the retained pages, so the \
             old number has to become one of the two. If it capped the whole pool:\n\n\
             [[kv.window]]\nbackend = \"vram\"\nsize = \"{0}\"\n\n\
             If it left room for parked conversations beyond the windows:\n\n\
             [kv.pages]\nsize = \"{0}\"",
            held("<old kv.pool_mb>m")
        ),
        _ => return None,
    })
}

/// [`removed_kv_key`] for an environment variable.
pub(crate) fn removed_kv_env(key: &str) -> Option<String> {
    let path = match key {
        "INFR_KV_OVERFLOW" => "kv.overflow",
        "INFR_KV_OVERFLOW_VRAM_MB" => "kv.overflow_vram_mb",
        "INFR_KV_OVERFLOW_RESERVE_MB" => "kv.overflow_reserve_mb",
        "INFR_KV_POOL_MB" => "kv.pool_mb",
        _ => return None,
    };
    removed_kv_key(path, None).map(|m| format!("{key} was removed. {m}"))
}

/// The environment variables [`removed_kv_env`] refuses.
pub(crate) const REMOVED_KV_ENV: &[&str] = &[
    "INFR_KV_OVERFLOW",
    "INFR_KV_OVERFLOW_VRAM_MB",
    "INFR_KV_OVERFLOW_RESERVE_MB",
    "INFR_KV_POOL_MB",
];

/// A `kv.*` path the file and `--set` refuse because only code sets it.
pub(crate) fn programmatic_kv_key(path: &str) -> Option<&'static str> {
    (path == "kv.store_backed").then_some(
        "`kv.store_backed` is not a config key: only the store-backed measurement harness sets \
         it. Keep the live window in `[[kv.window]]` and the prefix store in `[[kv.block]]`.",
    )
}

/// Every `[[kv.window]]` and `[[kv.block]]` entry of a layer has to name its `backend`: an entry
/// with only a `path` or a `size` is refused rather than guessed. Reports the first offender.
pub(crate) fn check_layer_backends(
    window: Option<&Vec<WindowEntry>>,
    block: Option<&Vec<KvBlock>>,
) -> Result<(), ConfigError> {
    let missing = |section: &str, i: usize, kinds: &str| ConfigError::Value {
        path: format!("kv.{section}.{i}.backend"),
        message: format!("`backend` is required ({kinds})"),
    };
    if let Some(i) = window.and_then(|w| w.iter().position(|e| e.backend.is_none())) {
        return Err(missing("window", i, "vram or host"));
    }
    if let Some(i) = block.and_then(|b| b.iter().position(|e| e.backend.is_none())) {
        return Err(missing("block", i, "redb or redis"));
    }
    Ok(())
}

/// One entry of an old `kv.tiers` list, as the file layer read it.
#[derive(Debug, Default, Clone)]
pub(crate) struct OldTier {
    pub backend: Option<String>,
    pub size: Option<String>,
    pub path: Option<String>,
    pub url: Option<String>,
}

/// The new TOML an old `[[kv.tiers]]` list becomes: storage entries to `[[kv.block]]`, a leading
/// `vram` followed by `host` to two windows, `host` alone to one, a lone `vram` to `[kv.pages]`.
pub(crate) fn tiers_replacement(tiers: &[OldTier]) -> String {
    let backend = |t: &OldTier| t.backend.as_deref().map(str::to_ascii_lowercase);
    let quoted = |v: &str| toml::Value::String(v.to_string()).to_string();
    let mut out = String::new();
    let mut section = |text: String| {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&text);
    };
    let first = tiers.first().and_then(backend);
    let second = tiers.get(1).and_then(backend);
    let window = |kind: &str, size: &Option<String>| {
        let mut text = format!("[[kv.window]]\nbackend = \"{kind}\"\n");
        match size {
            Some(size) => text.push_str(&format!("size = {}\n", quoted(size))),
            None if kind == "host" => text.push_str("size = \"<cap>\"\n"),
            None => {}
        }
        text
    };
    let mut skip = 0;
    match (first.as_deref(), second.as_deref()) {
        (Some("vram"), Some("host")) => {
            section(window("vram", &tiers[0].size));
            section(window("host", &tiers[1].size));
            skip = 2;
        }
        (Some("host"), _) => {
            section(window("host", &tiers[0].size));
            skip = 1;
        }
        (Some("vram"), _) => {
            let size = tiers[0]
                .size
                .as_deref()
                .map_or("<size>".to_string(), quoted);
            section(format!("[kv.pages]\nsize = {size}\n"));
            skip = 1;
        }
        _ => {}
    }
    for tier in &tiers[skip..] {
        let kind = backend(tier).unwrap_or_else(|| "redb".to_string());
        let mut text = format!("[[kv.block]]\nbackend = {}\n", quoted(&kind));
        if let Some(path) = &tier.path {
            text.push_str(&format!("path = {}\n", quoted(path)));
        }
        if let Some(url) = &tier.url {
            text.push_str(&format!("url = {}\n", quoted(url)));
        }
        if let (Some(size), "redb") = (&tier.size, kind.as_str()) {
            text.push_str(&format!("size = {}\n", quoted(size)));
        }
        section(text);
    }
    out
}
