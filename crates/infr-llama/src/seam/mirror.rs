//! A store-backed window (the first `[[kv.block]]` store, switched on programmatically by the
//! measurement harness) becomes the authoritative copy of a slot's live window: after every
//! forward that advances the window, the rows go out to the store and come back, and the slot's
//! buffers are rewritten from the store's answer. That is what makes the placement a property of
//! the config rather than of the engine — the slot holds a staging copy, the store holds the
//! truth.
//!
//! Three deliberate choices, each of which costs speed and buys a property the plan asked for:
//!
//! * The round trip is a *round* trip. Writing the rows out alone would leave the device as the
//!   authority it already is; reading them back through the store's own encoder is what shows the
//!   store can reconstruct the window, and it is what [`LiveWindow::hydrate`] later trusts.
//! * The window's extent id comes from a reserved range ([`super::kv_tier::LIVE_WINDOW_BASE`]),
//!   so a live window and a content-addressed prefix block share one store separated only by id
//!   space: the block index walks blocks, the mirror walks its own ids, and neither can evict the
//!   other.
//! * A failed step fails the generation. The store is the authoritative copy under this
//!   placement, so losing a step would leave the slot describing a window the store does not hold.
//!
//! This is an experiment, not a recommendation: with `kv.paged` it is also redundant (pages
//! resume from the parked pool, the window from the store), and it doubles the window's bytes
//! between the slot and the store. `docs/perf/kv-prefix.md` carries the measured cost.

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, ensure, Result};
use infr_core::backend::Backend;

use crate::Config;

use super::kv_store::BlockCache;
use super::kv_tier::{live_window_id, DeviceKv, KvRows, KvStore, LIVE_WINDOW_SLOTS};
use super::weights::{SeamKv, SlotView};

/// One step of one slot's window into the store behind [`LiveWindow`], taken after every forward
/// that advances the window. `slot` is the addressable view of the slot's rows, built from the
/// fields the runner already holds destructured.
pub(crate) trait Mirror: Send + Sync {
    fn step(&self, be: &dyn Backend, cfg: &Config, slot: SlotView<'_>) -> Result<()>;

    /// Load this slot's stored window into `kv` when the store still holds a prefix of `prompt`.
    /// Returns the rows adopted, 0 when there is nothing to adopt. Called on checkout, before the
    /// cross-request prefix cache gets a chance to restore a block: a window belongs to THIS slot
    /// and is row-exact, where a block restore is a different conversation's prefix.
    fn hydrate(
        &self,
        be: &dyn Backend,
        cfg: &Config,
        kv: &mut SeamKv,
        prompt: &[u32],
    ) -> Result<usize>;
}

/// The shared half of every slot's mirror: the one block cache the engine keeps, behind its lock.
/// A step holds that lock only for the store's own write and read-back — the device transfer on
/// either side of it runs unlocked, so a mirroring slot never blocks the park/relieve paths for the
/// duration of a copy. Hydration is once per checkout rather than per step, and holds it for the
/// whole adoption.
pub(crate) struct LiveWindow {
    cache: Arc<Mutex<BlockCache<Box<dyn KvStore>>>>,
}

impl LiveWindow {
    pub(crate) fn new(cache: Arc<Mutex<BlockCache<Box<dyn KvStore>>>>) -> Self {
        Self { cache }
    }

    pub(crate) fn cache(&self) -> Result<std::sync::MutexGuard<'_, BlockCache<Box<dyn KvStore>>>> {
        self.cache
            .lock()
            .map_err(|_| anyhow!("kv store lock poisoned"))
    }

    /// Load extent `id`'s window into `kv` when it is a prefix of `prompt`. Returns the rows
    /// restored, 0 when the store holds nothing for that extent or nothing that applies.
    pub(crate) fn hydrate(
        &self,
        be: &dyn Backend,
        cfg: &Config,
        kv: &mut SeamKv,
        prompt: &[u32],
        id: u64,
    ) -> Result<usize> {
        self.cache()?.hydrate(be, cfg, kv, prompt, id)
    }

    /// `(steps, bytes written, bytes read, bytes of the last step)`. The engine's own readers go
    /// through [`super::parallel::ParallelSeam::kv_mirror_stats`], so these two are for tests.
    #[cfg(test)]
    pub(crate) fn stats(&self) -> Result<(u64, u64, u64, u64)> {
        let c = self.cache()?;
        Ok((
            c.mirror_steps,
            c.mirror_written,
            c.mirror_read,
            c.mirror_last,
        ))
    }

    /// `(windows adopted, rows adopted)`.
    #[cfg(test)]
    pub(crate) fn hydrated(&self) -> Result<(u64, u64)> {
        let c = self.cache()?;
        Ok((c.hydrated, c.hydrated_tokens))
    }
}

/// One slot's mirror: the shared cache, this slot's extent id, and the tokens the extent's side
/// metadata already names. Empty means "this process has written none", so the first step always
/// rewrites the metadata — a window inherited from an older process must not be trusted on this
/// one without the slot re-deriving it.
struct SlotMirror {
    window: Arc<LiveWindow>,
    id: u64,
    recorded: Mutex<Vec<u32>>,
    /// The store's answer for the last step, kept so a step does not allocate it again.
    down: Mutex<KvRows>,
}

impl Mirror for SlotMirror {
    fn step(&self, be: &dyn Backend, cfg: &Config, slot: SlotView<'_>) -> Result<()> {
        let rows = slot.rows;
        if rows == 0 {
            return Ok(());
        }
        // A prefix of the rows, never more than the window: after a rewind the tail rows belong to
        // the previous conversation until the prefill reaches them again, so the metadata keeps
        // naming only what the rows are known to be.
        let tokens = &slot.cached[..slot.cached.len().min(rows)];
        let mut recorded = self
            .recorded
            .lock()
            .map_err(|_| anyhow!("kv mirror: the recorded-token lock is poisoned"))?;
        // The tokens, not just how many: a rewind that keeps the count leaves the extent's rows
        // and its metadata describing different conversations.
        let meta = (recorded.as_slice() != tokens).then_some(tokens);
        let started = std::time::Instant::now();
        // Both transfers run outside the shared cache lock: they touch this slot's buffers, and a
        // copy must not stall every other slot's park/restore for its duration.
        let mut up = KvRows::default();
        DeviceKv::on(self.id, be, cfg, slot).get(self.id, 0..rows, &mut up)?;
        let mut down = self
            .down
            .lock()
            .map_err(|_| anyhow!("kv mirror: the read-back lock is poisoned"))?;
        let bytes = {
            let mut cache = self.window.cache()?;
            let bytes = cache.mirror_put(self.id, up, meta)?;
            cache.mirror_read(self.id, rows, bytes, &mut down)?;
            bytes
        };
        DeviceKv::on(self.id, be, cfg, slot).write(self.id, 0, &down)?;
        if meta.is_some() {
            recorded.clear();
            recorded.extend_from_slice(tokens);
        }
        tracing::trace!(
            id = self.id,
            rows,
            bytes,
            meta = meta.is_some(),
            elapsed_ms = started.elapsed().as_millis() as i64,
            "kv store: mirrored a step"
        );
        Ok(())
    }

    fn hydrate(
        &self,
        be: &dyn Backend,
        cfg: &Config,
        kv: &mut SeamKv,
        prompt: &[u32],
    ) -> Result<usize> {
        let rows = self.window.hydrate(be, cfg, kv, prompt, self.id)?;
        if rows > 0 {
            // The extent's metadata may still name a longer window than the rows adopted here, so
            // there is nothing this process can treat as already written: the next step republishes
            // what the slot actually holds.
            self.recorded
                .lock()
                .map_err(|_| anyhow!("kv mirror: the recorded-token lock is poisoned"))?
                .clear();
        }
        Ok(rows)
    }
}

/// Attach slot `slot`'s mirror. The engine calls this once per slot at startup, after the fork
/// (which deliberately leaves the child without its parent's mirror).
pub(crate) fn attach(kv: &mut SeamKv, window: &Arc<LiveWindow>, slot: usize) -> Result<()> {
    ensure!(
        slot < LIVE_WINDOW_SLOTS as usize,
        "{} slots have no live-window id range (at most {LIVE_WINDOW_SLOTS})",
        slot + 1
    );
    kv.mirror = Some(Arc::new(SlotMirror {
        window: Arc::clone(window),
        id: live_window_id(slot),
        recorded: Mutex::new(Vec::new()),
        down: Mutex::new(KvRows::default()),
    }));
    Ok(())
}
