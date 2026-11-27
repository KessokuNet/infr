//! Embedding models (`docs/plan/22-embedding-models.md`): the one-shot bidirectional prefill and
//! the host-side pooling that turns its per-token head rows into one vector.
//!
//! EmbeddingGemma 2 is the first arch here, and it is a decoder-shaped transformer whose two
//! differences need no new graph ops: its mask is bidirectional (`Config::non_causal`, emitted as
//! `AttnMask::Canvas`/`AttnMask::Banded`), and its head projects to the embedding dimension instead
//! of the vocabulary (`Config::n_embd_out`), so the per-row "logits" the prefill tap captures ARE
//! the per-token embedding vectors. Mean pooling is then host arithmetic — and exact before the
//! projection, which is linear.

use super::runner::generate_dense_backend_bound;
use super::{cpu_upload_bind, TokenEmbd};
use crate::{Config, EngineConfig, PerLayerEmbd};
use anyhow::{bail, Result as AResult};
use infr_core::WeightSource;
use infr_cpu::CpuBackend;

/// Embed `tokens` with the model's own pooling and normalization, on the CPU reference backend.
///
/// `dimensions` truncates the pooled vector to its leading entries (Matryoshka, which is what
/// OpenAI's `dimensions` field asks for) and renormalizes afterwards; `None` keeps the native width.
pub(crate) fn embed_cpu(
    g: &dyn WeightSource,
    cfg: &Config,
    ec: &std::sync::Arc<EngineConfig>,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
    dimensions: Option<usize>,
) -> AResult<Vec<f32>> {
    if !cfg.non_causal || cfg.n_embd_out == 0 {
        bail!(
            "embed: not an embedding model — the GGUF must declare attention.causal = false and \
             embedding_length_out"
        );
    }
    if tokens.is_empty() {
        bail!("embed: empty input");
    }
    let cpu_be = CpuBackend::new_with(ec.clone());
    // The teacher-forced (`verify`) branch is the forward this wants: one batched pass over EVERY
    // token, with the head run over all of its rows. That is what a teacher-forced scoring harness
    // uses to read a whole prompt at once, and a bidirectional embedding pass is the same shape —
    // unlike the ordinary prefill path, which withholds the last prompt token for the decode loop.
    // Dense models only, which every embedding arch is.
    let mut rows = Vec::new();
    let mut state = None;
    generate_dense_backend_bound(
        None, // prebound
        &cpu_be,
        &cpu_upload_bind(&cpu_be),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        tokens,
        0,
        |_| {},
        &mut state,
        tokens.len() + 2,
        None,            // constraint
        Some(&mut rows), // verify: the batched all-rows forward
        None,            // verify_ids
        None,            // logits_out
        None,            // h_out
        None,            // denoise_req
        None,            // req
        None,            // prefill_logits
        None,            // route
        None,            // paged_blocks
        1,               // slot_copies
    )?;
    let dim = cfg.head_rows();
    if rows.len() != tokens.len() * dim {
        bail!(
            "embed: the batched forward returned {} values, not {} rows of {dim}",
            rows.len(),
            tokens.len()
        );
    }
    mean_pool_normalize(&rows, tokens.len(), dim, dimensions)
}

/// Mean-pool `rows` vectors of width `dim`, then L2-normalize; `dimensions` truncates the pooled
/// vector and renormalizes, because slicing a unit vector does not preserve unit length (the model
/// card is explicit that skipping this step degrades ranking silently).
pub(crate) fn mean_pool_normalize(
    vectors: &[f32],
    rows: usize,
    dim: usize,
    dimensions: Option<usize>,
) -> AResult<Vec<f32>> {
    if rows == 0 {
        bail!("embed: no rows to pool");
    }
    if vectors.len() != rows * dim {
        bail!(
            "embed: {} values cannot be {rows} rows of {dim}",
            vectors.len()
        );
    }
    let mut pooled = vec![0f64; dim];
    for row in vectors.chunks_exact(dim) {
        for (acc, v) in pooled.iter_mut().zip(row) {
            *acc += f64::from(*v);
        }
    }
    let inv_rows = 1.0 / rows as f64;
    let mut out: Vec<f32> = pooled.iter().map(|v| (v * inv_rows) as f32).collect();
    if let Some(keep) = dimensions {
        if keep == 0 || keep > dim {
            bail!("embed: dimensions {keep} is outside 1..={dim}");
        }
        out.truncate(keep);
    }
    l2_normalize(&mut out);
    Ok(out)
}

/// Scale `v` to unit length in place. A zero vector stays zero rather than becoming NaN.
fn l2_normalize(v: &mut [f32]) {
    let norm = v
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        let inv = (1.0 / norm) as f32;
        for x in v {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_averages_and_normalizes() {
        // Two 3-wide rows: the mean is [0.5, 1.0, 1.5], which normalizes to unit length.
        let v = mean_pool_normalize(&[0.0, 1.0, 2.0, 1.0, 1.0, 1.0], 2, 3, None).unwrap();
        assert_eq!(v.len(), 3);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm {norm}");
        // Direction preserved: ratios 1:2:3.
        assert!((v[1] / v[0] - 2.0).abs() < 1e-5, "{v:?}");
        assert!((v[2] / v[0] - 3.0).abs() < 1e-5, "{v:?}");
    }

    #[test]
    fn truncation_renormalizes() {
        let v = mean_pool_normalize(&[1.0, 1.0, 1.0, 1.0], 1, 4, Some(2)).unwrap();
        assert_eq!(v.len(), 2);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm {norm}");
    }

    #[test]
    fn rejects_a_shape_that_is_not_whole_rows_and_an_out_of_range_width() {
        assert!(mean_pool_normalize(&[1.0, 2.0, 3.0], 2, 2, None).is_err());
        assert!(mean_pool_normalize(&[], 0, 2, None).is_err());
        assert!(mean_pool_normalize(&[1.0, 2.0], 1, 2, Some(3)).is_err());
        assert!(mean_pool_normalize(&[1.0, 2.0], 1, 2, Some(0)).is_err());
    }

    #[test]
    fn a_zero_vector_stays_zero() {
        assert_eq!(
            mean_pool_normalize(&[0.0, 0.0], 1, 2, None).unwrap(),
            vec![0.0, 0.0]
        );
    }
}
