//! Parity and decode bandwidth for the fp32 K-quant GEMV (`native_gemv_kq_fma.comp`).
//! Run: cargo test -p infr-vulkan --release --test kq_fma_gemv -- --ignored --nocapture
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::DType;
use infr_vulkan::linear::pad_to_u32_align;
use infr_vulkan::VulkanBackend;

struct Rng(u64);
impl Rng {
    fn byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
    fn f32(&mut self) -> f32 {
        self.byte() as f32 / 128.0 - 1.0
    }
}

/// Valid Q4_K/Q6_K blocks: random payload with small positive scale fields.
fn synth(dt: DType, n_elems: usize, seed: u64) -> Vec<u8> {
    let (epb, bpb) = infr_gguf::block_layout(dt);
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n_elems / epb * bpb);
    for _ in 0..n_elems / epb {
        let mut b: Vec<u8> = (0..bpb).map(|_| rng.byte()).collect();
        let d = half::f16::from_f32(0.004 + rng.byte() as f32 * 1e-5).to_le_bytes();
        let m = half::f16::from_f32(0.002 + rng.byte() as f32 * 1e-5).to_le_bytes();
        match dt {
            DType::Q4K => {
                b[0..2].copy_from_slice(&d);
                b[2..4].copy_from_slice(&m);
            }
            DType::Q6K => b[208..210].copy_from_slice(&d),
            _ => unreachable!(),
        }
        out.extend_from_slice(&b);
    }
    out
}

fn upload(be: &VulkanBackend, bytes: &[u8], usage: BufferUsage) -> Box<dyn Buffer> {
    let b = be.alloc(bytes.len(), usage).unwrap();
    be.upload(b.as_ref(), bytes).unwrap();
    b
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn kq_fma_gemv_parity() {
    let be = VulkanBackend::new().unwrap();
    // (in_f, out_f, w_base): an out_f not divisible by NR, an odd block count per row (Q6_K blocks
    // alternate 2-byte alignment), a non-zero weight base, and a multi-superblock row.
    for dt in [DType::Q4K, DType::Q6K] {
        for (in_f, out_f, w_base) in [
            (256, 64, 0),
            (768, 30, 0),
            (2560, 301, 0),
            (768, 40, 768 * 3),
        ] {
            for (res, nr) in [(false, 4), (true, 4), (false, 8), (true, 8)] {
                for norm in [false, true] {
                    let n = w_base + in_f * out_f;
                    let bank = synth(dt, n, 0x6e ^ (in_f * out_f) as u64 ^ dt as u64);
                    let wq = infr_gguf::dequant::dequant_block(dt, &bank).unwrap();
                    let w = upload(&be, &pad_to_u32_align(&bank), BufferUsage::Weights);
                    let mut rng = Rng(7 + in_f as u64);
                    let xv: Vec<f32> = (0..in_f).map(|_| rng.f32()).collect();
                    let rv: Vec<f32> = (0..out_f).map(|_| rng.f32()).collect();
                    let gv: Vec<f32> = (0..in_f).map(|_| 1.0 + rng.f32() * 0.1).collect();
                    let eps = 1e-5f32;
                    // Host reference input: raw xv, or rmsnorm(xv, gamma, eps).
                    let xh: Vec<f64> = if norm {
                        let ss: f64 = xv.iter().map(|&v| (v as f64) * (v as f64)).sum();
                        let s = 1.0 / ((ss / in_f as f64) + eps as f64).sqrt();
                        xv.iter()
                            .zip(&gv)
                            .map(|(&v, &g)| (v as f64) * s * (g as f64))
                            .collect()
                    } else {
                        xv.iter().map(|&v| v as f64).collect()
                    };
                    let x = upload(&be, bytemuck::cast_slice(&xv), BufferUsage::Activations);
                    let r = upload(&be, bytemuck::cast_slice(&rv), BufferUsage::Activations);
                    let g = upload(&be, bytemuck::cast_slice(&gv), BufferUsage::Activations);
                    let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
                    let rec = be.recorder().unwrap();
                    let rr = res.then_some(r.as_ref());
                    let nn = norm.then_some((x.as_ref(), g.as_ref(), eps));
                    rec.linear_kq_fma_nr(
                        dt,
                        w.as_ref(),
                        w_base,
                        x.as_ref(),
                        rr,
                        y.as_ref(),
                        in_f,
                        out_f,
                        nr,
                        nn,
                    );
                    rec.finish().unwrap();
                    let mut got = vec![0f32; out_f];
                    be.download(y.as_ref(), bytemuck::cast_slice_mut(&mut got))
                        .unwrap();
                    for o in 0..out_f {
                        let row = &wq[w_base + o * in_f..w_base + (o + 1) * in_f];
                        let mut want: f64 = row.iter().zip(&xh).map(|(a, b)| *a as f64 * *b).sum();
                        if res {
                            want += rv[o] as f64;
                        }
                        let want = want as f32;
                        assert!(
                        (got[o] - want).abs() <= 1e-4 + 1e-4 * want.abs(),
                        "{dt:?} {in_f}x{out_f} w_base {w_base} res {res} nr {nr} norm {norm}: out {o} got {} want {want}",
                        got[o]
                    );
                    }
                    println!(
                        "{dt:?} {in_f}x{out_f} w_base {w_base} res {res} nr {nr} norm {norm}: OK"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn kq_fma_gemv_bw() {
    let be = VulkanBackend::new().unwrap();
    const N_W: usize = 4;
    const REPS: usize = 32;
    for (label, dt, in_f, out_f) in [
        ("gate+up", DType::Q4K, 2560, 20480),
        ("down q4", DType::Q4K, 10240, 2560),
        ("down q6", DType::Q6K, 10240, 2560),
        ("q", DType::Q4K, 2560, 2048),
        ("o", DType::Q4K, 2048, 2560),
        ("lm_head", DType::Q4K, 2560, 262144),
    ] {
        let bank = synth(dt, in_f * out_f, 3);
        let wbytes = bank.len();
        let n_w = if wbytes > 100 << 20 { 1 } else { N_W };
        let ws: Vec<_> = (0..n_w)
            .map(|_| upload(&be, &pad_to_u32_align(&bank), BufferUsage::Weights))
            .collect();
        let xv: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.01).sin()).collect();
        let x = upload(&be, bytemuck::cast_slice(&xv), BufferUsage::Activations);
        let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        let nblk = in_f / 32;
        let qa = be.alloc(in_f, BufferUsage::Activations).unwrap();
        let dact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let time = |f: &dyn Fn(&infr_vulkan::Recorder<'_>, &dyn Buffer)| -> f64 {
            let run = || {
                let rec = be.recorder().unwrap();
                for i in 0..REPS {
                    f(&rec, ws[i % n_w].as_ref());
                }
                rec.finish().unwrap();
            };
            run();
            let mut best = 0f64;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                run();
                let s = t.elapsed().as_secs_f64() / REPS as f64;
                best = best.max(wbytes as f64 / s / 1e9);
            }
            best
        };
        let mrow = time(&|rec, w| {
            rec.quant_q8(
                x.as_ref(),
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                1,
                in_f,
            );
            rec.linear_mmv_mrow(
                dt,
                w,
                0,
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                None,
                y.as_ref(),
                1,
                in_f,
                out_f,
            );
        });
        let fuseq = time(&|rec, w| {
            rec.linear_mmv_mrow_fuseq(dt, w, 0, x.as_ref(), None, y.as_ref(), 1, in_f, out_f)
        });
        let kq = |nr: usize| {
            time(&|rec, w| {
                rec.linear_kq_fma_nr(
                    dt,
                    w,
                    0,
                    x.as_ref(),
                    None,
                    y.as_ref(),
                    in_f,
                    out_f,
                    nr,
                    None,
                )
            })
        };
        let (k4, k8) = (kq(4), kq(8));
        let pick = be.recorder().unwrap().kq_fma_gemv_rows(out_f);
        let best = if pick == 4 { k4 } else { k8 };
        println!(
            "{label:8} {dt:?} {in_f}x{out_f} [{} MiB]: quant_q8+mrow {mrow:4.0}  mrow_fuseq {fuseq:4.0}  kq_fma nr4 {k4:4.0} nr8 {k8:4.0} GB/s  picked nr{pick} ({:+.0}%)",
            wbytes >> 20,
            (best / mrow - 1.0) * 100.0
        );
    }
}

fn host_gemv(wq: &[f32], x: &[f64], in_f: usize, out_f: usize) -> Vec<f64> {
    (0..out_f)
        .map(|o| {
            wq[o * in_f..(o + 1) * in_f]
                .iter()
                .zip(x)
                .map(|(a, b)| *a as f64 * b)
                .sum()
        })
        .collect()
}

/// A dispatch reading the fused GEMV's output in the same recording must see the finished write.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn kq_fma_gemv_norm_output_ordered_before_reader() {
    let be = VulkanBackend::new().unwrap();
    let (d0, d1, d2) = (1536usize, 2048usize, 2560usize);
    for (dt, nr) in [(DType::Q4K, 8), (DType::Q6K, 4)] {
        let b1 = synth(dt, d0 * d1, 0x31 ^ dt as u64);
        let b2 = synth(dt, d1 * d2, 0x32 ^ dt as u64);
        let (q1, q2) = (
            infr_gguf::dequant::dequant_block(dt, &b1).unwrap(),
            infr_gguf::dequant::dequant_block(dt, &b2).unwrap(),
        );
        let w1 = upload(&be, &pad_to_u32_align(&b1), BufferUsage::Weights);
        let w2 = upload(&be, &pad_to_u32_align(&b2), BufferUsage::Weights);
        let mut rng = Rng(19);
        let xv: Vec<f32> = (0..d0).map(|_| rng.f32()).collect();
        let gv: Vec<f32> = (0..d0).map(|_| 1.0 + rng.f32() * 0.1).collect();
        let eps = 1e-6f32;
        let ss: f64 = xv.iter().map(|&v| v as f64 * v as f64).sum();
        let s = 1.0 / (ss / d0 as f64 + eps as f64).sqrt();
        let xn: Vec<f64> = xv
            .iter()
            .zip(&gv)
            .map(|(&v, &g)| v as f64 * s * g as f64)
            .collect();
        let want = host_gemv(&q2, &host_gemv(&q1, &xn, d0, d1), d1, d2);
        let x = upload(&be, bytemuck::cast_slice(&xv), BufferUsage::Activations);
        let g = upload(&be, bytemuck::cast_slice(&gv), BufferUsage::Activations);
        let y1 = be.alloc(d1 * 4, BufferUsage::Activations).unwrap();
        let y2 = be.alloc(d2 * 4, BufferUsage::Activations).unwrap();
        for run in 0..8 {
            let rec = be.recorder().unwrap();
            let norm = Some((x.as_ref(), g.as_ref(), eps));
            rec.linear_kq_fma_nr(
                dt,
                w1.as_ref(),
                0,
                x.as_ref(),
                None,
                y1.as_ref(),
                d0,
                d1,
                nr,
                norm,
            );
            rec.linear_kq_fma_nr(
                dt,
                w2.as_ref(),
                0,
                y1.as_ref(),
                None,
                y2.as_ref(),
                d1,
                d2,
                nr,
                None,
            );
            rec.finish().unwrap();
            let mut got = vec![0f32; d2];
            be.download(y2.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for (o, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (*g as f64 - w).abs() <= 1e-3 + 1e-3 * w.abs(),
                    "{dt:?} run {run}: out {o} got {g} want {w}"
                );
            }
        }
        println!("{dt:?} nr{nr}: OK");
    }
}
