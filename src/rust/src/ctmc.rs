// =============================================================================
// Continuous-time Markov jump process (CTMC) intensity model — phase v1a.
//
// Estimates the transition-intensity parameters of a hidden Markov multistate
// component with an OBSERVED (clamped) state path, via adaptive random-walk
// Metropolis-Hastings on the proportional-intensity generator. Because the state
// path is fixed data in v1a, this block is conditionally independent of the
// outcome level model, so it runs as its own sampler and its draws are merged
// with the outcome fit in R. The pure primitives here (build_generator, expm,
// CtmcData::loglik) carry over to the v1b FFBS integration in the main sampler.
// See data-raw/DESIGN_regimes_hmm.md. FFI wrappers live in lib.rs.
//
// Model: q_kl(x) = q0_kl * exp(x' beta_kl)  (l != k),  q_kk = -sum_{l!=k} q_kl.
// Interval transition matrix P(delta; x) = exp(Q(x) * delta) (matrix exponential).
// A state interval a -> b split into piecewise-constant-covariate segments
// s = 1..S with durations delta_s contributes
//   log [ prod_s exp(Q(x_s) delta_s) ]_{a,b},
// so off-grid covariates (change times not coinciding with state observations)
// are handled by segmenting the interval.
// =============================================================================

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

fn normal_logpdf(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    -0.5 * z * z - sd.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Assemble the K x K generator Q for one covariate row.
pub fn build_generator(
    q0: &[f64],
    beta: &[DVector<f64>],
    x: &[f64],
    allowed: &[(usize, usize)],
    k: usize,
) -> DMatrix<f64> {
    let xv = DVector::from_column_slice(x);
    let mut q = DMatrix::<f64>::zeros(k, k);
    for (a, &(from, to)) in allowed.iter().enumerate() {
        let rate = q0[a] * beta[a].dot(&xv).exp();
        q[(from, to)] += rate;
        q[(from, from)] -= rate;
    }
    q
}

/// Matrix exponential of Q*delta. nalgebra's exp() is scaling-and-squaring Padé,
/// robust to the defective/repeated eigenvalues a generator can hit mid-chain
/// (unlike an eigendecomposition route). K is small (2-8).
pub fn expm(q: &DMatrix<f64>, delta: f64) -> DMatrix<f64> {
    (q * delta).exp()
}

/// Preprocessed transition data for repeated likelihood evaluation. Distinct
/// (covariate row, duration) segments are deduplicated so exp(Q*dt) is computed
/// once per distinct combo per proposal (a large speedup when covariates are
/// categorical and the time grid is regular).
pub struct CtmcData {
    k: usize,
    p: usize,
    seg_combo: Vec<usize>,              // per segment: index into `combos`
    combos: Vec<(Vec<f64>, f64)>,       // distinct (covariate row, duration)
    interval_segs: Vec<(usize, usize)>, // (start, count) into segment order, per interval
    from: Vec<usize>,                   // per interval: source state
    to: Vec<usize>,                     // per interval: destination state
    allowed: Vec<(usize, usize)>,       // allowed (k,l) transitions, indexes q0/beta
}

impl CtmcData {
    /// Build from the flat FFI arrays. Segments are supplied sorted by interval
    /// id (0-based, contiguous).
    pub fn new(
        k: usize, p: usize,
        x_trans: &[f64], seg_dt: &[f64], seg_interval: &[i32],
        interval_from: &[i32], interval_to: &[i32],
        allowed_from: &[i32], allowed_to: &[i32],
    ) -> CtmcData {
        let n_seg = seg_dt.len();
        let x = DMatrix::from_column_slice(n_seg, p, x_trans);

        // deduplicate (x_row, dt) combos (keyed by rounded bits)
        let mut map: HashMap<Vec<i64>, usize> = HashMap::new();
        let mut combos: Vec<(Vec<f64>, f64)> = Vec::new();
        let mut seg_combo = vec![0usize; n_seg];
        for s in 0..n_seg {
            let xrow: Vec<f64> = (0..p).map(|c| x[(s, c)]).collect();
            let mut key: Vec<i64> = xrow.iter().map(|&v| (v * 1e9).round() as i64).collect();
            key.push((seg_dt[s] * 1e9).round() as i64);
            let id = *map.entry(key).or_insert_with(|| {
                combos.push((xrow.clone(), seg_dt[s])); combos.len() - 1
            });
            seg_combo[s] = id;
        }

        let n_intervals = interval_from.len();
        let mut interval_segs = vec![(0usize, 0usize); n_intervals];
        for s in 0..n_seg {
            let iv = seg_interval[s] as usize;
            if interval_segs[iv].1 == 0 { interval_segs[iv].0 = s; }
            interval_segs[iv].1 += 1;
        }
        CtmcData {
            k, p, seg_combo, combos, interval_segs,
            from: interval_from.iter().map(|&v| v as usize).collect(),
            to: interval_to.iter().map(|&v| v as usize).collect(),
            allowed: allowed_from.iter().zip(allowed_to.iter())
                .map(|(&f, &t)| (f as usize, t as usize)).collect(),
        }
    }

    pub fn n_allowed(&self) -> usize { self.allowed.len() }
    pub fn p(&self) -> usize { self.p }

    fn loglik(&self, log_q0: &[f64], beta: &[DVector<f64>]) -> f64 {
        let q0: Vec<f64> = log_q0.iter().map(|&l| l.exp()).collect();
        // exp(Q(x)*dt) once per distinct combo
        let mats: Vec<DMatrix<f64>> = self.combos.iter().map(|(xrow, dt)| {
            expm(&build_generator(&q0, beta, xrow, &self.allowed, self.k), *dt)
        }).collect();
        let mut ll = 0.0;
        for iv in 0..self.from.len() {
            let (start, cnt) = self.interval_segs[iv];
            let mut p = DMatrix::<f64>::identity(self.k, self.k);
            for s in start..start + cnt { p *= &mats[self.seg_combo[s]]; }
            let pab = p[(self.from[iv], self.to[iv])].max(1e-300);
            ll += pab.ln();
        }
        ll
    }
}

// Diminishing Robbins-Monro adaptation gain during warmup.
fn adapt_gain(it: usize) -> f64 {
    1.0 / (1.0 + it as f64).powf(0.7)
}

/// One MH chain over {log q0, beta} per allowed transition. Returns an
/// (n_post x n_params) draws matrix, columns = [q0 (n_allowed), then beta
/// (allowed-major, covariate-minor)].
pub fn run_one_chain(
    data: &CtmcData,
    prior_logq0_mean: f64, prior_logq0_sd: f64,
    prior_beta_mean: f64, prior_beta_sd: f64,
    n_iter: usize, warmup: usize, init_step: f64, seed: u64,
) -> DMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let na = data.n_allowed();
    let p = data.p();
    let target = 0.234;

    let mut log_q0 = vec![prior_logq0_mean; na];
    let mut beta: Vec<DVector<f64>> = (0..na)
        .map(|_| DVector::from_element(p, prior_beta_mean)).collect();
    let mut step = vec![init_step; na];
    let mut cur_ll = data.loglik(&log_q0, &beta);

    let block_logprior = |lq0: f64, b: &DVector<f64>| -> f64 {
        let mut s = normal_logpdf(lq0, prior_logq0_mean, prior_logq0_sd);
        for c in 0..p { s += normal_logpdf(b[c], prior_beta_mean, prior_beta_sd); }
        s
    };

    let n_post = n_iter - warmup;
    let mut draws = DMatrix::<f64>::zeros(n_post, na * (1 + p));

    for it in 0..n_iter {
        for a in 0..na {
            let jump = Normal::new(0.0, step[a]).unwrap();
            let mut prop_lq0 = log_q0.clone();
            let mut prop_beta = beta.clone();
            prop_lq0[a] += jump.sample(&mut rng);
            for c in 0..p { prop_beta[a][c] += jump.sample(&mut rng); }

            let prop_ll = data.loglik(&prop_lq0, &prop_beta);
            let cur_lp = block_logprior(log_q0[a], &beta[a]);
            let prop_lp = block_logprior(prop_lq0[a], &prop_beta[a]);
            let log_ratio = (prop_ll + prop_lp) - (cur_ll + cur_lp);
            let accept = log_ratio >= 0.0 || rng.gen::<f64>().ln() < log_ratio;
            if accept { log_q0 = prop_lq0; beta = prop_beta; cur_ll = prop_ll; }

            if it < warmup {
                let acc = if accept { 1.0 } else { 0.0 };
                step[a] = (step[a] * (adapt_gain(it) * (acc - target)).exp()).clamp(1e-4, 10.0);
            }
        }
        if it >= warmup {
            let row = it - warmup;
            for a in 0..na { draws[(row, a)] = log_q0[a].exp(); }
            let mut col = na;
            for a in 0..na {
                for c in 0..p { draws[(row, col)] = beta[a][c]; col += 1; }
            }
        }
    }
    draws
}
