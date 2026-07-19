// =============================================================================
// In-loop latent-regime Gibbs block for run_chain_bjlm — v1c-b.
//
// This is the regime update COMPOSED with the rest of the outcome model. Unlike
// the standalone regime_hmm.rs (which owns the whole outcome), here the state
// level b0_state[s] is one additive term in a linear predictor that also carries
// the change-point, the latent GP and random intercepts. The caller passes
// `mu_full` = State::means_full (which already includes the current state
// offset); we form the state-excluded mean mu_no_state[i] = mu_full[i] -
// b0_state[path[i]] and use it as the FFBS emission base, exactly the
// cache-and-subtract pattern the GP block uses (sample_gp_state).
//
// One Gibbs sweep of the block: FFBS path -> b0_state levels (weighted
// conjugate / Polya-Gamma) -> misclassification E (Dirichlet) -> initial pi
// (Dirichlet) -> transition intensities {log q0, beta_q} (adaptive RW-MH). The
// smoothed marginals are accumulated for the Rao-Blackwellized occupancy output.
// =============================================================================

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution, Gamma, Normal};

use crate::ctmc::{build_generator, expm};
use crate::model::{OutcomeFamily, RegimeData, RegimeState};
use crate::polya_gamma::sample_pg;

fn normal_logpdf(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    -0.5 * z * z - sd.ln() - 0.9189385332046727
}
fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}
fn ln_gamma(z: f64) -> f64 {
    let c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let mut sum = 1.000000000190015;
    for (i, ci) in c.iter().enumerate() { sum += ci / (z + (i as f64) + 1.0); }
    let temp = z + 5.5;
    (z + 0.5) * temp.ln() - temp + (2.5066282746310005 * sum / z).ln()
}
fn logsumexp(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() { return m; }
    m + v.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}
fn sample_cat_logits(logits: &[f64], rng: &mut StdRng) -> usize {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = logits.iter().map(|&x| (x - m).exp()).collect();
    let tot: f64 = w.iter().sum();
    let mut u = rng.gen::<f64>() * tot;
    for (i, &wi) in w.iter().enumerate() { u -= wi; if u <= 0.0 { return i; } }
    w.len() - 1
}
fn dirichlet(alpha: &[f64], rng: &mut StdRng) -> Vec<f64> {
    let g: Vec<f64> = alpha.iter().map(|&a| Gamma::new(a.max(1e-6), 1.0).unwrap().sample(rng)).collect();
    let s: f64 = g.iter().sum();
    g.iter().map(|&x| x / s).collect()
}

// Family emission log-density of y at linear predictor eta (identity/logit/log).
fn emission_loglik(family: OutcomeFamily, y: f64, eta: f64, disp: f64) -> f64 {
    match family {
        OutcomeFamily::Binomial => y * eta - softplus(eta),
        OutcomeFamily::NegativeBinomial => {
            let r = disp; let mu = eta.exp();
            ln_gamma(y + r) - ln_gamma(r) - ln_gamma(y + 1.0)
                + r * (r / (r + mu)).ln() + y * (mu / (r + mu)).ln()
        }
        _ => normal_logpdf(y, eta, disp),
    }
}

fn adapt_gain(it: usize) -> f64 { 1.0 / (1.0 + it as f64).powf(0.7) }

// Interval transition matrices for one subject's time-sorted global obs indices.
fn subject_pmats(rg: &RegimeData, rs: &RegimeState, obs: &[usize]) -> Vec<DMatrix<f64>> {
    let q0: Vec<f64> = rs.log_q0.iter().map(|&x| x.exp()).collect();
    let mut mats = Vec::with_capacity(obs.len().saturating_sub(1));
    for w in obs.windows(2) {
        let (a, b) = (w[0], w[1]);
        let xrow: Vec<f64> = (0..rg.p_trans).map(|c| rg.x_trans[(a, c)]).collect();
        let q = build_generator(&q0, &rs.beta_q, &xrow, &rg.allowed, rg.n_states);
        mats.push(expm(&q, rg.obs_time[b] - rg.obs_time[a]));
    }
    mats
}

// One full Gibbs sweep of a regime block, composed with the rest of the mean.
// `mu_full` is State::means_full (includes the current state offset); `y` the
// response; `weights` the per-obs IPW weights; `disp` sigma (Gaussian) or r (NB).
// `occ` accumulates the forward-backward smoothed marginals (n x K, row-major by
// obs) when `accumulate` is true.
#[allow(clippy::too_many_arguments)]
pub fn regime_update(
    rg: &RegimeData, rs: &mut RegimeState,
    mu_full: &[f64], y: &[f64], weights: &[f64],
    family: OutcomeFamily, disp: f64,
    iter: usize, warmup: usize, accumulate: bool, occ: &mut [f64],
    rng: &mut StdRng,
) {
    let adapting = iter < warmup;
    let k = rg.n_states;
    let ncat = rg.n_cat;

    // State-excluded mean: mu_no_state[i] = mu_full[i] - b0_state[path[i]].
    let n = mu_full.len();
    let mut mu_no = vec![0.0; n];
    for i in 0..n { mu_no[i] = mu_full[i] - rs.b0_state[rs.state_path[i]]; }

    let e_log = rs.emat.map(|v| v.max(1e-300).ln());
    let pi_log: Vec<f64> = rs.pi_init.iter().map(|&v| v.max(1e-300).ln()).collect();

    // ---- 1. FFBS path per subject (+ smoothed marginals) ----
    for obs in &rg.subj_obs {
        let t = obs.len();
        if t == 0 { continue; }
        let pmats = subject_pmats(rg, rs, obs);
        let emiss = |j: usize, s: usize| -> f64 {
            let gi = obs[j];
            let mut l = weights[gi] * emission_loglik(family, y[gi], mu_no[gi] + rs.b0_state[s], disp);
            let ri = rg.obs_state[gi];
            if ri >= 0 { l += e_log[(s, ri as usize)]; }
            l
        };
        // forward
        let mut la = vec![vec![0.0; k]; t];
        for s in 0..k { la[0][s] = pi_log[s] + emiss(0, s); }
        for j in 1..t {
            let p = &pmats[j - 1];
            for l in 0..k {
                let terms: Vec<f64> = (0..k).map(|kk| la[j - 1][kk] + p[(kk, l)].max(1e-300).ln()).collect();
                la[j][l] = emiss(j, l) + logsumexp(&terms);
            }
        }
        // backward messages (for smoothed marginals)
        let mut lb = vec![vec![0.0; k]; t];
        for j in (0..t.saturating_sub(1)).rev() {
            let p = &pmats[j];
            for s in 0..k {
                let terms: Vec<f64> = (0..k).map(|l| p[(s, l)].max(1e-300).ln() + emiss(j + 1, l) + lb[j + 1][l]).collect();
                lb[j][s] = logsumexp(&terms);
            }
        }
        if accumulate {
            for j in 0..t {
                let lg: Vec<f64> = (0..k).map(|s| la[j][s] + lb[j][s]).collect();
                let z = logsumexp(&lg);
                let gi = obs[j];
                for s in 0..k { occ[gi * k + s] += (lg[s] - z).exp(); }
            }
        }
        // backward sample
        let mut path = vec![0usize; t];
        path[t - 1] = sample_cat_logits(&la[t - 1], rng);
        for j in (0..t.saturating_sub(1)).rev() {
            let p = &pmats[j];
            let logits: Vec<f64> = (0..k).map(|kk| la[j][kk] + p[(kk, path[j + 1])].max(1e-300).ln()).collect();
            path[j] = sample_cat_logits(&logits, rng);
        }
        for (jj, &s) in path.iter().enumerate() { rs.state_path[obs[jj]] = s; }
    }

    // ---- 2. b0_state levels: per-state weighted conjugate / PG draw ----
    // Each observation belongs to exactly one state, so the K-1 free levels are
    // conditionally independent (mirrors sample_random_effects_weighted grouped
    // by the path). Reference state ref_state stays pinned at 0.
    let prior_prec = 1.0 / (rg.prior_b0_sd * rg.prior_b0_sd);
    for s in 0..k {
        if s == rg.ref_state { rs.b0_state[s] = 0.0; continue; }
        let mut prec = prior_prec;
        let mut rhs = 0.0;
        for i in 0..n {
            if rs.state_path[i] != s { continue; }
            match family {
                OutcomeFamily::Binomial => {
                    let eta = mu_no[i] + rs.b0_state[s];
                    let omega = sample_pg(1.0, eta, rng).max(1e-9);
                    let kappa = y[i] - 0.5;
                    prec += weights[i] * omega;
                    rhs += weights[i] * (kappa - omega * mu_no[i]);
                }
                OutcomeFamily::NegativeBinomial => {
                    let log_r = disp.ln();
                    let eta = mu_no[i] + rs.b0_state[s]; // psi (log-mean)
                    let omega = sample_pg(y[i] + disp, eta - log_r, rng).max(1e-9);
                    let kappa = (y[i] - disp) / 2.0;
                    prec += weights[i] * omega;
                    rhs += weights[i] * (kappa + omega * log_r - omega * mu_no[i]);
                }
                _ => {
                    let w = weights[i] / (disp * disp);
                    prec += w;
                    rhs += w * (y[i] - mu_no[i]);
                }
            }
        }
        let mean = rhs / prec;
        let sd = (1.0 / prec).sqrt();
        rs.b0_state[s] = mean + sd * Normal::new(0.0, 1.0).unwrap().sample(rng);
    }

    // ---- 3. misclassification E: Dirichlet from (path, indicator) ----
    for a in 0..k {
        let mut counts = vec![0.0; ncat];
        for i in 0..n {
            if rs.state_path[i] == a && rg.obs_state[i] >= 0 { counts[rg.obs_state[i] as usize] += 1.0; }
        }
        let alpha: Vec<f64> = (0..ncat).map(|b| (if a == b { rg.e_diag } else { rg.e_offdiag }) + counts[b]).collect();
        let row = dirichlet(&alpha, rng);
        for b in 0..ncat { rs.emat[(a, b)] = row[b]; }
    }

    // ---- 4. initial distribution pi: Dirichlet from first-obs states ----
    let mut pc = vec![1.0; k];
    for obs in &rg.subj_obs { if let Some(&g0) = obs.first() { pc[rs.state_path[g0]] += 1.0; } }
    rs.pi_init = dirichlet(&pc, rng);

    // ---- 5. transition intensities: adaptive RW-MH, blocked per transition ----
    let na = rg.allowed.len();
    let pt = rg.p_trans;
    let mut cur_ll = intensity_loglik(rg, rs, &rs.log_q0, &rs.beta_q);
    for a in 0..na {
        let jump = Normal::new(0.0, rs.step[a]).unwrap();
        let mut pl = rs.log_q0.clone();
        let mut pb = rs.beta_q.clone();
        pl[a] += jump.sample(rng);
        for c in 0..pt { pb[a][c] += jump.sample(rng); }
        let prop_ll = intensity_loglik(rg, rs, &pl, &pb);
        let lp = |lq0: f64, b: &DVector<f64>| {
            let mut s = normal_logpdf(lq0, rg.prior_logq0_mean, rg.prior_logq0_sd);
            for c in 0..pt { s += normal_logpdf(b[c], 0.0, rg.prior_beta_q_sd); }
            s
        };
        let ratio = (prop_ll + lp(pl[a], &pb[a])) - (cur_ll + lp(rs.log_q0[a], &rs.beta_q[a]));
        let accept = ratio >= 0.0 || rng.gen::<f64>().ln() < ratio;
        if accept { rs.log_q0 = pl; rs.beta_q = pb; cur_ll = prop_ll; }
        if adapting {
            let acc = if accept { 1.0 } else { 0.0 };
            rs.step[a] = (rs.step[a] * (adapt_gain(iter) * (acc - 0.234)).exp()).clamp(1e-4, 10.0);
        }
    }
}

// Sum of ln P_interval[from, to] over all subject intervals, given intensities.
fn intensity_loglik(rg: &RegimeData, rs: &RegimeState, log_q0: &[f64], beta_q: &[DVector<f64>]) -> f64 {
    let q0: Vec<f64> = log_q0.iter().map(|&x| x.exp()).collect();
    let mut ll = 0.0;
    for obs in &rg.subj_obs {
        for w in obs.windows(2) {
            let (a, b) = (w[0], w[1]);
            let xrow: Vec<f64> = (0..rg.p_trans).map(|c| rg.x_trans[(a, c)]).collect();
            let q = build_generator(&q0, beta_q, &xrow, &rg.allowed, rg.n_states);
            let p = expm(&q, rg.obs_time[b] - rg.obs_time[a]);
            ll += p[(rs.state_path[a], rs.state_path[b])].max(1e-300).ln();
        }
    }
    ll
}
