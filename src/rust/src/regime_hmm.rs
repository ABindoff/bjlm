// =============================================================================
// Standalone Bayesian continuous-time hidden-Markov multistate REGRESSION with a
// misclassified observed indicator — phases v1b (Gaussian) and v1c-a (families).
//
// Outcome: a state-dependent linear predictor eta_it = x_it' beta + b0_{s_it}
// feeds one of three families,
//   Gaussian:  y ~ N(eta, sigma^2)
//   Binomial:  y ~ Binom(n_it, sigmoid(eta))                      (logit link)
//   NegBin:    y ~ NB(mean = exp(eta), size = r)                  (log link)
// where s_it is a latent continuous-time Markov state (proportional-intensity
// generator, ctmc.rs) observed through a misclassification matrix E:
//   r_it ~ Categorical(E_{s_it, .}),  E rows ~ Dirichlet (diagonally dominant).
// The outcome and transition models couple through the latent path, drawn by
// forward-filter/backward-sample (FFBS) over each subject's observation grid
// using exp(Q*delta) interval kernels. Given the path, the coefficients + state
// levels are a (weighted) conjugate Gaussian draw -- exact for Gaussian, and via
// Polya-Gamma augmentation for Binomial/NB (bjlm convention: psi = ln(mean); NB
// augments the log-odds eta = psi - ln r with b = y + r, kappa = (y - r)/2;
// Binomial b = 1, kappa = y - 0.5). The dispersion is inverse-gamma sigma
// (Gaussian) or RW-MH r (NB); E and the initial distribution are Dirichlet and
// the transition intensities are adaptive RW-MH (as in ctmc.rs). Self-contained
// (no GP / change-point); the main-loop composition is a later phase.
// See data-raw/DESIGN_regimes_hmm.md. FFI wrapper in lib.rs.
// =============================================================================

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Gamma, Normal};

use crate::ctmc::{build_generator, expm};
use crate::polya_gamma::sample_pg;

// Outcome families.
const FAM_GAUSSIAN: i32 = 0;
const FAM_BINOMIAL: i32 = 1;
const FAM_NEGBIN: i32 = 2;

fn normal_logpdf(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    -0.5 * z * z - sd.ln() - 0.9189385332046727 // 0.5*ln(2pi)
}

// ln(1 + exp(x)), numerically stable.
fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

// Lanczos ln Gamma (matches sampler_bjlm::ln_gamma).
fn ln_gamma(z: f64) -> f64 {
    let c = [
        76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
    ];
    let mut sum = 1.000000000190015;
    for (i, ci) in c.iter().enumerate() {
        sum += ci / (z + (i as f64) + 1.0);
    }
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

// Per-subject contiguous observation ranges (obs sorted by subject then time).
fn subject_ranges(obs_subj: &[i32]) -> Vec<(usize, usize)> {
    let n = obs_subj.len();
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < n {
        let s = obs_subj[i];
        let start = i;
        while i < n && obs_subj[i] == s { i += 1; }
        ranges.push((start, i));
    }
    ranges
}

struct HmmReg {
    n: usize, k: usize, r: usize, p_fixed: usize, p_trans: usize,
    family: i32,
    y: DVector<f64>,
    n_trials: Vec<f64>,             // binomial trials per obs (1.0 for Bernoulli / unused)
    x_fixed: DMatrix<f64>,          // n x p_fixed
    x_trans: DMatrix<f64>,          // n x p_trans (covariate row for the interval starting at obs)
    obs_state: Vec<i32>,            // observed indicator per obs (0-based, -1 missing)
    obs_time: Vec<f64>,
    subj: Vec<(usize, usize)>,      // per-subject obs ranges
    allowed: Vec<(usize, usize)>,
}

// Family emission log-density of y_i given linear predictor eta and dispersion
// (sigma for Gaussian, r for NB; ignored for Binomial). State-independent
// constants may be dropped (they cancel in the FFBS categorical normalisation)
// but are kept here for a well-scaled filter.
fn emission_loglik(d: &HmmReg, i: usize, eta: f64, disp: f64) -> f64 {
    match d.family {
        FAM_BINOMIAL => {
            let t = d.n_trials[i];
            d.y[i] * eta - t * softplus(eta)
        }
        FAM_NEGBIN => {
            let r = disp;
            let mu = eta.exp();
            ln_gamma(d.y[i] + r) - ln_gamma(r) - ln_gamma(d.y[i] + 1.0)
                + r * (r / (r + mu)).ln() + d.y[i] * (mu / (r + mu)).ln()
        }
        _ => normal_logpdf(d.y[i], eta, disp), // Gaussian
    }
}

// Interval transition matrices for a subject, given current q0/beta.
fn subject_pmats(d: &HmmReg, r0: usize, r1: usize, q0: &[f64], beta: &[DVector<f64>]) -> Vec<DMatrix<f64>> {
    let mut mats = Vec::with_capacity(r1 - r0 - 1);
    for j in r0..r1 - 1 {
        let xrow: Vec<f64> = (0..d.p_trans).map(|c| d.x_trans[(j, c)]).collect();
        let q = build_generator(q0, beta, &xrow, &d.allowed, d.k);
        mats.push(expm(&q, d.obs_time[j + 1] - d.obs_time[j]));
    }
    mats
}

// FFBS one subject. Returns the sampled state per observation (for the Gibbs
// updates) AND the forward-backward SMOOTHED marginals gamma[j][s] =
// p(s_j = s | y, theta) as probabilities (for the Rao-Blackwellized occupancy
// estimate -- averaging these across draws is lower-variance than counting the
// hard sampled path). `xbeta` is the fixed-effect predictor x_i'beta; the state
// adds b0_state[s].
fn ffbs_subject(
    d: &HmmReg, r0: usize, r1: usize,
    xbeta: &DVector<f64>, b0_state: &[f64], disp: f64,
    e_log: &DMatrix<f64>, pi_log: &[f64], pmats: &[DMatrix<f64>],
    rng: &mut StdRng,
) -> (Vec<usize>, Vec<Vec<f64>>) {
    let k = d.k;
    let t = r1 - r0;
    let emiss = |j: usize, s: usize| -> f64 {
        let eta = xbeta[r0 + j] + b0_state[s];
        let mut l = emission_loglik(d, r0 + j, eta, disp);
        let ri = d.obs_state[r0 + j];
        if ri >= 0 { l += e_log[(s, ri as usize)]; }
        l
    };
    // forward (log alpha): alpha_j(s) = p(y_1:j, s_j = s)
    let mut la = vec![vec![0.0; k]; t];
    for s in 0..k { la[0][s] = pi_log[s] + emiss(0, s); }
    for j in 1..t {
        let p = &pmats[j - 1];
        for l in 0..k {
            let terms: Vec<f64> = (0..k).map(|kk| la[j - 1][kk] + p[(kk, l)].max(1e-300).ln()).collect();
            la[j][l] = emiss(j, l) + logsumexp(&terms);
        }
    }
    // backward (log beta): beta_j(s) = p(y_{j+1:T} | s_j = s)
    let mut lb = vec![vec![0.0; k]; t];
    for j in (0..t - 1).rev() {
        let p = &pmats[j];
        for s in 0..k {
            let terms: Vec<f64> = (0..k).map(|l| p[(s, l)].max(1e-300).ln() + emiss(j + 1, l) + lb[j + 1][l]).collect();
            lb[j][s] = logsumexp(&terms);
        }
    }
    // smoothed marginals gamma_j(s) proportional to alpha_j(s) beta_j(s)
    let mut gamma = vec![vec![0.0; k]; t];
    for j in 0..t {
        let lg: Vec<f64> = (0..k).map(|s| la[j][s] + lb[j][s]).collect();
        let z = logsumexp(&lg);
        for s in 0..k { gamma[j][s] = (lg[s] - z).exp(); }
    }
    // backward sample the hard path (uses alpha; unchanged)
    let mut path = vec![0usize; t];
    path[t - 1] = sample_cat_logits(&la[t - 1], rng);
    for j in (0..t - 1).rev() {
        let p = &pmats[j];
        let logits: Vec<f64> = (0..k).map(|kk| la[j][kk] + p[(kk, path[j + 1])].max(1e-300).ln()).collect();
        path[j] = sample_cat_logits(&logits, rng);
    }
    (path, gamma)
}

// (weighted) conjugate Gaussian draw of (beta, b0_state_free) given the path.
// Design D = [x_fixed | state indicators for states 1..K-1]; state 0 is the
// corner. Per-observation weight w_i and pseudo-response wy_i (already
// weight-folded) reduce every family to the same normal equations:
//   precision = D' diag(w) D + diag(1/prior_var),  rhs = D' wy,
// theta ~ N(precision^{-1} rhs, precision^{-1}).
//   Gaussian:  w = 1/sigma^2,  wy = y / sigma^2
//   Binomial:  omega ~ PG(1, eta_cur),        w = omega, wy = (y - 0.5)
//   NegBin:    omega ~ PG(y + r, eta_cur-lnr), w = omega, wy = kappa + omega*ln r
// where eta_cur is the CURRENT linear predictor (previous coefficients + path).
fn draw_coefs(
    d: &HmmReg, path: &[usize], cur_beta: &DVector<f64>, cur_b0: &[f64], disp: f64,
    prior_beta_sd: f64, prior_b0_sd: f64, rng: &mut StdRng,
) -> (DVector<f64>, Vec<f64>) {
    let dcol = d.p_fixed + (d.k - 1);
    let mut dm = DMatrix::<f64>::zeros(d.n, dcol);
    for i in 0..d.n {
        for c in 0..d.p_fixed { dm[(i, c)] = d.x_fixed[(i, c)]; }
        let s = path[i];
        if s >= 1 { dm[(i, d.p_fixed + s - 1)] = 1.0; }
    }
    let xbeta = &d.x_fixed * cur_beta;

    let mut w = vec![0.0f64; d.n];
    let mut wy = vec![0.0f64; d.n];
    match d.family {
        FAM_BINOMIAL => {
            for i in 0..d.n {
                let eta = xbeta[i] + cur_b0[path[i]];
                let omega = sample_pg(1.0, eta, rng).max(1e-9);
                w[i] = omega;
                wy[i] = d.y[i] - 0.5 * d.n_trials[i];      // kappa = y - n/2
            }
        }
        FAM_NEGBIN => {
            let log_r = disp.ln();
            for i in 0..d.n {
                let eta = xbeta[i] + cur_b0[path[i]]; // psi = log-mean
                let omega = sample_pg(d.y[i] + disp, eta - log_r, rng).max(1e-9);
                let kappa = (d.y[i] - disp) / 2.0;
                w[i] = omega;
                wy[i] = kappa + omega * log_r;             // pseudo-response for psi, weight-folded
            }
        }
        _ => {
            let inv_s2 = 1.0 / (disp * disp);
            for i in 0..d.n { w[i] = inv_s2; wy[i] = d.y[i] * inv_s2; }
        }
    }

    // precision = D' diag(w) D + prior; rhs = D' wy
    let mut wx = dm.clone();
    for i in 0..d.n { for c in 0..dcol { wx[(i, c)] *= w[i]; } }
    let mut precision = dm.transpose() * &wx;
    for c in 0..dcol {
        let pv = if c < d.p_fixed { prior_beta_sd } else { prior_b0_sd };
        precision[(c, c)] += 1.0 / (pv * pv);
    }
    let rhs = dm.transpose() * DVector::from_vec(wy);
    let chol = precision.cholesky().expect("coef posterior precision not PD");
    let mean = chol.solve(&rhs);
    let z = DVector::from_iterator(dcol, (0..dcol).map(|_| Normal::new(0.0, 1.0).unwrap().sample(rng)));
    let y_samp = chol.l().transpose().solve_upper_triangular(&z)
        .expect("upper-triangular solve failed");
    let theta = mean + y_samp;

    let beta = DVector::from_iterator(d.p_fixed, (0..d.p_fixed).map(|c| theta[c]));
    let mut b0 = vec![0.0; d.k];
    for s in 1..d.k { b0[s] = theta[d.p_fixed + s - 1]; }
    (beta, b0)
}

// RW-MH update of the NB dispersion r on the log scale (weights = 1). Prior:
// r ~ Gamma(shape, rate) (bjlm convention: rate = 1/scale). Returns (r, step).
fn draw_nb_r(
    d: &HmmReg, eta: &[f64], r: f64, step_r: f64,
    r_shape: f64, r_rate: f64, adapting: bool, rng: &mut StdRng,
) -> (f64, f64) {
    let log_r = r.ln();
    let prop_log_r = log_r + Normal::new(0.0, step_r).unwrap().sample(rng);
    let prop_r = prop_log_r.exp();
    let mut ll_diff = 0.0;
    for i in 0..d.n {
        let mu = eta[i].exp();
        let ll_c = ln_gamma(d.y[i] + r) - ln_gamma(r)
            + r * (r / (r + mu)).ln() + d.y[i] * (mu / (r + mu)).ln();
        let ll_p = ln_gamma(d.y[i] + prop_r) - ln_gamma(prop_r)
            + prop_r * (prop_r / (prop_r + mu)).ln() + d.y[i] * (mu / (prop_r + mu)).ln();
        ll_diff += ll_p - ll_c;
    }
    let prior_c = (r_shape - 1.0) * log_r - r_rate * r;
    let prior_p = (r_shape - 1.0) * prop_log_r - r_rate * prop_r;
    let log_accept = ll_diff + (prior_p - prior_c) + (prop_log_r - log_r); // +Jacobian
    let acc = log_accept.exp().min(1.0);
    let new_r = if rng.gen::<f64>() < acc { prop_r } else { r };
    let new_step = if adapting { (step_r * (1.0 + 0.1 * (acc - 0.44))).max(0.005) } else { step_r };
    (new_r, new_step)
}

fn intensity_loglik(d: &HmmReg, path: &[usize], log_q0: &[f64], beta: &[DVector<f64>]) -> f64 {
    let q0: Vec<f64> = log_q0.iter().map(|&x| x.exp()).collect();
    let mut ll = 0.0;
    for &(r0, r1) in &d.subj {
        let mats = subject_pmats(d, r0, r1, &q0, beta);
        for (idx, j) in (r0..r1 - 1).enumerate() {
            ll += mats[idx][(path[j], path[j + 1])].max(1e-300).ln();
        }
    }
    ll
}

fn adapt_gain(it: usize) -> f64 { 1.0 / (1.0 + it as f64).powf(0.7) }

#[allow(clippy::too_many_arguments)]
fn run_one_chain(
    d: &HmmReg,
    prior_beta_sd: f64, prior_b0_sd: f64, sigma_shape: f64, sigma_scale: f64,
    r_init: f64, r_shape: f64, r_rate: f64,
    e_diag: f64, e_offdiag: f64,
    prior_logq0_mean: f64, prior_logq0_sd: f64, prior_beta_q_sd: f64,
    n_iter: usize, warmup: usize, init_step: f64, seed: u64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (k, r, na, pt) = (d.k, d.r, d.allowed.len(), d.p_trans);

    // init
    let mut beta = DVector::<f64>::zeros(d.p_fixed);
    let mut b0_state = vec![0.0; k];
    let mut sigma = 1.0;                 // Gaussian dispersion
    let mut r_disp = r_init;             // NB dispersion
    let mut step_r = 0.2;
    let mut e_mat = DMatrix::<f64>::from_element(k, r, 0.1);
    for i in 0..k { if i < r { e_mat[(i, i)] = 0.8; } }
    for i in 0..k { let s: f64 = (0..r).map(|j| e_mat[(i, j)]).sum(); for j in 0..r { e_mat[(i, j)] /= s; } }
    let mut pi = vec![1.0 / k as f64; k];
    let mut log_q0 = vec![prior_logq0_mean; na];
    let mut beta_q: Vec<DVector<f64>> = (0..na).map(|_| DVector::zeros(pt)).collect();
    let mut step = vec![init_step; na];
    let mut path = vec![0usize; d.n];

    // dispersion passed to the emission (sigma for Gaussian, r for NB, 1.0 else)
    let disp_of = |sigma: f64, r_disp: f64| -> f64 {
        match d.family { FAM_NEGBIN => r_disp, FAM_GAUSSIAN => sigma, _ => 1.0 }
    };

    let n_post = n_iter - warmup;
    let ncol = d.p_fixed + (k - 1) + 1 + na + na * pt + k * r + k;
    let mut draws = DMatrix::<f64>::zeros(n_post, ncol);
    // Rao-Blackwellized occupancy: sum of smoothed marginals over post-warmup
    // iterations, averaged at the end -> p(s_i = s | data).
    let mut occ = DMatrix::<f64>::zeros(d.n, k);

    for it in 0..n_iter {
        let e_log = e_mat.map(|v| v.max(1e-300).ln());
        let pi_log: Vec<f64> = pi.iter().map(|&v| v.max(1e-300).ln()).collect();
        let q0v: Vec<f64> = log_q0.iter().map(|&x| x.exp()).collect();
        let disp = disp_of(sigma, r_disp);

        // 1. FFBS path per subject (+ smoothed marginals for occupancy)
        let xbeta = &d.x_fixed * &beta;
        for &(r0, r1) in &d.subj {
            let mats = subject_pmats(d, r0, r1, &q0v, &beta_q);
            let (sp, gamma) = ffbs_subject(d, r0, r1, &xbeta, &b0_state, disp, &e_log, &pi_log, &mats, &mut rng);
            for (jj, &s) in sp.iter().enumerate() { path[r0 + jj] = s; }
            if it >= warmup {
                for (jj, g) in gamma.iter().enumerate() {
                    for s in 0..k { occ[(r0 + jj, s)] += g[s]; }
                }
            }
        }

        // 2. (beta, b0_state): weighted conjugate / PG-augmented draw
        let (nb, nb0) = draw_coefs(d, &path, &beta, &b0_state, disp, prior_beta_sd, prior_b0_sd, &mut rng);
        beta = nb; b0_state = nb0;

        // 3. dispersion
        let xbeta2 = &d.x_fixed * &beta;
        match d.family {
            FAM_GAUSSIAN => {
                let mut ssr = 0.0;
                for i in 0..d.n { let mu = xbeta2[i] + b0_state[path[i]]; ssr += (d.y[i] - mu).powi(2); }
                let sh = sigma_shape + d.n as f64 / 2.0;
                let sc = sigma_scale + ssr / 2.0;
                sigma = (1.0 / Gamma::new(sh, 1.0 / sc).unwrap().sample(&mut rng)).sqrt();
            }
            FAM_NEGBIN => {
                let eta: Vec<f64> = (0..d.n).map(|i| xbeta2[i] + b0_state[path[i]]).collect();
                let (nr, ns) = draw_nb_r(d, &eta, r_disp, step_r, r_shape, r_rate, it < warmup, &mut rng);
                r_disp = nr; step_r = ns;
            }
            _ => {}
        }

        // 4. E Dirichlet from (path, observed indicator)
        for a in 0..k {
            let mut counts = vec![0.0; r];
            for i in 0..d.n { if path[i] == a && d.obs_state[i] >= 0 { counts[d.obs_state[i] as usize] += 1.0; } }
            let alpha: Vec<f64> = (0..r).map(|b| (if a == b { e_diag } else { e_offdiag }) + counts[b]).collect();
            let row = dirichlet(&alpha, &mut rng);
            for b in 0..r { e_mat[(a, b)] = row[b]; }
        }

        // 5. pi Dirichlet from first-obs states
        let mut pc = vec![1.0; k];
        for &(r0, _) in &d.subj { pc[path[r0]] += 1.0; }
        pi = dirichlet(&pc, &mut rng);

        // 6. intensity RW-MH, blocked per allowed transition
        let mut cur_ll = intensity_loglik(d, &path, &log_q0, &beta_q);
        for a in 0..na {
            let jump = Normal::new(0.0, step[a]).unwrap();
            let mut pl = log_q0.clone(); let mut pb = beta_q.clone();
            pl[a] += jump.sample(&mut rng);
            for c in 0..pt { pb[a][c] += jump.sample(&mut rng); }
            let prop_ll = intensity_loglik(d, &path, &pl, &pb);
            let lp = |lq0: f64, b: &DVector<f64>| {
                let mut s = normal_logpdf(lq0, prior_logq0_mean, prior_logq0_sd);
                for c in 0..pt { s += normal_logpdf(b[c], 0.0, prior_beta_q_sd); }
                s
            };
            let ratio = (prop_ll + lp(pl[a], &pb[a])) - (cur_ll + lp(log_q0[a], &beta_q[a]));
            let accept = ratio >= 0.0 || rng.gen::<f64>().ln() < ratio;
            if accept { log_q0 = pl; beta_q = pb; cur_ll = prop_ll; }
            if it < warmup {
                let acc = if accept { 1.0 } else { 0.0 };
                step[a] = (step[a] * (adapt_gain(it) * (acc - 0.234)).exp()).clamp(1e-4, 10.0);
            }
        }

        // store: [beta, b0_state(K-1), disp, q0(na), beta_q(na*pt), E(k*r), pi(k)]
        if it >= warmup {
            let row = it - warmup;
            let mut c = 0;
            for i in 0..d.p_fixed { draws[(row, c)] = beta[i]; c += 1; }
            for s in 1..k { draws[(row, c)] = b0_state[s]; c += 1; }
            draws[(row, c)] = match d.family { FAM_NEGBIN => r_disp, FAM_GAUSSIAN => sigma, _ => 1.0 }; c += 1;
            for a in 0..na { draws[(row, c)] = log_q0[a].exp(); c += 1; }
            for a in 0..na { for cc in 0..pt { draws[(row, c)] = beta_q[a][cc]; c += 1; } }
            for a in 0..k { for b in 0..r { draws[(row, c)] = e_mat[(a, b)]; c += 1; } }
            for a in 0..k { draws[(row, c)] = pi[a]; c += 1; }
        }
    }
    occ /= n_post as f64;                          // posterior-mean occupancy
    (draws, occ)
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    n_states: usize, n_cat: usize, family: i32,
    y: &[f64], n_trials: &[f64], x_fixed: &[f64], p_fixed: usize, x_trans: &[f64], p_trans: usize,
    obs_state: &[i32], obs_subj: &[i32], obs_time: &[f64],
    allowed_from: &[i32], allowed_to: &[i32],
    prior_beta_sd: f64, prior_b0_sd: f64, sigma_shape: f64, sigma_scale: f64,
    r_init: f64, r_shape: f64, r_rate: f64,
    e_diag: f64, e_offdiag: f64,
    prior_logq0_mean: f64, prior_logq0_sd: f64, prior_beta_q_sd: f64,
    n_iter: usize, warmup: usize, chains: usize, seed: u64, init_step: f64,
) -> Vec<(DMatrix<f64>, DMatrix<f64>)> {
    let n = y.len();
    let nt = if n_trials.len() == n { n_trials.to_vec() } else { vec![1.0; n] };
    let d = HmmReg {
        n, k: n_states, r: n_cat, p_fixed, p_trans, family,
        y: DVector::from_column_slice(y),
        n_trials: nt,
        x_fixed: DMatrix::from_column_slice(n, p_fixed, x_fixed),
        x_trans: DMatrix::from_column_slice(n, p_trans, x_trans),
        obs_state: obs_state.to_vec(),
        obs_time: obs_time.to_vec(),
        subj: subject_ranges(obs_subj),
        allowed: allowed_from.iter().zip(allowed_to.iter())
            .map(|(&f, &t)| (f as usize, t as usize)).collect(),
    };
    (0..chains).map(|c| run_one_chain(
        &d, prior_beta_sd, prior_b0_sd, sigma_shape, sigma_scale, r_init, r_shape, r_rate,
        e_diag, e_offdiag, prior_logq0_mean, prior_logq0_sd, prior_beta_q_sd,
        n_iter, warmup, init_step, seed.wrapping_add(c as u64 * 1_000_003),
    )).collect()
}
