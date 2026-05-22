use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use rand_distr::{Normal, Gamma, Distribution};

use crate::model::{ModelData, Priors, State, log_truncated_normal_prior, sigmoid};
use crate::propensity::{PropensityState, PropensityPriors, PropensityData, sample_propensity};
use crate::weights::{WeightType, compute_weights, expand_weights_to_obs};
use crate::sampler::init_state;

// ---------------------------------------------------------------------------
// Joint Bayesian IPW sampler
//
// Architecture:
//   1. Propensity block (PG-augmented, cut-feedback)
//   2. Weight computation from propensity scores
//   3. Weighted outcome block (adapted from smoothbp's sampler)
//
// The propensity block updates α using ONLY p(T | X, α).
// The outcome block uses weights w_i(α) to weight the Gaussian likelihood.
// This is the "cut posterior" approach (Plummer 2015).
// ---------------------------------------------------------------------------

// ========================== Weighted Gibbs steps ==========================

fn sample_sigma_weighted(
    data: &ModelData, priors: &Priors, state: &mut State,
    weights: &[f64], rng: &mut StdRng,
) {
    let mu = state.means(data);
    let mut wss = 0.0;
    let mut wn = 0.0;
    for i in 0..data.n {
        let r = data.y[i] - mu[i];
        wss += weights[i] * r * r;
        wn += weights[i];
    }
    let shape = priors.sigma_shape + wn * 0.5;
    let scale = priors.sigma_scale + wss * 0.5;
    let gamma_dist = Gamma::new(shape, 1.0 / scale).unwrap();
    state.sigma = 1.0 / gamma_dist.sample(rng).sqrt();
}

fn sample_sigma_u_weighted(priors: &Priors, state: &mut State, rng: &mut StdRng) {
    // sigma_u doesn't depend on observation weights (it's a group-level prior)
    let ss = state.u_b0.dot(&state.u_b0);
    let n = state.u_b0.len() as f64;
    let shape = priors.sigma_u_shape + n * 0.5;
    let scale = priors.sigma_u_scale + ss * 0.5;
    let gamma_dist = Gamma::new(shape, 1.0 / scale).unwrap();
    state.sigma_u = 1.0 / gamma_dist.sample(rng).sqrt();
}

fn sample_linear_coefs_weighted(
    data: &ModelData, priors: &Priors, state: &mut State,
    weights: &[f64], rng: &mut StdRng,
) {
    let n = data.n;
    let sigma2 = state.sigma * state.sigma;

    let p_b0 = data.x_b0.ncols();
    let p_b1 = data.x_b1.ncols();
    let mut p_total = p_b0 + p_b1;
    for k in 0..data.n_breakpoints {
        p_total += data.x_deltas[k].ncols();
    }

    let mut x_full = DMatrix::<f64>::zeros(n, p_total);
    let mut prec_prior = DVector::<f64>::zeros(p_total);
    let mut mu_prior = DVector::<f64>::zeros(p_total);

    // b0
    x_full.view_mut((0, 0), (n, p_b0)).copy_from(&data.x_b0);
    for j in 0..p_b0 {
        prec_prior[j] = 1.0 / (priors.b0_sd[j] * priors.b0_sd[j]);
        mu_prior[j] = priors.b0_mean[j];
    }

    // b1
    let mut b1_design = data.x_b1.clone();
    if data.n_breakpoints > 0 {
        let om1 = state.omega_vec(0, &data.x_om[0]);
        for i in 0..n {
            let mut row = b1_design.row_mut(i);
            for j in 0..p_b1 {
                row[j] *= data.tau[i] - om1[i];
            }
        }
    } else {
        for i in 0..n {
            let mut row = b1_design.row_mut(i);
            for j in 0..p_b1 {
                row[j] *= data.tau[i];
            }
        }
    }
    // Apply gamma_b1 (Kuo-Mallick)
    for j in 0..p_b1 {
        if !state.gamma_b1[j] {
            let mut col = b1_design.column_mut(j);
            col.fill(0.0);
        }
    }
    x_full.view_mut((0, p_b0), (n, p_b1)).copy_from(&b1_design);
    for j in 0..p_b1 {
        prec_prior[p_b0 + j] = 1.0 / (priors.b1_sd[j] * priors.b1_sd[j]);
        mu_prior[p_b0 + j] = priors.b1_mean[j];
    }

    // deltas
    let mut offset = p_b0 + p_b1;
    for k in 0..data.n_breakpoints {
        let pk = data.x_deltas[k].ncols();
        let mut d_design = data.x_deltas[k].clone();
        let om = state.omega_vec(k, &data.x_om[k]);
        let rho = state.rho_vec(k, &data.x_rho[k]);
        for i in 0..n {
            let di = data.tau[i] - om[i];
            let si = sigmoid(di * rho[i]);
            let mut row = d_design.row_mut(i);
            for j in 0..pk {
                row[j] *= di * si;
            }
        }
        for j in 0..pk {
            if !state.gamma_deltas[k][j] {
                let mut col = d_design.column_mut(j);
                col.fill(0.0);
            }
        }
        x_full.view_mut((0, offset), (n, pk)).copy_from(&d_design);
        for j in 0..pk {
            prec_prior[offset + j] = 1.0 / (priors.delta_sd[k][j] * priors.delta_sd[k][j]);
            mu_prior[offset + j] = priors.delta_mean[k][j];
        }
        offset += pk;
    }

    // Sufficient statistics — WEIGHTED: X'WX and X'Wy
    let mut y_tilde = data.y.clone();
    if data.n_groups_b0 > 0 {
        for i in 0..n {
            let g = data.group_b0[i];
            if g >= 0 {
                y_tilde[i] -= state.u_b0[g as usize];
            }
        }
    }

    // Apply weights to design matrix: W^{1/2} X
    let mut w_x = x_full.clone();
    for i in 0..n {
        let mut row = w_x.row_mut(i);
        for j in 0..p_total {
            row[j] *= weights[i];
        }
    }

    // Apply weights to response: W y_tilde
    let mut w_y = y_tilde.clone();
    for i in 0..n {
        w_y[i] *= weights[i];
    }

    let xt = x_full.transpose();
    let mut precision = &xt * &w_x / sigma2; // X' W X / σ²
    for j in 0..p_total {
        precision[(j, j)] += prec_prior[j];
    }

    let cholesky = precision
        .cholesky()
        .expect("Weighted linear precision matrix not positive definite");
    let xty = &xt * &w_y / sigma2; // X' W y / σ²
    let rhs = xty + prec_prior.component_mul(&mu_prior);
    let mean = cholesky.solve(&rhs);

    let mut z = DVector::<f64>::zeros(p_total);
    let normal = Normal::new(0.0, 1.0).unwrap();
    for j in 0..p_total {
        z[j] = normal.sample(rng);
    }

    let y_samp = cholesky
        .l()
        .transpose()
        .solve_upper_triangular(&z)
        .expect("Failed to solve upper triangular system");
    let theta_new = mean + y_samp;

    // Check bounds for rejection
    let mut ok = true;
    let mut idx = 0;
    for j in 0..p_b0 {
        if theta_new[idx] < priors.b0_lb[j] || theta_new[idx] > priors.b0_ub[j] {
            ok = false;
            break;
        }
        idx += 1;
    }
    if ok {
        for j in 0..p_b1 {
            if theta_new[idx] < priors.b1_lb[j] || theta_new[idx] > priors.b1_ub[j] {
                ok = false;
                break;
            }
            idx += 1;
        }
    }
    if ok {
        for k in 0..data.n_breakpoints {
            for j in 0..data.x_deltas[k].ncols() {
                if theta_new[idx] < priors.delta_lb[k][j]
                    || theta_new[idx] > priors.delta_ub[k][j]
                {
                    ok = false;
                    break;
                }
                idx += 1;
            }
            if !ok {
                break;
            }
        }
    }

    if ok {
        state.beta_b0.copy_from(&theta_new.rows(0, p_b0));
        state.beta_b1.copy_from(&theta_new.rows(p_b0, p_b1));
        let mut offset = p_b0 + p_b1;
        for k in 0..data.n_breakpoints {
            let pk = data.x_deltas[k].ncols();
            state.beta_deltas[k].copy_from(&theta_new.rows(offset, pk));
            offset += pk;
        }
    }
}

fn sample_random_effects_weighted(
    data: &ModelData, _priors: &Priors, state: &mut State,
    weights: &[f64], rng: &mut StdRng,
) {
    let sigma2 = state.sigma * state.sigma;
    let sigma_u2 = state.sigma_u * state.sigma_u;
    let n_groups = data.n_groups_b0;

    let mut state_no_re = state.clone();
    state_no_re.u_b0.fill(0.0);
    let mu_fixed = state_no_re.means(data);
    let resid = &data.y - &mu_fixed;

    let mut sum_wr = vec![0.0f64; n_groups]; // weighted sum of residuals
    let mut sum_w = vec![0.0f64; n_groups]; // sum of weights
    for i in 0..data.n {
        let g = data.group_b0[i];
        if g >= 0 {
            sum_wr[g as usize] += weights[i] * resid[i];
            sum_w[g as usize] += weights[i];
        }
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    for j in 0..n_groups {
        let prec = sum_w[j] / sigma2 + 1.0 / sigma_u2;
        let post_sd = (1.0 / prec).sqrt();
        let post_mean = (sum_wr[j] / sigma2) / prec;
        state.u_b0[j] = post_mean + post_sd * normal.sample(rng);
    }
}

// ========================== Weighted HMC ==========================

const EPSILON_FLOOR: f64 = 1e-6;
const DIVERGENCE_THRESHOLD: f64 = 1000.0;
const MAX_REFLECTIONS: usize = 20;

struct HmcAdapt {
    p: usize,
    l_min: usize,
    l_max: usize,
    epsilon: f64,
    target_accept: f64,
    mu: f64,
    log_eps_bar: f64,
    h_bar: f64,
    gamma: f64,
    t0: f64,
    kappa: f64,
    da_count: usize,
    inv_mass: Vec<f64>,
    welford_n: usize,
    welford_mean: DVector<f64>,
    welford_m2: DVector<f64>,
    adapting: bool,
    n_divergent: usize,
}

impl HmcAdapt {
    fn new(p: usize, init_epsilon: f64, target_accept: f64, l_min: usize, l_max: usize) -> Self {
        HmcAdapt {
            p, l_min, l_max, epsilon: init_epsilon, target_accept,
            mu: (10.0 * init_epsilon).ln(), log_eps_bar: 0.0, h_bar: 0.0,
            gamma: 0.05, t0: 10.0, kappa: 0.75, da_count: 0,
            inv_mass: vec![1.0; p], welford_n: 0,
            welford_mean: DVector::<f64>::zeros(p), welford_m2: DVector::<f64>::zeros(p),
            adapting: true, n_divergent: 0,
        }
    }

    fn update_epsilon(&mut self, accept_prob: f64) {
        if !self.adapting { return; }
        let ap = if accept_prob.is_nan() { 0.0 } else { accept_prob.clamp(0.0, 1.0) };
        self.da_count += 1;
        let m = self.da_count as f64;
        let w = 1.0 / (m + self.t0);
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.target_accept - ap);
        let log_eps = self.mu - (m.sqrt() / self.gamma) * self.h_bar;
        self.epsilon = log_eps.exp().max(EPSILON_FLOOR);
        let mk = m.powf(-self.kappa);
        self.log_eps_bar = mk * log_eps + (1.0 - mk) * self.log_eps_bar;
    }

    fn record_energy_error(&mut self, delta_h: f64) {
        if !self.adapting && (delta_h.abs() > DIVERGENCE_THRESHOLD || delta_h.is_nan()) {
            self.n_divergent += 1;
        }
    }

    fn observe(&mut self, q: &DVector<f64>) {
        if !self.adapting { return; }
        self.welford_n += 1;
        let n = self.welford_n as f64;
        for k in 0..self.p {
            let delta = q[k] - self.welford_mean[k];
            self.welford_mean[k] += delta / n;
            let delta2 = q[k] - self.welford_mean[k];
            self.welford_m2[k] += delta * delta2;
        }
    }

    fn refresh_mass_matrix(&mut self) {
        if self.welford_n < 20 { return; }
        let n = self.welford_n as f64;
        for k in 0..self.p {
            let var_k = self.welford_m2[k] / (n - 1.0);
            self.inv_mass[k] = var_k.max(1e-8);
        }
    }

    fn freeze(&mut self) {
        self.epsilon = self.log_eps_bar.exp().max(EPSILON_FLOOR);
        self.adapting = false;
    }

    fn sample_l(&self, rng: &mut StdRng) -> usize {
        if self.l_min == self.l_max { self.l_min } else { rng.gen_range(self.l_min..=self.l_max) }
    }
}

// LinearCache (same as sampler.rs but used here for weighted HMC)
struct LinearCache {
    b0_fixed: DVector<f64>,
    re_contrib: DVector<f64>,
    b1_vals: DVector<f64>,
    delta_vals: Vec<DVector<f64>>,
}

impl LinearCache {
    fn build(state: &State, data: &ModelData) -> Self {
        let b0_fixed = &data.x_b0 * &state.beta_b0;
        let mut re_contrib = DVector::<f64>::zeros(data.n);
        if data.n_groups_b0 > 0 {
            for i in 0..data.n {
                let g = data.group_b0[i];
                if g >= 0 { re_contrib[i] = state.u_b0[g as usize]; }
            }
        }
        let mut b1_eff = state.beta_b1.clone();
        for j in 0..b1_eff.len() {
            if !state.gamma_b1[j] { b1_eff[j] = 0.0; }
        }
        let b1_vals = &data.x_b1 * &b1_eff;
        let mut delta_vals = Vec::with_capacity(data.n_breakpoints);
        for k in 0..data.n_breakpoints {
            let mut bd_eff = state.beta_deltas[k].clone();
            for j in 0..bd_eff.len() {
                if !state.gamma_deltas[k][j] { bd_eff[j] = 0.0; }
            }
            delta_vals.push(&data.x_deltas[k] * &bd_eff);
        }
        LinearCache { b0_fixed, re_contrib, b1_vals, delta_vals }
    }

    fn mu_without_segment(&self, data: &ModelData, state: &State, k: usize) -> DVector<f64> {
        let n = data.n;
        let mut mu = self.b0_fixed.clone() + &self.re_contrib;
        for i in 0..data.n_breakpoints {
            if i == k { continue; }
            let om = state.omega_vec(i, &data.x_om[i]);
            let rho = state.rho_vec(i, &data.x_rho[i]);
            for j in 0..n {
                let di = data.tau[j] - om[j];
                let si = sigmoid(di * rho[j]);
                mu[j] += self.delta_vals[i][j] * di * si;
            }
        }
        if data.n_breakpoints > 0 {
            if k != 0 {
                let om1 = state.omega_vec(0, &data.x_om[0]);
                for j in 0..n {
                    mu[j] += self.b1_vals[j] * (data.tau[j] - om1[j]);
                }
            }
        } else {
            for j in 0..n {
                mu[j] += self.b1_vals[j] * data.tau[j];
            }
        }
        mu
    }
}

fn hmc_step_om_weighted(
    data: &ModelData, priors: &Priors, state: &mut State, k: usize,
    cache: &LinearCache, weights: &[f64], adapt: &mut HmcAdapt, rng: &mut StdRng,
) {
    let p = adapt.p;
    let mut all_fixed = true;
    for j in 0..p {
        if priors.om_sd[k][j] > 0.0 { all_fixed = false; break; }
    }
    if all_fixed { return; }

    let sigma = state.sigma;
    let mu_base = cache.mu_without_segment(data, state, k);
    let is_om1 = k == 0 && data.n_breakpoints > 0;

    let energy_fn = |q: &DVector<f64>| -> (f64, DVector<f64>) {
        let om_k = &data.x_om[k] * q;
        let rho_k = state.rho_vec(k, &data.x_rho[k]);
        let delta_k = &cache.delta_vals[k];

        let mut mu = mu_base.clone();
        if is_om1 {
            for i in 0..data.n {
                mu[i] += cache.b1_vals[i] * (data.tau[i] - om_k[i]);
            }
        }
        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            mu[i] += delta_k[i] * di * si;
        }

        let r = &data.y - &mu;
        // WEIGHTED log-likelihood
        let mut ll = 0.0;
        for i in 0..data.n {
            ll += -0.5 * weights[i] * r[i] * r[i] / (sigma * sigma);
        }
        let lp = log_truncated_normal_prior(
            q.as_slice(), &priors.om_mean[k], &priors.om_sd[k],
            &priors.om_lb[k], &priors.om_ub[k],
        );

        let inv_s2 = 1.0 / (sigma * sigma);
        let mut grad = DVector::<f64>::zeros(p);
        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            let ri = rho_k[i];
            let bi = cache.delta_vals[k][i];
            let mut dmu_dom = -(bi * si + di * ri * si * (1.0 - si) * bi);
            if is_om1 { dmu_dom -= cache.b1_vals[i]; }
            // WEIGHTED gradient
            let factor = weights[i] * r[i] * inv_s2 * dmu_dom;
            for j in 0..p {
                grad[j] -= factor * data.x_om[k][(i, j)];
            }
        }
        for j in 0..p {
            grad[j] += (q[j] - priors.om_mean[k][j]) / (priors.om_sd[k][j] * priors.om_sd[k][j]);
        }
        (-ll - lp, grad)
    };

    let (q_new, accept) = hmc_sample(
        &state.beta_om[k], energy_fn, adapt, rng, &priors.om_lb[k], &priors.om_ub[k],
    );
    state.beta_om[k] = q_new;
    adapt.update_epsilon(accept);
}

fn hmc_step_rho_weighted(
    data: &ModelData, priors: &Priors, state: &mut State, k: usize,
    cache: &LinearCache, weights: &[f64], adapt: &mut HmcAdapt, rng: &mut StdRng,
) {
    let p = adapt.p;
    let mut all_fixed = true;
    for j in 0..p {
        if priors.rho_sd[k][j] > 0.0 { all_fixed = false; break; }
    }
    if all_fixed { return; }

    let sigma = state.sigma;
    let mu_base = cache.mu_without_segment(data, state, k);

    let energy_fn = |q: &DVector<f64>| -> (f64, DVector<f64>) {
        let rho_k = &data.x_rho[k] * q;
        let om_k = state.omega_vec(k, &data.x_om[k]);
        let delta_k = &cache.delta_vals[k];

        let mut mu = mu_base.clone();
        if k == 0 && data.n_breakpoints > 0 {
            let om1 = state.omega_vec(0, &data.x_om[0]);
            for i in 0..data.n { mu[i] += cache.b1_vals[i] * (data.tau[i] - om1[i]); }
        }
        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            mu[i] += delta_k[i] * di * si;
        }

        let r = &data.y - &mu;
        let mut ll = 0.0;
        for i in 0..data.n {
            ll += -0.5 * weights[i] * r[i] * r[i] / (sigma * sigma);
        }
        let lp = log_truncated_normal_prior(
            q.as_slice(), &priors.rho_mean[k], &priors.rho_sd[k],
            &priors.rho_lb[k], &priors.rho_ub[k],
        );

        let inv_s2 = 1.0 / (sigma * sigma);
        let mut grad = DVector::<f64>::zeros(p);
        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            let bi = cache.delta_vals[k][i];
            let dmu_drho = di * di * si * (1.0 - si) * bi;
            let factor = weights[i] * r[i] * inv_s2 * dmu_drho;
            for j in 0..p {
                grad[j] -= factor * data.x_rho[k][(i, j)];
            }
        }
        for j in 0..p {
            grad[j] += (q[j] - priors.rho_mean[k][j])
                / (priors.rho_sd[k][j] * priors.rho_sd[k][j]);
        }
        (-ll - lp, grad)
    };

    let (q_new, accept) = hmc_sample(
        &state.beta_rho[k], energy_fn, adapt, rng, &priors.rho_lb[k], &priors.rho_ub[k],
    );
    state.beta_rho[k] = q_new;
    adapt.update_epsilon(accept);
}

fn hmc_sample<F>(
    q0: &DVector<f64>, mut energy_fn: F, adapt: &mut HmcAdapt, rng: &mut StdRng,
    lb: &[f64], ub: &[f64],
) -> (DVector<f64>, f64)
where
    F: FnMut(&DVector<f64>) -> (f64, DVector<f64>),
{
    let p = adapt.p;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p0 = DVector::<f64>::from_iterator(
        p, (0..p).map(|i| normal.sample(rng) / adapt.inv_mass[i].sqrt()),
    );
    let kinetic0: f64 = (0..p).map(|i| p0[i] * p0[i] * adapt.inv_mass[i]).sum::<f64>() * 0.5;
    let (u0, mut grad) = energy_fn(q0);
    let h0 = u0 + kinetic0;

    let mut q = q0.clone();
    let mut mom = p0.clone();
    let l = adapt.sample_l(rng);
    let eps = adapt.epsilon;

    for _ in 0..l {
        for i in 0..p { mom[i] -= 0.5 * eps * grad[i]; }
        for i in 0..p {
            q[i] += eps * mom[i] * adapt.inv_mass[i];
            for _ in 0..MAX_REFLECTIONS {
                if q[i] < lb[i] { q[i] = 2.0 * lb[i] - q[i]; mom[i] = -mom[i]; }
                else if q[i] > ub[i] { q[i] = 2.0 * ub[i] - q[i]; mom[i] = -mom[i]; }
                else { break; }
            }
        }
        let (_, g_new) = energy_fn(&q);
        grad = g_new;
        for i in 0..p { mom[i] -= 0.5 * eps * grad[i]; }
    }

    let (u1, _) = energy_fn(&q);
    let kinetic1: f64 = (0..p).map(|i| mom[i] * mom[i] * adapt.inv_mass[i]).sum::<f64>() * 0.5;
    let h1 = u1 + kinetic1;

    let dh = h1 - h0;
    adapt.record_energy_error(dh);
    let accept_prob = (-dh).exp().min(1.0);
    if rng.gen_bool(accept_prob) { (q, accept_prob) } else { (q0.clone(), accept_prob) }
}

// ========================== Main Joint Chain ==========================

/// Configuration for the BIPW sampler.
pub struct BipwConfig {
    pub weight_type: WeightType,
    pub max_weight: f64,
}

/// Run the joint Bayesian IPW chain.
///
/// Returns a tuple of:
/// - draws: DMatrix of posterior draws (n_post × n_params)
/// - n_divergent: number of divergent transitions
///
/// The parameter columns are ordered:
/// [outcome params (β₀, u, β₁, δ, ω, ρ, σ, σ_u)] | [propensity α] | [mean_weight]
pub fn run_chain_bipw(
    outcome_data: &ModelData,
    outcome_priors: &Priors,
    prop_data: &PropensityData,
    prop_priors: &PropensityPriors,
    config: &BipwConfig,
    n_iter: usize,
    n_warmup: usize,
    step_om_init: f64,
    step_rho_init: f64,
    target_accept: f64,
    seed: u64,
    verbose: bool,
    chain_id: usize,
    n_chains: usize,
    progress_fn: &dyn Fn(usize, usize, usize, usize, bool),
) -> (DMatrix<f64>, usize) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Initialize outcome state
    let mut outcome_state = init_state(outcome_data, outcome_priors, &mut rng);

    // Initialize propensity state
    let mut prop_state = PropensityState::new(prop_data.p_prop, prop_data.n_subjects, &mut rng);

    let n_post = n_iter - n_warmup;
    let n_outcome_params = outcome_state.n_params(false, false, false);
    let n_prop_params = prop_data.p_prop;
    let n_total_params = n_outcome_params + n_prop_params + 1; // +1 for mean weight
    let mut draws = DMatrix::<f64>::zeros(n_post, n_total_params);

    let mut adapt_om: Vec<HmcAdapt> = (0..outcome_data.n_breakpoints)
        .map(|k| {
            HmcAdapt::new(
                outcome_data.x_om[k].ncols(), step_om_init, target_accept, 5, 15,
            )
        })
        .collect();
    let mut adapt_rho: Vec<HmcAdapt> = (0..outcome_data.n_breakpoints)
        .map(|k| {
            HmcAdapt::new(
                outcome_data.x_rho[k].ncols(), step_rho_init, target_accept, 5, 15,
            )
        })
        .collect();

    let report_every = (n_iter / 10).max(1);

    for iter in 0..n_iter {
        if verbose && iter % report_every == 0 {
            progress_fn(chain_id, n_chains, iter, n_iter, iter < n_warmup);
        }

        // === PROPENSITY BLOCK (cut-feedback) ===
        sample_propensity(prop_data, prop_priors, &mut prop_state, &mut rng);

        // === COMPUTE WEIGHTS ===
        let weights_subj = compute_weights(
            &prop_data.treatment, &prop_state.pi,
            config.weight_type, config.max_weight,
        );

        // Expand subject-level weights to observation-level
        let weights_obs = expand_weights_to_obs(
            &weights_subj, &outcome_data.group_b0, outcome_data.n,
        );

        // === OUTCOME BLOCK (weighted) ===
        sample_linear_coefs_weighted(
            outcome_data, outcome_priors, &mut outcome_state, &weights_obs, &mut rng,
        );

        if outcome_data.n_groups_b0 > 0 {
            sample_random_effects_weighted(
                outcome_data, outcome_priors, &mut outcome_state, &weights_obs, &mut rng,
            );
        }

        let cache = LinearCache::build(&outcome_state, outcome_data);
        for k in 0..outcome_data.n_breakpoints {
            hmc_step_om_weighted(
                outcome_data, outcome_priors, &mut outcome_state, k,
                &cache, &weights_obs, &mut adapt_om[k], &mut rng,
            );
            hmc_step_rho_weighted(
                outcome_data, outcome_priors, &mut outcome_state, k,
                &cache, &weights_obs, &mut adapt_rho[k], &mut rng,
            );
        }

        // Second pass of linear coefs (as in smoothbp)
        sample_linear_coefs_weighted(
            outcome_data, outcome_priors, &mut outcome_state, &weights_obs, &mut rng,
        );
        sample_sigma_weighted(
            outcome_data, outcome_priors, &mut outcome_state, &weights_obs, &mut rng,
        );
        if outcome_data.n_groups_b0 > 0 {
            sample_sigma_u_weighted(outcome_priors, &mut outcome_state, &mut rng);
        }

        // === HMC adaptation ===
        if iter < n_warmup {
            for k in 0..outcome_data.n_breakpoints {
                adapt_om[k].observe(&outcome_state.beta_om[k]);
                adapt_rho[k].observe(&outcome_state.beta_rho[k]);
                if (iter + 1) % 500 == 0 {
                    adapt_om[k].refresh_mass_matrix();
                    adapt_rho[k].refresh_mass_matrix();
                }
            }
        } else if iter == n_warmup {
            for k in 0..outcome_data.n_breakpoints {
                adapt_om[k].freeze();
                adapt_rho[k].freeze();
            }
        }

        // === Store draws ===
        if iter >= n_warmup {
            let row = iter - n_warmup;
            let outcome_draw = outcome_state.to_vec(false, false, false);
            for (col, &val) in outcome_draw.iter().enumerate() {
                draws[(row, col)] = val;
            }
            // Propensity parameters
            let offset = n_outcome_params;
            for j in 0..n_prop_params {
                draws[(row, offset + j)] = prop_state.alpha[j];
            }
            // Mean weight (diagnostic)
            let mean_w: f64 = weights_subj.iter().sum::<f64>() / weights_subj.len() as f64;
            draws[(row, offset + n_prop_params)] = mean_w;
        }
    }

    let n_div = adapt_om.iter().map(|h| h.n_divergent).sum::<usize>()
        + adapt_rho.iter().map(|h| h.n_divergent).sum::<usize>();
    (draws, n_div)
}
