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
// Dual-averaging step-size adapter for Metropolis-Hastings
// (Nesterov 2009; same scheme as Stan/NUTS uses for HMC)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct DualAvg {
    epsilon: f64,
    mu: f64,
    log_eps_bar: f64,
    h_bar: f64,
    da_count: usize,
    gamma: f64,
    t0: f64,
    kappa: f64,
    target_accept: f64,
}

impl DualAvg {
    fn new(init_epsilon: f64, target_accept: f64) -> Self {
        DualAvg {
            epsilon: init_epsilon,
            mu: (10.0 * init_epsilon).ln(),
            log_eps_bar: 0.0,
            h_bar: 0.0,
            da_count: 0,
            gamma: 0.05,
            t0: 10.0,
            kappa: 0.75,
            target_accept,
        }
    }

    fn update(&mut self, accept_prob: f64) {
        let ap = if accept_prob.is_nan() { 0.0 } else { accept_prob.clamp(0.0, 1.0) };
        self.da_count += 1;
        let m = self.da_count as f64;
        let w = 1.0 / (m + self.t0);
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.target_accept - ap);
        let log_eps = self.mu - (m.sqrt() / self.gamma) * self.h_bar;
        self.epsilon = log_eps.exp().clamp(1e-6, 5.0);
        let mk = m.powf(-self.kappa);
        self.log_eps_bar = mk * log_eps + (1.0 - mk) * self.log_eps_bar;
    }

    fn freeze(&mut self) {
        self.epsilon = self.log_eps_bar.exp().clamp(1e-6, 5.0);
    }
}

// ---------------------------------------------------------------------------
// Joint + componentwise adaptive MH for GP hyperparameters.
// Combines a 3D joint proposal (all hyperparams at once) with per-dimension
// componentwise proposals, each with independent dual-averaging step-size
// adaptation. A shared diagonal mass matrix is estimated via Welford's
// online algorithm during warmup.
// ---------------------------------------------------------------------------

struct GpHyperAdapt {
    /// Dual-averaging for the joint (3D) proposal
    joint_da: DualAvg,
    /// Dual-averaging for componentwise proposals: [alpha, rho, sigma_x]
    comp_da: [DualAvg; 3],
    /// Diagonal mass matrix (estimated variances on log scale)
    inv_mass: [f64; 3],
    /// Welford online variance estimation
    welford_n: usize,
    welford_mean: [f64; 3],
    welford_m2: [f64; 3],
}

impl GpHyperAdapt {
    fn new(init_epsilon: f64) -> Self {
        GpHyperAdapt {
            // Optimal joint acceptance for d=3: ~0.30
            joint_da: DualAvg::new(init_epsilon, 0.30),
            comp_da: [
                DualAvg::new(init_epsilon, 0.44),
                DualAvg::new(init_epsilon, 0.44),
                DualAvg::new(init_epsilon, 0.44),
            ],
            inv_mass: [1.0; 3],
            welford_n: 0,
            welford_mean: [0.0; 3],
            welford_m2: [0.0; 3],
        }
    }

    fn observe(&mut self, log_theta: &[f64; 3]) {
        self.welford_n += 1;
        let n = self.welford_n as f64;
        for k in 0..3 {
            let delta = log_theta[k] - self.welford_mean[k];
            self.welford_mean[k] += delta / n;
            let delta2 = log_theta[k] - self.welford_mean[k];
            self.welford_m2[k] += delta * delta2;
        }
    }

    fn refresh_mass_matrix(&mut self) {
        if self.welford_n < 20 { return; }
        let n = self.welford_n as f64;
        for k in 0..3 {
            let var_k = self.welford_m2[k] / (n - 1.0);
            self.inv_mass[k] = var_k.max(1e-8);
        }
    }

    fn freeze(&mut self) {
        self.joint_da.freeze();
        for da in &mut self.comp_da {
            da.freeze();
        }
    }
}

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

// ========================== Pointwise Log-Likelihood ==========================

/// Compute the pointwise log-likelihood vector for all observations.
/// Uses `means_full` which includes GP contributions.
fn compute_pointwise_log_lik(
    outcome_data: &ModelData,
    outcome_state: &State,
) -> Vec<f64> {
    let n = outcome_data.n;
    let mu = outcome_state.means_full(outcome_data);
    let mut ll = vec![0.0_f64; n];

    use crate::model::OutcomeFamily;
    match outcome_data.outcome_family {
        OutcomeFamily::Gaussian => {
            let sigma = outcome_state.sigma;
            let log_sigma = sigma.ln();
            let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
            for i in 0..n {
                let r = outcome_data.y[i] - mu[i];
                ll[i] = -0.5 * (r * r) / (sigma * sigma) - log_sigma - half_log_2pi;
            }
        }
        OutcomeFamily::NegativeBinomial => {
            let r = outcome_state.r;
            for i in 0..n {
                let y = outcome_data.y[i];
                let mu_i = mu[i].exp();
                // dnbinom(y, size=r, mu=mu_i, log=TRUE)
                // = lgamma(y+r) - lgamma(r) - lgamma(y+1) + r*ln(r/(r+mu_i)) + y*ln(mu_i/(r+mu_i))
                ll[i] = ln_gamma(y + r) - ln_gamma(r) - ln_gamma(y + 1.0)
                    + r * (r / (r + mu_i)).ln()
                    + y * (mu_i / (r + mu_i)).ln();
            }
        }
        OutcomeFamily::Binomial => {
            for i in 0..n {
                let eta = mu[i];
                let y = outcome_data.y[i];
                // dbinom(y, 1, logistic(eta), log=TRUE) = y*eta - log(1+exp(eta))
                let log1pexp = if eta > 20.0 { eta } else { (1.0 + eta.exp()).ln() };
                ll[i] = y * eta - log1pexp;
            }
        }
    }
    ll
}

// ========================== GP Sampler ==========================

fn sample_gp_state(
    outcome_data: &ModelData,
    prop_data: &PropensityData,
    outcome_state: &mut State,
    prop_state: &PropensityState,
    weights_obs: &[f64],
    rng: &mut StdRng,
) {
    // Elliptical Slice Sampling (Murray, Adams & MacKay, 2010)
    // For each GP block, for each subject:
    //   1. Draw nu ~ N(0, K) from the GP prior
    //   2. Set threshold: log_y = log_lik(x_current) + ln(U)
    //   3. Draw angle theta, propose x' = x*cos(theta) + nu*sin(theta)
    //   4. Accept if log_lik(x') > log_y, otherwise shrink bracket

    let n = outcome_data.n;

    // Compute base mu excluding ALL GPs (since data.x_b0 has 0s for GP columns)
    let mu_base = outcome_state.means(outcome_data);

    // Propensity eta_base (excludes GPs)
    let mut eta_base = DVector::zeros(prop_data.n_subjects);
    for i in 0..prop_data.n_subjects {
        let mut eta = 0.0;
        for j in 0..prop_data.p_prop {
            eta += prop_data.x_prop[(i, j)] * prop_state.alpha[j];
        }
        eta_base[i] = eta;
    }
    // Add ALL GP contributions to eta_base
    for (gp_idx, gp) in outcome_data.latent_gps.iter().enumerate() {
        if gp.p_prop_idx >= 0 {
            let col = gp.p_prop_idx as usize;
            let beta = prop_state.alpha[col];
            for s in 0..gp.subjects.len() {
                let subj = &gp.subjects[s];
                let gp_x = &outcome_state.gp_states[gp_idx].x[s];
                if !subj.trt_indices.is_empty() {
                    eta_base[s] += beta * gp_x[subj.trt_indices[0]];
                }
            }
        }
    }

    // Compute mu_full = mu_base + all GP contributions to outcome
    let mut mu_full = mu_base.clone();
    for (gp_idx, gp) in outcome_data.latent_gps.iter().enumerate() {
        if gp.p_b0_idx >= 0 {
            let col = gp.p_b0_idx as usize;
            let beta = outcome_state.beta_b0[col];
            for s in 0..gp.subjects.len() {
                let subj = &gp.subjects[s];
                let gp_x = &outcome_state.gp_states[gp_idx].x[s];
                for i in 0..subj.out_indices.len() {
                    let gidx = subj.out_global[i];
                    mu_full[gidx] += beta * gp_x[subj.out_indices[i]];
                }
            }
        }
        if gp.p_b1_idx >= 0 {
            let col = gp.p_b1_idx as usize;
            let beta = outcome_state.beta_b1[col];
            if outcome_state.gamma_b1[col] {
                let center = if outcome_data.n_breakpoints > 0 {
                    outcome_state.omega_vec(0, &outcome_data.x_om[0])
                } else {
                    DVector::zeros(n)
                };
                for s in 0..gp.subjects.len() {
                    let subj = &gp.subjects[s];
                    let gp_x = &outcome_state.gp_states[gp_idx].x[s];
                    for i in 0..subj.out_indices.len() {
                        let gidx = subj.out_global[i];
                        let t_val = if outcome_data.n_breakpoints > 0 {
                            outcome_data.tau[gidx] - center[gidx]
                        } else {
                            outcome_data.tau[gidx]
                        };
                        mu_full[gidx] += beta * t_val * gp_x[subj.out_indices[i]];
                    }
                }
            }
        }
    }

    let inv_sig2_y = 1.0 / (outcome_state.sigma * outcome_state.sigma);

    for (gp_idx, gp) in outcome_data.latent_gps.iter().enumerate() {
        let alpha = outcome_state.gp_states[gp_idx].alpha;
        let rho = outcome_state.gp_states[gp_idx].rho;
        let sigma_x = outcome_state.gp_states[gp_idx].sigma_x;
        let inv_sig2_x = 1.0 / (sigma_x * sigma_x);

        let beta_b0 = if gp.p_b0_idx >= 0 { outcome_state.beta_b0[gp.p_b0_idx as usize] } else { 0.0 };
        let beta_b1 = if gp.p_b1_idx >= 0 && outcome_state.gamma_b1[gp.p_b1_idx as usize] {
            outcome_state.beta_b1[gp.p_b1_idx as usize]
        } else { 0.0 };
        let beta_prop = if gp.p_prop_idx >= 0 { prop_state.alpha[gp.p_prop_idx as usize] } else { 0.0 };

        let center = if outcome_data.n_breakpoints > 0 {
            Some(outcome_state.omega_vec(0, &outcome_data.x_om[0]))
        } else {
            None
        };

        for s in 0..gp.subjects.len() {
            let subj = &gp.subjects[s];
            let nt = subj.times.len();
            if nt == 0 { continue; }

            // 1. Draw nu ~ N(0, K) from the GP prior
            let cov = crate::gp::compute_cov_matrix(&subj.times, alpha, rho, 1e-6);
            let chol = match cov.clone().cholesky() {
                Some(c) => c,
                None => {
                    // Fallback: add more jitter
                    let cov_jit = crate::gp::compute_cov_matrix(&subj.times, alpha, rho, 1e-3);
                    cov_jit.cholesky().unwrap_or_else(|| DMatrix::identity(nt, nt).cholesky().unwrap())
                }
            };
            let normal = Normal::new(0.0, 1.0).unwrap();
            let z_vec = DVector::from_iterator(nt, (0..nt).map(|_| normal.sample(rng)));
            let nu: Vec<f64> = (chol.l() * &z_vec).iter().cloned().collect();

            // Helper closure: compute observation log-likelihood for candidate GP values
            let current_x = &outcome_state.gp_states[gp_idx].x[s];

            let obs_log_lik = |x_cand: &[f64]| -> f64 {
                let mut ll = 0.0_f64;

                // Outcome likelihood contribution
                for i in 0..subj.out_indices.len() {
                    let tidx = subj.out_indices[i];
                    let gidx = subj.out_global[i];
                    let w = weights_obs[gidx];
                    let t_val = if let Some(c) = &center {
                        outcome_data.tau[gidx] - c[gidx]
                    } else {
                        outcome_data.tau[gidx]
                    };
                    let eff_beta = beta_b0 + beta_b1 * t_val;

                    // mu with current GP replaced by candidate
                    let mu_i = mu_full[gidx] - eff_beta * current_x[tidx] + eff_beta * x_cand[tidx];

                    use crate::model::OutcomeFamily;
                    match outcome_data.outcome_family {
                        OutcomeFamily::Gaussian => {
                            let r = outcome_data.y[gidx] - mu_i;
                            ll += -0.5 * w * inv_sig2_y * r * r;
                        }
                        OutcomeFamily::Binomial => {
                            let y = outcome_data.y[gidx];
                            let log1pexp = if mu_i > 20.0 { mu_i } else { (1.0 + mu_i.exp()).ln() };
                            ll += w * (y * mu_i - log1pexp);
                        }
                        OutcomeFamily::NegativeBinomial => {
                            let y = outcome_data.y[gidx];
                            let r_param = outcome_state.r;
                            let mu_exp = mu_i.exp();
                            ll += w * (ln_gamma(y + r_param) - ln_gamma(r_param) - ln_gamma(y + 1.0)
                                + r_param * (r_param / (r_param + mu_exp)).ln()
                                + y * (mu_exp / (r_param + mu_exp)).ln());
                        }
                    }
                }

                // GP observation likelihood: x_obs ~ N(x_true, sigma_x^2)
                for i in 0..subj.obs_indices.len() {
                    let tidx = subj.obs_indices[i];
                    let gidx = subj.obs_global[i];
                    let diff = gp.obs_val[gidx] - x_cand[tidx];
                    ll += -0.5 * inv_sig2_x * diff * diff;
                }

                // Propensity likelihood: T ~ Bernoulli(logistic(eta))
                if beta_prop != 0.0 && !subj.trt_indices.is_empty() {
                    let tidx = subj.trt_indices[0];
                    let eta = eta_base[s] - beta_prop * current_x[tidx] + beta_prop * x_cand[tidx];
                    let trt = prop_data.treatment[s];
                    let log1pexp = if eta > 20.0 { eta } else if eta < -20.0 { 0.0 } else { (1.0 + eta.exp()).ln() };
                    ll += trt * eta - log1pexp;
                }

                ll
            };

            // 2. Compute current log-likelihood and set threshold
            let current_ll = obs_log_lik(current_x);
            let log_y = current_ll + rng.gen::<f64>().ln();

            // 3. Draw initial angle and set bracket
            let mut theta_max: f64 = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
            let mut theta_min = theta_max - 2.0 * std::f64::consts::PI;
            let mut theta = theta_max;

            // 4. ESS loop: shrink bracket until proposal is accepted
            let mut accepted = false;
            for _ess_iter in 0..100 {
                // Propose on the ellipse
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let x_prop: Vec<f64> = (0..nt).map(|j| current_x[j] * cos_t + nu[j] * sin_t).collect();

                let prop_ll = obs_log_lik(&x_prop);

                if prop_ll > log_y {
                    // Accept: update mu_full, eta_base, and GP state
                    for i in 0..subj.out_indices.len() {
                        let tidx = subj.out_indices[i];
                        let gidx = subj.out_global[i];
                        let t_val = if let Some(c) = &center {
                            outcome_data.tau[gidx] - c[gidx]
                        } else {
                            outcome_data.tau[gidx]
                        };
                        let eff_beta = beta_b0 + beta_b1 * t_val;
                        mu_full[gidx] += eff_beta * (x_prop[tidx] - current_x[tidx]);
                    }
                    if beta_prop != 0.0 && !subj.trt_indices.is_empty() {
                        let tidx = subj.trt_indices[0];
                        eta_base[s] += beta_prop * (x_prop[tidx] - current_x[tidx]);
                    }
                    outcome_state.gp_states[gp_idx].x[s] = x_prop;
                    accepted = true;
                    break;
                }

                // Shrink the bracket
                if theta < 0.0 {
                    theta_min = theta;
                } else {
                    theta_max = theta;
                }
                // Draw new theta uniformly from shrunken bracket
                theta = theta_min + rng.gen::<f64>() * (theta_max - theta_min);
            }

            if !accepted {
                // ESS failed to find acceptable proposal after 100 iterations;
                // keep current values (this should be very rare)
            }
        }
    }
}


fn sample_gp_hyperparameters(
    outcome_data: &ModelData,
    outcome_state: &mut State,
    rng: &mut StdRng,
    adapts: &mut [GpHyperAdapt],
    adapting: bool,
) {
    let normal = Normal::new(0.0, 1.0).unwrap();

    for (gp_idx, gp) in outcome_data.latent_gps.iter().enumerate() {
        let adapt = &mut adapts[gp_idx];
        let mut alpha = outcome_state.gp_states[gp_idx].alpha;
        let mut rho = outcome_state.gp_states[gp_idx].rho;
        let mut sigma_x = outcome_state.gp_states[gp_idx].sigma_x;

        let compute_ll = |a: f64, r: f64, s: f64| -> f64 {
            let mut ll = 0.0;
            let inv_s2 = 1.0 / (s * s);
            for s_idx in 0..gp.subjects.len() {
                let subj = &gp.subjects[s_idx];
                let nt = subj.times.len();
                if nt == 0 { continue; }
                
                let cov = crate::gp::compute_cov_matrix(&subj.times, a, r, 1e-6);
                let chol = cov.cholesky().unwrap_or_else(|| {
                    DMatrix::identity(nt, nt).cholesky().unwrap()
                });
                
                let x = DVector::from_column_slice(&outcome_state.gp_states[gp_idx].x[s_idx]);
                let y = chol.l().solve_lower_triangular(&x).unwrap();
                let quad = y.dot(&y);
                let log_det: f64 = chol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>() * 2.0;
                ll -= 0.5 * quad + 0.5 * log_det;
                
                for i in 0..subj.obs_indices.len() {
                    let tidx = subj.obs_indices[i];
                    let gidx = subj.obs_global[i];
                    let val = gp.obs_val[gidx];
                    let err = val - x[tidx];
                    ll -= 0.5 * err * err * inv_s2 + s.ln();
                }
            }
            ll -= 0.5 * a.ln().powi(2) + a.ln();
            ll -= 0.5 * r.ln().powi(2) + r.ln();
            ll -= 0.5 * (s.ln() + 1.0).powi(2) + s.ln();
            ll
        };

        let mut current_ll = compute_ll(alpha, rho, sigma_x);

        // === Joint proposal on log scale ===
        {
            let eps = adapt.joint_da.epsilon;
            let a_prop = (alpha.ln() + eps * adapt.inv_mass[0].sqrt() * normal.sample(rng)).exp();
            let r_prop = (rho.ln() + eps * adapt.inv_mass[1].sqrt() * normal.sample(rng)).exp();
            let s_prop = (sigma_x.ln() + eps * adapt.inv_mass[2].sqrt() * normal.sample(rng)).exp();

            if a_prop.is_finite() && r_prop.is_finite() && s_prop.is_finite()
                && a_prop > 0.0 && r_prop > 0.0 && s_prop > 0.0
            {
                let ll_new = compute_ll(a_prop, r_prop, s_prop);
                let log_ratio = ll_new - current_ll;
                let accept_prob = if log_ratio.is_nan() { 0.0 } else { log_ratio.exp().min(1.0) };
                if rng.gen::<f64>() < accept_prob {
                    alpha = a_prop;
                    rho = r_prop;
                    sigma_x = s_prop;
                    current_ll = ll_new;
                }
                if adapting { adapt.joint_da.update(accept_prob); }
            } else {
                if adapting { adapt.joint_da.update(0.0); }
            }
        }

        // === Componentwise proposals ===
        // Alpha
        {
            let eps = adapt.comp_da[0].epsilon;
            let a_prop = (alpha.ln() + eps * adapt.inv_mass[0].sqrt() * normal.sample(rng)).exp();
            if a_prop.is_finite() && a_prop > 0.0 {
                let ll_new = compute_ll(a_prop, rho, sigma_x);
                let log_ratio = ll_new - current_ll;
                let accept_prob = if log_ratio.is_nan() { 0.0 } else { log_ratio.exp().min(1.0) };
                if rng.gen::<f64>() < accept_prob {
                    alpha = a_prop;
                    current_ll = ll_new;
                }
                if adapting { adapt.comp_da[0].update(accept_prob); }
            } else {
                if adapting { adapt.comp_da[0].update(0.0); }
            }
        }

        // Rho
        {
            let eps = adapt.comp_da[1].epsilon;
            let r_prop = (rho.ln() + eps * adapt.inv_mass[1].sqrt() * normal.sample(rng)).exp();
            if r_prop.is_finite() && r_prop > 0.0 {
                let ll_new = compute_ll(alpha, r_prop, sigma_x);
                let log_ratio = ll_new - current_ll;
                let accept_prob = if log_ratio.is_nan() { 0.0 } else { log_ratio.exp().min(1.0) };
                if rng.gen::<f64>() < accept_prob {
                    rho = r_prop;
                    current_ll = ll_new;
                }
                if adapting { adapt.comp_da[1].update(accept_prob); }
            } else {
                if adapting { adapt.comp_da[1].update(0.0); }
            }
        }

        // Sigma_x
        {
            let eps = adapt.comp_da[2].epsilon;
            let s_prop = (sigma_x.ln() + eps * adapt.inv_mass[2].sqrt() * normal.sample(rng)).exp();
            if s_prop.is_finite() && s_prop > 0.0 {
                let ll_new = compute_ll(alpha, rho, s_prop);
                let log_ratio = ll_new - current_ll;
                let accept_prob = if log_ratio.is_nan() { 0.0 } else { log_ratio.exp().min(1.0) };
                if rng.gen::<f64>() < accept_prob {
                    sigma_x = s_prop;
                }
                if adapting { adapt.comp_da[2].update(accept_prob); }
            } else {
                if adapting { adapt.comp_da[2].update(0.0); }
            }
        }

        // Write back
        outcome_state.gp_states[gp_idx].alpha = alpha;
        outcome_state.gp_states[gp_idx].rho = rho;
        outcome_state.gp_states[gp_idx].sigma_x = sigma_x;

        // Observe for mass matrix estimation
        if adapting {
            let log_theta = [alpha.ln(), rho.ln(), sigma_x.ln()];
            adapt.observe(&log_theta);
        }
    }
}

// ========================== Weighted Gibbs steps ==========================

fn sample_sigma_weighted(
    data: &ModelData, priors: &Priors, state: &mut State,
    weights: &[f64], rng: &mut StdRng,
) {
    let mu = state.means_full(data);
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

fn ln_gamma(mut z: f64) -> f64 {
    let c = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let mut sum = 1.000000000190015;
    for i in 0..6 {
        sum += c[i] / (z + (i as f64) + 1.0);
    }
    let temp = z + 5.5;
    (z + 0.5) * temp.ln() - temp + (2.5066282746310005 * sum / z).ln()
}

fn sample_r_weighted(data: &ModelData, priors: &Priors, state: &mut State, weights: &[f64], adapting: bool, rng: &mut StdRng) {
    let current_r = state.r;
    let log_r = current_r.ln();
    
    let normal = Normal::new(0.0, state.step_r).unwrap();
    let prop_log_r = log_r + normal.sample(rng);
    let prop_r = prop_log_r.exp();
    
    let mu = state.means_full(data); // mu is the linear predictor psi_i
    
    let mut log_lik_diff = 0.0;
    for i in 0..data.n {
        let y = data.y[i];
        let w = weights[i];
        let psi = mu[i];
        
        let ll_curr = ln_gamma(y + current_r) - ln_gamma(current_r) - current_r * (1.0 + psi.exp()).ln();
        let ll_prop = ln_gamma(y + prop_r) - ln_gamma(prop_r) - prop_r * (1.0 + psi.exp()).ln();
        
        log_lik_diff += w * (ll_prop - ll_curr);
    }
    
    let prior_curr = (priors.r_shape - 1.0) * log_r - priors.r_rate * current_r;
    let prior_prop = (priors.r_shape - 1.0) * prop_log_r - priors.r_rate * prop_r;
    
    // Jacobian for log transform is just adding prop_log_r - log_r to acceptance prob
    let log_accept = log_lik_diff + (prior_prop - prior_curr) + (prop_log_r - log_r);
    
    let mut accept_prob = log_accept.exp();
    if accept_prob > 1.0 { accept_prob = 1.0; }
    
    if rng.gen::<f64>() < accept_prob {
        state.r = prop_r;
    }
    
    if adapting {
        state.step_r = (state.step_r * (1.0 + 0.1 * (accept_prob - 0.44))).max(0.005);
    }
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
    let mut zero_var_indices = Vec::new();

    // b0
    x_full.view_mut((0, 0), (n, p_b0)).copy_from(&data.x_b0);
    // Overwrite GP columns in x_b0 part of x_full
    for (gp_idx, gp) in data.latent_gps.iter().enumerate() {
        if gp.p_b0_idx >= 0 {
            let col = gp.p_b0_idx as usize;
            for s in 0..data.latent_gps[gp_idx].subjects.len() {
                let subj = &gp.subjects[s];
                let gp_x = &state.gp_states[gp_idx].x[s];
                for i in 0..subj.out_indices.len() {
                    let global_idx = subj.out_global[i];
                    x_full[(global_idx, col)] = gp_x[subj.out_indices[i]];
                }
            }
        }
    }

    for j in 0..p_b0 {
        if priors.b0_sd[j] == 0.0 {
            prec_prior[j] = 1.0;
            zero_var_indices.push((j, priors.b0_mean[j]));
        } else {
            prec_prior[j] = 1.0 / (priors.b0_sd[j] * priors.b0_sd[j]);
        }
        mu_prior[j] = priors.b0_mean[j];
    }

    // b1
    let mut b1_design = data.x_b1.clone();
    // Overwrite GP columns in b1_design
    for (gp_idx, gp) in data.latent_gps.iter().enumerate() {
        if gp.p_b1_idx >= 0 {
            let col = gp.p_b1_idx as usize;
            for s in 0..data.latent_gps[gp_idx].subjects.len() {
                let subj = &gp.subjects[s];
                let gp_x = &state.gp_states[gp_idx].x[s];
                for i in 0..subj.out_indices.len() {
                    let global_idx = subj.out_global[i];
                    b1_design[(global_idx, col)] = gp_x[subj.out_indices[i]];
                }
            }
        }
    }
    
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
        let idx = p_b0 + j;
        if priors.b1_sd[j] == 0.0 {
            prec_prior[idx] = 1.0;
            zero_var_indices.push((idx, priors.b1_mean[j]));
        } else {
            prec_prior[idx] = 1.0 / (priors.b1_sd[j] * priors.b1_sd[j]);
        }
        mu_prior[idx] = priors.b1_mean[j];
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
            let idx = offset + j;
            if priors.delta_sd[k][j] == 0.0 {
                prec_prior[idx] = 1.0;
                zero_var_indices.push((idx, priors.delta_mean[k][j]));
            } else {
                prec_prior[idx] = 1.0 / (priors.delta_sd[k][j] * priors.delta_sd[k][j]);
            }
            mu_prior[idx] = priors.delta_mean[k][j];
        }
        offset += pk;
    }

    // Zero out columns in x_full for zero variance priors
    for &(idx, _) in &zero_var_indices {
        let mut col = x_full.column_mut(idx);
        col.fill(0.0);
    }

    // Get current beta vector for PG linear predictor
    let mut beta_current = DVector::<f64>::zeros(p_total);
    beta_current.view_mut((0, 0), (p_b0, 1)).copy_from(&state.beta_b0);
    beta_current.view_mut((p_b0, 0), (p_b1, 1)).copy_from(&state.beta_b1);
    let mut offset_tmp = p_b0 + p_b1;
    for k in 0..data.n_breakpoints {
        let pk = data.x_deltas[k].ncols();
        beta_current.view_mut((offset_tmp, 0), (pk, 1)).copy_from(&state.beta_deltas[k]);
        offset_tmp += pk;
    }

    let mut w_x = x_full.clone();
    let mut w_y = DVector::<f64>::zeros(n);
    let mut inv_sig2_eff = 1.0 / sigma2; // For Gaussian

    use crate::model::OutcomeFamily;
    match data.outcome_family {
        OutcomeFamily::Gaussian => {
            // Sufficient statistics — WEIGHTED: X'WX and X'Wy
            let mut y_tilde = data.y.clone();
            if data.n_groups_b0 > 0 {
                for i in 0..n {
                    let g = if data.group_b0.is_empty() { -1 } else { data.group_b0[i] };
                    if g >= 0 {
                        y_tilde[i] -= state.u_b0[g as usize];
                    }
                }
            }
            for i in 0..n {
                let mut row = w_x.row_mut(i);
                for j in 0..p_total {
                    row[j] *= weights[i];
                }
                w_y[i] = y_tilde[i] * weights[i];
            }
        }
        OutcomeFamily::Binomial => {
            inv_sig2_eff = 1.0;
            let c_vec = &x_full * &beta_current;
            for i in 0..n {
                let mut c_i = c_vec[i];
                let g = if data.group_b0.is_empty() { -1 } else { data.group_b0[i] };
                if g >= 0 {
                    c_i += state.u_b0[g as usize];
                }
                let omega = crate::polya_gamma::sample_pg(1.0, c_i, rng);
                let kappa = data.y[i] - 0.5;
                
                let w_eff = weights[i] * omega;
                let mut row = w_x.row_mut(i);
                for j in 0..p_total {
                    row[j] *= w_eff;
                }
                
                // y_tilde_i = (kappa / omega - u_b0)
                // w_y[i] = y_tilde_i * w_eff 
                //        = (kappa / omega - u_b0) * weights[i] * omega 
                //        = kappa * weights[i] - u_b0 * weights[i] * omega
                let u_b0_val = if g >= 0 { state.u_b0[g as usize] } else { 0.0 };
                w_y[i] = kappa * weights[i] - u_b0_val * w_eff;
            }
        }
        OutcomeFamily::NegativeBinomial => {
            inv_sig2_eff = 1.0;
            let c_vec = &x_full * &beta_current;
            let r = state.r;
            for i in 0..n {
                let mut c_i = c_vec[i];
                let g = if data.group_b0.is_empty() { -1 } else { data.group_b0[i] };
                if g >= 0 {
                    c_i += state.u_b0[g as usize];
                }
                let omega = crate::polya_gamma::sample_pg(data.y[i] + r, c_i, rng);
                let kappa = (data.y[i] - r) / 2.0;
                
                let w_eff = weights[i] * omega;
                let mut row = w_x.row_mut(i);
                for j in 0..p_total {
                    row[j] *= w_eff;
                }
                
                // y_tilde_i = (kappa / omega - u_b0)
                // w_y[i] = y_tilde_i * w_eff 
                //        = (kappa / omega - u_b0) * weights[i] * omega 
                //        = kappa * weights[i] - u_b0 * weights[i] * omega
                let u_b0_val = if g >= 0 { state.u_b0[g as usize] } else { 0.0 };
                w_y[i] = kappa * weights[i] - u_b0_val * w_eff;
            }
        }
    }

    let xt = x_full.transpose();
    let mut precision = &xt * &w_x * inv_sig2_eff; 
    for j in 0..p_total {
        precision[(j, j)] += prec_prior[j];
    }

    let cholesky = precision
        .cholesky()
        .expect("Weighted linear precision matrix not positive definite");
    let xty = &xt * &w_y * inv_sig2_eff; 
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
    let mut theta_new = mean + y_samp;
    for &(idx, val) in &zero_var_indices {
        theta_new[idx] = val;
    }

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
    let mu_fixed = state_no_re.means_full(data);
    let resid = &data.y - &mu_fixed;

    let mut sum_wr = vec![0.0f64; n_groups]; // weighted sum of residuals
    let mut sum_w = vec![0.0f64; n_groups]; // sum of weights
    
    use crate::model::OutcomeFamily;
    for i in 0..data.n {
        let g = if data.group_b0.is_empty() { -1 } else { data.group_b0[i] };
        if g >= 0 {
            match data.outcome_family {
                OutcomeFamily::Gaussian => {
                    sum_wr[g as usize] += weights[i] * resid[i];
                    sum_w[g as usize] += weights[i];
                }
                OutcomeFamily::Binomial => {
                    let c_i = mu_fixed[i] + state.u_b0[g as usize]; // Need old u_b0 for PG
                    let omega = crate::polya_gamma::sample_pg(1.0, c_i, rng);
                    let kappa = data.y[i] - 0.5;
                    sum_wr[g as usize] += weights[i] * (kappa - omega * mu_fixed[i]);
                    sum_w[g as usize] += weights[i] * omega;
                }
                OutcomeFamily::NegativeBinomial => {
                    let c_i = mu_fixed[i] + state.u_b0[g as usize];
                    let r_param = state.r;
                    let omega = crate::polya_gamma::sample_pg(data.y[i] + r_param, c_i, rng);
                    let kappa = (data.y[i] - r_param) / 2.0;
                    sum_wr[g as usize] += weights[i] * (kappa - omega * mu_fixed[i]);
                    sum_w[g as usize] += weights[i] * omega;
                }
            }
        }
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    for j in 0..n_groups {
        let prec = sum_w[j] / sigma2 + 1.0 / sigma_u2;
        let post_sd = (1.0 / prec).sqrt();
        let post_mean = (sum_wr[j] / sigma2) / prec;
        state.u_b0[j] = post_mean + post_sd * normal.sample(rng);
    }
    
    // Sweep Centering for Identifiability
    if data.x_b0.ncols() > 0 {
        let mut has_intercept = true;
        for i in 0..data.n {
            if (data.x_b0[(i, 0)] - 1.0).abs() > 1e-6 {
                has_intercept = false;
                break;
            }
        }
        if has_intercept {
            let mean_u = state.u_b0.iter().sum::<f64>() / n_groups as f64;
            for j in 0..n_groups {
                state.u_b0[j] -= mean_u;
            }
            state.beta_b0[0] += mean_u;
        }
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
                let g = if data.group_b0.is_empty() { -1 } else { data.group_b0[i] };
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

fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
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

        let mut ll = 0.0;
        let mut grad_ll_mu = DVector::<f64>::zeros(data.n);

        match data.outcome_family {
            crate::model::OutcomeFamily::Gaussian => {
                let inv_s2 = 1.0 / (sigma * sigma);
                for i in 0..data.n {
                    let ri = data.y[i] - mu[i];
                    ll += -0.5 * weights[i] * ri * ri * inv_s2;
                    grad_ll_mu[i] = weights[i] * ri * inv_s2;
                }
            },
            crate::model::OutcomeFamily::Binomial => {
                for i in 0..data.n {
                    let expit_mu = sigmoid(mu[i]);
                    ll += weights[i] * (data.y[i] * mu[i] - softplus(mu[i]));
                    grad_ll_mu[i] = weights[i] * (data.y[i] - expit_mu);
                }
            },
            crate::model::OutcomeFamily::NegativeBinomial => {
                let r_param = state.r;
                for i in 0..data.n {
                    let expit_mu = sigmoid(mu[i]);
                    ll += weights[i] * (data.y[i] * mu[i] - (data.y[i] + r_param) * softplus(mu[i]));
                    grad_ll_mu[i] = weights[i] * (data.y[i] - (data.y[i] + r_param) * expit_mu);
                }
            }
        }

        let lp = log_truncated_normal_prior(
            q.as_slice(), &priors.om_mean[k], &priors.om_sd[k],
            &priors.om_lb[k], &priors.om_ub[k],
        );

        let mut grad = DVector::<f64>::zeros(p);
        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            let ri = rho_k[i];
            let bi = cache.delta_vals[k][i];
            let mut dmu_dom = -(bi * si + di * ri * si * (1.0 - si) * bi);
            if is_om1 { dmu_dom -= cache.b1_vals[i]; }
            let factor = grad_ll_mu[i] * dmu_dom;
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

        let mut ll = 0.0;
        let mut grad_ll_mu = DVector::<f64>::zeros(data.n);

        match data.outcome_family {
            crate::model::OutcomeFamily::Gaussian => {
                let inv_s2 = 1.0 / (sigma * sigma);
                for i in 0..data.n {
                    let ri = data.y[i] - mu[i];
                    ll += -0.5 * weights[i] * ri * ri * inv_s2;
                    grad_ll_mu[i] = weights[i] * ri * inv_s2;
                }
            },
            crate::model::OutcomeFamily::Binomial => {
                for i in 0..data.n {
                    let expit_mu = sigmoid(mu[i]);
                    ll += weights[i] * (data.y[i] * mu[i] - softplus(mu[i]));
                    grad_ll_mu[i] = weights[i] * (data.y[i] - expit_mu);
                }
            },
            crate::model::OutcomeFamily::NegativeBinomial => {
                let r_param = state.r;
                for i in 0..data.n {
                    let expit_mu = sigmoid(mu[i]);
                    ll += weights[i] * (data.y[i] * mu[i] - (data.y[i] + r_param) * softplus(mu[i]));
                    grad_ll_mu[i] = weights[i] * (data.y[i] - (data.y[i] + r_param) * expit_mu);
                }
            }
        }

        let lp = log_truncated_normal_prior(
            q.as_slice(), &priors.rho_mean[k], &priors.rho_sd[k],
            &priors.rho_lb[k], &priors.rho_ub[k],
        );

        let mut grad = DVector::<f64>::zeros(p);
        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            let bi = cache.delta_vals[k][i];
            let dmu_drho = di * di * si * (1.0 - si) * bi;
            let factor = grad_ll_mu[i] * dmu_drho;
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

/// Configuration for the BJLM sampler.
pub struct BjlmConfig {
    pub weight_type: WeightType,
    pub max_weight: f64,
}

/// Run the joint Bayesian IPW chain.
///
/// Returns a tuple of:
/// - draws: DMatrix of posterior draws (n_post × n_params)
/// - log_lik: DMatrix of pointwise log-likelihoods (n_post × n_obs)
/// - n_divergent: number of divergent transitions
///
/// The parameter columns are ordered:
/// [outcome params (β₀, u, β₁, δ, ω, ρ, σ, σ_u)] | [propensity α] | [mean_weight]
pub fn run_chain_bjlm(
    outcome_data: &ModelData,
    outcome_priors: &Priors,
    prop_data: &PropensityData,
    prop_priors: &PropensityPriors,
    config: &BjlmConfig,
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
) -> (DMatrix<f64>, DMatrix<f64>, usize) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Initialize outcome state
    let mut outcome_state = init_state(outcome_data, outcome_priors, &mut rng);

    // Initialize propensity state
    let mut prop_state = PropensityState::new(prop_data.p_prop, prop_data.n_subjects, &mut rng);

    // Initialize GP hyperparameter adapters (dual-averaging + mass matrix)
    let mut gp_hyper_adapts: Vec<GpHyperAdapt> = outcome_data.latent_gps.iter()
        .map(|_| GpHyperAdapt::new(0.1))
        .collect();

    let n_post = n_iter - n_warmup;
    let n_outcome_params = outcome_state.n_params(false, false, false, outcome_data.outcome_family.clone());
    let n_prop_params = prop_data.p_prop;
    let n_total_params = n_outcome_params + n_prop_params + 1; // +1 for mean weight
    let mut draws = DMatrix::<f64>::zeros(n_post, n_total_params);
    let n_obs = outcome_data.n;
    let mut log_lik = DMatrix::<f64>::zeros(n_post, n_obs);

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
        sample_propensity(prop_data, prop_priors, &mut prop_state, outcome_data, &outcome_state, &mut rng);

        // === COMPUTE WEIGHTS ===
        let weights_subj = compute_weights(
            &prop_data.treatment, &prop_state.pi,
            config.weight_type, config.max_weight,
        );

        // Expand subject-level weights to observation-level
        let weights_obs = if outcome_data.group_prop.is_empty() || outcome_data.group_prop[0] == usize::MAX {
            // Cross-sectional: n_subjects == n_obs, use weights directly
            weights_subj.clone()
        } else {
            // Convert group_prop (Vec<usize>) to Vec<i32> for expand_weights_to_obs
            let group_prop_i32: Vec<i32> = outcome_data.group_prop.iter().map(|&x| x as i32).collect();
            expand_weights_to_obs(
                &weights_subj, &group_prop_i32, outcome_data.n,
            )
        };

        // === GP BLOCK ===
        sample_gp_state(
            outcome_data, prop_data, &mut outcome_state, &prop_state, &weights_obs, &mut rng,
        );
        sample_gp_hyperparameters(
            outcome_data, &mut outcome_state, &mut rng, &mut gp_hyper_adapts, iter < n_warmup,
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
        if matches!(outcome_data.outcome_family, crate::model::OutcomeFamily::Gaussian) {
            sample_sigma_weighted(
                outcome_data, outcome_priors, &mut outcome_state, &weights_obs, &mut rng,
            );
        }
        if outcome_data.n_groups_b0 > 0 {
            sample_sigma_u_weighted(outcome_priors, &mut outcome_state, &mut rng);
        }
        if matches!(outcome_data.outcome_family, crate::model::OutcomeFamily::NegativeBinomial) {
            sample_r_weighted(
                outcome_data, outcome_priors, &mut outcome_state, &weights_obs, iter < n_warmup, &mut rng,
            );
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
            // GP hyperparameter mass matrix refresh
            if (iter + 1) % 500 == 0 {
                for gp_adapt in gp_hyper_adapts.iter_mut() {
                    gp_adapt.refresh_mass_matrix();
                }
            }
        } else if iter == n_warmup {
            for k in 0..outcome_data.n_breakpoints {
                adapt_om[k].freeze();
                adapt_rho[k].freeze();
            }
            // Freeze GP hyperparameter adapters
            for gp_adapt in gp_hyper_adapts.iter_mut() {
                gp_adapt.freeze();
            }
        }

        // === Store draws ===
        if iter >= n_warmup {
            let row = iter - n_warmup;
            let outcome_draw = outcome_state.to_vec(false, false, false, outcome_data.outcome_family.clone());
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

            // Compute pointwise log-likelihood (uses means_full with GP contributions)
            let ll = compute_pointwise_log_lik(outcome_data, &outcome_state);
            for (col, &val) in ll.iter().enumerate() {
                log_lik[(row, col)] = val;
            }
        }
    }

    let n_div = adapt_om.iter().map(|h| h.n_divergent).sum::<usize>()
        + adapt_rho.iter().map(|h| h.n_divergent).sum::<usize>();
    (draws, log_lik, n_div)
}
