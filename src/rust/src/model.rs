use nalgebra::{DMatrix, DVector};

// ---------------------------------------------------------------------------
// Data passed in from R
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutcomeFamily {
    Gaussian,
    Binomial,
    NegativeBinomial,
}

pub struct ModelData {
    pub outcome_family: OutcomeFamily,
    pub y: DVector<f64>,
    pub tau: DVector<f64>,
    pub x_b0: DMatrix<f64>,
    pub x_b1: DMatrix<f64>,
    /// List of design matrices for slope changes at each breakpoint
    pub x_deltas: Vec<DMatrix<f64>>,
    /// List of design matrices for breakpoint locations
    pub x_om: Vec<DMatrix<f64>>,
    /// List of design matrices for transition sharpness
    pub x_rho: Vec<DMatrix<f64>>,
    /// 0-based group indices for b0 random intercept; -1 if observation has no RE
    pub group_b0: Vec<i32>,
    pub n_groups_b0: usize,
    pub group_prop: Vec<usize>,
    pub n: usize,
    pub n_breakpoints: usize,
    /// Indicates if a coefficient in x_om[k] is a random effect (hierarchical)
    pub re_mask_om: Vec<Vec<bool>>,
    /// List of latent GP data configurations
    pub latent_gps: Vec<GpData>,
}

pub struct GpData {
    pub name: String,
    pub obs_time: Vec<f64>,
    pub obs_val: Vec<f64>,
    pub obs_group: Vec<usize>,
    
    pub trt_time: Vec<f64>,
    pub trt_group: Vec<usize>,
    
    pub out_time: Vec<f64>,
    pub out_group: Vec<usize>,
    
    pub p_b0_idx: i32,
    pub p_b1_idx: i32,
    pub p_prop_idx: i32,
    
    // Processed data per subject
    pub subjects: Vec<GpSubjectData>,
}

pub struct GpSubjectData {
    pub times: Vec<f64>,
    pub obs_indices: Vec<usize>, // maps local obs to times[]
    pub obs_global: Vec<usize>,  // global indices in obs_val[]
    pub trt_indices: Vec<usize>,
    pub trt_global: Vec<usize>,
    pub out_indices: Vec<usize>,
    pub out_global: Vec<usize>,
}

impl GpData {
    pub fn process(&mut self, n_subjects: usize) {
        self.subjects.clear();
        let mut obs_by_subj = vec![Vec::new(); n_subjects];
        let mut trt_by_subj = vec![Vec::new(); n_subjects];
        let mut out_by_subj = vec![Vec::new(); n_subjects];
        
        for (i, &g) in self.obs_group.iter().enumerate() { obs_by_subj[g].push((i, self.obs_time[i])); }
        for (i, &g) in self.trt_group.iter().enumerate() { trt_by_subj[g].push((i, self.trt_time[i])); }
        for (i, &g) in self.out_group.iter().enumerate() { out_by_subj[g].push((i, self.out_time[i])); }
        
        for s in 0..n_subjects {
            let mut all_times = Vec::new();
            for &(_, t) in &obs_by_subj[s] { all_times.push(t); }
            for &(_, t) in &trt_by_subj[s] { all_times.push(t); }
            for &(_, t) in &out_by_subj[s] { all_times.push(t); }
            
            // Sort and deduplicate
            all_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            all_times.dedup();
            
            let find_idx = |t: f64| all_times.iter().position(|&x| (x - t).abs() < 1e-8).unwrap();
            
            let mut obs_indices = vec![0; obs_by_subj[s].len()];
            let mut obs_global = vec![0; obs_by_subj[s].len()];
            for (local_idx, &(global_idx, t)) in obs_by_subj[s].iter().enumerate() {
                obs_indices[local_idx] = find_idx(t);
                obs_global[local_idx] = global_idx;
            }
            
            let mut trt_indices = vec![0; trt_by_subj[s].len()];
            let mut trt_global = vec![0; trt_by_subj[s].len()];
            for (local_idx, &(global_idx, t)) in trt_by_subj[s].iter().enumerate() {
                trt_indices[local_idx] = find_idx(t);
                trt_global[local_idx] = global_idx;
            }
            
            let mut out_indices = vec![0; out_by_subj[s].len()];
            let mut out_global = vec![0; out_by_subj[s].len()];
            for (local_idx, &(global_idx, t)) in out_by_subj[s].iter().enumerate() {
                out_indices[local_idx] = find_idx(t);
                out_global[local_idx] = global_idx;
            }
            
            self.subjects.push(GpSubjectData {
                times: all_times,
                obs_indices,
                obs_global,
                trt_indices,
                trt_global,
                out_indices,
                out_global,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Prior hyperparameters
// ---------------------------------------------------------------------------

/// Priors for all regression coefficients.
/// Organized by parameter group.
pub struct Priors {
    pub b0_mean: Vec<f64>,
    pub b0_sd: Vec<f64>,
    pub b0_lb: Vec<f64>,
    pub b0_ub: Vec<f64>,

    pub b1_mean: Vec<f64>,
    pub b1_sd: Vec<f64>,
    pub b1_lb: Vec<f64>,
    pub b1_ub: Vec<f64>,

    pub delta_mean: Vec<Vec<f64>>,
    pub delta_sd: Vec<Vec<f64>>,
    pub delta_lb: Vec<Vec<f64>>,
    pub delta_ub: Vec<Vec<f64>>,

    pub om_mean: Vec<Vec<f64>>,
    pub om_sd: Vec<Vec<f64>>,
    pub om_lb: Vec<Vec<f64>>,
    pub om_ub: Vec<Vec<f64>>,

    pub rho_mean: Vec<Vec<f64>>,
    pub rho_sd: Vec<Vec<f64>>,
    pub rho_lb: Vec<Vec<f64>>,
    pub rho_ub: Vec<Vec<f64>>,

    pub sigma_shape: f64,
    pub sigma_scale: f64,
    pub sigma_u_shape: f64,
    pub sigma_u_scale: f64,

    pub sigma_re_om_shape: f64,
    pub sigma_re_om_scale: f64,

    pub r_shape: f64,
    pub r_rate: f64,

    pub p_b0: usize,
    pub p_b1: usize,
    pub p_deltas: Vec<usize>,
    pub p_om: Vec<usize>,
    pub p_rho: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Spike-and-slab configuration
// ---------------------------------------------------------------------------

pub struct SpikeSlabConfig {
    /// Whether b1 is eligible for spike-and-slab
    pub b1_spike_mask: Vec<bool>,
    /// Spike-and-slab for each breakpoint's delta coefficients
    pub delta_spike_mask: Vec<Vec<bool>>,

    pub pi_init: f64,

    /// Beta hyperprior shape parameters for pi (if learn_pi > 0)
    pub beta_a: f64,
    pub beta_b: f64,
}

// ---------------------------------------------------------------------------
// Sampler state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct State {
    pub beta_b0: DVector<f64>,
    pub u_b0: DVector<f64>,
    pub beta_b1: DVector<f64>,
    pub beta_deltas: Vec<DVector<f64>>,
    pub beta_om: Vec<DVector<f64>>,
    pub beta_rho: Vec<DVector<f64>>,
    pub sigma: f64,
    pub sigma_u: f64,
    /// Auxiliary scale for the half-Cauchy(0, A) prior on sigma_u via the
    /// inverse-gamma parameter-expansion (Wand 2011). Kept in State because its
    /// full conditional is sampled jointly with sigma_u.
    pub a_u: f64,
    /// Metropolis step size for the ancillary (non-centred) sigma_u update in the
    /// ASIS interweave. Adapted during warmup.
    pub step_sigma_u: f64,
    /// Inclusion indicators for b1
    pub gamma_b1: Vec<bool>,
    /// Inclusion indicators for each breakpoint's deltas
    pub gamma_deltas: Vec<Vec<bool>>,
    /// Current inclusion probability
    pub pi: f64,
    /// Learned standard deviation for omega random effects at each breakpoint
    pub sigma_re_om: Vec<f64>,
    /// Negative binomial overdispersion parameter
    pub r: f64,
    pub step_r: f64,
    /// States for latent Gaussian Processes
    pub gp_states: Vec<GpState>,
}

#[derive(Clone)]
pub struct GpState {
    /// Latent GP values for each subject at all their evaluation points
    pub x: Vec<Vec<f64>>,
    pub rho: f64,
    pub alpha: f64,
    pub sigma_x: f64,
    
    // MH adaptation
    pub step_alpha: f64,
    pub step_rho: f64,
    pub step_sigma: f64,
}

impl State {
    pub fn n_params(&self, include_gammas: bool, learn_pi: bool, hierarchical: bool, outcome_family: OutcomeFamily) -> usize {
        let mut n = self.beta_b0.len() + self.u_b0.len() + self.beta_b1.len() + 2;
        for i in 0..self.beta_deltas.len() {
            n += self.beta_deltas[i].len();
            n += self.beta_om[i].len();
            n += self.beta_rho[i].len();
        }
        if include_gammas {
            // Inclusion indicators
            n += self.gamma_b1.len();
            for g_vec in &self.gamma_deltas {
                n += g_vec.len();
            }
        }
        if learn_pi { n += 1; }
        // One sigma_re_om per breakpoint
        if hierarchical {
            n += self.sigma_re_om.len();
        }
        if outcome_family == OutcomeFamily::NegativeBinomial {
            n += 1;
        }
        n += self.gp_states.len() * 3; // alpha, rho, sigma_x per GP
        n
    }

    pub fn to_vec(&self, include_gammas: bool, learn_pi: bool, hierarchical: bool, outcome_family: OutcomeFamily) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params(include_gammas, learn_pi, hierarchical, outcome_family));
        v.extend_from_slice(self.beta_b0.as_slice());
        v.extend_from_slice(self.u_b0.as_slice());
        v.extend_from_slice(self.beta_b1.as_slice());
        for b in &self.beta_deltas { v.extend_from_slice(b.as_slice()); }
        for b in &self.beta_om { v.extend_from_slice(b.as_slice()); }
        for b in &self.beta_rho { v.extend_from_slice(b.as_slice()); }
        v.push(self.sigma);
        v.push(self.sigma_u);
        
        if include_gammas {
            // Gammas
            for &g in &self.gamma_b1 { v.push(if g { 1.0 } else { 0.0 }); }
            for g_vec in &self.gamma_deltas {
                for &g in g_vec { v.push(if g { 1.0 } else { 0.0 }); }
            }
        }
        
        if learn_pi {
            v.push(self.pi);
        }
        if hierarchical {
            for &s in &self.sigma_re_om { v.push(s); }
        }
        if outcome_family == OutcomeFamily::NegativeBinomial {
            v.push(self.r);
        }
        
        for gp in &self.gp_states {
            v.push(gp.alpha);
            v.push(gp.rho);
            v.push(gp.sigma_x);
        }
        
        v
    }

    // ------------------------------------------------------------------
    // Derived quantities
    // ------------------------------------------------------------------

    pub fn omega_vec(&self, k: usize, x_om: &DMatrix<f64>) -> DVector<f64> {
        x_om * &self.beta_om[k]
    }

    pub fn rho_vec(&self, k: usize, x_rho: &DMatrix<f64>) -> DVector<f64> {
        x_rho * &self.beta_rho[k]
    }

    pub fn means(&self, data: &ModelData) -> DVector<f64> {
        let n = data.n;
        let mut mu = &data.x_b0 * &self.beta_b0;

        // Add random intercepts
        if data.n_groups_b0 > 0 {
            for i in 0..n {
                let g = data.group_b0[i];
                if g >= 0 { mu[i] += self.u_b0[g as usize]; }
            }
        }

        // Segment 1 (initial slope)
        let mut b1_eff = self.beta_b1.clone();
        for j in 0..b1_eff.len() {
            if !self.gamma_b1[j] { b1_eff[j] = 0.0; }
        }
        let b1_vals = &data.x_b1 * &b1_eff;

        if data.n_breakpoints > 0 {
            // Center at first breakpoint for segment 1
            let om1 = self.omega_vec(0, &data.x_om[0]);
            for i in 0..n {
                mu[i] += b1_vals[i] * (data.tau[i] - om1[i]);
            }
        } else {
            // Linear model fallback
            for i in 0..n {
                mu[i] += b1_vals[i] * data.tau[i];
            }
        }

        // Breakpoints
        for k in 0..data.n_breakpoints {
            let om = self.omega_vec(k, &data.x_om[k]);
            let rho = self.rho_vec(k, &data.x_rho[k]);
            
            let mut bd_eff = self.beta_deltas[k].clone();
            for j in 0..bd_eff.len() {
                if !self.gamma_deltas[k][j] { bd_eff[j] = 0.0; }
            }
            let b_delta = &data.x_deltas[k] * &bd_eff;
            
            for i in 0..n {
                let di = data.tau[i] - om[i];
                let si = sigmoid(di * rho[i]);
                mu[i] += b_delta[i] * di * si;
            }
        }
        mu
    }

    pub fn means_full(&self, data: &ModelData) -> DVector<f64> {
        let n = data.n;
        let mut mu = self.means(data);

        // Add contributions of ALL GPs
        for (gp_idx, gp) in data.latent_gps.iter().enumerate() {
            if gp.p_b0_idx >= 0 {
                let col = gp.p_b0_idx as usize;
                let beta = self.beta_b0[col];
                for s in 0..gp.subjects.len() {
                    let subj = &gp.subjects[s];
                    let gp_x = &self.gp_states[gp_idx].x[s];
                    for i in 0..subj.out_indices.len() {
                        let gidx = subj.out_global[i];
                        mu[gidx] += beta * gp_x[subj.out_indices[i]];
                    }
                }
            }
            if gp.p_b1_idx >= 0 {
                let col = gp.p_b1_idx as usize;
                let beta = self.beta_b1[col];
                if self.gamma_b1[col] {
                    // b1 is interacted with time
                    let center = if data.n_breakpoints > 0 {
                        self.omega_vec(0, &data.x_om[0])
                    } else {
                        DVector::zeros(n)
                    };
                    for s in 0..gp.subjects.len() {
                        let subj = &gp.subjects[s];
                        let gp_x = &self.gp_states[gp_idx].x[s];
                        for i in 0..subj.out_indices.len() {
                            let gidx = subj.out_global[i];
                            let t_val = if data.n_breakpoints > 0 {
                                data.tau[gidx] - center[gidx]
                            } else {
                                data.tau[gidx]
                            };
                            mu[gidx] += beta * t_val * gp_x[subj.out_indices[i]];
                        }
                    }
                }
            }
        }
        mu
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub fn log_truncated_normal_prior(
    values: &[f64],
    means: &[f64],
    sds: &[f64],
    lbs: &[f64],
    ubs: &[f64],
) -> f64 {
    let log_sqrt2pi = 0.5 * std::f64::consts::TAU.ln();
    let mut lp = 0.0;
    for i in 0..values.len() {
        let v = values[i];
        if v < lbs[i] || v > ubs[i] {
            return f64::NEG_INFINITY;
        }
        let z = (v - means[i]) / sds[i];
        lp -= 0.5 * z * z + sds[i].ln() + log_sqrt2pi;
    }
    lp
}
