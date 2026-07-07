// Shared initialisers used by the production joint sampler (sampler_bjlm.rs):
// the per-chain `State` constructor and the spike-and-slab inclusion-probability
// Gibbs update. The former stand-alone, outcome-only engines (`run_chain` /
// `run_chain_ss`) and their HMC/Gibbs helpers were removed together with the
// unused `run_mcmc` / `run_mcmc_ss` FFI entry points; `sampler_bjlm` only ever
// imported these two functions from this module.

use nalgebra::DVector;
use rand::rngs::StdRng;
use rand_distr::{Normal, Gamma, Distribution};

use crate::model::{ModelData, Priors, State, SpikeSlabConfig, GpState};

pub fn sample_pi(ss: &SpikeSlabConfig, state: &mut State, rng: &mut StdRng) {
    let mut n1 = 0.0;
    let mut n0 = 0.0;
    for &g in &state.gamma_b1 { if g { n1 += 1.0; } else { n0 += 1.0; } }
    for g_vec in &state.gamma_deltas {
        for &g in g_vec { if g { n1 += 1.0; } else { n0 += 1.0; } }
    }
    let a = ss.beta_a + n1;
    let b = ss.beta_b + n0;
    let gamma_a = Gamma::new(a, 1.0).unwrap();
    let gamma_b = Gamma::new(b, 1.0).unwrap();
    let x = gamma_a.sample(rng);
    let y = gamma_b.sample(rng);
    state.pi = x / (x + y);
}

pub fn init_state(data: &ModelData, priors: &Priors, rng: &mut StdRng) -> State {
    let jitter = Normal::new(0.0, 0.01).unwrap();
    let beta_b0 = DVector::from_iterator(data.x_b0.ncols(), (0..data.x_b0.ncols()).map(|i| {
        let m = priors.b0_mean[i];
        if priors.b0_sd[i] > 0.0 { m + jitter.sample(rng) } else { m }
    }));
    let u_b0 = DVector::zeros(data.n_groups_b0);
    let beta_b1 = DVector::from_iterator(data.x_b1.ncols(), (0..data.x_b1.ncols()).map(|i| {
        let m = priors.b1_mean[i];
        if priors.b1_sd[i] > 0.0 { m + jitter.sample(rng) } else { m }
    }));
    let mut beta_deltas = Vec::new();
    let mut beta_om = Vec::new();
    let mut beta_rho = Vec::new();
    let mut gamma_deltas = Vec::new();

    for k in 0..data.n_breakpoints {
        beta_deltas.push(DVector::from_iterator(data.x_deltas[k].ncols(), (0..data.x_deltas[k].ncols()).map(|i| {
            let m = priors.delta_mean[k][i];
            if priors.delta_sd[k][i] > 0.0 { m + jitter.sample(rng) } else { m }
        })));
        beta_om.push(DVector::from_iterator(data.x_om[k].ncols(), (0..data.x_om[k].ncols()).map(|i| {
            let m = priors.om_mean[k][i];
            if priors.om_sd[k][i] > 0.0 { m + jitter.sample(rng) } else { m }
        })));
        beta_rho.push(DVector::from_iterator(data.x_rho[k].ncols(), (0..data.x_rho[k].ncols()).map(|i| {
            let m = priors.rho_mean[k][i];
            if priors.rho_sd[k][i] > 0.0 { m + jitter.sample(rng) } else { m }
        })));
        gamma_deltas.push(vec![true; data.x_deltas[k].ncols()]);
    }

    let mut gp_states = Vec::new();
    for gp in &data.latent_gps {
        let mut subj_x = Vec::new();
        for subj in &gp.subjects {
            subj_x.push(vec![0.0; subj.times.len()]);
        }
        gp_states.push(GpState {
            x: subj_x,
            rho: 1.0,
            alpha: 1.0,
            sigma_x: 0.5,
            step_alpha: 0.1,
            step_rho: 0.1,
            step_sigma: 0.1,
        });
    }

    State {
        beta_b0, u_b0, beta_b1, beta_deltas, beta_om, beta_rho,
        sigma: 1.0, sigma_u: 1.0, a_u: 1.0, step_sigma_u: 0.1,
        gamma_b1: vec![true; data.x_b1.ncols()],
        gamma_deltas, pi: 0.5,
        sigma_re_om: vec![1.0; data.n_breakpoints],
        a_re_om: vec![1.0; data.n_breakpoints],
        step_sigma_re_om: vec![0.1; data.n_breakpoints],
        gp_states,
        r: 1.0,
        step_r: 0.1,
    }
}
