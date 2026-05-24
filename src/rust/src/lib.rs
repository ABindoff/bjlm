use extendr_api::prelude::*;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

pub mod gp;

mod model;
mod sampler;
mod sampler_re;
mod polya_gamma;
mod propensity;
mod weights;
mod sampler_bjlm;

use model::{ModelData, Priors, SpikeSlabConfig};
use sampler::{run_chain, run_chain_ss};
use sampler_re::{run_chain_re, run_chain_re_ss};
use propensity::{PropensityData, PropensityPriors};
use weights::WeightType;
use sampler_bjlm::{BjlmConfig, run_chain_bjlm};

// Helper: build DMatrix from a flat column-major slice + dimensions
fn flat_to_dmatrix(data: &[f64], nrow: usize, ncol: usize) -> DMatrix<f64> {
    DMatrix::from_column_slice(nrow, ncol, data)
}

fn list_to_vec_dmatrix(list: List, nrow: usize, p_vec: &[i32]) -> Vec<DMatrix<f64>> {
    list.iter()
        .zip(p_vec.iter())
        .map(|(robj, &p)| {
            let data: Vec<f64> = robj.1.as_real_vector().unwrap();
            flat_to_dmatrix(&data, nrow, p as usize)
        })
        .collect()
}

/// @noRd
/// @keywords internal
#[extendr]
fn run_mcmc(
    y: &[f64],
    tau: &[f64],
    x_b0: &[f64], p_b0: i32,
    x_b1: &[f64], p_b1: i32,
    x_deltas: List, p_deltas: &[i32],
    x_om: List, p_om: &[i32],
    x_rho: List, p_rho: &[i32],
    group_b0: &[i32],
    n_groups_b0: i32,
    prior_mean_b0: &[f64], prior_sd_b0: &[f64], prior_lb_b0: &[f64], prior_ub_b0: &[f64],
    prior_mean_b1: &[f64], prior_sd_b1: &[f64], prior_lb_b1: &[f64], prior_ub_b1: &[f64],
    prior_mean_deltas: List, prior_sd_deltas: List, prior_lb_deltas: List, prior_ub_deltas: List,
    prior_mean_om: List, prior_sd_om: List, prior_lb_om: List, prior_ub_om: List,
    prior_mean_rho: List, prior_sd_rho: List, prior_lb_rho: List, prior_ub_rho: List,
    sigma_shape: f64,
    sigma_scale: f64,
    sigma_u_shape: f64,
    sigma_u_scale: f64,
    step_om: f64,
    step_rho: f64,
    target_accept: f64,
    chains: i32,
    iter: i32,
    warmup: i32,
    seed: i32,
    verbose: bool,
    n_cores: i32,
) -> List {
    let mut p_deltas = p_deltas;
    let mut p_om = p_om;
    let mut p_rho = p_rho;
    let mut group_b0 = group_b0;

    if p_deltas.len() == 1 && p_deltas[0] == -1 { p_deltas = &[]; }
    if p_om.len() == 1 && p_om[0] == -1 { p_om = &[]; }
    if p_rho.len() == 1 && p_rho[0] == -1 { p_rho = &[]; }
    if group_b0.len() == 1 && group_b0[0] == -1 { group_b0 = &[]; }

    let n = y.len();
    let n_bp = p_deltas.len();

    let data = ModelData { outcome_family: crate::model::OutcomeFamily::Gaussian,
        y: DVector::from_column_slice(y),
        tau: DVector::from_column_slice(tau),
        x_b0: flat_to_dmatrix(x_b0, n, p_b0 as usize),
        x_b1: flat_to_dmatrix(x_b1, n, p_b1 as usize),
        x_deltas: list_to_vec_dmatrix(x_deltas, n, p_deltas),
        x_om: list_to_vec_dmatrix(x_om, n, p_om),
        x_rho: list_to_vec_dmatrix(x_rho, n, p_rho),
        group_b0: group_b0.to_vec(),
        n_groups_b0: n_groups_b0 as usize,
        n_breakpoints: n_bp,
        n,
        re_mask_om: Vec::new(),
        latent_gps: Vec::new(),
    };

    let priors = Priors { r_shape: 0.0, r_rate: 0.0,
        b0_mean: prior_mean_b0.to_vec(),
        b0_sd: prior_sd_b0.to_vec(),
        b0_lb: prior_lb_b0.to_vec(),
        b0_ub: prior_ub_b0.to_vec(),
        b1_mean: prior_mean_b1.to_vec(),
        b1_sd: prior_sd_b1.to_vec(),
        b1_lb: prior_lb_b1.to_vec(),
        b1_ub: prior_ub_b1.to_vec(),
        delta_mean: prior_mean_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_sd: prior_sd_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_lb: prior_lb_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_ub: prior_ub_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_mean: prior_mean_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_sd: prior_sd_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_lb: prior_lb_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_ub: prior_ub_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_mean: prior_mean_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_sd: prior_sd_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_lb: prior_lb_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_ub: prior_ub_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        sigma_shape,
        sigma_scale,
        sigma_u_shape,
        sigma_u_scale,
        sigma_re_om_shape: 1.0,
        sigma_re_om_scale: 1.0,
        p_b0: p_b0 as usize,
        p_b1: p_b1 as usize,
        p_deltas: p_deltas.iter().map(|&p| p as usize).collect(),
        p_om: p_om.iter().map(|&p| p as usize).collect(),
        p_rho: p_rho.iter().map(|&p| p as usize).collect(),
    };

    let n_chains = chains as usize;
    let n_iter = iter as usize;
    let n_warmup = warmup as usize;
    let base_seed = seed as u64;
    let n_cores = (n_cores as usize).max(1);

    let results: Vec<(DMatrix<f64>, usize)> = if n_cores > 1 && n_chains > 1 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n_cores).build().unwrap();
        pool.install(|| {
            (0..n_chains).into_par_iter().map(|c| {
                let seed = base_seed.wrapping_add(c as u64 * 1_000_003);
                run_chain(&data, &priors, n_iter, n_warmup, step_om, step_rho, target_accept, seed, false, c, n_chains, &|_,_,_,_,_| {})
            }).collect()
        })
    } else {
        (0..n_chains).map(|c| {
            let seed = base_seed.wrapping_add(c as u64 * 1_000_003);
            run_chain(&data, &priors, n_iter, n_warmup, step_om, step_rho, target_accept, seed, verbose, c, n_chains, &|_,_,_,_,_| {})
        }).collect()
    };

    let chain_results: Vec<Robj> = results.into_iter().map(|(draws, _)| {
        let nr = draws.nrows();
        let nc = draws.ncols();
        let flat: Vec<f64> = draws.iter().cloned().collect();
        RMatrix::new_matrix(nr, nc, |r, c| flat[c * nr + r]).into()
    }).collect();

    list!(draws = chain_results)
}

/// @noRd
/// @keywords internal
#[extendr]
fn run_mcmc_ss(
    y: &[f64],
    tau: &[f64],
    x_b0: &[f64], p_b0: i32,
    x_b1: &[f64], p_b1: i32,
    x_deltas: List, p_deltas: &[i32],
    x_om: List, p_om: &[i32],
    x_rho: List, p_rho: &[i32],
    group_b0: &[i32],
    n_groups_b0: i32,
    prior_mean_b0: &[f64], prior_sd_b0: &[f64], prior_lb_b0: &[f64], prior_ub_b0: &[f64],
    prior_mean_b1: &[f64], prior_sd_b1: &[f64], prior_lb_b1: &[f64], prior_ub_b1: &[f64],
    prior_mean_deltas: List, prior_sd_deltas: List, prior_lb_deltas: List, prior_ub_deltas: List,
    prior_mean_om: List, prior_sd_om: List, prior_lb_om: List, prior_ub_om: List,
    prior_mean_rho: List, prior_sd_rho: List, prior_lb_rho: List, prior_ub_rho: List,
    sigma_shape: f64,
    sigma_scale: f64,
    sigma_u_shape: f64,
    sigma_u_scale: f64,
    step_om: f64,
    step_rho: f64,
    target_accept: f64,
    b1_spike_mask: &[i32],
    delta_spike_mask: List,
    pi_init: f64,
    pi_beta_a: f64,
    pi_beta_b: f64,
    chains: i32,
    iter: i32,
    warmup: i32,
    seed: i32,
    verbose: bool,
    n_cores: i32,
) -> List {
    let mut p_deltas = p_deltas;
    let mut p_om = p_om;
    let mut p_rho = p_rho;
    let mut group_b0 = group_b0;
    let mut b1_spike_mask = b1_spike_mask;

    if p_deltas.len() == 1 && p_deltas[0] == -1 { p_deltas = &[]; }
    if p_om.len() == 1 && p_om[0] == -1 { p_om = &[]; }
    if p_rho.len() == 1 && p_rho[0] == -1 { p_rho = &[]; }
    if group_b0.len() == 1 && group_b0[0] == -1 { group_b0 = &[]; }
    if b1_spike_mask.len() == 1 && b1_spike_mask[0] == -1 { b1_spike_mask = &[]; }

    let n = y.len();
    let n_bp = p_deltas.len();

    let data = ModelData { outcome_family: crate::model::OutcomeFamily::Gaussian,
        y: DVector::from_column_slice(y),
        tau: DVector::from_column_slice(tau),
        x_b0: flat_to_dmatrix(x_b0, n, p_b0 as usize),
        x_b1: flat_to_dmatrix(x_b1, n, p_b1 as usize),
        x_deltas: list_to_vec_dmatrix(x_deltas, n, p_deltas),
        x_om: list_to_vec_dmatrix(x_om, n, p_om),
        x_rho: list_to_vec_dmatrix(x_rho, n, p_rho),
        group_b0: group_b0.to_vec(),
        n_groups_b0: n_groups_b0 as usize,
        n_breakpoints: n_bp,
        n,
        re_mask_om: Vec::new(),
        latent_gps: Vec::new(),
    };

    let priors = Priors { r_shape: 0.0, r_rate: 0.0,
        b0_mean: prior_mean_b0.to_vec(),
        b0_sd: prior_sd_b0.to_vec(),
        b0_lb: prior_lb_b0.to_vec(),
        b0_ub: prior_ub_b0.to_vec(),
        b1_mean: prior_mean_b1.to_vec(),
        b1_sd: prior_sd_b1.to_vec(),
        b1_lb: prior_lb_b1.to_vec(),
        b1_ub: prior_ub_b1.to_vec(),
        delta_mean: prior_mean_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_sd: prior_sd_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_lb: prior_lb_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_ub: prior_ub_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_mean: prior_mean_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_sd: prior_sd_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_lb: prior_lb_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_ub: prior_ub_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_mean: prior_mean_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_sd: prior_sd_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_lb: prior_lb_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_ub: prior_ub_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        sigma_shape,
        sigma_scale,
        sigma_u_shape,
        sigma_u_scale,
        sigma_re_om_shape: 1.0,
        sigma_re_om_scale: 1.0,
        p_b0: p_b0 as usize,
        p_b1: p_b1 as usize,
        p_deltas: p_deltas.iter().map(|&p| p as usize).collect(),
        p_om: p_om.iter().map(|&p| p as usize).collect(),
        p_rho: p_rho.iter().map(|&p| p as usize).collect(),
    };

    let ss_config = SpikeSlabConfig {
        b1_spike_mask: b1_spike_mask.iter().map(|&v| v != 0).collect(),
        delta_spike_mask: delta_spike_mask.iter().map(|r| r.1.as_integer_vector().unwrap().iter().map(|&v| v != 0).collect()).collect(),
        pi_init: pi_init,
        beta_a: pi_beta_a,
        beta_b: pi_beta_b,
    };

    let n_chains = chains as usize;
    let n_iter = iter as usize;
    let n_warmup = warmup as usize;
    let base_seed = seed as u64;
    let n_cores = (n_cores as usize).max(1);

    let results: Vec<(DMatrix<f64>, usize)> = if n_cores > 1 && n_chains > 1 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n_cores).build().unwrap();
        pool.install(|| {
            (0..n_chains).into_par_iter().map(|c| {
                let seed = base_seed.wrapping_add(c as u64 * 1_000_003);
                run_chain_ss(&data, &priors, &ss_config, n_iter, n_warmup, step_om, step_rho, target_accept, seed, false, c, n_chains, &|_,_,_,_,_| {})
            }).collect()
        })
    } else {
        (0..n_chains).map(|c| {
            let seed = base_seed.wrapping_add(c as u64 * 1_000_003);
            run_chain_ss(&data, &priors, &ss_config, n_iter, n_warmup, step_om, step_rho, target_accept, seed, verbose, c, n_chains, &|_,_,_,_,_| {})
        }).collect()
    };

    let chain_results: Vec<Robj> = results.into_iter().map(|(draws, _)| {
        let nr = draws.nrows();
        let nc = draws.ncols();
        let flat: Vec<f64> = draws.iter().cloned().collect();
        RMatrix::new_matrix(nr, nc, |r, c| flat[c * nr + r]).into()
    }).collect();

    list!(draws = chain_results)
}

/// @noRd
/// @keywords internal
#[extendr]
fn run_mcmc_re(
    y: &[f64],
    tau: &[f64],
    x_b0: &[f64], p_b0: i32,
    x_b1: &[f64], p_b1: i32,
    x_deltas: List, p_deltas: &[i32],
    x_om: List, p_om: &[i32],
    x_rho: List, p_rho: &[i32],
    group_b0: &[i32],
    n_groups_b0: i32,
    re_mask_om: List,
    prior_mean_b0: &[f64], prior_sd_b0: &[f64], prior_lb_b0: &[f64], prior_ub_b0: &[f64],
    prior_mean_b1: &[f64], prior_sd_b1: &[f64], prior_lb_b1: &[f64], prior_ub_b1: &[f64],
    prior_mean_deltas: List, prior_sd_deltas: List, prior_lb_deltas: List, prior_ub_deltas: List,
    prior_mean_om: List, prior_sd_om: List, prior_lb_om: List, prior_ub_om: List,
    prior_mean_rho: List, prior_sd_rho: List, prior_lb_rho: List, prior_ub_rho: List,
    sigma_shape: f64,
    sigma_scale: f64,
    sigma_u_shape: f64,
    sigma_u_scale: f64,
    sigma_re_om_shape: f64,
    sigma_re_om_scale: f64,
    step_om: f64,
    step_rho: f64,
    target_accept: f64,
    chains: i32,
    iter: i32,
    warmup: i32,
    seed: i32,
    verbose: bool,
    n_cores: i32,
) -> List {
    let mut p_deltas = p_deltas;
    let mut p_om = p_om;
    let mut p_rho = p_rho;
    let mut group_b0 = group_b0;

    if p_deltas.len() == 1 && p_deltas[0] == -1 { p_deltas = &[]; }
    if p_om.len() == 1 && p_om[0] == -1 { p_om = &[]; }
    if p_rho.len() == 1 && p_rho[0] == -1 { p_rho = &[]; }
    if group_b0.len() == 1 && group_b0[0] == -1 { group_b0 = &[]; }

    let n = y.len();
    let n_bp = p_deltas.len();

    let data = ModelData { outcome_family: crate::model::OutcomeFamily::Gaussian,
        y: DVector::from_column_slice(y),
        tau: DVector::from_column_slice(tau),
        x_b0: flat_to_dmatrix(x_b0, n, p_b0 as usize),
        x_b1: flat_to_dmatrix(x_b1, n, p_b1 as usize),
        x_deltas: list_to_vec_dmatrix(x_deltas, n, p_deltas),
        x_om: list_to_vec_dmatrix(x_om, n, p_om),
        x_rho: list_to_vec_dmatrix(x_rho, n, p_rho),
        group_b0: group_b0.to_vec(),
        n_groups_b0: n_groups_b0 as usize,
        n_breakpoints: n_bp,
        n,
        re_mask_om: re_mask_om.iter().map(|r| r.1.as_integer_vector().unwrap().iter().map(|&v| v != 0).collect()).collect(),
        latent_gps: Vec::new(),
    };

    let priors = Priors { r_shape: 0.0, r_rate: 0.0,
        b0_mean: prior_mean_b0.to_vec(),
        b0_sd: prior_sd_b0.to_vec(),
        b0_lb: prior_lb_b0.to_vec(),
        b0_ub: prior_ub_b0.to_vec(),
        b1_mean: prior_mean_b1.to_vec(),
        b1_sd: prior_sd_b1.to_vec(),
        b1_lb: prior_lb_b1.to_vec(),
        b1_ub: prior_ub_b1.to_vec(),
        delta_mean: prior_mean_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_sd: prior_sd_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_lb: prior_lb_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_ub: prior_ub_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_mean: prior_mean_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_sd: prior_sd_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_lb: prior_lb_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_ub: prior_ub_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_mean: prior_mean_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_sd: prior_sd_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_lb: prior_lb_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_ub: prior_ub_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        sigma_shape,
        sigma_scale,
        sigma_u_shape,
        sigma_u_scale,
        sigma_re_om_shape,
        sigma_re_om_scale,
        p_b0: p_b0 as usize,
        p_b1: p_b1 as usize,
        p_deltas: p_deltas.iter().map(|&p| p as usize).collect(),
        p_om: p_om.iter().map(|&p| p as usize).collect(),
        p_rho: p_rho.iter().map(|&p| p as usize).collect(),
    };

    let n_chains = chains as usize;
    let n_iter = iter as usize;
    let n_warmup = warmup as usize;
    let base_seed = seed as u64;
    let n_cores = (n_cores as usize).max(1);

    let results: Vec<(DMatrix<f64>, usize)> = if n_cores > 1 && n_chains > 1 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n_cores).build().unwrap();
        pool.install(|| {
            (0..n_chains).into_par_iter().map(|c| {
                let seed = base_seed.wrapping_add(c as u64 * 1_000_003);
                run_chain_re(&data, &priors, n_iter, n_warmup, step_om, step_rho, target_accept, seed, false, c, n_chains, &|_,_,_,_,_| {})
            }).collect()
        })
    } else {
        (0..n_chains).map(|c| {
            let seed = base_seed.wrapping_add(c as u64 * 1_000_003);
            run_chain_re(&data, &priors, n_iter, n_warmup, step_om, step_rho, target_accept, seed, verbose, c, n_chains, &|_,_,_,_,_| {})
        }).collect()
    };

    let chain_results: Vec<Robj> = results.into_iter().map(|(draws, _)| {
        let nr = draws.nrows();
        let nc = draws.ncols();
        let flat: Vec<f64> = draws.iter().cloned().collect();
        RMatrix::new_matrix(nr, nc, |r, c| flat[c * nr + r]).into()
    }).collect();

    list!(draws = chain_results)
}

/// @noRd
/// @keywords internal
#[extendr]
fn run_mcmc_re_ss(
    y: &[f64],
    tau: &[f64],
    x_b0: &[f64], p_b0: i32,
    x_b1: &[f64], p_b1: i32,
    x_deltas: List, p_deltas: &[i32],
    x_om: List, p_om: &[i32],
    x_rho: List, p_rho: &[i32],
    group_b0: &[i32],
    n_groups_b0: i32,
    re_mask_om: List,
    prior_mean_b0: &[f64], prior_sd_b0: &[f64], prior_lb_b0: &[f64], prior_ub_b0: &[f64],
    prior_mean_b1: &[f64], prior_sd_b1: &[f64], prior_lb_b1: &[f64], prior_ub_b1: &[f64],
    prior_mean_deltas: List, prior_sd_deltas: List, prior_lb_deltas: List, prior_ub_deltas: List,
    prior_mean_om: List, prior_sd_om: List, prior_lb_om: List, prior_ub_om: List,
    prior_mean_rho: List, prior_sd_rho: List, prior_lb_rho: List, prior_ub_rho: List,
    sigma_shape: f64,
    sigma_scale: f64,
    sigma_u_shape: f64,
    sigma_u_scale: f64,
    sigma_re_om_shape: f64,
    sigma_re_om_scale: f64,
    step_om: f64,
    step_rho: f64,
    target_accept: f64,
    b1_spike_mask: &[i32],
    delta_spike_mask: List,
    pi_init: f64,
    pi_beta_a: f64,
    pi_beta_b: f64,
    chains: i32,
    iter: i32,
    warmup: i32,
    seed: i32,
    verbose: bool,
    n_cores: i32,
) -> List {
    let mut p_deltas = p_deltas;
    let mut p_om = p_om;
    let mut p_rho = p_rho;
    let mut group_b0 = group_b0;
    let mut b1_spike_mask = b1_spike_mask;

    if p_deltas.len() == 1 && p_deltas[0] == -1 { p_deltas = &[]; }
    if p_om.len() == 1 && p_om[0] == -1 { p_om = &[]; }
    if p_rho.len() == 1 && p_rho[0] == -1 { p_rho = &[]; }
    if group_b0.len() == 1 && group_b0[0] == -1 { group_b0 = &[]; }
    if b1_spike_mask.len() == 1 && b1_spike_mask[0] == -1 { b1_spike_mask = &[]; }

    let n = y.len();
    let n_bp = p_deltas.len();

    let data = ModelData { outcome_family: crate::model::OutcomeFamily::Gaussian,
        y: DVector::from_column_slice(y),
        tau: DVector::from_column_slice(tau),
        x_b0: flat_to_dmatrix(x_b0, n, p_b0 as usize),
        x_b1: flat_to_dmatrix(x_b1, n, p_b1 as usize),
        x_deltas: list_to_vec_dmatrix(x_deltas, n, p_deltas),
        x_om: list_to_vec_dmatrix(x_om, n, p_om),
        x_rho: list_to_vec_dmatrix(x_rho, n, p_rho),
        group_b0: group_b0.to_vec(),
        n_groups_b0: n_groups_b0 as usize,
        n_breakpoints: n_bp,
        n,
        re_mask_om: re_mask_om.iter().map(|r| r.1.as_integer_vector().unwrap().iter().map(|&v| v != 0).collect()).collect(),
        latent_gps: Vec::new(),
    };

    let priors = Priors { r_shape: 0.0, r_rate: 0.0,
        b0_mean: prior_mean_b0.to_vec(),
        b0_sd: prior_sd_b0.to_vec(),
        b0_lb: prior_lb_b0.to_vec(),
        b0_ub: prior_ub_b0.to_vec(),
        b1_mean: prior_mean_b1.to_vec(),
        b1_sd: prior_sd_b1.to_vec(),
        b1_lb: prior_lb_b1.to_vec(),
        b1_ub: prior_ub_b1.to_vec(),
        delta_mean: prior_mean_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_sd: prior_sd_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_lb: prior_lb_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_ub: prior_ub_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_mean: prior_mean_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_sd: prior_sd_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_lb: prior_lb_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_ub: prior_ub_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_mean: prior_mean_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_sd: prior_sd_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_lb: prior_lb_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_ub: prior_ub_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        sigma_shape,
        sigma_scale,
        sigma_u_shape,
        sigma_u_scale,
        sigma_re_om_shape,
        sigma_re_om_scale,
        p_b0: p_b0 as usize,
        p_b1: p_b1 as usize,
        p_deltas: p_deltas.iter().map(|&p| p as usize).collect(),
        p_om: p_om.iter().map(|&p| p as usize).collect(),
        p_rho: p_rho.iter().map(|&p| p as usize).collect(),
    };

    let ss = SpikeSlabConfig {
        b1_spike_mask: b1_spike_mask.iter().map(|&v| v != 0).collect(),
        delta_spike_mask: delta_spike_mask.iter().map(|r| r.1.as_integer_vector().unwrap().iter().map(|&v| v != 0).collect()).collect(),
        pi_init,
        beta_a: pi_beta_a,
        beta_b: pi_beta_b,
    };

    let n_chains = chains as usize;
    let n_iter = iter as usize;
    let n_warmup = warmup as usize;
    let base_seed = seed as u64;
    let n_cores = (n_cores as usize).max(1);

    let results: Vec<(DMatrix<f64>, usize)> = if n_cores > 1 && n_chains > 1 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n_cores).build().unwrap();
        pool.install(|| {
            (0..n_chains).into_par_iter().map(|c| {
                let seed = base_seed.wrapping_add(c as u64 * 1_000_003);
                run_chain_re_ss(&data, &priors, &ss, n_iter, n_warmup, step_om, step_rho, target_accept, seed, false, c, n_chains, &|_,_,_,_,_| {})
            }).collect()
        })
    } else {
        (0..n_chains).map(|c| {
            let seed = base_seed.wrapping_add(c as u64 * 1_000_003);
            run_chain_re_ss(&data, &priors, &ss, n_iter, n_warmup, step_om, step_rho, target_accept, seed, verbose, c, n_chains, &|_,_,_,_,_| {})
        }).collect()
    };

    let chain_results: Vec<Robj> = results.into_iter().map(|(draws, _)| {
        let nr = draws.nrows();
        let nc = draws.ncols();
        let flat: Vec<f64> = draws.iter().cloned().collect();
        RMatrix::new_matrix(nr, nc, |r, c| flat[c * nr + r]).into()
    }).collect();

    list!(draws = chain_results)
}

/// @noRd
/// @keywords internal
#[extendr]
fn run_bjlm(
    // Outcome data
    y: &[f64],
    tau: &[f64],
    x_b0: &[f64], p_b0: i32,
    x_b1: &[f64], p_b1: i32,
    x_deltas: List, p_deltas: &[i32],
    x_om: List, p_om: &[i32],
    x_rho: List, p_rho: &[i32],
    group_b0: &[i32],
    n_groups_b0: i32,
    // Outcome priors
    prior_mean_b0: &[f64], prior_sd_b0: &[f64], prior_lb_b0: &[f64], prior_ub_b0: &[f64],
    prior_mean_b1: &[f64], prior_sd_b1: &[f64], prior_lb_b1: &[f64], prior_ub_b1: &[f64],
    prior_mean_deltas: List, prior_sd_deltas: List, prior_lb_deltas: List, prior_ub_deltas: List,
    prior_mean_om: List, prior_sd_om: List, prior_lb_om: List, prior_ub_om: List,
    prior_mean_rho: List, prior_sd_rho: List, prior_lb_rho: List, prior_ub_rho: List,
    sigma_shape: f64,
    sigma_scale: f64,
    sigma_u_shape: f64,
    sigma_u_scale: f64,
    prior_r_shape: f64,
    prior_r_rate: f64,
    // Propensity data
    x_prop: &[f64], p_prop: i32,
    latent_gps: List,
    treatment: &[f64],
    n_subjects: i32,
    // Propensity priors
    prop_prior_sd: f64,
    // Weight config
    weight_type: i32,
    max_weight: f64,
    // MCMC settings
    step_om: f64,
    step_rho: f64,
    target_accept: f64,
    chains: i32,
    iter: i32,
    warmup: i32,
    seed: i32,
    verbose: bool,
    n_cores: i32,
    outcome_family: &str,
) -> List {
    let mut p_deltas = p_deltas;
    let mut p_om = p_om;
    let mut p_rho = p_rho;
    let mut group_b0 = group_b0;

    if p_deltas.len() == 1 && p_deltas[0] == -1 { p_deltas = &[]; }
    if p_om.len() == 1 && p_om[0] == -1 { p_om = &[]; }
    if p_rho.len() == 1 && p_rho[0] == -1 { p_rho = &[]; }
    if group_b0.len() == 1 && group_b0[0] == -1 { group_b0 = &[]; }

    let n = y.len();
    let n_bp = p_deltas.len();
    let n_subj = n_subjects as usize;
    let p_pr = p_prop as usize;

    let outcome_family_enum = match outcome_family {
        "binomial" => crate::model::OutcomeFamily::Binomial,
        "negative_binomial" => crate::model::OutcomeFamily::NegativeBinomial,
        _ => crate::model::OutcomeFamily::Gaussian,
    };

    let outcome_data = ModelData {
        outcome_family: outcome_family_enum,
        y: DVector::from_column_slice(y),
        tau: DVector::from_column_slice(tau),
        x_b0: flat_to_dmatrix(x_b0, n, p_b0 as usize),
        x_b1: flat_to_dmatrix(x_b1, n, p_b1 as usize),
        x_deltas: list_to_vec_dmatrix(x_deltas, n, p_deltas),
        x_om: list_to_vec_dmatrix(x_om, n, p_om),
        x_rho: list_to_vec_dmatrix(x_rho, n, p_rho),
        group_b0: group_b0.to_vec(),
        n_groups_b0: n_groups_b0 as usize,
        n_breakpoints: n_bp,
        n,
        re_mask_om: Vec::new(),
        latent_gps: {
            let mut gps = Vec::new();
            for (_, robj) in latent_gps.iter() {
                let gp_list = robj.as_list().unwrap();
                let mut gp = crate::model::GpData {
                    name: gp_list.dollar("name").unwrap().as_str().unwrap().to_string(),
                    obs_time: gp_list.dollar("obs_time").unwrap().as_real_vector().unwrap(),
                    obs_val: gp_list.dollar("obs_val").unwrap().as_real_vector().unwrap(),
                    obs_group: gp_list.dollar("obs_group").unwrap().as_integer_vector().unwrap().into_iter().map(|g| g as usize).collect(),
                    
                    trt_time: gp_list.dollar("trt_time").unwrap().as_real_vector().unwrap(),
                    trt_group: gp_list.dollar("trt_group").unwrap().as_integer_vector().unwrap().into_iter().map(|g| g as usize).collect(),
                    
                    out_time: gp_list.dollar("out_time").unwrap().as_real_vector().unwrap(),
                    out_group: gp_list.dollar("out_group").unwrap().as_integer_vector().unwrap().into_iter().map(|g| g as usize).collect(),
                    
                    p_b0_idx: gp_list.dollar("p_b0_idx").unwrap().as_integer_vector().unwrap()[0],
                    p_b1_idx: gp_list.dollar("p_b1_idx").unwrap().as_integer_vector().unwrap()[0],
                    p_prop_idx: gp_list.dollar("p_prop_idx").unwrap().as_integer_vector().unwrap()[0],
                    subjects: Vec::new(),
                };
                gp.process(n_subj);
                gps.push(gp);
            }
            gps
        },
    };

    let outcome_priors = Priors {
        b0_mean: prior_mean_b0.to_vec(),
        b0_sd: prior_sd_b0.to_vec(),
        b0_lb: prior_lb_b0.to_vec(),
        b0_ub: prior_ub_b0.to_vec(),
        b1_mean: prior_mean_b1.to_vec(),
        b1_sd: prior_sd_b1.to_vec(),
        b1_lb: prior_lb_b1.to_vec(),
        b1_ub: prior_ub_b1.to_vec(),
        delta_mean: prior_mean_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_sd: prior_sd_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_lb: prior_lb_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        delta_ub: prior_ub_deltas.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_mean: prior_mean_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_sd: prior_sd_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_lb: prior_lb_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        om_ub: prior_ub_om.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_mean: prior_mean_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_sd: prior_sd_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_lb: prior_lb_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        rho_ub: prior_ub_rho.iter().map(|r| r.1.as_real_vector().unwrap()).collect(),
        sigma_shape,
        sigma_scale,
        sigma_u_shape,
        sigma_u_scale,
        sigma_re_om_shape: 1.0,
        sigma_re_om_scale: 1.0,
        r_shape: prior_r_shape,
        r_rate: prior_r_rate,
        p_b0: p_b0 as usize,
        p_b1: p_b1 as usize,
        p_deltas: p_deltas.iter().map(|&p| p as usize).collect(),
        p_om: p_om.iter().map(|&p| p as usize).collect(),
        p_rho: p_rho.iter().map(|&p| p as usize).collect(),
    };

    // Propensity data: subject-level design matrix
    let x_prop_mat = DMatrix::from_column_slice(n_subj, p_pr, x_prop);
    let prop_data = PropensityData {
        x_prop: x_prop_mat,
        treatment: treatment.to_vec(),
        n_subjects: n_subj,
        p_prop: p_pr,
    };

    let prop_priors = PropensityPriors::isotropic(p_pr, prop_prior_sd);

    let config = BjlmConfig {
        weight_type: WeightType::from_i32(weight_type),
        max_weight,
    };

    let n_chains = chains as usize;
    let n_iter = iter as usize;
    let n_warmup = warmup as usize;
    let base_seed = seed as u64;
    let n_cores_val = (n_cores as usize).max(1);

    let results: Vec<(DMatrix<f64>, usize)> = if n_cores_val > 1 && n_chains > 1 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n_cores_val).build().unwrap();
        pool.install(|| {
            (0..n_chains).into_par_iter().map(|c| {
                let seed = base_seed.wrapping_add(c as u64 * 1_000_003);
                run_chain_bjlm(
                    &outcome_data, &outcome_priors, &prop_data, &prop_priors, &config,
                    n_iter, n_warmup, step_om, step_rho, target_accept,
                    seed, false, c, n_chains, &|_,_,_,_,_| {},
                )
            }).collect()
        })
    } else {
        (0..n_chains).map(|c| {
            let seed = base_seed.wrapping_add(c as u64 * 1_000_003);
            run_chain_bjlm(
                &outcome_data, &outcome_priors, &prop_data, &prop_priors, &config,
                n_iter, n_warmup, step_om, step_rho, target_accept,
                seed, verbose, c, n_chains, &|_,_,_,_,_| {},
            )
        }).collect()
    };

    let chain_results: Vec<Robj> = results.into_iter().map(|(draws, _)| {
        let nr = draws.nrows();
        let nc = draws.ncols();
        let flat: Vec<f64> = draws.iter().cloned().collect();
        RMatrix::new_matrix(nr, nc, |r, c| flat[c * nr + r]).into()
    }).collect();

    list!(draws = chain_results)
}

extendr_module! {
    mod bjlm;
    fn run_mcmc;
    fn run_mcmc_ss;
    fn run_mcmc_re;
    fn run_mcmc_re_ss;
    fn run_bjlm;
}
