use crate::model::sigmoid;

// ---------------------------------------------------------------------------
// IPW weight types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum WeightType {
    Ate,
    Att,
    StabilisedAte,
    StabilisedAtt,
    /// Uniform (all weights = 1): an UNWEIGHTED outcome fit. Used for the
    /// conditional outcome regression E[Y|X,T] that G-computation and textbook
    /// AIPW require (the weighted MSM fit is not a conditional regression).
    Uniform,
}

impl WeightType {
    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => WeightType::Ate,
            1 => WeightType::Att,
            2 => WeightType::StabilisedAte,
            3 => WeightType::StabilisedAtt,
            4 => WeightType::Uniform,
            _ => WeightType::StabilisedAte, // default
        }
    }
}

// ---------------------------------------------------------------------------
// Weight computation
// ---------------------------------------------------------------------------

/// Compute IPW weights from propensity scores.
///
/// # Arguments
/// * `treatment`  - Binary treatment indicators (0/1) for each *subject*
/// * `pi`         - Propensity scores π_i for each subject
/// * `weight_type` - Type of weights (ATE, ATT, stabilised)
/// * `max_weight`  - Trimming threshold (weights clamped to this value)
///
/// # Returns
/// A vector of weights, one per subject.
pub fn compute_weights(
    treatment: &[f64],
    pi: &[f64],
    weight_type: WeightType,
    max_weight: f64,
) -> Vec<f64> {
    let n = treatment.len();

    // Uniform weights = unweighted fit (conditional outcome regression), independent
    // of treatment type or propensity. Return early before any pi-based logic.
    if let WeightType::Uniform = weight_type {
        return vec![1.0; n];
    }

    let is_continuous = treatment.iter().any(|&t| t != 0.0 && t != 1.0);

    if is_continuous {
        let mean_t = treatment.iter().sum::<f64>() / n as f64;
        let sum_sq_diff: f64 = treatment.iter().map(|&t| (t - mean_t) * (t - mean_t)).sum();
        let var_t = if n > 1 { sum_sq_diff / ((n - 1) as f64) } else { 1.0 };
        let sd_t = if var_t > 0.0 { var_t.sqrt() } else { 1.0 };
        
        let mut weights = vec![0.0; n];
        for i in 0..n {
            // Clamp conditional density to avoid division by zero or extreme weights
            let cond_dens = pi[i].max(1e-10);
            
            let diff = treatment[i] - mean_t;
            let marg_dens = (1.0 / (2.0 * std::f64::consts::PI * var_t).sqrt()) * 
                            (-0.5 * diff * diff / var_t).exp();
            
            let is_stabilised = match weight_type {
                WeightType::StabilisedAte | WeightType::StabilisedAtt => true,
                _ => false,
            };
            
            weights[i] = if is_stabilised {
                marg_dens / cond_dens
            } else {
                1.0 / cond_dens
            };
            
            weights[i] = weights[i].min(max_weight);
        }
        return weights;
    }

    let p_marginal = treatment.iter().sum::<f64>() / n as f64;

    // If treatment has no variation (all treated or all control),
    // then propensity weighting is not meaningful/applicable (it's a dummy treatment model).
    // In this case, we bypass weighting by returning a vector of 1.0.
    if p_marginal == 0.0 || p_marginal == 1.0 {
        return vec![1.0; n];
    }

    let mut weights = vec![0.0; n];

    for i in 0..n {
        // Clamp pi to prevent extreme weights
        let pi_i = pi[i].clamp(1e-6, 1.0 - 1e-6);

        weights[i] = match weight_type {
            WeightType::Ate => {
                if treatment[i] > 0.5 {
                    1.0 / pi_i
                } else {
                    1.0 / (1.0 - pi_i)
                }
            }
            WeightType::Att => {
                if treatment[i] > 0.5 {
                    1.0
                } else {
                    pi_i / (1.0 - pi_i)
                }
            }
            WeightType::StabilisedAte => {
                if treatment[i] > 0.5 {
                    p_marginal / pi_i
                } else {
                    (1.0 - p_marginal) / (1.0 - pi_i)
                }
            }
            WeightType::StabilisedAtt => {
                if treatment[i] > 0.5 {
                    1.0
                } else {
                    p_marginal * pi_i / ((1.0 - p_marginal) * (1.0 - pi_i))
                }
            }
            // Unreachable: handled by the early return above. Present for exhaustiveness.
            WeightType::Uniform => 1.0,
        };

        weights[i] = weights[i].min(max_weight);
    }
    weights
}

/// Compute propensity scores from design matrix and coefficients.
pub fn compute_propensity_scores(
    x_prop: &[f64],
    n: usize,
    p: usize,
    alpha: &[f64],
) -> Vec<f64> {
    let mut pi = vec![0.0; n];
    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..p {
            eta += x_prop[j * n + i] * alpha[j]; // column-major access
        }
        pi[i] = sigmoid(eta);
    }
    pi
}

/// Expand subject-level weights to observation-level weights.
///
/// In longitudinal data, each subject has multiple observations. This function
/// maps subject-level weights to observation-level using group indices.
///
/// # Arguments
/// * `weights_subj` - Subject-level weights (length n_subjects)
/// * `group_indices` - For each observation, the subject index (0-based)
/// * `n_obs` - Total number of observations
pub fn expand_weights_to_obs(
    weights_subj: &[f64],
    group_indices: &[i32],
    n_obs: usize,
) -> Vec<f64> {
    let mut weights_obs = vec![1.0; n_obs];
    for i in 0..n_obs {
        let g = group_indices[i];
        if g >= 0 {
            weights_obs[i] = weights_subj[g as usize];
        }
    }
    weights_obs
}
