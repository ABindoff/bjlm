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
}

impl WeightType {
    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => WeightType::Ate,
            1 => WeightType::Att,
            2 => WeightType::StabilisedAte,
            3 => WeightType::StabilisedAtt,
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
