use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};

use crate::polya_gamma::sample_pg;

// ---------------------------------------------------------------------------
// Pólya-Gamma augmented Gibbs sampler for logistic regression
//
// Implements the propensity score model:
//   T_i | X_i, α ~ Bernoulli(sigmoid(X_i' α))
//
// Using PG augmentation, this becomes a conjugate update:
//   ω_i | α ~ PG(1, X_i' α)
//   α | ω, T ~ N(m_ω, Σ_ω)
//
// where:
//   Σ_ω = (X' Ω X + B^{-1})^{-1}
//   m_ω = Σ_ω (X' (T - 1/2) + B^{-1} b)
//   Ω = diag(ω_1, ..., ω_n)
//   Prior: α ~ N(b, B)
// ---------------------------------------------------------------------------

/// State for the propensity model.
pub struct PropensityState {
    /// Regression coefficients for propensity model
    pub alpha: DVector<f64>,
    /// PG latent variables (one per subject)
    pub omega_pg: Vec<f64>,
    /// Propensity scores (cached after each update)
    pub pi: Vec<f64>,
}

/// Priors for the propensity model.
pub struct PropensityPriors {
    /// Prior mean for α
    pub mean: DVector<f64>,
    /// Prior precision matrix B^{-1}
    pub precision: DMatrix<f64>,
}

impl PropensityPriors {
    /// Create isotropic normal prior: α_j ~ N(0, prior_sd^2)
    pub fn isotropic(p: usize, prior_sd: f64) -> Self {
        let prec = 1.0 / (prior_sd * prior_sd);
        PropensityPriors {
            mean: DVector::zeros(p),
            precision: DMatrix::from_diagonal(&DVector::from_element(p, prec)),
        }
    }
}

/// Propensity data (subject-level design matrix and treatment).
pub struct PropensityData {
    /// Design matrix for propensity model (n_subjects × p_prop)
    pub x_prop: DMatrix<f64>,
    /// Treatment indicator (0/1), length n_subjects
    pub treatment: Vec<f64>,
    /// Number of subjects
    pub n_subjects: usize,
    /// Number of propensity covariates
    pub p_prop: usize,
}

impl PropensityState {
    /// Initialise with zeros + small jitter.
    pub fn new(p: usize, n_subjects: usize, rng: &mut StdRng) -> Self {
        let jitter = Normal::new(0.0, 0.01).unwrap();
        let alpha = DVector::from_iterator(p, (0..p).map(|_| jitter.sample(rng)));
        let omega_pg = vec![0.25; n_subjects]; // E[PG(1,0)] = 0.25
        let pi = vec![0.5; n_subjects];
        PropensityState { alpha, omega_pg, pi }
    }
}

/// One Gibbs iteration of the propensity block (PG-augmented).
///
/// This is a cut-feedback update: it uses ONLY the propensity likelihood
/// p(T | X, α), NOT the outcome data. This prevents the outcome from
/// influencing the propensity score estimates.
///
/// Returns: nothing (updates state in place). Propensity scores are
/// recomputed and cached in state.pi.
pub fn sample_propensity(
    data: &PropensityData,
    priors: &PropensityPriors,
    state: &mut PropensityState,
    outcome_data: &crate::model::ModelData,
    outcome_state: &crate::model::State,
    rng: &mut StdRng,
) {
    let n = data.n_subjects;
    let p = data.p_prop;

    // Helper to get x_prop(i, j) with GP injected
    let get_x = |i: usize, j: usize| -> f64 {
        for (gp_idx, gp) in outcome_data.latent_gps.iter().enumerate() {
            if gp.p_prop_idx == j as i32 {
                let subj = &gp.subjects[i];
                let gp_x = &outcome_state.gp_states[gp_idx].x[i];
                if !subj.trt_indices.is_empty() {
                    return gp_x[subj.trt_indices[0]];
                }
            }
        }
        data.x_prop[(i, j)]
    };

    let is_continuous = data.treatment.iter().any(|&t| t != 0.0 && t != 1.0);

    if is_continuous {
        // --- Continuous exposure: generalized propensity score (GPS) under a normal linear regression model ---
        let mut sigma_a_sq = state.omega_pg[0];
        if sigma_a_sq <= 0.0 || sigma_a_sq.is_nan() {
            sigma_a_sq = 1.0;
        }

        // Build normal equations X' X and X' T
        let mut xtx = DMatrix::<f64>::zeros(p, p);
        let mut xtt = DVector::<f64>::zeros(p);

        for i in 0..n {
            let ti = data.treatment[i];
            for j in 0..p {
                let xij = get_x(i, j);
                xtt[j] += xij * ti;
                for l in j..p {
                    let v = xij * get_x(i, l);
                    xtx[(j, l)] += v;
                    if l != j {
                        xtx[(l, j)] += v;
                    }
                }
            }
        }

        // Precision = X' X / sigma_a_sq + Prior Precision
        let precision = (&xtx / sigma_a_sq) + &priors.precision;

        let chol = precision
            .cholesky()
            .expect("Propensity precision matrix not positive definite");

        let rhs = (&xtt / sigma_a_sq) + &priors.precision * &priors.mean;
        let mean = chol.solve(&rhs);

        // Sample alpha ~ N(mean, precision^{-1})
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = DVector::from_iterator(p, (0..p).map(|_| normal.sample(rng)));
        let v = chol
            .l()
            .transpose()
            .solve_upper_triangular(&z)
            .expect("Failed to solve upper triangular system");
        state.alpha = mean + v;

        // Sample sigma_a_sq ~ Inv-Gamma(a_post, b_post)
        let mut sum_sq_resid = 0.0;
        for i in 0..n {
            let mut pred = 0.0;
            for j in 0..p {
                pred += get_x(i, j) * state.alpha[j];
            }
            let resid = data.treatment[i] - pred;
            sum_sq_resid += resid * resid;
        }
        let a_post = 1.0 + (n as f64) / 2.0;
        let b_post = 1.0 + 0.5 * sum_sq_resid;

        let gamma_dist = rand_distr::Gamma::new(a_post, 1.0 / b_post).unwrap();
        let sampled_prec = gamma_dist.sample(rng);
        let new_sigma_a_sq = 1.0 / sampled_prec;
        state.omega_pg[0] = new_sigma_a_sq;

        // Recompute propensity conditional densities f(T_i | X_i, alpha, sigma_a_sq)
        for i in 0..n {
            let mut pred = 0.0;
            for j in 0..p {
                pred += get_x(i, j) * state.alpha[j];
            }
            let diff = data.treatment[i] - pred;
            let density = (1.0 / (2.0 * std::f64::consts::PI * new_sigma_a_sq).sqrt()) * 
                          (-0.5 * diff * diff / new_sigma_a_sq).exp();
            state.pi[i] = density;
        }
    } else {
        // --- Binary exposure: logistic regression with PG augmentation ---
        // Step 1: Sample PG latent variables
        //   ω_i | α ~ PG(1, X_i' α)
        for i in 0..n {
            let mut psi = 0.0;
            for j in 0..p {
                psi += get_x(i, j) * state.alpha[j];
            }
            state.omega_pg[i] = sample_pg(1.0, psi, rng);
        }

        // Step 2: Build weighted normal equations
        //   Σ_ω = (X' Ω X + B^{-1})^{-1}
        //   m_ω = Σ_ω (X' κ + B^{-1} b)
        //   where κ_i = T_i - 0.5

        // X' Ω X
        let mut xtox = DMatrix::<f64>::zeros(p, p);
        let mut xtk = DVector::<f64>::zeros(p);

        for i in 0..n {
            let wi = state.omega_pg[i];
            let ki = data.treatment[i] - 0.5;
            for j in 0..p {
                let xij = get_x(i, j);
                xtk[j] += xij * ki;
                for l in j..p {
                    let v = xij * wi * get_x(i, l);
                    xtox[(j, l)] += v;
                    if l != j {
                        xtox[(l, j)] += v;
                    }
                }
            }
        }

        // Add prior precision
        let precision = xtox + &priors.precision;

        // Solve via Cholesky
        let chol = precision
            .cholesky()
            .expect("Propensity precision matrix not positive definite");

        let rhs = xtk + &priors.precision * &priors.mean;
        let mean = chol.solve(&rhs);

        // Sample α ~ N(mean, precision^{-1})
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = DVector::from_iterator(p, (0..p).map(|_| normal.sample(rng)));
        let v = chol
            .l()
            .transpose()
            .solve_upper_triangular(&z)
            .expect("Failed to solve upper triangular system in propensity sampling");
        state.alpha = mean + v;

        // Step 3: Recompute propensity scores
        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += get_x(i, j) * state.alpha[j];
            }
            state.pi[i] = crate::model::sigmoid(eta);
        }
    }
}
