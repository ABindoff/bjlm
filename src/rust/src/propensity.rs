use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};

use crate::polya_gamma::sample_pg1;

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
    rng: &mut StdRng,
) {
    let n = data.n_subjects;
    let p = data.p_prop;

    // Step 1: Sample PG latent variables
    //   ω_i | α ~ PG(1, X_i' α)
    for i in 0..n {
        let mut psi = 0.0;
        for j in 0..p {
            psi += data.x_prop[(i, j)] * state.alpha[j];
        }
        state.omega_pg[i] = sample_pg1(psi, rng);
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
            let xij = data.x_prop[(i, j)];
            xtk[j] += xij * ki;
            for l in j..p {
                let v = xij * wi * data.x_prop[(i, l)];
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
    // L' * v = z => v = (L')^{-1} z
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
            eta += data.x_prop[(i, j)] * state.alpha[j];
        }
        state.pi[i] = crate::model::sigmoid(eta);
    }
}
