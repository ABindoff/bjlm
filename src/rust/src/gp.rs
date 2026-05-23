use nalgebra::{DMatrix, DVector};

/// Compute Squared Exponential covariance matrix
pub fn compute_cov_matrix(
    times: &[f64],
    alpha: f64,
    rho: f64,
    jitter: f64,
) -> DMatrix<f64> {
    let n = times.len();
    let mut k = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let dist = times[i] - times[j];
            let val = alpha * alpha * (-0.5 * (dist * dist) / (rho * rho)).exp();
            k[(i, j)] = val;
            if i != j {
                k[(j, i)] = val;
            }
        }
        k[(i, i)] += jitter;
    }
    k
}
