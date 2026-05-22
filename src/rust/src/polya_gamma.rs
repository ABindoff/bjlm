use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution, Exp1, StandardNormal};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Pólya-Gamma PG(1, c) sampler
//
// Reference: Polson, Scott, Windle (2013). "Bayesian Inference for Logistic
// Models Using Pólya–Gamma Latent Variables." JASA 108(504): 1339–1349.
//
// Uses the method of Devroye (2009) as described in Windle et al. (2014).
// ---------------------------------------------------------------------------

const TRUNC: f64 = 0.64;
const PI_SQ_OVER_8: f64 = PI * PI / 8.0;

/// Sample from PG(1, c).
///
/// PG(1, c) has mean tanh(c/2) / (2c) and is symmetric in c.
pub fn sample_pg1(c: f64, rng: &mut StdRng) -> f64 {
    let z = c.abs() * 0.5;

    // For c ≈ 0, use the series representation
    if z < 1e-12 {
        return sample_pg1_zero(rng);
    }

    let k = PI_SQ_OVER_8 + z * z * 0.5;

    // Compute mixture weights for proposal
    // p: mass from truncated exponential (x >= TRUNC)
    // q: mass from truncated inverse-Gaussian (x < TRUNC)
    let p = PI * 0.5 / k * (-k * TRUNC).exp();
    let q = 2.0 * (-z).exp() * cdf_inverse_gaussian(TRUNC, 1.0 / z, 1.0);

    loop {
        let u: f64 = rng.gen();
        let x: f64;

        if u * (p + q) < p {
            // Propose from truncated exponential: x ~ Exp(K) + TRUNC
            let e: f64 = Exp1.sample(rng);
            x = TRUNC + e / k;
        } else {
            // Propose from truncated inverse-Gaussian
            x = sample_truncated_ig(1.0 / z, 1.0, TRUNC, rng);
        }

        // Accept/reject using alternating series
        let mut s = a_coef(0, x);
        let y: f64 = rng.gen::<f64>() * s;
        let mut n = 0u32;

        loop {
            n += 1;
            if n % 2 == 1 {
                // Odd: subtract
                s -= a_coef(n, x);
                if y <= s {
                    return x / 4.0; // PG(1, c) = J*(1, z) / 4
                }
            } else {
                // Even: add
                s += a_coef(n, x);
                if y > s {
                    break; // reject, retry outer loop
                }
            }
            if n > 200 {
                // Safety: shouldn't happen, but avoid infinite loop
                return x / 4.0;
            }
        }
    }
}

/// Coefficients for the alternating series.
/// a_n(x) = pi * (n + 0.5) * exp(-(n + 0.5)^2 * pi^2 * x / 2)
fn a_coef(n: u32, x: f64) -> f64 {
    let nh = n as f64 + 0.5;
    PI * nh * (-nh * nh * PI * PI * x * 0.5).exp()
}

/// PG(1, 0) via truncated series.
/// PG(1, 0) = sum_{k=0}^inf G_k / ((k+0.5)^2 * 4 * pi^2)
/// where G_k ~ Exp(1).
fn sample_pg1_zero(rng: &mut StdRng) -> f64 {
    let mut x = 0.0;
    for k in 0..20 {
        let g: f64 = Exp1.sample(rng);
        let kh = k as f64 + 0.5;
        x += g / (kh * kh * 4.0 * PI * PI);
    }
    x
}

/// CDF of the inverse Gaussian distribution IG(mu, lambda) at x.
fn cdf_inverse_gaussian(x: f64, mu: f64, lambda: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let sqrt_lx = (lambda / x).sqrt();
    let t1 = normal_cdf(sqrt_lx * (x / mu - 1.0));
    let t2 = (2.0 * lambda / mu).exp() * normal_cdf(-sqrt_lx * (x / mu + 1.0));
    t1 + t2
}

/// Standard normal CDF (using the error function approximation).
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Approximation of the error function.
/// Uses the Abramowitz and Stegun approximation (max error ~1.5e-7).
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Sample from a truncated inverse Gaussian IG(mu, lambda=1)
/// truncated to (0, trunc].
///
/// Uses the standard sampling method for IG:
///   1. Generate v ~ N(0,1), chi2 = v^2
///   2. x = mu + mu^2*chi2/(2*lambda) - mu/(2*lambda)*sqrt(4*mu*lambda*chi2 + mu^2*chi2^2)
///   3. Accept x with probability mu/(mu+x), else return mu^2/x
///   4. Reject if x >= trunc
///
/// For very small mu (mu < trunc), the acceptance rate is high.
/// For mu >> trunc, we use the approximation that the truncated IG
/// is approximately a truncated half-normal.
fn sample_truncated_ig(mu: f64, lambda: f64, trunc: f64, rng: &mut StdRng) -> f64 {
    if mu > trunc {
        // mu > trunc: use right-truncated normal approximation
        // IG(mu, 1) for large mu is approximately N(mu, mu^3/lambda)
        // We just sample from the half-normal method
        loop {
            let y: f64 = StandardNormal.sample(rng);
            let y = y * y; // chi-squared(1)
            let x = mu + mu * mu * y / (2.0 * lambda)
                - mu / (2.0 * lambda)
                    * (4.0 * mu * lambda * y + mu * mu * y * y).sqrt();
            if x <= 0.0 || x.is_nan() {
                continue;
            }
            let u: f64 = rng.gen();
            let result = if u <= mu / (mu + x) { x } else { mu * mu / x };
            if result < trunc && result > 0.0 {
                return result;
            }
        }
    } else {
        // mu <= trunc: standard IG sampling, reject if >= trunc
        loop {
            let y: f64 = StandardNormal.sample(rng);
            let y = y * y;
            let x = mu + mu * mu * y / (2.0 * lambda)
                - mu / (2.0 * lambda)
                    * (4.0 * mu * lambda * y + mu * mu * y * y).sqrt();
            if x <= 0.0 || x.is_nan() {
                continue;
            }
            let u: f64 = rng.gen();
            let result = if u <= mu / (mu + x) { x } else { mu * mu / x };
            if result < trunc && result > 0.0 {
                return result;
            }
        }
    }
}

/// Verify PG sampler: compute sample mean and compare with E[PG(1,c)] = tanh(c/2)/(2c).
#[allow(dead_code)]
pub fn test_pg_mean(c: f64, n_samples: usize) -> (f64, f64) {
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(42);
    let mut sum = 0.0;
    for _ in 0..n_samples {
        sum += sample_pg1(c, &mut rng);
    }
    let sample_mean = sum / n_samples as f64;
    let true_mean = if c.abs() < 1e-12 {
        0.25 // lim_{c->0} tanh(c/2)/(2c) = 1/4
    } else {
        (c * 0.5).tanh() / (2.0 * c)
    };
    (sample_mean, true_mean)
}
