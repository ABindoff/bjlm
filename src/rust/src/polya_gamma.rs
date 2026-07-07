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
    // A non-finite tilt (from a diverged linear predictor upstream) would make the
    // rejection loop below spin forever (every proposal is NaN and is skipped).
    // Fail fast with an actionable message instead of hanging the session.
    if !c.is_finite() {
        panic!("Polya-Gamma sampler received a non-finite tilt (c = {c}); the chain \
                diverged (non-finite linear predictor). Check priors/data/scaling.");
    }
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

/// Coefficients a_n(x) for the alternating series in Devroye's J*(1,0) sampler.
///
/// The series representation of the J* density is PIECEWISE (Devroye 2009;
/// Windle et al. 2014, as in the BayesLogit reference implementation): the two
/// forms are analytically equal but only one is a decreasing-in-`n` alternating
/// sequence in each regime, and the accept/reject squeeze in `sample_pg1`
/// REQUIRES that monotonicity to be a valid bound. The truncated-inverse-Gaussian
/// proposal branch only ever produces `x < TRUNC`, so using the large-x form there
/// (which is non-monotone for small x, e.g. a_1 > a_0 at x = 0.01) silently
/// corrupts the accepted draws. Match the regime to `x`:
///   x >  TRUNC:  a_n(x) = K * exp(-K^2 x / 2)
///   x <= TRUNC:  a_n(x) = K * (pi x / 2)^(-3/2) * exp(-2 (n+1/2)^2 / x)
/// with K = (n + 1/2) * pi.
fn a_coef(n: u32, x: f64) -> f64 {
    let nh = n as f64 + 0.5;
    let k = nh * PI;
    if x > TRUNC {
        k * (-0.5 * k * k * x).exp()
    } else if x > 0.0 {
        let expnt = k.ln() - 1.5 * ((0.5 * PI).ln() + x.ln()) - 2.0 * nh * nh / x;
        expnt.exp()
    } else {
        0.0
    }
}

/// PG(1, 0) via the truncated Gamma series (Polson, Scott & Windle 2013):
/// PG(1, 0) = sum_{k=0}^inf G_k / ((k+0.5)^2 * 2 * pi^2),  G_k ~ Exp(1).
/// The denominator constant is 2*pi^2, not 4*pi^2: with 4*pi^2 the mean is
/// 1/8 instead of the correct E[PG(1,0)] = 1/4 (since sum 1/(k+1/2)^2 = pi^2/2).
fn sample_pg1_zero(rng: &mut StdRng) -> f64 {
    let mut x = 0.0;
    for k in 0..20 {
        let g: f64 = Exp1.sample(rng);
        let kh = k as f64 + 0.5;
        x += g / (kh * kh * 2.0 * PI * PI);
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
    // Numerically stable exact Inverse Gaussian sampler.
    // Standard formula: x = mu + (mu^2 * y)/(2*lambda) - (mu/(2*lambda)) * sqrt(4*mu*lambda*y + mu^2*y^2)
    // Suffers from catastrophic cancellation when mu is large.
    // Instead we use the identity A - B = (A^2 - B^2) / (A + B) = mu^2 / (A + B)
    // where A = mu + (mu^2 * y)/(2*lambda)
    // and   B = (mu/(2*lambda)) * sqrt(4*mu*lambda*y + mu^2*y^2)
    loop {
        let y: f64 = StandardNormal.sample(rng);
        let y = y * y; // chi-squared(1)
        
        let a = mu + (mu * mu * y) / (2.0 * lambda);
        let b = (mu / (2.0 * lambda)) * (4.0 * mu * lambda * y + mu * mu * y * y).sqrt();
        let x = (mu * mu) / (a + b);
        
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

/// Sample from PG(b, c) for float b > 0.
///
/// For b=1, uses the exact sample_pg1 method.
/// For integer b, sums b independent PG(1, c) draws.
/// For float b, approximates via a Gamma distribution with matched mean and variance.
pub fn sample_pg(b: f64, c: f64, rng: &mut StdRng) -> f64 {
    if !c.is_finite() || !b.is_finite() {
        panic!("Polya-Gamma sampler received non-finite parameters (b = {b}, c = {c}); \
                the chain diverged. Check priors/data/scaling.");
    }
    let c_abs = c.abs();

    // For large c, the exact sampler is exponentially inefficient.
    // We fall back to the Gamma approximation.
    if b == 1.0 && c_abs <= 5.0 {
        return sample_pg1(c, rng);
    }
    
    // If b is a SMALL integer, we can sum PG(1, c) exactly. The cap matters: for
    // NB the caller passes b = y + r, and r initialises to exactly 1.0, so integer
    // b = y + 1 arrives on the first sweep -- summing y draws per observation is an
    // effective hang for large counts (a high-count SBC rep burned 16 CPU-hours in
    // one call). For b > 50 the moment-matched Gamma below is CLT-accurate anyway.
    if (b.round() - b).abs() < 1e-9 && b > 0.0 && b <= 50.0 && c_abs <= 5.0 {
        let n = b.round() as u32;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += sample_pg1(c, rng);
        }
        return sum;
    }
    
    // For non-integer float b, use a Gamma approximation
    // E[PG(b, c)] = (b / 2c) * tanh(c/2)
    // V[PG(b, c)] = b / (4 * c^3) * (sinh(c) - c) / cosh^2(c/2)
    // If c is very small, use limits:
    // mean -> b / 4
    // var -> b / 24
    let mean;
    let var;
    let c_abs = c.abs();
    
    if c_abs < 1e-6 {
        mean = b / 4.0;
        var = b / 24.0;
    } else {
        let tanh_half = (c_abs * 0.5).tanh();
        mean = (b / (2.0 * c_abs)) * tanh_half;
        
        // Numerically stable variance computation
        let cosh_half = (c_abs * 0.5).cosh();
        let sinh_c = c_abs.sinh();
        var = (b / (4.0 * c_abs * c_abs * c_abs)) * (sinh_c - c_abs) / (cosh_half * cosh_half);
    }
    
    // Match moments to Gamma(shape, rate)
    let shape = (mean * mean) / var;
    let rate = mean / var;
    
    use rand_distr::Gamma;
    if let Ok(dist) = Gamma::new(shape, 1.0 / rate) {
        dist.sample(rng)
    } else {
        mean // Fallback if shape/rate are invalid
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

// ---------------------------------------------------------------------------
// Validation harness for the PG(1, c) sampler.
//
// Runs at `cargo test` (dev-time only; NOT part of R CMD check). Guards the
// Devroye accept/reject primitive against the two defects that biased it
// historically: the missing small-x branch in `a_coef` (which corrupted every
// truncated-inverse-Gaussian proposal, x < TRUNC) and the factor-of-2 in
// `sample_pg1_zero`. Both are caught here: the moment test pins mean AND
// variance to closed forms, and the KS test compares the sampler's law to the
// exact Polson-Scott-Windle Gamma-series representation.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    // Exact PG(1, c) via the infinite Gamma series (Polson, Scott & Windle 2013,
    // eq. 1): PG(1,c) = (1/(2 pi^2)) sum_{k>=1} g_k / ((k-1/2)^2 + c^2/(4 pi^2)),
    // g_k ~ Exp(1). Truncating the tail at `n_terms` biases the draw downward by
    // O(1/n_terms) (each dropped term has mean < 1/(2 pi^2 (k-1/2)^2)); n_terms
    // = 4000 keeps that below ~1e-5, negligible against the test tolerances.
    fn pg1_series_reference(c: f64, n_terms: usize, rng: &mut StdRng) -> f64 {
        let shift = c * c / (4.0 * PI * PI);
        let mut x = 0.0;
        for k in 1..=n_terms {
            let g: f64 = Exp1.sample(rng);
            let km = k as f64 - 0.5;
            x += g / (km * km + shift);
        }
        x / (2.0 * PI * PI)
    }

    fn analytic_mean(c: f64) -> f64 {
        if c.abs() < 1e-9 { 0.25 } else { (c * 0.5).tanh() / (2.0 * c) }
    }

    // Var[PG(1,c)] = (1/(4 c^3)) (sinh c - c) / cosh^2(c/2); limit 1/24 as c -> 0.
    fn analytic_var(c: f64) -> f64 {
        if c.abs() < 1e-4 {
            1.0 / 24.0
        } else {
            let ch = (c * 0.5).cosh();
            (c.sinh() - c) / (4.0 * c * c * c) / (ch * ch)
        }
    }

    // Two-sample Kolmogorov-Smirnov statistic.
    fn ks_two_sample(a: &mut [f64], b: &mut [f64]) -> f64 {
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let (na, nb) = (a.len() as f64, b.len() as f64);
        let (mut i, mut j) = (0usize, 0usize);
        let mut d: f64 = 0.0;
        while i < a.len() && j < b.len() {
            let v = a[i].min(b[j]);
            while i < a.len() && a[i] <= v { i += 1; }
            while j < b.len() && b[j] <= v { j += 1; }
            d = d.max((i as f64 / na - j as f64 / nb).abs());
        }
        d
    }

    #[test]
    fn pg1_moments_match_analytic() {
        let mut rng = StdRng::seed_from_u64(20_260_706);
        let n = 400_000usize;
        // c = 0 exercises sample_pg1_zero; the rest exercise both proposal
        // branches (truncated-IG for x < TRUNC, truncated-exponential above).
        for &c in &[0.0, 0.3, 0.8, 1.5, 3.0, 5.0] {
            let (mut s, mut s2) = (0.0f64, 0.0f64);
            for _ in 0..n {
                let x = sample_pg1(c, &mut rng);
                s += x;
                s2 += x * x;
            }
            let mean = s / n as f64;
            let var = s2 / n as f64 - mean * mean;
            let (em, ev) = (analytic_mean(c), analytic_var(c));
            assert!(
                (mean - em).abs() < 0.004,
                "PG(1,{c}) mean {mean:.5} vs analytic {em:.5}"
            );
            assert!(
                (var - ev).abs() < 0.05 * ev + 5e-4,
                "PG(1,{c}) var {var:.5} vs analytic {ev:.5}"
            );
        }
    }

    #[test]
    fn pg1_ks_vs_series_reference() {
        let mut rng = StdRng::seed_from_u64(11_235_813);
        let m = 30_000usize;
        // KS critical value at ~1e-3 significance for n = m = 30_000 is
        // 1.95 * sqrt(2/m) ~= 0.016; 0.03 leaves headroom against Monte-Carlo
        // noise while still failing decisively on a mis-specified a_coef
        // (the missing small-x branch distorts the whole x < TRUNC region).
        for &c in &[0.4, 1.0, 2.5] {
            let mut samp: Vec<f64> = (0..m).map(|_| sample_pg1(c, &mut rng)).collect();
            let mut refr: Vec<f64> = (0..m).map(|_| pg1_series_reference(c, 4000, &mut rng)).collect();
            let d = ks_two_sample(&mut samp, &mut refr);
            assert!(d < 0.03, "PG(1,{c}) KS D = {d:.4} vs series reference (threshold 0.03)");
        }
    }

    // A non-finite tilt must fail fast, not hang (the historical bug: the
    // rejection loop skipped every NaN proposal forever).
    #[test]
    #[should_panic(expected = "non-finite")]
    fn pg1_panics_on_nonfinite_tilt() {
        let mut rng = StdRng::seed_from_u64(1);
        let _ = sample_pg1(f64::NAN, &mut rng);
    }
}
