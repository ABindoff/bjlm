# SBC certificate: latent-GP change-point model

Simulation-Based Calibration (Talts et al. 2018) for the latent Gaussian-process
confounder model with a smooth change-point and a random intercept. Harness:
`data-raw/sbc_latent_gp.R`.

## Model (deliberately clean)

Outcome (Gaussian), one breakpoint, per-subject latent GP confounder measured by a
noisy covariate, random intercept, and **no treatment / propensity / IPW** so the
fitted target is a genuine Bayesian posterior rather than a weighted pseudo-posterior:

```
y_ij  = b0 + u_j + b1*(tau_ij - omega)
        + delta*(tau_ij - omega)*sigmoid(rho*(tau_ij - omega))
        + b0_gp * f_j(tau_ij) + N(0, sigma^2)
X_ij  = f_j(tau_ij) + N(0, sigma_x^2)          # noisy measurement of the GP
f_j   ~ GP(0, SE kernel; alpha, rho_gp)
u_j   ~ N(0, sigma_u)
```

Eligible for the collapsed (Rao-Blackwellised) GP hyperparameter sampler, so this
certifies that path.

## Priors (drawn == fitted)

| parameter | prior |
|---|---|
| b0, b0_gp | N(0, 1) |
| b1 | N(0, 0.5) |
| delta | N(0, 0.8) |
| omega | N(5, 1.5) truncated [0.5, 9.5] |
| rho | N(4, 1.5) truncated [1, 10] |
| sigma | sigma^2 ~ InvGamma(5, 2) |
| sigma_u | half-Cauchy(0, 0.5) on the SD |
| gp_alpha, gp_rho | lognormal(0, 1) |
| gp_sigma_x | lognormal(-1, 1) |

## Protocol

Draw theta from the prior, simulate, fit (2 chains), rank each true value among
ESS-thinned posterior draws (fractional rank), test rank uniformity with a 10-bin
chi-square GOF. b0 is ranked as `b0 + mean(u)` (the identified level; b0 aliases with
the random-intercept mean). Bonferroni threshold for 11 parameters at 0.05 is
p > 0.0045.

Two runs:
- **Primary**: 128 reps, 1500 iter / 750 warmup, thinning by `ess_bulk`.
- **Confirmatory**: 64 reps, 3000 iter / 1500 warmup, thinning by
  `min(ess_bulk, ess_tail)` (stricter). Its role: separate genuine miscalibration
  from ESS-thinning artifacts (residual autocorrelation inflates the extreme ranks
  and mimics over-confidence).

## Result: 10 / 11 calibrated; gp_sigma_x flags

Zero fit failures across 128 + 64 = 192 simulated datasets.

| parameter | primary p | confirmatory p | verdict |
|---|---|---|---|
| omega (change-point) | 0.066 | 0.28 | PASS |
| rho | 0.42 | 0.48 | PASS |
| delta | 0.084 | 0.71 | PASS (U-shape was thinning) |
| b1 | 0.049 | 0.48 | PASS |
| b0 (as b0+mean u) | 0.69 | 0.68 | PASS |
| sigma | 0.35 | 0.40 | PASS (U-shape was thinning) |
| sigma_u | 0.34 | 0.24 | PASS |
| b0_gp | 0.35 | 0.88 | PASS (U-shape was thinning) |
| gp_alpha | 0.096 | 0.037 | PASS (> Bonferroni) |
| gp_rho | 0.28 | 0.22 | PASS |
| **gp_sigma_x** | 0.032 | **0.0010** | **FLAG** |

The mild U-shapes seen in the primary run (delta, sigma, b0_gp) **flattened** under
stricter thinning, confirming they were residual-autocorrelation artifacts, not
sampler defects. The change-point (omega, rho) and the GP marginal-SD / lengthscale
(alpha, rho_gp) are calibrated.

**gp_sigma_x (GP observation-noise SD) is genuinely miscalibrated**: under stricter
thinning it *sharpened* rather than flattened (p = 0.001, fails Bonferroni), with a
right-skewed rank histogram (truth in the upper tail) -> the posterior is
systematically biased **low**. This is consistent with the point-recovery runs, where
sigma_x landed ~0.35 against a true 0.40 repeatedly. Leading hypothesis: the latent
field absorbs a little of the measurement noise, deflating the estimated sigma_x.
This is a real, mild finding for follow-up; it does not affect the change-point or the
other GP hyperparameters.

## Bottom line

The latent-GP change-point sampler is **calibrated for the change-point and the GP
signal (alpha, rho_gp), with a documented mild low-bias in the GP observation-noise
SD (sigma_x)** to be resolved. SBC did its job: it separated three thinning artifacts
from one real defect.
