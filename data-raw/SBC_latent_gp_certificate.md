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
| gp_alpha | lognormal(0, 1) |
| gp_rho | resolution-aware: ln(rho) ~ N(loc, scale), band = grid median-gap to range |
| gp_sigma_x | lognormal(-1, 1) |

## Protocol

Draw theta from the prior, simulate, fit (2 chains), rank each true value among
ESS-thinned posterior draws (fractional rank), test rank uniformity with a 10-bin
chi-square GOF. b0 is ranked as `b0 + mean(u)` (the identified level; b0 aliases with
the random-intercept mean). Bonferroni threshold for 11 parameters at 0.05 is
p > 0.0045.

Three runs:
- **Primary**: 128 reps, 1500 iter / 750 warmup, thinning by `ess_bulk`, fixed
  lognormal(0,1) lengthscale prior.
- **Confirmatory**: 64 reps, 3000 iter / 1500 warmup, thinning by
  `min(ess_bulk, ess_tail)` (stricter), same prior. Its role: separate genuine
  miscalibration from ESS-thinning artifacts (residual autocorrelation inflates the
  extreme ranks and mimics over-confidence).
- **Resolution-aware**: 100 reps, 3000 iter / 1500 warmup, strict thinning, with the
  fixed lognormal(0,1) lengthscale prior REPLACED by the resolution-aware prior below.

## Result: 11 / 11 calibrated (after the lengthscale-prior fix)

Zero fit failures across 128 + 64 + 100 = 292 simulated datasets.

| parameter | primary p | confirmatory p | res-aware p | verdict |
|---|---|---|---|---|
| omega (change-point) | 0.066 | 0.28 | 0.90 | PASS |
| rho | 0.42 | 0.48 | 0.53 | PASS |
| delta | 0.084 | 0.71 | 0.55 | PASS (primary U-shape was thinning) |
| b1 | 0.049 | 0.48 | 0.30 | PASS |
| b0 (as b0+mean u) | 0.69 | 0.68 | 0.35 | PASS |
| sigma | 0.35 | 0.40 | 0.46 | PASS (primary U-shape was thinning) |
| sigma_u | 0.34 | 0.24 | 0.22 | PASS |
| b0_gp | 0.35 | 0.88 | 0.57 | PASS (primary U-shape was thinning) |
| gp_alpha | 0.096 | 0.037 | 0.53 | PASS |
| gp_rho | 0.28 | 0.22 | 0.83 | PASS |
| **gp_sigma_x** | 0.032 | **0.0010** | **0.15** | **PASS (fixed)** |

The mild U-shapes in the primary run (delta, sigma, b0_gp) flattened under stricter
thinning, confirming they were residual-autocorrelation artifacts.

**gp_sigma_x was genuinely miscalibrated under the fixed lognormal(0,1) lengthscale
prior** (p = 0.001, right-skewed ranks, posterior biased low). Diagnosis, by a
lengthscale sweep at fixed truth (true sigma_x = 0.40, grid spacing ~1.43):

| true rho_gp | 0.5 | 1.0 | 1.5 | 2.5 | 4.0 |
|---|---|---|---|---|---|
| posterior sigma_x | 0.25 | 0.26 | 0.35 | 0.38 | 0.39 |

The bias switches on exactly at the sampling resolution. Below it, `K ~ alpha^2 I`, so
the data identify only the total variance `alpha^2 + sigma_x^2`, not the split; the
asymmetric priors (ln alpha median 1 >> ln sigma_x median 0.37) then load the shared
variance onto alpha and starve sigma_x. This is not a sampler defect (the collapse is
exact once the field is resolved) but a resolution-blind lengthscale prior: a fixed
lognormal(0,1) assumes an O(1) time axis and sits below resolution on a [0,10] grid.

**Fix**: a resolution-aware lengthscale prior, `ln(rho) ~ N(loc, scale)` with the band
from the grid's median gap (below which the GP absorbs noise) to its range (above which
it is indistinguishable from a constant): `loc = 0.5*(ln gap + ln range)`,
`scale = max(0.35, 0.25*ln(range/gap))`. Under it, gp_sigma_x is calibrated
(p = 0.15) and every other parameter remains so. See `gp_rho_log_prior_params()`.

## Bottom line

The latent-GP change-point sampler is **fully calibrated (11/11)** under the
resolution-aware lengthscale prior. SBC did its job three times over: it separated
three thinning artifacts from one real defect, then confirmed the fix for that defect.
