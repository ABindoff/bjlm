## Submission Summary
This is the initial CRAN release of the `bjlm` package. `bjlm` fits Bayesian
joint longitudinal models that combine a propensity-weighted (marginal
structural) component with a piecewise change-point outcome model, using
Pólya-Gamma augmented Gibbs sampling and a modular cut posterior. The MCMC
engine is written in Rust via the `extendr` framework.

## Test environments
* Local Windows 11, R 4.6.0
* (Recommended: run R-hub / win-builder checks before actual submission)

## R CMD check results
0 errors | 1 warning | 0 notes

* Warning: checking Rust compilation ... WARNING Downloads Rust crates
  * This is expected as the package uses the `extendr` framework. `Cargo.lock`
    is included for reproducible builds. If required by CRAN, we can vendor all
    Rust dependencies in a subsequent submission.

## Notes for the submission
* This is a new release.
* Parameter recovery and posterior consistency are validated against `brms`
  (Stan) and raw `rstan`; that comparison lives in the `model_comparisons`
  vignette, which ships with its code chunks set to `eval = FALSE` because it
  depends on the heavy `brms`/`rstan` Suggests and compiles a Stan model. All
  vignettes build without those optional packages.
