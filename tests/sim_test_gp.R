# ==============================================================================
# Simulation Study: Latent Gaussian Process Parameter Recovery & Coverage
# ==============================================================================
# Sourced via Rscript tests/sim_test_gp.R after devtools::load_all()

library(dplyr)
library(posterior)

set.seed(123)
cat("=== Simulating Joint Model Data with Latent GP Confounder ===\n")

# Simulate data using our new simulate_bjlm() generator
# 25 subjects, 8 time points each = 200 total observations
true_b0_int <- 0.5
true_b0_trt <- 2.0
true_b0_gp <- 2.0
true_b1 <- 0.2
true_delta_int <- -0.2
true_delta_trt <- 0.4
true_omega <- 5.0
true_rho <- 5.0
true_sigma <- 0.5

true_gp_alpha <- 1.0
true_gp_rho <- 3.0
true_gp_sigma_x <- 0.4
true_sigma_u <- 0.5

dat <- simulate_bjlm(
  n_subj = 25L,
  n_obs = 8L,
  b0 = true_b0_int,
  b0_trt = true_b0_trt,
  b1 = true_b1,
  omegas = c(true_omega),
  rhos = c(true_rho),
  deltas_int = c(true_delta_int),
  deltas_trt = c(true_delta_trt),
  sigma = true_sigma,
  sigma_u = true_sigma_u,
  gp_confounder = TRUE,
  gp_alpha = true_gp_alpha,
  gp_rho = true_gp_rho,
  gp_sigma_x = true_gp_sigma_x,
  b0_gp = true_b0_gp,
  trt_gp = 0.5,
  seed = 123L
)

cat(sprintf("Generated %d observations for %d subjects.\n", nrow(dat), length(unique(dat$subject))))

cat("\n=== Fitting Joint GP BJLM Model ===\n")
fit <- bjlm_model() |>
  propensity(Trt ~ 1, data = dat[!duplicated(dat$subject), ]) |>
  outcome(
    y ~ tau,
    b0 = ~ 1 + Trt + X_obs + (1 | subject),
    b1 = ~ 1,
    deltas = list(~ 1 + Trt),
    omega = list(~ 1),
    rho = list(~ 1),
    data = dat
  ) |>
  latent_gp(
    name = "X_obs",
    data = dat,
    time_var = "tau",
    obs_var = "X_obs",
    subject = "subject",
    time_out_var = "tau",
    time_trt_var = "tau"
  ) |>
  compile() |>
  fit(
    priors = bjlm_priors(
      outcome = smoothbp_priors(
        b0 = prior_normal(0, 10),
        b1 = prior_normal(0, 2),
        deltas = prior_normal(0, 2),
        omega = prior_normal(5, 3, lb = 0, ub = 10),
        rho = prior_normal(3, 3, lb = 0.1, ub = 20),
        sigma = prior_invgamma(1, 1),
        sigma_u = prior_halfcauchy(1)
      ),
      propensity = prior_normal(0, 2.5)
    ),
    chains = 2L,
    iter = 3000L,
    warmup = 1500L,
    seed = 123L,
    cores = 2L,
    verbose = TRUE,
    step_om = 0.02,
    step_rho = 0.02,
    target_accept = 0.8
  )

cat("\n=== Summarizing Posterior Draws & Checking Parameter Recovery ===\n")
draws_summary <- posterior::summarise_draws(
  fit$draws,
  mean,
  sd,
  ~ posterior::quantile2(.x, probs = c(0.025, 0.975)),
  rhat,
  ess_bulk
)

# Map parameter name in fit to the true simulated value
validation_params <- list(
  "b0_(Intercept)" = true_b0_int,
  "b0_Trt" = true_b0_trt,
  "b0_X_obs" = true_b0_gp,
  "b1_(Intercept)" = true_b1,
  "delta1_(Intercept)" = true_delta_int,
  "delta1_Trt" = true_delta_trt,
  "omega1_(Intercept)" = true_omega,
  "rho1_(Intercept)" = true_rho,
  "sigma" = true_sigma,
  "sigma_u" = true_sigma_u,
  "X_obs_alpha" = true_gp_alpha,
  "X_obs_rho" = true_gp_rho,
  "X_obs_sigma_x" = true_gp_sigma_x
)

cat(sprintf("\n%-20s | %-8s | %-8s | %-18s | %-8s | %-8s\n", 
            "Parameter", "Truth", "Mean", "95% Credible Int", "Rhat", "ESS"))
cat(paste(rep("-", 80), collapse = ""), "\n")

n_covered <- 0
for (p_name in names(validation_params)) {
  truth <- validation_params[[p_name]]
  
  if (!p_name %in% draws_summary$variable) {
    cat(sprintf("%-20s | %-8.3f | %-8s | %-18s | %-8s | %-8s (Not Found)\n", 
                p_name, truth, "-", "-", "-", "-"))
    next
  }
  
  row <- draws_summary[draws_summary$variable == p_name, ]
  p_mean <- row$mean
  p_lo <- row$q2.5
  p_hi <- row$q97.5
  p_rhat <- row$rhat
  p_ess <- row$ess_bulk
  
  covered <- (truth >= p_lo && truth <= p_hi)
  if (covered) n_covered <- n_covered + 1
  
  covered_str <- if (covered) "✓" else "✗"
  ci_str <- sprintf("[%.3f, %.3f] %s", p_lo, p_hi, covered_str)
  
  cat(sprintf("%-20s | %-8.3f | %-8.3f | %-18s | %-8.3f | %-8.0f\n", 
              p_name, truth, p_mean, ci_str, p_rhat, p_ess))
}

coverage_rate <- n_covered / length(validation_params)
cat(paste(rep("-", 80), collapse = ""), "\n")
cat(sprintf("Overall Credible Interval Coverage: %d/%d (%.1f%%)\n", 
            n_covered, length(validation_params), 100 * coverage_rate))

if (coverage_rate >= 0.8) {
  cat("\nSUCCESS: Parameter recovery and nominal credible interval coverage validated successfully!\n")
} else {
  cat("\nWARNING: Nominal coverage rate is below 80%. Check posterior chain diagnostics/mixing.\n")
}

cat("\n=== Testing recovery_plot() on the fitted Joint GP Model ===\n")
p_rec <- recovery_plot(fit, dat)
if (inherits(p_rec, "ggplot")) {
  cat("SUCCESS: recovery_plot() generated a valid ggplot object successfully!\n")
} else {
  cat("FAILURE: recovery_plot() failed to return a ggplot object.\n")
}

cat("\nDone.\n")
