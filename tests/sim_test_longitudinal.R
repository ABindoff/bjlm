# ============================================================
# Simulation 2: Longitudinal piecewise trajectory with 
# confounding, random intercepts, and one breakpoint
#
# DGP:
#   100 subjects × 10 timepoints (tau = 0..9)
#   X1 ~ N(0, 1)   confounder (baseline cognition)
#   X2 ~ N(0, 1)   confounder (education)
#   Trt | X ~ Bernoulli(logistic(0.3 + 0.6*X1 - 0.4*X2))
#   u_i ~ N(0, sigma_u=2)   random intercept
#
#   Piecewise model (1 breakpoint at omega=5):
#     b0_intercept = 50, b0_X1 = 2
#     b1 = -0.3             (initial slope, decline/year)
#     delta_intercept = -0.2 (additional decline for controls)
#     delta_Trt = 0.4       (CAUSAL EFFECT: treatment slows decline)
#     rho = 5               (sharp transition)
#     sigma = 1.5
#     sigma_u = 2.0
#
#   Before breakpoint: slope = -0.3 for all
#   After breakpoint:
#     Controls: slope = -0.3 + (-0.2) = -0.5 (faster decline)
#     Treated:  slope = -0.3 + (-0.2 + 0.4) = -0.1 (slowed decline)
#
#   True causal effect: delta_Trt = 0.4
# ============================================================
# Package is loaded via devtools::load_all() before sourcing

sigmoid <- function(x) ifelse(x >= 0, 1/(1+exp(-x)), exp(x)/(1+exp(x)))

set.seed(314)
n_subj <- 100
n_time <- 10
tau_vals <- 0:(n_time - 1)

# --- Subject-level ---
X1 <- rnorm(n_subj)
X2 <- rnorm(n_subj)
pi_true <- plogis(0.3 + 0.6 * X1 - 0.4 * X2)
Trt <- rbinom(n_subj, 1, pi_true)
u <- rnorm(n_subj, 0, 2.0)  # random intercepts

cat(sprintf("Treatment prevalence: %.1f%% (%d/%d)\n",
  100 * mean(Trt), sum(Trt), n_subj))

# --- Generate longitudinal data ---
true_b0_int <- 50
true_b0_X1 <- 2
true_b1 <- -0.3
true_delta_int <- -0.2
true_delta_trt <- 0.4   # CAUSAL EFFECT
true_omega <- 5
true_rho <- 5
true_sigma <- 1.5
true_sigma_u <- 2.0

dat_long <- do.call(rbind, lapply(1:n_subj, function(i) {
  data.frame(
    subject = rep(i, n_time),
    tau = tau_vals,
    X1 = rep(X1[i], n_time),
    X2 = rep(X2[i], n_time),
    Trt = rep(Trt[i], n_time),
    u = rep(u[i], n_time)
  )
}))

# Compute piecewise mean
dat_long$di <- dat_long$tau - true_omega
dat_long$si <- sigmoid(dat_long$di * true_rho)
dat_long$delta_i <- true_delta_int + true_delta_trt * dat_long$Trt
dat_long$mu <- true_b0_int + true_b0_X1 * dat_long$X1 + dat_long$u +
  true_b1 * dat_long$di +
  dat_long$delta_i * dat_long$di * dat_long$si
dat_long$score <- dat_long$mu + rnorm(nrow(dat_long), 0, true_sigma)

# Clean up helper columns
dat_long <- dat_long[, c("subject", "tau", "score", "X1", "X2", "Trt")]

cat(sprintf("Data: %d obs, %d subjects, %d timepoints\n",
  nrow(dat_long), n_subj, n_time))

# --- Naive analysis (no IPW, no confounder adjustment) ---
cat("\n=== Naive analysis ===\n")
# Simple OLS ignoring confounders and clustering
# Won't recover delta_Trt correctly due to confounding
cat("  (Naive analysis not straightforward for piecewise models, skipping)\n")

# --- bipw ---
cat("\n=== Fitting bipw model ===\n")
cat("  (4 chains x 4000 iter, 1 breakpoint, random intercepts)\n\n")

fit <- bipw(
  outcome    = score ~ tau,
  b0         = ~ 1 + X1 + (1 | subject),
  b1         = ~ 1,
  deltas     = list(~ 1 + Trt),
  omega      = list(~ 1),
  rho        = list(~ 1),
  propensity = Trt ~ X1 + X2,
  weights    = "stabilised_ate",
  max_weight = 20,
  data       = dat_long,
  propensity_prior_sd = 2.5,
  outcome_priors = list(
    b0_mean = c(0, 0), b0_sd = c(10, 5),
    b0_lb = c(-Inf, -Inf), b0_ub = c(Inf, Inf),
    b1_mean = c(0), b1_sd = c(2),
    b1_lb = c(-Inf), b1_ub = c(Inf),
    delta_mean = list(c(0, 0)), delta_sd = list(c(2, 2)),
    delta_lb = list(c(-Inf, -Inf)), delta_ub = list(c(Inf, Inf)),
    om_mean = list(c(5)), om_sd = list(c(3)),
    om_lb = list(c(0)), om_ub = list(c(9)),
    rho_mean = list(c(3)), rho_sd = list(c(3)),
    rho_lb = list(c(0.1)), rho_ub = list(c(20)),
    sigma_shape = 1, sigma_scale = 1,
    sigma_u_shape = 1, sigma_u_scale = 1
  ),
  chains     = 4L,
  iter       = 4000L,
  warmup     = 2000L,
  seed       = 314L,
  verbose    = TRUE,
  cores      = 1L,
  step_om    = 0.02,
  step_rho   = 0.02,
  target_accept = 0.8
)

cat("\n=== Model summary ===\n")
print(fit)

cat("\n=== Outcome model ===\n")
summary(fit, model = "outcome")

cat("\n=== Propensity model ===\n")
summary(fit, model = "propensity")

cat("\n=== Causal effect (delta_Trt: treatment effect on slope change) ===\n")
ce <- causal_effect(fit, param = "delta1_Trt")

cat("\n=== True parameters ===\n")
cat(sprintf("  b0_intercept: %.1f\n", true_b0_int))
cat(sprintf("  b0_X1:        %.1f\n", true_b0_X1))
cat(sprintf("  b1:           %.1f\n", true_b1))
cat(sprintf("  delta_int:    %.1f\n", true_delta_int))
cat(sprintf("  delta_Trt:    %.1f  <-- CAUSAL EFFECT\n", true_delta_trt))
cat(sprintf("  omega:        %.1f\n", true_omega))
cat(sprintf("  rho:          %.1f\n", true_rho))
cat(sprintf("  sigma:        %.1f\n", true_sigma))
cat(sprintf("  sigma_u:      %.1f\n", true_sigma_u))

cat("\n=== Weight diagnostics ===\n")
weight_diagnostics(fit)

cat("\n\nDone.\n")
