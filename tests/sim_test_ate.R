# ============================================================
# Simulation 1: Basic ATE recovery (cross-sectional)
# ============================================================
# Package is loaded via devtools::load_all() before sourcing this script

set.seed(42)
n <- 300

# --- Generate confounded data ---
X1 <- rnorm(n)
X2 <- rnorm(n)
pi_true <- plogis(0.5 + 0.8 * X1 - 0.3 * X2)
Trt <- rbinom(n, 1, pi_true)

# True ATE = 2.0
Y <- 5 + 2.0 * Trt + 1.5 * X1 + 0.5 * X2 + rnorm(n)

dat <- data.frame(
  Y = Y,
  tau = rep(0, n),
  Trt = Trt,
  X1 = X1,
  X2 = X2
)

cat("=== Naive estimate (biased) ===\n")
naive <- lm(Y ~ Trt, data = dat)
cat(sprintf("  Naive ATE: %.3f (SE: %.3f)\n",
  coef(naive)["Trt"], summary(naive)$coefficients["Trt", "Std. Error"]))

cat("\n=== OLS with all confounders (oracle, correctly specified) ===\n")
ols <- lm(Y ~ Trt + X1 + X2, data = dat)
cat(sprintf("  OLS ATE: %.3f (SE: %.3f)\n",
  coef(ols)["Trt"], summary(ols)$coefficients["Trt", "Std. Error"]))

cat(sprintf("  True ATE: 2.000\n\n"))

cat("=== Fitting bipw model (4 chains x 4000 iter) ===\n")
fit <- bipw(
  outcome    = Y ~ tau,
  b0         = ~ 1 + Trt + X1 + X2,
  b1         = ~ 1,
  propensity = Trt ~ X1 + X2,
  weights    = "stabilised_ate",
  max_weight = 20,
  data       = dat,
  propensity_prior_sd = 2.5,
  chains     = 4L,
  iter       = 4000L,
  warmup     = 2000L,
  seed       = 123L,
  verbose    = TRUE,
  cores      = 1L
)

cat("\n=== bipw model ===\n")
print(fit)

cat("\n=== Outcome model ===\n")
summary(fit, model = "outcome")

cat("\n=== Propensity model ===\n")
summary(fit, model = "propensity")

cat("\n=== Causal effect (ATE for Trt) ===\n")
ce <- causal_effect(fit, param = "b0_Trt")

cat("\n=== Weight diagnostics ===\n")
weight_diagnostics(fit)

cat("\n\nDone.\n")
