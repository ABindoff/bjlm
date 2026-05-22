# Diagnostic: inspect bjlm output structure and raw draws

set.seed(42)
n <- 300
X1 <- rnorm(n)
X2 <- rnorm(n)
Trt <- rbinom(n, 1, plogis(0.5 + 0.8 * X1 - 0.3 * X2))
Y <- 5 + 2.0 * Trt + 1.5 * X1 + 0.5 * X2 + rnorm(n)

dat <- data.frame(Y = Y, tau = rep(0, n), Trt = Trt, X1 = X1, X2 = X2)

cat("=== bjlm short run (200 iter, 1 chain) ===\n")
fit <- bjlm(
  outcome    = Y ~ tau,
  b0         = ~ 1 + Trt + X1 + X2,
  b1         = ~ 1,
  propensity = Trt ~ X1 + X2,
  weights    = "stabilised_ate",
  max_weight = 20,
  data       = dat,
  chains     = 1L,
  iter       = 200L,
  warmup     = 100L,
  seed       = 42L,
  verbose    = TRUE,
  cores      = 1L
)

# Check dimensions
cat("\n--- Draws structure ---\n")
if (inherits(fit$draws, "draws_array")) {
  cat(sprintf("  class: %s\n", paste(class(fit$draws), collapse = ", ")))
  cat(sprintf("  dims: %s\n", paste(dim(fit$draws), collapse = " x ")))
  cat(sprintf("  variables: %s\n", paste(posterior::variables(fit$draws), collapse = ", ")))
}

# Check first 10 draws of each param
cat("\n--- First 10 draws from chain 1 ---\n")
draws_mat <- posterior::as_draws_matrix(fit$draws)
vars <- posterior::variables(draws_mat)
for (v in vars) {
  vals <- as.numeric(draws_mat[1:min(10, nrow(draws_mat)), v])
  cat(sprintf("  %-20s: %s\n", v, paste(round(vals, 4), collapse = ", ")))
}

# Check that sigma values are all positive
cat("\n--- Sigma check ---\n")
sigma_vals <- as.numeric(draws_mat[, "sigma"])
cat(sprintf("  sigma min: %.4f, max: %.4f, all > 0: %s\n",
  min(sigma_vals), max(sigma_vals), all(sigma_vals > 0)))

sigma_u_vals <- as.numeric(draws_mat[, "sigma_u"])
cat(sprintf("  sigma_u min: %.4f, max: %.4f, all > 0: %s\n",
  min(sigma_u_vals), max(sigma_u_vals), all(sigma_u_vals > 0)))

# Mean weight trace
w_vals <- as.numeric(draws_mat[, "mean_weight"])
cat(sprintf("\n--- Mean weight ---\n"))
cat(sprintf("  min: %.4f, max: %.4f, mean: %.4f\n",
  min(w_vals), max(w_vals), mean(w_vals)))

# Also check the correctly-specified OLS for reference
cat("\n=== OLS with all confounders (correctly specified) ===\n")
ols <- lm(Y ~ Trt + X1 + X2, data = dat)
cat(sprintf("  OLS ATE: %.4f (SE: %.4f)\n",
  coef(ols)["Trt"], summary(ols)$coefficients["Trt", "Std. Error"]))

cat("\nDone.\n")
