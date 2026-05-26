test_that("G-computation and AIPW work successfully for causal effect estimation", {
  skip_if_not_installed("posterior")
  
  set.seed(42)
  n <- 80
  
  X1 <- rnorm(n)
  
  # 1. Binary treatment, binomial outcome
  Trt_bin <- rbinom(n, 1, plogis(0.3 + 0.8 * X1))
  Y_bin <- rbinom(n, 1, plogis(0.5 + 1.0 * Trt_bin + 0.5 * X1))
  dat_bin <- data.frame(Y = Y_bin, tau = rep(0, n), Trt = Trt_bin, X1 = X1)
  
  fit_bin <- bjlm_model() |>
    propensity(Trt ~ X1, data = dat_bin, family = binomial("logit")) |>
    outcome(Y ~ X1, b0 = ~ 1 + Trt, data = dat_bin, family = binomial("logit")) |>
    compile() |>
    fit(chains = 1L, iter = 60L, warmup = 30L, seed = 123L, verbose = FALSE)

  # A. G-computation ATE
  expect_silent(ate_sum <- fitted(fit_bin, type = "ate", summary = TRUE))
  expect_s3_class(ate_sum, "data.frame")
  expect_named(ate_sum, c(".observation", "fitted_mean", "fitted_Q2.5", "fitted_Q97.5"))
  expect_equal(nrow(ate_sum), 1)

  expect_silent(ate_draws <- fitted(fit_bin, type = "ate", summary = FALSE))
  expect_true(is.matrix(ate_draws))
  expect_equal(dim(ate_draws), c(30, 1))
  expect_equal(colnames(ate_draws), "ATE")

  # B. G-computation RR
  expect_silent(rr_sum <- fitted(fit_bin, type = "rr", summary = TRUE))
  expect_s3_class(rr_sum, "data.frame")
  expect_equal(nrow(rr_sum), 1)

  expect_silent(rr_draws <- fitted(fit_bin, type = "rr", summary = FALSE))
  expect_true(is.matrix(rr_draws))
  expect_equal(dim(rr_draws), c(30, 1))
  expect_equal(colnames(rr_draws), "RR")

  # C. AIPW ATE
  expect_silent(aipw_ate_sum <- fitted(fit_bin, type = "aipw_ate", summary = TRUE))
  expect_s3_class(aipw_ate_sum, "data.frame")
  expect_equal(nrow(aipw_ate_sum), 1)

  expect_silent(aipw_ate_draws <- fitted(fit_bin, type = "aipw_ate", summary = FALSE))
  expect_true(is.matrix(aipw_ate_draws))
  expect_equal(dim(aipw_ate_draws), c(30, 1))
  expect_equal(colnames(aipw_ate_draws), "AIPW_ATE")

  # D. AIPW RR
  expect_silent(aipw_rr_sum <- fitted(fit_bin, type = "aipw_rr", summary = TRUE))
  expect_s3_class(aipw_rr_sum, "data.frame")
  expect_equal(nrow(aipw_rr_sum), 1)

  expect_silent(aipw_rr_draws <- fitted(fit_bin, type = "aipw_rr", summary = FALSE))
  expect_true(is.matrix(aipw_rr_draws))
  expect_equal(dim(aipw_rr_draws), c(30, 1))
  expect_equal(colnames(aipw_rr_draws), "AIPW_RR")
  
  # E. Propensity truncation check
  expect_silent(aipw_trunc <- fitted(fit_bin, type = "aipw_ate", summary = FALSE, truncation = 0.05))
  expect_equal(dim(aipw_trunc), c(30, 1))

  # 2. Gaussian outcome causal effect check
  Y_gauss <- 5.0 + 2.0 * Trt_bin + 1.5 * X1 + rnorm(n)
  dat_gauss <- data.frame(Y = Y_gauss, tau = rep(0, n), Trt = Trt_bin, X1 = X1)
  
  fit_gauss <- bjlm_model() |>
    propensity(Trt ~ X1, data = dat_gauss, family = binomial("logit")) |>
    outcome(Y ~ X1, b0 = ~ 1 + Trt, data = dat_gauss, family = gaussian("identity")) |>
    compile() |>
    fit(chains = 1L, iter = 60L, warmup = 30L, seed = 123L, verbose = FALSE)

  # G-comp ATE & AIPW ATE work for Gaussian outcomes
  expect_silent(fitted(fit_gauss, type = "ate", summary = FALSE))
  expect_silent(fitted(fit_gauss, type = "aipw_ate", summary = FALSE))

  # Risk Ratio should error for Gaussian outcomes
  expect_error(fitted(fit_gauss, type = "rr"), "Risk Ratio is only applicable to binomial outcome models")
  expect_error(fitted(fit_gauss, type = "aipw_rr"), "Risk Ratio is only applicable to binomial outcome models")

  # 3. Model without propensity block error check
  fit_no_prop <- bjlm_model() |>
    outcome(Y ~ X1, b0 = ~ 1 + Trt, data = dat_gauss, family = gaussian("identity")) |>
    compile() |>
    fit(chains = 1L, iter = 60L, warmup = 30L, seed = 123L, verbose = FALSE)
    
  expect_error(fitted(fit_no_prop, type = "ate"), "requires a fitted propensity score model")
  expect_error(fitted(fit_no_prop, type = "aipw_ate"), "requires a fitted propensity score model")
})
