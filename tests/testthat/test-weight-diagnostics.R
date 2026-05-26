test_that("weight_diagnostics works for binary and continuous propensity models", {
  skip_if_not_installed("posterior")

  set.seed(42)
  n <- 80

  # Confounded data
  X1 <- rnorm(n)
  
  # 1. Binary treatment
  Trt_bin <- rbinom(n, 1, plogis(0.3 + 0.8 * X1))
  Y_bin <- 5.0 + 2.0 * Trt_bin + 1.5 * X1 + rnorm(n, 0, 1.0)
  dat_bin <- data.frame(Y = Y_bin, tau = rep(0, n), Trt = Trt_bin, X1 = X1)

  fit_bin <- bjlm_model() |>
    propensity(Trt ~ X1, data = dat_bin, family = binomial("logit")) |>
    outcome(Y ~ X1, b0 = ~ 1 + Trt, data = dat_bin) |>
    compile() |>
    fit(chains = 1L, iter = 60L, warmup = 30L, seed = 123L, verbose = FALSE)

  # Check that weight_diagnostics prints the expected output and returns list
  expect_output(
    res_bin <- weight_diagnostics(fit_bin),
    "=== Weight Diagnostics ==="
  )
  expect_type(res_bin, "list")
  expect_named(res_bin, c("weights", "ess", "trimmed_prop", "max_untrimmed"))
  expect_length(res_bin$weights, n)
  expect_length(res_bin$ess, 30) # S = iter - warmup = 60 - 30 = 30 draws
  expect_length(res_bin$trimmed_prop, 30)
  expect_length(res_bin$max_untrimmed, 30)

  # 2. Continuous treatment
  Trt_cont <- rnorm(n, 0.3 + 0.8 * X1, 0.5)
  Y_cont <- 5.0 + 2.0 * Trt_cont + 1.5 * X1 + rnorm(n, 0, 1.0)
  dat_cont <- data.frame(Y = Y_cont, tau = rep(0, n), Trt = Trt_cont, X1 = X1)

  fit_cont <- bjlm_model() |>
    propensity(Trt ~ X1, data = dat_cont, family = gaussian("identity")) |>
    outcome(Y ~ X1, b0 = ~ 1 + Trt, data = dat_cont) |>
    compile() |>
    fit(chains = 1L, iter = 60L, warmup = 30L, seed = 123L, verbose = FALSE)

  # Check that weight_diagnostics prints the expected output and returns list
  expect_output(
    res_cont <- weight_diagnostics(fit_cont),
    "Continuous \\(Generalized Propensity Score\\)"
  )
  expect_type(res_cont, "list")
  expect_named(res_cont, c("weights", "ess", "trimmed_prop", "max_untrimmed"))
  expect_length(res_cont$weights, n)
  expect_length(res_cont$ess, 30)
  expect_length(res_cont$trimmed_prop, 30)
  expect_length(res_cont$max_untrimmed, 30)
})
