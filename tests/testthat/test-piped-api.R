test_that("Piped API model specification, validation, compilation, and fitting work successfully", {
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("posterior")

  set.seed(42)
  n <- 50

  # Simple confounded dataset
  X1 <- rnorm(n)
  pi_true <- plogis(0.5 + 0.8 * X1)
  Trt <- rbinom(n, 1, pi_true)
  Y <- 5 + 2.0 * Trt + 1.5 * X1 + rnorm(n)

  dat <- data.frame(
    Y = Y,
    tau = rep(0, n),
    Trt = Trt,
    X1 = X1
  )

  # ---- Spec construction ----
  spec <- bjlm_model()
  expect_s3_class(spec, "bjlm_model")
  expect_null(spec$propensity)
  expect_null(spec$outcome)

  # Add propensity
  spec <- spec |> propensity(Trt ~ X1, data = dat)
  expect_type(spec$propensity, "list")
  expect_equal(spec$propensity$formula, Trt ~ X1)

  # Add outcome (zero-breakpoint speed shortcut)
  spec <- spec |> outcome(Y ~ X1, data = dat)
  expect_true(spec$outcome$zero_breakpoint)

  # Print spec method test
  expect_output(print(spec))

  # ---- Strict Bayesian Cut Validation check ----
  bad_spec <- bjlm_model() |>
    propensity(Trt ~ X1 + Y, data = dat) |> # Violation: outcome Y in propensity!
    outcome(Y ~ X1, data = dat)
  expect_error(compile(bad_spec), "Bayesian Cut violation")

  # ---- Validation of incomplete models ----
  expect_error(compile(bjlm_model() |> propensity(Trt ~ X1, data = dat)), "missing outcome")
  expect_error(compile(bjlm_model() |> outcome(Y ~ X1, data = dat)), "missing propensity")

  # ---- Compilation ----
  compiled <- compile(spec)
  expect_s3_class(compiled, "bjlm_compiled_model")
  expect_true(compiled$zero_breakpoint)
  expect_output(print(compiled))

  # ---- Fitting (Zero-breakpoint speed shortcut) ----
  fit <- compiled |> fit(
    chains = 2L,
    iter = 100L,
    warmup = 50L,
    seed = 123L,
    verbose = FALSE,
    cores = 1L
  )

  expect_s3_class(fit, "bjlm_fit")
  expect_output(print(fit))
  expect_output(summary(fit, model = "both"))

  # Verify b1 has been fixed at exactly 0.0 in posterior
  draws <- posterior::as_draws_matrix(fit$draws)
  expect_true(all(draws[, "b1_(Intercept)"] == 0.0))

  # ---- Fitting (Piecewise change-point model via Piped API) ----
  pw_spec <- bjlm_model() |>
    propensity(Trt ~ X1, data = dat) |>
    outcome(
      formula = Y ~ tau,
      b0 = ~ 1 + Trt + X1,
      b1 = ~ 1,
      data = dat
    )
  expect_false(pw_spec$outcome$zero_breakpoint)

  pw_compiled <- compile(pw_spec)
  expect_false(pw_compiled$zero_breakpoint)

  pw_fit <- pw_compiled |> fit(
    chains = 2L,
    iter = 100L,
    warmup = 50L,
    seed = 123L,
    verbose = FALSE,
    cores = 1L
  )
  expect_s3_class(pw_fit, "bjlm_fit")

  # ---- tab_bjlm formatting test ----
  res_kable <- tab_bjlm(fit, pw_fit, labels = c("Zero-BP Model", "Piecewise Model"))
  # Even if gt is not installed, it falls back to kable
  if (requireNamespace("gt", quietly = TRUE)) {
    expect_s3_class(res_kable, "gt_tbl")
  } else {
    expect_s3_class(res_kable, "knitr_kable")
  }
})
