test_that("bridge_sampler / bayes_factor error clearly on bjlm_fit (invalid target)", {
  skip_if_not_installed("bridgesampling")

  set.seed(1)
  n <- 40
  dat <- data.frame(Y = rnorm(n), tau = rep(0, n),
                    Trt = rbinom(n, 1, 0.5), X1 = rnorm(n))
  cm <- bjlm_model() |>
    propensity(Trt ~ X1, data = dat) |>
    outcome(Y ~ X1, b0 = ~ 1 + Trt, data = dat, family = gaussian("identity")) |>
    compile()
  fit <- suppressMessages(
    fit(cm, chains = 1L, iter = 60L, warmup = 30L, seed = 1L, verbose = FALSE))

  # The outcome model is an IPW-weighted MSM (pseudo-posterior): no valid
  # marginal likelihood / Bayes factor. Must error, not return a plausible number.
  expect_error(bridgesampling::bridge_sampler(fit), "marginal structural")
  expect_error(bridgesampling::bayes_factor(fit, fit), "marginal structural")
})

test_that("recovery_plot matches bjlm indexed change-point params (delta1/omega1/rho1)", {
  skip_if_not_installed("ggplot2")

  dat <- simulate_smoothbp(n_subj = 15, n_obs = 6, seed = 7)
  cm <- bjlm_model() |>
    outcome(y ~ tau, b0 = ~ 1 + (1 | subject),
            deltas = list(~ 1), omega = list(~ 1), rho = list(~ 1),
            data = dat) |>
    compile()
  fit <- suppressMessages(
    fit(cm, chains = 1L, iter = 80L, warmup = 40L, seed = 7L, verbose = FALSE))

  p <- suppressWarnings(recovery_plot(fit, dat))
  present <- as.character(p$data$nm)
  # Before the fix these silently dropped for a bjlm_fit (b2_/omega_/rho_ names).
  expect_true(all(c("b2", "omega", "rho") %in% present))
})
