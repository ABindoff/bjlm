test_that("fit() validates MCMC controls (clear errors, not cryptic ones)", {
  set.seed(1); n <- 30
  dat <- data.frame(Y = rnorm(n), tau = rep(0, n), X1 = rnorm(n))
  cm <- bjlm_model() |>
    outcome(Y ~ X1, b0 = ~ 1, data = dat, family = gaussian("identity")) |>
    compile()

  expect_error(fit(cm, chains = 0L,  iter = 40L),               "chains")
  expect_error(fit(cm, iter = 1L),                              "iter")
  expect_error(fit(cm, iter = 40L, warmup = 40L),               "warmup") # warmup >= iter
  expect_error(fit(cm, iter = 40L, warmup = -5L),               "warmup")
  expect_error(fit(cm, iter = 40L, cores = 0L),                 "cores")
})

test_that("pp_check draws replicates on the correct scale per family", {
  skip_if_not_installed("bayesplot")
  set.seed(2); n <- 60
  X1 <- rnorm(n)

  # --- binomial: replicates must be 0/1, not continuous Normal draws ---
  Yb <- rbinom(n, 1, plogis(0.2 + 0.7 * X1))
  db <- data.frame(Y = Yb, tau = rep(0, n), X1 = X1)
  cb <- bjlm_model() |>
    outcome(Y ~ X1, b0 = ~ 1, data = db, family = binomial("logit")) |>
    compile()
  fb <- suppressMessages(fit(cb, chains = 1L, iter = 60L, warmup = 30L, seed = 1L, verbose = FALSE))
  pb <- pp_check(fb, n_draws = 10)
  expect_s3_class(pb, "ggplot")
  yrep_b <- pb$data$value
  expect_true(all(yrep_b %in% c(0, 1)))   # Bernoulli, not Gaussian

  # --- negative binomial: replicates must be non-negative integer counts ---
  Yn <- rnbinom(n, size = 5, mu = exp(1.2 + 0.3 * X1))
  dn <- data.frame(Y = Yn, tau = rep(0, n), X1 = X1)
  cn <- bjlm_model() |>
    outcome(Y ~ X1, b0 = ~ 1, data = dn, family = "negative_binomial") |>
    compile()
  fn <- suppressMessages(fit(cn, chains = 1L, iter = 60L, warmup = 30L, seed = 1L, verbose = FALSE))
  pn <- pp_check(fn, n_draws = 10)
  expect_s3_class(pn, "ggplot")
  yrep_n <- pn$data$value
  expect_true(all(yrep_n >= 0 & yrep_n == round(yrep_n)))  # counts, not Normal
})

test_that("a random change-point fit returns the learned sigma_re_om", {
  set.seed(3); ns <- 12; nt <- 8
  subj    <- rep(seq_len(ns), each = nt)
  time    <- rep(seq(0, 10, length.out = nt), ns)
  omega_i <- (5 + rnorm(ns, 0, 1))[subj]
  d       <- time - omega_i
  y       <- 2 + 0.2 * d - 0.8 * d * plogis(3 * d) + rnorm(ns * nt, 0, 0.5)
  dat     <- data.frame(y = y, time = time, subj = factor(subj))

  cm <- bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1, b1 = ~ 1,
            deltas = list(~ 1), omega = list(~ 1 + (1 | subj)), rho = list(~ 1),
            data = dat) |>
    compile()
  fit <- suppressMessages(fit(cm, chains = 1L, iter = 60L, warmup = 30L, seed = 3L, verbose = FALSE))

  vars <- posterior::variables(fit$draws)
  expect_true("sigma_re_om1" %in% vars)           # was silently dropped before
  s <- as.numeric(posterior::as_draws_matrix(fit$draws)[, "sigma_re_om1"])
  expect_true(all(s > 0))
  expect_gt(stats::sd(s), 0)                       # actually sampled, not a constant
})

test_that("a fixed change-point fit does NOT gain a sigma_re_om column", {
  set.seed(4); n <- 40
  dat <- data.frame(y = rnorm(n), time = rep(0, n))
  cm <- bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1, b1 = ~ 1,
            deltas = list(~ 1), omega = list(~ 1), rho = list(~ 1), data = dat) |>
    compile()
  fit <- suppressMessages(fit(cm, chains = 1L, iter = 60L, warmup = 30L, seed = 4L, verbose = FALSE))
  expect_false(any(grepl("^sigma_re_om", posterior::variables(fit$draws))))
})
