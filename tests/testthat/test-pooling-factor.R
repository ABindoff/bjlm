test_that("pooling_factor reports both random-effect blocks", {
  set.seed(6); ns <- 15; nt <- 6
  subj    <- rep(seq_len(ns), each = nt)
  time    <- rep(seq(0, 10, length.out = nt), ns)
  omega_i <- (5 + rnorm(ns, 0, 1))[subj]
  u_i     <- rnorm(ns, 0, 0.8)[subj]
  d       <- time - omega_i
  y       <- 2 + u_i + 0.2 * d - 0.8 * d * plogis(3 * d) + rnorm(ns * nt, 0, 0.5)
  dat     <- data.frame(y = y, time = time, subj = factor(subj))

  cm <- bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1 + (1 | subj), b1 = ~ 1, deltas = list(~ 1),
            omega = list(~ 1 + (1 | subj)), rho = list(~ 1), data = dat) |>
    compile()
  fit <- suppressMessages(fit(cm, chains = 1L, iter = 80L, warmup = 40L, seed = 6L, verbose = FALSE))

  pf <- suppressMessages(pooling_factor(fit))
  expect_s3_class(pf, "bjlm_pooling_factor")
  expect_true(all(c("intercept", "changepoint") %in% pf$type))
  expect_equal(sum(pf$type == "intercept"),   ns)   # one coordinate per subject
  expect_equal(sum(pf$type == "changepoint"), ns)
  expect_true(all(pf$pi >= 0 & pf$pi <= 1))
  expect_true(all(pf$n_obs == nt))
  expect_true(all(pf$recommendation %in% c("centred (OK)", "borderline", "non-centred")))
})

test_that("pooling_factor closed form matches fibr::prior_fraction (conformance)", {
  skip_if_not_installed("fibr")
  # prior_fraction() arrived in fibr 0.1.x; older builds only ship smoothbp_advisor.
  skip_if_not("prior_fraction" %in% getNamespaceExports("fibr"),
              "installed fibr does not export prior_fraction()")
  set.seed(7); ns <- 12; nt <- 6
  subj <- rep(seq_len(ns), each = nt)
  time <- rep(seq(0, 10, length.out = nt), ns)
  y    <- 2 + rnorm(ns, 0, 0.8)[subj] + rnorm(ns * nt, 0, 0.5)
  dat  <- data.frame(y = y, time = time, subj = factor(subj))

  cm <- bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1 + (1 | subj), b1 = ~ 1, deltas = list(~ 1),
            omega = list(~ 1), rho = list(~ 1), data = dat) |>
    compile()
  fit <- suppressMessages(fit(cm, chains = 1L, iter = 80L, warmup = 40L, seed = 7L, verbose = FALSE))

  pf  <- suppressMessages(pooling_factor(fit))
  # fibr's default method IS the canonical closed form; feed it bjlm's ingredients.
  ref <- fibr::prior_fraction(1 / pf$prior_sd^2, lik_information = pf$lik_info)
  expect_equal(pf$pi, ref$pi, tolerance = 1e-10)
})

test_that("pooling_factor returns NULL with a message when there are no random effects", {
  set.seed(8); n <- 30
  dat <- data.frame(y = rnorm(n), time = rep(0, n))
  cm  <- bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, deltas = list(~ 1),
            omega = list(~ 1), rho = list(~ 1), data = dat) |>
    compile()
  fit <- suppressMessages(fit(cm, chains = 1L, iter = 60L, warmup = 30L, seed = 8L, verbose = FALSE))
  expect_message(res <- pooling_factor(fit), "no group-level")
  expect_null(res)
})
