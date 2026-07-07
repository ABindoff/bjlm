test_that("prior_spike_slab() validates and constructs", {
  s <- prior_spike_slab()
  expect_s3_class(s, "smoothbp_spike_slab")
  expect_equal(s$family, "spike_slab")
  expect_equal(s$pi, 0.5)
  expect_false(s$learn_pi)

  s2 <- prior_spike_slab(pi = 0.2, slab = prior_normal(0, 5), learn_pi = TRUE, a = 2, b = 3)
  expect_true(s2$learn_pi)
  expect_equal(s2$a, 2); expect_equal(s2$b, 3)
  expect_equal(s2$slab$sd, 5)

  expect_error(prior_spike_slab(pi = 1.5), "pi")
  expect_error(prior_spike_slab(pi = 0),   "pi")
  expect_error(prior_spike_slab(slab = prior_gamma(1, 1)))   # slab must be normal
  expect_error(prior_spike_slab(learn_pi = "yes"))
})

test_that("print.smoothbp_spike_slab shows the specification", {
  expect_output(print(prior_spike_slab(pi = 0.3)), "SpikeSlab\\(pi = 0.3")
  expect_output(print(prior_spike_slab(learn_pi = TRUE, a = 2, b = 3)), "Beta\\(2, 3\\)")
})

test_that("spike-and-slab selects real modifiers over a null one; pip() is well-formed", {
  set.seed(11); ns <- 40; nt <- 10
  subj <- rep(seq_len(ns), each = nt)
  time <- rep(seq(0, 10, length.out = nt), ns)
  Trt  <- rbinom(ns, 1, 0.5)[subj]
  Xn   <- rnorm(ns)[subj]                         # null covariate
  d    <- time - 5
  delta_i <- -1.0 + 1.5 * Trt                     # slope change depends on Trt, NOT Xn
  y    <- 2 + 0.2 * d + delta_i * d * plogis(4 * d) + rnorm(ns * nt, 0, 0.4)
  dat  <- data.frame(y = y, time = time, Trt = Trt, Xn = Xn, subj = factor(subj))

  cm <- bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, deltas = list(~ 1 + Trt + Xn),
            omega = list(~ 1), rho = list(~ 1), data = dat) |>
    compile()
  fit <- suppressMessages(fit(cm, spike = prior_spike_slab(pi = 0.5),
                              chains = 2L, iter = 600L, warmup = 300L, seed = 11L, verbose = FALSE))

  p <- pip(fit)
  expect_s3_class(p, "smoothbp_pip")
  expect_named(p, c("parameter", "pip", "lower", "upper"))
  expect_true(all(p$pip   >= 0 & p$pip   <= 1))
  expect_true(all(p$lower >= 0 & p$upper <= 1 & p$lower <= p$upper))

  gp <- function(nm) p$pip[p$parameter == nm]
  # real change-point and real modifier are selected; the null covariate is not
  expect_gt(gp("delta1_(Intercept)"), 0.5)
  expect_gt(gp("delta1_Trt"),         0.5)
  expect_lt(gp("delta1_Xn"),          0.5)

  skip_if_not_installed("ggplot2")
  expect_s3_class(plot(p), "ggplot")
})

test_that("pip() errors on a fit without spike-and-slab", {
  set.seed(12); n <- 30
  dat <- data.frame(y = rnorm(n), time = rep(0, n), X = rnorm(n))
  cm  <- bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1 + X, data = dat) |>
    compile()
  fit <- suppressMessages(fit(cm, chains = 1L, iter = 60L, warmup = 30L, seed = 12L, verbose = FALSE))
  expect_error(pip(fit), "spike")
})

test_that("learn_pi = TRUE samples pi_ss; FALSE omits it", {
  set.seed(13); ns <- 20; nt <- 8
  subj <- rep(seq_len(ns), each = nt); time <- rep(seq(0, 10, length.out = nt), ns)
  Trt  <- rbinom(ns, 1, 0.5)[subj]
  d    <- time - 5
  y    <- 2 + 0.2 * d + (-0.8 + 1.0 * Trt) * d * plogis(3 * d) + rnorm(ns * nt, 0, 0.5)
  dat  <- data.frame(y = y, time = time, Trt = Trt, subj = factor(subj))
  cm <- bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, deltas = list(~ 1 + Trt),
            omega = list(~ 1), rho = list(~ 1), data = dat) |>
    compile()

  fit_learn <- suppressMessages(fit(cm, spike = prior_spike_slab(learn_pi = TRUE),
                                    chains = 1L, iter = 200L, warmup = 100L, seed = 13L, verbose = FALSE))
  vars_l <- posterior::variables(fit_learn$draws)
  expect_true("pi_ss" %in% vars_l)
  expect_gt(stats::sd(as.numeric(posterior::as_draws_matrix(fit_learn$draws)[, "pi_ss"])), 0)

  fit_fixed <- suppressMessages(fit(cm, spike = prior_spike_slab(learn_pi = FALSE),
                                    chains = 1L, iter = 200L, warmup = 100L, seed = 13L, verbose = FALSE))
  expect_false("pi_ss" %in% posterior::variables(fit_fixed$draws))
})
