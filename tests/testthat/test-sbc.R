# SBC: fast structural tests (must run on CRAN) plus an opt-in uniformity
# certification gated behind BJLM_SBC_CERT (too slow for routine runs).

.sbc_test_model <- function(ns = 15L, nt = 6L, seed = 1L) {
  set.seed(seed)
  subj <- rep(seq_len(ns), each = nt)
  time <- rep(seq(0, 6, length.out = nt), ns)
  Trt  <- rbinom(ns, 1, 0.5)[subj]
  dat  <- data.frame(y = rnorm(ns * nt), time = time, Trt = Trt, subj = factor(subj))
  bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, deltas = list(~ 1 + Trt),
            omega = list(~ 1), rho = list(~ 1), data = dat) |>
    compile()
}

.sbc_test_priors <- function() {
  bjlm_priors(outcome = smoothbp_priors(
    b0 = prior_normal(0, 3), b1 = prior_normal(0, 1), deltas = prior_normal(0, 1),
    omega = prior_normal(3, 1, lb = 0), rho = prior_normal(4, 1, lb = 0),
    sigma = prior_invgamma(3, 2), sigma_u = prior_halfcauchy(1)))
}

test_that(".sbc_draw_prior names are a subset of the fitted draw columns", {
  cm <- .sbc_test_model()
  pri <- .sbc_test_priors()
  gp_info <- bjlm:::.sbc_gp_info(cm)
  set.seed(42)
  draw <- bjlm:::.sbc_draw_prior(cm, pri$outcome, NULL, gp_info)
  fit <- suppressMessages(fit(cm, priors = pri, chains = 1L, iter = 120L,
                              warmup = 60L, seed = 1L, verbose = FALSE))
  cols <- posterior::variables(fit$draws)
  # Every drawn truth aligns to a real posterior column (guards naming drift).
  expect_true(all(names(draw$theta) %in% cols))
  expect_true(all(c("b0_(Intercept)", "b1_(Intercept)", "delta1_Trt",
                    "omega1_(Intercept)", "sigma") %in% names(draw$theta)))
})

test_that(".sbc_simulate replaces y and preserves the design", {
  cm <- .sbc_test_model()
  pri <- .sbc_test_priors()
  gp_info <- bjlm:::.sbc_gp_info(cm)
  set.seed(7)
  draw <- bjlm:::.sbc_draw_prior(cm, pri$outcome, NULL, gp_info)
  orig <- cm$model$outcome$data
  sim <- bjlm:::.sbc_simulate(cm, draw, gp_info)
  expect_s3_class(sim, "data.frame")
  expect_equal(nrow(sim), nrow(orig))
  expect_false(isTRUE(all.equal(sim$y, orig$y)))
  expect_true(all(is.finite(sim$y)))
  expect_identical(sim$time, orig$time)      # covariates untouched
})

test_that("sbc() end-to-end returns well-formed ranks and prints", {
  cm <- .sbc_test_model()
  pri <- .sbc_test_priors()
  # rhat_threshold high so the structural check is not gated by convergence.
  res <- suppressMessages(sbc(cm, priors = pri, reps = 4L, iter = 200L,
                              warmup = 100L, chains = 1L, seed = 3L,
                              rhat_threshold = 5, verbose = FALSE))
  expect_s3_class(res, "bjlm_sbc")
  expect_true(all(res$ranks >= 0 & res$ranks <= 1, na.rm = TRUE))
  expect_true(all(c("omega1_(Intercept)", "rho1_(Intercept)", "sigma") %in% colnames(res$ranks)))
  expect_equal(res$n_kept + res$n_discard, 4L)
  expect_output(print(res), "Simulation-based calibration")
  skip_if_not_installed("ggplot2")
  expect_s3_class(plot(res), "ggplot")
})

test_that("spike-and-slab excludes selected coefficients from rank targets", {
  cm <- .sbc_test_model()
  pri <- .sbc_test_priors()
  res <- suppressMessages(sbc(cm, priors = pri,
                              spike = prior_spike_slab(pi = 0.5, slab = prior_normal(0, 2)),
                              reps = 3L, iter = 200L, warmup = 100L, chains = 1L,
                              seed = 4L, rhat_threshold = 5, verbose = FALSE))
  expect_false(any(grepl("^delta", colnames(res$ranks))))   # spike-eligible
  expect_false(any(grepl("^b1", colnames(res$ranks))))
  expect_true("omega1_(Intercept)" %in% colnames(res$ranks))
})

test_that("the rank-shape classifier reads the standard SBC histograms", {
  flagged <- 0.001; ok <- 0.5
  expect_equal(bjlm:::.sbc_classify(runif(50), ok)$shape, "uniform")
  expect_equal(bjlm:::.sbc_classify(c(rep(0.02, 10), rep(0.98, 10)), flagged)$shape, "U")   # both tails
  expect_equal(bjlm:::.sbc_classify(rep(0.5, 20), flagged)$shape, "n")                        # middle
  expect_equal(bjlm:::.sbc_classify(rep(0.02, 20), flagged)$shape, "left")                    # biased high
  expect_equal(bjlm:::.sbc_classify(rep(0.98, 20), flagged)$shape, "right")                   # biased low
})

test_that(".sbc_verdicts summarises a rank matrix", {
  set.seed(1)
  ranks <- cbind(a = runif(30), b = runif(30))
  v <- bjlm:::.sbc_verdicts(ranks)
  expect_named(v, c("parameter", "n_used", "mean_rank", "p_value", "calibrated", "verdict"))
  expect_equal(nrow(v), 2L)
  expect_true(all(v$n_used == 30L))
})

test_that("sbc() errors clearly on misuse", {
  cm <- .sbc_test_model()
  # a fit that dropped its compiled model
  fake <- structure(list(compiled_model = NULL), class = "bjlm_fit")
  expect_error(sbc(fake), "compiled model")
  # prior-sensitivity needs a latent GP
  expect_error(sbc_prior_sensitivity(cm, grid = c(0.5, 1)), "latent GP")
})

test_that("SBC uniformity certification (opt-in, slow)", {
  skip_on_cran()
  skip_if(!nzchar(Sys.getenv("BJLM_SBC_CERT")), "set BJLM_SBC_CERT=1 to run the SBC cert")
  cm <- .sbc_test_model(ns = 25L)
  pri <- .sbc_test_priors()
  res <- suppressMessages(sbc(cm, priors = pri, reps = 60L, iter = 1500L,
                              warmup = 750L, chains = 2L, seed = 100L,
                              rhat_threshold = 1.1, verbose = FALSE))
  v <- res$verdicts
  # Every default quantity should pass the Bonferroni-adjusted uniformity test.
  expect_true(all(v$calibrated %in% TRUE),
              info = paste(v$parameter, round(v$p_value, 3), collapse = "; "))
})
