# regimes(): phase v0 — known-state level switching (desugars to a b0 factor).

test_that("regime emission/prior constructors build the right objects", {
  expect_s3_class(exact(), "regime_obs_model")
  expect_equal(exact()$type, "exact")
  expect_s3_class(confusion(diag = 8, offdiag = 1), "regime_obs_model")
  expect_equal(confusion()$type, "confusion")
  expect_error(confusion(diag = 0))
  expect_s3_class(regime_priors(), "regime_priors")
})

test_that("regimes() validates its arguments and appends a block", {
  m <- bjlm_model()
  expect_error(regimes(list(), name = "r", data = data.frame(s = 1), n_states = 2, obs_state = "s"),
               "bjlm_model")
  expect_error(regimes(m, name = "r", n_states = 2, obs_state = "s"), "data")
  d <- data.frame(s = c("A", "B"))
  expect_error(regimes(m, name = "r", data = d, n_states = 2), "obs_state")
  expect_error(regimes(m, name = "r", data = d, n_states = 1, obs_state = "s"), "n_states")
  expect_error(regimes(m, name = "r", data = d, n_states = 2, obs_state = "s", obs_model = 1), "obs_model")

  m2 <- regimes(m, name = "regime", data = d, n_states = 2, states = c("A", "B"),
                obs_state = "s", obs_model = exact(), ref_state = "A")
  expect_length(m2$regimes, 1L)
  expect_equal(m2$regimes[[1]]$name, "regime")
})

.regime_data <- function(seed = 1, ns = 25L, nt = 6L) {
  set.seed(seed)
  subj <- rep(seq_len(ns), each = nt)
  time <- rep(seq(0, 5, length.out = nt), ns)
  state <- sample(c("A", "B", "C"), ns * nt, replace = TRUE)
  lev <- c(A = 0, B = 1.0, C = -0.5)[state]
  y <- 2 + 0.1 * time + lev + rnorm(ns * nt, 0, 0.4)
  data.frame(y = y, time = time, state = state, subj = factor(subj))
}

test_that("compile() desugars a known-state regime into a b0 factor", {
  dat <- .regime_data()
  cm <- bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, data = dat) |>
    regimes(name = "regime", data = dat, n_states = 3, states = c("A", "B", "C"),
            time_var = "time", subject = "subj", obs_state = "state",
            obs_model = exact(), ref_state = "A") |>
    compile()
  expect_true("state" %in% all.vars(cm$b0_formula))
  # the observed-state column is now a factor with the reference level first
  expect_true(is.factor(cm$model$outcome$data$state))
  expect_equal(levels(cm$model$outcome$data$state)[1], "A")
})

test_that("v0 capability gates fire", {
  dat <- .regime_data()
  base <- function() bjlm_model() |> outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, data = dat)
  # misclassification not yet active
  expect_error(
    base() |> regimes(name = "r", data = dat, n_states = 3, states = c("A","B","C"),
                      obs_state = "state", obs_model = confusion(), ref_state = "A") |> compile(),
    "exact")
  # covariate-dependent transitions recorded but warn
  expect_warning(
    base() |> regimes(name = "r", data = dat, n_states = 3, states = c("A","B","C"),
                      obs_state = "state", obs_model = exact(), transition = ~ time,
                      ref_state = "A") |> compile(),
    "transition")
  # missing observed-state column
  expect_error(
    base() |> regimes(name = "r", data = dat, n_states = 3, obs_state = "nope",
                      obs_model = exact()) |> compile(),
    "not found")
  # bad reference state
  expect_error(
    base() |> regimes(name = "r", data = dat, n_states = 3, states = c("A","B","C"),
                      obs_state = "state", obs_model = exact(), ref_state = "Z") |> compile(),
    "ref_state")
})

test_that("a known-state regime fit recovers the level offsets", {
  skip_on_cran()
  dat <- .regime_data(seed = 2, ns = 30L)
  fit <- suppressMessages(
    bjlm_model() |>
      outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, data = dat) |>
      regimes(name = "regime", data = dat, n_states = 3, states = c("A","B","C"),
              obs_state = "state", obs_model = exact(), ref_state = "A") |>
      compile() |>
      fit(chains = 2L, iter = 800L, warmup = 400L, seed = 2L, verbose = FALSE))
  sm <- posterior::summarise_draws(fit$draws, "mean")
  gm <- function(v) sm$mean[sm$variable == v]
  expect_true("b0_stateB" %in% sm$variable && "b0_stateC" %in% sm$variable)
  expect_gt(gm("b0_stateB"), 0.4)     # truth +1.0
  expect_lt(gm("b0_stateC"), -0.1)    # truth -0.5
})

test_that("SBC certifies regime level recovery via custom functionals", {
  skip_on_cran()
  dat <- .regime_data(seed = 3, ns = 20L)
  cm <- bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, data = dat) |>
    regimes(name = "regime", data = dat, n_states = 3, states = c("A","B","C"),
            obs_state = "state", obs_model = exact(), ref_state = "A") |>
    compile()
  res <- suppressMessages(sbc(cm, reps = 4L, iter = 200L, warmup = 100L, chains = 1L,
                              seed = 5L, rhat_threshold = 5, verbose = FALSE,
                              functionals = list(
                                stateB = function(p) p[["b0_stateB"]],
                                stateC = function(p) p[["b0_stateC"]])))
  expect_true(all(c("stateB", "stateC") %in% colnames(res$ranks)))
  expect_true(all(res$ranks >= 0 & res$ranks <= 1, na.rm = TRUE))
})
