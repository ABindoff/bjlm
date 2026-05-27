# Tests for population() block and population_predict()
# Existing coverage (test-piped-api.R lines 254-400):
#   builder errors, compile/fit pass-through, type="response" summary+draws,
#   type="ate" summary, zero-bp tau ignored, no-spec error,
#   piecewise trajectory 3-row df, summary=FALSE matrix column names.
# This file covers the remaining cases.

# ---- shared helpers ----------------------------------------------------------

.make_binary_fit <- function(n = 60, seed = 42, chains = 1L,
                              iter = 80L, warmup = 40L) {
  set.seed(seed)
  X1  <- rnorm(n)
  Trt <- rbinom(n, 1, plogis(0.3 + 0.6 * X1))
  Y   <- rbinom(n, 1, plogis(0.5 + 1.2 * Trt + 0.4 * X1))
  dat <- data.frame(Y = Y, tau = rep(0, n), Trt = Trt, X1 = X1)

  census <- data.frame(X1 = c(-0.5, 0.5), Trt = c(0, 1), N_pop = c(600L, 400L))

  bjlm_model() |>
    propensity(Trt ~ X1, data = dat, family = binomial("logit")) |>
    outcome(Y ~ X1, b0 = ~ 1 + Trt, data = dat, family = binomial("logit")) |>
    population(cells = census, weight = "N_pop") |>
    compile() |>
    fit(chains = chains, iter = iter, warmup = warmup,
        seed = seed, verbose = FALSE)
}

.make_gaussian_fit <- function(n = 60, seed = 7, chains = 1L,
                                iter = 80L, warmup = 40L) {
  set.seed(seed)
  X1  <- rnorm(n)
  Trt <- rbinom(n, 1, plogis(0.3 + 0.5 * X1))
  Y   <- 4 + 1.5 * Trt + 0.8 * X1 + rnorm(n)
  dat <- data.frame(Y = Y, tau = rep(0, n), Trt = Trt, X1 = X1)

  census <- data.frame(X1 = c(-0.5, 0.5), Trt = c(0, 1), N_pop = c(600L, 400L))

  bjlm_model() |>
    propensity(Trt ~ X1, data = dat) |>
    outcome(Y ~ X1, data = dat) |>
    population(cells = census, weight = "N_pop") |>
    compile() |>
    fit(chains = chains, iter = iter, warmup = warmup,
        seed = seed, verbose = FALSE)
}

# ---- 1. population() builder ------------------------------------------------

test_that("population() builder validates inputs thoroughly", {
  skip_if_not_installed("posterior")

  base <- bjlm_model() |> outcome(Y ~ X1, data = data.frame(Y = 1, tau = 0, X1 = 0))

  census <- data.frame(X1 = c(-0.5, 0.5), N_pop = c(600L, 400L))

  # Not a bjlm_model
  expect_error(population(list(), cells = census), "bjlm_model")

  # Empty cells
  expect_error(
    population(base, cells = data.frame(X1 = numeric(0))),
    "at least one row"
  )

  # Weight: not a string
  expect_error(population(base, cells = census, weight = 1L), "single string")

  # Weight: negative value
  census_neg <- data.frame(X1 = c(-0.5, 0.5), N_pop = c(-1L, 400L))
  expect_error(population(base, cells = census_neg, weight = "N_pop"), "non-negative")

  # Weight: all zero
  census_zero <- data.frame(X1 = c(-0.5, 0.5), N_pop = c(0L, 0L))
  expect_error(population(base, cells = census_zero, weight = "N_pop"), "positive value")

  # Weight: NA values
  census_na <- data.frame(X1 = c(-0.5, 0.5), N_pop = c(NA_real_, 400))
  expect_error(population(base, cells = census_na, weight = "N_pop"), "non-negative")

  # Strata: not a formula
  expect_error(population(base, cells = census, strata = "X1"), "one-sided formula")

  # Strata: variable missing from cells
  expect_error(
    population(base, cells = census, strata = ~ X1 + age),
    "Strata variable"
  )

  # at: not a list
  expect_error(population(base, cells = census, at = "tau=0"), "named list")

  # Valid call should succeed silently
  expect_silent(population(base, cells = census, weight = "N_pop", strata = ~ X1))
})

# ---- 2. compile()-time warnings ---------------------------------------------

test_that("compile() warns when census cells are missing model covariates", {
  skip_if_not_installed("posterior")

  set.seed(1)
  n <- 30
  dat <- data.frame(
    Y = rnorm(n), tau = rep(0, n),
    X1 = rnorm(n), X2 = rnorm(n),
    Trt = rbinom(n, 1, 0.5)
  )
  # Census intentionally omits X2
  census <- data.frame(X1 = c(-0.5, 0.5), Trt = c(0, 1))

  spec <- bjlm_model() |>
    propensity(Trt ~ X1, data = dat) |>
    outcome(Y ~ X1, b0 = ~ 1 + Trt + X1 + X2, data = dat) |>
    population(cells = census, at = list())

  expect_warning(compile(spec), "missing fixed-effect covariate")
})

test_that("compile() warns when piecewise model has no tau in cells and no at$tau", {
  skip_if_not_installed("posterior")

  set.seed(2)
  n_subj <- 8; n_obs <- 4
  dat <- data.frame(
    subject = rep(1:n_subj, each = n_obs),
    tau     = rep(seq(0, 3, length.out = n_obs), n_subj),
    X1      = rep(rnorm(n_subj), each = n_obs),
    Trt     = rep(rbinom(n_subj, 1, 0.5), each = n_obs),
    Y       = rnorm(n_subj * n_obs)
  )
  # Census has no tau column and no at$tau supplied
  census <- data.frame(X1 = c(-0.5, 0.5), Trt = c(0, 1))

  spec <- bjlm_model() |>
    propensity(Trt ~ X1, data = dat[!duplicated(dat$subject), ]) |>
    outcome(Y ~ tau, b0 = ~ 1 + Trt + X1 + (1 | subject), b1 = ~ 1, data = dat) |>
    population(cells = census)

  expect_warning(compile(spec), "time variable")
})

# ---- 3. population_predict: type = "rr" (binomial) --------------------------

test_that("population_predict type='rr' works for binomial outcome", {
  skip_if_not_installed("posterior")

  fit_bin <- .make_binary_fit()

  rr_sum <- population_predict(fit_bin, type = "rr", seed = 1L)
  expect_s3_class(rr_sum, "data.frame")
  expect_named(rr_sum, c("pop_mean", "pop_Q2.5", "pop_Q97.5"))
  expect_true(rr_sum$pop_mean > 0)
  expect_true(rr_sum$pop_Q2.5 < rr_sum$pop_mean)
  expect_true(rr_sum$pop_mean < rr_sum$pop_Q97.5)

  rr_draws <- population_predict(fit_bin, type = "rr", summary = FALSE, seed = 1L)
  n_draws <- nrow(posterior::as_draws_matrix(fit_bin$draws))
  expect_type(rr_draws, "double")
  expect_length(rr_draws, n_draws)
  expect_true(all(rr_draws > 0))
})

test_that("population_predict type='rr' errors for non-binomial outcome", {
  skip_if_not_installed("posterior")

  fit_gauss <- .make_gaussian_fit()
  expect_error(
    population_predict(fit_gauss, type = "rr"),
    "binomial"
  )
})

# ---- 4. population_predict: propensity errors --------------------------------

test_that("population_predict type='ate'/'rr' errors without propensity model", {
  skip_if_not_installed("posterior")

  set.seed(3)
  n <- 40
  dat <- data.frame(Y = rnorm(n), tau = rep(0, n), X1 = rnorm(n))
  census <- data.frame(X1 = c(-0.5, 0.5), N_pop = c(500L, 500L))

  fit_no_prop <- bjlm_model() |>
    outcome(Y ~ X1, data = dat) |>
    population(cells = census, weight = "N_pop") |>
    compile() |>
    fit(chains = 1L, iter = 40L, warmup = 20L, seed = 3L, verbose = FALSE)

  expect_error(population_predict(fit_no_prop, type = "ate"), "propensity")
  expect_error(population_predict(fit_no_prop, type = "rr"), "propensity|binomial",
               perl = TRUE)
})

test_that("population_predict type='ate' errors when trt_var not in census cells", {
  skip_if_not_installed("posterior")

  set.seed(4)
  n <- 40
  X1  <- rnorm(n)
  Trt <- rbinom(n, 1, plogis(0.3 * X1))
  Y   <- 3 + Trt + 0.5 * X1 + rnorm(n)
  dat <- data.frame(Y = Y, tau = rep(0, n), Trt = Trt, X1 = X1)

  # Census deliberately omits Trt
  census_no_trt <- data.frame(X1 = c(-0.5, 0.5), N_pop = c(500L, 500L))

  fit_res <- bjlm_model() |>
    propensity(Trt ~ X1, data = dat) |>
    outcome(Y ~ X1, data = dat) |>
    population(cells = census_no_trt, weight = "N_pop") |>
    compile() |>
    fit(chains = 1L, iter = 40L, warmup = 20L, seed = 4L, verbose = FALSE)

  expect_error(population_predict(fit_res, type = "ate"), "Treatment variable")
})

# ---- 5. population argument override at call time ---------------------------

test_that("population_predict accepts a population argument overriding the stored spec", {
  skip_if_not_installed("posterior")

  fit_gauss <- .make_gaussian_fit()

  # Supply an alternative census at call time
  alt_census <- data.frame(X1 = c(-1, 0, 1), Trt = c(0, 0, 0), N_pop = c(100L, 200L, 100L))
  alt_pop <- list(cells = alt_census, weight = "N_pop", strata = NULL, at = list())

  result <- population_predict(fit_gauss, population = alt_pop,
                               type = "response", seed = 10L)
  expect_s3_class(result, "data.frame")
  expect_named(result, c("pop_mean", "pop_Q2.5", "pop_Q97.5"))
})

test_that("population_predict errors when no spec in fit and none supplied", {
  skip_if_not_installed("posterior")

  set.seed(5)
  n <- 40
  dat <- data.frame(Y = rnorm(n), tau = rep(0, n), X1 = rnorm(n))

  bare_fit <- bjlm_model() |>
    outcome(Y ~ X1, data = dat) |>
    compile() |>
    fit(chains = 1L, iter = 40L, warmup = 20L, seed = 5L, verbose = FALSE)

  expect_error(population_predict(bare_fit), "No population specification")
})

# ---- 6. at override at call time --------------------------------------------

test_that("at argument at call time overrides stored at", {
  skip_if_not_installed("posterior")

  set.seed(6)
  n_subj <- 8; n_obs <- 5
  dat <- data.frame(
    subject = rep(1:n_subj, each = n_obs),
    tau     = rep(seq(0, 4, length.out = n_obs), n_subj),
    X1      = rep(rnorm(n_subj), each = n_obs),
    Trt     = rep(rbinom(n_subj, 1, 0.5), each = n_obs),
    Y       = rnorm(n_subj * n_obs)
  )
  dat_subj <- dat[!duplicated(dat$subject), ]
  census <- data.frame(X1 = c(-0.5, 0.5), Trt = c(0, 0), N_pop = c(500L, 500L))

  fit_res <- suppressWarnings(
    bjlm_model() |>
      propensity(Trt ~ X1, data = dat_subj) |>
      outcome(Y ~ tau, b0 = ~ 1 + X1 + (1 | subject), b1 = ~ 1, data = dat) |>
      population(cells = census, weight = "N_pop", at = list(tau = c(0, 2))) |>
      compile() |>
      fit(chains = 1L, iter = 60L, warmup = 30L, seed = 6L, verbose = FALSE)
  )

  # Override stored at$tau (c(0,2)) with a different grid
  traj <- population_predict(fit_res, at = list(tau = c(1, 3)), type = "response", seed = 1L)
  expect_equal(nrow(traj), 2L)
  expect_equal(traj$tau, c(1, 3))
})

# ---- 7. Weight effects -------------------------------------------------------

test_that("different census weights produce different population means", {
  skip_if_not_installed("posterior")

  fit_gauss <- .make_gaussian_fit()

  # Replace stored cells with two different weight vectors
  cells_base <- fit_gauss$population$cells

  pop_equal <- list(
    cells  = transform(cells_base, N_pop = c(1L, 1L)),
    weight = "N_pop", strata = NULL, at = list()
  )
  pop_skewed <- list(
    cells  = transform(cells_base, N_pop = c(990L, 10L)),
    weight = "N_pop", strata = NULL, at = list()
  )

  r_equal   <- population_predict(fit_gauss, population = pop_equal,
                                  type = "response", seed = 42L)
  r_skewed  <- population_predict(fit_gauss, population = pop_skewed,
                                  type = "response", seed = 42L)

  expect_false(isTRUE(all.equal(r_equal$pop_mean, r_skewed$pop_mean, tolerance = 0.01)))
})

test_that("uniform weights and equal N_pop weights give the same pop_mean", {
  skip_if_not_installed("posterior")

  fit_gauss <- .make_gaussian_fit()
  cells_base <- fit_gauss$population$cells

  pop_uniform <- list(cells = cells_base[, setdiff(names(cells_base), "N_pop")],
                      weight = NULL, strata = NULL, at = list())
  pop_equal   <- list(cells = transform(cells_base, N_pop = c(1L, 1L)),
                      weight = "N_pop", strata = NULL, at = list())

  r_uniform <- population_predict(fit_gauss, population = pop_uniform,
                                  type = "response", seed = 99L)
  r_equal   <- population_predict(fit_gauss, population = pop_equal,
                                  type = "response", seed = 99L)

  expect_equal(r_uniform$pop_mean, r_equal$pop_mean, tolerance = 1e-10)
})

# ---- 8. Reproducibility with seed -------------------------------------------

test_that("population_predict is reproducible with the same seed", {
  skip_if_not_installed("posterior")

  fit_gauss <- .make_gaussian_fit()

  r1 <- population_predict(fit_gauss, type = "response", summary = FALSE, seed = 77L)
  r2 <- population_predict(fit_gauss, type = "response", summary = FALSE, seed = 77L)
  expect_equal(r1, r2)
})

test_that("population_predict differs across different seeds (RE model)", {
  skip_if_not_installed("posterior")

  set.seed(8)
  n_subj <- 10; n_obs <- 6
  dat <- data.frame(
    subject = rep(1:n_subj, each = n_obs),
    tau     = rep(seq(0, 5, length.out = n_obs), n_subj),
    X1      = rep(rnorm(n_subj), each = n_obs),
    Trt     = rep(rbinom(n_subj, 1, 0.5), each = n_obs),
    Y       = rnorm(n_subj * n_obs, sd = 2)
  )
  census <- data.frame(X1 = c(-0.5, 0.5), Trt = c(0, 1), N_pop = c(500L, 500L))

  fit_re <- suppressWarnings(
    bjlm_model() |>
      propensity(Trt ~ X1, data = dat[!duplicated(dat$subject), ]) |>
      outcome(Y ~ tau, b0 = ~ 1 + X1 + (1 | subject), b1 = ~ 1, data = dat) |>
      population(cells = census, weight = "N_pop", at = list(tau = 0)) |>
      compile() |>
      fit(chains = 1L, iter = 60L, warmup = 30L, seed = 8L, verbose = FALSE)
  )

  r_s1 <- population_predict(fit_re, type = "response", summary = FALSE, seed = 1L)
  r_s2 <- population_predict(fit_re, type = "response", summary = FALSE, seed = 2L)

  # Different seeds → different RE draws → different results for RE model
  expect_false(isTRUE(all.equal(r_s1, r_s2, tolerance = 1e-12)))
})

# ---- 9. ate trajectory -------------------------------------------------------

test_that("population_predict type='ate' works over a tau grid", {
  skip_if_not_installed("posterior")

  set.seed(9)
  n_subj <- 10; n_obs <- 5
  dat <- data.frame(
    subject = rep(1:n_subj, each = n_obs),
    tau     = rep(seq(0, 4, length.out = n_obs), n_subj),
    X1      = rep(rnorm(n_subj), each = n_obs),
    Trt     = rep(rbinom(n_subj, 1, 0.5), each = n_obs),
    Y       = rnorm(n_subj * n_obs)
  )
  census <- data.frame(X1 = c(-0.5, 0.5), Trt = c(0, 1), N_pop = c(500L, 500L))

  fit_res <- suppressWarnings(
    bjlm_model() |>
      propensity(Trt ~ X1, data = dat[!duplicated(dat$subject), ]) |>
      outcome(Y ~ tau, b0 = ~ 1 + Trt + X1 + (1 | subject), b1 = ~ 1, data = dat) |>
      population(cells = census, weight = "N_pop", at = list(tau = c(0, 2, 4))) |>
      compile() |>
      fit(chains = 1L, iter = 60L, warmup = 30L, seed = 9L, verbose = FALSE)
  )

  traj_ate <- population_predict(fit_res, type = "ate", seed = 1L)
  expect_s3_class(traj_ate, "data.frame")
  expect_equal(nrow(traj_ate), 3L)
  expect_named(traj_ate, c("tau", "pop_mean", "pop_Q2.5", "pop_Q97.5"))
  expect_equal(traj_ate$tau, c(0, 2, 4))

  # summary=FALSE → matrix
  mat_ate <- population_predict(fit_res, type = "ate", summary = FALSE, seed = 1L)
  expect_true(is.matrix(mat_ate))
  expect_equal(ncol(mat_ate), 3L)
  expect_equal(colnames(mat_ate), c("tau=0", "tau=2", "tau=4"))
})
