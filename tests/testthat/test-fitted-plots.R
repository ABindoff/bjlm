test_that("fitted S3 method, diagnostics, and plotting functions work as expected", {
  skip_if_not_installed("posterior")
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("loo")
  skip_if_not_installed("bayesplot")

  set.seed(123)
  dat <- data.frame(
    subject_id = rep(1:5, each = 3),
    tau = rep(0:2, times = 5),
    Trt = rep(rbinom(5, 1, 0.5), each = 3),
    X_shared = rnorm(15),
    Y = rnorm(15)
  )
  dat_subjects <- dat[!duplicated(dat$subject_id), ]
  
  spec <- bjlm_model() |>
    propensity(Trt ~ X_shared, data = dat_subjects) |>
    outcome(
      formula = Y ~ tau,
      b0 = ~ 1 + Trt + X_shared + (1 | subject_id),
      b1 = ~ 1,
      data = dat
    )
    
  expect_warning(
    compiled <- compile(spec),
    "Shared covariate name"
  )
  
  fit_res <- compiled |> fit(
    chains = 1L,
    iter = 10L,
    warmup = 5L,
    seed = 123L,
    verbose = FALSE,
    cores = 1L
  )
  
  # 1. Test fitted() with in-sample predictions
  pred_in <- fitted(fit_res)
  expect_s3_class(pred_in, "data.frame")
  expect_equal(nrow(pred_in), 15)
  expect_true(all(c("fitted_mean", "fitted_Q2.5", "fitted_Q97.5") %in% colnames(pred_in)))
  
  # 2. Test fitted() with summary = FALSE (returns raw draw matrix)
  pred_raw <- fitted(fit_res, summary = FALSE)
  expect_true(is.matrix(pred_raw))
  expect_equal(dim(pred_raw), c(5, 15)) # 5 draws (10 iter - 5 warmup) by 15 observations
  
  # 3. Test fitted() with newdata (out-of-sample prediction)
  new_dat <- data.frame(
    subject_id = c(1, 2, 99), # 99 is a new subject
    tau = c(1.5, 2.5, 0.5),
    Trt = c(1, 0, 1),
    X_shared = c(0.5, -0.5, 0.0)
  )
  pred_out <- fitted(fit_res, newdata = new_dat)
  expect_s3_class(pred_out, "data.frame")
  expect_equal(nrow(pred_out), 3)
  
  # 4. Test model diagnostics S3 methods
  ll <- log_lik(fit_res)
  expect_true(is.matrix(ll))
  expect_equal(dim(ll), c(5, 15))
  
  l_res <- suppressWarnings(loo(fit_res))
  expect_s3_class(l_res, "loo")
  
  w_res <- suppressWarnings(waic(fit_res))
  expect_s3_class(w_res, "waic")
  
  # pp_check returns a ggplot object
  pp <- pp_check(fit_res, n_draws = 2)
  expect_s3_class(pp, "ggplot")
  
  # 5. Test advanced plotting methods
  p_pred_pop <- plot_predictions(fit_res, type = "population")
  expect_s3_class(p_pred_pop, "ggplot")
  
  p_pred_sub <- plot_predictions(fit_res, type = "subject", n_subjects = 2)
  expect_s3_class(p_pred_sub, "ggplot")
  # The per-subject 95% credible band must actually be drawn (it was previously
  # computed and then discarded).
  expect_true("GeomRibbon" %in% vapply(p_pred_sub$layers,
                                       function(l) class(l$geom)[1], character(1)))
  
  p_pred_both <- plot_predictions(fit_res, type = "both", n_subjects = 2)
  expect_s3_class(p_pred_both, "ggplot")
  
  # plot_gp should throw warning since there's no GP block in this model
  expect_warning(plot_gp(fit_res), "No latent Gaussian Process draws found")
  
  p_prop_overlap <- plot_propensity(fit_res, type = "overlap")
  expect_s3_class(p_prop_overlap, "ggplot")
  
  p_prop_weights <- plot_propensity(fit_res, type = "weights")
  expect_s3_class(p_prop_weights, "ggplot")
  
  p_prop_both <- plot_propensity(fit_res, type = "both")
  expect_type(p_prop_both, "list")
  expect_s3_class(p_prop_both$overlap, "ggplot")
  expect_s3_class(p_prop_both$weights, "ggplot")
})

test_that("plot_predictions draws subject bands for a GP model without a random intercept", {
  skip_if_not_installed("ggplot2")
  set.seed(7)
  ns <- 6L; nt <- 5L
  dat <- data.frame(
    Y = rnorm(ns * nt),
    tau = rep(seq(0, 4, length.out = nt), ns),
    X_obs = rnorm(ns * nt),
    series = factor(rep(seq_len(ns), each = nt))
  )
  fit_gp <- bjlm_model() |>
    outcome(Y ~ tau, b0 = ~ 1 + X_obs, b1 = ~ 1, deltas = list(~ 1),
            omega = list(~ 1), rho = list(~ 1), data = dat) |>
    latent_gp(name = "X_obs", data = dat, obs_var = "X_obs", time_var = "tau",
              time_out_var = "tau", time_trt_var = "tau", subject = "series") |>
    compile() |>
    fit(chains = 1L, iter = 12L, warmup = 6L, seed = 7L, verbose = FALSE)

  # No random intercept, so the subject grouping must fall back to the GP's
  # subject variable; the subject band was invisible before that fallback.
  expect_null(fit_gp$subject_var)
  p <- plot_predictions(fit_gp, type = "subject", n_subjects = 3)
  expect_s3_class(p, "ggplot")
  geoms <- vapply(p$layers, function(l) class(l$geom)[1], character(1))
  expect_true("GeomRibbon" %in% geoms)   # credible band drawn
  expect_true("GeomLine" %in% geoms)     # fitted trajectory drawn
})

test_that("recovery_plot works with bjlm_fit objects", {
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("posterior")
  
  set.seed(42)
  dat <- simulate_smoothbp(n_subj = 5, n_obs = 3, seed = 42)
  
  fit <- bjlm_model() |> 
    outcome(y ~ tau, b0 = ~ 1 + (1 | subject), b1 = ~ 1, data = dat) |> 
    compile() |> 
    fit(iter = 20, warmup = 10, chains = 1, verbose = FALSE)
    
  p <- recovery_plot(fit, dat)
  expect_s3_class(p, "ggplot")
})
