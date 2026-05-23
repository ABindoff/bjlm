test_that("Leave-Future-Out Cross-Validation (LFO-CV) works as expected", {
  skip_if_not_installed("posterior")
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("loo")
  
  set.seed(456)
  dat <- data.frame(
    subject_id = rep(1:5, each = 4),
    tau = rep(0:3, times = 5),
    Trt = rep(rbinom(5, 1, 0.5), each = 4),
    X_shared = rnorm(20),
    Y = rnorm(20)
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
  
  # Fit with 100 post-warmup draws for realistic PSIS tail estimation
  fit_res <- compiled |> fit(
    chains = 1L,
    iter = 200L,
    warmup = 100L,
    seed = 456L,
    verbose = FALSE,
    cores = 1L
  )
  
  # 1. Test basic LFO-CV run (approximate where k < threshold)
  # Starting evaluation from tau = 1 (meaning training on tau <= 1, predicting tau = 2 and 3)
  # Since the out-of-sample Pareto k for this small model is ~1.38, we use k_threshold = 2.0
  # to verify the approximate prediction path (refit = FALSE) without triggering a refit.
  suppressWarnings(
    lfo_approx <- lfo_cv(fit_res, min_tau = 1, k_threshold = 2.0, verbose = FALSE)
  )
  
  expect_s3_class(lfo_approx, "bjlm_lfo")
  expect_type(lfo_approx$elpd_lfo, "double")
  expect_s3_class(lfo_approx$pointwise, "data.frame")
  expect_s3_class(lfo_approx$diagnostics, "data.frame")
  
  # The evaluated time points should be > min_tau (which means tau = 2 or 3)
  expect_true(all(lfo_approx$pointwise$tau %in% c(2, 3)))
  expect_equal(nrow(lfo_approx$diagnostics), 2) # predicting tau = 2 and 3
  expect_equal(lfo_approx$diagnostics$predict_time[1], 2)
  expect_equal(lfo_approx$diagnostics$predict_time[2], 3)
  expect_false(lfo_approx$diagnostics$refit[1])
  
  # 2. Test exact LFO-CV run by forcing k_threshold = -1.0 (forces refits at every step)
  suppressWarnings(
    lfo_exact <- lfo_cv(fit_res, min_tau = 1, k_threshold = -1.0, verbose = FALSE)
  )
  expect_s3_class(lfo_exact, "bjlm_lfo")
  expect_true(lfo_exact$diagnostics$refit[1])
  
  # 3. Test print S3 method
  output <- capture.output(print(lfo_approx))
  expect_true(any(grepl("Leave-Future-Out Cross-Validation", output)))
  expect_true(any(grepl("Total LFO ELPD", output)))
  expect_true(any(grepl("Model refits", output)))
  
  # 4. Test plot S3 method (should return ggplot)
  p <- plot(lfo_approx)
  expect_s3_class(p, "ggplot")
  
  # 5. Check error conditions
  # min_tau larger than grid
  expect_error(lfo_cv(fit_res, min_tau = 99), "min_tau must be smaller than the maximum time point")
  # min_tau smaller than grid
  expect_error(lfo_cv(fit_res, min_tau = -5), "min_tau is smaller than the minimum time point")
})
