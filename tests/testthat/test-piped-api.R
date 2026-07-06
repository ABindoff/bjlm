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
  expect_s3_class(compile(bjlm_model() |> outcome(Y ~ X1, data = dat)), "bjlm_compiled_model")

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

test_that("Piped API: dataset alignment, warning, and compile_report generation work for separate subject and observation-level datasets", {
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("posterior")

  # 1. Create a simulated dataset with separate subject-level and observation-level data
  set.seed(123)
  n_subjects <- 15
  n_obs_per_subject <- 5

  # Subject-level dataset (propensity)
  # Trt is subject-level
  # X_shared is subject-level, but outcome also has an observation-level X_shared
  # X_only_subject is subject-level only
  dat_subjects <- data.frame(
    subject_id = 1:n_subjects,
    Trt = rbinom(n_subjects, 1, 0.5),
    X_shared = rnorm(n_subjects, mean = 2, sd = 0.5),
    X_only_subject = rnorm(n_subjects, mean = -1, sd = 0.5)
  )

  # Observation-level dataset (outcome)
  # tau is time-varying
  # X_shared is observation-level (time-varying)
  dat_observations <- data.frame(
    subject_id = rep(1:n_subjects, each = n_obs_per_subject),
    tau = rep(0:(n_obs_per_subject - 1), times = n_subjects),
    X_shared = rnorm(n_subjects * n_obs_per_subject, mean = 0, sd = 1)
  )

  # Generate outcome Y
  # Y depends on Trt (subject-level), X_shared (observation-level), and X_only_subject (subject-level, needs expansion)
  # Expand columns manually for simulation
  trt_expanded <- dat_subjects$Trt[match(dat_observations$subject_id, dat_subjects$subject_id)]
  x_sub_expanded <- dat_subjects$X_only_subject[match(dat_observations$subject_id, dat_subjects$subject_id)]

  dat_observations$Y <- 10 + 2.0 * trt_expanded + 1.5 * dat_observations$X_shared + 0.8 * x_sub_expanded + rnorm(nrow(dat_observations), sd = 0.5)

  # 2. Build model spec
  spec <- bjlm_model() |>
    propensity(Trt ~ X_shared, data = dat_subjects) |>
    outcome(
      formula = Y ~ tau,
      b0 = ~ 1 + Trt + X_shared + X_only_subject + (1 | subject_id),
      b1 = ~ 1,
      data = dat_observations
    )

  # 3. Compile and check warnings/messages.
  # compile() no longer writes compile_report.md to the working directory (CRAN
  # policy); it is opt-in via options(bjlm.compile_report=). Point it at a tempfile.
  report_file <- tempfile(fileext = ".md")
  withr::local_options(bjlm.compile_report = report_file)

  # We expect a warning about shared covariate name 'X_shared' (compile() also
  # emits an informational DR note for the shared covariate; suppress messages).
  suppressMessages(
    expect_warning(
      compiled <- compile(spec),
      "Shared covariate name\\(s\\) detected: 'X_shared'"
    )
  )

  expect_s3_class(compiled, "bjlm_compiled_model")

  # Check that the expanded columns are added to the outcome data
  expect_true("X_only_subject" %in% names(compiled$model$outcome$data))
  expect_true("Trt" %in% names(compiled$model$outcome$data))

  # Check that X_shared was NOT overwritten (independent scoping)
  # The mean of dat_observations$X_shared should be around 0, not 2
  expect_equal(compiled$model$outcome$data$X_shared, dat_observations$X_shared)

  # 4. Verify the opt-in compile report contains the Mermaid flowchart
  expect_true(file.exists(report_file))
  report_text <- paste(readLines(report_file), collapse = "\n")

  expect_true(grepl("graph TD", report_text))
  expect_true(grepl("Propensity Block", report_text))
  expect_true(grepl("Outcome Block", report_text))
  expect_true(grepl("Shared variables: X_shared", report_text))

  # 5. Fit the model to ensure it runs without "Allocation from iterator error"
  fit_res <- compiled |> fit(
    chains = 2L,
    iter = 40L,
    warmup = 20L,
    seed = 123L,
    verbose = FALSE,
    cores = 1L
  )

  expect_s3_class(fit_res, "bjlm_fit")
})

test_that("flowchart S3 methods extract and format Mermaid diagrams correctly", {
  skip_if_not_installed("posterior")
  
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
  
  # 1. Check flowchart on compiled model
  fc_comp <- flowchart(compiled)
  expect_s3_class(fc_comp, "bjlm_flowchart")
  expect_true(inherits(unclass(fc_comp), "character"))
  expect_true(grepl("graph TD", fc_comp))
  expect_true(grepl("Align by subject ID: subject_id", fc_comp))
  expect_true(grepl("Shared variables:.*X_shared", fc_comp))
  expect_output(print(fc_comp), "```mermaid")
  
  # 2. Check flowchart on fit object
  fit_res <- compiled |> fit(
    chains = 1L,
    iter = 10L,
    warmup = 5L,
    seed = 123L,
    verbose = FALSE,
    cores = 1L
  )
  
  fc_fit <- flowchart(fit_res)
  expect_s3_class(fc_fit, "bjlm_flowchart")
  expect_true(grepl("graph TD", fc_fit))
  expect_true(grepl("Align by subject ID: subject_id", fc_fit))
  expect_true(grepl("Shared variables:.*X_shared", fc_fit))
})

test_that("population() block builds correctly and population_predict() returns valid draws", {
  skip_if_not_installed("posterior")

  set.seed(42)
  n <- 60

  age  <- sample(c("young", "old"), n, replace = TRUE)
  sex  <- sample(c("M", "F"),       n, replace = TRUE)
  X1   <- ifelse(age == "old", 1, 0) + rnorm(n, sd = 0.5)
  pi_t <- plogis(0.3 + 0.5 * X1)
  Trt  <- rbinom(n, 1, pi_t)
  Y    <- 4 + 1.5 * Trt + 0.8 * X1 + rnorm(n)

  dat <- data.frame(Y = Y, tau = rep(0, n), Trt = Trt, X1 = X1, age = age, sex = sex)

  # Census: 4 cells (age x sex) with population counts
  census <- data.frame(
    age    = c("young", "young", "old", "old"),
    sex    = c("M",     "F",     "M",   "F"),
    N_pop  = c(1200,    1100,    900,   800),
    Trt    = c(0.4,     0.4,     0.6,   0.6),  # pop treatment prevalence (not used for PATE)
    X1     = c(0.1,    -0.1,     0.8,   0.7)
  )

  # ---- population() builder validation ----
  bad_model <- bjlm_model() |> outcome(Y ~ X1, data = dat)
  expect_error(
    population(bad_model, cells = "not_a_df"),
    "'cells' must be a data frame"
  )
  expect_error(
    population(bad_model, cells = census, weight = "missing_col"),
    "Weight column 'missing_col' not found"
  )
  expect_error(
    population(bad_model, cells = census, strata = ~ age + missing_var),
    "Strata variable"
  )

  # ---- Valid model spec with population block ----
  spec <- bjlm_model() |>
    propensity(Trt ~ X1, data = dat) |>
    outcome(Y ~ X1, data = dat) |>
    population(
      cells   = census,
      weight  = "N_pop",
      strata  = ~ age + sex,
      at      = list()
    )

  expect_false(is.null(spec$population))
  expect_equal(nrow(spec$population$cells), 4L)
  expect_output(print(spec), "Population Block")
  expect_output(print(spec), "Census cells: 4")

  # ---- compile() passes population through ----
  compiled <- compile(spec)
  expect_false(is.null(compiled$population))
  expect_equal(nrow(compiled$population$cells), 4L)
  expect_output(print(compiled), "Census cells:  4")

  # ---- fit() passes population through ----
  fit_res <- compiled |> fit(
    chains  = 2L,
    iter    = 100L,
    warmup  = 50L,
    seed    = 7L,
    verbose = FALSE,
    cores   = 1L
  )
  expect_false(is.null(fit_res$population))

  # ---- population_predict: type = "response" ----
  pop_resp <- population_predict(fit_res, type = "response", seed = 1L)
  expect_s3_class(pop_resp, "data.frame")
  expect_named(pop_resp, c("pop_mean", "pop_Q2.5", "pop_Q97.5"))
  expect_true(pop_resp$pop_Q2.5 < pop_resp$pop_mean)
  expect_true(pop_resp$pop_mean < pop_resp$pop_Q97.5)

  # ---- population_predict: summary = FALSE returns numeric vector ----
  pop_draws <- population_predict(fit_res, type = "response", summary = FALSE, seed = 1L)
  expect_type(pop_draws, "double")
  expect_length(pop_draws, nrow(posterior::as_draws_matrix(fit_res$draws)))

  # ---- population_predict: type = "ate" (PATE) ----
  # census must have Trt column (already does); population_predict sets trt_var to 1/0
  pop_ate <- population_predict(fit_res, type = "ate", seed = 1L)
  expect_s3_class(pop_ate, "data.frame")
  expect_named(pop_ate, c("pop_mean", "pop_Q2.5", "pop_Q97.5"))

  # ---- population_predict: at$tau trajectory (zero-bp model: no-op, but ensure no error) ----
  # For a zero-breakpoint model, at$tau is silently ignored (tau doesn't affect predictions).
  pop_at <- population_predict(fit_res, at = list(tau = c(0, 1, 2)), type = "response", seed = 1L)
  # Should return a single-row data frame (zero-bp ignores tau grid)
  expect_s3_class(pop_at, "data.frame")

  # ---- population_predict: error when no population spec ----
  bare_spec <- bjlm_model() |> outcome(Y ~ X1, data = dat)
  bare_fit  <- compile(bare_spec) |> fit(
    chains = 1L, iter = 40L, warmup = 20L, seed = 1L, verbose = FALSE
  )
  expect_error(population_predict(bare_fit), "No population specification")
})

test_that("population_predict trajectory returns one row per tau value for piecewise models", {
  skip_if_not_installed("posterior")

  set.seed(99)
  n_subj <- 10; n_obs <- 6
  subj <- rep(1:n_subj, each = n_obs)
  tau  <- rep(seq(0, 5, length.out = n_obs), times = n_subj)
  X1   <- rep(rnorm(n_subj), each = n_obs)
  Trt  <- rep(rbinom(n_subj, 1, 0.5), each = n_obs)
  Y    <- 3 + 0.5 * Trt + 0.4 * X1 - 0.1 * tau + rnorm(n_subj * n_obs, sd = 0.5)
  dat  <- data.frame(subject = subj, tau = tau, Y = Y, X1 = X1, Trt = Trt)
  dat_subj <- dat[!duplicated(dat$subject), ]

  census <- data.frame(X1 = c(-0.5, 0.5), Trt = c(0, 1), N_pop = c(500, 500))

  spec <- bjlm_model() |>
    propensity(Trt ~ X1, data = dat_subj) |>
    outcome(
      formula = Y ~ tau,
      b0      = ~ 1 + Trt + X1 + (1 | subject),
      b1      = ~ 1,
      data    = dat
    ) |>
    population(cells = census, weight = "N_pop", at = list(tau = c(0, 2, 4)))

  compiled <- suppressWarnings(compile(spec))
  fit_res  <- compiled |> fit(
    chains = 1L, iter = 80L, warmup = 40L, seed = 5L, verbose = FALSE
  )

  # Trajectory: should return 3 rows (one per tau)
  traj <- population_predict(fit_res, type = "response", seed = 2L)
  expect_s3_class(traj, "data.frame")
  expect_equal(nrow(traj), 3L)
  expect_named(traj, c("tau", "pop_mean", "pop_Q2.5", "pop_Q97.5"))
  expect_equal(traj$tau, c(0, 2, 4))

  # summary = FALSE returns a matrix with 3 columns
  traj_mat <- population_predict(fit_res, type = "response", summary = FALSE, seed = 2L)
  expect_true(is.matrix(traj_mat))
  expect_equal(ncol(traj_mat), 3L)
  expect_equal(colnames(traj_mat), c("tau=0", "tau=2", "tau=4"))
})
