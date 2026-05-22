test_that("bipw fit, printing, and plotting methods work correctly", {
  skip_if_not_installed("ggplot2")
  
  set.seed(42)
  n <- 50
  
  # Simple confounded data
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
  
  # Run a fast model
  fit <- bipw(
    outcome    = Y ~ tau,
    b0         = ~ 1 + Trt + X1,
    b1         = ~ 1,
    propensity = Trt ~ X1,
    weights    = "stabilised_ate",
    data       = dat,
    chains     = 2L,
    iter       = 100L,
    warmup     = 50L,
    seed       = 123L,
    verbose    = FALSE,
    cores      = 1L
  )
  
  expect_s3_class(fit, "bipw_fit")
  
  # Test output printing & summary methods
  expect_output(print(fit))
  expect_output(summary(fit, model = "outcome"))
  expect_output(summary(fit, model = "propensity"))
  
  # Test diagnostics and causal effect extracts
  ce <- causal_effect(fit, param = "b0_Trt")
  expect_type(ce, "list")
  expect_named(ce, c("parameter", "mean", "sd", "lower", "upper", "draws", "prob"))
  expect_type(ce$draws, "double")
  expect_length(ce$draws, 100) # 2 chains x 50 post-warmup draws
  
  wd <- weight_diagnostics(fit)
  expect_type(wd, "double")
  expect_length(wd, 100)
  
  # Test plotting methods (trace_plot and S3 plot)
  p_trace <- trace_plot(fit)
  expect_s3_class(p_trace, "ggplot")
  
  p_dens <- trace_plot(fit, type = "density")
  expect_s3_class(p_dens, "ggplot")
  
  p_both <- trace_plot(fit, type = "both")
  expect_type(p_both, "list")
  expect_named(p_both, c("trace", "density"))
  expect_s3_class(p_both$trace, "ggplot")
  expect_s3_class(p_both$density, "ggplot")
  
  p_plot <- plot(fit)
  expect_s3_class(p_plot, "ggplot")
})
