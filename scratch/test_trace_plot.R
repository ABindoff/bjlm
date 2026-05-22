# Fast verification script for trace_plot on bipw_fit objects
library(devtools)
load_all(".")

set.seed(42)
n <- 100

# Confounded data
X1 <- rnorm(n)
X2 <- rnorm(n)
pi_true <- plogis(0.5 + 0.8 * X1 - 0.3 * X2)
Trt <- rbinom(n, 1, pi_true)
Y <- 5 + 2.0 * Trt + 1.5 * X1 + 0.5 * X2 + rnorm(n)

dat <- data.frame(
  Y = Y,
  tau = rep(0, n),
  Trt = Trt,
  X1 = X1,
  X2 = X2
)

cat("=== Fitting fast bipw model ===\n")
fit <- bipw(
  outcome    = Y ~ tau,
  b0         = ~ 1 + Trt + X1 + X2,
  b1         = ~ 1,
  propensity = Trt ~ X1 + X2,
  weights    = "stabilised_ate",
  max_weight = 20,
  data       = dat,
  chains     = 2L,
  iter       = 200L,
  warmup     = 100L,
  seed       = 123L,
  verbose    = FALSE,
  cores      = 1L
)

cat("=== Fit finished. Testing trace_plot() ===\n")
p_trace <- trace_plot(fit)
print(p_trace)

cat("=== Testing S3 plot() method ===\n")
p_plot <- plot(fit, type = "both")
print(p_plot$trace)
print(p_plot$density)

cat("=== Verification complete! No errors! ===\n")
