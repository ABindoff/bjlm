# test_suite_draft.R
devtools::load_all(".")
set.seed(42)

cat("=== Scenario 1: Basic Linear Model, No Breakpoints ===\n")
# Small N, basic setup
n <- 200
dat1 <- data.frame(id = 1:n, tau = runif(n, 0, 10), Trt = rbinom(n, 1, 0.5))
dat1$Y <- 2 + 0.5 * dat1$tau + 1.5 * dat1$Trt + rnorm(n)

fit1 <- bjlm_model() |>
  propensity(Trt ~ 1, data = dat1) |>
  outcome(Y ~ tau, b0 = ~ 1 + Trt, b1 = ~ 1, data = dat1) |>
  compile() |>
  fit(iter = 100, warmup = 50, chains = 1, verbose = FALSE)

print(tab_bjlm(fit1))

cat("\n=== Scenario 2: Logistic Regression with 1 Breakpoint & Extreme Weights ===\n")
# Test Polya-Gamma sampler, IPW weight trimming
n <- 300
tau <- runif(n, 0, 20)
# Create extreme propensity scores
pi_trt <- plogis(-2 + 0.2 * tau)
Trt <- rbinom(n, 1, pi_trt)
# Outcome with a shift at tau = 10
Y_prob <- plogis(0.5 + 0.2 * tau + 2 * Trt - 0.5 * pmax(0, tau - 10))
Y <- rbinom(n, 1, Y_prob)
dat2 <- data.frame(id = 1:n, tau = tau, Trt = Trt, Y = Y)

fit2 <- bjlm_model() |>
  propensity(Trt ~ tau, data = dat2) |>
  outcome(Y ~ tau, b0 = ~ 1 + Trt, b1 = ~ 1, deltas = list(~ 1), omega = list(~ 1), rho = list(~ 1), data = dat2, family = "binomial") |>
  compile() |>
  fit(iter = 200, warmup = 100, chains = 1, max_weight = 10, verbose = FALSE)

print(tab_bjlm(fit2))

cat("\n=== Scenario 3: Negative Binomial with Random Intercepts & GP ===\n")
n_subj <- 50
n_obs <- 5
tau_out <- runif(n_subj * n_obs, 0, 15)
id_out <- rep(1:n_subj, each = n_obs)
# True confounder
X_true <- sin(tau_out / 3)
Trt_expanded <- rbinom(n_subj * n_obs, 1, plogis(X_true))

# NB outcome
mu <- exp(0.5 + 0.1 * tau_out + Trt_expanded + X_true + rnorm(n_subj)[id_out])
Y_nb <- rnbinom(n_subj * n_obs, mu = mu, size = 2)

dat3 <- data.frame(id = factor(id_out), tau = tau_out, Trt = Trt_expanded, Y = Y_nb, X_obs = X_true + rnorm(n_subj*n_obs, 0, 0.2))

# Sub-sample for propensity (baseline only)
dat3_prop <- dat3[!duplicated(dat3$id), ]

fit3 <- bjlm_model() |>
  propensity(Trt ~ X_obs, data = dat3_prop) |>
  outcome(Y ~ tau, b0 = ~ 1 + Trt + X_obs + (1 | id), b1 = ~ 1, data = dat3, family = "negative_binomial") |>
  latent_gp("X_obs", data = dat3, time_var = "tau", obs_var = "X_obs", subject = "id", time_out_var = "tau", time_trt_var = "tau") |>
  compile() |>
  fit(iter = 150, warmup = 50, chains = 1, verbose = FALSE)

print(tab_bjlm(fit3))
cat("\n=== Tests completed ===\n")
