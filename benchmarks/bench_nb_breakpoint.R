devtools::load_all(".")
library(brms)
library(dplyr)
library(tidyr)
library(ggplot2)
library(bayesplot)

set.seed(42)

# 1. Simulate Negative Binomial Data with a Breakpoint and Time-varying Confounding
n_subjects <- 100
n_obs_per_subject <- 10
n_total <- n_subjects * n_obs_per_subject

# True parameters
true_b0_int <- 0.5
true_b0_trt <- 1.5
true_b0_x <- -0.5
true_b1_int <- 0.8
true_om <- 5.0
true_rho <- 2.0
true_r <- 5.0 # Dispersion parameter (brms uses "shape")
true_sigma_u <- 0.5

# Generate data
dat <- expand.grid(
  tau = 1:n_obs_per_subject,
  id = 1:n_subjects
) |> 
  arrange(id, tau) |>
  mutate(
    # Time-varying confounder
    X_obs = rnorm(n(), mean = 0, sd = 1),
    # Treatment probability depends on X_obs (confounding)
    pi_trt = plogis(0.5 + 1.2 * X_obs),
    Trt = rbinom(n(), 1, pi_trt),
    # Random intercepts
    u = rep(rnorm(n_subjects, 0, true_sigma_u), each = n_obs_per_subject),
    # Breakpoint function
    d = tau - true_om,
    s = plogis(d * true_rho),
    # Linear Predictor
    eta = true_b0_int + u + true_b0_trt * Trt + true_b0_x * X_obs + true_b1_int * d * s,
    # Negative Binomial Outcome
    mu = exp(eta),
    # Negative Binomial parameterization in R: size = r, mu = mu
    Y = rnbinom(n(), size = true_r, mu = mu)
)

# 2. brms Implementation (Two-step IPW)
cat("\n=== Fitting brms model ===\n")
# Step A: Estimate weights using logistic regression
weight_mod <- glm(Trt ~ X_obs, data = dat, family = binomial())
dat$pi_hat <- predict(weight_mod, type = "response")
dat$w <- ifelse(dat$Trt == 1, 1 / dat$pi_hat, 1 / (1 - dat$pi_hat))

# Step B: Fit brms non-linear model
brms_form <- bf(
  Y | weights(w) ~ b0 + b1 * (tau - om) * inv_logit(rho * (tau - om)),
  b0 ~ 1 + Trt + X_obs + (1 | id),
  b1 ~ 1,
  om ~ 1,
  rho ~ 1,
  nl = TRUE
)

# Priors to match bjlm default weak priors
bpriors <- c(
  prior(normal(0, 5), nlpar = "b0"),
  prior(normal(0, 5), nlpar = "b1"),
  prior(normal(0, 5), nlpar = "om"),
  prior(normal(0, 5), nlpar = "rho", lb = 0)
)

t_brms <- system.time({
brms_fit <- brm(
  brms_form,
  data = dat,
  family = negbinomial(),
  prior = bpriors,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 42,
  backend = "rstan",
  silent = 2,
  refresh = 0
)
})

# 3. bjlm Implementation (Joint Model)
cat("\n=== Fitting bjlm model ===\n")
t_bjlm <- system.time({
bjlm_fit <- bjlm_model() |>
  propensity(Trt ~ X_obs, data = dat) |>
  outcome(
    Y ~ tau,
    b0 = ~ 1 + Trt + X_obs + (1 | id),
    b1 = ~ 1,
    deltas = list(~ 1),
    omega = list(~ 1),
    rho = list(~ 1),
    data = dat,
    family = "negative_binomial"
  ) |>
  compile() |>
  fit(chains = 4, iter = 2000, warmup = 1000, cores = 4, seed = 42)
})

# 4. Extract Posteriors and Compare
brms_draws <- as_draws_df(brms_fit) |> 
  as_tibble() |>
  select(
    `b0_(Intercept)` = b_b0_Intercept,
    `b0_Trt` = b_b0_Trt,
    `b0_X_obs` = b_b0_X_obs,
    `b1_(Intercept)` = b_b1_Intercept,
    om = b_om_Intercept,
    rho = b_rho_Intercept,
    shape = shape,
    sigma_u = sd_id__b0_Intercept
  ) |>
  mutate(model = "brms (Two-step IPW)")

cat("bjlm column names:\n")
print(colnames(as_draws_df(bjlm_fit$draws)))

bjlm_draws <- as_draws_df(bjlm_fit$draws) |> 
  as_tibble() |>
  select(
    `b0_(Intercept)` = `b0_(Intercept)`,
    `b0_Trt` = b0_Trt,
    `b0_X_obs` = b0_X_obs,
    `b1_(Intercept)` = `delta1_(Intercept)`, # brms b1 is actually the slope change (delta)
    om = `omega1_(Intercept)`,
    rho = `rho1_(Intercept)`,
    shape = r,
    sigma_u = sigma_u
  ) |>
  mutate(model = "bjlm (Joint Model)")

combined_draws <- bind_rows(brms_draws, bjlm_draws)

# Plotting overlaps
params_to_plot <- c("b0_(Intercept)", "b0_Trt", "b0_X_obs", "b1_(Intercept)", "om", "rho", "shape", "sigma_u")

plot_list <- list()
for (p in params_to_plot) {
  plt <- ggplot(combined_draws, aes(x = .data[[p]], fill = model)) +
    geom_density(alpha = 0.5) +
    theme_minimal() +
    labs(title = paste("Posterior:", p), x = "Value", y = "Density") +
    scale_fill_manual(values = c("brms (Two-step IPW)" = "blue", "bjlm (Joint Model)" = "orange")) +
    theme(legend.position = "bottom")
  plot_list[[p]] <- plt
}

library(patchwork)
final_plot <- wrap_plots(plot_list, ncol = 2) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
ggsave("benchmarks/bench_nb_breakpoint.png", final_plot, width = 12, height = 16)

# Print Summary
cat("\n=== Benchmark Summary ===\n")
cat("brms execution time: ", t_brms["elapsed"], " seconds\n")
cat("bjlm execution time: ", t_bjlm["elapsed"], " seconds\n")
cat("Speedup factor: ", t_brms["elapsed"] / t_bjlm["elapsed"], "x\n")

saveRDS(list(brms = brms_fit, bjlm = bjlm_fit), "benchmarks/bench_nb_breakpoint.rds")
