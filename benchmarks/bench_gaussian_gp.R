devtools::load_all(".")
library(rstan)
library(dplyr)
library(ggplot2)
library(bayesplot)
library(posterior)

set.seed(42)

# 1. Simulate Gaussian Joint Model Data
N <- 100
time <- sort(runif(N, 0, 10))

true_b0_int <- 0.5
true_b0_x <- 2.0
true_sigma_y <- 1.0

true_alpha <- 1.0
true_rho <- 3.0  # Increased lengthscale for smoother GP
true_sigma_x <- 0.5 # Increased noise to make GP sampling easier

# Generate GP
dist_mat <- as.matrix(dist(time))
K <- true_alpha^2 * exp(-0.5 * (dist_mat / true_rho)^2) + diag(1e-9, N)
L <- t(chol(K))
z <- rnorm(N)
X_true <- as.numeric(L %*% z)

# Generate observed data
X_obs <- X_true + rnorm(N, 0, true_sigma_x)
Y <- true_b0_int + true_b0_x * X_true + rnorm(N, 0, true_sigma_y)

dat <- data.frame(
  time = time,
  X_obs = X_obs,
  Y = Y,
  dummy_trt = rbinom(N, 1, 0.5),
  id = rep(1, N) # bjlm requires id column for longitudinal structure even if N_subj=1
)

weight_mod <- glm(dummy_trt ~ 1, data = dat, family = binomial())
dat$pi_hat <- predict(weight_mod, type = "response")
dat$w <- ifelse(dat$dummy_trt == 1, 1 / dat$pi_hat, 1 / (1 - dat$pi_hat))

cat("\n=== Fitting rstan model ===\n")
stan_data <- list(
  N = N,
  Y = Y,
  X_obs = X_obs,
  time = time,
  w = dat$w
)

stan_mod <- stan_model("benchmarks/joint_gp.stan")

t_stan <- system.time({
stan_fit <- sampling(
  stan_mod,
  data = stan_data,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 42,
  refresh = 0
)
})

cat("\n=== Fitting bjlm model ===\n")
t_bjlm <- system.time({
bjlm_fit <- bjlm_model() |>
  propensity(dummy_trt ~ 1, data = dat) |>
  outcome(
    Y ~ time,
    b0 = ~ 1 + X_obs,
    data = dat,
    family = "gaussian"
  ) |>
  latent_gp(
    name = "X_obs",
    data = dat,
    time_var = "time",
    obs_var = "X_obs",
    subject = "id",
    time_out_var = "time",
    time_trt_var = "time"
  ) |>
  compile() |>
  fit(chains = 4, iter = 2000, warmup = 1000, cores = 4, seed = 42)
})

# Since stan has sigma_y_sq, let's extract it and compute sigma_y
stan_draws <- as.data.frame(stan_fit, pars = c("b0_int", "b0_x", "sigma_y_sq", "alpha", "rho", "sigma_x")) |>
  select(
    `b0_(Intercept)` = b0_int,
    `b0_X_obs` = b0_x,
    sigma = sigma_y_sq,
    X_obs_alpha = alpha,
    X_obs_rho = rho,
    X_obs_sigma_x = sigma_x
  ) |>
  mutate(sigma = sqrt(sigma)) |>
  mutate(model = "rstan")

bjlm_draws <- as_draws_df(bjlm_fit$draws) |> 
  as_tibble() |>
  select(
    `b0_(Intercept)` = `b0_(Intercept)`,
    `b0_X_obs` = b0_X_obs,
    sigma = sigma,
    X_obs_alpha = X_obs_alpha,
    X_obs_rho = X_obs_rho,
    X_obs_sigma_x = X_obs_sigma_x
  ) |>
  mutate(model = "bjlm")

combined_draws <- bind_rows(stan_draws, bjlm_draws)

params_to_plot <- c("b0_(Intercept)", "b0_X_obs", "sigma", "X_obs_alpha", "X_obs_rho", "X_obs_sigma_x")

plot_list <- list()
wasserstein_dists <- numeric(length(params_to_plot))
names(wasserstein_dists) <- params_to_plot

for (p in params_to_plot) {
  plt <- ggplot(combined_draws, aes(x = .data[[p]], fill = model)) +
    geom_density(alpha = 0.5) +
    theme_minimal() +
    labs(title = paste("Posterior:", p), x = "Value", y = "Density") +
    scale_fill_manual(values = c("rstan" = "blue", "bjlm" = "orange")) +
    theme(legend.position = "bottom")
  plot_list[[p]] <- plt
  
  x_stan <- stan_draws[[p]]
  x_bjlm <- bjlm_draws[[p]]
  
  if (length(x_stan) == length(x_bjlm)) {
    w_dist <- mean(abs(sort(x_stan) - sort(x_bjlm)))
  } else {
    q <- seq(0, 1, length.out = 1000)
    w_dist <- mean(abs(quantile(x_stan, q) - quantile(x_bjlm, q)))
  }
  wasserstein_dists[p] <- w_dist
}

library(patchwork)
final_plot <- wrap_plots(plot_list, ncol = 2) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
ggsave("benchmarks/bench_gaussian_gp.png", final_plot, width = 12, height = 12)

cat("\n=== Benchmark Summary ===\n")
cat("rstan execution time: ", t_stan["elapsed"], " seconds\n")
cat("bjlm execution time: ", t_bjlm["elapsed"], " seconds\n")

cat("\n=== Wasserstein Distances (rstan vs bjlm) ===\n")
for (p in names(wasserstein_dists)) {
  cat(sprintf("  %-20s : %.4f\n", p, wasserstein_dists[p]))
}

max_w <- max(wasserstein_dists)
if (max_w < 0.1) {
    cat("\nSUCCESS: All parameters have Wasserstein distance < 0.1 (Max:", max_w, ")\n")
} else {
    cat("\nWARNING: Some parameters have Wasserstein distance > 0.1 (Max:", max_w, ")\n")
}

saveRDS(list(stan = stan_fit, bjlm = bjlm_fit), "benchmarks/bench_gaussian_gp.rds")
