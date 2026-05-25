devtools::load_all(".")
library(brms)
library(dplyr)
library(tidyr)
library(ggplot2)
library(bayesplot)

set.seed(42)

n_subjects <- 100
n_obs_per_subject <- 10

true_b0_int <- 0.5
true_b0_trt <- 1.5
true_b0_x <- -0.5
true_b1_int <- 0.8
true_om <- 5.0
true_rho <- 2.0
true_r <- 5.0
true_sigma_u <- 0.5

dat <- expand.grid(
  tau = 1:n_obs_per_subject,
  id = 1:n_subjects
) |> 
  arrange(id, tau) |>
  mutate(
    X_obs = rnorm(n(), mean = 0, sd = 1),
    pi_trt = plogis(0.5 + 1.2 * X_obs),
    Trt = rbinom(n(), 1, pi_trt),
    u = rep(rnorm(n_subjects, 0, true_sigma_u), each = n_obs_per_subject),
    d = tau - true_om,
    s = plogis(d * true_rho),
    # Add a pre-slope of 0.2 to test both b1 and delta
    eta = true_b0_int + u + true_b0_trt * Trt + true_b0_x * X_obs + 0.2 * d + true_b1_int * d * s,
    mu = exp(eta),
    Y = rnbinom(n(), size = true_r, mu = mu)
)

weight_mod <- glm(Trt ~ X_obs, data = dat, family = binomial())
dat$pi_hat <- predict(weight_mod, type = "response")
dat$w <- ifelse(dat$Trt == 1, 1 / dat$pi_hat, 1 / (1 - dat$pi_hat))

cat("\n=== Fitting brms model ===\n")
brms_form <- bf(
  Y | weights(w) ~ b0 + b1 * (tau - 5.0) + delta * (tau - 5.0) * inv_logit(2.0 * (tau - 5.0)),
  b0 ~ 1 + Trt + X_obs + (1 | id),
  b1 ~ 1,
  delta ~ 1,
  nl = TRUE
)

bpriors <- c(
  prior(normal(0, 5), nlpar = "b0"),
  prior(normal(0, 5), nlpar = "b1"),
  prior(normal(0, 5), nlpar = "delta")
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

cat("\n=== Fitting bjlm model ===\n")
# Create priors with narrow normals to fix omega and rho
my_priors <- bjlm_priors(
  outcome = smoothbp_priors(
    omega = list(prior_normal(5.0, 1e-6)),
    rho = list(prior_normal(2.0, 1e-6))
  )
)

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
  fit(chains = 4, iter = 2000, warmup = 1000, cores = 4, seed = 42, priors = my_priors)
})

brms_draws <- as_draws_df(brms_fit) |> 
  as_tibble() |>
  select(
    `b0_(Intercept)` = b_b0_Intercept,
    `b0_Trt` = b_b0_Trt,
    `b0_X_obs` = b_b0_X_obs,
    `b1_(Intercept)` = b_b1_Intercept,
    `delta1_(Intercept)` = b_delta_Intercept,
    shape = shape,
    sigma_u = sd_id__b0_Intercept
  ) |>
  mutate(model = "brms")

bjlm_draws <- as_draws_df(bjlm_fit$draws) |> 
  as_tibble() |>
  select(
    `b0_(Intercept)` = `b0_(Intercept)`,
    `b0_Trt` = b0_Trt,
    `b0_X_obs` = b0_X_obs,
    `b1_(Intercept)` = `b1_(Intercept)`,
    `delta1_(Intercept)` = `delta1_(Intercept)`,
    shape = r,
    sigma_u = sigma_u
  ) |>
  # Adjust bjlm b0_Intercept by adding log(r) to match brms parametrization
  mutate(`b0_(Intercept)` = `b0_(Intercept)` + log(shape)) |>
  mutate(model = "bjlm")

combined_draws <- bind_rows(brms_draws, bjlm_draws)

params_to_plot <- c("b0_(Intercept)", "b0_Trt", "b0_X_obs", "b1_(Intercept)", "delta1_(Intercept)", "shape", "sigma_u")

plot_list <- list()
wasserstein_dists <- numeric(length(params_to_plot))
names(wasserstein_dists) <- params_to_plot

for (p in params_to_plot) {
  plt <- ggplot(combined_draws, aes(x = .data[[p]], fill = model)) +
    geom_density(alpha = 0.5) +
    theme_minimal() +
    labs(title = paste("Posterior:", p), x = "Value", y = "Density") +
    scale_fill_manual(values = c("brms" = "blue", "bjlm" = "orange")) +
    theme(legend.position = "bottom")
  plot_list[[p]] <- plt
  
  # Calculate Wasserstein distance
  x_brms <- brms_draws[[p]]
  x_bjlm <- bjlm_draws[[p]]
  
  if (length(x_brms) == length(x_bjlm)) {
    w_dist <- mean(abs(sort(x_brms) - sort(x_bjlm)))
  } else {
    # If different number of samples, use quantile approximation
    q <- seq(0, 1, length.out = 1000)
    w_dist <- mean(abs(quantile(x_brms, q) - quantile(x_bjlm, q)))
  }
  wasserstein_dists[p] <- w_dist
}

library(patchwork)
final_plot <- wrap_plots(plot_list, ncol = 2) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
ggsave("benchmarks/bench_nb_fixed.png", final_plot, width = 12, height = 14)

cat("\n=== Benchmark Summary ===\n")
cat("brms execution time: ", t_brms["elapsed"], " seconds\n")
cat("bjlm execution time: ", t_bjlm["elapsed"], " seconds\n")

cat("\n=== Wasserstein Distances (brms vs bjlm) ===\n")
for (p in names(wasserstein_dists)) {
  cat(sprintf("  %-20s : %.4f\n", p, wasserstein_dists[p]))
}

max_w <- max(wasserstein_dists)
if (max_w < 0.1) {
    cat("\nSUCCESS: All parameters have Wasserstein distance < 0.1 (Max:", max_w, ")\n")
} else {
    cat("\nWARNING: Some parameters have Wasserstein distance > 0.1 (Max:", max_w, ")\n")
}
