library(ggplot2)
library(dplyr)
library(posterior)
library(patchwork)

res <- readRDS("benchmarks/bench_nb_breakpoint.rds")
brms_fit <- res$brms
bjlm_fit <- res$bjlm

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

final_plot <- wrap_plots(plot_list, ncol = 2) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
ggsave("benchmarks/bench_nb_breakpoint.png", final_plot, width = 12, height = 16)
