library(posterior)

cat("\n--- NB Benchmark (brms vs bjlm) ---\n")
res1 <- readRDS("benchmarks/bench_nb_breakpoint.rds")
brms_sum <- summarise_draws(res1$brms, mean, sd)
bjlm_sum <- summarise_draws(res1$bjlm$draws, mean, sd)

# map bjlm names to brms for comparison
cat("\nBJLM:\n")
print(bjlm_sum[bjlm_sum$variable %in% c("b0_(Intercept)", "b0_Trt", "b0_X_obs", "delta1_(Intercept)", "omega1_(Intercept)", "rho1_(Intercept)", "r", "sigma_u"), ])
cat("\nBRMS:\n")
print(brms_sum[brms_sum$variable %in% c("b_b0_Intercept", "b_b0_Trt", "b_b0_X_obs", "b_b1_Intercept", "b_om_Intercept", "b_rho_Intercept", "shape", "sd_id__b0_Intercept"), ])

cat("\n--- GP Benchmark (rstan vs bjlm) ---\n")
res2 <- readRDS("benchmarks/bench_gaussian_gp.rds")
rstan_sum <- summarise_draws(res2$rstan, mean, sd)
bjlm2_sum <- summarise_draws(res2$bjlm$draws, mean, sd)

cat("\nBJLM:\n")
print(bjlm2_sum[bjlm2_sum$variable %in% c("b0_(Intercept)", "b0_X_obs", "sigma", "alpha_(Intercept)", "rho_(Intercept)", "sigma_x_(Intercept)"), ])
cat("\nRSTAN:\n")
print(rstan_sum[rstan_sum$variable %in% c("b0_int", "b0_x", "sigma_y", "alpha", "rho", "sigma_x"), ])
