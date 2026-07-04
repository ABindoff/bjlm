# =============================================================================
# SBC (Talts et al. 2018) for the latent-GP change-point model.
# See data-raw/SBC_latent_gp_certificate.md for the model, protocol, and results.
#
# Usage (after devtools::load_all()):
#   SBC_N=128 Rscript data-raw/sbc_latent_gp.R
# Env: SBC_N reps (default 16), SBC_START first rep (default 1),
#      SBC_OUT ranks RDS path, SBC_ITER iterations (default 3000).
# Re-running appends to SBC_OUT; the uniformity table is printed at the end.
# =============================================================================
suppressMessages({ library(posterior) })
if (!exists("simulate_bjlm")) suppressMessages(devtools::load_all(quiet = TRUE))

N     <- as.integer(Sys.getenv("SBC_N", "16"))
START <- as.integer(Sys.getenv("SBC_START", "1"))
OUT   <- Sys.getenv("SBC_OUT", "sbc_latent_gp_ranks.rds")
ITER  <- as.integer(Sys.getenv("SBC_ITER", "3000"))

# ---- priors (drawn == fitted) ----------------------------------------------
rtn <- function(m, s, lb, ub) qnorm(runif(1, pnorm(lb, m, s), pnorm(ub, m, s)), m, s)
draw_prior <- function() list(
  b0 = rnorm(1, 0, 1), b1 = rnorm(1, 0, 0.5), delta = rnorm(1, 0, 0.8),
  omega = rtn(5, 1.5, 0.5, 9.5), rho = rtn(4, 1.5, 1, 10),
  sigma = 1 / sqrt(rgamma(1, shape = 5, rate = 2)),   # sigma^2 ~ IG(5,2)
  sigma_u = abs(rcauchy(1, 0, 0.5)),                  # half-Cauchy(0,0.5) on SD
  b0_gp = rnorm(1, 0, 1),
  gp_alpha = exp(rnorm(1, 0, 1)), gp_rho = exp(rnorm(1, 0, 1)),
  gp_sigma_x = exp(rnorm(1, -1, 1))                   # lognormal(-1,1)
)
fit_priors <- bjlm_priors(outcome = smoothbp_priors(
  b0 = prior_normal(0, 1), b1 = prior_normal(0, 0.5), deltas = prior_normal(0, 0.8),
  omega = prior_normal(5, 1.5, lb = 0.5, ub = 9.5), rho = prior_normal(4, 1.5, lb = 1, ub = 10),
  sigma = prior_invgamma(5, 2), sigma_u = prior_halfcauchy(0.5)))

pmap <- c("b0_(Intercept)"="b0", "b1_(Intercept)"="b1", "delta1_(Intercept)"="delta",
          "omega1_(Intercept)"="omega", "rho1_(Intercept)"="rho", "sigma"="sigma",
          "sigma_u"="sigma_u", "b0_X_obs"="b0_gp", "X_obs_alpha"="gp_alpha",
          "X_obs_rho"="gp_rho", "X_obs_sigma_x"="gp_sigma_x")

# ESS-thinned fractional rank (min of bulk/tail ESS = conservative thinning).
frank <- function(dr, var, truth) {
  d  <- as_draws_array(subset_draws(dr, variable = var))
  eb <- tryCatch(ess_bulk(d), error = function(e) NA_real_)
  et <- tryCatch(ess_tail(d), error = function(e) NA_real_)
  ess <- suppressWarnings(min(eb, et, na.rm = TRUE))
  v <- as.vector(d)
  if (!is.finite(ess) || ess < 2) ess <- length(v)
  vt <- v[seq(1L, length(v), by = max(1L, floor(length(v) / ess)))]
  mean(vt < truth)
}

acc <- if (file.exists(OUT)) readRDS(OUT) else NULL
for (rep in START:(START + N - 1L)) {
  set.seed(1000 + rep)
  tp <- draw_prior()
  dat <- tryCatch(simulate_bjlm(n_subj = 20L, n_obs = 8L, b0 = tp$b0, b0_trt = 0,
      b1 = tp$b1, omegas = tp$omega, rhos = tp$rho, deltas_int = tp$delta, deltas_trt = 0,
      sigma = tp$sigma, sigma_u = tp$sigma_u, gp_confounder = TRUE, gp_alpha = tp$gp_alpha,
      gp_rho = tp$gp_rho, gp_sigma_x = tp$gp_sigma_x, b0_gp = tp$b0_gp, trt_gp = 0,
      seed = 1000 + rep), error = function(e) NULL)
  if (is.null(dat)) { cat(sprintf("rep %d: sim failed\n", rep)); next }
  u_mean <- mean(attr(dat, "true_params")$u)
  fit <- tryCatch(
    bjlm_model() |>
      outcome(y ~ tau, b0 = ~ 1 + X_obs + (1 | subject), b1 = ~ 1,
              deltas = list(~ 1), omega = list(~ 1), rho = list(~ 1), data = dat) |>
      latent_gp(name = "X_obs", data = dat, time_var = "tau", obs_var = "X_obs",
                subject = "subject", time_out_var = "tau", time_trt_var = "tau") |>
      compile() |>
      fit(priors = fit_priors, chains = 2L, iter = ITER, warmup = ITER %/% 2L,
          seed = 1000 + rep, cores = 2L, step_om = 0.02, step_rho = 0.02, target_accept = 0.8),
    error = function(e) { cat(sprintf("rep %d: fit failed: %s\n", rep, conditionMessage(e))); NULL })
  if (is.null(fit)) next
  row <- list(rep = rep)
  for (vn in names(pmap)) {
    truth <- if (pmap[[vn]] == "b0") tp$b0 + u_mean else tp[[pmap[[vn]]]]
    row[[pmap[[vn]]]] <- tryCatch(frank(fit$draws, vn, truth), error = function(e) NA_real_)
  }
  acc <- rbind(acc, as.data.frame(row))
  saveRDS(acc, OUT)
  cat(sprintf("rep %d done (%d total)\n", rep, nrow(acc)))
}

# ---- uniformity table ------------------------------------------------------
params <- setdiff(names(acc), "rep"); B <- 10; M <- nrow(acc)
cat(sprintf("\nSBC uniformity: %d reps, %d-bin chi-square. Bonferroni p>%.4f passes.\n",
            M, B, 0.05 / length(params)))
for (p in params) {
  x <- acc[[p]][!is.na(acc[[p]])]; n <- length(x)
  h <- as.numeric(table(cut(x, breaks = seq(0, 1, length.out = B + 1), include.lowest = TRUE)))
  chisq <- sum((h - n / B)^2 / (n / B)); pval <- pchisq(chisq, df = B - 1, lower.tail = FALSE)
  cat(sprintf("  %-11s chisq %6.2f  p %.4f  %s\n", p, chisq, pval,
              if (pval > 0.05 / length(params)) "PASS" else "FLAG"))
}
