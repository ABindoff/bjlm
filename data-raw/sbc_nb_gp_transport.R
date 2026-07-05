# =============================================================================
# SBC (Talts et al. 2018) for the NB + latent-GP conditional-transport sampler,
# in the HIGH-COUNT regime (Fable's warning: a wrong fix looks calibrated on low
# counts). Model: NB outcome, one change-point, latent GP confounder on b0, no
# random intercept, no IPW. Transport is the default NB+GP path.
#
# Usage (after devtools::load_all()):
#   SBC_N=128 Rscript data-raw/sbc_nb_gp_transport.R
# Env: SBC_N reps (default 16), SBC_START first rep (default 1),
#      SBC_OUT ranks RDS path, SBC_ITER iterations (default 3000).
# =============================================================================
suppressMessages({ library(posterior) })
if (!exists("simulate_bjlm")) suppressMessages(devtools::load_all(quiet = TRUE))

N     <- as.integer(Sys.getenv("SBC_N", "16"))
START <- as.integer(Sys.getenv("SBC_START", "1"))
OUT   <- Sys.getenv("SBC_OUT", "sbc_nb_gp_transport_ranks.rds")
ITER  <- as.integer(Sys.getenv("SBC_ITER", "3000"))

# Transport is the default NB+GP path (BJLM_GP_NO_TRANSPORT=1 would force whitened).

# ---- priors (drawn == fitted) ----------------------------------------------
rtn <- function(m, s, lb, ub) qnorm(runif(1, pnorm(lb, m, s), pnorm(ub, m, s)), m, s)
.gt <- seq(0, 10, length.out = 8L)
.gap <- median(diff(.gt)); .rng <- max(.gt) - min(.gt)
RHO_LOC <- 0.5 * (log(.gap) + log(.rng))
RHO_SCALE <- max(0.35, 0.25 * log(.rng / .gap))
draw_prior <- function() list(
  b0 = rnorm(1, 2.5, 0.5),                            # high-count intercept (psi = ln mean)
  b0_gp = rnorm(1, 0, 1),
  b1 = rnorm(1, 0, 0.3), delta = rnorm(1, 0, 0.8),
  omega = rtn(5, 1.5, 0.5, 9.5), rho = rtn(4, 1.5, 1, 10),
  r = rgamma(1, shape = 2, scale = 5),                # NB dispersion, mean 10
  gp_alpha = exp(rnorm(1, 0, 1)),
  gp_rho = exp(rnorm(1, RHO_LOC, RHO_SCALE)),
  gp_sigma_x = exp(rnorm(1, -1, 1))
)
fit_priors <- bjlm_priors(outcome = smoothbp_priors(
  b0 = list("(Intercept)" = prior_normal(2.5, 0.5), "X_obs" = prior_normal(0, 1)),
  b1 = prior_normal(0, 0.3), deltas = prior_normal(0, 0.8),
  omega = prior_normal(5, 1.5, lb = 0.5, ub = 9.5), rho = prior_normal(4, 1.5, lb = 1, ub = 10),
  sigma = prior_invgamma(5, 2), sigma_u = prior_halfcauchy(0.5),
  r = prior_gamma(2, 5)))   # prior_gamma(shape, SCALE): mean 10, matches the draw above

pmap <- c("b0_(Intercept)"="b0", "b0_X_obs"="b0_gp", "b1_(Intercept)"="b1",
          "delta1_(Intercept)"="delta", "omega1_(Intercept)"="omega", "rho1_(Intercept)"="rho",
          "r"="r", "X_obs_alpha"="gp_alpha", "X_obs_rho"="gp_rho", "X_obs_sigma_x"="gp_sigma_x")

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

ns <- 20L; nt <- 8L; tau0 <- .gt
D <- as.matrix(dist(tau0))
sig <- function(x) 1 / (1 + exp(-x))

acc <- if (file.exists(OUT)) readRDS(OUT) else NULL
for (rep in START:(START + N - 1L)) {
  set.seed(4000 + rep)
  tp <- draw_prior()
  K <- tp$gp_alpha^2 * exp(-0.5 * (D / tp$gp_rho)^2) + diag(1e-8, nt)
  Lk <- t(chol(K))
  rows <- list()
  for (j in 1:ns) {
    f <- as.numeric(Lk %*% rnorm(nt))
    Xobs <- f + rnorm(nt, 0, tp$gp_sigma_x)
    d <- tau0 - tp$omega
    psi <- tp$b0 + tp$b0_gp * f + tp$b1 * d + tp$delta * d * sig(tp$rho * d)
    y <- rnbinom(nt, size = tp$r, mu = exp(psi))     # psi = ln(mean): bjlm convention
    rows[[j]] <- data.frame(subject = j, tau = tau0, y = y, X_obs = Xobs)
  }
  dat <- do.call(rbind, rows); dat$subject <- factor(dat$subject)
  if (any(!is.finite(dat$y))) { cat(sprintf("rep %d: bad sim, skipped\n", rep)); next }
  fit <- tryCatch(
    bjlm_model() |>
      outcome(y ~ tau, b0 = ~ 1 + X_obs, b1 = ~ 1, deltas = list(~1),
              omega = list(~1), rho = list(~1), data = dat, family = "negative_binomial") |>
      latent_gp(name = "X_obs", data = dat, time_var = "tau", obs_var = "X_obs",
                subject = "subject", time_out_var = "tau", time_trt_var = "tau") |>
      compile() |>
      fit(priors = fit_priors, chains = 2L, iter = ITER, warmup = ITER %/% 2L,
          seed = 4000 + rep, cores = 2L, step_om = 0.02, step_rho = 0.02),
    error = function(e) { cat(sprintf("rep %d: fit failed: %s\n", rep, conditionMessage(e))); NULL })
  if (is.null(fit)) next
  row <- list(rep = rep)
  for (vn in names(pmap))
    row[[pmap[[vn]]]] <- tryCatch(frank(fit$draws, vn, tp[[pmap[[vn]]]]), error = function(e) NA_real_)
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
