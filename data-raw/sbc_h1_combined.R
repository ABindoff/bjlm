## Combined outcome-block SBC (Gate H1): H1a (half-Cauchy sigma_u + ASIS) AND
## H1b (Rao-Blackwell spike-slab) together. Draws the change-point coefficients
## with spike-slab indicators, plus sigma, sigma_u, u, omega, rho from their
## priors; simulates; fits WITH spike-slab on; checks rank uniformity of the
## continuous parameters (the interaction of the funnel fix with active
## selection is exactly what this certifies). Talts et al. (2018) protocol.
##
## Run: Rscript data-raw/sbc_h1_combined.R

suppressMessages({ library(posterior) })
devtools::load_all("C:/Users/bindoffa/antigravity_projects/bjlm", quiet = TRUE)

N_SBC <- 200L; N_SUBJ <- 25L; N_OBS <- 6L
ITER <- 1500L; WARMUP <- 800L; N_BINS <- 16L
SLAB_SD <- 2.0; PI <- 0.5

# Priors used for BOTH generation and fitting (omega/rho tightened to stay in
# the tau range [0,6] so the change point is identifiable).
om_prior  <- list(m = 3, s = 1)      # N(3,1) lb 0
rho_prior <- list(m = 4, s = 1)      # N(4,1) lb 0
rtnorm_pos <- function(m, s) { repeat { x <- rnorm(1, m, s); if (x > 0.05) return(x) } }

pri <- bjlm_priors(outcome = smoothbp_priors(
  b0 = prior_normal(0, 3), b1 = prior_normal(0, SLAB_SD),
  deltas = prior_normal(0, SLAB_SD),
  omega = prior_normal(om_prior$m, om_prior$s, lb = 0),
  rho   = prior_normal(rho_prior$m, rho_prior$s, lb = 0),
  sigma = prior_invgamma(3, 2), sigma_u = prior_halfcauchy(1)))

par_names <- c("b0_(Intercept)", "sigma", "sigma_u")
frac_rank <- function(d1, truth) {
  ess <- tryCatch(max(1L, floor(ess_bulk(d1))), error = function(e) 100L)
  v <- as.vector(as_draws_matrix(d1)); L <- length(v); thin <- max(1L, floor(L / ess))
  vt <- v[seq(1L, L, by = thin)]; sum(vt < truth) / length(vt)
}
sig <- function(x) 1 / (1 + exp(-x))

ranks <- matrix(NA_real_, N_SBC, length(par_names), dimnames = list(NULL, par_names))
n_ok <- 0L
for (s in seq_len(N_SBC)) {
  set.seed(31415L + s)
  sigma_u <- min(abs(rcauchy(1, 0, 1)), 8)          # half-Cauchy(0,1), guard extreme
  sigma   <- 1 / sqrt(rgamma(1, 3, rate = 2))
  b0 <- rnorm(1, 0, 3)
  gb1 <- rbinom(1, 1, PI); bb1 <- gb1 * rnorm(1, 0, SLAB_SD)
  gdi <- rbinom(1, 1, PI); bdi <- gdi * rnorm(1, 0, SLAB_SD)
  gdt <- rbinom(1, 1, PI); bdt <- gdt * rnorm(1, 0, SLAB_SD)
  omega <- rtnorm_pos(om_prior$m, om_prior$s); rho <- rtnorm_pos(rho_prior$m, rho_prior$s)
  u <- rnorm(N_SUBJ, 0, sigma_u); Trt <- rbinom(N_SUBJ, 1, 0.5)
  subj <- rep(seq_len(N_SUBJ), each = N_OBS)
  rows <- list()
  for (j in seq_len(N_SUBJ)) {
    tau <- seq(0, 6, length.out = N_OBS); d <- tau - omega
    delta_j <- bdi + bdt * Trt[j]
    mu <- b0 + u[j] + bb1 * d + delta_j * d * sig(rho * d)
    rows[[j]] <- data.frame(subject = j, tau = tau, y = mu + rnorm(N_OBS, 0, sigma), Trt = Trt[j])
  }
  dat <- do.call(rbind, rows); dat$subject <- factor(dat$subject)

  fit <- tryCatch(
    bjlm_model() |>
      outcome(y ~ tau, b0 = ~ 1 + (1 | subject), b1 = ~ 1,
              deltas = list(~ 1 + Trt), omega = list(~ 1), rho = list(~ 1), data = dat) |>
      compile() |>
      fit(priors = pri, spike = prior_spike_slab(pi = PI, slab = prior_normal(0, SLAB_SD)),
          chains = 1L, iter = ITER, warmup = WARMUP, seed = s),
    error = function(e) NULL)
  if (is.null(fit) || !all(is.finite(as.array(fit$draws)))) next
  dr <- fit$draws
  ranks[s, "b0_(Intercept)"] <- frac_rank(subset_draws(dr, variable = "b0_(Intercept)"), b0 + mean(u))
  ranks[s, "sigma"]          <- frac_rank(subset_draws(dr, variable = "sigma"), sigma)
  ranks[s, "sigma_u"]        <- frac_rank(subset_draws(dr, variable = "sigma_u"), sigma_u)
  n_ok <- n_ok + 1L
  if (s %% 25L == 0L) cat(sprintf("  %d/%d (%d valid)\n", s, N_SBC, n_ok))
}

chisq_unif <- function(r) {
  r <- r[is.finite(r)]; br <- seq(0, 1, length.out = N_BINS + 1L)
  o <- as.vector(table(cut(r, br, include.lowest = TRUE))); e <- length(r) / N_BINS
  pchisq(sum((o - e)^2 / e), N_BINS - 1L, lower.tail = FALSE)
}
cat(sprintf("\nCombined outcome-block SBC (spike-slab ON): %d valid replicates\n", n_ok))
res <- data.frame(parameter = par_names,
  mean_rank = round(colMeans(ranks, na.rm = TRUE), 3),
  pvalue = round(vapply(par_names, function(p) chisq_unif(ranks[, p]), numeric(1)), 3))
res$pass <- res$pvalue > 0.01
print(res, row.names = FALSE)
cat(sprintf("\nsigma_u under active selection: p = %.3f -> %s\n",
            res$pvalue[res$parameter == "sigma_u"],
            if (res$pvalue[res$parameter == "sigma_u"] > 0.01) "PASS" else "FAIL"))
