## SBC certificate for H1a: half-Cauchy sigma_u prior + ASIS interweave on the
## random-effects funnel. Random-intercept Gaussian model (the cleanest exercise
## of sample_random_effects_weighted + sample_re_ancillary_weighted + the
## half-Cauchy-aux sigma_u update). Draws all parameters from their priors,
## simulates, fits, and checks rank uniformity (ESS-thinned fractional ranks,
## chi-squared GOF) per the Talts et al. (2018) protocol.
##
## Run (loads the freshly-built package): Rscript data-raw/sbc_h1a_random_effects.R

suppressMessages({ library(posterior) })
devtools::load_all("C:/Users/bindoffa/antigravity_projects/bjlm", quiet = TRUE)

N_SBC   <- 256L
N_SUBJ  <- 25L
N_OBS   <- 6L
ITER    <- 1500L
WARMUP  <- 750L
N_BINS  <- 16L
A_HC    <- 1.0          # half-Cauchy(0, A) scale for sigma_u (== prior scale)

# Priors used for BOTH generation and fitting (so SBC is valid).
pri <- bjlm_priors(outcome = smoothbp_priors(
  b0      = prior_normal(0, 3),
  sigma   = prior_invgamma(3, 2),      # sigma^2 ~ IG(3,2): well-identified residual SD
  sigma_u = prior_invgamma(1, A_HC)    # scale = A -> half-Cauchy(0, A) under new code
))

par_names <- c("b0_(Intercept)", "sigma", "sigma_u")

frac_rank <- function(draws_1d, truth) {
  # ESS-thin to approx-independent draws, return fractional rank in [0,1].
  ess <- tryCatch(max(1L, floor(ess_bulk(draws_1d))), error = function(e) length(draws_1d))
  v   <- as.vector(as_draws_matrix(draws_1d))
  L   <- length(v); thin <- max(1L, floor(L / ess))
  vt  <- v[seq(1L, L, by = thin)]
  sum(vt < truth) / length(vt)
}

ranks <- matrix(NA_real_, N_SBC, length(par_names), dimnames = list(NULL, par_names))
n_ok <- 0L
for (s in seq_len(N_SBC)) {
  set.seed(20260704L + s)
  sigma_u <- abs(rcauchy(1L, 0, A_HC))              # half-Cauchy(0, A)
  sigma   <- 1 / sqrt(rgamma(1L, shape = 3, rate = 2))  # sigma^2 ~ IG(3,2)
  b0      <- rnorm(1L, 0, 3)
  u       <- rnorm(N_SUBJ, 0, sigma_u)
  subj    <- rep(seq_len(N_SUBJ), each = N_OBS)
  y       <- b0 + u[subj] + rnorm(N_SUBJ * N_OBS, 0, sigma)
  dat     <- data.frame(subject = factor(subj), y = y)

  fit <- tryCatch(
    bjlm_model() |>
      outcome(y ~ 1 + (1 | subject), data = dat, family = gaussian()) |>
      compile() |>
      fit(priors = pri, chains = 1L, iter = ITER, warmup = WARMUP, seed = s),
    error = function(e) NULL
  )
  if (is.null(fit)) next
  dr <- fit$draws
  ok <- all(is.finite(as.array(dr)))
  if (!ok) next
  # The sampler imposes sum-to-zero on u (b0 absorbs the RE mean), so the
  # identifiable intercept it targets is b0 + mean(u). Rank against that.
  ranks[s, "b0_(Intercept)"] <- frac_rank(subset_draws(dr, variable = "b0_(Intercept)"), b0 + mean(u))
  ranks[s, "sigma"]          <- frac_rank(subset_draws(dr, variable = "sigma"), sigma)
  ranks[s, "sigma_u"]        <- frac_rank(subset_draws(dr, variable = "sigma_u"), sigma_u)
  n_ok <- n_ok + 1L
  if (s %% 32L == 0L) cat(sprintf("  %d/%d done (%d valid)\n", s, N_SBC, n_ok))
}

chisq_unif <- function(r) {
  r  <- r[is.finite(r)]
  br <- seq(0, 1, length.out = N_BINS + 1L)
  o  <- as.vector(table(cut(r, br, include.lowest = TRUE)))
  e  <- length(r) / N_BINS
  stat <- sum((o - e)^2 / e)
  pchisq(stat, N_BINS - 1L, lower.tail = FALSE)
}

cat(sprintf("\nH1a SBC: %d valid replicates (random-intercept Gaussian model)\n", n_ok))
res <- data.frame(
  parameter = par_names,
  mean_rank = round(colMeans(ranks, na.rm = TRUE), 3),
  pvalue    = round(vapply(par_names, function(p) chisq_unif(ranks[, p]), numeric(1)), 3)
)
res$pass <- res$pvalue > 0.01
print(res, row.names = FALSE)
cat(sprintf("\nsigma_u SBC (the H1a target): p = %.3f -> %s\n",
            res$pvalue[res$parameter == "sigma_u"],
            if (res$pvalue[res$parameter == "sigma_u"] > 0.01) "PASS" else "FAIL"))
saveRDS(ranks, "data-raw/sbc_h1a_ranks.rds")
