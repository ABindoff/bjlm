# =============================================================================
# Reporting for bjlm_sbc: turn SBC rank histograms into a plain-language verdict
# so a non-expert can read the result, plus the prior-sensitivity mode.
# =============================================================================

# Bare column names used inside ggplot2::aes() (evaluated in the data frame).
utils::globalVariables(c("rank", "width", "min_p", "p", "diff", "lo", "hi", "coverage"))

# Classify one rank vector's departure from uniform into a plain-English verdict.
# rank = fraction of posterior draws below the truth:
#   uniform         -> calibrated
#   U-shaped (mass at 0 & 1)  -> intervals too NARROW (overconfident)
#   n-shaped (mass in middle) -> intervals too WIDE (under-confident)
#   low mean rank   -> truth sits low in the posterior -> estimates biased HIGH
#   high mean rank  -> estimates biased LOW
.sbc_classify <- function(r, p_value) {
  r <- r[is.finite(r)]
  mean_r <- mean(r)
  lower <- mean(r < 0.1)                     # each tail expected ~0.1 under uniform
  upper <- mean(r > 0.9)
  if (p_value > 0.05)
    return(list(shape = "uniform", verdict = "well calibrated"))
  # Mass at BOTH ends -> too-narrow (overconfident); at ONE end -> bias; in the
  # MIDDLE -> too-wide (under-confident).
  if (lower > 0.18 && upper > 0.18)
    return(list(shape = "U", verdict = "intervals too NARROW (overconfident)"))
  if (lower > 0.18 && upper <= 0.18)
    return(list(shape = "left", verdict = "estimates biased HIGH (truth in lower tail)"))
  if (upper > 0.18 && lower <= 0.18)
    return(list(shape = "right", verdict = "estimates biased LOW (truth in upper tail)"))
  if (lower < 0.05 && upper < 0.05)
    return(list(shape = "n", verdict = "intervals too WIDE (under-confident)"))
  if (mean_r < 0.42)
    return(list(shape = "left", verdict = "estimates biased HIGH (truth in lower tail)"))
  if (mean_r > 0.58)
    return(list(shape = "right", verdict = "estimates biased LOW (truth in upper tail)"))
  list(shape = "other", verdict = "mild departure from uniform")
}

# Per-parameter uniformity p-value (binned chi-square) + verdict data frame.
.sbc_verdicts <- function(ranks) {
  n_kept <- nrow(ranks)
  B <- max(4L, min(10L, n_kept %/% 5L))
  params <- colnames(ranks)
  alpha_bonf <- 0.05 / length(params)
  rows <- lapply(params, function(p) {
    r <- ranks[, p]; r <- r[is.finite(r)]
    n <- length(r)
    if (n < 2) return(data.frame(parameter = p, n_used = n, mean_rank = NA_real_,
                                 p_value = NA_real_, calibrated = NA,
                                 verdict = "too few replicates", stringsAsFactors = FALSE))
    br <- seq(0, 1, length.out = B + 1L)
    o <- as.numeric(table(cut(r, br, include.lowest = TRUE)))
    e <- n / B
    chisq <- sum((o - e)^2 / e)
    pval <- stats::pchisq(chisq, df = B - 1L, lower.tail = FALSE)
    cls <- .sbc_classify(r, pval)
    data.frame(parameter = p, n_used = n, mean_rank = mean(r), p_value = pval,
               calibrated = pval > alpha_bonf, verdict = cls$verdict,
               stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}

#' Report a simulation-based calibration result
#'
#' Prints the discard/convergence summary and a per-quantity verdict table that
#' translates each rank histogram's shape into plain language (well calibrated,
#' intervals too narrow/wide, or biased). `summary()` returns that table as a
#' data frame.
#'
#' @param x,object A `bjlm_sbc` object from [sbc()].
#' @param ... Unused.
#' @return `print()` returns `x` invisibly; `summary()` returns the verdict
#'   data frame.
#' @seealso [sbc()], [plot.bjlm_sbc()].
#' @export
print.bjlm_sbc <- function(x, ...) {
  cat("Simulation-based calibration (bjlm)\n")
  cat(sprintf("  %d replicates requested, %d fit ok, %d discarded (%.0f%% failed/degenerate fits)\n",
              x$n_reps, x$n_kept, x$n_discard,
              100 * x$n_discard / max(1L, x$n_reps)))
  s <- x$settings
  cat(sprintf("  fit: %d chains x %d iter (warmup %d), family '%s'%s%s\n",
              s$chains, s$iter, s$warmup, s$family,
              if (s$n_breakpoints > 0) sprintf(", %d breakpoint(s)", s$n_breakpoints) else "",
              if (isTRUE(s$spike)) ", spike-and-slab" else ""))
  if (x$n_discard > 0.2 * x$n_reps)
    cat("  ! high fit-failure rate: many fits errored or produced degenerate data;\n",
        "    that is itself a model/sampler signal, not just a calibration result.\n", sep = "")
  if (!is.null(x$nonconv) && any(x$nonconv > 0))
    cat(sprintf("  note: per-quantity draws with Rhat > %.3f were dropped (see n_used);\n    a low n_used means that quantity is often non-identified on prior-drawn data.\n",
                s$rhat_threshold))
  cat("\n")
  v <- x$verdicts
  v$mean_rank <- round(v$mean_rank, 3)
  v$p_value <- signif(v$p_value, 3)
  mark <- ifelse(is.na(v$calibrated), "?", ifelse(v$calibrated, "OK", "FLAG"))
  w <- max(nchar(v$parameter))
  cat(sprintf("  %-*s  %-4s  %-6s  %-9s  %-8s  %s\n", w, "quantity", "flag", "n_used", "mean_rank", "p_value", "reading"))
  for (i in seq_len(nrow(v)))
    cat(sprintf("  %-*s  %-4s  %-6s  %-9s  %-8s  %s\n", w, v$parameter[i], mark[i],
                format(v$n_used[i]), format(v$mean_rank[i]), format(v$p_value[i]), v$verdict[i]))
  n_flag <- sum(!v$calibrated %in% TRUE & !is.na(v$calibrated))
  cat("\n")
  if (n_flag == 0)
    cat("  All quantities look calibrated (Bonferroni-adjusted).\n")
  else
    cat(sprintf("  %d quantit%s flagged. See plot(<result>) for the rank histograms.\n",
                n_flag, if (n_flag == 1) "y" else "ies"))
  invisible(x)
}

#' @rdname print.bjlm_sbc
#' @export
summary.bjlm_sbc <- function(object, ...) object$verdicts

#' Plot SBC rank histograms with a uniform envelope
#'
#' The default `"ecdf"` style plots, per quantity, the empirical CDF of the
#' fractional ranks minus the uniform CDF, inside a \strong{simultaneous}
#' confidence band (Säilynoja, Bürkner & Vehtari 2022). A calibrated quantity
#' stays entirely within the band; the ECDF-difference view is more sensitive
#' and free of the bin-width choice that the rank histogram (`style = "hist"`)
#' depends on. A curve leaving the band \emph{downward then upward} (an S) means
#' overconfident intervals, an inverted-S means under-confident, and a one-sided
#' excursion means bias.
#'
#' @param x A `bjlm_sbc` object.
#' @param style `"ecdf"` (default) for ECDF-difference panels with a
#'   simultaneous band, or `"hist"` for rank histograms.
#' @param conf Simultaneous coverage of the ECDF band. Default `0.95`.
#' @param ... Unused.
#' @return A `ggplot` object.
#' @references Säilynoja, T., Bürkner, P.-C., & Vehtari, A. (2022). Graphical
#'   test for discrete uniformity and its applications in goodness-of-fit
#'   evaluation and multiple sample comparison. \emph{Statistics and Computing}.
#' @export
plot.bjlm_sbc <- function(x, style = c("ecdf", "hist"), conf = 0.95, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("plot.bjlm_sbc requires the 'ggplot2' package.", call. = FALSE)
  style <- match.arg(style)
  ranks <- x$ranks
  n_kept <- nrow(ranks)

  if (style == "hist") {
    B <- max(4L, min(10L, n_kept %/% 5L))
    df <- data.frame(parameter = rep(colnames(ranks), each = n_kept),
                     rank = as.vector(ranks), stringsAsFactors = FALSE)
    df <- df[is.finite(df$rank), ]
    e <- n_kept / B
    lo <- stats::qbinom(0.005, n_kept, 1 / B); hi <- stats::qbinom(0.995, n_kept, 1 / B)
    return(
      ggplot2::ggplot(df, ggplot2::aes(x = rank)) +
        ggplot2::annotate("rect", xmin = 0, xmax = 1, ymin = lo, ymax = hi, fill = "grey85") +
        ggplot2::geom_hline(yintercept = e, colour = "grey55", linetype = 2) +
        ggplot2::geom_histogram(breaks = seq(0, 1, length.out = B + 1L),
                                fill = "#3B6EA5", colour = "white", linewidth = 0.2) +
        ggplot2::facet_wrap(~ parameter, scales = "free_y") +
        ggplot2::labs(title = "Simulation-based calibration",
                      subtitle = sprintf("%d replicates — flat within the grey band = calibrated", n_kept),
                      x = "posterior rank of the true value", y = "count") +
        ggplot2::theme_bw() +
        ggplot2::theme(strip.text = ggplot2::element_text(size = 8)))
  }

  # ECDF-difference panels with a per-N cached simultaneous band.
  params <- colnames(ranks)
  band_cache <- list()
  rows <- lapply(params, function(p) {
    r <- ranks[, p]; r <- r[is.finite(r)]
    N <- length(r)
    if (N < 2) return(NULL)
    key <- as.character(N)
    if (is.null(band_cache[[key]])) band_cache[[key]] <<- .sbc_ecdf_band(N, conf = conf)
    band <- band_cache[[key]]
    if (is.null(band)) return(NULL)
    d <- .sbc_ecdf_diff(r, band)
    d$parameter <- p
    d
  })
  dd <- do.call(rbind, Filter(Negate(is.null), rows))
  if (is.null(dd)) stop("Too few replicates to draw ECDF bands.", call. = FALSE)

  ggplot2::ggplot(dd, ggplot2::aes(x = p)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = lo, ymax = hi), fill = "grey80") +
    ggplot2::geom_hline(yintercept = 0, linetype = 2, colour = "grey50") +
    ggplot2::geom_step(ggplot2::aes(y = diff), colour = "#3B6EA5", linewidth = 0.6) +
    ggplot2::facet_wrap(~ parameter) +
    ggplot2::labs(
      title = "Simulation-based calibration (ECDF difference)",
      subtitle = sprintf("%d replicates — within the grey band (%.0f%% simultaneous) = calibrated",
                         n_kept, 100 * conf),
      x = "fractional rank", y = "ECDF - uniform") +
    ggplot2::theme_bw() +
    ggplot2::theme(strip.text = ggplot2::element_text(size = 8))
}

# Simultaneous confidence band for the ECDF of N i.i.d. Uniform(0,1) fractional
# ranks (Säilynoja, Bürkner & Vehtari 2022), by Monte-Carlo: find the pointwise
# level gamma whose two-sided envelope has `conf` SIMULTANEOUS coverage. Returns
# the band as ECDF proportions on a [0,1] grid. Seeded locally (and the RNG state
# is restored) so a plot is reproducible without disturbing the caller's stream.
.sbc_ecdf_band <- function(N, conf = 0.95, M = 2000L, K = 100L, seed = 20240714L) {
  if (N < 2) return(NULL)
  if (exists(".Random.seed", envir = .GlobalEnv)) {
    old <- get(".Random.seed", envir = .GlobalEnv)
    on.exit(assign(".Random.seed", old, envir = .GlobalEnv), add = TRUE)
  }
  set.seed(seed)
  grid <- seq(0, 1, length.out = K)
  null_counts <- function() {
    R <- matrix(stats::runif(N * M), nrow = M)              # M x N
    vapply(grid, function(p) rowSums(R <= p), numeric(M))   # M x K
  }
  cal <- null_counts(); ev <- null_counts()
  cover_at <- function(gamma) {
    lo <- apply(cal, 2, stats::quantile, probs = gamma / 2,     type = 1)
    hi <- apply(cal, 2, stats::quantile, probs = 1 - gamma / 2, type = 1)
    inside <- rowMeans((ev >= matrix(lo, nrow(ev), length(lo), byrow = TRUE)) &
                       (ev <= matrix(hi, nrow(ev), length(hi), byrow = TRUE))) == 1
    list(cov = mean(inside), lo = lo, hi = hi)
  }
  g_lo <- 1e-4; g_hi <- 0.5
  for (it in 1:40) { g <- (g_lo + g_hi) / 2; if (cover_at(g)$cov >= conf) g_lo <- g else g_hi <- g }
  b <- cover_at(g_lo)
  data.frame(p = grid, lo = b$lo / N, hi = b$hi / N)
}

# ECDF-of-ranks minus uniform, with the band expressed as difference from uniform.
.sbc_ecdf_diff <- function(ranks, band) {
  ranks <- ranks[is.finite(ranks)]
  obs <- vapply(band$p, function(p) mean(ranks <= p), numeric(1))
  data.frame(p = band$p, diff = obs - band$p, lo = band$lo - band$p, hi = band$hi - band$p)
}

# ---- prior-sensitivity mode -------------------------------------------------

#' Prior-sensitivity sweep over the GP lengthscale
#'
#' Two related questions about how the GP lengthscale prior affects the
#' trend-versus-GP decomposition, selected by whether you supply `gen_priors`.
#'
#' \strong{Sampler-robustness mode (default, `gen_priors = NULL`).} Runs a
#' self-consistent [sbc()] at each prior width (truth drawn from the *same*
#' prior used to fit) and reports the slope (`delta`) uniformity p-value. This
#' checks whether the *sampler* stays calibrated as the prior geometry changes.
#' Because draw == fit, an exact sampler is uniform at \emph{every} width, so a
#' dip here flags where the GP-hyperparameter geometry trips the sampler, not
#' where the GP "absorbs the trend"; the curve is typically non-monotonic and
#' should be read as a sampler-difficulty map, not an identifiability threshold.
#'
#' \strong{Coverage mode (supply `gen_priors`).} Answers the identifiability
#' question directly. Data are generated from a \emph{fixed} reality
#' (`gen_priors` for the coefficients and a fixed GP lengthscale `gen_lengthscale`),
#' then \emph{fit} with an increasingly wide GP lengthscale prior. It reports the
#' coverage of the true slope change by its posterior interval. As the fit prior
#' widens, the GP gains the freedom to absorb the forced trend and the slope's
#' coverage falls below nominal, giving a concrete threshold for how much
#' lengthscale prior information your attribution needs. This deliberately breaks
#' draw == fit, so coverage (not rank uniformity) is the honest metric.
#'
#' @param object A `bjlm_compiled_model` or `bjlm_fit` with at least one latent GP.
#' @param grid Numeric vector of lengthscale-prior widths (`sdlog` of a
#'   log-normal centred at the resolution-aware location). Larger = more diffuse.
#' @param priors The \emph{fitting} priors (your real analysis priors) whose GP
#'   lengthscale is swept. Defaults to a fit's own `priors_used`, else
#'   [bjlm_priors()].
#' @param gp_name Which GP to vary (defaults to the first).
#' @param target Regex selecting the target coefficient(s): the uniformity
#'   functionals (default mode) or the coverage columns (coverage mode). Default
#'   `"^delta"` (the slope differences).
#' @param gen_priors Coverage mode only: a [bjlm_priors()] bundle defining the
#'   fixed data-generating reality. It is used only to \emph{generate} data (the
#'   fit still uses `priors`); supplying it switches the function into coverage
#'   mode. Make it informative (e.g. a real, non-zero slope change) so there is a
#'   true trend the GP could steal.
#' @param gen_lengthscale Coverage mode only: the true GP lengthscale used to
#'   generate data (held fixed across the sweep). Defaults to the resolution-aware
#'   prior median.
#' @param gen_amplitude,gen_sigma_x Coverage mode only: the true GP marginal SD
#'   and observation-noise SD used to generate data. Pin these to a realistic
#'   signal-to-noise so the forced trend is not swamped by natural variability;
#'   if `NULL`, they are drawn from the model's GP priors (which can be too
#'   diffuse to leave the slope identifiable).
#' @param level Coverage mode only: nominal central posterior-interval level.
#'   Default `0.9`.
#' @param rhat_threshold Convergence gate applied per target coefficient.
#'   Default `1.05`.
#' @param reps,iter,warmup,chains,cores,seed Replicates and MCMC controls at each
#'   width. `warmup` defaults to `iter / 2`.
#' @param ... Passed to [sbc()] (default mode) or [fit()] (coverage mode).
#'
#' @return A `bjlm_sbc_sensitivity` object; `print()` and `plot()` show either
#'   the uniformity p-value or the coverage against prior width.
#' @seealso [sbc()].
#' @export
sbc_prior_sensitivity <- function(object, grid, priors = NULL, gp_name = NULL,
                                   target = "^delta", gen_priors = NULL,
                                   gen_lengthscale = NULL, gen_amplitude = NULL,
                                   gen_sigma_x = NULL, level = 0.9,
                                   rhat_threshold = 1.05, reps = 50L,
                                   iter = 1000L, warmup = NULL, chains = 2L,
                                   cores = 1L, seed = 1L, ...) {
  cm <- if (inherits(object, "bjlm_fit")) object$compiled_model else object
  if (!inherits(cm, "bjlm_compiled_model"))
    stop("`object` must be a bjlm_compiled_model or bjlm_fit.", call. = FALSE)
  gps <- cm$model$latent_gps
  if (is.null(gps) || length(gps) == 0)
    stop("sbc_prior_sensitivity() needs a model with at least one latent GP.", call. = FALSE)
  gi <- if (is.null(gp_name)) 1L else match(gp_name, vapply(gps, function(g) g$name, character(1)))
  if (is.na(gi)) stop(sprintf("No latent GP named '%s'.", gp_name), call. = FALSE)

  info <- .sbc_gp_info(cm)[[gi]]
  loc <- .sbc_lengthscale_params(info$time_grid)$loc
  fit_priors <- priors %||% (if (inherits(object, "bjlm_fit")) object$priors_used else NULL)
  if (is.null(warmup)) warmup <- iter %/% 2L

  if (!is.null(gen_priors)) {
    if (is.null(fit_priors)) fit_priors <- bjlm_priors()
    return(.sbc_coverage_sweep(cm, gi, info, loc, grid, target, fit_priors, gen_priors,
                               gen_lengthscale, gen_amplitude, gen_sigma_x,
                               level, rhat_threshold,
                               reps, iter, warmup, chains, cores, seed, ...))
  }

  base_priors <- fit_priors
  base_spike  <- if (inherits(object, "bjlm_fit")) object$spike else NULL

  rows <- list()
  for (w in grid) {
    cm_w <- .sbc_set_gp_lengthscale(cm, gi, prior_lognormal(meanlog = loc, sdlog = w))
    message(sprintf("sbc_prior_sensitivity: lengthscale width sdlog = %.3g", w))
    res <- tryCatch(
      sbc.bjlm_compiled_model(cm_w, priors = base_priors, spike = base_spike,
                              reps = reps, iter = iter, warmup = warmup, chains = chains,
                              cores = cores, seed = seed, verbose = FALSE, ...),
      error = function(e) { message("  failed: ", conditionMessage(e)); NULL })
    if (is.null(res)) next
    v <- res$verdicts
    tv <- v[grepl(target, v$parameter), , drop = FALSE]
    finite_p <- tv$p_value[is.finite(tv$p_value)]
    rows[[length(rows) + 1L]] <- data.frame(
      width = w,
      min_p = if (length(finite_p)) min(finite_p) else NA_real_,
      n_flagged = sum(tv$calibrated %in% FALSE),
      n_kept = res$n_kept,
      stringsAsFactors = FALSE)
  }
  if (length(rows) == 0) stop("All grid points failed.", call. = FALSE)
  structure(list(mode = "uniformity", table = do.call(rbind, rows),
                 target = target, gp_name = info$name, loc = loc),
            class = "bjlm_sbc_sensitivity")
}

# Set one of a GP's hyperpriors (alpha / rho / sigma_x) on a compiled model.
.sbc_set_gp_hyperprior <- function(cm, gi, which, prior) {
  gp_pr <- cm$model$latent_gps[[gi]]$priors %||% gp_priors()
  gp_pr[[which]] <- prior
  cm$model$latent_gps[[gi]]$priors <- gp_pr
  cm
}
.sbc_set_gp_lengthscale <- function(cm, gi, prior) .sbc_set_gp_hyperprior(cm, gi, "rho", prior)
# A near-degenerate log-normal that pins a positive hyperparameter at `value`.
.sbc_fixed_prior <- function(value) prior_lognormal(log(value), 1e-8)

# Coverage mode: generate from a fixed reality (gen_priors), fit with the user's
# analysis priors (fit_priors) under a widening GP lengthscale, and report the
# target coefficient's interval coverage.
.sbc_coverage_sweep <- function(cm, gi, info, loc, grid, target, fit_priors, gen_priors,
                                gen_lengthscale, gen_amplitude, gen_sigma_x,
                                level, rhat_threshold,
                                reps, iter, warmup, chains, cores, seed, ...) {
  if (!inherits(gen_priors, "bjlm_priors"))
    stop("`gen_priors` must be a bjlm_priors() bundle.", call. = FALSE)
  gen_L <- gen_lengthscale %||% exp(loc)
  # Generation model: GP lengthscale pinned at the true value (near-degenerate
  # log-normal); the amplitude and obs-noise are pinned too when supplied, so the
  # forced signal is not swamped by natural variability of an uncontrolled scale.
  # Everything else is drawn from gen_priors.
  cm_gen <- .sbc_set_gp_lengthscale(cm, gi, .sbc_fixed_prior(gen_L))
  if (!is.null(gen_amplitude)) cm_gen <- .sbc_set_gp_hyperprior(cm_gen, gi, "alpha", .sbc_fixed_prior(gen_amplitude))
  if (!is.null(gen_sigma_x)) cm_gen <- .sbc_set_gp_hyperprior(cm_gen, gi, "sigma_x", .sbc_fixed_prior(gen_sigma_x))
  gp_info_gen <- .sbc_gp_info(cm_gen)
  gen_outcome <- gen_priors$outcome %||% smoothbp_priors()
  alpha <- (1 - level) / 2

  rows <- list()
  for (w in grid) {
    cm_fit <- .sbc_set_gp_lengthscale(cm, gi, prior_lognormal(loc, w))
    message(sprintf("sbc_prior_sensitivity (coverage): lengthscale width sdlog = %.3g", w))
    covered <- logical(0); biases <- numeric(0); nok <- 0L
    for (rep in seq_len(reps)) {
      set.seed(seed + rep)                       # identical data across widths (paired)
      draw <- .sbc_draw_prior(cm_gen, gen_outcome, NULL, gp_info_gen)
      sim  <- tryCatch(.sbc_simulate(cm_gen, draw, gp_info_gen), error = function(e) NULL)
      if (is.null(sim)) next
      cm_rep <- cm_fit
      cm_rep$model$outcome$data <- sim
      for (i in seq_along(cm_rep$model$latent_gps)) cm_rep$model$latent_gps[[i]]$data <- sim
      fit <- tryCatch(
        fit.bjlm_compiled_model(cm_rep, priors = fit_priors, chains = chains,
            iter = iter, warmup = warmup, cores = cores, seed = seed + rep,
            verbose = FALSE, ...),
        error = function(e) NULL)
      if (is.null(fit)) next
      vn <- posterior::variables(fit$draws)
      tcols <- grep(target, vn, value = TRUE)
      tcols <- tcols[tcols %in% names(draw$theta)]
      if (length(tcols) == 0) next
      any_used <- FALSE
      for (tc in tcols) {
        sub <- posterior::subset_draws(fit$draws, variable = tc)
        rh <- tryCatch(posterior::rhat(sub), error = function(e) NA_real_)
        if (is.finite(rh) && rh > rhat_threshold) next
        dd <- as.numeric(posterior::as_draws_matrix(sub)[, tc])
        q <- stats::quantile(dd, c(alpha, 1 - alpha), names = FALSE)
        td <- draw$theta[[tc]]
        covered <- c(covered, td >= q[1] && td <= q[2])
        biases  <- c(biases, mean(dd) - td)
        any_used <- TRUE
      }
      if (any_used) nok <- nok + 1L
    }
    rows[[length(rows) + 1L]] <- data.frame(
      width = w,
      coverage = if (length(covered)) mean(covered) else NA_real_,
      bias = if (length(biases)) mean(biases) else NA_real_,
      n_kept = nok, stringsAsFactors = FALSE)
  }
  if (length(rows) == 0) stop("All grid points failed.", call. = FALSE)
  structure(list(mode = "coverage", table = do.call(rbind, rows), target = target,
                 gp_name = info$name, loc = loc, level = level, gen_lengthscale = gen_L),
            class = "bjlm_sbc_sensitivity")
}

#' @export
print.bjlm_sbc_sensitivity <- function(x, ...) {
  if (identical(x$mode, "coverage")) {
    cat(sprintf("SBC prior sensitivity (coverage) — GP '%s' lengthscale width vs '%s' recovery\n",
                x$gp_name, x$target))
    cat(sprintf("  generated at fixed lengthscale %.3g; nominal interval level %.0f%%\n",
                x$gen_lengthscale, 100 * x$level))
    tt <- x$table
    tt$coverage <- round(tt$coverage, 3); tt$bias <- signif(tt$bias, 3)
    print(tt, row.names = FALSE)
    # Materially under-covered: below nominal by more than ~2 binomial SEs.
    se <- sqrt(x$level * (1 - x$level) / pmax(1L, x$table$n_kept))
    broke <- x$table$width[which(x$table$coverage < x$level - 2 * se)]
    if (length(broke))
      cat(sprintf("\n  Slope coverage first drops materially below %.0f%% at width >= %.3g\n  (the GP begins absorbing the forced trend).\n",
                  100 * x$level, min(broke)))
    else
      cat(sprintf("\n  Slope coverage holds near %.0f%% across the grid.\n", 100 * x$level))
    return(invisible(x))
  }
  cat(sprintf("SBC prior sensitivity (sampler robustness) — GP '%s' lengthscale width vs '%s' uniformity\n",
              x$gp_name, x$target))
  tt <- x$table
  tt$min_p <- signif(tt$min_p, 3)
  print(tt, row.names = FALSE)
  broke <- tt$width[which(tt$min_p < 0.05)]
  if (length(broke))
    cat(sprintf("\n  Slope uniformity first dips below p = 0.05 at width = %.3g.\n  (A sampler-difficulty signal; may be non-monotonic. For an identifiability\n   threshold, re-run in coverage mode with gen_priors.)\n", min(broke)))
  else
    cat("\n  Slope uniformity holds across the grid.\n")
  invisible(x)
}

#' @export
plot.bjlm_sbc_sensitivity <- function(x, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("plot.bjlm_sbc_sensitivity requires the 'ggplot2' package.", call. = FALSE)
  if (identical(x$mode, "coverage")) {
    return(
      ggplot2::ggplot(x$table, ggplot2::aes(x = width, y = coverage)) +
        ggplot2::geom_hline(yintercept = x$level, colour = "#cc3333", linetype = 2) +
        ggplot2::geom_line(colour = "#3B6EA5") +
        ggplot2::geom_point(size = 2) +
        ggplot2::coord_cartesian(ylim = c(0, 1)) +
        ggplot2::labs(
          title = sprintf("Slope recovery vs GP lengthscale prior width (GP '%s')", x$gp_name),
          subtitle = sprintf("as the prior widens below the red line (nominal %.0f%%), the GP absorbs the forced trend",
                             100 * x$level),
          x = "GP lengthscale prior width (sdlog)",
          y = sprintf("%.0f%% interval coverage of '%s'", 100 * x$level, x$target)) +
        ggplot2::theme_bw())
  }
  ggplot2::ggplot(x$table, ggplot2::aes(x = width, y = min_p)) +
    ggplot2::geom_hline(yintercept = 0.05, colour = "#cc3333", linetype = 2) +
    ggplot2::geom_line(colour = "#3B6EA5") +
    ggplot2::geom_point(size = 2) +
    ggplot2::labs(
      title = sprintf("Sampler robustness vs GP lengthscale prior width (GP '%s')", x$gp_name),
      subtitle = "self-consistent SBC uniformity; a dip flags sampler difficulty, not identifiability",
      x = "GP lengthscale prior width (sdlog)",
      y = sprintf("min uniformity p-value over '%s'", x$target)) +
    ggplot2::theme_bw()
}
