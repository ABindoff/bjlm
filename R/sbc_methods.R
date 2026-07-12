# =============================================================================
# Reporting for bjlm_sbc: turn SBC rank histograms into a plain-language verdict
# so a non-expert can read the result, plus the prior-sensitivity mode.
# =============================================================================

# Bare column names used inside ggplot2::aes() (evaluated in the data frame).
utils::globalVariables(c("rank", "width", "min_p"))

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
#' One panel per calibrated quantity showing the rank histogram; a well-
#' calibrated quantity is flat within the shaded band (the ~99% interval of bin
#' counts expected under uniformity). Departures translate to the verdicts in
#' [print.bjlm_sbc]: a U-shape means overconfident intervals, a hump means
#' under-confident intervals, and a slope means bias.
#'
#' @param x A `bjlm_sbc` object.
#' @param ... Unused.
#' @return A `ggplot` object.
#' @export
plot.bjlm_sbc <- function(x, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("plot.bjlm_sbc requires the 'ggplot2' package.", call. = FALSE)
  ranks <- x$ranks
  n_kept <- nrow(ranks)
  B <- max(4L, min(10L, n_kept %/% 5L))
  df <- data.frame(
    parameter = rep(colnames(ranks), each = n_kept),
    rank = as.vector(ranks),
    stringsAsFactors = FALSE
  )
  df <- df[is.finite(df$rank), ]
  # Uniform envelope: bin count ~ Binomial(n_kept, 1/B).
  e <- n_kept / B
  lo <- stats::qbinom(0.005, n_kept, 1 / B)
  hi <- stats::qbinom(0.995, n_kept, 1 / B)
  ggplot2::ggplot(df, ggplot2::aes(x = rank)) +
    ggplot2::annotate("rect", xmin = 0, xmax = 1, ymin = lo, ymax = hi, fill = "grey85") +
    ggplot2::geom_hline(yintercept = e, colour = "grey55", linetype = 2) +
    ggplot2::geom_histogram(breaks = seq(0, 1, length.out = B + 1L),
                            fill = "#3B6EA5", colour = "white", linewidth = 0.2) +
    ggplot2::facet_wrap(~ parameter, scales = "free_y") +
    ggplot2::labs(
      title = "Simulation-based calibration",
      subtitle = sprintf("%d replicates — flat within the grey band = calibrated", n_kept),
      x = "posterior rank of the true value", y = "count") +
    ggplot2::theme_bw() +
    ggplot2::theme(strip.text = ggplot2::element_text(size = 8))
}

# ---- prior-sensitivity mode -------------------------------------------------

#' SBC prior-sensitivity sweep for the GP lengthscale
#'
#' Answers "how much prior information does my trend-versus-GP decomposition
#' require?" by running [sbc()] over a grid of GP lengthscale prior widths and
#' reporting where the slope (`delta`) calibration breaks. A wide lengthscale
#' prior lets the GP absorb genuine trend, degrading the calibration of the
#' slope-difference functionals; the sweep shows the width at which that happens.
#'
#' @param object A `bjlm_compiled_model` or `bjlm_fit` with at least one latent GP.
#' @param grid Numeric vector of lengthscale-prior widths (`sdlog` of a
#'   log-normal centred at the resolution-aware location). Larger = more diffuse.
#' @param gp_name Which GP to vary (defaults to the first).
#' @param target Regex selecting the functionals to summarise. Default `"^delta"`
#'   (the slope differences).
#' @param reps,iter,chains,cores,seed,... Passed to [sbc()] at each grid point.
#'
#' @return A `bjlm_sbc_sensitivity` object; `print()` and `plot()` show the
#'   minimum slope-calibration p-value against prior width.
#' @seealso [sbc()].
#' @export
sbc_prior_sensitivity <- function(object, grid, gp_name = NULL,
                                   target = "^delta", reps = 50L,
                                   iter = 1000L, chains = 2L, cores = 1L,
                                   seed = 1L, ...) {
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
  base_priors <- if (inherits(object, "bjlm_fit")) object$priors_used else NULL
  base_spike  <- if (inherits(object, "bjlm_fit")) object$spike else NULL

  rows <- list()
  for (w in grid) {
    cm_w <- cm
    gp_pr <- cm_w$model$latent_gps[[gi]]$priors %||% gp_priors()
    gp_pr$rho <- prior_lognormal(meanlog = loc, sdlog = w)
    cm_w$model$latent_gps[[gi]]$priors <- gp_pr
    message(sprintf("sbc_prior_sensitivity: lengthscale width sdlog = %.3g", w))
    res <- tryCatch(
      sbc.bjlm_compiled_model(cm_w, priors = base_priors, spike = base_spike,
                              reps = reps, iter = iter, chains = chains,
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
  structure(list(table = do.call(rbind, rows), target = target,
                 gp_name = info$name, loc = loc),
            class = "bjlm_sbc_sensitivity")
}

#' @export
print.bjlm_sbc_sensitivity <- function(x, ...) {
  cat(sprintf("SBC prior sensitivity — GP '%s' lengthscale width vs '%s' calibration\n",
              x$gp_name, x$target))
  tt <- x$table
  tt$min_p <- signif(tt$min_p, 3)
  print(tt, row.names = FALSE)
  broke <- tt$width[which(tt$min_p < 0.05)]
  if (length(broke))
    cat(sprintf("\n  Slope calibration first breaks (p < 0.05) at width >= %.3g.\n", min(broke)))
  else
    cat("\n  Slope calibration holds across the grid.\n")
  invisible(x)
}

#' @export
plot.bjlm_sbc_sensitivity <- function(x, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("plot.bjlm_sbc_sensitivity requires the 'ggplot2' package.", call. = FALSE)
  ggplot2::ggplot(x$table, ggplot2::aes(x = width, y = min_p)) +
    ggplot2::geom_hline(yintercept = 0.05, colour = "#cc3333", linetype = 2) +
    ggplot2::geom_line(colour = "#3B6EA5") +
    ggplot2::geom_point(size = 2) +
    ggplot2::labs(
      title = sprintf("Prior sensitivity of slope calibration (GP '%s')", x$gp_name),
      subtitle = "below the red line, the slope decomposition is miscalibrated",
      x = "GP lengthscale prior width (sdlog)",
      y = sprintf("min uniformity p-value over '%s'", x$target)) +
    ggplot2::theme_bw()
}
