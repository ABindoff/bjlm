#' Per-coordinate pooling factor (prior fraction) for a bjlm fit
#'
#' @description
#' For each group-level (random-effect) coordinate of a fitted \code{bjlm} model,
#' computes the \emph{pooling factor} / \emph{prior fraction} of Gelman and Pardoe
#' (2006):
#' \deqn{\pi_j = \frac{G_{\text{prior}}}{G_{\text{prior}} + G_{\text{lik},j}},}
#' the share of the coordinate's posterior precision contributed by the prior
#' rather than by its own data. \eqn{\pi_j \approx 1} is prior-dominated (the
#' centring/non-centring funnel regime, where a non-centred parameterisation
#' helps); \eqn{\pi_j \approx 0} is data-driven.
#'
#' It covers both random-effect blocks of a bjlm model:
#' \itemize{
#'   \item \strong{Random intercepts} (\code{b0 = ~ 1 + (1 | group)}). Linear
#'     coordinate: \eqn{G_{\text{prior}} = 1/\sigma_u^2} and
#'     \eqn{G_{\text{lik},j} = \sum_{i \in j} I_i} with \eqn{I_i} the per-observation
#'     GLM Fisher weight of the outcome family.
#'   \item \strong{Random change-points} (\code{omega = list(~ 1 + (1 | group))}).
#'     Nonlinear coordinate: \eqn{G_{\text{prior}} = 1/\sigma_{re,k}^2} and
#'     \eqn{G_{\text{lik},j} = \sum_{i \in j} (\partial\mu_i/\partial\omega_{k})^2 I_i},
#'     with the analytic smooth-transition gradient
#'     \eqn{\partial\mu_i/\partial\omega = -[\delta_i s_i(1 + d_i \rho_i (1-s_i))]}
#'     (minus \eqn{b_{1,i}} at the first breakpoint), \eqn{d_i = \tau_i - \omega_i},
#'     \eqn{s_i = \mathrm{logistic}(d_i \rho_i)}.
#'   }
#'
#' This is the diagnostic provided by the \pkg{fibr} package (whose
#' \code{prior_fraction()} it matches coordinate for coordinate); it is
#' reimplemented natively here so that no extra dependency is required. It is a
#' read-only prior-influence report, not a convergence diagnostic: nothing is
#' refit. Quantities are evaluated at the posterior mean. GP contributions to the
#' linear predictor are not included in the family information (a note is emitted).
#'
#' @param fit A \code{bjlm_fit} object.
#' @param ... Unused.
#'
#' @return A data frame of class \code{bjlm_pooling_factor} with one row per
#'   coordinate: \code{type} (intercept/changepoint), \code{group}, \code{coef},
#'   \code{level}, \code{n_obs}, \code{prior_sd}, \code{lik_info}, \code{pi}, and
#'   \code{recommendation} (centred / borderline / non-centred). Has \code{print}
#'   and \code{plot} methods. Returns \code{invisible(NULL)} for a model with no
#'   random effects.
#'
#' @references
#' Gelman and Pardoe (2006), \emph{Technometrics} 48(2):241--251.
#'
#' @export
pooling_factor <- function(fit, ...) UseMethod("pooling_factor")

utils::globalVariables(c("n_plot", "pi", "type"))

# Per-observation likelihood (Fisher) information w.r.t. the linear predictor:
# the GLM IRLS working weight (d mu / d eta)^2 / Var(y | mu). Matches
# fibr:::.glm_information for the families bjlm supports.
.pf_information <- function(family, eta, dispersion = 1) {
  switch(family,
    gaussian          = rep(1 / dispersion, length(eta)),      # 1 / sigma^2
    binomial          = { p <- stats::plogis(eta); p * (1 - p) },
    negative_binomial = { m <- exp(eta); m / (1 + m / dispersion) }, # NB2, disp = r
    stop("pooling_factor(): outcome family '", family, "' is not supported.",
         call. = FALSE)
  )
}

#' @export
pooling_factor.bjlm_fit <- function(fit, ...) {
  `%||%` <- function(a, b) if (is.null(a)) b else a
  dm_mat <- as.matrix(posterior::as_draws_matrix(fit$draws))
  vars   <- colnames(dm_mat)

  family <- fit$outcome_family %||% fit$model$outcome$family$family %||% "gaussian"
  sigma  <- if ("sigma" %in% vars) mean(dm_mat[, "sigma"]) else 1
  disp   <- switch(family,
                   gaussian          = sigma^2,
                   negative_binomial = if ("r" %in% vars) mean(dm_mat[, "r"]) else 1,
                   binomial          = 1,
                   1)

  # Rebuild the FITTING design (with random-effect dummy columns + re_mask); the
  # prediction builder uses model.matrix() and cannot represent (1 | group).
  b0_full <- fit$model$outcome$b0 %||% fit$b0_formula
  design  <- tryCatch(
    .build_design_matrices(b0_full, fit$b1_formula, fit$deltas, fit$omega, fit$rho, fit$data),
    error = function(e)
      stop("pooling_factor(): could not rebuild the model design (", conditionMessage(e),
           ").", call. = FALSE))

  tau  <- as.double(fit$data[[all.vars(fit$outcome_formula)[2]]])
  n    <- length(tau)
  n_bp <- length(design$X_deltas)

  if (length(fit$model$latent_gps %||% list()) > 0)
    message("pooling_factor(): latent-GP contributions are omitted from the ",
            "linear predictor used for the family information (minor for binomial/NB).")

  # Posterior-mean coefficient vectors (plain, and spike-and-slab effective).
  bm <- function(nms) vapply(nms, function(nm)
    if (nm %in% vars) mean(dm_mat[, nm]) else 0, numeric(1))
  eff <- function(prefix, gamma_prefix, cols) vapply(cols, function(cc) {
    bn <- paste0(prefix, cc); if (!(bn %in% vars)) return(0)
    b <- dm_mat[, bn]; gn <- paste0(gamma_prefix, cc)
    if (gn %in% vars) b <- b * dm_mat[, gn]
    mean(b)
  }, numeric(1))

  beta_b0 <- bm(paste0("b0_", design$col_names_b0))
  b1_eff  <- eff("b1_", "gamma_b1_", design$col_names_b1)
  b1v     <- as.vector(design$X_b1 %*% b1_eff)

  om <- rho <- del <- vector("list", n_bp)
  for (k in seq_len(n_bp)) {
    om[[k]]  <- as.vector(design$X_om[[k]]  %*% bm(paste0("omega", k, "_", design$col_names_om[[k]])))
    rho[[k]] <- as.vector(design$X_rho[[k]] %*% bm(paste0("rho",   k, "_", design$col_names_rho[[k]])))
    del[[k]] <- as.vector(design$X_deltas[[k]] %*%
                          eff(paste0("delta", k, "_"), paste0("gamma_delta", k, "_"),
                              design$col_names_deltas[[k]]))
  }

  # Linear predictor at the posterior mean (mirrors the Rust mean function).
  eta <- as.vector(design$X_b0 %*% beta_b0)
  if (design$n_groups_b0 > 0) {
    u <- bm(paste0("u_", design$group_levels_b0))
    for (i in seq_len(n)) if (design$group_b0[i] >= 0) eta[i] <- eta[i] + u[design$group_b0[i] + 1L]
  }
  if (n_bp > 0) eta <- eta + b1v * (tau - om[[1]]) else eta <- eta + b1v * tau
  for (k in seq_len(n_bp)) {
    d <- tau - om[[k]]; s <- stats::plogis(d * rho[[k]])
    eta <- eta + del[[k]] * d * s
  }
  info <- .pf_information(family, eta, disp)

  rows <- list()

  # ---- Random-intercept block (linear) ----
  if (design$n_groups_b0 > 0 && "sigma_u" %in% vars) {
    sig_re <- mean(dm_mat[, "sigma_u"]); pp <- 1 / sig_re^2
    for (gi in seq_len(design$n_groups_b0)) {
      m    <- design$group_b0 == (gi - 1L)
      Glik <- sum(info[m])                            # Z = 1 for the intercept
      rows[[length(rows) + 1L]] <- data.frame(
        type = "intercept", group = fit$subject_var %||% "group",
        coef = "(Intercept)", level = design$group_levels_b0[gi],
        n_obs = sum(m), prior_sd = sig_re, lik_info = Glik,
        pi = pp / (pp + Glik), stringsAsFactors = FALSE)
    }
  }

  # ---- Random change-point block(s) (nonlinear) ----
  for (k in seq_len(n_bp)) {
    re_mask <- attr(design$X_om[[k]], "re_mask")
    if (is.null(re_mask)) next
    re_cols <- which(re_mask == 1L)
    sre_nm  <- paste0("sigma_re_om", k)
    if (!length(re_cols) || !(sre_nm %in% vars)) next
    sig_re <- mean(dm_mat[, sre_nm]); pp <- 1 / sig_re^2
    d   <- tau - om[[k]]; s <- stats::plogis(d * rho[[k]])
    dmu <- -(del[[k]] * s * (1 + d * rho[[k]] * (1 - s)))
    if (k == 1L) dmu <- dmu - b1v
    cn <- design$col_names_om[[k]]
    for (ji in re_cols) {
      m    <- design$X_om[[k]][, ji] == 1
      Glik <- sum(dmu[m]^2 * info[m])
      rows[[length(rows) + 1L]] <- data.frame(
        type = "changepoint", group = paste0("omega", k),
        coef = "changepoint", level = sub("^re_", "", cn[ji]),
        n_obs = sum(m), prior_sd = sig_re, lik_info = Glik,
        pi = pp / (pp + Glik), stringsAsFactors = FALSE)
    }
  }

  if (!length(rows)) {
    message("pooling_factor(): no group-level (random) effects found. Add a ",
            "random intercept (b0 = ~ 1 + (1 | group)) or random change-point ",
            "(omega = list(~ 1 + (1 | group))).")
    return(invisible(NULL))
  }

  out <- do.call(rbind, rows)
  out$pi[!is.finite(out$pi)] <- 1                     # zero total precision -> all prior
  out$recommendation <- ifelse(out$pi > 0.6, "non-centred",
                        ifelse(out$pi < 0.4, "centred (OK)", "borderline"))
  rownames(out) <- NULL
  class(out) <- c("bjlm_pooling_factor", "data.frame")
  out
}

#' @export
print.bjlm_pooling_factor <- function(x, threshold = 0.8, ...) {
  n  <- nrow(x)
  hi <- sum(x$pi > threshold, na.rm = TRUE)
  cat(sprintf("<bjlm pooling factor>  %d coordinate(s)\n", n))
  cat(sprintf("  prior-dominated (pi > %.2f): %d (%.0f%%)\n",
              threshold, hi, 100 * hi / max(n, 1L)))
  cat(sprintf("  pi range: [%.3f, %.3f], median %.3f\n",
              min(x$pi, na.rm = TRUE), max(x$pi, na.rm = TRUE),
              stats::median(x$pi, na.rm = TRUE)))
  cat("  (pi near 1 = mostly prior/shrinkage -> consider non-centred; near 0 = data-driven)\n")
  print(utils::head(as.data.frame(x[order(-x$pi), ]), 10), row.names = FALSE)
  if (n > 10) cat(sprintf("  ... %d more rows\n", n - 10))
  invisible(x)
}

#' @export
plot.bjlm_pooling_factor <- function(x, threshold = 0.8, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("plot(): the 'ggplot2' package is required.", call. = FALSE)
  df <- as.data.frame(x)
  df$n_plot <- pmax(df$n_obs, 0.5)
  ggplot2::ggplot(df, ggplot2::aes(x = n_plot, y = pi, colour = type)) +
    ggplot2::geom_hline(yintercept = threshold, linetype = 2, colour = "grey50") +
    ggplot2::geom_point(alpha = 0.8) +
    ggplot2::scale_x_log10() +
    ggplot2::coord_cartesian(ylim = c(0, 1)) +
    ggplot2::labs(x = "observations loading on the coordinate (log scale)",
                  y = expression(pi[j] ~ "(pooling factor)"), colour = NULL,
                  title = "Pooling factor by coordinate",
                  subtitle = "high = prior-dominated (mostly shrinkage); low = data-driven") +
    ggplot2::theme_minimal()
}
