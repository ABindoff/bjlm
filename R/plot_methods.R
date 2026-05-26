#' Trace and density plots for a smoothbp_fit
#'
#' A thin wrapper around \code{\link{trace_plot}} for the standard
#' \code{plot()} interface.
#'
#' @param x   A \code{smoothbp_fit} object.
#' @param type One of \code{"trace"} (default), \code{"density"}, or
#'   \code{"both"}.
#' @param pars Character vector of parameter names.  Defaults to all
#'   non-random-effect parameters.
#' @param ...  Passed to \code{\link{trace_plot}}.
#' @return A \code{ggplot} object, or a named list of two when
#'   \code{type = "both"}.
#' @export
plot.smoothbp_fit <- function(x, type = "trace", pars = NULL, ...) {
  if (is.null(pars)) {
    all_pars <- posterior::variables(x$draws)
    pars <- all_pars[!grepl("^u\\[", all_pars)]
  }
  trace_plot(x, pars = pars, type = type, ...)
}


#' Trace and density plots for a bjlm_fit
#'
#' A thin wrapper around \code{\link{trace_plot}} for the standard
#' \code{plot()} interface.
#'
#' @param x   A \code{bjlm_fit} object.
#' @param type One of \code{"trace"} (default), \code{"density"}, or
#'   \code{"both"}.
#' @param pars Character vector of parameter names.  Defaults to all
#'   non-random-effect parameters.
#' @param ...  Passed to \code{\link{trace_plot}}.
#' @return A \code{ggplot} object, or a named list of two when
#'   \code{type = "both"}.
#' @export
plot.bjlm_fit <- function(x, type = "trace", pars = NULL, ...) {
  if (is.null(pars)) {
    all_pars <- posterior::variables(x$draws)
    pars <- all_pars[!grepl("^u\\[", all_pars)]
  }
  trace_plot(x, pars = pars, type = type, ...)
}

#' Plot posterior inclusion probabilities
#'
#' @param x A `smoothbp_pip` object.
#' @param ... Unused.
#'
#' @return A `ggplot` object.
#' @export
plot.smoothbp_pip <- function(x, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting PIPs.")
  }
  
  # Try to extract breakpoint index for coloring/grouping
  # Parameter names are like delta1_var, delta2_var
  x$breakpoint <- NA_integer_
  idx <- grepl("^delta([0-9]+)_", x$parameter)
  if (any(idx)) {
    x$breakpoint[idx] <- as.integer(sub("^delta([0-9]+)_.*", "\\1", x$parameter[idx]))
  }
  
  x$type <- ifelse(is.na(x$breakpoint), "Baseline Slope (b1)", paste("Breakpoint", x$breakpoint))
  
  p <- ggplot2::ggplot(x, ggplot2::aes(x = pip, y = stats::reorder(parameter, pip), 
                                      xmin = lower, xmax = upper, color = type)) +
    ggplot2::geom_vline(xintercept = 0.5, linetype = "dashed", alpha = 0.3) +
    ggplot2::geom_errorbarh(height = 0.2) +
    ggplot2::geom_point(size = 2.5) +
    ggplot2::scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    ggplot2::labs(
      title    = "Posterior Inclusion Probabilities (PIP)",
      subtitle = "Points show mean; bars show 95% HDI (Beta posterior)",
      x        = "Probability of Inclusion",
      y        = "Parameter",
      color    = "Model Component"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      legend.position  = "bottom"
    )

  # If we have multiple breakpoints, facetting makes it easier to read
  n_bp <- length(unique(x$breakpoint[!is.na(x$breakpoint)]))
  if (n_bp > 1) {
    p <- p + ggplot2::facet_wrap(~ type, scales = "free_y", ncol = 1)
  }
  
  p
}

# ---------------------------------------------------------------------------
# Internal helper: draws_df -> long data frame
# ---------------------------------------------------------------------------

.draws_to_long <- function(draws_obj, pars) {
  draws_df   <- posterior::as_draws_df(draws_obj[, , pars, drop = FALSE])
  param_cols <- setdiff(names(draws_df), c(".chain", ".iteration", ".draw"))
  do.call(rbind, lapply(param_cols, function(p) {
    data.frame(
      parameter = p,
      iteration = draws_df$.iteration,
      chain     = factor(draws_df$.chain),
      value     = draws_df[[p]],
      stringsAsFactors = FALSE
    )
  }))
}

#' Trace plots with automatic poor-mixing highlighting
#'
#' Produces per-parameter trace plots from a \code{smoothbp_fit} or \code{bjlm_fit} object.
#' Parameters with \eqn{\hat{R} > 1.05} are flagged with a light-red
#' background and their panel labels include the \eqn{\hat{R}} value and a
#' warning symbol.  Parameters with low bulk-ESS (< 100) are further annotated.
#'
#' @param fit  A \code{smoothbp_fit} or \code{bjlm_fit} object.
#' @param pars Character vector of parameter names to include.  Defaults to
#'   all non-random-effect parameters.
#' @param type One of \code{"trace"} (default), \code{"density"}, or
#'   \code{"both"}.
#' @param rhat_thresh Rhat threshold above which a parameter is flagged as
#'   poorly mixing.  Default \code{1.05}.
#' @param ess_thresh   Bulk-ESS threshold below which a parameter is flagged.
#'   Default \code{100}.
#'
#' @return A \code{ggplot} object (or a named list of two when
#'   \code{type = "both"}).
#' @export
trace_plot <- function(
    fit,
    pars        = NULL,
    type        = "trace",
    rhat_thresh = 1.05,
    ess_thresh  = 100
) {
  if (!inherits(fit, c("smoothbp_fit", "bjlm_fit"))) {
    stop("`fit` must be a smoothbp_fit or bjlm_fit object.")
  }
  if (!type %in% c("trace", "density", "both")) {
    stop('`type` must be one of "trace", "density", or "both".')
  }

  if (is.null(pars)) {
    all_pars <- posterior::variables(fit$draws)
    pars <- all_pars[!grepl("^u\\[", all_pars)]
  }

  # ---- Compute mixing diagnostics -----------------------------------------
  diag_df <- .mixing_diagnostics(fit, pars, rhat_thresh, ess_thresh)

  # ---- Build long draws data frame ----------------------------------------
  long <- .draws_to_long(fit$draws, pars)

  # Attach diagnostics and relabel parameters
  long <- merge(long, diag_df[, c("parameter", "label", "flag")],
                by = "parameter", all.x = TRUE)

  # Use labelled parameter as the facet variable
  long$param_label <- long$label

  # Build background data for flagged parameters (only for trace)
  bad_params <- diag_df$parameter[diag_df$flag]

  if (type %in% c("trace", "both")) {
    p_trace <- .build_trace(long, bad_params, rhat_thresh, ess_thresh)
  }

  if (type %in% c("density", "both")) {
    p_dens <- .build_density(long)
  }

  # Print mixing summary to console if any flags
  n_bad <- sum(diag_df$flag)
  if (n_bad > 0) {
    bad_names <- diag_df$parameter[diag_df$flag]
    bad_rhats <- diag_df$rhat[diag_df$flag]
    message(sprintf(
      "%d parameter(s) flagged (Rhat > %.2f or ESS < %d): %s",
      n_bad, rhat_thresh, ess_thresh,
      paste(sprintf("%s (%.3f)", bad_names, bad_rhats), collapse = ", ")
    ))
  }

  if (type == "trace")   return(p_trace)
  if (type == "density") return(p_dens)
  list(trace = p_trace, density = p_dens)
}

# ---------------------------------------------------------------------------
# Internal: compute per-parameter Rhat and ESS, build flag + label
# ---------------------------------------------------------------------------

.mixing_diagnostics <- function(fit, pars, rhat_thresh, ess_thresh) {
  # Compute Rhat and ESS per parameter, guarding against empty results
  draws_sub <- fit$draws[, , pars, drop = FALSE]
  
  rhats <- vapply(pars, function(p) {
    tryCatch(posterior::rhat(draws_sub[, , p, drop = FALSE]), error = function(e) NA_real_)
  }, numeric(1))
  
  ess <- vapply(pars, function(p) {
    tryCatch(posterior::ess_bulk(draws_sub[, , p, drop = FALSE]), error = function(e) NA_real_)
  }, numeric(1))
  
  names(rhats) <- pars
  names(ess)   <- pars

  flag <- !is.na(rhats) & (rhats > rhat_thresh | (!is.na(ess) & ess < ess_thresh))

  # Build human-readable panel labels
  labels <- vapply(pars, function(p) {
    r <- rhats[p]
    e <- ess[p]
    rhat_str <- if (!is.na(r)) sprintf("Rhat=%.3f", r) else "Rhat=NA"
    ess_str  <- if (!is.na(e)) sprintf("ESS=%d", round(e)) else "ESS=NA"
    warn <- if (!is.na(r) && (r > rhat_thresh || (!is.na(e) && e < ess_thresh))) " \u26a0" else ""
    sprintf("%s\n%s  %s%s", p, rhat_str, ess_str, warn)
  }, character(1))

  data.frame(
    parameter = pars,
    rhat      = rhats,
    ess       = ess,
    flag      = flag,
    label     = labels,
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

# ---------------------------------------------------------------------------
# Internal: build trace ggplot
# ---------------------------------------------------------------------------

.build_trace <- function(long, bad_params, rhat_thresh, ess_thresh) {
  # Background rectangles for flagged parameters (very light red wash)
  has_bad <- length(bad_params) > 0
  if (has_bad) {
    bg_data <- unique(long[long$parameter %in% bad_params,
                           c("param_label", "iteration")])
    bg_data <- do.call(rbind, lapply(
      unique(bg_data$param_label), function(lbl) {
        iters <- bg_data$iteration[bg_data$param_label == lbl]
        data.frame(
          param_label = lbl,
          xmin = min(iters), xmax = max(iters),
          ymin = -Inf,        ymax = Inf,
          stringsAsFactors = FALSE
        )
      }
    ))
  }

  p <- ggplot2::ggplot(long,
         ggplot2::aes(x = iteration, y = value, colour = chain)) +
    ggplot2::geom_line(alpha = 0.7, linewidth = 0.3) +
    ggplot2::facet_wrap(~ param_label, scales = "free_y") +
    ggplot2::labs(
      title    = "Trace plots",
      subtitle = if (has_bad)
        sprintf("Parameters flagged (\u26a0) have Rhat > %.2f or ESS < %d",
                rhat_thresh, ess_thresh)
      else
        "All parameters appear well-mixed",
      x = "Post-warmup iteration",
      y = "Value"
    ) +
    ggplot2::theme(
      legend.position  = "bottom",
      strip.text       = ggplot2::element_text(size = 7.5, lineheight = 1.1),
      plot.subtitle    = ggplot2::element_text(
        colour = if (has_bad) "#cc3333" else "grey40", size = 9
      )
    )

  # Overlay red background on bad-mixing panels
  if (has_bad) {
    p <- p + ggplot2::geom_rect(
      data        = bg_data,
      ggplot2::aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      fill        = "#ff4444",
      alpha       = 0.07,
      inherit.aes = FALSE
    )
    # Re-add lines on top of background (layer ordering)
    p <- p + ggplot2::geom_line(alpha = 0.7, linewidth = 0.3)
  }

  p
}

# ---------------------------------------------------------------------------
# Internal: build density ggplot
# ---------------------------------------------------------------------------

.build_density <- function(long) {
  ggplot2::ggplot(long,
    ggplot2::aes(x = value, colour = chain, fill = chain)) +
    ggplot2::geom_density(alpha = 0.2) +
    ggplot2::facet_wrap(~ param_label, scales = "free") +
    ggplot2::labs(
      title = "Posterior densities",
      x = "Value", y = "Density"
    ) +
    ggplot2::theme(
      legend.position = "bottom",
      strip.text      = ggplot2::element_text(size = 7.5, lineheight = 1.1)
    )
}

# Suppress CRAN check warnings for ggplot variables
Propensity <- Treatment <- Weight <- y_fit <- .data <- NULL

#' Plot outcome predictions and piecewise trajectories
#'
#' @param fit A `bjlm_fit` object.
#' @param type One of `"population"` (default), `"subject"`, or `"both"`.
#' @param subjects Optional vector of subject IDs to plot when `type` is `"subject"` or `"both"`.
#' @param n_subjects Number of random subjects to select if `subjects` is NULL.
#' @param ... Unused.
#'
#' @return A `ggplot` object.
#' @export
plot_predictions <- function(fit, type = c("population", "subject", "both"), subjects = NULL, n_subjects = 5, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting predictions.")
  }
  type <- match.arg(type)
  
  # Parse outcome variable names
  outcome_vars <- all.vars(fit$outcome_formula)
  y_name <- outcome_vars[1]
  tau_name <- outcome_vars[2]
  
  # Base plot with raw data
  p <- ggplot2::ggplot(fit$data, ggplot2::aes(x = .data[[tau_name]])) +
    ggplot2::geom_point(ggplot2::aes(y = .data[[y_name]]), alpha = 0.3, colour = "grey50", size = 1.2) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", size = 13, colour = "#2c3e50"),
      plot.subtitle = ggplot2::element_text(size = 10, colour = "grey40"),
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = "bottom"
    )
  
  scale_colors <- character(0)
  
  if (type %in% c("population", "both")) {
    pop_data <- fit$data
    if (!is.null(fit$subject_var)) {
      pop_data[[fit$subject_var]] <- "__POPULATION_LEVEL__"
    }
    pop_pred <- stats::fitted(fit, newdata = pop_data, summary = TRUE)
    
    pop_df <- fit$data
    pop_df$y_fit <- pop_pred$fitted_mean
    pop_df$lo <- pop_pred$fitted_Q2.5
    pop_df$hi <- pop_pred$fitted_Q97.5
    
    p <- p +
      ggplot2::geom_ribbon(data = pop_df, ggplot2::aes(ymin = lo, ymax = hi), fill = "#2b5c8f", alpha = 0.15) +
      ggplot2::geom_line(data = pop_df, ggplot2::aes(y = y_fit, colour = "Population Trajectory"), linewidth = 1.2)
      
    scale_colors["Population Trajectory"] <- "#2b5c8f"
      
    # Plot vertical lines for estimated breakpoints (omega)
    draw_mat <- posterior::as_draws_matrix(fit$draws)
    col_names <- colnames(draw_mat)
    om_cols <- grep("^omega[0-9]+_", col_names, value = TRUE)
    if (length(om_cols) > 0) {
      for (om_var in om_cols) {
        om_mean <- mean(as.numeric(draw_mat[, om_var]))
        p <- p + ggplot2::geom_vline(xintercept = om_mean, linetype = "dashed", colour = "#d35400", alpha = 0.6)
      }
    }
  }
  
  if (type %in% c("subject", "both") && !is.null(fit$subject_var)) {
    sub_var <- fit$subject_var
    all_subs <- unique(fit$data[[sub_var]])
    if (is.null(subjects)) {
      subjects <- sample(all_subs, min(n_subjects, length(all_subs)))
    }
    
    sub_df <- fit$data[fit$data[[sub_var]] %in% subjects, , drop = FALSE]
    if (nrow(sub_df) > 0) {
      sub_pred <- stats::fitted(fit, newdata = sub_df, summary = TRUE)
      sub_df$y_fit <- sub_pred$fitted_mean
      sub_df$lo <- sub_pred$fitted_Q2.5
      sub_df$hi <- sub_pred$fitted_Q97.5
      
      # Convert subjects to factor for distinct colors
      sub_df[[sub_var]] <- as.factor(sub_df[[sub_var]])
      
      p <- p +
        ggplot2::geom_line(data = sub_df, ggplot2::aes(y = y_fit, group = .data[[sub_var]], colour = "Subject-specific Fitted"), 
                           linewidth = 0.8, alpha = 0.8, linetype = "dashed")
                           
      scale_colors["Subject-specific Fitted"] <- "#2ecc71"
    }
  }
  
  p <- p + ggplot2::scale_colour_manual(values = scale_colors, name = "Model Fit") +
    ggplot2::labs(
      title = "Posterior Fitted Trajectories",
      subtitle = "Shaded band shows 95% credible interval; points are observed data.",
      x = tau_name,
      y = y_name
    )
  
  p
}

#' Plot time-varying latent Gaussian Process (GP) deviations
#'
#' @param fit A `bjlm_fit` object.
#' @param subjects Optional vector of subject IDs to plot.
#' @param n_subjects Number of random subjects to select if `subjects` is NULL.
#' @param ... Unused.
#'
#' @return A `ggplot` object.
#' @export
plot_gp <- function(fit, subjects = NULL, n_subjects = 5, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting GP curves.")
  }
  
  # Look for parameters starting with gp_ in the posterior draws
  col_names <- colnames(posterior::as_draws_matrix(fit$draws))
  gp_cols <- grep("^gp_", col_names, value = TRUE)
  
  if (length(gp_cols) == 0) {
    warning("No latent Gaussian Process draws found in this fitted model. ",
         "The GP sampler is currently a work-in-progress. Returning skeleton plot.", call. = FALSE)
  }
  
  # Skeleton implementation
  ggplot2::ggplot() + 
    ggplot2::labs(title = "Latent Gaussian Process Curves")
}

#' Plot propensity score diagnostics and weight balance
#'
#' @param fit A `bjlm_fit` object.
#' @param type One of `"overlap"` (default), `"weights"`, or `"both"`.
#' @param ... Unused.
#'
#' @return A `ggplot` object, or a list of two ggplot objects if `type = "both"`.
#' @export
plot_propensity <- function(fit, type = c("overlap", "weights", "both"), ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting propensity diagnostics.")
  }
  type <- match.arg(type)
  
  if (is.null(fit$propensity_formula)) {
    warning("No propensity score model specified in this fit. Skipping propensity plotting.")
    return(NULL)
  }
  
  prop_vars <- all.vars(fit$propensity_formula)
  treatment_name <- fit$treatment_name
  prop_covariate_names <- prop_vars[prop_vars != treatment_name]
  
  sub_var <- fit$subject_var
  if (!is.null(sub_var)) {
    sub_data <- fit$data[!duplicated(fit$data[[sub_var]]), ]
  } else {
    sub_data <- fit$data
  }
  
  prop_formula <- stats::reformulate(prop_covariate_names)
  x_prop <- stats::model.matrix(prop_formula, data = sub_data)
  
  draw_mat <- posterior::as_draws_matrix(fit$draws)
  alpha_cols <- grep("^alpha_", colnames(draw_mat), value = TRUE)
  
  if (length(alpha_cols) == 0) {
    stop("Propensity model coefficients (alpha) not found in posterior draws.")
  }
  
  alpha_means <- colMeans(draw_mat[, alpha_cols, drop = FALSE])
  alpha_clean_names <- sub("^alpha_", "", alpha_cols)
  match_idx <- match(colnames(x_prop), alpha_clean_names)
  alpha_aligned <- alpha_means[match_idx]
  
  linear_predictor <- as.vector(x_prop %*% alpha_aligned)
  prop_scores <- 1 / (1 + exp(-linear_predictor))
  
  treatment <- sub_data[[treatment_name]]
  
  df_plot <- data.frame(
    Propensity = prop_scores,
    Treatment = factor(treatment, levels = c(0, 1), labels = c("Control", "Treated"))
  )
  
  p_overlap <- ggplot2::ggplot(df_plot, ggplot2::aes(x = Propensity, fill = Treatment)) +
    ggplot2::geom_density(alpha = 0.4, colour = "transparent") +
    ggplot2::scale_fill_manual(values = c("Control" = "#e74c3c", "Treated" = "#2ecc71")) +
    ggplot2::labs(
      title = "Propensity Score Overlap (Common Support)",
      subtitle = "Density distributions of estimated propensity scores by treatment group.",
      x = "Estimated Propensity Score",
      y = "Density"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", size = 12, colour = "#2c3e50"),
      plot.subtitle = ggplot2::element_text(size = 9, colour = "grey40"),
      legend.position = "bottom"
    )
    
  p_trt <- mean(treatment)
  
  if (fit$weight_type == "stabilised_ate") {
    w <- ifelse(treatment == 1, p_trt / prop_scores, (1 - p_trt) / (1 - prop_scores))
  } else if (fit$weight_type == "ate") {
    w <- ifelse(treatment == 1, 1 / prop_scores, 1 / (1 - prop_scores))
  } else if (fit$weight_type == "att") {
    w <- ifelse(treatment == 1, 1, prop_scores / (1 - prop_scores))
  } else if (fit$weight_type == "stabilised_att") {
    w <- ifelse(treatment == 1, p_trt, p_trt * prop_scores / (1 - prop_scores))
  } else {
    w <- rep(1, length(treatment))
  }
  
  w <- pmin(w, fit$max_weight)
  
  df_plot$Weight <- w
  
  p_weights <- ggplot2::ggplot(df_plot, ggplot2::aes(x = Treatment, y = Weight, fill = Treatment)) +
    ggplot2::geom_violin(alpha = 0.5, colour = "transparent", scale = "width") +
    ggplot2::geom_boxplot(width = 0.15, fill = "white", outlier.shape = 16, outlier.size = 1.5, outlier.alpha = 0.5) +
    ggplot2::scale_fill_manual(values = c("Control" = "#e74c3c", "Treated" = "#2ecc71")) +
    ggplot2::labs(
      title = "IPW Weight Distribution",
      subtitle = sprintf("Distribution of calculated weights (type: %s, max trim: %g)", fit$weight_type, fit$max_weight),
      x = "Treatment Status",
      y = "Inverse Probability Weight"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", size = 12, colour = "#2c3e50"),
      plot.subtitle = ggplot2::element_text(size = 9, colour = "grey40"),
      legend.position = "none"
    )
    
  if (type == "overlap") {
    return(p_overlap)
  } else if (type == "weights") {
    return(p_weights)
  } else {
    return(list(overlap = p_overlap, weights = p_weights))
  }
}

