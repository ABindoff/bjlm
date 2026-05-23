#' Numerically stable log-sum-exp helper
#'
#' Avoids underflow or overflow when computing log(sum(exp(x))).
#' @param x Numeric vector.
#' @return A single numeric value.
#' @keywords internal
.log_sum_exp <- function(x) {
  max_x <- max(x)
  if (is.infinite(max_x)) {
    return(max_x)
  }
  max_x + log(sum(exp(x - max_x)))
}

#' Reconstruct a bjlm_model specification from a bjlm_fit object
#'
#' Helper function for LFO-CV refitting.
#' @param object A `bjlm_fit` object.
#' @param data A data frame containing the subset of data for refitting.
#' @return A `bjlm_model` specification.
#' @keywords internal
.reconstruct_model <- function(object, data) {
  spec <- bjlm_model() |>
    propensity(formula = object$propensity_formula, data = data) |>
    outcome(
      formula = object$outcome_formula,
      b0 = object$b0_formula,
      b1 = object$b1_formula,
      deltas = object$deltas,
      omega = object$omega,
      rho = object$rho,
      data = data
    )
  spec$latent_gps <- object$model$latent_gps
  spec
}

#' Helper to compute out-of-sample pointwise log-likelihood matrix
#'
#' @param fit_active A fitted `bjlm_fit` model.
#' @param validation_data A data frame containing held-out observations.
#' @param y_name Character. Name of response variable.
#' @return An S x N matrix of log-likelihoods.
#' @keywords internal
.compute_oos_log_lik <- function(fit_active, validation_data, y_name) {
  pred_draws <- fitted(fit_active, newdata = validation_data, summary = FALSE)
  sigma_draws <- as.numeric(posterior::as_draws_matrix(fit_active$draws)[, "sigma"])
  
  y_obs <- as.double(validation_data[[y_name]])
  ll_matrix <- matrix(0, nrow = nrow(pred_draws), ncol = nrow(validation_data))
  for (i in seq_along(y_obs)) {
    ll_matrix[, i] <- stats::dnorm(y_obs[i], mean = pred_draws[, i], sd = sigma_draws, log = TRUE)
  }
  ll_matrix
}

#' Leave-Future-Out Cross-Validation (LFO-CV) for BJLM Models
#'
#' Performs approximate or exact Leave-Future-Out (LFO) cross-validation for a fitted
#' longitudinal model (`bjlm_fit`) using Pareto Smoothed Importance Sampling (PSIS)
#' to minimize model refits.
#'
#' @param object A `bjlm_fit` object.
#' @param t_var Character. Name of the time variable in the outcome dataset. If `NULL`
#'   (default), it is parsed from the outcome formula (the second variable on the RHS).
#' @param t_grid Numeric vector. The grid of time points at which splits should occur.
#'   Defaults to the sorted unique values of `t_var` in the dataset.
#' @param min_tau Numeric. The minimum time point to use for the initial model fit.
#'   Observations where time <= `min_tau` will be used for training, and prediction starts for
#'   the next time step. If `NULL`, defaults to the 2nd unique time point in `t_grid`.
#' @param k_threshold Numeric. The Pareto k shape diagnostic threshold (default 0.7) to trigger a model refit.
#' @param verbose Logical. Whether to print progress messages during evaluation.
#' @param ... Extra arguments passed to [fit()].
#'
#' @return An object of class `bjlm_lfo` containing:
#'   \item{elpd_lfo}{The overall expected log predictive density.}
#'   \item{pointwise}{A data frame with pointwise ELPDs for the evaluated observations.}
#'   \item{diagnostics}{A step-by-step diagnostic logging of Pareto k and refits.}
#'   \item{t_var}{The time variable name.}
#'   \item{t_grid}{The time grid evaluated.}
#'   \item{min_tau}{The baseline evaluation start time.}
#'   \item{k_threshold}{The Pareto k threshold used.}
#' @export
lfo_cv <- function(object, t_var = NULL, t_grid = NULL, min_tau = NULL, k_threshold = 0.7, verbose = TRUE, ...) {
  if (!inherits(object, "bjlm_fit")) {
    stop("`object` must be a `bjlm_fit` object.")
  }
  
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("The `loo` package is required to run LFO-CV.")
  }
  if (!requireNamespace("posterior", quietly = TRUE)) {
    stop("The `posterior` package is required to run LFO-CV.")
  }
  
  # 1. Parse outcome response and time variables
  outcome_vars <- all.vars(object$outcome_formula)
  if (length(outcome_vars) < 2) {
    stop("Outcome formula must specify both response and time variables (e.g. Y ~ tau).")
  }
  y_name <- outcome_vars[1]
  
  if (is.null(t_var)) {
    t_var <- outcome_vars[2]
  }
  
  if (!t_var %in% names(object$data)) {
    stop(sprintf("Time variable '%s' not found in the outcome dataset.", t_var))
  }
  
  # 2. Build time grid
  if (is.null(t_grid)) {
    t_grid <- sort(unique(object$data[[t_var]]))
  } else {
    t_grid <- sort(t_grid)
  }
  
  if (length(t_grid) < 2) {
    stop("t_grid must have at least 2 unique time points to perform LFO-CV.")
  }
  
  # 3. Resolve starting time step index min_idx
  if (is.null(min_tau)) {
    min_idx <- 2L
    min_tau <- t_grid[2]
  } else {
    matching_idx <- which(t_grid <= min_tau)
    if (length(matching_idx) == 0) {
      stop("min_tau is smaller than the minimum time point in the time grid.")
    }
    min_idx <- max(matching_idx)
    min_tau <- t_grid[min_idx]
  }
  
  if (min_idx >= length(t_grid)) {
    stop("min_tau must be smaller than the maximum time point in the time grid to allow evaluation on future time points.")
  }
  
  # 4. Prepare data structures
  elpd_pointwise <- rep(NA_real_, nrow(object$data))
  
  diagnostics <- data.frame(
    step = seq(min_idx, length(t_grid) - 1L),
    time_step = t_grid[min_idx:(length(t_grid) - 1L)],
    predict_time = t_grid[(min_idx + 1L):length(t_grid)],
    n_train_obs = NA_integer_,
    n_train_subjects = NA_integer_,
    pareto_k = NA_real_,
    refit = FALSE
  )
  
  # Extract fitting parameters from the original call
  chains <- object$chains
  iter <- object$iter
  warmup <- object$warmup
  seed <- object$call$seed %||% sample.int(.Machine$integer.max, 1L)
  cores <- object$call$cores %||% 1L
  
  # 5. Fit the initial model (data up to min_tau)
  if (verbose) {
    message(sprintf("Fitting initial model on training data up to %s = %s...", t_var, as.character(min_tau)))
  }
  
  data_init <- object$data[object$data[[t_var]] <= min_tau, , drop = FALSE]
  spec_init <- .reconstruct_model(object, data_init)
  compiled_init <- compile(spec_init)
  
  fit_active <- fit(
    compiled_init,
    priors = object$priors,
    chains = chains,
    iter = iter,
    warmup = warmup,
    seed = seed,
    verbose = FALSE,
    cores = cores,
    ...
  )
  
  active_log_weights <- NULL
  last_refit_idx <- min_idx
  
  # 6. Execute LFO loop
  for (step_idx in seq_along(diagnostics$step)) {
    t <- diagnostics$step[step_idx]
    t_val <- diagnostics$predict_time[step_idx]
    
    val_indices <- which(object$data[[t_var]] == t_val)
    if (length(val_indices) == 0) {
      next
    }
    validation_data <- object$data[val_indices, , drop = FALSE]
    
    # pointwise log-likelihood matrix under current fit_active
    ll_matrix <- .compute_oos_log_lik(fit_active, validation_data, y_name)
    S <- nrow(ll_matrix)
    
    # Prediction:
    if (t == last_refit_idx) {
      # EXACT predictive density
      elpd_pointwise[val_indices] <- apply(ll_matrix, 2, function(col) {
        .log_sum_exp(col) - log(S)
      })
      diagnostics$pareto_k[step_idx] <- NA_real_
    } else {
      # APPROXIMATE predictive density using cached importance weights
      elpd_pointwise[val_indices] <- apply(ll_matrix, 2, function(col) {
        .log_sum_exp(col + active_log_weights)
      })
    }
    
    diagnostics$n_train_obs[step_idx] <- nrow(fit_active$data)
    diagnostics$n_train_subjects[step_idx] <- fit_active$n_subjects
    
    # 7. Check if we need to refit for the next time step
    if (step_idx == nrow(diagnostics)) {
      # Final time point predicted, no more steps
      next
    }
    
    # added data since last refit: (T_last_refit, T_t+1]
    added_indices <- which(object$data[[t_var]] > t_grid[last_refit_idx] & object$data[[t_var]] <= t_val)
    if (length(added_indices) == 0) {
      # No new data to check, transition is trivial
      next
    }
    added_data <- object$data[added_indices, , drop = FALSE]
    
    ll_added <- .compute_oos_log_lik(fit_active, added_data, y_name)
    log_ratios <- rowSums(ll_added)
    
    # Run PSIS to estimate k and new weights
    psis_res <- tryCatch({
      suppressWarnings(loo::psis(log_ratios))
    }, error = function(e) {
      if (verbose) {
        message(sprintf("PSIS calculation failed at step %d: %s. Forcing refit.", step_idx, e$message))
      }
      NULL
    })
    
    if (is.null(psis_res)) {
      k_val <- Inf
    } else {
      k_val <- psis_res$diagnostics$pareto_k
    }
    
    diagnostics$pareto_k[step_idx] <- k_val
    
    if (k_val < k_threshold) {
      # Stable importance weights, no refit
      active_log_weights <- as.numeric(psis_res$log_weights)
      if (verbose) {
        message(sprintf("Step %d/%d: Predicted %s = %s (PSIS k = %.3f, no refit)",
                        step_idx, nrow(diagnostics), t_var, as.character(t_val), k_val))
      }
    } else {
      # Unstable importance weights, must refit
      diagnostics$refit[step_idx] <- TRUE
      if (verbose) {
        message(sprintf("Step %d/%d: Predicted %s = %s (PSIS k = %.3f >= %.2f, refitting...)",
                        step_idx, nrow(diagnostics), t_var, as.character(t_val), k_val, k_threshold))
      }
      
      data_new <- object$data[object$data[[t_var]] <= t_val, , drop = FALSE]
      spec_new <- .reconstruct_model(object, data_new)
      compiled_new <- compile(spec_new)
      
      fit_active <- fit(
        compiled_new,
        priors = object$priors,
        chains = chains,
        iter = iter,
        warmup = warmup,
        seed = seed,
        verbose = FALSE,
        cores = cores,
        ...
      )
      
      last_refit_idx <- t + 1L
      active_log_weights <- NULL
    }
  }
  
  # Calculate approximate SE of total ELPD
  # Assuming independent pointwise observations (standard for cross-validation approximations)
  elpd_se <- sqrt(length(eval_indices) * stats::var(elpd_pointwise[eval_indices], na.rm = TRUE))
  
  estimates <- matrix(NA_real_, nrow = 1, ncol = 2)
  rownames(estimates) <- c("elpd_loo")  # Named elpd_loo for compatibility with loo::loo_compare
  colnames(estimates) <- c("Estimate", "SE")
  estimates["elpd_loo", "Estimate"] <- elpd_lfo
  estimates["elpd_loo", "SE"] <- elpd_se

  pointwise_df <- data.frame(
    observation = eval_indices,
    tau = object$data[[t_var]][eval_indices],
    elpd_loo = elpd_pointwise[eval_indices]
  )
  colnames(pointwise_df)[2] <- t_var
  
  res <- list(
    estimates = estimates,
    elpd_lfo = elpd_lfo,
    pointwise = as.matrix(pointwise_df),
    diagnostics = diagnostics,
    t_var = t_var,
    t_grid = t_grid,
    min_tau = min_tau,
    min_idx = min_idx,
    k_threshold = k_threshold
  )
  class(res) <- c("bjlm_lfo", "loo")
  res
}

#' @export
print.bjlm_lfo <- function(x, ...) {
  cat("Leave-Future-Out Cross-Validation (LFO-CV) for BJLM\n")
  cat("===================================================\n")
  cat("Total LFO ELPD:      ", round(x$elpd_lfo, 3), "\n")
  cat("Time variable:       ", x$t_var, "\n")
  cat("Time grid points:    ", length(x$t_grid), " (evaluating ", length(x$t_grid) - x$min_idx, " future steps)\n")
  cat("Min training time:   ", x$min_tau, "\n")
  cat("Refit threshold (k): ", x$k_threshold, "\n")
  
  n_refits <- sum(x$diagnostics$refit)
  cat("Model refits:        ", n_refits, " (out of ", nrow(x$diagnostics), " steps)\n")
  
  cat("\nStep-by-step diagnostics:\n")
  # Format numeric column beautifully
  df_print <- x$diagnostics
  df_print$pareto_k <- round(df_print$pareto_k, 3)
  df_print$pareto_k[is.na(df_print$pareto_k)] <- "-"
  print(df_print, row.names = FALSE)
  
  invisible(x)
}

#' Plot LFO-CV Diagnostics
#'
#' Premium visualization of the Pareto k diagnostics and refit events over the time grid.
#'
#' @param x A `bjlm_lfo` object.
#' @param ... Unused.
#' @importFrom ggplot2 ggplot aes geom_point geom_line geom_hline theme_minimal labs scale_color_manual scale_shape_manual theme element_text
#' @export
plot.bjlm_lfo <- function(x, ...) {
  # Declare variables for CRAN check compliance
  time <- k <- refit <- NULL
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package is required to plot LFO-CV diagnostics.")
  }
  
  df <- x$diagnostics
  
  # Prepare plotting data frame
  plot_df <- data.frame(
    time = df$predict_time,
    k = df$pareto_k,
    refit = ifelse(df$refit, "Yes (Refit)", "No"),
    stringsAsFactors = FALSE
  )
  
  # For exact prediction steps (which don't calculate importance weights), Pareto k is NA.
  # We represent exact steps or refits explicitly.
  plot_df$refit[is.na(plot_df$k)] <- "Exact Prediction"
  # Set a dummy visual k for plotting exact steps if needed, or leave NA to avoid lines.
  
  # Theme and aesthetic setup (Google Fonts/premium style)
  k_thresh <- x$k_threshold
  
  p <- ggplot2::ggplot(plot_df, ggplot2::aes(x = time, y = k)) +
    # Threshold line
    ggplot2::geom_hline(yintercept = k_thresh, linetype = "dashed", color = "#E06666", linewidth = 0.8) +
    # Horizontal line at 0.5 (ideal PSIS target)
    ggplot2::geom_hline(yintercept = 0.5, linetype = "dotted", color = "#999999", linewidth = 0.5) +
    # Draw path connecting approximate steps
    ggplot2::geom_line(color = "#357ABD", linewidth = 0.8, alpha = 0.6) +
    # High-quality stylized points
    ggplot2::geom_point(ggplot2::aes(color = refit, shape = refit), size = 3) +
    # Custom color palette matching the package guidelines
    ggplot2::scale_color_manual(
      values = c("No" = "#4A90E2", "Yes (Refit)" = "#D0021B", "Exact Prediction" = "#2ECC71")
    ) +
    ggplot2::scale_shape_manual(
      values = c("No" = 16, "Yes (Refit)" = 17, "Exact Prediction" = 15)
    ) +
    ggplot2::theme_minimal(base_family = "sans") +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", size = 14, color = "#2C3E50"),
      plot.subtitle = ggplot2::element_text(size = 11, color = "#7F8C8D"),
      axis.title = ggplot2::element_text(face = "bold", color = "#2C3E50"),
      legend.title = ggplot2::element_text(face = "bold"),
      legend.position = "bottom",
      panel.grid.minor = ggplot2::element_blank()
    ) +
    ggplot2::labs(
      title = "Leave-Future-Out Cross-Validation Diagnostics",
      subtitle = sprintf("Time Variable: %s | Refit Threshold: k = %.2f", x$t_var, k_thresh),
      x = sprintf("Future Prediction Time (%s)", x$t_var),
      y = "Pareto k Diagnostic",
      color = "Refit Triggered",
      shape = "Refit Triggered"
    )
  
  p
}
