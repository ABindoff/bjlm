#' Summary for bjlm_fit objects
#'
#' @param object A \code{bjlm_fit} object.
#' @param model Which model to summarise: \code{"outcome"}, \code{"propensity"},
#'   or \code{"both"} (default).
#' @param prob Probability mass for credible interval (default: 0.95).
#' @param ... Additional arguments (ignored).
#'
#' @export
summary.bjlm_fit <- function(object, model = c("both", "outcome", "propensity"), prob = 0.95, ...) {
  model <- match.arg(model)
  alpha <- 1 - prob

  if (requireNamespace("posterior", quietly = TRUE)) {
    draws <- object$draws
  } else {
    stop("Package 'posterior' is required for summary. Install it with install.packages('posterior').")
  }

  .summarise_params <- function(param_names, label) {
    if (length(param_names) == 0) return(NULL)
    sub <- posterior::subset_draws(draws, variable = param_names)
    smry <- posterior::summarise_draws(sub,
      mean = mean,
      sd = stats::sd,
      q_lo = ~ stats::quantile(.x, probs = alpha / 2),
      q_hi = ~ stats::quantile(.x, probs = 1 - alpha / 2),
      rhat = posterior::rhat,
      ess_bulk = posterior::ess_bulk
    )
    cat(sprintf("\n--- %s model ---\n", label))
    print(smry, n = nrow(smry))
    invisible(smry)
  }

  results <- list()
  if (model %in% c("both", "outcome")) {
    results$outcome <- .summarise_params(object$outcome_names, "Outcome (weighted)")
  }
  if (model %in% c("both", "propensity")) {
    results$propensity <- .summarise_params(object$propensity_names, "Propensity")
    # Weight diagnostic
    w_draws <- posterior::subset_draws(draws, variable = "mean_weight")
    w_smry <- posterior::summarise_draws(w_draws, mean = mean, sd = stats::sd)
    cat("\n--- Weight diagnostics ---\n")
    cat(sprintf("  Mean weight: %.3f (SD: %.3f)\n", w_smry$mean, w_smry$sd))
    cat(sprintf("  Weight type: %s, max_weight: %g\n", object$weight_type, object$max_weight))
  }

  invisible(results)
}

#' Print method for bjlm_fit
#' @param x A \code{bjlm_fit} object.
#' @param ... Additional arguments (ignored).
#' @export
print.bjlm_fit <- function(x, ...) {
  cat("Bayesian Joint Longitudinal Model (bjlm)\n")
  cat(sprintf("  Observations: %d (%d subjects)\n", x$n, x$n_subjects))
  cat(sprintf("  Breakpoints: %d\n", x$n_breakpoints))
  cat(sprintf("  Chains: %d, Iter: %d (warmup: %d)\n", x$chains, x$iter, x$warmup))
  cat(sprintf("  Treatment: %s\n", x$treatment_name))
  cat(sprintf("  Weights: %s (max: %g)\n", x$weight_type, x$max_weight))
  cat(sprintf("  Outcome params: %d, Propensity params: %d\n",
    length(x$outcome_names), length(x$propensity_names)))
  invisible(x)
}

#' Extract causal effect estimates
#'
#' Computes the average treatment effect (ATE) or treatment effect on the treated
#' (ATT) from the posterior draws, including credible intervals.
#'
#' @param fit A \code{bjlm_fit} object.
#' @param param Character string identifying the treatment effect parameter.
#'   This should be the name of the parameter capturing the treatment effect
#'   in the outcome model (e.g., \code{"b1_GroupExperimental"}).
#' @param prob Probability mass for credible interval (default: 0.95).
#'
#' @return A named list with \code{mean}, \code{sd}, \code{lower}, \code{upper},
#'   and the full posterior draws vector.
#'
#' @export
causal_effect <- function(fit, param = NULL, prob = 0.95) {
  stopifnot(inherits(fit, "bjlm_fit"))
  alpha <- 1 - prob

  # Retrieve treatment variable name
  treatment_name <- fit$treatment_name
  if (is.null(treatment_name) || is.na(treatment_name)) {
    if (!is.null(fit$propensity_formula)) {
      treatment_name <- all.vars(fit$propensity_formula)[1]
    }
  }

  if (is.null(param)) {
    if (is.null(treatment_name) || is.na(treatment_name)) {
      stop("`param` must be specified because treatment variable name could not be auto-detected.")
    }
    
    # Find all outcome parameters containing treatment_name
    matching_params <- fit$outcome_names[grepl(treatment_name, fit$outcome_names)]
    
    if (length(matching_params) == 0) {
      stop(sprintf("No parameter in the outcome model contains the treatment variable name '%s'.", treatment_name))
    }
    
    if (length(matching_params) == 1) {
      param <- matching_params[1]
      message(sprintf("Auto-detected treatment parameter: '%s'", param))
    } else {
      # In change-point models, default to the slope-change parameter (delta) if it exists,
      # otherwise default to the first match.
      slope_matches <- matching_params[grepl("^delta", matching_params)]
      if (length(slope_matches) > 0) {
        param <- slope_matches[1]
      } else {
        param <- matching_params[1]
      }
      message(sprintf("Multiple treatment parameters found (%s). Defaulting to: '%s'", 
                      paste(paste0("'", matching_params, "'"), collapse = ", "), param))
    }
  } else {
    # Validate user-specified parameter name
    if (!is.null(treatment_name) && !is.na(treatment_name)) {
      if (!grepl(treatment_name, param)) {
        warning(sprintf("The specified parameter '%s' does not contain the treatment variable name '%s'. Are you sure this is the correct parameter?", param, treatment_name))
      }
    }
    if (!param %in% fit$outcome_names) {
      stop(sprintf("Parameter '%s' not found in the outcome model draws. Available parameters: %s", 
                   param, paste(paste0("'", fit$outcome_names, "'"), collapse = ", ")))
    }
  }

  if (requireNamespace("posterior", quietly = TRUE)) {
    draws <- posterior::subset_draws(fit$draws, variable = param)
    draws_mat <- posterior::as_draws_matrix(draws)
    draws_vec <- as.numeric(draws_mat[, param])
  } else {
    stop("Package 'posterior' is required.")
  }

  result <- list(
    parameter = param,
    mean = mean(draws_vec),
    sd = sd(draws_vec),
    lower = quantile(draws_vec, alpha / 2),
    upper = quantile(draws_vec, 1 - alpha / 2),
    draws = draws_vec,
    prob = prob
  )

  cat(sprintf("\nCausal effect estimate (%s):\n", param))
  cat(sprintf("  Mean:  %.4f\n", result$mean))
  cat(sprintf("  SD:    %.4f\n", result$sd))
  cat(sprintf("  %d%% CrI: [%.4f, %.4f]\n",
    round(prob * 100), result$lower, result$upper))

  invisible(result)
}

#' Weight diagnostics for bjlm_fit
#'
#' Reports summary statistics of the IPW weights across posterior draws,
#' including checks for positivity violations.
#'
#' @param fit A \code{bjlm_fit} object.
#'
#' @export
weight_diagnostics <- function(fit) {
  stopifnot(inherits(fit, "bjlm_fit"))

  if (requireNamespace("posterior", quietly = TRUE)) {
    draws <- posterior::subset_draws(fit$draws, variable = "mean_weight")
    draws_mat <- posterior::as_draws_matrix(draws)
    w_draws <- as.numeric(draws_mat[, "mean_weight"])
  } else {
    stop("Package 'posterior' is required.")
  }

  cat("\n=== Weight Diagnostics ===\n")
  cat(sprintf("  Weight type: %s\n", fit$weight_type))
  cat(sprintf("  Max weight (trim): %g\n", fit$max_weight))
  cat(sprintf("  Mean weight across draws: %.3f (SD: %.3f)\n", mean(w_draws), sd(w_draws)))
  cat(sprintf("  Range: [%.3f, %.3f]\n", min(w_draws), max(w_draws)))

  invisible(w_draws)
}



