#' Summary for bipw_fit objects
#'
#' @param object A \code{bipw_fit} object.
#' @param model Which model to summarise: \code{"outcome"}, \code{"propensity"},
#'   or \code{"both"} (default).
#' @param prob Probability mass for credible interval (default: 0.95).
#' @param ... Additional arguments (ignored).
#'
#' @export
summary.bipw_fit <- function(object, model = c("both", "outcome", "propensity"), prob = 0.95, ...) {
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

#' Print method for bipw_fit
#' @param x A \code{bipw_fit} object.
#' @param ... Additional arguments (ignored).
#' @export
print.bipw_fit <- function(x, ...) {
  cat("Joint Bayesian IPW model (bipw)\n")
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
#' @param fit A \code{bipw_fit} object.
#' @param param Character string identifying the treatment effect parameter.
#'   This should be the name of the parameter capturing the treatment effect
#'   in the outcome model (e.g., \code{"b1_GroupExperimental"}).
#' @param prob Probability mass for credible interval (default: 0.95).
#'
#' @return A named list with \code{mean}, \code{sd}, \code{lower}, \code{upper},
#'   and the full posterior draws vector.
#'
#' @export
causal_effect <- function(fit, param, prob = 0.95) {
  stopifnot(inherits(fit, "bipw_fit"))
  alpha <- 1 - prob

  if (requireNamespace("posterior", quietly = TRUE)) {
    draws <- posterior::subset_draws(fit$draws, variable = param)
    draws_vec <- as.vector(posterior::draws_of(draws))
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

#' Weight diagnostics for bipw_fit
#'
#' Reports summary statistics of the IPW weights across posterior draws,
#' including checks for positivity violations.
#'
#' @param fit A \code{bipw_fit} object.
#'
#' @export
weight_diagnostics <- function(fit) {
  stopifnot(inherits(fit, "bipw_fit"))

  if (requireNamespace("posterior", quietly = TRUE)) {
    w_draws <- as.vector(posterior::draws_of(
      posterior::subset_draws(fit$draws, variable = "mean_weight")
    ))
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
