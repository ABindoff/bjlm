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
#' including checks for positivity violations, Effective Sample Size (ESS),
#' trimming proportion, and a text-based distribution histogram.
#'
#' @param fit A \code{bjlm_fit} object.
#'
#' @return Invisibly, a list containing observation-level mean weights, ESS draws,
#'   trimmed proportions, and untrimmed max weights.
#' @export
weight_diagnostics <- function(fit) {
  stopifnot(inherits(fit, "bjlm_fit"))

  # 1. Subject-level data
  subject_var <- fit$subject_var
  subject_data <- if (!is.null(subject_var) && subject_var %in% names(fit$data)) {
    fit$data[!duplicated(fit$data[[subject_var]]), , drop = FALSE]
  } else {
    fit$data
  }
  
  # 2. Extract propensity model info
  prop_formula <- fit$propensity_formula
  if (is.null(prop_formula)) {
    # If no propensity model was specified (outcome-only model), weights are all 1.0
    cat("\n=== Weight Diagnostics ===\n")
    cat("  Propensity Model:          None (Outcome-only model fitted)\n")
    cat("  Weight Type:               None\n")
    cat("  All weights are identical to 1.0.\n")
    return(invisible(list(
      weights = rep(1.0, nrow(fit$data)),
      ess = rep(nrow(fit$data), fit$iter - fit$warmup),
      trimmed_prop = rep(0, fit$iter - fit$warmup),
      max_untrimmed = rep(1.0, fit$iter - fit$warmup)
    )))
  }
  
  treatment_var <- all.vars(prop_formula)[1]
  rhs_formula <- formula(delete.response(terms(prop_formula)))
  
  x_prop <- model.matrix(rhs_formula, data = subject_data)
  treatment <- as.numeric(subject_data[[treatment_var]])
  n_subj <- length(treatment)
  
  # 3. Extract alpha draws
  alpha_names <- fit$propensity_names
  if (is.null(alpha_names)) {
    alpha_names <- colnames(posterior::as_draws_matrix(fit$draws))
    alpha_names <- alpha_names[grepl("^alpha_", alpha_names)]
  }
  
  if (length(alpha_names) == 0) {
    stop("Could not find propensity parameter draws (alpha) in the fitted model.")
  }
  
  alpha_draws <- posterior::subset_draws(fit$draws, variable = alpha_names)
  alpha_mat <- posterior::as_draws_matrix(alpha_draws)
  
  # Ensure column order matches colnames(x_prop)
  expected_names <- paste0("alpha_", colnames(x_prop))
  colnames_alpha <- colnames(alpha_mat)
  col_indices <- sapply(colnames(x_prop), function(col) {
    target <- paste0("alpha_", col)
    idx <- which(colnames_alpha == target)
    if (length(idx) == 0) {
      idx <- which(grepl(col, colnames_alpha, fixed = TRUE))
    }
    if (length(idx) == 0) {
      stop(sprintf("Could not find propensity coefficient for covariate '%s' in posterior draws.", col))
    }
    idx[1]
  })
  
  alpha_mat <- alpha_mat[, col_indices, drop = FALSE]
  class(alpha_mat) <- "matrix"
  S <- nrow(alpha_mat)
  
  # 4. Determine if treatment is continuous
  is_continuous <- !all(treatment %in% c(0, 1))
  
  # 5. Compute subject-level weights for each draw
  W <- matrix(NA_real_, nrow = S, ncol = n_subj)
  trimmed_prop <- numeric(S)
  max_untrimmed <- numeric(S)
  
  max_w <- fit$max_weight
  w_type <- fit$weight_type
  
  if (is_continuous) {
    mean_t <- mean(treatment)
    var_t <- var(treatment)
    sd_t <- sd(treatment)
    
    for (s in seq_len(S)) {
      alpha_s <- alpha_mat[s, ]
      pred_s <- as.vector(x_prop %*% alpha_s)
      
      # Estimate residual variance for this draw
      resid_sq <- (treatment - pred_s)^2
      sigma_a_sq_s <- mean(resid_sq)
      
      # Conditional density f(T_i | X_i)
      cond_dens <- dnorm(treatment, mean = pred_s, sd = sqrt(sigma_a_sq_s))
      cond_dens <- pmax(cond_dens, 1e-10) # prevent division by zero
      
      # Marginal density f(T_i)
      marg_dens <- dnorm(treatment, mean = mean_t, sd = sd_t)
      
      is_stabilised <- w_type %in% c("stabilised_ate", "stabilised_att")
      
      untrimmed <- if (is_stabilised) {
        marg_dens / cond_dens
      } else {
        1.0 / cond_dens
      }
      
      max_untrimmed[s] <- max(untrimmed)
      trimmed_w <- pmin(untrimmed, max_w)
      W[s, ] <- trimmed_w
      trimmed_prop[s] <- mean(untrimmed >= max_w - 1e-6)
    }
  } else {
    p_marginal <- mean(treatment)
    for (s in seq_len(S)) {
      alpha_s <- alpha_mat[s, ]
      eta_s <- as.vector(x_prop %*% alpha_s)
      pi_s <- 1 / (1 + exp(-eta_s))
      pi_s <- pmax(pmin(pi_s, 1 - 1e-6), 1e-6)
      
      untrimmed <- if (w_type == "ate") {
        ifelse(treatment > 0.5, 1.0 / pi_s, 1.0 / (1.0 - pi_s))
      } else if (w_type == "att") {
        ifelse(treatment > 0.5, 1.0, pi_s / (1.0 - pi_s))
      } else if (w_type == "stabilised_ate") {
        ifelse(treatment > 0.5, p_marginal / pi_s, (1.0 - p_marginal) / (1.0 - pi_s))
      } else if (w_type == "stabilised_att") {
        ifelse(treatment > 0.5, 1.0, p_marginal * pi_s / ((1.0 - p_marginal) * (1.0 - pi_s)))
      } else {
        ifelse(treatment > 0.5, 1.0 / pi_s, 1.0 / (1.0 - pi_s))
      }
      
      max_untrimmed[s] <- max(untrimmed)
      trimmed_w <- pmin(untrimmed, max_w)
      W[s, ] <- trimmed_w
      trimmed_prop[s] <- mean(untrimmed >= max_w - 1e-6)
    }
  }

  # 6. Expand to observation level
  if (!is.null(subject_var) && subject_var %in% names(fit$data)) {
    obs_subjects <- fit$data[[subject_var]]
    subj_indices <- match(obs_subjects, subject_data[[subject_var]])
    W_obs <- W[, subj_indices, drop = FALSE]
  } else {
    W_obs <- W
  }
  
  N_obs <- ncol(W_obs)
  
  # Calculate ESS for each draw
  ess_draws <- numeric(S)
  for (s in seq_len(S)) {
    w_s <- W_obs[s, ]
    mean_w <- mean(w_s)
    sd_w <- sd(w_s)
    cv_sq <- if (mean_w > 0) (sd_w / mean_w)^2 else 0
    ess_draws[s] <- N_obs / (1 + cv_sq)
  }
  
  # Mean weight for each observation
  obs_mean_weights <- colMeans(W_obs)
  
  # 7. Print premium diagnostics report
  cat("\n=== Weight Diagnostics ===\n")
  cat(sprintf("  Exposure Type:             %s\n", if (is_continuous) "Continuous (Generalized Propensity Score)" else "Binary (Logistic propensity model)"))
  cat(sprintf("  Weight Type:               %s\n", w_type))
  cat(sprintf("  Max weight (trim):         %g\n", max_w))
  cat(sprintf("  Posterior Mean ESS:        %.1f (%.1f%% of sample size %d)\n", 
              mean(ess_draws), 100 * mean(ess_draws) / N_obs, N_obs))
  cat(sprintf("  Mean untrimmed max weight: %.1f [Range of max: %.1f, %.1f]\n", 
              mean(max_untrimmed), min(max_untrimmed), max(max_untrimmed)))
  cat(sprintf("  Mean prop. weights trimmed: %.2f%%\n", 100 * mean(trimmed_prop)))
  
  # Positivity Check
  med_max_untrimmed <- median(max_untrimmed)
  if (med_max_untrimmed > 50) {
    cat("  WARNING: High untrimmed weights suggest potential propensity score positivity violations!\n")
  } else {
    cat("  Positivity check:          Passed (no extreme untrimmed weights detected)\n")
  }
  
  cat("\n  Observation-level weight distribution summary:\n")
  smry <- quantile(obs_mean_weights, probs = c(0, 0.25, 0.5, 0.75, 1))
  cat(sprintf("    Min: %.3f | 25%%: %.3f | Median: %.3f | 75%%: %.3f | Max: %.3f\n\n",
              smry[1], smry[2], smry[3], smry[4], smry[5]))
  
  # Text-based histogram of mean weights
  if (sd(obs_mean_weights) > 1e-6) {
    bins <- cut(obs_mean_weights, breaks = seq(min(obs_mean_weights), max(obs_mean_weights), length.out = 6), include.lowest = TRUE)
    bin_counts <- table(bins)
    cat("  Weight distribution histogram (text-based):\n")
    for (bin_name in names(bin_counts)) {
      count <- bin_counts[bin_name]
      pct <- 100 * count / length(obs_mean_weights)
      bar <- paste(rep("■", round(pct / 4)), collapse = "")
      cat(sprintf("    %-18s : %5.1f%% | %s\n", bin_name, pct, bar))
    }
  } else {
    cat("  Weight distribution histogram: all weights are identical (1.0).\n")
  }
  cat("\n")

  invisible(list(
    weights = obs_mean_weights,
    ess = ess_draws,
    trimmed_prop = trimmed_prop,
    max_untrimmed = max_untrimmed
  ))
}

#' Extract Posterior Inclusion Probabilities from a spike-and-slab bjlm fit
#'
#' Computes the posterior inclusion probability (PIP) for each coefficient
#' that was assigned a spike-and-slab prior. The result carries the
#' `smoothbp_pip` class (from the \pkg{smoothbp} package) so that
#' [plot.smoothbp_pip()] can be used directly.
#'
#' @param x A `bjlm_fit` object fitted with `spike = prior_spike_slab(...)`.
#' @param ... Unused.
#'
#' @return A data frame of class `smoothbp_pip` with columns `parameter`,
#'   `pip`, `lower`, and `upper` (95\% credible interval on the inclusion
#'   probability, derived from a Beta posterior).
#' @seealso [prior_spike_slab()], [plot.smoothbp_pip()]
#' @export
pip <- function(x, ...) UseMethod("pip")

#' @export
pip.bjlm_fit <- function(x, ...) {
  if (is.null(x$spike)) {
    stop("`pip()` requires a model fitted with `spike = prior_spike_slab(...)`. ",
         "This fit has no spike-and-slab prior.")
  }

  gamma_cols <- grep("^gamma_", x$outcome_names, value = TRUE)
  if (length(gamma_cols) == 0L) {
    stop("No gamma columns found in draws. ",
         "Ensure the model was fitted with `spike = prior_spike_slab(...)`.")
  }

  draws_mat <- as.matrix(posterior::subset_draws(x$draws, variable = gamma_cols))

  pip_vals <- colMeans(draws_mat)
  n_draws  <- nrow(draws_mat)

  # Beta posterior CI: Beta(1 + n1, 1 + n0)
  n1 <- round(pip_vals * n_draws)
  n0 <- n_draws - n1
  lower <- qbeta(0.025, n1 + 1, n0 + 1)
  upper <- qbeta(0.975, n1 + 1, n0 + 1)

  # Strip "gamma_" prefix for display
  param_names <- sub("^gamma_", "", gamma_cols)

  result <- data.frame(
    parameter = param_names,
    pip       = pip_vals,
    lower     = lower,
    upper     = upper,
    stringsAsFactors = FALSE
  )
  rownames(result) <- NULL
  structure(result, class = c("smoothbp_pip", "data.frame"))
}

