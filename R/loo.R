#' Extract pointwise log-likelihood from a bjlm_fit object
#'
#' @param object A \code{bjlm_fit} object.
#' @param ... Additional arguments passed to \code{fitted}.
#'
#' @return A matrix of size S x N containing pointwise log-likelihoods.
#' @export
log_lik.bjlm_fit <- function(object, ...) {
  # Use Rust-side pre-computed log-lik matrix if available
  # (includes GP contributions that the R-side fitted() misses)
  if (!is.null(object$log_lik_matrix)) {
    ll_matrix <- object$log_lik_matrix
  } else {
    # Fallback: R-side computation (for backward compatibility with old fits)
    family <- object$outcome_family
    if (is.null(family)) {
      if (!is.null(object$model$outcome$family$family)) {
        family <- object$model$outcome$family$family
      } else {
        family <- "gaussian"
      }
    }
    
    # Get predictions on the link scale (log-mean for negative binomial, log-odds for binomial)
    pred_draws <- fitted(object, summary = FALSE, type = "link", ...)
    
    outcome_var <- object$outcome_var
    if (is.null(outcome_var)) {
      outcome_var <- all.vars(object$outcome_formula)[1]
    }
    y_obs <- as.double(object$data[[outcome_var]])
    
    S <- nrow(pred_draws)
    N <- ncol(pred_draws)
    ll_matrix <- matrix(NA_real_, nrow = S, ncol = N)
    
    draws_mat <- posterior::as_draws_matrix(object$draws)
    
    if (family == "gaussian") {
      sigma_draws <- as.numeric(draws_mat[, "sigma"])
      for (i in seq_len(N)) {
        ll_matrix[, i] <- stats::dnorm(y_obs[i], mean = pred_draws[, i], sd = sigma_draws, log = TRUE)
      }
    } else if (family == "negative_binomial") {
      r_draws <- as.numeric(draws_mat[, "r"])
      mu_draws <- exp(pred_draws) # Exponentiate link-scale to get response-scale mean mu
      for (i in seq_len(N)) {
        ll_matrix[, i] <- stats::dnbinom(y_obs[i], size = r_draws, mu = mu_draws[, i], log = TRUE)
      }
    } else if (family == "binomial") {
      p_draws <- 1 / (1 + exp(-pred_draws)) # Sigmoid link-scale to get probability
      trials <- rep(1, N)
      for (i in seq_len(N)) {
        ll_matrix[, i] <- stats::dbinom(y_obs[i], size = trials[i], prob = p_draws[, i], log = TRUE)
      }
    } else {
      stop("Unsupported family for log_lik: ", family)
    }
  }

  # Apply weighting if applicable
  weight_type <- object$weight_type
  
  # Check if there is treatment variation in the data
  has_trt_variation <- TRUE
  if (!is.null(object$propensity_formula)) {
    prop_vars <- all_vars(object$propensity_formula)
    if (length(prop_vars) > 0) {
      trt_var <- prop_vars[1]
      if (trt_var %in% names(object$data)) {
        trt_vals <- object$data[[trt_var]]
        if (length(unique(trt_vals)) <= 1) {
          has_trt_variation <- FALSE
        }
      }
    }
  }

  if (!is.null(weight_type) && weight_type != "none" && has_trt_variation) {
    # Reconstruct weights from the draws of alpha
    prop_vars <- all.vars(object$propensity_formula)
    trt_var <- prop_vars[1]
    cov_vars <- prop_vars[-1]

    subject_var <- object$subject_var
    if (!is.null(subject_var)) {
      group_factor <- as.factor(object$data[[subject_var]])
      first_idx <- !duplicated(object$data[[subject_var]])
      subject_data <- object$data[first_idx, , drop = FALSE]
      subject_data <- subject_data[match(levels(group_factor), subject_data[[subject_var]]), , drop = FALSE]
    } else {
      subject_data <- object$data
    }

    x_prop <- stats::model.matrix(stats::reformulate(cov_vars), data = subject_data)
    draws_mat <- posterior::as_draws_matrix(object$draws)
    alpha_cols <- grep("^alpha_", colnames(draws_mat), value = TRUE)
    alpha_draws <- as.matrix(draws_mat[, alpha_cols, drop = FALSE])

    eta <- alpha_draws %*% t(x_prop)
    pi_hat <- 1 / (1 + exp(-eta))
    trt <- subject_data[[trt_var]]
    w_draws <- matrix(0, nrow = nrow(pi_hat), ncol = ncol(pi_hat))

    max_weight <- object$max_weight

    for (i in seq_along(trt)) {
      if (trt[i] == 1) {
        w_draws[, i] <- 1 / pi_hat[, i]
      } else {
        w_draws[, i] <- 1 / (1 - pi_hat[, i])
      }
      if (!is.null(max_weight)) {
        w_draws[, i] <- pmin(w_draws[, i], max_weight)
      }
    }

    if (!is.null(subject_var)) {
      obs_group <- as.integer(as.factor(object$data[[subject_var]]))
      w_obs_draws <- w_draws[, obs_group, drop = FALSE]
    } else {
      w_obs_draws <- w_draws
    }

    w_obs <- colMeans(w_obs_draws)
    
    # Scale log-likelihood by the posterior mean weights
    ll_matrix <- sweep(ll_matrix, 2, w_obs, "*")
  }
  
  colnames(ll_matrix) <- paste0("y[", seq_len(ncol(ll_matrix)), "]")
  return(ll_matrix)
}

#' Extract pointwise log-likelihood from a smoothbp_fit object
#'
#' @param object A \code{smoothbp_fit} object.
#' @param ... Additional arguments passed to \code{fitted}.
#'
#' @return A matrix of size S x N containing pointwise log-likelihoods.
#' @export
log_lik.smoothbp_fit <- function(object, ...) {
  log_lik.bjlm_fit(object, ...)
}

#' Compute LOO-IC for bjlm_fit objects
#' 
#' @param x A `bjlm_fit` object.
#' @param ... Additional arguments passed to `loo::loo`.
#' @export
loo.bjlm_fit <- function(x, ...) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("The 'loo' package is required.")
  }
  ll <- log_lik(x)
  loo::loo(ll, ...)
}

#' Compute WAIC for bjlm_fit objects
#' 
#' @param x A `bjlm_fit` object.
#' @param ... Additional arguments passed to `loo::waic`.
#' @export
waic.bjlm_fit <- function(x, ...) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("The 'loo' package is required.")
  }
  ll <- log_lik(x)
  loo::waic(ll, ...)
}

#' Compute LOO-IC for smoothbp_fit objects
#' 
#' @param x A `smoothbp_fit` object.
#' @param ... Additional arguments passed to `loo::loo`.
#' @export
loo.smoothbp_fit <- function(x, ...) {
  loo.bjlm_fit(x, ...)
}

#' Compute WAIC for smoothbp_fit objects
#' 
#' @param x A `smoothbp_fit` object.
#' @param ... Additional arguments passed to `loo::waic`.
#' @export
waic.smoothbp_fit <- function(x, ...) {
  waic.bjlm_fit(x, ...)
}
