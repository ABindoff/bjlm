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
