#' Extract pointwise log-likelihood from a bjlm_fit object
#'
#' @param object A \code{bjlm_fit} object.
#' @param ... Additional arguments passed to \code{fitted}.
#'
#' @return A matrix of size S x N containing pointwise log-likelihoods.
#' @export
log_lik.bjlm_fit <- function(object, ...) {
  family <- object$outcome_family
  if (is.null(family)) {
    family <- "gaussian"
  }
  
  pred_draws <- fitted(object, summary = FALSE, ...)
  y_obs <- as.double(object$data[[object$outcome_var]])
  
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
    for (i in seq_len(N)) {
      ll_matrix[, i] <- stats::dnbinom(y_obs[i], size = r_draws, mu = pred_draws[, i], log = TRUE)
    }
  } else if (family == "binomial") {
    # Default to Bernoulli trials = 1
    trials <- rep(1, N)
    for (i in seq_len(N)) {
      ll_matrix[, i] <- stats::dbinom(y_obs[i], size = trials[i], prob = pred_draws[, i], log = TRUE)
    }
  } else {
    stop("Unsupported family for log_lik: ", family)
  }
  
  colnames(ll_matrix) <- paste0("y[", seq_len(N), "]")
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

#' Compute LOO-IC for smoothbp_fit objects
#' 
#' @param x A `smoothbp_fit` object.
#' @param ... Additional arguments passed to `loo::loo`.
#' @export
loo.smoothbp_fit <- function(x, ...) {
  loo.bjlm_fit(x, ...)
}
