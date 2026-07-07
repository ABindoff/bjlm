# Bridge sampling / Bayes factor methods for bjlm.
#
# Marginal-likelihood and Bayes-factor computation are NOT supported for bjlm_fit
# objects. The outcome model is an IPW-weighted marginal structural model, i.e. a
# pseudo-posterior whose normalising constant is not a Bayesian marginal
# likelihood, so a Bayes factor from it is not well defined; spike-and-slab fits
# additionally contain discrete inclusion indicators that bridge sampling cannot
# marginalise. The methods below therefore error with guidance rather than return a
# plausible-looking but meaningless value. (The legacy smoothbp_fit bridge sampler
# and its Gaussian log-posterior were removed: this package never constructs a
# smoothbp_fit, and the log-posterior modelled neither the IPW weighting nor the
# NB/GP families, so it could not produce a valid marginal likelihood here.)
#
# For predictive model comparison use lfo_cv() or loo(); for variable selection
# use pip().

.fit_has_spike_slab <- function(fit) {
  if (!is.null(fit$spike)) return(TRUE)
  nm <- tryCatch(colnames(posterior::as_draws_matrix(fit$draws)),
                 error = function(e) character(0))
  any(grepl("^gamma_", nm))
}

.bridge_bjlm_msg <- function(fit) {
  paste0(
    "bridge_sampler()/bayes_factor() are not supported for bjlm_fit objects: the ",
    "outcome model is an IPW-weighted marginal structural model (a pseudo-posterior), ",
    "so its normalising constant is not a Bayesian marginal likelihood and a Bayes ",
    "factor from it is not well defined",
    if (.fit_has_spike_slab(fit))
      " (this fit also uses spike-and-slab, whose discrete indicators cannot be marginalised)"
    else "",
    ". For predictive model comparison use lfo_cv() or loo(); for variable selection use pip()."
  )
}

#' Bridge sampling / Bayes factors for bjlm_fit objects (unsupported)
#'
#' Marginal-likelihood and Bayes-factor computation are not supported for
#' \code{bjlm_fit} objects. The outcome model is an IPW-weighted marginal
#' structural model (a pseudo-posterior), so its normalising constant is not a
#' Bayesian marginal likelihood; spike-and-slab fits additionally contain discrete
#' inclusion indicators that bridge sampling cannot marginalise. These methods
#' therefore error with guidance. For predictive model comparison use
#' \code{\link{lfo_cv}} or \code{\link[loo]{loo}}; for variable selection use
#' \code{\link{pip}}.
#'
#' @param samples,x1,x2 A \code{bjlm_fit} object.
#' @param log,... Unused.
#' @return These methods do not return; they stop with an explanatory error.
#' @name bridge_sampler_bjlm
#' @importFrom bridgesampling bridge_sampler
#' @method bridge_sampler bjlm_fit
#' @export
bridge_sampler.bjlm_fit <- function(samples, ...) {
  stop(.bridge_bjlm_msg(samples), call. = FALSE)
}

#' @rdname bridge_sampler_bjlm
#' @importFrom bridgesampling bayes_factor
#' @method bayes_factor bjlm_fit
#' @export
bayes_factor.bjlm_fit <- function(x1, x2, log = FALSE, ...) {
  stop(.bridge_bjlm_msg(x1), call. = FALSE)
}
