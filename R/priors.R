#' Specify a normal (or truncated normal) prior for a regression coefficient
#'
#' @param mean Prior mean. Default 0.
#' @param sd Prior standard deviation. Default 1.
#' @param lb Lower bound (use `-Inf` for unconstrained). Default `-Inf`.
#' @param ub Upper bound (use `Inf` for unconstrained). Default `Inf`.
#'
#' @return A `smoothbp_prior` object.
#' @export
prior_normal <- function(mean = 0, sd = 1, lb = -Inf, ub = Inf) {
  stopifnot(sd >= 0, lb < ub)
  structure(
    list(family = "normal", mean = mean, sd = sd, lb = lb, ub = ub),
    class = "smoothbp_prior"
  )
}

#' Fix a parameter at a specific value
#'
#' Used within `omega` or `rho` lists in \code{\link{bjlm}} to specify that a
#' parameter is fixed and should not be estimated.
#'
#' @param value The fixed value(s) (numeric scalar or vector).
#'
#' @return A `smoothbp_fixed` object.
#' @export
fixed <- function(value) {
  stopifnot(is.numeric(value))
  structure(value, class = "smoothbp_fixed")
}

#' @rdname fixed
#' @export
prior_fixed <- function(value) {
  fixed(value)
}

#' Specify an inverse-gamma prior for a variance component
#'
#' @param shape Shape parameter (> 0).
#' @param scale Scale parameter (> 0).
#'
#' @return A `smoothbp_prior` object.
#' @export
prior_invgamma <- function(shape = 1, scale = 1) {
  stopifnot(shape > 0, scale > 0)
  structure(
    list(family = "invgamma", shape = shape, scale = scale),
    class = "smoothbp_prior"
  )
}

#' Specify a half-Cauchy prior for a scale (standard-deviation) parameter
#'
#' Weakly-informative prior for a hierarchical standard deviation (Gelman 2006).
#' Unlike an inverse-gamma prior on the variance, it places mass arbitrarily close
#' to zero, so it does not impose a spurious floor on the estimated SD. Sampled via
#' the inverse-gamma parameter-expansion (Wand 2011), so the Gibbs updates stay
#' conjugate.
#'
#' @param scale Half-Cauchy scale `A` (> 0); larger is more diffuse.
#'
#' @return A `smoothbp_prior` object.
#' @export
prior_halfcauchy <- function(scale = 1) {
  stopifnot(scale > 0)
  structure(
    # `shape` is an unused placeholder for the C interface; the half-Cauchy is
    # fully determined by `scale` (which the Rust sampler reads as A).
    list(family = "halfcauchy", shape = 0, scale = scale),
    class = "smoothbp_prior"
  )
}

#' Specify a gamma prior for a parameter
#'
#' @param shape Shape parameter (> 0).
#' @param scale Scale parameter (> 0).
#'
#' @return A `smoothbp_prior` object.
#' @export
prior_gamma <- function(shape = 1, scale = 1) {
  stopifnot(shape > 0, scale > 0)
  structure(
    list(family = "gamma", shape = shape, scale = scale),
    class = "smoothbp_prior"
  )
}

#' Specify a log-normal prior for a positive parameter
#'
#' `ln(theta) ~ Normal(meanlog, sdlog)`. Natural weakly-informative prior for a
#' positive scale or lengthscale; sampled exactly on the log scale.
#'
#' @param meanlog,sdlog Mean and SD of the underlying normal on the log scale.
#' @return A `smoothbp_prior` object.
#' @export
prior_lognormal <- function(meanlog = 0, sdlog = 1) {
  stopifnot(sdlog > 0)
  structure(list(family = "lognormal", meanlog = meanlog, sdlog = sdlog),
            class = "smoothbp_prior")
}

#' Specify a half-normal prior for a scale (standard-deviation) parameter
#'
#' @param sd Scale of the half-normal (> 0); larger is more diffuse.
#' @return A `smoothbp_prior` object.
#' @export
prior_halfnormal <- function(sd = 1) {
  stopifnot(sd > 0)
  structure(list(family = "halfnormal", sd = sd), class = "smoothbp_prior")
}

#' Specify a half-t prior for a scale (standard-deviation) parameter
#'
#' Heavier-tailed weakly-informative alternative to the half-normal; `df = 1` is the
#' half-Cauchy.
#'
#' @param df Degrees of freedom (> 0).
#' @param scale Scale (> 0).
#' @return A `smoothbp_prior` object.
#' @export
prior_halft <- function(df = 3, scale = 1) {
  stopifnot(df > 0, scale > 0)
  structure(list(family = "halft", df = df, scale = scale), class = "smoothbp_prior")
}

#' Resolution-aware GP lengthscale prior (default for `rho` in [gp_priors()])
#'
#' A log-normal on the GP lengthscale whose location and scale are set from the data's
#' time grid: the band runs from the median inter-point spacing (below which the GP
#' cannot be resolved and absorbs observation noise, biasing the noise SD low) to the
#' range (above which the GP is indistinguishable from a constant). Prevents the
#' sub-resolution regime that a fixed, scale-blind lengthscale prior admits.
#'
#' @return A `smoothbp_prior` object.
#' @export
prior_lengthscale <- function() {
  structure(list(family = "lengthscale"), class = "smoothbp_prior")
}

# Integer family codes shared with the Rust sampler (see log_scale_prior()).
.gp_prior_code <- c(lognormal = 0L, halfnormal = 1L, halfcauchy = 2L,
                    invgamma = 3L, gamma = 4L, halft = 5L, lengthscale = 6L)

# Encode a GP hyperprior as c(family_code, p1, p2) for the FFI.
.gp_prior_encode <- function(p) {
  vals <- switch(p$family,
    lognormal   = c(p$meanlog, p$sdlog),
    halfnormal  = c(p$sd, 0),
    halfcauchy  = c(p$scale, 0),
    invgamma    = c(p$shape, p$scale),
    gamma       = c(p$shape, p$scale),
    halft       = c(p$df, p$scale),
    lengthscale = c(0, 0),
    stop(sprintf("Unsupported GP prior family '%s'.", p$family)))
  as.double(c(.gp_prior_code[[p$family]], vals))
}

#' Priors for the latent Gaussian-process hyperparameters
#'
#' Bundle of priors for a latent-GP block (see [latent_gp()]). Each prior compiles to
#' a fast, closed-form log-density on the log scale, so the sampler stays exact and
#' cannot panic on an arbitrary distribution.
#'
#' @param alpha Prior for the GP marginal SD. A standard-deviation family:
#'   [prior_lognormal()], [prior_halfnormal()], [prior_halfcauchy()], [prior_halft()],
#'   or [prior_gamma()].
#' @param rho Prior for the GP lengthscale: [prior_lengthscale()] (resolution-aware,
#'   the recommended default), [prior_lognormal()], [prior_invgamma()] (Betancourt's
#'   lengthscale prior), [prior_gamma()], or [prior_halft()].
#' @param sigma_x Prior for the GP observation-noise SD. Same family set as `alpha`.
#'
#' @details Inverse-gamma is offered only for `rho` (directly on the lengthscale); it
#'   is deliberately not offered for the SD slots, where it is conventionally placed on
#'   the variance and would be ambiguous.
#'
#' @return A `gp_priors` object.
#' @export
gp_priors <- function(alpha   = prior_lognormal(0, 1),
                      rho     = prior_lengthscale(),
                      sigma_x = prior_lognormal(-1, 1)) {
  sd_ok  <- c("lognormal", "halfnormal", "halfcauchy", "halft", "gamma")
  rho_ok <- c("lengthscale", "lognormal", "invgamma", "gamma", "halft")
  chk <- function(p, nm, ok) {
    if (!inherits(p, "smoothbp_prior"))
      stop(sprintf("`%s` must be a prior object (e.g. prior_lognormal()).", nm))
    if (!p$family %in% ok)
      stop(sprintf("`%s` prior family '%s' is not supported; choose one of: %s.",
                   nm, p$family, paste(ok, collapse = ", ")))
  }
  chk(alpha, "alpha", sd_ok)
  chk(rho, "rho", rho_ok)
  chk(sigma_x, "sigma_x", sd_ok)
  structure(list(alpha = alpha, rho = rho, sigma_x = sigma_x), class = "gp_priors")
}

#' @export
print.smoothbp_prior <- function(x, ...) {
  if (x$family == "normal") {
    cat(sprintf("Normal(mean=%g, sd=%g", x$mean, x$sd))
    if (is.finite(x$lb) || is.finite(x$ub)) {
      cat(sprintf(", lb=%s, ub=%s", format(x$lb), format(x$ub)))
    }
    cat(")\n")
  } else if (x$family == "invgamma") {
    cat(sprintf("InvGamma(shape=%g, scale=%g)\n", x$shape, x$scale))
  } else if (x$family == "gamma") {
    cat(sprintf("Gamma(shape=%g, scale=%g)\n", x$shape, x$scale))
  } else if (x$family == "halfcauchy") {
    cat(sprintf("HalfCauchy(scale=%g)\n", x$scale))
  } else if (x$family == "lognormal") {
    cat(sprintf("LogNormal(meanlog=%g, sdlog=%g)\n", x$meanlog, x$sdlog))
  } else if (x$family == "halfnormal") {
    cat(sprintf("HalfNormal(sd=%g)\n", x$sd))
  } else if (x$family == "halft") {
    cat(sprintf("HalfT(df=%g, scale=%g)\n", x$df, x$scale))
  } else if (x$family == "lengthscale") {
    cat("Lengthscale(resolution-aware)\n")
  }
  invisible(x)
}

#' @export
print.gp_priors <- function(x, ...) {
  cat("GP hyperpriors:\n")
  for (nm in c("alpha", "rho", "sigma_x")) {
    cat(sprintf("  %-8s: ", nm)); print(x[[nm]])
  }
  invisible(x)
}

#' Collect priors for all model parameters
#'
#' Each argument accepts either:
#' - A single `prior_normal()` applied to all coefficients of that parameter, or
#' - A named list mapping coefficient names (matching column names of the design
#'   matrix) to individual `prior_normal()` objects.
#'
#' For multi-breakpoint models, `deltas`, `omega`, and `rho` can also be
#' **lists of prior specifications** (one per breakpoint slot). If a single
#' specification is provided, it is applied to all slots.
#'
#' @param b0      Prior(s) for `b0` regression coefficients.
#' @param b1      Prior(s) for `b1` regression coefficients.
#' @param deltas  Prior(s) for slope change coefficients (one list per segment).
#' @param omega   Prior(s) for `omega` coefficients (one list per segment).
#' @param rho     Prior(s) for `rho` coefficients (one list per segment).
#' @param sigma   `prior_invgamma()` for residual SD.
#' @param sigma_u `prior_halfcauchy()` for the random-effect SD. Half-Cauchy
#'   avoids the spurious near-zero floor an inverse-gamma variance prior imposes.
#' @param sigma_re_om `prior_halfcauchy()` for the random change-point SD on omega.
#' @param r `prior_gamma()` for Negative Binomial overdispersion parameter.
#'
#' @return A `smoothbp_priors` list.
#' @export
smoothbp_priors <- function(
    b0      = prior_normal(0, 10),
    b1      = prior_normal(0, 2),
    deltas  = prior_normal(0, 2),
    omega   = prior_normal(3, 2, lb = 0),
    rho     = prior_normal(3, 2, lb = 0),
    sigma   = prior_invgamma(1, 1),
    sigma_u = prior_halfcauchy(1),
    sigma_re_om = prior_halfcauchy(1),
    r       = prior_gamma(1, 1)
) {
  if (!inherits(sigma, "smoothbp_prior") || sigma$family != "invgamma")
    stop("`sigma` (residual SD) must be prior_invgamma().")
  if (!inherits(r, "smoothbp_prior") || r$family != "gamma")
    stop("`r` (NB overdispersion) must be prior_gamma().")
  # Random-effect SDs use a half-Cauchy(0, scale) prior via the inverse-gamma
  # auxiliary sampler (Wand 2011); the sampler does not read an inverse-gamma
  # here, so we reject it rather than silently ignore it. Half-Cauchy also avoids
  # the spurious near-zero floor an IG variance prior imposes on a hierarchical SD.
  for (nm in c("sigma_u", "sigma_re_om")) {
    p <- get(nm)
    if (!inherits(p, "smoothbp_prior") || p$family != "halfcauchy")
      stop(sprintf(
        "`%s` must be prior_halfcauchy(scale=): random-effect SDs use a half-Cauchy prior. %s",
        nm,
        if (inherits(p, "smoothbp_prior") && p$family == "invgamma")
          "Inverse-gamma is no longer supported for RE SDs (it imposes a near-zero variance floor)."
        else ""))
  }
  structure(
    list(b0 = b0, b1 = b1, deltas = deltas, omega = omega, rho = rho,
         sigma = sigma, sigma_u = sigma_u, sigma_re_om = sigma_re_om, r = r),
    class = "smoothbp_priors"
  )
}

#' @export
print.smoothbp_priors <- function(x, ...) {
  cat("smoothbp priors:\n")
  for (nm in c("b0", "b1", "deltas", "omega", "rho", "sigma", "sigma_u", "sigma_re_om")) {
    cat(sprintf("  %-8s: ", nm))
    if (is.list(x[[nm]]) && !inherits(x[[nm]], "smoothbp_prior")) {
        cat("<list of priors>\n")
    } else {
        print(x[[nm]])
    }
  }
  invisible(x)
}

#' Generate evenly spaced priors for candidate breakpoints
#'
#' This helper function generates a list of `prior_normal` objects for `omega`
#' (breakpoint locations) that are evenly spaced across the range of your time
#' variable `tau`. This is highly recommended when using `smoothbp_ss()` to
#' ensure the candidate breakpoints cover the entire domain without clumping.
#'
#' For hierarchical models where `omega` has random effects (e.g., `~ 1 + (1 | group)`),
#' this function automatically names the prior `(Intercept)` so it applies correctly
#' to the global market mean, while the random effects are handled automatically by
#' the `sigma_re_om` shrinkage variance.
#'
#' @param K Number of candidate breakpoints.
#' @param tau_min Minimum value of the time/covariate variable.
#' @param tau_max Maximum value of the time/covariate variable.
#'
#' @return A list of length `K` containing prior specifications for `omega`.
#' @export
space_omega_priors <- function(K, tau_min, tau_max) {
  stopifnot(K >= 1, tau_max > tau_min)
  
  # Pad the edges so we don't push breakpoints right to the very limits
  means <- seq(tau_min, tau_max, length.out = K + 2)[2:(K + 1)]
  
  # Standard deviation heuristic: width of interval / K
  sd_val <- (tau_max - tau_min) / K
  
  lapply(means, function(m) {
    list(
      "(Intercept)" = prior_normal(mean = m, sd = sd_val, lb = tau_min, ub = tau_max)
    )
  })
}

# ---------------------------------------------------------------------------
# Internal helpers: expand priors to per-coefficient vectors
# ---------------------------------------------------------------------------

.expand_prior <- function(prior_spec, coef_names) {
  n <- length(coef_names)
  if (n == 0) return(data.frame(name=character(), mean=numeric(), sd=numeric(), lb=numeric(), ub=numeric(), stringsAsFactors=FALSE))
  if (inherits(prior_spec, "smoothbp_prior") && prior_spec$family == "normal") {
    data.frame(
      name = coef_names,
      mean = prior_spec$mean,
      sd   = prior_spec$sd,
      lb   = prior_spec$lb,
      ub   = prior_spec$ub,
      stringsAsFactors = FALSE
    )
  } else if (is.list(prior_spec) && !inherits(prior_spec, "smoothbp_prior")) {
    default <- prior_spec[["."]] %||% prior_normal(0, 10)
    out <- data.frame(
      name = coef_names,
      mean = default$mean,
      sd   = default$sd,
      lb   = default$lb,
      ub   = default$ub,
      stringsAsFactors = FALSE
    )
    for (nm in intersect(names(prior_spec), coef_names)) {
      p <- prior_spec[[nm]]
      idx <- which(coef_names == nm)
      out$mean[idx] <- p$mean
      out$sd[idx]   <- p$sd
      out$lb[idx]   <- p$lb
      out$ub[idx]   <- p$ub
    }
    out
  } else {
    stop("Prior must be a prior_normal() or a named list of prior_normal() objects.")
  }
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

#' Specify a spike-and-slab prior for variable selection
#'
#' Used with spike-and-slab models to place a point-mass spike at zero on selected
#' coefficients (Kuo-Mallick formulation). The underlying Rust samplers
#' (`sample_pi`, `sample_gamma`) and the `smoothbp_spike_slab` / `smoothbp_pip`
#' classes originate from the \pkg{smoothbp} package (Bindoff 2025).
#'
#' @param pi Prior inclusion probability. Default `0.5`.
#' @param slab A [prior_normal()] object for the slab component.
#' @param learn_pi Logical; if `TRUE`, place a `Beta(a, b)` hyperprior on pi.
#' @param a Shape parameter for the Beta hyperprior. Default `1`.
#' @param b Shape parameter for the Beta hyperprior. Default `1`.
#'
#' @return A `smoothbp_spike_slab` object.
#' @seealso [pip()] to extract posterior inclusion probabilities after fitting.
#' @export
prior_spike_slab <- function(pi = 0.5, slab = prior_normal(0, 2),
                             learn_pi = FALSE, a = 1, b = 1) {
  stopifnot(
    inherits(slab, "smoothbp_prior"),
    slab$family == "normal",
    is.numeric(pi),
    all(pi > 0 & pi < 1),
    is.logical(learn_pi)
  )
  structure(
    list(family = "spike_slab", pi = pi, slab = slab,
         learn_pi = learn_pi, a = a, b = b),
    class = "smoothbp_spike_slab"
  )
}

#' @export
print.smoothbp_spike_slab <- function(x, ...) {
  if (x$learn_pi) {
    cat(sprintf("SpikeSlab(pi~Beta(%g,%g), slab=Normal(%g, %g), spike_intercept=%s)\n",
                x$a, x$b, x$slab$mean, x$slab$sd, x$spike_intercept))
  } else {
    cat(sprintf("SpikeSlab(pi=%s, slab=Normal(%g, %g), spike_intercept=%s)\n",
                paste(format(x$pi), collapse = ","),
                x$slab$mean, x$slab$sd, x$spike_intercept))
  }
  invisible(x)
}

#' Collect priors for both outcome and propensity models in a joint BJLM model
#'
#' @param outcome An object of class \code{smoothbp_priors} specifying priors
#'   for the outcome model. See \code{\link{smoothbp_priors}} for details.
#' @param propensity An object of class \code{smoothbp_prior} (of normal family)
#'   specifying the prior for the propensity coefficients. Default is \code{prior_normal(0, 2.5)}.
#'
#' @return A \code{bjlm_priors} list.
#' @export
bjlm_priors <- function(
    outcome = smoothbp_priors(),
    propensity = prior_normal(0, 2.5)
) {
  stopifnot(
    "outcome must be a smoothbp_priors object" = inherits(outcome, "smoothbp_priors"),
    "propensity must be a normal smoothbp_prior" = inherits(propensity, "smoothbp_prior") && propensity$family == "normal"
  )
  structure(
    list(outcome = outcome, propensity = propensity),
    class = "bjlm_priors"
  )
}

#' Alias for smoothbp_priors
#'
#' @inheritParams smoothbp_priors
#' @export
bjlm_outcome_priors <- function(
    b0      = prior_normal(0, 10),
    b1      = prior_normal(0, 2),
    deltas  = prior_normal(0, 2),
    omega   = prior_normal(3, 2, lb = 0),
    rho     = prior_normal(3, 2, lb = 0),
    sigma   = prior_invgamma(1, 1),
    sigma_u = prior_halfcauchy(1),
    sigma_re_om = prior_halfcauchy(1),
    r       = prior_gamma(1, 1)
) {
  smoothbp_priors(
    b0 = b0, b1 = b1, deltas = deltas, omega = omega, rho = rho,
    sigma = sigma, sigma_u = sigma_u, sigma_re_om = sigma_re_om, r = r
  )
}

#' @export
print.bjlm_priors <- function(x, ...) {
  cat("bjlm priors:\n")
  cat("  propensity model:\n")
  cat("    ")
  print(x$propensity)
  cat("  outcome model:\n")
  print(x$outcome)
  invisible(x)
}
