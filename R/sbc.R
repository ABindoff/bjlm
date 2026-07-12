# =============================================================================
# Simulation-based calibration (SBC; Talts et al. 2018) for bjlm models.
#
# SBC checks whether the sampler + priors actually recover truth on data like
# the user's. The contract is three small, reusable pieces:
#   .sbc_draw_prior()  draws a full parameter set from the SAME priors the fit
#                      will use (guaranteed via the shared prior-vector helper).
#   .sbc_simulate()    simulates a dataset from that parameter set, reusing the
#                      fitter's own mean function (.build_predictions) so the
#                      generative and inferential means cannot drift apart.
#   .sbc_align_draws() / .sbc_score_functional()  extract named posterior
#                      quantities aligned to the truth and rank them.
# The public entry point is sbc(), a generic over compiled models and fits.
# =============================================================================

# ---- small numeric helpers --------------------------------------------------

# Draw one truncated normal by inverse-CDF; sd == 0 returns the (fixed) mean.
.sbc_rtnorm <- function(mean, sd, lb = -Inf, ub = Inf) {
  if (!is.finite(sd) || sd <= 0) return(mean)
  lo <- stats::pnorm(lb, mean, sd)
  hi <- stats::pnorm(ub, mean, sd)
  if (hi <= lo) return(mean)
  stats::qnorm(stats::runif(1, lo, hi), mean, sd)
}

# Draw a whole per-coefficient block (a data.frame with name/mean/sd/lb/ub).
.sbc_draw_block <- function(df) {
  if (nrow(df) == 0) return(stats::setNames(numeric(0), character(0)))
  vals <- vapply(seq_len(nrow(df)),
                 function(i) .sbc_rtnorm(df$mean[i], df$sd[i], df$lb[i], df$ub[i]),
                 numeric(1))
  stats::setNames(vals, df$name)
}

# Resolution-aware GP lengthscale prior params (mirror of the Rust
# gp_rho_log_prior_params(): ln(rho) ~ Normal(loc, scale)).
.sbc_lengthscale_params <- function(time_grid) {
  g <- sort(unique(as.numeric(time_grid)))
  spacing <- stats::median(diff(g))
  rng <- max(g) - min(g)
  loc <- 0.5 * (log(spacing) + log(rng))
  scale <- max(0.35, 0.25 * log(rng / spacing))
  list(loc = loc, scale = scale)
}

# Draw one GP hyperparameter from its prior (SD slots and the lengthscale).
.sbc_draw_gp_hyper <- function(prior, time_grid = NULL) {
  switch(prior$family,
    lognormal   = exp(stats::rnorm(1, prior$meanlog, prior$sdlog)),
    halfnormal  = abs(stats::rnorm(1, 0, prior$sd)),
    halfcauchy  = abs(stats::rcauchy(1, 0, prior$scale)),
    halft       = abs(prior$scale * stats::rt(1, prior$df)),
    gamma       = stats::rgamma(1, shape = prior$shape, scale = prior$scale),
    invgamma    = 1 / stats::rgamma(1, shape = prior$shape, rate = prior$scale),
    lengthscale = {
      p <- .sbc_lengthscale_params(time_grid); exp(stats::rnorm(1, p$loc, p$scale))
    },
    stop(sprintf("Cannot draw GP hyperparameter from prior family '%s'.", prior$family))
  )
}

# Build the outcome design matrices exactly as bjlm() does (same column names
# and ordering), from a compiled model's stored formulas + a data frame.
.sbc_design <- function(cm, data) {
  b0_fixed <- .parse_re(cm$b0_formula)$fixed
  x_b0 <- stats::model.matrix(b0_fixed, data = data)
  x_b1 <- if (is.null(cm$b1_formula)) stats::model.matrix(~ 0, data = data) else
            stats::model.matrix(cm$b1_formula, data = data)
  n_bp <- length(cm$deltas)
  x_deltas <- if (n_bp > 0) lapply(cm$deltas, function(f) stats::model.matrix(f, data = data)) else list()
  x_om     <- if (n_bp > 0) lapply(cm$omega,  function(f) .build_mm(f, data)) else list()
  x_rho    <- if (n_bp > 0) lapply(cm$rho,    function(f) stats::model.matrix(f, data = data)) else list()
  list(x_b0 = x_b0, x_b1 = x_b1, x_deltas = x_deltas, x_om = x_om, x_rho = x_rho, n_bp = n_bp)
}

# Per-GP metadata: name, channels, subject/time columns, prior bundle, and the
# first-subject time grid used for the resolution-aware lengthscale prior.
.sbc_gp_info <- function(cm) {
  gps <- cm$model$latent_gps
  if (is.null(gps) || length(gps) == 0) return(list())
  lapply(gps, function(gp) {
    gd <- gp$data
    sv <- gd[[gp$subject]]
    grid <- sort(unique(as.numeric(gd[[gp$time_var]][sv == sv[1]])))
    list(name = gp$name, subject = gp$subject, time_var = gp$time_var,
         obs_var = gp$obs_var, priors = gp$priors %||% gp_priors(),
         time_grid = grid)
  })
}

# ---- 1. draw a full parameter set from the prior ---------------------------

# Returns list(theta = named numeric of ALL draw-column values, meta = ...).
# `theta` names match the posterior draw columns exactly, so truth and draws
# share one namespace and a functional is a single closure over it.
.sbc_draw_prior <- function(cm, outcome_priors, spike, gp_info) {
  data <- cm$model$outcome$data
  d <- .sbc_design(cm, data)
  gp_names <- vapply(gp_info, function(g) g$name, character(1))

  # Random change-points (a random effect in the omega formula) are not yet
  # supported: the shared mean engine (.build_predictions) builds the omega
  # design with plain model.matrix, which cannot represent per-subject
  # deviations. Fail clearly rather than silently miscalibrate.
  if (any(vapply(d$x_om, function(X) any(.get_re_masks(list(X))[[1]] == 1L), logical(1))))
    stop("SBC does not yet support random change-points (a random effect in the ",
         "`omega` formula). Calibrate the fixed-effect specification, or drop the ",
         "(1 | group) term from omega for the SBC run.", call. = FALSE)

  opv <- .build_outcome_prior_vectors(
    outcome_priors, spike, d$x_b0, d$x_b1, d$x_deltas, d$x_om, d$x_rho,
    gp_names = gp_names)
  pv <- opv$pv

  theta <- numeric(0)

  # b0 (fixed), then random intercepts u_<level> ~ N(0, sigma_u).
  b0 <- .sbc_draw_block(pv$b0)
  theta <- c(theta, stats::setNames(b0, paste0("b0_", names(b0))))

  re_group <- .parse_re(cm$b0_formula)$re_group
  has_re <- !is.null(re_group)
  re_levels <- character(0)
  sigma_u <- NA_real_
  if (has_re) {
    sigma_u <- .sbc_draw_gp_hyper(outcome_priors$sigma_u)   # half-Cauchy scale
    re_levels <- levels(as.factor(data[[re_group]]))
    u <- stats::rnorm(length(re_levels), 0, sigma_u)
    theta <- c(theta, stats::setNames(u, paste0("u_", re_levels)))
  }

  # b1 (initial slope).
  b1 <- .sbc_draw_block(pv$b1)
  if (length(b1)) theta <- c(theta, stats::setNames(b1, paste0("b1_", names(b1))))

  # deltas / omega / rho, per breakpoint (fixed effects; random change-points
  # are gated out above).
  n_bp <- d$n_bp
  for (k in seq_len(n_bp)) {
    dk <- .sbc_draw_block(pv$deltas[[k]])
    theta <- c(theta, stats::setNames(dk, paste0("delta", k, "_", names(dk))))
    ok <- .sbc_draw_block(pv$om[[k]])
    theta <- c(theta, stats::setNames(ok, paste0("omega", k, "_", names(ok))))
    rk <- .sbc_draw_block(pv$rho[[k]])
    theta <- c(theta, stats::setNames(rk, paste0("rho", k, "_", names(rk))))
  }

  # sigma (residual SD): variance ~ inverse-gamma(shape, scale).
  sig <- outcome_priors$sigma
  theta["sigma"] <- 1 / sqrt(stats::rgamma(1, shape = sig$shape, rate = sig$scale))
  theta["sigma_u"] <- if (has_re) sigma_u else 0

  # spike-and-slab inclusion indicators (Bernoulli(pi)) and, if learnt, pi.
  use_spike <- !is.null(spike) && inherits(spike, "smoothbp_spike_slab")
  if (use_spike) {
    if (!is.null(opv$b1_spike_mask) && length(b1)) {
      g <- ifelse(opv$b1_spike_mask == 1L, stats::rbinom(length(b1), 1, spike$pi), 1L)
      theta <- c(theta, stats::setNames(as.numeric(g), paste0("gamma_b1_", names(b1))))
    }
    for (k in seq_len(n_bp)) {
      nm <- pv$deltas[[k]]$name
      g <- stats::rbinom(length(nm), 1, spike$pi)
      theta <- c(theta, stats::setNames(as.numeric(g), paste0("gamma_delta", k, "_", nm)))
    }
    if (spike$learn_pi) theta["pi_ss"] <- spike$pi
  }

  # r (NB overdispersion): Gamma(shape, scale).
  family <- cm$model$outcome$family$family %||% "gaussian"
  if (family == "negative_binomial") {
    theta["r"] <- stats::rgamma(1, shape = outcome_priors$r$shape, scale = outcome_priors$r$scale)
  }

  # GP hyperparameters: alpha, rho (lengthscale), sigma_x.
  for (g in gp_info) {
    theta[paste0(g$name, "_alpha")]   <- .sbc_draw_gp_hyper(g$priors$alpha)
    theta[paste0(g$name, "_rho")]     <- .sbc_draw_gp_hyper(g$priors$rho, g$time_grid)
    theta[paste0(g$name, "_sigma_x")] <- .sbc_draw_gp_hyper(g$priors$sigma_x)
  }

  meta <- list(
    n_bp = n_bp, family = family, has_re = has_re, re_levels = re_levels,
    b1_names = names(b1),
    b1_spike_mask = opv$b1_spike_mask,
    delta_names = if (n_bp > 0) lapply(pv$deltas, function(x) x$name) else list(),
    om_names = if (n_bp > 0) lapply(pv$om, function(x) x$name) else list(),
    rho_names = if (n_bp > 0) lapply(pv$rho, function(x) x$name) else list(),
    om_masks = if (n_bp > 0) lapply(d$x_om, function(X) .get_re_masks(list(X))[[1]]) else list(),
    has_om_re = FALSE, use_spike = use_spike,
    gp_names = gp_names, gp_channels = vapply(gp_info, function(g) {
      if (g$name %in% colnames(d$x_b0)) "b0" else if (g$name %in% colnames(d$x_b1)) "b1" else "none"
    }, character(1))
  )
  list(theta = theta, meta = meta)
}

# ---- 2. simulate a dataset from the drawn parameters -----------------------

# Reuses .build_predictions(summary = FALSE) for the mean, then samples y from
# the outcome family. Returns the model's data frame with y (and any GP-observed
# covariate) replaced. `latent GP field f` is generated here and substituted
# into the GP design column so the existing mean code produces loading * f.
.sbc_simulate <- function(cm, draw, gp_info) {
  theta <- draw$theta
  data <- cm$model$outcome$data
  y_name <- all.vars(cm$model$outcome$formula)[1]
  tau_name <- all.vars(cm$model$outcome$formula)[2]
  n <- nrow(data)
  jitter <- 1e-6

  mu_data <- data                     # gp column -> latent f for the mean
  sim_data <- data                    # gp column -> observed f + noise for the fit
  gp_replacements <- list()
  for (g in gp_info) {
    alpha <- theta[[paste0(g$name, "_alpha")]]
    rho   <- theta[[paste0(g$name, "_rho")]]
    sig_x <- theta[[paste0(g$name, "_sigma_x")]]
    subj  <- as.factor(data[[g$subject]])
    tau   <- as.numeric(data[[tau_name]])
    f_full <- numeric(n); xobs_full <- numeric(n)
    for (lv in levels(subj)) {
      idx <- which(subj == lv)
      tg <- tau[idx]
      D <- as.matrix(stats::dist(tg))
      K <- alpha^2 * exp(-0.5 * (D / rho)^2) + diag(jitter, length(tg))
      L <- t(chol(K))
      f <- as.numeric(L %*% stats::rnorm(length(tg)))
      f_full[idx] <- f
      xobs_full[idx] <- f + stats::rnorm(length(tg), 0, sig_x)
    }
    mu_data[[g$name]]  <- f_full
    sim_data[[g$name]] <- xobs_full
    sim_data[[g$obs_var]] <- xobs_full   # GP observation-model column (may differ)
    gp_replacements[[g$name]] <- xobs_full
  }

  # One-row "draws" object so .build_predictions computes the mean from theta.
  dm1 <- matrix(theta, nrow = 1, dimnames = list(NULL, names(theta)))
  shim <- list(
    draws = posterior::as_draws_matrix(dm1),
    data = data,
    outcome_formula = cm$model$outcome$formula,
    b0_formula = cm$b0_formula, b1_formula = cm$b1_formula,
    deltas = cm$deltas, omega = cm$omega, rho = cm$rho,
    subject_var = cm$subject_var, n_subjects = cm$n_subjects,
    model = list(outcome = list(family = cm$model$outcome$family))
  )
  mu <- as.numeric(.build_predictions(shim, newdata = mu_data, type = "link", summary = FALSE)[1, ])

  family <- cm$model$outcome$family$family %||% "gaussian"
  y <- switch(family,
    gaussian          = stats::rnorm(n, mu, theta[["sigma"]]),
    binomial          = stats::rbinom(n, 1, stats::plogis(mu)),
    negative_binomial = stats::rnbinom(n, size = theta[["r"]], mu = exp(mu)),
    stop(sprintf("SBC does not support outcome family '%s'.", family)))

  sim_data[[y_name]] <- y
  attr(sim_data, "sbc_gp_obs") <- gp_replacements
  sim_data
}

# ---- 3. score a functional against the truth --------------------------------

# ESS-thinned fractional rank of the truth within the (transformed) posterior
# draws, plus the split-Rhat of the transformed quantity for the convergence
# gate. `fun` maps a named draw vector -> scalar; `M` is an iter x chain matrix.
.sbc_score_functional <- function(mat, chain, iter, fun, truth) {
  fv <- apply(mat, 1L, fun)
  nch <- max(chain); nit <- max(iter)
  M <- matrix(NA_real_, nit, nch)
  M[cbind(iter, chain)] <- fv
  da <- posterior::as_draws_array(array(M, dim = c(nit, nch, 1L)))
  eb <- tryCatch(posterior::ess_bulk(da), error = function(e) NA_real_)
  et <- tryCatch(posterior::ess_tail(da), error = function(e) NA_real_)
  rh <- tryCatch(posterior::rhat(da),     error = function(e) NA_real_)
  ess <- suppressWarnings(min(eb, et, na.rm = TRUE))
  v <- as.vector(M)
  if (!is.finite(ess) || ess < 2) ess <- length(v)
  vt <- v[seq(1L, length(v), by = max(1L, floor(length(v) / ess)))]
  list(rank = mean(vt < truth), rhat = rh)
}

# ---- default target functionals --------------------------------------------

# Build the interpretable default functionals from the drawn parameter
# structure: initial slope b1, each slope-difference delta, each breakpoint
# omega and sharpness rho, GP amplitude/lengthscale + loading, and the scales
# sigma / sigma_u / sigma_re_om / r. Spike-eligible coefficients are excluded
# (their marginal has an atom at 0; calibrate those via pip() instead). Each
# functional is list(name, truth_fun, draw_fun); default extractors are a
# single-column lookup, with the random-intercept truth = b0 + mean(u).
.sbc_default_functionals <- function(theta, meta) {
  fns <- list()
  add <- function(name, col = name, truth_fun = NULL, draw_fun = NULL) {
    if (is.null(draw_fun)) draw_fun <- function(v) v[[col]]
    if (is.null(truth_fun)) truth_fun <- draw_fun
    fns[[length(fns) + 1L]] <<- list(name = name, truth_fun = truth_fun, draw_fun = draw_fun)
  }
  nm <- names(theta)

  # b1: include columns that are NOT spike-eligible (GP-loading b1 columns, or
  # any b1 when no spike is active).
  for (j in seq_along(meta$b1_names)) {
    col <- paste0("b1_", meta$b1_names[j])
    eligible <- meta$use_spike && !is.null(meta$b1_spike_mask) && meta$b1_spike_mask[j] == 1L
    if (!eligible && col %in% nm) add(col)
  }
  # deltas (slope differences): exclude when spike-and-slab is active.
  if (!meta$use_spike) {
    for (k in seq_len(meta$n_bp)) for (cn in meta$delta_names[[k]]) {
      col <- paste0("delta", k, "_", cn); if (col %in% nm) add(col)
    }
  }
  # omega (breakpoints) and rho (sharpness): fixed-effect columns only.
  for (k in seq_len(meta$n_bp)) {
    mask <- meta$om_masks[[k]]
    for (j in seq_along(meta$om_names[[k]])) {
      if (mask[j] == 0L) { col <- paste0("omega", k, "_", meta$om_names[[k]][j]); if (col %in% nm) add(col) }
    }
    for (cn in meta$rho_names[[k]]) { col <- paste0("rho", k, "_", cn); if (col %in% nm) add(col) }
    if (meta$has_om_re && !meta$use_spike && paste0("sigma_re_om", k) %in% nm)
      add(paste0("sigma_re_om", k))
  }
  # GP amplitude, lengthscale, obs-noise, and the loading coefficient.
  for (g in seq_along(meta$gp_names)) {
    gname <- meta$gp_names[g]
    for (suff in c("_alpha", "_rho", "_sigma_x")) {
      col <- paste0(gname, suff); if (col %in% nm) add(col)
    }
    load_col <- paste0(meta$gp_channels[g], "_", gname)   # e.g. b0_X_obs
    if (load_col %in% nm) add(load_col)
  }
  # scales. The residual SD 'sigma' only enters the Gaussian likelihood; for
  # binomial/NB it is an unconstrained nuisance, so it is not a default target.
  if (meta$family == "gaussian" && "sigma" %in% nm) add("sigma")
  if (meta$has_re && "sigma_u" %in% nm) add("sigma_u")
  if (meta$family == "negative_binomial" && "r" %in% nm) add("r")

  fns
}

# Wrap user-supplied functionals (closures over a named parameter vector) so
# they share the truth_fun == draw_fun contract.
.sbc_wrap_user_functionals <- function(functionals) {
  if (is.null(functionals)) return(list())
  if (!is.list(functionals) || is.null(names(functionals)))
    stop("`functionals` must be a named list of functions of a parameter vector.")
  lapply(seq_along(functionals), function(i) {
    f <- functionals[[i]]
    if (!is.function(f)) stop("Each entry of `functionals` must be a function.")
    list(name = names(functionals)[i], truth_fun = f, draw_fun = f)
  })
}

# ---- public interface -------------------------------------------------------

#' Simulation-based calibration for a bjlm model
#'
#' Runs simulation-based calibration (SBC; Talts et al. 2018) to check whether
#' the sampler and priors recover truth on data resembling yours. For each of
#' `reps` replicates a full parameter set is drawn from the prior, a dataset is
#' simulated from it, the model is re-fit, and the rank of each true value
#' within its posterior is recorded. If the model is calibrated those ranks are
#' uniform; systematic departures reveal over- or under-confident intervals or
#' bias. The returned object prints a plain-language verdict per quantity.
#'
#' You can call `sbc()` on a compiled model (to check calibration \emph{before}
#' committing to a fit) or on a fitted model (which reuses that fit's model,
#' priors, and design).
#'
#' \strong{What is checked.} By default the interpretable, identifiability-
#' revealing quantities: the initial slope `b1`, each slope-difference `delta`
#' (the "did it accelerate" numbers), each breakpoint `omega` and sharpness
#' `rho`, the GP amplitude and lengthscale (and its loading), and the scale
#' parameters (`sigma`, `sigma_u`, `r`). Pass extra `functionals` as closures of
#' the named parameter vector. Spike-and-slab-selected coefficients are excluded
#' from rank calibration (their marginal has an atom at zero); calibrate those
#' with [pip()] instead.
#'
#' \strong{Robustness guards.} Posterior draws are thinned toward independence
#' (SBC assumes independent draws) using the bulk/tail effective sample size,
#' and replicates whose fit did not converge (max split-Rhat above
#' `rhat_threshold`, or non-finite draws) are discarded and reported, so a
#' sampler failure on a hard simulated dataset is never mistaken for a
#' calibration failure.
#'
#' @param object A `bjlm_compiled_model` (from [compile()]) or a `bjlm_fit`.
#' @param priors A [bjlm_priors()] bundle. Defaults to the same priors the fit
#'   would use. The prior draws and the re-fit always share this bundle.
#' @param spike Optional [prior_spike_slab()] for variable selection; must match
#'   what you intend to fit with.
#' @param reps Number of simulate-and-fit replicates. Default `50`; `>= 100`
#'   gives more power to detect miscalibration.
#' @param functionals Optional named list of extra target functionals, each a
#'   function of the named parameter vector returning a scalar.
#' @param iter,warmup,chains,cores,seed Passed to [fit()] for each replicate.
#'   `warmup` defaults to `iter / 2`.
#' @param rhat_threshold Convergence gate; replicates whose worst target-quantity
#'   split-Rhat exceeds this are discarded. Default `1.01`.
#' @param verbose Print progress. Default `TRUE`.
#' @param ... Further arguments passed to [fit()].
#'
#' @return A `bjlm_sbc` object with the rank matrix, per-quantity uniformity
#'   p-values and verdicts, and the discard rate. See [print.bjlm_sbc] and
#'   [plot.bjlm_sbc].
#' @references Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A.
#'   (2018). Validating Bayesian inference algorithms with simulation-based
#'   calibration. arXiv:1804.06788.
#' @seealso [sbc_prior_sensitivity()], [pip()], [recovery_plot()].
#' @export
sbc <- function(object, ...) {
  UseMethod("sbc")
}

#' @rdname sbc
#' @export
sbc.bjlm_fit <- function(object, reps = 50, functionals = NULL,
                         iter = NULL, warmup = NULL, chains = NULL,
                         cores = 1L, seed = 1L, rhat_threshold = 1.01,
                         verbose = TRUE, ...) {
  cm <- object$compiled_model
  if (is.null(cm))
    stop("This fit does not retain its compiled model; re-fit with a current ",
         "version of bjlm, or call sbc() on the compiled model directly.", call. = FALSE)
  dots <- object$fit_dots %||% list()
  sbc.bjlm_compiled_model(
    cm,
    priors = object$priors_used,
    spike = object$spike,
    reps = reps, functionals = functionals,
    iter = iter %||% object$iter %||% 1000L,
    warmup = warmup %||% object$warmup,
    chains = chains %||% object$chains %||% 2L,
    cores = cores, seed = seed, rhat_threshold = rhat_threshold,
    verbose = verbose, ...)
}

#' @rdname sbc
#' @export
sbc.bjlm_compiled_model <- function(object, priors = NULL, spike = NULL,
                                    reps = 50L, functionals = NULL,
                                    iter = 1000L, warmup = NULL, chains = 2L,
                                    cores = 1L, seed = 1L, rhat_threshold = 1.01,
                                    verbose = TRUE, ...) {
  if (object$zero_breakpoint && is.null(priors)) priors <- object$shortcut_priors
  if (is.null(priors)) priors <- bjlm_priors()
  if (!inherits(priors, "bjlm_priors"))
    stop("`priors` must be a bjlm_priors() bundle.", call. = FALSE)
  outcome_priors <- priors$outcome %||% smoothbp_priors()
  if (is.null(warmup)) warmup <- iter %/% 2L

  gp_info <- .sbc_gp_info(object)
  n_bp <- length(object$deltas)

  # ---- one-time guidance / caveats ----
  if (!isTRUE(object$model$auto_propensity))
    message("sbc(): the outcome fit is IPW-weighted (a marginal structural model), so ",
            "SBC checks the pseudo-posterior; for a pure calibration check consider the ",
            "unweighted outcome specification.")
  if (n_bp >= 2)
    message("sbc(): with >= 2 breakpoints, per-index omega/delta ranks assume the ",
            "change-points are order-identified; if they can swap, use ",
            "space_omega_priors() or interpret only order-invariant functionals.")
  .sbc_identifiability_note(object, outcome_priors)
  if (reps < 100)
    message(sprintf("sbc(): running %d replicates; >= 100 gives more power to detect miscalibration.", reps))

  functionals_built <- NULL
  ranks <- NULL
  rank_names <- NULL
  n_discard <- 0L          # rep-level: fit failed / non-finite / degenerate sim
  discard_reasons <- character(0)
  n_fit_ok <- 0L
  nonconv <- NULL          # per-functional count of Rhat-filtered draws

  for (rep in seq_len(reps)) {
    set.seed(seed + rep)
    draw <- .sbc_draw_prior(object, outcome_priors, spike, gp_info)
    sim <- tryCatch(.sbc_simulate(object, draw, gp_info),
                    error = function(e) { discard_reasons[[length(discard_reasons) + 1L]] <<- paste("simulate:", conditionMessage(e)); NULL })
    if (is.null(sim)) { n_discard <- n_discard + 1L; next }

    # degenerate-simulation guard (heavy-tailed count draws can blow up).
    yy <- sim[[all.vars(object$model$outcome$formula)[1]]]
    if (any(!is.finite(yy)) || (draw$meta$family == "negative_binomial" && max(yy) > 1e6)) {
      n_discard <- n_discard + 1L
      discard_reasons[length(discard_reasons) + 1L] <- "degenerate simulated data"; next
    }

    cm_rep <- object
    cm_rep$model$outcome$data <- sim
    if (length(gp_info) > 0)
      for (i in seq_along(cm_rep$model$latent_gps)) cm_rep$model$latent_gps[[i]]$data <- sim

    fit <- tryCatch(
      fit.bjlm_compiled_model(cm_rep, priors = priors, spike = spike,
          chains = chains, iter = iter, warmup = warmup, cores = cores,
          seed = seed + rep, verbose = FALSE, ...),
      error = function(e) { discard_reasons[[length(discard_reasons) + 1L]] <<- paste("fit:", conditionMessage(e)); NULL })
    if (is.null(fit)) { n_discard <- n_discard + 1L; next }

    if (!all(is.finite(as.array(fit$draws)))) {
      n_discard <- n_discard + 1L
      discard_reasons[length(discard_reasons) + 1L] <- "non-finite draws"; next
    }

    if (is.null(functionals_built)) {
      functionals_built <- c(.sbc_default_functionals(draw$theta, draw$meta),
                             .sbc_wrap_user_functionals(functionals))
      rank_names <- vapply(functionals_built, function(f) f$name, character(1))
      nonconv <- stats::setNames(integer(length(rank_names)), rank_names)
    }

    ndf <- as.data.frame(posterior::as_draws_df(fit$draws))
    vn <- posterior::variables(fit$draws)
    mat <- as.matrix(ndf[, vn, drop = FALSE])
    ch <- as.integer(ndf$.chain); it <- as.integer(ndf$.iteration)

    # Per-functional convergence filtering: a quantity whose own transformed
    # Rhat exceeds the gate is recorded as NA (not counted), rather than
    # discarding the whole replicate -- so a non-identified change-point does
    # not throw away the perfectly good sigma/b1 ranks from the same fit.
    row_rank <- rep(NA_real_, length(functionals_built))
    for (fi in seq_along(functionals_built)) {
      f <- functionals_built[[fi]]
      truth <- tryCatch(f$truth_fun(draw$theta), error = function(e) NA_real_)
      sc <- tryCatch(.sbc_score_functional(mat, ch, it, f$draw_fun, truth),
                     error = function(e) list(rank = NA_real_, rhat = NA_real_))
      if (is.finite(sc$rhat) && sc$rhat > rhat_threshold) {
        nonconv[fi] <- nonconv[fi] + 1L
      } else {
        row_rank[fi] <- sc$rank
      }
    }

    ranks <- rbind(ranks, row_rank)
    n_fit_ok <- n_fit_ok + 1L
    if (verbose && (rep %% 10L == 0L || rep == reps))
      message(sprintf("  sbc: %d/%d replicates (%d fit ok, %d discarded)", rep, reps, n_fit_ok, n_discard))
  }

  if (is.null(ranks) || nrow(ranks) == 0L)
    stop("SBC produced no usable replicates (every fit failed or was degenerate). ",
         "Check the model/priors.", call. = FALSE)
  colnames(ranks) <- rank_names

  .bjlm_sbc(
    ranks = ranks,
    n_reps = reps, n_kept = n_fit_ok, n_discard = n_discard,
    nonconv = nonconv, discard_reasons = discard_reasons,
    settings = list(iter = iter, warmup = warmup, chains = chains,
                    rhat_threshold = rhat_threshold, spike = !is.null(spike),
                    n_breakpoints = n_bp, family = object$model$outcome$family$family %||% "gaussian")
  )
}

# Warn when the omega prior places substantial mass outside the observed time
# range (change-points there are unidentified for reasons unrelated to the
# sampler, which would masquerade as miscalibration).
.sbc_identifiability_note <- function(cm, outcome_priors) {
  om <- outcome_priors$omega
  if (!inherits(om, "smoothbp_prior") || om$family != "normal") return(invisible())
  tau <- suppressWarnings(as.numeric(cm$model$outcome$data[[all.vars(cm$model$outcome$formula)[2]]]))
  if (length(tau) == 0 || all(is.na(tau))) return(invisible())
  lo <- min(tau, na.rm = TRUE); hi <- max(tau, na.rm = TRUE)
  if (om$sd <= 0) return(invisible())
  outside <- stats::pnorm(lo, om$mean, om$sd) + (1 - stats::pnorm(hi, om$mean, om$sd))
  if (outside > 0.25)
    message(sprintf(
      "sbc(): the omega prior places ~%.0f%% of its mass outside the observed time range [%.2f, %.2f]; ",
      100 * outside, lo, hi),
      "change-point calibration there is conditional on identifiability, not a pure sampler verdict.")
  invisible()
}

# Constructor + uniformity scoring for the result object.
.bjlm_sbc <- function(ranks, n_reps, n_kept, n_discard, nonconv = NULL,
                      discard_reasons, settings) {
  verdicts <- .sbc_verdicts(ranks)
  if (!is.null(nonconv)) verdicts$n_nonconv <- as.integer(nonconv[verdicts$parameter])
  structure(
    list(ranks = ranks, n_reps = n_reps, n_kept = n_kept, n_discard = n_discard,
         nonconv = nonconv, discard_reasons = discard_reasons, settings = settings,
         verdicts = verdicts),
    class = "bjlm_sbc"
  )
}
