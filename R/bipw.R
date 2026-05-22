#' Fit a joint Bayesian IPW model
#'
#' Fits a joint propensity-weighted piecewise regression model using
#' Pólya-Gamma augmented Gibbs sampling for the propensity model and
#' a weighted smoothbp outcome model. Implements the "cut posterior"
#' (modular Bayes) approach to prevent outcome-to-propensity feedback.
#'
#' @param outcome A two-sided formula of the form \code{y ~ tau}, where
#'   \code{y} is the outcome variable and \code{tau} is the time variable.
#' @param b0 A one-sided formula for the intercept model (e.g., \code{~ 1 + Group}).
#'   Supports random intercepts via \code{(1 | subject)}.
#' @param b1 A one-sided formula for the initial slope model.
#' @param deltas A list of one-sided formulas for slope-change parameters at each breakpoint.
#' @param omega A list of one-sided formulas for breakpoint location parameters.
#' @param rho A list of one-sided formulas for transition sharpness parameters.
#' @param propensity A two-sided formula of the form \code{treatment ~ covariates},
#'   specifying the propensity score model. The LHS must be a binary (0/1) variable.
#' @param weights Character string specifying the weight type:
#'   \code{"stabilised_ate"} (default), \code{"ate"}, \code{"att"}, or \code{"stabilised_att"}.
#' @param max_weight Numeric. Trimming threshold for extreme weights (default: 20).
#' @param data A data frame containing all variables.
#' @param outcome_priors A list of priors for the outcome model parameters.
#'   See \code{\link{smoothbp_priors}} for details.
#' @param propensity_prior_sd Numeric. Standard deviation for the isotropic normal
#'   prior on propensity model coefficients (default: 2.5, weakly informative for logistic).
#' @param chains Integer. Number of MCMC chains (default: 4).
#' @param iter Integer. Total number of iterations per chain (default: 5000).
#' @param warmup Integer. Number of warmup iterations (default: half of \code{iter}).
#' @param seed Integer. Random seed for reproducibility.
#' @param verbose Logical. Print progress messages (default: TRUE).
#' @param cores Integer. Number of parallel cores (default: 1).
#' @param step_om,step_rho Initial HMC step sizes for omega and rho parameters.
#' @param target_accept Target acceptance rate for HMC (default: 0.8).
#'
#' @return An object of class \code{"bipw_fit"} containing:
#'   \describe{
#'     \item{outcome_draws}{Posterior draws for outcome model parameters.}
#'     \item{propensity_draws}{Posterior draws for propensity model coefficients.}
#'     \item{weight_draws}{Posterior draws for mean weights (diagnostic).}
#'     \item{outcome_names}{Parameter names for the outcome model.}
#'     \item{propensity_names}{Parameter names for the propensity model.}
#'     \item{call}{The matched call.}
#'   }
#'
#' @export
bipw <- function(
    outcome,
    b0, b1,
    deltas = list(),
    omega = list(),
    rho = list(),
    propensity,
    weights = c("stabilised_ate", "ate", "att", "stabilised_att"),
    max_weight = 20,
    data,
    outcome_priors = NULL,
    propensity_prior_sd = 2.5,
    chains = 4L,
    iter = 5000L,
    warmup = NULL,
    seed = NULL,
    verbose = TRUE,
    cores = 1L,
    step_om = 0.01,
    step_rho = 0.01,
    target_accept = 0.8
) {
  cl <- match.call()
  weights <- match.arg(weights)
  weight_type_int <- match(weights, c("ate", "att", "stabilised_ate", "stabilised_att")) - 1L

  if (is.null(warmup)) warmup <- floor(iter / 2)
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  # ---- Parse outcome formula ----
  outcome_vars <- all.vars(outcome)
  y_name <- outcome_vars[1]
  tau_name <- outcome_vars[2]
  y <- data[[y_name]]
  tau <- data[[tau_name]]
  n <- length(y)

  # ---- Parse propensity formula ----
  prop_vars <- all.vars(propensity)
  treatment_name <- prop_vars[1]
  prop_covariate_names <- prop_vars[-1]

  # Propensity model is subject-level, so we need to identify unique subjects
  # Use the random effect grouping variable from b0 if available
  re_info <- .parse_re(b0)
  if (!is.null(re_info$re_var)) {
    group_var <- re_info$re_var
    group_factor <- as.factor(data[[group_var]])
    group_indices <- as.integer(group_factor) - 1L
    n_groups <- nlevels(group_factor)

    # Extract subject-level data (first occurrence of each subject)
    first_idx <- !duplicated(data[[group_var]])
    subject_data <- data[first_idx, , drop = FALSE]
    n_subjects <- nrow(subject_data)
  } else {
    # Cross-sectional: each observation is a subject
    group_indices <- rep(-1L, n)
    n_groups <- 0L
    subject_data <- data
    n_subjects <- n
  }

  treatment <- subject_data[[treatment_name]]
  stopifnot("Treatment variable must be binary (0/1)" = all(treatment %in% c(0, 1)))

  # Build propensity design matrix (subject-level)
  prop_formula <- reformulate(prop_covariate_names)
  x_prop <- model.matrix(prop_formula, data = subject_data)
  p_prop <- ncol(x_prop)
  prop_names <- paste0("alpha_", colnames(x_prop))

  # ---- Build outcome design matrices (same as smoothbp) ----
  b0_fixed_formula <- .strip_re(b0)
  x_b0 <- model.matrix(b0_fixed_formula, data = data)

  x_b1 <- model.matrix(b1, data = data)

  n_bp <- length(deltas)

  x_deltas_list <- lapply(deltas, function(f) model.matrix(f, data = data))
  x_om_list <- lapply(omega, function(f) model.matrix(f, data = data))
  x_rho_list <- lapply(rho, function(f) model.matrix(f, data = data))

  p_b0 <- ncol(x_b0)
  p_b1 <- ncol(x_b1)
  p_deltas <- if (n_bp > 0) sapply(x_deltas_list, ncol) else -1L
  p_om <- if (n_bp > 0) sapply(x_om_list, ncol) else -1L
  p_rho <- if (n_bp > 0) sapply(x_rho_list, ncol) else -1L

  # ---- Build outcome priors (reuse smoothbp infrastructure) ----
  if (is.null(outcome_priors)) {
    outcome_priors <- .default_bipw_priors(p_b0, p_b1, n_bp, p_deltas, p_om, p_rho)
  }

  # ---- Parameter names ----
  b0_names <- paste0("b0_", colnames(x_b0))
  re_names <- if (n_groups > 0) paste0("u_", levels(group_factor)) else character(0)
  b1_names <- paste0("b1_", colnames(x_b1))
  delta_names <- if (n_bp > 0) {
    unlist(lapply(seq_len(n_bp), function(k) paste0("delta", k, "_", colnames(x_deltas_list[[k]]))))
  } else character(0)
  om_names <- if (n_bp > 0) {
    unlist(lapply(seq_len(n_bp), function(k) paste0("omega", k, "_", colnames(x_om_list[[k]]))))
  } else character(0)
  rho_names <- if (n_bp > 0) {
    unlist(lapply(seq_len(n_bp), function(k) paste0("rho", k, "_", colnames(x_rho_list[[k]]))))
  } else character(0)
  outcome_names <- c(b0_names, re_names, b1_names, delta_names, om_names, rho_names, "sigma", "sigma_u")

  # ---- Call Rust sampler ----
  raw <- run_bipw(
    y = y,
    tau = tau,
    x_b0 = as.double(x_b0), p_b0 = as.integer(p_b0),
    x_b1 = as.double(x_b1), p_b1 = as.integer(p_b1),
    x_deltas = if (n_bp > 0) lapply(x_deltas_list, as.double) else list(-1),
    p_deltas = as.integer(p_deltas),
    x_om = if (n_bp > 0) lapply(x_om_list, as.double) else list(-1),
    p_om = as.integer(p_om),
    x_rho = if (n_bp > 0) lapply(x_rho_list, as.double) else list(-1),
    p_rho = as.integer(p_rho),
    group_b0 = if (n_groups > 0) group_indices else -1L,
    n_groups_b0 = as.integer(n_groups),
    prior_mean_b0 = outcome_priors$b0_mean,
    prior_sd_b0 = outcome_priors$b0_sd,
    prior_lb_b0 = outcome_priors$b0_lb,
    prior_ub_b0 = outcome_priors$b0_ub,
    prior_mean_b1 = outcome_priors$b1_mean,
    prior_sd_b1 = outcome_priors$b1_sd,
    prior_lb_b1 = outcome_priors$b1_lb,
    prior_ub_b1 = outcome_priors$b1_ub,
    prior_mean_deltas = if (n_bp > 0) outcome_priors$delta_mean else list(-1),
    prior_sd_deltas = if (n_bp > 0) outcome_priors$delta_sd else list(-1),
    prior_lb_deltas = if (n_bp > 0) outcome_priors$delta_lb else list(-1),
    prior_ub_deltas = if (n_bp > 0) outcome_priors$delta_ub else list(-1),
    prior_mean_om = if (n_bp > 0) outcome_priors$om_mean else list(-1),
    prior_sd_om = if (n_bp > 0) outcome_priors$om_sd else list(-1),
    prior_lb_om = if (n_bp > 0) outcome_priors$om_lb else list(-1),
    prior_ub_om = if (n_bp > 0) outcome_priors$om_ub else list(-1),
    prior_mean_rho = if (n_bp > 0) outcome_priors$rho_mean else list(-1),
    prior_sd_rho = if (n_bp > 0) outcome_priors$rho_sd else list(-1),
    prior_lb_rho = if (n_bp > 0) outcome_priors$rho_lb else list(-1),
    prior_ub_rho = if (n_bp > 0) outcome_priors$rho_ub else list(-1),
    sigma_shape = outcome_priors$sigma_shape,
    sigma_scale = outcome_priors$sigma_scale,
    sigma_u_shape = outcome_priors$sigma_u_shape,
    sigma_u_scale = outcome_priors$sigma_u_scale,
    x_prop = as.double(x_prop), p_prop = as.integer(p_prop),
    treatment = as.double(treatment),
    n_subjects = as.integer(n_subjects),
    prop_prior_sd = propensity_prior_sd,
    weight_type = as.integer(weight_type_int),
    max_weight = max_weight,
    step_om = step_om,
    step_rho = step_rho,
    target_accept = target_accept,
    chains = as.integer(chains),
    iter = as.integer(iter),
    warmup = as.integer(warmup),
    seed = as.integer(seed),
    verbose = verbose,
    n_cores = as.integer(cores)
  )

  # ---- Post-process draws ----
  n_outcome <- length(outcome_names)
  n_alpha <- p_prop
  n_total <- n_outcome + n_alpha + 1  # +1 for mean_weight

  all_names <- c(outcome_names, prop_names, "mean_weight")

  draws_list <- lapply(seq_len(chains), function(c) {
    mat <- raw$draws[[c]]
    colnames(mat) <- all_names
    mat
  })

  # Build posterior draws object
  if (requireNamespace("posterior", quietly = TRUE)) {
    n_post <- nrow(draws_list[[1]])
    # Build the 3D array correctly: dims are (iteration, chain, variable)
    # We must fill it explicitly because unlist + array has wrong fill order
    arr <- array(NA_real_, dim = c(n_post, chains, n_total),
                 dimnames = list(NULL, paste0("chain_", seq_len(chains)), all_names))
    for (c in seq_len(chains)) {
      arr[, c, ] <- draws_list[[c]]
    }
    draws_array <- posterior::as_draws_array(arr)
  } else {
    draws_array <- draws_list
  }

  structure(
    list(
      draws = draws_array,
      outcome_names = outcome_names,
      propensity_names = prop_names,
      weight_name = "mean_weight",
      call = cl,
      data = data,
      n = n,
      n_subjects = n_subjects,
      n_breakpoints = n_bp,
      weight_type = weights,
      max_weight = max_weight,
      treatment_name = treatment_name,
      chains = chains,
      iter = iter,
      warmup = warmup
    ),
    class = "bipw_fit"
  )
}

# ---- Internal helpers ----

#' Parse random effects from a formula like ~ 1 + x + (1 | group)
#' @noRd
.parse_re <- function(formula) {
  # Check for (1 | group) pattern
  terms_str <- deparse(formula)
  re_match <- regmatches(terms_str, regexpr("\\(1\\s*\\|\\s*(\\w+)\\)", terms_str, perl = TRUE))
  if (length(re_match) == 0 || re_match == "") {
    return(list(re_var = NULL))
  }
  re_var <- gsub(".*\\|\\s*(\\w+)\\).*", "\\1", re_match)
  list(re_var = re_var)
}

#' Strip random effects from a formula
#' @noRd
.strip_re <- function(formula) {
  terms_str <- deparse(formula)
  cleaned <- gsub("\\+?\\s*\\(1\\s*\\|\\s*\\w+\\)\\s*\\+?", "", terms_str)
  cleaned <- gsub("\\s+\\+\\s+$", "", cleaned)
  cleaned <- gsub("^\\s+\\+\\s+", "", cleaned)
  as.formula(cleaned)
}

#' Default priors for bipw (mirrors smoothbp defaults)
#' @noRd
.default_bipw_priors <- function(p_b0, p_b1, n_bp, p_deltas, p_om, p_rho) {
  priors <- list(
    b0_mean = rep(0, p_b0),
    b0_sd = rep(5, p_b0),
    b0_lb = rep(-Inf, p_b0),
    b0_ub = rep(Inf, p_b0),
    b1_mean = rep(0, p_b1),
    b1_sd = rep(2, p_b1),
    b1_lb = rep(-Inf, p_b1),
    b1_ub = rep(Inf, p_b1),
    sigma_shape = 1,
    sigma_scale = 1,
    sigma_u_shape = 1,
    sigma_u_scale = 1
  )

  if (n_bp > 0) {
    priors$delta_mean <- lapply(p_deltas, function(p) rep(0, p))
    priors$delta_sd <- lapply(p_deltas, function(p) rep(2, p))
    priors$delta_lb <- lapply(p_deltas, function(p) rep(-Inf, p))
    priors$delta_ub <- lapply(p_deltas, function(p) rep(Inf, p))
    priors$om_mean <- lapply(p_om, function(p) rep(0, p))
    priors$om_sd <- lapply(p_om, function(p) rep(5, p))
    priors$om_lb <- lapply(p_om, function(p) rep(-Inf, p))
    priors$om_ub <- lapply(p_om, function(p) rep(Inf, p))
    priors$rho_mean <- lapply(p_rho, function(p) rep(1, p))
    priors$rho_sd <- lapply(p_rho, function(p) rep(2, p))
    priors$rho_lb <- lapply(p_rho, function(p) rep(0, p))
    priors$rho_ub <- lapply(p_rho, function(p) rep(Inf, p))
  }
  priors
}
