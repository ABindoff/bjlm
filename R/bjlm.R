#' Fit a joint Bayesian Joint Longitudinal Model (BJLM)
#'
#' Fits a joint propensity-weighted piecewise regression model using
#' Pólya-Gamma augmented Gibbs sampling for the propensity model and
#' a weighted outcome model. Implements the "cut posterior"
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
#' @param priors A \code{\link{bjlm_priors}} object specifying priors for both
#'   outcome and propensity models (optional). If supplied, takes precedence
#'   over \code{outcome_priors} and \code{propensity_prior_sd}.
#' @param outcome_priors A list of priors for the outcome model parameters (legacy).
#'   See \code{\link{smoothbp_priors}} for details.
#' @param propensity_prior_sd Numeric. Standard deviation for the isotropic normal
#'   prior on propensity model coefficients (legacy, default: 2.5).
#' @param chains Integer. Number of MCMC chains (default: 4).
#' @param iter Integer. Total number of iterations per chain (default: 5000).
#' @param warmup Integer. Number of warmup iterations (default: half of \code{iter}).
#' @param seed Integer. Random seed for reproducibility.
#' @param verbose Logical. Print progress messages (default: TRUE).
#' @param cores Integer. Number of parallel cores (default: 1).
#' @param step_om,step_rho Initial HMC step sizes for omega and rho parameters.
#' @param target_accept Target acceptance rate for HMC (default: 0.8).
#'
#' @return An object of class \code{"bjlm_fit"} containing posterior draws and metadata.
#'
#' @export
bjlm <- function(
    outcome,
    b0, b1,
    deltas = list(),
    omega = list(),
    rho = list(),
    latent_gps = list(),
    propensity,
    weights = c("stabilised_ate", "ate", "att", "stabilised_att"),
    max_weight = 20,
    data,
    priors = NULL,
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
  y <- as.double(data[[y_name]])
  tau <- as.double(data[[tau_name]])
  n <- length(y)

  # ---- Parse propensity formula ----
  prop_vars <- all.vars(propensity)
  treatment_name <- prop_vars[1]
  prop_covariate_names <- prop_vars[-1]

  # Propensity model is subject-level, so we need to identify unique subjects
  re_info <- .parse_re(b0)
  if (!is.null(re_info$re_group)) {
    group_var <- re_info$re_group
    group_factor <- as.factor(data[[group_var]])
    group_indices <- as.integer(group_factor) - 1L
    n_groups <- nlevels(group_factor)

    # Extract subject-level data (first occurrence of each subject) and align with levels(group_factor)
    first_idx <- !duplicated(data[[group_var]])
    subject_data <- data[first_idx, , drop = FALSE]
    subject_data <- subject_data[match(levels(group_factor), subject_data[[group_var]]), , drop = FALSE]
    n_subjects <- nrow(subject_data)
  } else {
    # Cross-sectional
    group_indices <- rep(-1L, n)
    n_groups <- 0L
    subject_data <- data
    n_subjects <- n
  }

  treatment <- subject_data[[treatment_name]]
  stopifnot("Treatment variable must be binary (0/1)" = all(treatment %in% c(0, 1)))

  # Build propensity design matrix
  prop_formula <- reformulate(prop_covariate_names)
  x_prop <- model.matrix(prop_formula, data = subject_data)
  p_prop <- ncol(x_prop)
  prop_names <- paste0("alpha_", colnames(x_prop))

  # ---- Build outcome design matrices ----
  b0_fixed_formula <- re_info$fixed
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

  # ---- Resolve priors ----
  if (!is.null(priors)) {
    if (inherits(priors, "bjlm_priors")) {
      outcome_priors <- priors$outcome
      propensity_prior_sd <- priors$propensity$sd
    } else {
      stop("`priors` must be an object of class `bjlm_priors` created by bjlm_priors().")
    }
  }

  if (is.null(outcome_priors)) {
    outcome_priors <- smoothbp_priors()
  }

  # Expand outcome priors into vectors
  dm_meta <- list(
    col_names_b0 = colnames(x_b0),
    col_names_b1 = colnames(x_b1),
    col_names_deltas = if (n_bp > 0) lapply(x_deltas_list, colnames) else list(),
    col_names_om = if (n_bp > 0) lapply(x_om_list, colnames) else list(),
    col_names_rho = if (n_bp > 0) lapply(x_rho_list, colnames) else list(),
    X_om = if (n_bp > 0) x_om_list else list(),
    X_rho = if (n_bp > 0) x_rho_list else list()
  )
  pv <- .build_prior_vectors(outcome_priors, dm_meta)

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
  gp_hyper_names <- character(0)
  if (length(latent_gps) > 0) {
    gp_hyper_names <- unlist(lapply(latent_gps, function(gp) {
      paste0(gp$name, c("_alpha", "_rho", "_sigma_x"))
    }))
  }
  outcome_names <- c(b0_names, re_names, b1_names, delta_names, om_names, rho_names, "sigma", "sigma_u", gp_hyper_names)

  # ---- Process Latent GPs ----
  gp_list <- list()
  if (length(latent_gps) > 0) {
    gp_list <- lapply(latent_gps, function(gp) {
      
      p_b0_idx <- match(gp$name, colnames(x_b0)) - 1L
      if (is.na(p_b0_idx)) p_b0_idx <- -1L
      
      p_b1_idx <- match(gp$name, colnames(x_b1)) - 1L
      if (is.na(p_b1_idx)) p_b1_idx <- -1L
      
      p_prop_idx <- match(gp$name, colnames(x_prop)) - 1L
      if (is.na(p_prop_idx)) p_prop_idx <- -1L
      
      # Coerce subject variable in GP data to match the factor levels of the outcome subject variable
      gp_data <- gp$data
      gp_data[[gp$subject]] <- factor(gp_data[[gp$subject]], levels = levels(group_factor))
      # Drop missing subjects
      gp_data <- gp_data[!is.na(gp_data[[gp$subject]]), , drop = FALSE]
      
      list(
        name = gp$name,
        obs_time = as.double(gp_data[[gp$time_var]]),
        obs_val = as.double(gp_data[[gp$obs_var]]),
        obs_group = as.integer(gp_data[[gp$subject]]) - 1L,
        
        trt_time = as.double(subject_data[[gp$time_trt_var]]),
        trt_group = as.integer(factor(subject_data[[group_var]], levels = levels(group_factor))) - 1L,
        
        out_time = as.double(data[[gp$time_out_var]]),
        out_group = as.integer(factor(data[[group_var]], levels = levels(group_factor))) - 1L,
        
        p_b0_idx = as.integer(p_b0_idx),
        p_b1_idx = as.integer(p_b1_idx),
        p_prop_idx = as.integer(p_prop_idx)
      )
    })
  }

  # ---- Call Rust sampler ----
  raw <- run_bjlm(
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
    prior_mean_b0 = pv$b0$mean,
    prior_sd_b0 = pv$b0$sd,
    prior_lb_b0 = pv$b0$lb,
    prior_ub_b0 = pv$b0$ub,
    prior_mean_b1 = pv$b1$mean,
    prior_sd_b1 = pv$b1$sd,
    prior_lb_b1 = pv$b1$lb,
    prior_ub_b1 = pv$b1$ub,
    prior_mean_deltas = if (n_bp > 0) lapply(pv$deltas, `[[`, "mean") else list(-1),
    prior_sd_deltas = if (n_bp > 0) lapply(pv$deltas, `[[`, "sd") else list(-1),
    prior_lb_deltas = if (n_bp > 0) lapply(pv$deltas, `[[`, "lb") else list(-1),
    prior_ub_deltas = if (n_bp > 0) lapply(pv$deltas, `[[`, "ub") else list(-1),
    prior_mean_om = if (n_bp > 0) lapply(pv$om, `[[`, "mean") else list(-1),
    prior_sd_om = if (n_bp > 0) lapply(pv$om, `[[`, "sd") else list(-1),
    prior_lb_om = if (n_bp > 0) lapply(pv$om, `[[`, "lb") else list(-1),
    prior_ub_om = if (n_bp > 0) lapply(pv$om, `[[`, "ub") else list(-1),
    prior_mean_rho = if (n_bp > 0) lapply(pv$rho, `[[`, "mean") else list(-1),
    prior_sd_rho = if (n_bp > 0) lapply(pv$rho, `[[`, "sd") else list(-1),
    prior_lb_rho = if (n_bp > 0) lapply(pv$rho, `[[`, "lb") else list(-1),
    prior_ub_rho = if (n_bp > 0) lapply(pv$rho, `[[`, "ub") else list(-1),
    sigma_shape = outcome_priors$sigma$shape,
    sigma_scale = outcome_priors$sigma$scale,
    sigma_u_shape = outcome_priors$sigma_u$shape,
    sigma_u_scale = outcome_priors$sigma_u$scale,
    x_prop = as.double(x_prop), p_prop = as.integer(p_prop),
    latent_gps = gp_list,
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
  n_total <- n_outcome + n_alpha + 1

  all_names <- c(outcome_names, prop_names, "mean_weight")

  draws_list <- lapply(seq_len(chains), function(c) {
    mat <- raw$draws[[c]]
    colnames(mat) <- all_names
    mat
  })

  if (requireNamespace("posterior", quietly = TRUE)) {
    n_post <- nrow(draws_list[[1]])
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
      warmup = warmup,
      priors = priors,
      outcome_priors = outcome_priors,
      propensity_prior_sd = propensity_prior_sd,
      subject_var = re_info$re_group,
      merged_cols = character(0),
      shared_cols = character(0),
      zero_breakpoint = FALSE,
      propensity_formula = propensity,
      outcome_formula = outcome,
      b0_formula = b0_fixed_formula,
      b1_formula = b1,
      deltas = deltas,
      omega = omega,
      rho = rho
    ),
    class = "bjlm_fit"
  )
}


