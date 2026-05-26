#' Simulate data from the smooth change-point model
#'
#' Generates synthetic data from the model used by \code{\link{bjlm}},
#' including optional between-subject random intercepts.  True parameter values
#' are stored as the \code{"true_params"} attribute so they can be compared
#' against posterior estimates.
#'
#' The data-generating model is:
#' \deqn{y_{ij} = (b0 + u_j) + b1 \cdot d_{ij} + b2 \cdot d_{ij} \cdot \sigma(d_{ij} \cdot \rho) + \varepsilon_{ij}}
#' where \eqn{d_{ij} = \tau_{ij} - \omega} and \eqn{\sigma(\cdot)} is the
#' logistic sigmoid.
#'
#' @param n_subj    Number of subjects (groups).  Set to \code{1} and
#'   \code{sigma_u = 0} for a single-group simulation.
#' @param n_obs     Observations per subject.  May be a scalar (same for all
#'   subjects) or a length-\code{n_subj} integer vector for unbalanced designs.
#' @param b0        Overall intercept.
#' @param b1        Pre-change-point slope.
#' @param b2        Change in slope at the change-point.
#' @param omega     Change-point location (must be positive).
#' @param rho       Sharpness of the transition (must be positive; larger values
#'   give a sharper kink).
#' @param sigma     Residual standard deviation.
#' @param sigma_u   Between-subject SD for random intercepts.  Set to \code{0}
#'   to suppress random effects.  Default \code{0.5}.
#' @param tau_range Numeric vector of length 2 giving the range of the time
#'   variable.  Observations are evenly spaced within this range for each
#'   subject.  Default \code{c(0, 6)}.
#' @param seed      Integer seed for reproducibility.  Sampled randomly if
#'   \code{NULL} (default).
#'
#' @return A \code{data.frame} with columns:
#'   \describe{
#'     \item{\code{subject}}{Subject identifier (factor).}
#'     \item{\code{tau}}{Time variable.}
#'     \item{\code{mu}}{Noise-free conditional mean \eqn{\mu_{ij}}.}
#'     \item{\code{y}}{Observed response.}
#'   }
#'   The attribute \code{"true_params"} is a named list containing the
#'   data-generating values of \code{b0}, \code{b1}, \code{b2}, \code{omega},
#'   \code{rho}, \code{sigma}, \code{sigma_u}, the vector of subject-level
#'   deviations \code{u}, and the \code{seed} used.
#'
#' @examples
#' dat <- simulate_smoothbp(
#'   n_subj = 20, n_obs = 8,
#'   b0 = 5, b1 = -0.3, b2 = 1.2,
#'   omega = 3, rho = 4, sigma = 0.4, sigma_u = 0.5,
#'   seed = 42
#' )
#' head(dat)
#' attr(dat, "true_params")
#'
#' @export
simulate_smoothbp <- function(
    n_subj    = 20L,
    n_obs     = 8L,
    b0        = 5.0,
    b1        = -0.3,
    b2        = 1.2,
    omega     = 3.0,
    rho       = 4.0,
    sigma     = 0.4,
    sigma_u   = 0.5,
    tau_range = c(0, 6),
    seed      = NULL
) {
  # ---- Validation -----------------------------------------------------------
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)
  set.seed(seed)

  stopifnot(
    is.numeric(omega), omega > 0,
    is.numeric(rho),   rho   > 0,
    is.numeric(sigma), sigma > 0,
    is.numeric(sigma_u), sigma_u >= 0,
    length(tau_range) == 2, tau_range[1] < tau_range[2],
    n_subj >= 1L
  )

  # Allow unbalanced designs
  if (length(n_obs) == 1L) n_obs <- rep(as.integer(n_obs), n_subj)
  if (length(n_obs) != n_subj) {
    stop("`n_obs` must be a scalar or a vector of length `n_subj`.")
  }

  # ---- Generate random intercepts ------------------------------------------
  u_j <- if (sigma_u > 0) rnorm(n_subj, 0, sigma_u) else rep(0.0, n_subj)

  # ---- Logistic sigmoid ----------------------------------------------------
  .sigmoid <- function(x) 1.0 / (1.0 + exp(-x))

  # ---- Build rows for each subject -----------------------------------------
  rows <- vector("list", n_subj)
  for (j in seq_len(n_subj)) {
    tau_j <- seq(tau_range[1], tau_range[2], length.out = n_obs[j])
    d_j   <- tau_j - omega
    s_j   <- .sigmoid(d_j * rho)
    mu_j  <- (b0 + u_j[j]) + b1 * d_j + b2 * d_j * s_j
    y_j   <- mu_j + rnorm(n_obs[j], 0, sigma)

    rows[[j]] <- data.frame(
      subject = j,
      tau     = tau_j,
      mu      = mu_j,
      y       = y_j
    )
  }

  dat         <- do.call(rbind, rows)
  dat$subject <- factor(dat$subject)
  rownames(dat) <- NULL

  attr(dat, "true_params") <- list(
    b0      = b0,
    b1      = b1,
    b2      = b2,
    omega   = omega,
    rho     = rho,
    sigma   = sigma,
    sigma_u = sigma_u,
    u       = u_j,
    seed    = seed
  )

  dat
}

#' Print true parameters from a simulated dataset
#'
#' Convenience function to display the data-generating parameters stored in the
#' \code{"true_params"} attribute of a dataset returned by
#' \code{\link{simulate_smoothbp}}.
#'
#' @param dat A \code{data.frame} returned by \code{simulate_smoothbp}.
#' @return The \code{true_params} list, invisibly.
#' @export
true_params <- function(dat) {
  tp <- attr(dat, "true_params")
  if (is.null(tp)) stop("`dat` does not have a `true_params` attribute.")

  scalar_params <- tp[!names(tp) %in% c("u", "X1", "pi", "X_true")]
  cat("Data-generating parameters:\n")
  for (nm in names(scalar_params)) {
    # Format vectors nicely
    val <- scalar_params[[nm]]
    if (length(val) > 1) {
      val_str <- paste(round(val, 3), collapse = ", ")
      cat(sprintf("  %-15s [%s]\n", paste0(nm, ":"), val_str))
    } else {
      cat(sprintf("  %-15s %s\n", paste0(nm, ":"), round(val, 3)))
    }
  }
  invisible(tp)
}

#' Simulate data from the Bayesian Joint Longitudinal Model (BJLM)
#'
#' A comprehensive data generator supporting multiple change-points, binary treatment assignment,
#' propensity score confounding, subject-level random intercepts, and continuous latent Gaussian Process
#' confounders. This function is designed for validation and simulation studies.
#'
#' @param n_subj Number of subjects. Default is 50.
#' @param n_obs Observations per subject. Default is 10.
#' @param b0 Baseline overall intercept. Default is 50.0.
#' @param b0_trt Causal effect of treatment on the intercept. Default is 2.0.
#' @param b1 Baseline pre-change-point slope. Default is -0.3.
#' @param omegas Vector of change-point locations. Default is \code{c(5.0)}.
#' @param rhos Vector of transition sharpness parameters. Default is \code{c(5.0)}.
#' @param deltas_int Vector of baseline slope changes at each change-point. Default is \code{c(-0.2)}.
#' @param deltas_trt Vector of treatment-specific slope changes at each change-point. Default is \code{c(0.4)}.
#' @param sigma Residual standard deviation of the outcome. Default is 1.0.
#' @param sigma_u Subject-level random intercept standard deviation. Default is 1.5.
#' @param tau_range Range of time variable. Default is \code{c(0, 10)}.
#' @param trt_intercept Baseline log-odds of treatment. Default is 0.3.
#' @param trt_x Coefficient for baseline confounder in the propensity model. Default is 0.6.
#' @param gp_confounder Logical. If \code{TRUE}, simulate a time-varying latent GP confounder. Default is \code{FALSE}.
#' @param gp_alpha Marginal standard deviation of the latent GP. Default is 1.0.
#' @param gp_rho Lengthscale of the latent GP. Default is 3.0.
#' @param gp_sigma_x Observation noise standard deviation for the noisy GP covariate. Default is 0.5.
#' @param b0_gp Coefficient for the latent GP on the outcome intercept. Default is 2.0.
#' @param trt_gp Coefficient for the baseline latent GP value on treatment assignment. Default is 0.5.
#' @param seed Reproducibility seed. Default is \code{NULL}.
#'
#' @return A \code{data.frame} with the generated dataset. True parameters are stored in the
#'   \code{"true_params"} attribute.
#' @export
simulate_bjlm <- function(
    n_subj = 50L,
    n_obs = 10L,
    b0 = 50.0,
    b0_trt = 2.0,
    b1 = -0.3,
    omegas = c(5.0),
    rhos = c(5.0),
    deltas_int = c(-0.2),
    deltas_trt = c(0.4),
    sigma = 1.0,
    sigma_u = 1.5,
    tau_range = c(0, 10),
    trt_intercept = 0.3,
    trt_x = 0.6,
    gp_confounder = FALSE,
    gp_alpha = 1.0,
    gp_rho = 3.0,
    gp_sigma_x = 0.5,
    b0_gp = 2.0,
    trt_gp = 0.5,
    seed = NULL
) {
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)
  set.seed(seed)

  n_bp <- length(omegas)
  stopifnot(
    length(rhos) == n_bp,
    length(deltas_int) == n_bp,
    length(deltas_trt) == n_bp,
    n_subj >= 1L
  )

  # Balanced or unbalanced design
  if (length(n_obs) == 1L) n_obs <- rep(as.integer(n_obs), n_subj)
  
  # 1. Subject-level baseline confounders and random effects
  X1 <- rnorm(n_subj, 0, 1)
  u_j <- if (sigma_u > 0) rnorm(n_subj, 0, sigma_u) else rep(0.0, n_subj)
  
  # 2. Simulate Latent GP Confounders if requested
  gp_list <- list()
  if (gp_confounder) {
    for (j in seq_len(n_subj)) {
      tau_j <- seq(tau_range[1], tau_range[2], length.out = n_obs[j])
      dist_mat <- as.matrix(dist(tau_j))
      K <- gp_alpha^2 * exp(-0.5 * (dist_mat / gp_rho)^2) + diag(1e-9, n_obs[j])
      L <- t(chol(K))
      z <- rnorm(n_obs[j])
      X_true <- as.numeric(L %*% z)
      X_obs <- X_true + rnorm(n_obs[j], 0, gp_sigma_x)
      
      gp_list[[j]] <- list(
        X_true = X_true,
        X_obs = X_obs,
        tau = tau_j
      )
    }
  }

  # 3. Propensity and Treatment assignment at baseline
  # Baseline GP is the first value (representing t=0 or baseline)
  X_gp_base <- if (gp_confounder) sapply(gp_list, function(gp) gp$X_true[1]) else rep(0.0, n_subj)
  
  pi_true <- if (gp_confounder) {
    stats::plogis(trt_intercept + trt_gp * X_gp_base)
  } else {
    stats::plogis(trt_intercept + trt_x * X1)
  }
  
  Trt <- stats::rbinom(n_subj, 1, pi_true)
  
  # 4. Generate longitudinal data
  .sigmoid <- function(x) 1.0 / (1.0 + exp(-x))
  
  rows <- vector("list", n_subj)
  for (j in seq_len(n_subj)) {
    tau_j <- seq(tau_range[1], tau_range[2], length.out = n_obs[j])
    
    # Baseline outcomes
    mu_j <- (b0 + u_j[j]) + b0_trt * Trt[j]
    
    # Latent GP effect
    if (gp_confounder) {
      mu_j <- mu_j + b0_gp * gp_list[[j]]$X_true
    } else {
      mu_j <- mu_j + 1.5 * X1[j] # Default confounder effect if not GP
    }
    
    # Piecewise trend
    # Add baseline pre-change-point slope
    d_base <- tau_j - omegas[1] # standard slope reference point
    mu_j <- mu_j + b1 * d_base
    
    # Add breakpoints
    for (k in seq_len(n_bp)) {
      d_k <- tau_j - omegas[k]
      s_k <- .sigmoid(d_k * rhos[k])
      delta_k <- deltas_int[k] + deltas_trt[k] * Trt[j]
      mu_j <- mu_j + delta_k * d_k * s_k
    }
    
    y_j <- mu_j + rnorm(n_obs[j], 0, sigma)
    
    df_j <- data.frame(
      subject = rep(j, n_obs[j]),
      tau     = tau_j,
      mu      = mu_j,
      y       = y_j,
      X1      = rep(X1[j], n_obs[j]),
      Trt     = rep(Trt[j], n_obs[j])
    )
    
    if (gp_confounder) {
      df_j$X_obs <- gp_list[[j]]$X_obs
    }
    
    rows[[j]] <- df_j
  }
  
  dat <- do.call(rbind, rows)
  dat$subject <- factor(dat$subject)
  rownames(dat) <- NULL
  
  true_p_list <- list(
    b0 = b0,
    b0_trt = b0_trt,
    b1 = b1,
    omegas = omegas,
    rhos = rhos,
    deltas_int = deltas_int,
    deltas_trt = deltas_trt,
    sigma = sigma,
    sigma_u = sigma_u,
    u = u_j,
    X1 = X1,
    pi = pi_true,
    seed = seed
  )
  
  if (gp_confounder) {
    true_p_list$gp_alpha = gp_alpha
    true_p_list$gp_rho = gp_rho
    true_p_list$gp_sigma_x = gp_sigma_x
    true_p_list$b0_gp = b0_gp
    true_p_list$trt_gp = trt_gp
    true_p_list$X_true = unlist(lapply(gp_list, `[[`, "X_true"))
  }
  
  attr(dat, "true_params") <- true_p_list
  dat
}
