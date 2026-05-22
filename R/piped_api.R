#' Initialise a Bayesian Joint Longitudinal Model (BJLM)
#'
#' @return A `bjlm_model` object.
#' @export
bjlm_model <- function() {
  model <- list(
    propensity = NULL,
    outcome = NULL,
    latent_gps = list()
  )
  class(model) <- "bjlm_model"
  model
}

#' Add a propensity score block to a BJLM model
#'
#' @param model A `bjlm_model` object.
#' @param formula A two-sided formula of the form `treatment ~ covariates`,
#'   specifying the propensity model. The LHS must be a binary (0/1) variable.
#' @param data A data frame containing propensity score covariates.
#' @param family A description of the error distribution and link function.
#'   Currently supports `binomial("logit")` (default).
#'
#' @return The modified `bjlm_model` object.
#' @export
propensity <- function(model, formula, data = NULL, family = binomial("logit")) {
  if (!inherits(model, "bjlm_model")) stop("First argument must be a bjlm_model object.")
  
  # Validate family
  if (is.character(family)) family <- get(family, mode = "function", envir = parent.frame())()
  if (!inherits(family, "family") || family$family != "binomial" || family$link != "logit") {
    stop("Currently only binomial('logit') is supported for the propensity model.")
  }

  model$propensity <- list(
    formula = formula,
    data = data,
    family = family
  )
  model
}

#' Add an outcome block to a BJLM model
#'
#' Specifies the outcome model. Can fit either a general piecewise change-point
#' model or a standard regression (mixed-effects or fixed) using the speed shortcut.
#'
#' @param model A `bjlm_model` object.
#' @param formula A two-sided formula. For piecewise models, of the form `y ~ tau`
#'   where `y` is the outcome and `tau` is the time variable. For standard models
#'   (no breakpoints), of the form `y ~ covariates`.
#' @param b0 A one-sided formula for the intercept model (piecewise models only).
#'   Supports random intercepts via `(1 | subject)`.
#' @param b1 A one-sided formula for the initial slope model (piecewise models only).
#' @param deltas A list of one-sided formulas for slope-change parameters (piecewise models only).
#' @param omega A list of one-sided formulas for breakpoint locations (piecewise models only).
#' @param rho A list of one-sided formulas for transition sharpness parameters (piecewise models only).
#' @param data A data frame containing all outcome variables.
#' @param family Error distribution and link. Currently only `gaussian("identity")` is supported.
#'
#' @return The modified `bjlm_model` object.
#' @export
outcome <- function(
  model,
  formula,
  b0 = NULL,
  b1 = NULL,
  deltas = list(),
  omega = list(),
  rho = list(),
  data = NULL,
  family = gaussian("identity")
) {
  if (!inherits(model, "bjlm_model")) stop("First argument must be a bjlm_model object.")

  if (is.character(family)) family <- get(family, mode = "function", envir = parent.frame())()
  if (!inherits(family, "family") || family$family != "gaussian" || family$link != "identity") {
    stop("Currently only gaussian('identity') is supported for the outcome model.")
  }

  # Detect zero-breakpoint speed shortcut:
  # b0 and b1 must both be omitted (NULL) and all breakpoint parameters empty.
  zero_breakpoint <- is.null(b0) && is.null(b1) && length(deltas) == 0 && length(omega) == 0 && length(rho) == 0

  model$outcome <- list(
    formula = formula,
    b0 = b0,
    b1 = b1,
    deltas = deltas,
    omega = omega,
    rho = rho,
    data = data,
    family = family,
    zero_breakpoint = zero_breakpoint
  )
  model
}

#' Add a latent Gaussian Process (GP) confounder to a BJLM model
#'
#' @param model A `bjlm_model` object.
#' @param name Character name of the GP confounder.
#' @param time Name of the time variable in the outcome dataset.
#' @param subject Name of the subject grouping variable.
#' @param kernel Kernel function to use. Currently supports `"se"` (Squared Exponential).
#'
#' @return The modified `bjlm_model` object.
#' @export
latent_gp <- function(model, name, time, subject, kernel = "se") {
  if (!inherits(model, "bjlm_model")) stop("First argument must be a bjlm_model object.")
  
  gp <- list(
    name = name,
    time = time,
    subject = subject,
    kernel = kernel
  )
  
  model$latent_gps[[length(model$latent_gps) + 1]] <- gp
  model
}

#' Compile a BJLM model
#'
#' Performs key alignment, demographic matching, strict Bayesian Cut validation,
#' and validates all GP specifications.
#'
#' @param model A `bjlm_model` object.
#'
#' @return A `bjlm_compiled_model` object ready to be fitted.
#' @export
compile <- function(model) {
  if (!inherits(model, "bjlm_model")) stop("Must be a bjlm_model object.")
  if (is.null(model$propensity)) stop("Model is missing propensity() specification.")
  if (is.null(model$outcome)) stop("Model is missing outcome() specification.")

  out_data <- model$outcome$data
  prop_data <- model$propensity$data %||% out_data

  if (is.null(out_data)) stop("Outcome dataset is missing.")
  
  # ---- Strict Bayesian Cut Check ----
  outcome_vars <- all.vars(model$outcome$formula)
  outcome_lhs <- outcome_vars[1]
  prop_vars <- all.vars(model$propensity$formula)
  if (outcome_lhs %in% prop_vars) {
    stop(sprintf("Bayesian Cut violation: outcome variable '%s' cannot be used in the propensity score model.", outcome_lhs))
  }

  # ---- Parse Random Intercepts ----
  # If zero-breakpoint, we parse random intercepts from the main outcome formula.
  # Otherwise, we parse them from the b0 formula.
  if (model$outcome$zero_breakpoint) {
    re_info <- .parse_re(model$outcome$formula)
  } else {
    re_info <- .parse_re(model$outcome$b0)
  }
  
  subject_var <- re_info$re_group

  # ---- GP Validation ----
  if (length(model$latent_gps) > 0) {
    gp_names <- sapply(model$latent_gps, `[[`, "name")
    if (any(duplicated(gp_names))) {
      stop("All latent GP blocks must have unique names.")
    }
    for (gp in model$latent_gps) {
      if (!is.null(subject_var) && gp$subject != subject_var) {
        stop(sprintf("GP subject variable '%s' does not match the random intercept subject variable '%s'.", gp$subject, subject_var))
      }
      if (!gp$time %in% names(out_data)) {
        stop(sprintf("GP time variable '%s' not found in outcome data.", gp$time))
      }
      if (!gp$subject %in% names(out_data)) {
        stop(sprintf("GP subject variable '%s' not found in outcome data.", gp$subject))
      }
    }
  }

  # ---- Align datasets ----
  if (!is.null(subject_var)) {
    # Longitudinal alignment: match subject-level propensity rows with outcome levels
    stopifnot("Subject variable must be present in outcome data" = subject_var %in% names(out_data))
    stopifnot("Subject variable must be present in propensity data" = subject_var %in% names(prop_data))
    
    group_factor <- as.factor(out_data[[subject_var]])
    group_indices <- as.integer(group_factor) - 1L
    n_groups <- nlevels(group_factor)
    
    idx_match <- match(levels(group_factor), prop_data[[subject_var]])
    if (any(is.na(idx_match))) {
      missing_subjects <- levels(group_factor)[is.na(idx_match)]
      stop(sprintf("Subject(s) in outcome data not found in propensity data: %s", 
                   paste(head(missing_subjects), collapse = ", ")))
    }
    subject_data <- prop_data[idx_match, , drop = FALSE]
    n_subjects <- nrow(subject_data)
  } else {
    # Cross-sectional alignment: 1:1 mapping
    group_indices <- rep(-1L, nrow(out_data))
    n_groups <- 0L
    subject_data <- prop_data
    n_subjects <- nrow(prop_data)
    if (nrow(out_data) != nrow(prop_data)) {
      stop("For cross-sectional models (no random intercept), outcome and propensity datasets must have the same number of rows.")
    }
  }

  # ---- Prepare Model Formulations & Zero-Breakpoint Shortcuts ----
  y <- as.double(out_data[[outcome_lhs]])
  n <- length(y)

  if (model$outcome$zero_breakpoint) {
    # Zero-Breakpoint Shortcut:
    # RHS of outcome formula (without random intercept) becomes b0 fixed formula.
    b0_formula <- re_info$fixed
    b1_formula <- ~ 1
    deltas <- list()
    omega <- list()
    rho <- list()
    tau <- rep(0.0, n)
    
    # We will pass a standard outcome design matrix x_b0.
    # And we will force b1 to be fixed at 0.0 in compile().
    # This is done by setting b1_fixed = TRUE or prior variance = 0.
    # Let's create an outcome prior that fixes b1 at 0.0 automatically.
    shortcut_priors <- bjlm_priors(
      outcome = smoothbp_priors(
        b1 = prior_normal(mean = 0, sd = 0) # Fixed at 0.0
      )
    )
  } else {
    # Piecewise model
    b0_formula <- re_info$fixed
    b1_formula <- model$outcome$b1
    deltas <- model$outcome$deltas
    omega <- model$outcome$omega
    rho <- model$outcome$rho
    
    tau_name <- outcome_vars[2]
    tau <- as.double(out_data[[tau_name]])
    shortcut_priors <- NULL
  }

  compiled <- list(
    model = model,
    y = y,
    tau = tau,
    b0_formula = b0_formula,
    b1_formula = b1_formula,
    deltas = deltas,
    omega = omega,
    rho = rho,
    group_indices = group_indices,
    n_groups = n_groups,
    subject_data = subject_data,
    n_subjects = n_subjects,
    treatment_name = prop_vars[1],
    prop_covariate_names = prop_vars[-1],
    shortcut_priors = shortcut_priors,
    zero_breakpoint = model$outcome$zero_breakpoint
  )
  class(compiled) <- "bjlm_compiled_model"
  compiled
}

#' Fit a compiled model
#'
#' @param object A compiled model object.
#' @param ... Additional arguments.
#'
#' @return A fitted model object.
#' @export
fit <- function(object, ...) {
  UseMethod("fit")
}

#' Fit a compiled BJLM model
#'
#' @param object A `bjlm_compiled_model` object.
#' @param priors A `bjlm_priors` object specifying priors.
#' @param ... Additional arguments passed to the fitting engine `bjlm()`.
#'
#' @return A `bjlm_fit` object.
#' @export
fit.bjlm_compiled_model <- function(object, priors = NULL, ...) {
  # If it is a zero-breakpoint shortcut and priors are NULL, use our preset shortcut priors
  if (object$zero_breakpoint && is.null(priors)) {
    priors <- object$shortcut_priors
  } else if (object$zero_breakpoint && !is.null(priors)) {
    # Make sure prior for b1 is forced to be fixed at 0.0
    priors$outcome$b1 <- prior_normal(mean = 0, sd = 0)
  }

  bjlm(
    outcome = if (object$zero_breakpoint) {
      # Dummy outcome formula: LHS ~ dummy_tau
      # We construct it dynamically
      stats::reformulate("dummy_tau", response = all.vars(object$model$outcome$formula)[1])
    } else {
      object$model$outcome$formula
    },
    b0 = object$b0_formula,
    b1 = object$b1_formula,
    deltas = object$deltas,
    omega = object$omega,
    rho = object$rho,
    propensity = object$model$propensity$formula,
    data = if (object$zero_breakpoint) {
      d <- object$model$outcome$data
      d$dummy_tau <- 0.0
      d
    } else {
      object$model$outcome$data
    },
    priors = priors,
    ...
  )
}

#' @export
print.bjlm_model <- function(x, ...) {
  cat("Bayesian Joint Longitudinal Model (BJLM) Specification\n")
  cat("====================================================\n")
  
  if (is.null(x$propensity)) {
    cat("Propensity Score Model: [Not specified]\n")
  } else {
    cat("Propensity Score Model:\n")
    cat("  Formula: ", deparse(x$propensity$formula), "\n")
  }
  
  if (is.null(x$outcome)) {
    cat("Outcome Model: [Not specified]\n")
  } else {
    cat("Outcome Model:\n")
    cat("  Formula: ", deparse(x$outcome$formula), "\n")
    if (x$outcome$zero_breakpoint) {
      cat("  Type: Standard regression (no breakpoints speed shortcut)\n")
    } else {
      cat("  Type: Piecewise change-point model\n")
      cat("  Fixed Intercept (b0): ", deparse(x$outcome$b0), "\n")
      cat("  Initial Slope (b1): ", deparse(x$outcome$b1), "\n")
      cat("  Slope Changes (deltas): ", length(x$outcome$deltas), " breakpoint(s)\n")
    }
  }
  
  if (length(x$latent_gps) > 0) {
    cat("Latent Gaussian Processes:\n")
    for (gp in x$latent_gps) {
      cat(sprintf("  - %s (time=%s, subject=%s, kernel=%s)\n", 
                  gp$name, gp$time, gp$subject, gp$kernel))
    }
  }
  
  invisible(x)
}

#' @export
print.bjlm_compiled_model <- function(x, ...) {
  cat("Compiled Bayesian Joint Longitudinal Model (BJLM)\n")
  cat("===============================================\n")
  cat("Observations: ", length(x$y), "\n")
  cat("Subjects:     ", x$n_subjects, "\n")
  if (x$n_groups > 0) {
    cat("Groups/REs:   ", x$n_groups, "\n")
  }
  cat("Model Type:   ")
  if (x$zero_breakpoint) {
    cat("Standard regression (zero-breakpoint speed shortcut)\n")
  } else {
    cat("Piecewise change-point model with ", length(x$deltas), " breakpoint(s)\n")
  }
  
  invisible(x)
}
