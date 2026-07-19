#' Initialise a Bayesian Joint Longitudinal Model (BJLM)
#'
#' @return A `bjlm_model` object.
#' @export
bjlm_model <- function() {
  model <- list(
    propensity = NULL,
    outcome = NULL,
    latent_gps = list(),
    regimes = list()
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
  if (!inherits(family, "family")) stop("family must be a family object.")
  if (!((family$family == "binomial" && family$link == "logit") || 
        (family$family == "gaussian" && family$link == "identity"))) {
    stop("Only binomial('logit') or gaussian('identity') are supported for the propensity model.")
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

  if (is.character(family)) {
    family <- tolower(family)
    if (family %in% c("negbin", "nbinom", "negative_binomial", "negative binomial")) {
      family <- list(family = "negative_binomial", link = "log")
      class(family) <- "family"
    } else if (family %in% c("binomial", "logistic", "logit")) {
      family <- binomial("logit")
    } else if (family %in% c("poisson", "cloglog", "tobit", "probit")) {
      family <- list(family = family, link = "unknown")
      class(family) <- "family"
    } else {
      family <- get(family, mode = "function", envir = parent.frame())()
    }
  }
  
  if (!inherits(family, "family")) {
    stop("family must be a family object or a valid string.")
  }

  supported_families <- c("gaussian", "binomial", "negative_binomial")
  if (!family$family %in% supported_families) {
    if (family$family %in% c("poisson", "cloglog", "tobit", "probit")) {
      stop(sprintf("Family '%s' is planned but not yet implemented.", family$family))
    }
    stop(sprintf("Family '%s' is not supported. Supported families: %s", 
                 family$family, paste(supported_families, collapse = ", ")))
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
#' @param data A data frame containing the observed covariate data.
#' @param obs_var The column name of the noisy continuous observations in `data`.
#' @param time_var The column name of the time variable in `data`.
#' @param time_trt_var The column name of the time variable in the propensity data.
#' @param time_out_var The column name of the time variable in the outcome data.
#' @param subject Name of the subject grouping variable.
#' @param kernel Kernel function to use. Currently supports `"se"` (Squared Exponential).
#' @param priors A [gp_priors()] bundle for the GP hyperparameters (marginal SD
#'   `alpha`, lengthscale `rho`, observation-noise SD `sigma_x`). Defaults to
#'   `gp_priors()`, i.e. lognormal(0,1) on `alpha`, the resolution-aware lengthscale
#'   prior on `rho`, and lognormal(-1,1) on `sigma_x`.
#'
#' @return The modified `bjlm_model` object.
#' @export
latent_gp <- function(model, name, data, obs_var, time_var, time_trt_var, time_out_var, subject, kernel = "se",
                      priors = gp_priors()) {
  if (!inherits(model, "bjlm_model")) stop("First argument must be a bjlm_model object.")
  if (missing(data) || is.null(data)) stop("Must provide 'data' argument for latent_gp (containing noisy covariate observations).")
  if (!inherits(priors, "gp_priors")) stop("`priors` must be a gp_priors() object.")

  gp <- list(
    name = name,
    data = data,
    obs_var = obs_var,
    time_var = time_var,
    time_trt_var = time_trt_var,
    time_out_var = time_out_var,
    subject = subject,
    kernel = kernel,
    priors = priors
  )
  
  model$latent_gps[[length(model$latent_gps) + 1]] <- gp
  model
}

#' Add a population calibration block to a BJLM model
#'
#' Enables population-level G-computation via census weights. After fitting,
#' use \code{\link{population_predict}} to estimate population-average outcome
#' trajectories and Population Average Treatment Effects (PATE).
#'
#' @details
#' The population block replaces sample weights with census weights in the
#' G-computation path. For each posterior draw \eqn{s}, the population-average
#' outcome at time \eqn{\tau} is:
#' \deqn{Q^{(s)}(\tau) = \sum_c w_c \cdot g^{-1}\!\left(\hat{\mu}_c^{(s)}(\tau) + u_c^{(s)}\right)}
#' where \eqn{w_c} are normalised census weights, \eqn{\hat{\mu}_c^{(s)}} is the
#' fixed-effects linear predictor for cell \eqn{c}, and
#' \eqn{u_c^{(s)} \sim \mathcal{N}(0,\,\sigma_u^{(s)\,2})} is a freshly drawn
#' random effect that marginalises over the posterior of \eqn{\sigma_u} rather
#' than conditioning on any observed subject.
#'
#' For PATE/RR estimation (\code{type = "ate"} or \code{type = "rr"} in
#' \code{\link{population_predict}}), the \emph{same} random-effect draw is
#' applied to both counterfactuals, so RE terms cancel exactly for Gaussian
#' outcomes and variance is reduced for other families.
#'
#' @param model A \code{bjlm_model} object.
#' @param cells A data frame of census cells, one row per demographic stratum.
#'   Must contain all fixed-effect covariates referenced in the outcome model.
#'   Random-effect group variables should \emph{not} be included; their
#'   uncertainty is marginalised automatically.
#' @param weight Character name of the column in \code{cells} containing
#'   population counts or sampling weights. Weights are normalised to sum to
#'   one internally. If \code{NULL}, all cells are weighted equally.
#' @param strata A one-sided formula documenting the stratification variables
#'   (e.g., \code{~ age + sex + region}). Informational; used for validation
#'   and display only. All named variables must be present in \code{cells}.
#' @param at A named list of covariate overrides applied to \code{cells} before
#'   prediction. Typical use: \code{at = list(tau = 3)} to evaluate at a fixed
#'   time point, or \code{at = list(tau = 0:5)} for a population trajectory.
#'
#' @return The modified \code{bjlm_model} object.
#' @export
population <- function(model, cells, weight = NULL, strata = NULL, at = list()) {
  if (!inherits(model, "bjlm_model")) stop("First argument must be a bjlm_model object.")
  if (!is.data.frame(cells)) stop("'cells' must be a data frame.")
  if (nrow(cells) == 0L) stop("'cells' must have at least one row.")

  if (!is.null(weight)) {
    if (!is.character(weight) || length(weight) != 1L)
      stop("'weight' must be a single string naming a column in 'cells'.")
    if (!weight %in% names(cells))
      stop(sprintf("Weight column '%s' not found in 'cells'.", weight))
    wv <- cells[[weight]]
    if (anyNA(wv) || any(wv < 0))
      stop("Population weights must be non-negative and non-missing.")
    if (sum(wv) == 0)
      stop("Population weights must sum to a positive value.")
  }

  if (!is.null(strata)) {
    if (!inherits(strata, "formula"))
      stop("'strata' must be a one-sided formula (e.g., ~ age + sex + region).")
    missing_sv <- setdiff(all.vars(strata), names(cells))
    if (length(missing_sv) > 0)
      stop(sprintf("Strata variable(s) not found in 'cells': %s.",
                   paste(missing_sv, collapse = ", ")))
  }

  if (!is.list(at)) stop("'at' must be a named list (e.g., list(tau = 0:5)).")

  model$population <- list(
    cells  = cells,
    weight = weight,
    strata = strata,
    at     = at
  )
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
  if (is.null(model$outcome)) stop("Model is missing outcome() specification.")

  if (is.null(model$propensity)) {
    # Auto-generate a dummy propensity block to keep Rust backend compatible
    out_data <- model$outcome$data
    if (is.null(out_data)) stop("Outcome dataset is missing.")
    
    # Generate a unique dummy treatment name
    dummy_trt_name <- "dummy_trt"
    while (dummy_trt_name %in% names(out_data)) {
      dummy_trt_name <- paste0(dummy_trt_name, "_gp")
    }
    
    # Inject constant treatment into a copy of out_data
    out_data[[dummy_trt_name]] <- 1L
    model$outcome$data <- out_data
    
    model$propensity <- list(
      formula = stats::as.formula(paste0(dummy_trt_name, " ~ 1")),
      data = out_data,
      family = stats::binomial("logit")
    )
    model$weight_type <- "none"
    model$auto_propensity <- TRUE
  }

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

  # ---- GP Validation & Placeholders ----
  if (length(model$latent_gps) > 0) {
    gp_names <- sapply(model$latent_gps, `[[`, "name")
    if (any(duplicated(gp_names))) {
      stop("All latent GP blocks must have unique names.")
    }
    for (gp in model$latent_gps) {
      if (!is.null(subject_var) && gp$subject != subject_var) {
        stop(sprintf("GP subject variable '%s' does not match the random intercept subject variable '%s'.", gp$subject, subject_var))
      }
      
      # Validate columns in gp$data
      if (!gp$time_var %in% names(gp$data)) {
        stop(sprintf("GP time variable '%s' not found in covariate data.", gp$time_var))
      }
      if (!gp$obs_var %in% names(gp$data)) {
        stop(sprintf("GP obs variable '%s' not found in covariate data.", gp$obs_var))
      }
      if (!gp$subject %in% names(gp$data)) {
        stop(sprintf("GP subject variable '%s' not found in covariate data.", gp$subject))
      }
      
      # Inject placeholders into propensity and outcome data to satisfy model.matrix
      if (!gp$name %in% names(prop_data)) {
        prop_data[[gp$name]] <- 0.0
      }
      if (!gp$name %in% names(out_data)) {
        out_data[[gp$name]] <- 0.0
      }
    }
  }

  # ---- Align datasets ----
  shared_cols <- character(0)
  merged_cols <- character(0)
  
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
    
    # Check for name collisions and merge subject-level columns from prop_data to out_data
    out_idx_match <- match(out_data[[subject_var]], prop_data[[subject_var]])
    prop_cols <- setdiff(names(prop_data), subject_var)
    
    shared_cols <- intersect(prop_cols, names(out_data))
    if (length(shared_cols) > 0 && !identical(prop_data, out_data)) {
      warning(sprintf(
        "Shared covariate name(s) detected: %s. Scoping is independent: the propensity model uses the subject-level version, and the outcome model uses the observation-level version.",
        paste(paste0("'", shared_cols, "'"), collapse = ", ")
      ), call. = FALSE)
    }
    
    for (col in prop_cols) {
      if (!col %in% names(out_data)) {
        out_data[[col]] <- prop_data[[col]][out_idx_match]
        merged_cols <- c(merged_cols, col)
      }
    }
    
    if (length(merged_cols) > 0) {
      message(sprintf(
        "Expanded subject-level covariate(s) into outcome dataset: %s",
        paste(paste0("'", merged_cols, "'"), collapse = ", ")
      ))
    }
    
    # Save the updated out_data and prop_data back to model
    model$outcome$data <- out_data
    model$propensity$data <- prop_data
    
  } else {
    # Cross-sectional alignment: 1:1 mapping
    group_indices <- rep(-1L, nrow(out_data))
    n_groups <- 0L
    subject_data <- prop_data
    n_subjects <- nrow(prop_data)
    if (nrow(out_data) != nrow(prop_data)) {
      stop("For cross-sectional models (no random intercept), outcome and propensity datasets must have the same number of rows.")
    }
    
    # Check for name collisions in cross-sectional
    prop_cols <- setdiff(names(prop_data), subject_var)
    shared_cols <- intersect(prop_cols, names(out_data))
    if (length(shared_cols) > 0 && !identical(prop_data, out_data)) {
      warning(sprintf(
        "Shared covariate name(s) detected: %s. Scoping is independent: the propensity model uses the subject-level version, and the outcome model uses the observation-level version.",
        paste(paste0("'", shared_cols, "'"), collapse = ", ")
      ), call. = FALSE)
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

  # ---- Regime (HMM) desugaring ----
  # Two modes, chosen by the emission model of the regime block(s):
  #  * v0/v1a (obs_model = exact()): the states are KNOWN, so a regime-dependent
  #    level is a factor in the b0 design. We augment b0_formula + data here and
  #    the rest of the pipeline (design, conjugate level Gibbs, draw naming,
  #    prediction, SBC) works unchanged; v1a then fits the transition intensities
  #    from the clamped path and merges them in fit().
  #  * v1b (obs_model = confusion()): the states are LATENT (the observed
  #    indicator is a misclassified emission). The whole outcome+transition+
  #    emission model is then fit jointly by FFBS in run_regime_hmm(), so we do
  #    NOT desugar into a factor -- fit() routes to .regime_hmm_fit() instead.
  regime_mode <- NULL
  if (length(model$regimes %||% list()) > 0) {
    regime_mode <- .regime_mode(model$regimes)
    if (identical(regime_mode, "v1b")) {
      .validate_regimes_v1b(model, out_data)      # errors on unsupported combos
    } else {
      .rg <- .desugar_regimes_v0(model$regimes, out_data, b0_formula)
      b0_formula <- .rg$b0_formula        # used by SBC, prediction, zero-bp fit
      out_data <- .rg$data
      model$outcome$data <- out_data
      # fit()'s piecewise branch reads model$outcome$b0 (which retains any RE
      # term), so append the state factor(s) there too, keeping RE intact.
      if (!is.null(model$outcome$b0)) {
        model$outcome$b0 <- stats::update.formula(
          model$outcome$b0, stats::reformulate(c(".", .rg$state_cols)))
      }
    }
  }

  # ---- Population block validation ----
  if (!is.null(model$population)) {
    pop_cells <- model$population$cells
    pop_at    <- model$population$at %||% list()

    # Collect fixed-effect covariate names from outcome sub-formulas
    fv <- character(0)
    if (model$outcome$zero_breakpoint) {
      fv <- c(fv, all.vars(re_info$fixed))
    } else {
      if (!is.null(re_info$fixed)) fv <- c(fv, all.vars(re_info$fixed))
      if (!is.null(model$outcome$b1)) fv <- c(fv, all.vars(model$outcome$b1))
      for (f in c(model$outcome$deltas, model$outcome$omega, model$outcome$rho)) {
        fv <- c(fv, all.vars(f))
      }
    }
    fv <- unique(setdiff(fv, c(outcome_lhs, subject_var %||% character(0))))
    missing_fv <- setdiff(fv, names(pop_cells))
    if (length(missing_fv) > 0) {
      warning(sprintf(
        "Census cells are missing fixed-effect covariate(s) used in the outcome model: %s. Ensure these are present before calling population_predict().",
        paste(missing_fv, collapse = ", ")
      ), call. = FALSE)
    }

    if (!model$outcome$zero_breakpoint && length(outcome_vars) >= 2) {
      tau_nm_pop <- outcome_vars[2]
      if (!tau_nm_pop %in% names(pop_cells) && is.null(pop_at[["tau"]])) {
        warning(sprintf(
          "Census cells do not contain the time variable '%s' and 'at$tau' is not set in population(). Specify at = list(tau = ...) or add '%s' to cells.",
          tau_nm_pop, tau_nm_pop
        ), call. = FALSE)
      }
    }
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
    zero_breakpoint = model$outcome$zero_breakpoint,
    subject_var = subject_var,
    merged_cols = merged_cols,
    shared_cols = shared_cols,
    population  = model$population,
    regime_mode = regime_mode
  )
  class(compiled) <- "bjlm_compiled_model"

  # Optionally write a compile report. Off by default: writing to the working
  # directory on every compile() is surprising and violates CRAN policy (packages
  # must not write outside tempdir() unasked). Opt in with a path via
  # options(bjlm.compile_report = "path/to/report.md"), or TRUE for the tempdir.
  report_target <- getOption("bjlm.compile_report", FALSE)
  if (!isFALSE(report_target)) {
    report_path <- if (isTRUE(report_target)) {
      file.path(tempdir(), "compile_report.md")
    } else {
      as.character(report_target)
    }
    tryCatch({
      .write_compile_report(compiled, report_path)
    }, error = function(e) {
      warning("Could not write compile report: ", e$message, call. = FALSE)
    })
  }

  .dr_shared_confounder_note(compiled)

  compiled
}

# One-time informational note (NOT a warning) when a covariate appears in BOTH the
# propensity model and the outcome design. This is CORRECT and expected for doubly
# robust G-computation/AIPW (both nuisance models use the confounders), so a warning
# would cry wolf. But the choice of where a variable belongs -- confounder (propensity),
# effect modifier or precision covariate (outcome) -- needs subject-matter judgement,
# and in the IPW-weighted MSM fit a shared variable is adjusted twice. Surface it so an
# expert can confirm intent. Suppress with options(bjlm.quiet_dr = TRUE).
.dr_shared_confounder_note <- function(compiled) {
  if (isTRUE(getOption("bjlm.quiet_dr", FALSE))) return(invisible())
  if (isTRUE(compiled$model$auto_propensity)) return(invisible())
  prop_fml <- compiled$model$propensity$formula
  if (is.null(prop_fml)) return(invisible())

  gather_vars <- function(x) {
    if (is.null(x)) return(character(0))
    if (inherits(x, "formula")) return(all.vars(x))
    if (is.list(x)) return(unlist(lapply(x, gather_vars)))
    character(0)
  }
  prop_all  <- all.vars(prop_fml)
  trt_var   <- if (length(prop_all) > 0) prop_all[1] else character(0)
  prop_conf <- setdiff(prop_all, trt_var)
  out_vars  <- unique(gather_vars(list(
    compiled$b0_formula, compiled$b1_formula,
    compiled$deltas, compiled$omega, compiled$rho
  )))
  shared <- setdiff(intersect(prop_conf, out_vars), compiled$subject_var)
  if (length(shared) == 0) return(invisible())

  message(
    "bjlm: ", paste0("'", shared, "'", collapse = ", "),
    " appear(s) in both the propensity and outcome models. This is expected for ",
    "G-computation / AIPW, but in the IPW-weighted (MSM) fit such variables are ",
    "adjusted twice (weights and outcome regression) -- confirm that is intended. ",
    "See ?fitted.bjlm_fit; silence with options(bjlm.quiet_dr = TRUE)."
  )
  invisible()
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
#' @param dr Logical. If `TRUE`, eagerly fit the auxiliary *unweighted* outcome
#'   regression used by G-computation and AIPW (`fitted(type = "ate"/"rr"/"aipw_*")`)
#'   at fit time. If `FALSE` (default), that fit is deferred until first requested and
#'   then cached on the returned object. Requires a propensity model.
#' @param ... Additional arguments passed to the fitting engine `bjlm()`.
#'
#' @return A `bjlm_fit` object.
#' @export
fit.bjlm_compiled_model <- function(object, priors = NULL, dr = FALSE, ...) {
  # Latent-regime (misclassified indicator) models have two engines:
  #  * ISOLATED (v1b): the whole outcome IS the regime level model -> the
  #    standalone FFBS sampler run_regime_hmm() (fast, no main-loop cost).
  #  * IN-LOOP (v1c-b): the regime COMPOSES with a latent GP and/or change-point,
  #    so the FFBS runs as a Gibbs step inside run_chain_bjlm() via bjlm(regimes=).
  # Use the in-loop engine when a latent GP is present (composition needed) or
  # when explicitly requested (BJLM_REGIME_INLOOP=1, for A/B against isolated).
  regime_list <- list()
  if (identical(object$regime_mode, "v1b")) {
    inloop <- nzchar(Sys.getenv("BJLM_REGIME_INLOOP")) ||
              length(object$model$latent_gps %||% list()) > 0
    if (!inloop) return(.regime_hmm_fit(object, priors = priors, ...))
    regime_list <- .build_regime_list(object)
  }

  # If it is a zero-breakpoint shortcut and priors are NULL, use our preset shortcut priors
  if (object$zero_breakpoint && is.null(priors)) {
    priors <- object$shortcut_priors
  } else if (object$zero_breakpoint && !is.null(priors)) {
    # Make sure prior for b1 is forced to be fixed at 0.0
    priors$outcome$b1 <- prior_normal(mean = 0, sd = 0)
  }

  fit_obj <- bjlm(
    outcome = if (object$zero_breakpoint) {
      # Dummy outcome formula: LHS ~ dummy_tau
      # We construct it dynamically
      stats::reformulate("dummy_tau", response = all.vars(object$model$outcome$formula)[1])
    } else {
      object$model$outcome$formula
    },
    b0 = if (object$zero_breakpoint) {
      if (!is.null(object$subject_var)) {
        fixed_chars <- deparse(object$b0_formula)
        fixed_chars <- sub("^\\s*~\\s*", "", paste(fixed_chars, collapse = " "))
        new_fml_str <- sprintf("~ %s + (1 | %s)", fixed_chars, object$subject_var)
        stats::as.formula(new_fml_str, env = environment(object$b0_formula))
      } else {
        object$b0_formula
      }
    } else {
      object$model$outcome$b0
    },
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
    latent_gps = object$model$latent_gps,
    priors = priors,
    outcome_family = object$model$outcome$family$family %||% "gaussian",
    propensity_family = object$model$propensity$family$family %||% "binomial",
    regimes = regime_list,
    ...
  )

  # Attach compilation metadata for reproducibility and flowchart generation
  fit_obj$model <- object$model
  fit_obj$subject_var <- object$subject_var
  fit_obj$merged_cols <- object$merged_cols
  fit_obj$shared_cols <- object$shared_cols
  fit_obj$zero_breakpoint <- object$zero_breakpoint
  fit_obj$propensity_formula <- if (isTRUE(object$model$auto_propensity)) NULL else object$model$propensity$formula
  fit_obj$treatment_name <- if (isTRUE(object$model$auto_propensity)) NULL else object$treatment_name
  fit_obj$outcome_formula <- object$model$outcome$formula
  
  # Attach individual parameter formulas for prediction and plotting S3 methods
  fit_obj$b0_formula <- object$b0_formula
  fit_obj$b1_formula <- object$b1_formula
  fit_obj$deltas <- object$deltas
  fit_obj$omega <- object$omega
  fit_obj$rho <- object$rho
  fit_obj$population <- object$population

  # Retain what the lazy unweighted-outcome refit needs (the conditional outcome
  # regression E[Y|X,T] behind G-computation/AIPW; see .ensure_unweighted).
  # `cache` is an environment (reference semantics) so a draw computed on first
  # use persists across subsequent fitted() calls on the same object.
  fit_obj$compiled_model <- object
  fit_obj$priors_used <- priors
  fit_obj$fit_dots <- list(...)
  fit_obj$cache <- new.env(parent = emptyenv())

  # Eager precompute if requested (only meaningful with a propensity model).
  if (isTRUE(dr) && !is.null(fit_obj$propensity_formula)) {
    .ensure_unweighted(fit_obj)
  }

  # v1a (exact/known states): fit and attach the continuous-time transition
  # intensities. Under clamped states these are independent of the outcome level
  # model, so they are sampled separately and merged into fit_obj$draws. For the
  # latent-regime engines (v1b isolated / v1c-b in-loop) the intensities are
  # already sampled jointly, so this is skipped.
  if (length(object$model$regimes %||% list()) > 0 &&
      !identical(object$regime_mode, "v1b")) {
    fit_obj <- .attach_regime_intensities(fit_obj, object)
  }

  fit_obj
}

# Fit (once) and cache the auxiliary UNWEIGHTED outcome regression that
# G-computation and textbook AIPW require. The primary fit is IPW-weighted (an
# MSM), so its draws are NOT a conditional outcome regression E[Y|X,T]; these
# estimators need the unweighted fit for the outcome arm. Returns the unweighted
# posterior draws (same parameterisation/column names as object$draws).
.ensure_unweighted <- function(object) {
  cache <- object$cache
  if (!is.null(cache) && !is.null(cache$draws_unweighted)) {
    return(cache$draws_unweighted)
  }
  if (is.null(object$compiled_model)) {
    stop("This fit does not retain the compiled model needed to fit the unweighted ",
         "outcome regression for G-computation/AIPW. Re-fit with a current version ",
         "of bjlm (or use fitted(type = 'link'/'response')).", call. = FALSE)
  }
  if (is.null(object$propensity_formula)) {
    stop("G-computation/AIPW require a propensity model.", call. = FALSE)
  }
  message("bjlm: fitting the unweighted outcome regression E[Y|X,T] for ",
          "G-computation / AIPW (one-time; cached on this fit object). ",
          "Pass dr = TRUE to fit() to precompute this at fit time.")
  args <- object$fit_dots %||% list()
  args$weights <- "none"      # uniform weights => unweighted conditional fit
  args$verbose <- FALSE
  uw <- do.call(fit, c(list(object$compiled_model, priors = object$priors_used), args))
  if (is.null(cache)) cache <- new.env(parent = emptyenv())
  cache$draws_unweighted <- uw$draws
  cache$outcome_names <- uw$outcome_names
  object$cache <- cache
  cache$draws_unweighted
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
      cat(sprintf("  - %s (time=%s, obs=%s, subject=%s, kernel=%s)\n",
                  gp$name, gp$time_var, gp$obs_var, gp$subject, gp$kernel))
    }
  }

  if (!is.null(x$population)) {
    pop <- x$population
    cat("Population Block:\n")
    cat(sprintf("  Census cells: %d rows\n", nrow(pop$cells)))
    if (!is.null(pop$weight)) cat(sprintf("  Weight column: %s\n", pop$weight))
    if (!is.null(pop$strata)) cat(sprintf("  Strata: %s\n", deparse(pop$strata)))
    if (length(pop$at) > 0) {
      at_desc <- paste(names(pop$at), collapse = ", ")
      cat(sprintf("  at: %s\n", at_desc))
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
  if (!is.null(x$population)) {
    cat("Census cells: ", nrow(x$population$cells), "\n")
  }

  invisible(x)
}

.write_compile_report <- function(compiled, file_path = "compile_report.md") {
  model <- compiled$model
  n_obs <- length(compiled$y)
  n_sub <- compiled$n_subjects
  subject_var <- compiled$subject_var
  merged_cols <- compiled$merged_cols
  shared_cols <- compiled$shared_cols

  # Dynamic Mermaid flowchart
  flowchart <- "graph TD\n"
  flowchart <- paste0(flowchart, "    classDef prop fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px;\n")
  flowchart <- paste0(flowchart, "    classDef out fill:#e3f2fd,stroke:#1565c0,stroke-width:1px;\n")
  flowchart <- paste0(flowchart, "    classDef align fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;\n")
  flowchart <- paste0(flowchart, "    classDef collision fill:#ffebee,stroke:#c62828,stroke-width:1px;\n\n")

  flowchart <- paste0(flowchart, "    subgraph \"Propensity Block (Subject-level)\"\n")
  flowchart <- paste0(flowchart, "        PD[\"Propensity Data<br/>", n_sub, " subjects\"]:::prop\n")
  flowchart <- paste0(flowchart, "        PF[\"Formula: ", deparse(model$propensity$formula), "\"]:::prop\n")
  flowchart <- paste0(flowchart, "        PD --> PF\n")
  flowchart <- paste0(flowchart, "    end\n\n")

  flowchart <- paste0(flowchart, "    subgraph \"Outcome Block (Observation-level)\"\n")
  flowchart <- paste0(flowchart, "        OD[\"Outcome Data<br/>", n_obs, " observations\"]:::out\n")
  flowchart <- paste0(flowchart, "        OF[\"Formula: ", deparse(model$outcome$formula), "\"]:::out\n")
  flowchart <- paste0(flowchart, "        OD --> OF\n")
  flowchart <- paste0(flowchart, "    end\n\n")

  if (!is.null(subject_var)) {
    flowchart <- paste0(flowchart, "    %% Alignment & Merging\n")
    flowchart <- paste0(flowchart, "    PD -->|\"Align by subject ID: ", subject_var, "\"| OD\n")
    if (length(merged_cols) > 0) {
      flowchart <- paste0(flowchart, "    PD -.->|\"Expand subject-level: ", paste(merged_cols, collapse = ", "), "\"| OD\n")
    }
  } else {
    flowchart <- paste0(flowchart, "    PD -->|\"Cross-sectional alignment (1:1 rows)\"| OD\n")
  }

  if (length(shared_cols) > 0) {
    flowchart <- paste0(flowchart, "    %% Collision Scoping\n")
    flowchart <- paste0(flowchart, "    SC[\"Shared variables: ", paste(shared_cols, collapse = ", "), "<br/>Independent Block-Scoping\"]:::collision\n")
    flowchart <- paste0(flowchart, "    PF -.-> SC\n")
    flowchart <- paste0(flowchart, "    OF -.-> SC\n")
  }

  content <- c(
    "# BJLM Model Compilation & Alignment Report",
    "",
    "This diagnostic report was automatically generated during model compilation to provide complete transparency on how datasets are aligned, variables are scoped, and dimensions are verified.",
    "",
    "## 1. Model Summary",
    "",
    sprintf("- **Model Type:** %s", if (compiled$zero_breakpoint) "Standard regression (zero-breakpoint speed shortcut)" else "Piecewise change-point model"),
    sprintf("- **Total Observations (Outcome):** %d", n_obs),
    sprintf("- **Total Subjects:** %d", n_sub),
    sprintf("- **Grouping Variable (Random Intercept):** %s", if (is.null(subject_var)) "None (Cross-Sectional)" else subject_var),
    "",
    "## 2. Alignment Flowchart",
    "",
    "```mermaid",
    flowchart,
    "```",
    "",
    "## 3. Variable Resolution Log",
    ""
  )

  if (length(shared_cols) > 0) {
    content <- c(content, 
      "### [WARNING] Shared Covariate Names Detected",
      "",
      sprintf("The following covariate(s) are present in **both** propensity and outcome datasets: **%s**.", paste(paste0("`", shared_cols, "`"), collapse = ", ")),
      "",
      "> [!IMPORTANT]",
      "> **Transparency Resolution Rule:**",
      "> Scoping is kept strictly independent. The outcome model uses the version in the outcome dataset, while the propensity model uses the version in the propensity dataset. This is correct if they represent different observations (e.g. baseline vs. time-varying measurements of the same covariate).",
      ""
    )
  } else {
    content <- c(content,
      "### Variable Names Scoping",
      "No shared covariate names were detected across datasets. Scoping is clean and separate.",
      ""
    )
  }

  if (length(merged_cols) > 0) {
    content <- c(content,
      "### [INFO] Subject-Level Covariates Expanded",
      "",
      sprintf("The following subject-level covariate(s) from the propensity dataset were automatically expanded and aligned to the observation-level outcome dataset: **%s**.", paste(paste0("`", merged_cols, "`"), collapse = ", ")),
      "",
      "This ensures all outcome formulas (which run at the observation-level) can resolve these covariates with correct dimensions matching the total number of longitudinal observations.",
      ""
    )
  }

  writeLines(content, file_path)
}

#' Generate a Mermaid flowchart of model alignment and scoping
#'
#' Produces a Mermaid flowchart diagram showing how the propensity and
#' outcome datasets are aligned, variable scoping, and any name collisions.
#'
#' @param x A `bjlm_compiled_model` or `bjlm_fit` object.
#' @param ... Additional arguments.
#'
#' @return A character string of class `bjlm_flowchart` containing the Mermaid code.
#' @export
flowchart <- function(x, ...) {
  UseMethod("flowchart")
}

#' @export
flowchart.bjlm_compiled_model <- function(x, ...) {
  .build_flowchart(x)
}

#' @export
flowchart.bjlm_fit <- function(x, ...) {
  .build_flowchart(x)
}

#' @export
print.bjlm_flowchart <- function(x, ...) {
  cat("```mermaid\n")
  cat(x)
  cat("```\n")
  invisible(x)
}

#' Render a flowchart in the RStudio Viewer pane
#'
#' Passes the Mermaid diagram produced by [flowchart()] to the RStudio Viewer.
#' Uses `DiagrammeR` if available, otherwise falls back to `htmltools`.
#'
#' @param x A `bjlm_compiled_model`, `bjlm_fit`, or `bjlm_flowchart` object.
#' @param ... Unused.
#'
#' @return `x`, invisibly.
#' @export
view_flowchart <- function(x, ...) {
  fc <- if (inherits(x, "bjlm_flowchart")) x else flowchart(x)
  mermaid_src <- as.character(fc)

  if (requireNamespace("htmltools", quietly = TRUE)) {
    html <- htmltools::browsable(htmltools::HTML(paste0(
      '<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>',
      '<div class="mermaid">', mermaid_src, '</div>',
      '<script>mermaid.initialize({startOnLoad:true,theme:"default"})</script>'
    )))
    print(html)
  } else if (requireNamespace("DiagrammeR", quietly = TRUE)) {
    print(DiagrammeR::mermaid(mermaid_src))
  } else {
    stop("Install 'htmltools' (recommended) or 'DiagrammeR' to view flowcharts in the Viewer pane.")
  }

  invisible(x)
}

.build_flowchart <- function(x) {
  # Standardise attributes based on object class
  if (inherits(x, "bjlm_compiled_model")) {
    n_obs <- length(x$y)
    n_sub <- x$n_subjects
    subject_var <- x$subject_var
    merged_cols <- x$merged_cols
    shared_cols <- x$shared_cols
    prop_fml <- x$model$propensity$formula
    out_fml <- x$model$outcome$formula
    latent_gps <- x$model$latent_gps
  } else if (inherits(x, "bjlm_fit")) {
    n_obs <- x$n
    n_sub <- x$n_subjects
    subject_var <- x$subject_var
    merged_cols <- x$merged_cols
    shared_cols <- x$shared_cols
    prop_fml <- x$propensity_formula
    out_fml <- x$outcome_formula
    latent_gps <- x$model$latent_gps
  } else {
    stop("Must be a bjlm_compiled_model or bjlm_fit object.")
  }
  if (is.null(latent_gps)) latent_gps <- list()

  flowchart <- "graph TD\n"
  flowchart <- paste0(flowchart, "    classDef prop fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px;\n")
  flowchart <- paste0(flowchart, "    classDef out fill:#e3f2fd,stroke:#1565c0,stroke-width:1px;\n")
  flowchart <- paste0(flowchart, "    classDef gp fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px;\n")
  flowchart <- paste0(flowchart, "    classDef collision fill:#ffebee,stroke:#c62828,stroke-width:1px;\n\n")

  flowchart <- paste0(flowchart, "    subgraph \"Propensity Block (Subject-level)\"\n")
  flowchart <- paste0(flowchart, "        PD[\"Propensity Data<br/>", n_sub, " subjects\"]:::prop\n")
  flowchart <- paste0(flowchart, "        PF[\"Formula: ", deparse(prop_fml), "\"]:::prop\n")
  flowchart <- paste0(flowchart, "        PD --> PF\n")
  flowchart <- paste0(flowchart, "    end\n\n")

  flowchart <- paste0(flowchart, "    subgraph \"Outcome Block (Observation-level)\"\n")
  flowchart <- paste0(flowchart, "        OD[\"Outcome Data<br/>", n_obs, " observations\"]:::out\n")
  flowchart <- paste0(flowchart, "        OF[\"Formula: ", deparse(out_fml), "\"]:::out\n")
  flowchart <- paste0(flowchart, "        OD --> OF\n")
  flowchart <- paste0(flowchart, "    end\n\n")

  # Latent GP blocks
  for (i in seq_along(latent_gps)) {
    gp <- latent_gps[[i]]
    gp_id   <- paste0("GP", i)
    gp_d_id <- paste0("GP", i, "D")
    n_gp_obs <- nrow(gp$data)
    kernel_label <- toupper(gp$kernel %||% "SE")
    flowchart <- paste0(flowchart, "    subgraph \"Latent GP: ", gp$name, "\"\n")
    flowchart <- paste0(flowchart, "        ", gp_d_id, "[\"", gp$obs_var, " data<br/>",
                        n_gp_obs, " observations\"]:::gp\n")
    flowchart <- paste0(flowchart, "        ", gp_id, "[\"", gp$name,
                        " ~ GP(", kernel_label, " kernel)\"]:::gp\n")
    flowchart <- paste0(flowchart, "        ", gp_d_id, " --> ", gp_id, "\n")
    flowchart <- paste0(flowchart, "    end\n\n")
    flowchart <- paste0(flowchart, "    ", gp_id, " -->|\"Query at ", gp$time_trt_var, "\"| PF\n")
    flowchart <- paste0(flowchart, "    ", gp_id, " -->|\"Query at ", gp$time_out_var, "\"| OF\n\n")
  }

  if (!is.null(subject_var)) {
    flowchart <- paste0(flowchart, "    %% Alignment & Merging\n")
    flowchart <- paste0(flowchart, "    PD -->|\"Align by subject ID: ", subject_var, "\"| OD\n")
    if (length(merged_cols) > 0) {
      flowchart <- paste0(flowchart, "    PD -.->|\"Expand subject-level: ", paste(merged_cols, collapse = ", "), "\"| OD\n")
    }
  } else {
    flowchart <- paste0(flowchart, "    PD -->|\"Cross-sectional alignment (1:1 rows)\"| OD\n")
  }

  if (length(shared_cols) > 0) {
    flowchart <- paste0(flowchart, "    %% Collision Scoping\n")
    flowchart <- paste0(flowchart, "    SC[\"Shared variables: ", paste(shared_cols, collapse = ", "), "<br/>Independent Block-Scoping\"]:::collision\n")
    flowchart <- paste0(flowchart, "    PF -.-> SC\n")
    flowchart <- paste0(flowchart, "    OF -.-> SC\n")
  }

  structure(flowchart, class = "bjlm_flowchart")
}

#' Fitted values, G-computation, and Doubly Robust AIPW for bjlm_fit objects
#'
#' Estimates marginal counterfactual predictions and causal effects (ATE and marginal Risk Ratio)
#' using either G-computation (standardisation) or doubly robust Augmented Inverse Probability Weighting (AIPW).
#'
#' @details
#' \subsection{Two outcome fits: weighted (MSM) and unweighted (conditional)}{
#'   The primary \code{bjlm} outcome model is fitted with the estimated
#'   inverse-probability weights carried through MCMC, i.e. a Bayesian
#'   \emph{marginal structural model} (MSM); this is what \code{type = "link"} and
#'   \code{"response"} return. The causal estimators below (\code{"ate"}, \code{"rr"},
#'   \code{"aipw_ate"}, \code{"aipw_rr"}) instead require the \emph{conditional} outcome
#'   regression \eqn{E[Y \mid X, T]}, so they are built from an auxiliary
#'   \strong{unweighted} outcome fit. That fit is done once, on first use, and cached on
#'   the object (pass \code{dr = TRUE} to \code{fit()} to precompute it). Using the
#'   weighted MSM fit here would double-count the propensity.
#' }
#' \subsection{G-computation (\code{type = "ate"}, \code{"rr"})}{
#'   Predict individual-level potential outcomes \eqn{\hat{Y}_i(a)} under a counterfactual
#'   treatment \eqn{a \in \{0, 1\}} from the unweighted outcome regression, then average:
#'   \deqn{\hat{\mu}_a^{(s)} = \frac{1}{N} \sum_{i=1}^N \hat{Y}_i(a)^{(s)}}
#'   \deqn{\text{RR}^{(s)} = \hat{\mu}_1^{(s)} / \hat{\mu}_0^{(s)}, \qquad \text{ATE}^{(s)} = \hat{\mu}_1^{(s)} - \hat{\mu}_0^{(s)}}
#'   This is standard G-computation (consistent if the outcome model is correctly specified).
#' }
#' \subsection{AIPW (\code{type = "aipw_ate"}, \code{"aipw_rr"})}{
#'   Doubly robust: the unweighted outcome regression \eqn{\hat{Y}_i(a)} augmented by an
#'   IPW term using the propensity \eqn{\hat{\pi}_i} from the propensity model,
#'   \deqn{\hat{\mu}_a^{\text{AIPW},(s)} = \frac{1}{N} \sum_{i=1}^N \left( \hat{Y}_i(a)^{(s)} + \frac{\mathbb{1}\{T_i=a\} (Y_i - \hat{Y}_i(a)^{(s)})}{\hat{\pi}_{i,a}^{(s)}} \right)}
#'   consistent if \emph{either} the propensity or the outcome model is correctly specified.
#'   It is normal, and desirable, for confounders to appear in \emph{both} the propensity
#'   and outcome models.
#' }
#'
#' @param object A \code{bjlm_fit} object.
#' @param newdata An optional data frame. If omitted, the training data is used. For AIPW, must contain observed outcome and treatment columns.
#' @param type The type of prediction or causal effect to return:
#'   \itemize{
#'     \item \code{"link"}: Posterior linear predictor draws.
#'     \item \code{"response"}: Posterior expected value draws on the response scale.
#'     \item \code{"rr"}: G-computation marginal Risk Ratio.
#'     \item \code{"ate"}: G-computation Average Treatment Effect.
#'     \item \code{"aipw_ate"}: Doubly robust AIPW Average Treatment Effect.
#'     \item \code{"aipw_rr"}: Doubly robust AIPW marginal Risk Ratio.
#'   }
#' @param summary Logical; if \code{TRUE} (default), returns a summary data frame of posterior statistics;
#'   if \code{FALSE}, returns the full posterior draws matrix.
#' @param link_override Passed to \code{.build_predictions}.
#' @param truncation Numeric clipping threshold for propensity scores to maintain AIPW numerical stability (default: \code{1e-5}).
#' @param ... Additional arguments (ignored).
#'
#' @return If \code{summary = TRUE}, a data frame. If \code{summary = FALSE}, a matrix of draws.
#' @export
fitted.bjlm_fit <- function(object, newdata = NULL, type = c("link", "response", "rr", "ate", "aipw_ate", "aipw_rr"), summary = TRUE, link_override = NULL, truncation = NULL, ...) {
  type <- match.arg(type)
  data <- if (is.null(newdata)) object$data else newdata

  if (type %in% c("rr", "ate", "aipw_ate", "aipw_rr")) {
    fam <- object$model$outcome$family$family %||% "gaussian"
    if (type %in% c("rr", "aipw_rr") && fam != "binomial") {
      stop("Risk Ratio is only applicable to binomial outcome models.")
    }
    
    prop_formula <- object$propensity_formula
    if (is.null(prop_formula)) {
      stop("A causal effect or AIPW estimation requires a fitted propensity score model.")
    }
    
    trt_var <- all.vars(prop_formula)[1]
    if (is.na(trt_var) || is.null(trt_var)) {
      stop("Could not identify treatment variable from propensity formula.")
    }
    
    if (!trt_var %in% names(data)) {
      stop(sprintf("Observed treatment column '%s' is required in data.", trt_var))
    }
    trt <- as.double(data[[trt_var]])
    
    if (!all(trt %in% c(0, 1))) {
      stop("Causal effect estimation for these types is only supported for binary treatments/exposures (treatment must be 0 or 1).")
    }

    # 1. Compute potential outcomes under Trt=1 and Trt=0 from the UNWEIGHTED
    #    conditional outcome regression E[Y|X,T]. The primary fit is IPW-weighted
    #    (an MSM); standardising / augmenting that would double-count the propensity.
    #    G-computation and AIPW both require the unweighted outcome arm.
    obj_pred <- object
    obj_pred$draws <- .ensure_unweighted(object)

    data1 <- data
    data1[[trt_var]] <- 1
    p1_draws <- .build_predictions(obj_pred, newdata = data1, type = "response", summary = FALSE)

    data0 <- data
    data0[[trt_var]] <- 0
    p0_draws <- .build_predictions(obj_pred, newdata = data0, type = "response", summary = FALSE)
    
    n_draws <- nrow(p1_draws)
    n_obs <- ncol(p1_draws)

    if (type %in% c("rr", "ate")) {
      # G-computation
      p1_marg <- rowMeans(p1_draws)
      p0_marg <- rowMeans(p0_draws)
      
      if (type == "rr") {
        rr_draws <- as.matrix(p1_marg / p0_marg)
        colnames(rr_draws) <- "RR"
        if (!summary) return(rr_draws)
        return(data.frame(
          .observation = 1,
          fitted_mean = mean(rr_draws),
          fitted_Q2.5 = stats::quantile(rr_draws, 0.025),
          fitted_Q97.5 = stats::quantile(rr_draws, 0.975)
        ))
      } else {
        ate_draws <- as.matrix(p1_marg - p0_marg)
        colnames(ate_draws) <- "ATE"
        if (!summary) return(ate_draws)
        return(data.frame(
          .observation = 1,
          fitted_mean = mean(ate_draws),
          fitted_Q2.5 = stats::quantile(ate_draws, 0.025),
          fitted_Q97.5 = stats::quantile(ate_draws, 0.975)
        ))
      }
    } else {
      # AIPW (doubly robust): unweighted conditional outcome regression Yhat(a)
      # (via obj_pred above) + IPW augmentation with pi from the propensity model.
      # This is the textbook augmentation and is consistent if EITHER the
      # propensity or the outcome model is correct.
      outcome_vars <- all.vars(object$outcome_formula)
      y_name <- outcome_vars[1]
      if (!y_name %in% names(data)) {
        stop(sprintf("Observed outcome column '%s' is required in data for AIPW estimation.", y_name))
      }
      y_obs <- as.double(data[[y_name]])

      # Reconstruct propensity scores
      subject_var <- object$subject_var
      subject_data <- if (!is.null(subject_var) && subject_var %in% names(data)) {
        data[!duplicated(data[[subject_var]]), , drop = FALSE]
      } else {
        data
      }
      
      rhs_formula <- formula(delete.response(terms(prop_formula)))
      x_prop <- model.matrix(rhs_formula, data = subject_data)
      
      # Extract alpha draws
      alpha_names <- object$propensity_names
      if (is.null(alpha_names)) {
        alpha_names <- colnames(posterior::as_draws_matrix(object$draws))
        alpha_names <- alpha_names[grepl("^alpha_", alpha_names)]
      }
      
      if (length(alpha_names) == 0) {
        stop("Could not find propensity parameter draws (alpha) in the fitted model.")
      }
      
      alpha_draws <- posterior::subset_draws(object$draws, variable = alpha_names)
      alpha_mat <- posterior::as_draws_matrix(alpha_draws)
      
      # Ensure column order matches colnames(x_prop)
      colnames_alpha <- colnames(alpha_mat)
      col_indices <- sapply(colnames(x_prop), function(col) {
        target <- paste0("alpha_", col)
        idx <- which(colnames_alpha == target)
        if (length(idx) == 0) {
          idx <- which(grepl(col, colnames_alpha, fixed = TRUE))
        }
        if (length(idx) == 0) {
          stop(sprintf("Could not find propensity coefficient for covariate '%s' in posterior draws.", col))
        }
        idx[1]
      })
      
      alpha_mat <- alpha_mat[, col_indices, drop = FALSE]
      class(alpha_mat) <- "matrix"

      # Vectorized linear predictor calculation
      eta_mat <- alpha_mat %*% t(x_prop)
      pi_mat <- 1 / (1 + exp(-eta_mat))
      
      # Propensity Score Truncation / Clipping
      trunc_min <- 1e-5
      trunc_max <- 1 - 1e-5
      if (!is.null(truncation)) {
        if (length(truncation) == 1) {
          trunc_min <- truncation
          trunc_max <- 1 - truncation
        } else if (length(truncation) == 2) {
          trunc_min <- truncation[1]
          trunc_max <- truncation[2]
        }
      }
      pi_mat <- pmax(pmin(pi_mat, trunc_max), trunc_min)
      
      # Expand to observation level
      if (!is.null(subject_var) && subject_var %in% names(data)) {
        obs_subjects <- data[[subject_var]]
        subj_indices <- match(obs_subjects, subject_data[[subject_var]])
        pi_obs <- pi_mat[, subj_indices, drop = FALSE]
      } else {
        pi_obs <- pi_mat
      }

      # Compute individual AIPW terms fully vectorized
      trt_mat <- matrix(trt, nrow = n_draws, ncol = n_obs, byrow = TRUE)
      y_mat <- matrix(y_obs, nrow = n_draws, ncol = n_obs, byrow = TRUE)
      
      aipw1_mat <- p1_draws + (trt_mat * (y_mat - p1_draws)) / pi_obs
      aipw0_mat <- p0_draws + ((1 - trt_mat) * (y_mat - p0_draws)) / (1 - pi_obs)
      
      mu1_aipw <- rowMeans(aipw1_mat)
      mu0_aipw <- rowMeans(aipw0_mat)
      
      if (type == "aipw_ate") {
        aipw_ate_draws <- as.matrix(mu1_aipw - mu0_aipw)
        colnames(aipw_ate_draws) <- "AIPW_ATE"
        if (!summary) return(aipw_ate_draws)
        return(data.frame(
          .observation = 1,
          fitted_mean = mean(aipw_ate_draws),
          fitted_Q2.5 = stats::quantile(aipw_ate_draws, 0.025),
          fitted_Q97.5 = stats::quantile(aipw_ate_draws, 0.975)
        ))
      } else {
        aipw_rr_draws <- as.matrix(mu1_aipw / mu0_aipw)
        colnames(aipw_rr_draws) <- "AIPW_RR"
        if (!summary) return(aipw_rr_draws)
        return(data.frame(
          .observation = 1,
          fitted_mean = mean(aipw_rr_draws),
          fitted_Q2.5 = stats::quantile(aipw_rr_draws, 0.025),
          fitted_Q97.5 = stats::quantile(aipw_rr_draws, 0.975)
        ))
      }
    }
  }

  .build_predictions(object, newdata, type, summary, ...)
}

#' Population-level G-computation for bjlm_fit objects
#'
#' Estimates the population-average outcome trajectory or the Population Average
#' Treatment Effect (PATE) by applying census weights to G-computation
#' predictions. Random effects for census cells are marginalised over the
#' posterior of \eqn{\sigma_u} via Monte Carlo rather than conditioned on any
#' observed subject.
#'
#' @param object A \code{bjlm_fit} object. Should have been fitted from a model
#'   with a \code{\link{population}} block, or the \code{population} argument
#'   must be supplied here.
#' @param ... Additional arguments (ignored).
#' @export
population_predict <- function(object, ...) UseMethod("population_predict")

#' @rdname population_predict
#'
#' @param population Optional population spec (a list with fields \code{cells},
#'   \code{weight}, \code{strata}, \code{at}) to use instead of the one stored
#'   in \code{object$population}.
#' @param type Character. One of:
#'   \itemize{
#'     \item \code{"response"}: Population-weighted mean outcome on the response
#'       scale, marginalising over random effects.
#'     \item \code{"ate"}: Population Average Treatment Effect (PATE),
#'       \eqn{E_{\text{pop}}[Y(1)] - E_{\text{pop}}[Y(0)]}. Requires a
#'       propensity model.
#'     \item \code{"rr"}: Population marginal Risk Ratio,
#'       \eqn{E_{\text{pop}}[Y(1)] / E_{\text{pop}}[Y(0)]}. Requires a
#'       propensity model and a binomial outcome.
#'   }
#' @param at Named list of covariate overrides applied to the census cells
#'   before prediction, overriding any \code{at} stored in the population spec.
#'   Use \code{at = list(tau = 0:5)} to obtain a population trajectory.
#' @param summary Logical. If \code{TRUE} (default), returns a data frame of
#'   posterior statistics (\code{pop_mean}, \code{pop_Q2.5}, \code{pop_Q97.5},
#'   plus a \code{tau} column when \code{at$tau} has length > 1). If
#'   \code{FALSE}, returns a numeric vector of posterior draws (or a matrix with
#'   one column per tau value).
#' @param seed Integer seed for the Monte Carlo marginalisation of random
#'   effects. Set for reproducibility.
#'
#' @return A data frame (if \code{summary = TRUE}) or a numeric vector/matrix
#'   (if \code{summary = FALSE}).
#' @export
population_predict.bjlm_fit <- function(
  object,
  population = NULL,
  type       = c("response", "ate", "rr"),
  at         = NULL,
  summary    = TRUE,
  seed       = NULL,
  ...
) {
  type <- match.arg(type)

  pop_spec <- population %||% object$population
  if (is.null(pop_spec)) {
    stop("No population specification found. Add a population() block to the model, or supply the 'population' argument.")
  }

  cells      <- pop_spec$cells
  weight_col <- pop_spec$weight
  at_vals    <- at %||% pop_spec$at %||% list()

  # Normalised census weights
  w <- if (!is.null(weight_col) && weight_col %in% names(cells)) {
    wv <- as.double(cells[[weight_col]])
    wv / sum(wv)
  } else {
    rep(1.0 / nrow(cells), nrow(cells))
  }

  fam <- object$model$outcome$family$family %||% "gaussian"

  if (type == "rr" && fam != "binomial") {
    stop("type = 'rr' requires a binomial outcome model.")
  }
  if (type %in% c("ate", "rr")) {
    if (is.null(object$propensity_formula)) {
      stop(sprintf("type = '%s' requires a fitted propensity model.", type))
    }
    trt_var <- all.vars(object$propensity_formula)[1]
    if (!trt_var %in% names(cells)) {
      stop(sprintf(
        "Treatment variable '%s' not found in census cells. Include it to define the population treatment distribution.",
        trt_var
      ))
    }
  }

  # For zero-breakpoint models the outcome formula is Y ~ dummy_tau; inject it.
  if (isTRUE(object$zero_breakpoint)) cells[["dummy_tau"]] <- 0.0

  # Apply non-tau at_vals overrides to cells now (tau is handled per-iteration below)
  for (var_nm in setdiff(names(at_vals), "tau")) {
    cells[[var_nm]] <- at_vals[[var_nm]]
  }

  outcome_vars   <- all.vars(object$outcome_formula)
  tau_name_local <- outcome_vars[2]

  tau_grid <- if (!is.null(at_vals[["tau"]]) && !isTRUE(object$zero_breakpoint)) {
    as.double(at_vals[["tau"]])
  } else {
    NA_real_
  }

  if (!is.null(seed)) set.seed(seed)

  # Causal PATE (ate/rr) standardises the CONDITIONAL outcome regression E[Y|X,T],
  # so it uses the auxiliary UNWEIGHTED outcome fit (the primary fit is an
  # IPW-weighted MSM; standardising that would double-count the propensity, exactly
  # as in fitted(type = "ate")). Descriptive response/link predictions keep the
  # primary (weighted) fit.
  pred_object <- object
  if (type %in% c("ate", "rr")) {
    pred_object$draws <- .ensure_unweighted(object)
  }

  draw_mat  <- posterior::as_draws_matrix(pred_object$draws)
  col_names <- colnames(draw_mat)
  n_draws   <- nrow(draw_mat)
  n_cells   <- nrow(cells)

  sigma_u_col <- match("sigma_u", col_names)
  has_re <- !is.null(object$subject_var) && !is.na(sigma_u_col)

  # Pre-draw standard normals for RE marginalisation once; scale per draw inside.
  # z_mat[s, c] * sigma_u_draws[s] gives u_c^(s) ~ N(0, sigma_u^(s)).
  z_mat          <- NULL
  sigma_u_draws  <- NULL
  if (has_re) {
    z_mat         <- matrix(rnorm(n_draws * n_cells), nrow = n_draws, ncol = n_cells)
    sigma_u_draws <- as.numeric(draw_mat[, sigma_u_col])
  }

  inv_link <- switch(fam,
    gaussian          = identity,
    binomial          = function(x) 1 / (1 + exp(-x)),
    negative_binomial = exp,
    identity
  )

  # Returns n_draws x n_cells link-scale prediction matrix, no observed RE applied.
  .lp_mat <- function(cells_eval) {
    .build_predictions(pred_object, newdata = cells_eval, type = "link", summary = FALSE)
  }

  # Applies RE marginalisation and census weighting; returns n_draws-length vector.
  .weighted_response <- function(lp_mat, re_mat) {
    if (!is.null(re_mat)) lp_mat <- lp_mat + re_mat
    as.vector(inv_link(lp_mat) %*% w)
  }

  # Core computation for a single tau value (NA = use tau column already in cells).
  .pop_draws_at_tau <- function(tau_val) {
    cells_eval <- cells
    if (!is.na(tau_val)) cells_eval[[tau_name_local]] <- tau_val

    # RE matrix: same draw used for both counterfactuals to reduce variance.
    re_mat <- if (has_re) sweep(z_mat, 1, sigma_u_draws, `*`) else NULL

    if (type == "response") {
      return(.weighted_response(.lp_mat(cells_eval), re_mat))
    }

    # PATE / Population RR
    cells_1 <- cells_eval; cells_1[[trt_var]] <- 1.0
    cells_0 <- cells_eval; cells_0[[trt_var]] <- 0.0

    q1 <- .weighted_response(.lp_mat(cells_1), re_mat)
    q0 <- .weighted_response(.lp_mat(cells_0), re_mat)

    if (type == "ate") q1 - q0 else q1 / q0
  }

  # Handle trajectory (at$tau vector) vs single-point
  if (length(tau_grid) > 1) {
    draws_list <- lapply(tau_grid, .pop_draws_at_tau)

    if (!summary) {
      out_mat <- do.call(cbind, draws_list)
      colnames(out_mat) <- paste0("tau=", tau_grid)
      return(out_mat)
    }

    rows <- lapply(seq_along(tau_grid), function(i) {
      d <- draws_list[[i]]
      data.frame(
        tau       = tau_grid[i],
        pop_mean  = mean(d),
        pop_Q2.5  = stats::quantile(d, 0.025),
        pop_Q97.5 = stats::quantile(d, 0.975)
      )
    })
    return(do.call(rbind, rows))
  }

  pop_draws <- .pop_draws_at_tau(if (is.na(tau_grid)) NA_real_ else tau_grid)

  if (!summary) return(pop_draws)

  data.frame(
    pop_mean  = mean(pop_draws),
    pop_Q2.5  = stats::quantile(pop_draws, 0.025),
    pop_Q97.5 = stats::quantile(pop_draws, 0.975)
  )
}

.build_predictions_dm <- function(object, data) {
  b0_fml <- object$b0_formula
  b1_fml <- object$b1_formula
  deltas_fml <- object$deltas
  omega_fml <- object$omega
  rho_fml <- object$rho

  # For zero-breakpoint models, b0_formula is stored as the two-sided outcome
  # formula (e.g. Y ~ X1). Strip the LHS so model.matrix doesn't look for the
  # response in newdata (which may not have it, e.g. census cells).
  if (length(b0_fml) == 3L) {
    b0_fml <- stats::as.formula(
      paste("~", paste(deparse(b0_fml[[3L]]), collapse = " ")),
      env = environment(b0_fml)
    )
  }

  X_b0 <- stats::model.matrix(b0_fml, data = data)
  if (is.null(b1_fml)) {
    X_b1 <- stats::model.matrix(~ 0, data = data)
  } else {
    X_b1 <- stats::model.matrix(b1_fml, data = data)
  }
  
  n_bp <- length(deltas_fml)
  X_deltas <- if (n_bp > 0) lapply(deltas_fml, function(f) stats::model.matrix(f, data = data)) else list()
  X_om     <- if (n_bp > 0) lapply(omega_fml,  function(f) stats::model.matrix(f, data = data)) else list()
  X_rho    <- if (n_bp > 0) lapply(rho_fml,    function(f) stats::model.matrix(f, data = data)) else list()
  
  subject_var <- object$subject_var
  n_groups_b0 <- object$n_subjects
  
  if (!is.null(subject_var) && subject_var %in% names(data)) {
    group_levels <- levels(as.factor(object$data[[subject_var]]))
    gfac <- factor(data[[subject_var]], levels = group_levels)
    group_b0 <- ifelse(is.na(gfac), -1L, as.integer(gfac) - 1L)
  } else {
    group_b0 <- rep(-1L, nrow(data))
  }
  
  list(
    X_b0 = X_b0,
    X_b1 = X_b1,
    X_deltas = X_deltas,
    X_om = X_om,
    X_rho = X_rho,
    group_b0 = group_b0,
    n_groups_b0 = n_groups_b0
  )
}

.build_predictions <- function(object, newdata = NULL, type = "link", summary = TRUE, ...) {
  if (is.null(newdata)) {
    data <- object$data
  } else {
    data <- newdata
  }

  outcome_vars <- all.vars(object$outcome_formula)
  y_name <- outcome_vars[1]
  tau_name <- outcome_vars[2]
  tau <- as.double(data[[tau_name]])

  dm <- .build_predictions_dm(object, data)

  n <- length(tau)
  draw_mat  <- posterior::as_draws_matrix(object$draws)
  col_names <- colnames(draw_mat)
  n_draws   <- nrow(draw_mat)
  n_bp      <- length(dm$X_deltas)

  b0_cols  <- match(paste0("b0_", colnames(dm$X_b0)), col_names)
  b1_cols  <- if (ncol(dm$X_b1) > 0) match(paste0("b1_", colnames(dm$X_b1)), col_names) else integer(0)
  
  subject_var <- object$subject_var
  group_levels <- if (!is.null(subject_var)) levels(as.factor(object$data[[subject_var]])) else character(0)
  u_cols <- if (length(group_levels) > 0) match(paste0("u_", group_levels), col_names) else integer(0)

  delta_cols_list <- lapply(seq_len(n_bp), function(k) match(paste0("delta", k, "_", colnames(dm$X_deltas[[k]])), col_names))
  om_cols_list    <- lapply(seq_len(n_bp), function(k) match(paste0("omega", k, "_", colnames(dm$X_om[[k]])), col_names))
  rho_cols_list   <- lapply(seq_len(n_bp), function(k) match(paste0("rho", k, "_", colnames(dm$X_rho[[k]])), col_names))

  gamma_b1_cols <- which(grepl("^gamma_b1_", col_names))
  gamma_delta_cols_list <- lapply(seq_len(n_bp), function(k) which(grepl(paste0("^gamma_delta", k, "_"), col_names)))

  fitted_draws <- matrix(0, nrow = n_draws, ncol = n)
  for (s in seq_len(n_draws)) {
    mu_i <- as.vector(dm$X_b0 %*% as.numeric(draw_mat[s, b0_cols]))
    b1_vals <- if (ncol(dm$X_b1) == 0) {
      rep(0, n)
    } else {
      beta_b1 <- as.numeric(draw_mat[s, b1_cols])
      if (length(gamma_b1_cols) > 0) beta_b1 <- beta_b1 * as.numeric(draw_mat[s, gamma_b1_cols])
      as.vector(dm$X_b1 %*% beta_b1)
    }
    
    if (n_bp > 0) {
      om1_i <- as.vector(dm$X_om[[1]] %*% as.numeric(draw_mat[s, om_cols_list[[1]]]))
      mu_i  <- mu_i + b1_vals * (tau - om1_i)
    } else {
      mu_i  <- mu_i + b1_vals * tau
    }
    
    for (k in seq_len(n_bp)) {
      b_delta <- as.numeric(draw_mat[s, delta_cols_list[[k]]])
      if (length(gamma_delta_cols_list[[k]]) > 0) b_delta <- b_delta * as.numeric(draw_mat[s, gamma_delta_cols_list[[k]]])
      delta_i <- as.vector(dm$X_deltas[[k]] %*% b_delta)
      om_i    <- as.vector(dm$X_om[[k]] %*% as.numeric(draw_mat[s, om_cols_list[[k]]]))
      rho_i   <- as.vector(dm$X_rho[[k]] %*% as.numeric(draw_mat[s, rho_cols_list[[k]]]))
      di <- tau - om_i
      si <- 1 / (1 + exp(-di * rho_i))
      mu_i <- mu_i + delta_i * di * si
    }
    
    if (length(u_cols) > 0 && !any(is.na(u_cols))) {
      u_b0 <- as.numeric(draw_mat[s, u_cols])
      for (i in seq_len(n)) {
        g <- dm$group_b0[i]
        if (g >= 0L) mu_i[i] <- mu_i[i] + u_b0[g + 1L]
      }
    }
    
    fitted_draws[s, ] <- mu_i
  }

  fam <- object$model$outcome$family$family %||% "gaussian"
  if (type == "response") {
    if (fam == "binomial") {
      fitted_draws <- 1 / (1 + exp(-fitted_draws))
    } else if (fam == "negative_binomial") {
      fitted_draws <- exp(fitted_draws)
    }
  } else if (type == "rr") {
    stop("type='rr' should be handled by fitted.bjlm_fit wrapper.")
  }

  if (!summary) return(fitted_draws)
  data.frame(
    .observation = seq_len(n),
    fitted_mean  = colMeans(fitted_draws),
    fitted_Q2.5  = apply(fitted_draws, 2, stats::quantile, probs = 0.025),
    fitted_Q97.5 = apply(fitted_draws, 2, stats::quantile, probs = 0.975)
  )
}



#' @importFrom bayesplot pp_check
#' @export
pp_check.bjlm_fit <- function(object, n_draws = 50, ...) {
  outcome_vars <- all.vars(object$outcome_formula)
  y_name <- outcome_vars[1]
  y_obs <- as.double(object$data[[y_name]])

  family <- object$outcome_family
  if (is.null(family)) family <- object$model$outcome$family$family
  if (is.null(family)) family <- "gaussian"

  # Draw replicates on the correct scale for the outcome family. `fitted(type =
  # "link")` returns the linear predictor (identity/logit/log for gaussian/
  # binomial/negative_binomial); replicate from the matching sampling distribution.
  eta_mat   <- fitted(object, summary = FALSE, type = "link")
  draws_mat <- posterior::as_draws_matrix(object$draws)
  idx       <- sample(nrow(eta_mat), min(n_draws, nrow(eta_mat)))
  n         <- length(y_obs)

  y_rep <- switch(family,
    gaussian = {
      sigma_draws <- as.numeric(draws_mat[, "sigma"])
      do.call(rbind, lapply(idx, function(s)
        stats::rnorm(n, mean = eta_mat[s, ], sd = sigma_draws[s])))
    },
    binomial = do.call(rbind, lapply(idx, function(s)
      stats::rbinom(n, size = 1, prob = 1 / (1 + exp(-eta_mat[s, ]))))),
    negative_binomial = {
      r_draws <- as.numeric(draws_mat[, "r"])
      do.call(rbind, lapply(idx, function(s)
        stats::rnbinom(n, size = r_draws[s], mu = exp(eta_mat[s, ]))))
    },
    stop("pp_check() does not support outcome family '", family, "'.", call. = FALSE)
  )
  bayesplot::ppc_dens_overlay(y_obs, y_rep)
}

#' Pointwise log-likelihood matrix
#'
#' @param object A fitted model object.
#' @param ... Additional arguments passed to methods.
#' @export
log_lik <- function(object, ...) UseMethod("log_lik")

#' @importFrom loo loo
#' @export
loo::loo

#' @importFrom loo waic
#' @export
loo::waic

#' @importFrom bayesplot pp_check
#' @export
bayesplot::pp_check



