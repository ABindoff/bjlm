# Internal utilities for parsing formulas and building design matrices

# ---------------------------------------------------------------------------
# Parse random-effect terms from a formula, returning:
#   $fixed   : formula with RE terms removed (for fixed-effect part)
#   $re_group: character name of the grouping variable (or NULL)
# ---------------------------------------------------------------------------
.parse_re <- function(formula) {
  tt  <- terms(formula, keep.order = TRUE)
  trm <- attr(tt, "term.labels")
  re_terms <- grep("^1\\s*\\|", trm, value = TRUE)

  if (length(re_terms) == 0L) {
    return(list(fixed = formula, re_group = NULL))
  }
  if (length(re_terms) > 1L) {
    stop("smoothbp supports at most one random-intercept term per parameter.")
  }

  group_name <- trimws(sub("^1\\s*\\|\\s*", "", re_terms[1]))

  # Rebuild fixed formula without the RE term
  fixed_trm <- setdiff(trm, re_terms)
  has_int   <- attr(tt, "intercept") == 1L
  fixed_rhs <- if (length(fixed_trm) == 0L) {
    if (has_int) "1" else "0"
  } else {
    paste(c(if (!has_int) "0", fixed_trm), collapse = " + ")
  }
  fixed_fml <- stats::as.formula(paste("~", fixed_rhs))
  environment(fixed_fml) <- environment(formula)

  list(fixed = fixed_fml, re_group = group_name)
}

# ---------------------------------------------------------------------------
# Build a single design matrix from a formula, returning it with a
# 're_mask' attribute (integer vector: 1 = random effect, 0 = fixed).
#
# For  ~ (1 | group):
#   - Column 1: intercept (grand mean, fixed, mask=0)
#   - Columns 2..K+1: one dummy per level of 'group' using identity coding
#     (all K levels present, mask=1).  Shrinking these to 0 = shrinking
#     toward the grand mean.
#
# For plain ~ x (no RE terms): standard model.matrix, all mask=0.
# ---------------------------------------------------------------------------
.build_mm <- function(formula, data) {
  if (inherits(formula, "smoothbp_fixed")) {
    X <- matrix(as.numeric(formula), nrow = nrow(data), ncol = 1)
    colnames(X) <- "(Intercept)"
    attr(X, "re_mask") <- 0L
    # If it's a vector, we fix the coefficient at 1.0. 
    # If it's a scalar, we fix it at the scalar value and use a column of 1s.
    # Actually, it's simpler to always treat it as a column in the design matrix
    # and fix the coefficient at 1.0 if it's a vector, OR keep it as a column of 1s
    # and fix the coefficient at the scalar value.
    if (length(formula) > 1) {
      attr(X, "fixed_value") <- 1.0
    } else {
      X[] <- 1.0
      attr(X, "fixed_value") <- as.numeric(formula)
    }
    return(X)
  }
  parsed <- .parse_re(formula)

  if (is.null(parsed$re_group)) {
    # No random effects: standard design matrix
    X <- stats::model.matrix(parsed$fixed, data = data)
    attr(X, "re_mask") <- integer(ncol(X))          # all fixed
    return(X)
  }

  # ---- (1 | group) formula -----------------------------------------------
  # Fixed part (usually just an intercept)
  X_fixed <- stats::model.matrix(parsed$fixed, data = data)

  # Random part: one dummy column per level of group (identity coding)
  grp       <- factor(data[[parsed$re_group]])
  lvls      <- levels(grp)
  K         <- length(lvls)
  X_re      <- matrix(0L, nrow = nrow(data), ncol = K)
  colnames(X_re) <- paste0("re_", parsed$re_group, "_", lvls)
  for (j in seq_len(K)) X_re[grp == lvls[j], j] <- 1L

  X <- cbind(X_fixed, X_re)
  attr(X, "re_mask") <- c(integer(ncol(X_fixed)),   # fixed columns: 0
                           rep(1L, K))               # RE columns:    1
  X
}

# ---------------------------------------------------------------------------
# Build design matrices for all parameters (including multiple segments).
# ---------------------------------------------------------------------------
.build_design_matrices <- function(
    b0_fml, b1_fml, deltas_fml, omega_fml, rho_fml,
    data
) {
  mk_list_mm <- function(fml_list, dat) {
    if (!is.list(fml_list)) fml_list <- list(fml_list)
    lapply(fml_list, function(f) .build_mm(f, dat))
  }

  # b0: may carry a (1|group) RE for the random intercept.
  # We do NOT use .build_mm for b0 because the Rust sampler handles b0's random 
  # intercept natively via the group_b0 mechanism and sigma_u, rather than as 
  # identity-coded dummy columns in the design matrix.
  b0_parsed <- .parse_re(b0_fml)
  X_b0      <- stats::model.matrix(b0_parsed$fixed, data = data)
  attr(X_b0, "re_mask") <- integer(ncol(X_b0))
  X_b1     <- .build_mm(b1_fml,    data)
  X_deltas <- mk_list_mm(deltas_fml, data)
  X_om     <- mk_list_mm(omega_fml,  data)
  X_rho    <- mk_list_mm(rho_fml,   data)

  n_bp <- length(X_deltas)
  if (length(X_om) != n_bp || length(X_rho) != n_bp) {
    stop("Number of formulas for deltas, omega, and rho must match.")
  }

  # b0 classic RE grouping (for random intercepts via group_b0 mechanism)
  group_b0        <- integer(nrow(data)) - 1L
  n_groups_b0     <- 0L
  group_levels_b0 <- character(0)

  if (!is.null(b0_parsed$re_group)) {
    gfactor         <- factor(data[[b0_parsed$re_group]])
    group_levels_b0 <- levels(gfactor)
    n_groups_b0     <- nlevels(gfactor)
    group_b0        <- as.integer(gfactor) - 1L
  }

  # Capture training model.frames so that in-formula transforms (scale, log,
  # poly, ns, bs, etc.) can be replayed correctly on new data at prediction
  # time.  The terms object stored on a model.frame carries the attributes
  # (center, scale coefficients, knots, ...) from stateful transforms.
  train_mfs <- .capture_train_model_frames(
    b0_fml, b1_fml, deltas_fml, omega_fml, rho_fml, data
  )

  list(
    X_b0     = X_b0,
    X_b1     = X_b1,
    X_deltas = X_deltas,
    X_om     = X_om,
    X_rho    = X_rho,
    group_b0        = group_b0,
    n_groups_b0     = n_groups_b0,
    group_levels_b0 = group_levels_b0,
    col_names_b0     = colnames(X_b0),
    col_names_b1     = colnames(X_b1),
    col_names_deltas = lapply(X_deltas, colnames),
    col_names_om     = lapply(X_om,     colnames),
    col_names_rho    = lapply(X_rho,    colnames),
    train_model_frames = train_mfs
  )
}

# ---------------------------------------------------------------------------
# Capture training model.frames for prediction-time transform replay.
#
# When a formula contains stateful transforms like scale(), poly(), ns(),
# bs(), etc., model.frame() stores the transform attributes (center, scale,
# knots, degree, ...) on the resulting object's terms.  At prediction time,
# passing these terms to model.frame(tt, newdata, xlev) causes R to replay
# the transform with the original (training-time) attributes rather than
# re-computing them from scratch on newdata.
#
# Stateless transforms (log, sqrt, I, exp, ...) are unaffected — they are
# simply re-evaluated, which is the correct behaviour.
# ---------------------------------------------------------------------------
.capture_train_model_frames <- function(b0_fml, b1_fml, deltas_fml, omega_fml, rho_fml, data) {
  safe_mf <- function(fml) {
    if (inherits(fml, "smoothbp_fixed")) return(NULL)
    fixed <- .parse_re(fml)$fixed
    tryCatch(
      stats::model.frame(fixed, data = data, na.action = stats::na.pass),
      error = function(e) NULL
    )
  }
  safe_mf_list <- function(fml_list) {
    if (!is.list(fml_list)) fml_list <- list(fml_list)
    lapply(fml_list, safe_mf)
  }

  list(
    b0     = safe_mf(b0_fml),
    b1     = safe_mf(b1_fml),
    deltas = safe_mf_list(deltas_fml),
    omega  = safe_mf_list(omega_fml),
    rho    = safe_mf_list(rho_fml)
  )
}

# ---------------------------------------------------------------------------
# Extract the re_mask for a list of design matrices (e.g. X_om).
# Returns a list of integer vectors, one per breakpoint.
# ---------------------------------------------------------------------------
.get_re_masks <- function(X_list) {
  lapply(X_list, function(X) {
    m <- attr(X, "re_mask")
    if (is.null(m)) integer(ncol(X)) else m
  })
}

# ---------------------------------------------------------------------------
# Check whether ANY design matrix in a list has random effects
# ---------------------------------------------------------------------------
.has_re <- function(X_list) {
  any(sapply(.get_re_masks(X_list), function(m) any(m == 1L)))
}

# ---------------------------------------------------------------------------
# Build concatenated prior vectors
# ---------------------------------------------------------------------------
.build_prior_vectors <- function(priors, dm) {
  expand_list <- function(spec_list, nms_list) {
    if (!is.list(spec_list) || inherits(spec_list, "smoothbp_prior")) {
        spec_list <- rep(list(spec_list), length(nms_list))
    }
    lapply(seq_along(nms_list), function(i) .expand_prior(spec_list[[i]], nms_list[[i]]))
  }

  p_b0 <- .expand_prior(priors$b0, dm$col_names_b0)
  p_b1 <- .expand_prior(priors$b1, dm$col_names_b1)

  p_deltas <- expand_list(priors$deltas, dm$col_names_deltas)
  
  # Handle fixed omega/rho by overriding priors if needed
  expand_segment_priors <- function(spec_list, dm_list, names_list) {
    if (!is.list(spec_list) || inherits(spec_list, "smoothbp_prior")) {
      spec_list <- rep(list(spec_list), length(names_list))
    }
    lapply(seq_along(names_list), function(i) {
      if (!is.null(attr(dm_list[[i]], "fixed_value"))) {
        # This segment is fixed!
        val <- attr(dm_list[[i]], "fixed_value")
        .expand_prior(prior_normal(mean = val, sd = 0), names_list[[i]])
      } else {
        .expand_prior(spec_list[[i]], names_list[[i]])
      }
    })
  }

  p_om     <- expand_segment_priors(priors$omega,  dm$X_om,  dm$col_names_om)
  p_rho    <- expand_segment_priors(priors$rho,    dm$X_rho, dm$col_names_rho)

  list(
    b0 = p_b0,
    b1 = p_b1,
    deltas = p_deltas,
    om = p_om,
    rho = p_rho
  )
}

# ---------------------------------------------------------------------------
# Build the per-coefficient outcome prior vectors AND the spike-and-slab masks,
# applying the slab-SD override for spike-eligible b1/delta columns.
#
# This is the single source of truth for the outcome priors the sampler
# actually uses. It is called by bjlm() at fit time and by .sbc_draw_prior()
# during simulation-based calibration, so that "draw == fit" holds by shared
# construction rather than by two copies of the same logic drifting apart.
#
# Inputs are the already-built design matrices (so column names and ordering
# match exactly) plus the names of any latent-GP columns, which are modelled
# separately and are NOT spike-eligible.
#
# Returns:
#   $pv               : list(b0, b1, deltas, om, rho) of per-column data.frames
#                       (name/mean/sd/lb/ub), with slab SDs already applied.
#   $b1_spike_mask    : integer vector (1 = spike-eligible) or NULL if no spike.
#   $delta_spike_mask : list of integer vectors per breakpoint, or NULL.
# ---------------------------------------------------------------------------
.build_outcome_prior_vectors <- function(outcome_priors, spike,
                                         x_b0, x_b1, x_deltas_list,
                                         x_om_list, x_rho_list,
                                         gp_names = character(0)) {
  n_bp <- length(x_deltas_list)
  dm_meta <- list(
    col_names_b0     = colnames(x_b0),
    col_names_b1     = colnames(x_b1),
    col_names_deltas = if (n_bp > 0) lapply(x_deltas_list, colnames) else list(),
    col_names_om     = if (n_bp > 0) lapply(x_om_list, colnames) else list(),
    col_names_rho    = if (n_bp > 0) lapply(x_rho_list, colnames) else list(),
    X_om             = if (n_bp > 0) x_om_list else list(),
    X_rho            = if (n_bp > 0) x_rho_list else list()
  )
  pv <- .build_prior_vectors(outcome_priors, dm_meta)

  use_spike <- !is.null(spike) && inherits(spike, "smoothbp_spike_slab")
  b1_spike_mask <- NULL
  delta_spike_mask <- NULL
  if (use_spike) {
    # GP columns are modelled separately and are NOT spike-eligible.
    b1_spike_mask    <- as.integer(!colnames(x_b1) %in% gp_names)
    delta_spike_mask <- lapply(x_deltas_list, function(dm) rep(1L, ncol(dm)))

    slab_sd <- spike$slab$sd
    for (j in seq_len(ncol(x_b1))) {
      if (b1_spike_mask[j] == 1L) pv$b1$sd[j] <- slab_sd
    }
    for (k in seq_len(n_bp)) {
      for (j in seq_len(ncol(x_deltas_list[[k]]))) {
        if (delta_spike_mask[[k]][j] == 1L) pv$deltas[[k]]$sd[j] <- slab_sd
      }
    }
  }

  list(pv = pv, b1_spike_mask = b1_spike_mask, delta_spike_mask = delta_spike_mask)
}

# ---------------------------------------------------------------------------
# Build the full parameter name vector
# ---------------------------------------------------------------------------
.param_names <- function(dm, pv, learn_pi = FALSE) {
  names <- c(
    paste0("b0_", pv$b0$name),
    if (dm$n_groups_b0 > 0) paste0("u[", dm$group_levels_b0, "]") else character(0),
    paste0("b1_", pv$b1$name)
  )

  for (i in seq_along(pv$deltas)) {
    names <- c(names, paste0("delta", i, "_", pv$deltas[[i]]$name))
  }
  for (i in seq_along(pv$om)) {
    names <- c(names, paste0("omega", i, "_", pv$om[[i]]$name))
  }
  for (i in seq_along(pv$rho)) {
    names <- c(names, paste0("rho", i, "_", pv$rho[[i]]$name))
  }

  names <- c(names, "sigma", "sigma_u")

  names
}
