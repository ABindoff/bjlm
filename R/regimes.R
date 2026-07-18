# =============================================================================
# regimes(): a discrete latent-regime (hidden Markov multistate) component for
# the outcome model. See data-raw/DESIGN_regimes_hmm.md for the full design.
#
# PHASING. The public API is forward-compatible with the full continuous-time
# model (covariate/treatment-dependent transition intensities, a misclassified
# observed indicator, hierarchical subjects). This file implements PHASE v0:
# a regime-dependent LEVEL with the states held fixed at an observed indicator
# (obs_model = exact()). Because the states are known, a state-dependent level
# is exactly a factor in the b0 design, so v0 desugars into the existing
# conjugate machinery (compile() -> .desugar_regimes_v0) with no CTMC math.
# Later phases (v1a intensities, v1b FFBS + misclassification, v1c composition)
# replace the internals; the API does not change.
# =============================================================================

#' Emission model for an observed regime indicator: exact (known states)
#'
#' Declares that the observed regime indicator equals the true latent state with
#' no error, i.e. the states are known. This is the only emission supported in
#' the current phase; the misclassification emission [confusion()] activates in
#' a later phase.
#'
#' @return A `regime_obs_model` object.
#' @seealso [regimes()], [confusion()].
#' @export
exact <- function() {
  structure(list(type = "exact"), class = "regime_obs_model")
}

#' Emission model for an observed regime indicator: misclassification
#'
#' Declares that the observed regime indicator is a noisy emission of the true
#' latent state through a confusion (misclassification) matrix with
#' diagonally-dominant Dirichlet rows. With this emission the states are
#' \strong{latent}: the whole model (outcome level, transition intensities,
#' misclassification, initial state and noise/dispersion) is fit jointly by
#' forward-filter backward-sample (FFBS). Supports gaussian, binomial (logit) and
#' negative-binomial (log) outcomes -- the non-Gaussian coefficient/level draw
#' uses Polya-Gamma augmentation. Requires a change-point-free outcome (e.g.
#' `outcome(y ~ time)`): the regime process replaces the smoothed change-point.
#'
#' @param diag,offdiag Dirichlet concentration on the diagonal (correct
#'   classification) and off-diagonal (misclassification) entries. `diag` should
#'   dominate to anchor state labels. Defaults `8` and `1`.
#' @return A `regime_obs_model` object.
#' @seealso [regimes()], [exact()].
#' @export
confusion <- function(diag = 8, offdiag = 1) {
  stopifnot(diag > 0, offdiag > 0)
  structure(list(type = "confusion", diag = diag, offdiag = offdiag),
            class = "regime_obs_model")
}

#' Priors for a regime block
#'
#' Bundle of priors for the regime component. In the current phase only `level`
#' is meaningful (the state-dependent intercept offsets); `intensity` and
#' `init` are recorded for forward compatibility with the transition-intensity
#' and initial-state models of later phases.
#'
#' @param level Prior for the state-dependent level offsets (a [prior_normal()]).
#' @param intensity Prior for transition log-intensities (later phases).
#' @param init Prior for the initial-state distribution (later phases).
#' @return A `regime_priors` object.
#' @export
regime_priors <- function(level = prior_normal(0, 5),
                          intensity = NULL, init = NULL) {
  structure(list(level = level, intensity = intensity, init = init),
            class = "regime_priors")
}

#' Add a discrete latent-regime (hidden Markov multistate) component
#'
#' Adds a discrete latent regime process to the outcome model: the outcome level
#' switches with a latent state that (in the full model) follows a continuous-time
#' Markov jump process with covariate/treatment-dependent transition intensities
#' and is observed through a possibly-misclassified indicator. This generalizes
#' the smooth change-point (a single monotone transition) to recurring,
#' stochastic, covariate-driven regime switches — the natural description of
#' relapsing-remitting disease (e.g. MS phenotype/EDSS states) and staged
#' progression (e.g. dementia CDR stages).
#'
#' \strong{Emission modes.} With `obs_model = exact()` the states are known: the
#' regime-dependent \emph{level} is a factor in the outcome design and the
#' covariate-dependent transition intensities (`transition`) are fit from the
#' clamped path. With `obs_model = confusion()` the states are \emph{latent} (the
#' observed indicator is misclassified) and the whole model, including the level,
#' the intensities, the misclassification matrix and the initial-state
#' distribution, is fit jointly by forward-filter backward-sample (FFBS); this
#' requires a change-point-free outcome (the regime process replaces the smoothed
#' change-point) and supports gaussian, binomial (logit) and negative-binomial
#' (log) families -- the non-Gaussian coefficient/level draw uses Polya-Gamma
#' augmentation. Composing a smoothed change-point or a latent GP with a latent
#' regime is a later phase. See `data-raw/DESIGN_regimes_hmm.md`.
#'
#' @param model A `bjlm_model` object.
#' @param name Character label for this regime block.
#' @param data A data frame. In the current phase the observed-state column must
#'   also be present in the outcome data.
#' @param n_states Integer number of states K.
#' @param states Optional character vector of state names (length `n_states`),
#'   used to label the observed indicator and the reported level offsets.
#' @param time_var,subject Time and subject-grouping column names (used from the
#'   FFBS phase; recorded now).
#' @param switch A formula naming which outcome parameter(s) switch with the
#'   state and their within-state model. Current phase supports `level ~ 1`.
#' @param obs_state Column name of the observed (in the current phase, exact)
#'   regime indicator.
#' @param obs_model Emission model for the indicator: [exact()] (current phase)
#'   or [confusion()] (later).
#' @param obs_true Optional column marking rows whose state is exactly known
#'   (later phases).
#' @param transition One-sided formula for covariate/treatment-dependent
#'   transition intensities (later phases). `~ 1` is homogeneous.
#' @param ref_state Reference state (name or index) for the level corner
#'   constraint and the transition/initial-state reference.
#' @param init Initial-state model (later phases).
#' @param hierarchical,re Subject-level random effects on the transition/emission
#'   model (later phases).
#' @param priors A [regime_priors()] bundle.
#'
#' @return The modified `bjlm_model` object.
#' @seealso [exact()], [confusion()], [regime_priors()].
#' @export
regimes <- function(model, name, data, n_states, states = NULL,
                    time_var = NULL, subject = NULL,
                    switch = level ~ 1,
                    obs_state, obs_model = exact(), obs_true = NULL,
                    transition = ~ 1, ref_state = 1,
                    init = ~ 1, hierarchical = TRUE, re = ~ (1 | subject),
                    priors = regime_priors()) {
  if (!inherits(model, "bjlm_model"))
    stop("First argument must be a bjlm_model object.")
  if (missing(data) || is.null(data)) stop("`regimes()` requires a `data` argument.")
  if (missing(obs_state)) stop("`regimes()` requires `obs_state` (the observed regime column).")
  if (is.null(time_var) || is.null(subject))
    stop("`regimes()` requires `time_var` and `subject`: they define the ",
         "per-subject transition intervals for the intensity model.")
  if (!inherits(obs_model, "regime_obs_model"))
    stop("`obs_model` must be exact() or confusion().")
  if (!inherits(priors, "regime_priors"))
    stop("`priors` must be a regime_priors() object.")
  n_states <- as.integer(n_states)
  if (length(n_states) != 1L || is.na(n_states) || n_states < 2L)
    stop("`n_states` must be an integer >= 2.")
  if (!is.null(states) && length(states) != n_states)
    stop("`states` must have length n_states.")

  block <- list(
    name = name, data = data, n_states = n_states, states = states,
    time_var = time_var, subject = subject, switch = switch,
    obs_state = obs_state, obs_model = obs_model, obs_true = obs_true,
    transition = transition, ref_state = ref_state, init = init,
    hierarchical = hierarchical, re = re, priors = priors
  )
  model$regimes[[length(model$regimes) + 1L]] <- block
  model
}

# ---------------------------------------------------------------------------
# v0 desugaring: known-state level-switching == a factor in the b0 design.
# Called from compile(). Validates each regime block, coerces the observed-state
# column to a factor with `ref_state` as the reference level (the corner
# constraint), appends it to the b0 fixed formula, and returns the augmented
# formula + data. Everything downstream (design build, conjugate level Gibbs,
# draw naming as b0_<col><state>, prediction, and SBC) then works unchanged.
# ---------------------------------------------------------------------------
.desugar_regimes_v0 <- function(regime_blocks, out_data, b0_formula) {
  state_cols <- character(0)
  for (blk in regime_blocks) {
    # --- v0 capability gates (warn/err on not-yet-active features) ---
    if (!identical(blk$obs_model$type, "exact"))
      stop(sprintf("regimes('%s'): the current phase supports obs_model = exact() ",
                   blk$name %||% ""),
           "(known states) only; misclassification via confusion() lands in a later phase.",
           call. = FALSE)
    sw_lhs <- if (inherits(blk$switch, "formula") && length(blk$switch) == 3L)
      all.vars(blk$switch[[2L]]) else character(0)
    if (!identical(sw_lhs, "level"))
      stop("regimes(): the current phase switches the level only (`switch = level ~ 1`).",
           call. = FALSE)

    sc <- blk$obs_state
    if (!sc %in% names(out_data))
      stop(sprintf("regimes('%s'): observed-state column '%s' not found in the outcome data ",
                   blk$name %||% "", sc),
           "(the current phase requires the observed state in the outcome data frame).",
           call. = FALSE)

    # --- coerce the observed state to a factor with ref_state as reference ---
    raw <- out_data[[sc]]
    lv  <- blk$states %||% sort(unique(as.character(raw)))
    if (length(lv) > blk$n_states)
      stop(sprintf("regimes('%s'): observed column '%s' has %d distinct values but n_states = %d.",
                   blk$name %||% "", sc, length(lv), blk$n_states), call. = FALSE)
    ref <- if (is.numeric(blk$ref_state)) lv[blk$ref_state] else as.character(blk$ref_state)
    if (!ref %in% lv)
      stop(sprintf("regimes('%s'): ref_state '%s' is not among the states.", blk$name %||% "", ref),
           call. = FALSE)
    fac <- factor(as.character(raw), levels = c(ref, setdiff(lv, ref)))
    if (anyNA(fac))
      stop(sprintf("regimes('%s'): observed-state column '%s' has values outside `states`.",
                   blk$name %||% "", sc), call. = FALSE)
    out_data[[sc]] <- fac

    # --- append the state factor to the b0 fixed formula (corner = ref level) ---
    # model.matrix() resolves the term against `data`, so the formula environment
    # is immaterial here.
    b0_formula <- stats::update.formula(b0_formula, stats::reformulate(c(".", sc)))
    state_cols <- c(state_cols, sc)
  }
  list(b0_formula = b0_formula, data = out_data, state_cols = state_cols)
}

# ---------------------------------------------------------------------------
# v1a: fit the continuous-time transition intensities for one regime block,
# given the CLAMPED observed state path. Under clamped states this block is
# conditionally independent of the outcome level model, so it is sampled on its
# own (run_ctmc_mh) with chains/iter/warmup matched to the outcome fit, and the
# q0 / beta draws are merged into the fit's draws array. Builds one interval per
# consecutive observation pair per subject (covariate taken at the interval
# start; off-grid covariate splitting is a later refinement).
# ---------------------------------------------------------------------------
.regime_intensity_fit <- function(blk, data, chains, iter, warmup, seed) {
  sv <- blk$subject; tv <- blk$time_var; sc <- blk$obs_state
  for (nm in c(sv, tv, sc)) {
    if (is.null(nm) || !nm %in% names(data))
      stop(sprintf("regimes('%s'): column '%s' needed for the transition model is not in the outcome data.",
                   blk$name %||% "", nm %||% "<NULL>"), call. = FALSE)
  }
  K <- blk$n_states
  fac <- data[[sc]]                                  # coerced to factor (ref first) in compile()
  st  <- as.integer(fac) - 1L                        # 0-based state; ref = 0
  stnames <- levels(fac)
  subj <- as.factor(data[[sv]]); tm <- as.numeric(data[[tv]])

  # transition design without intercept (~1 -> no covariates)
  X <- stats::model.matrix(blk$transition, data = data)
  X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
  p <- ncol(X); covnames <- colnames(X)

  # all off-diagonal transitions allowed
  allowed <- do.call(rbind, lapply(0:(K - 1L), function(a)
    do.call(rbind, lapply(setdiff(0:(K - 1L), a), function(b) c(a, b)))))

  seg_dt <- numeric(0); seg_iv <- integer(0); ifrom <- integer(0)
  ito <- integer(0); xr <- numeric(0); iv <- 0L
  for (lv in levels(subj)) {
    o <- which(subj == lv); o <- o[order(tm[o])]
    if (length(o) < 2L) next
    for (j in seq_len(length(o) - 1L)) {
      seg_dt <- c(seg_dt, tm[o[j + 1L]] - tm[o[j]]); seg_iv <- c(seg_iv, iv)
      ifrom <- c(ifrom, st[o[j]]); ito <- c(ito, st[o[j + 1L]])
      if (p > 0) xr <- c(xr, X[o[j], ])
      iv <- iv + 1L
    }
  }
  if (iv == 0L) stop("regimes(): no usable transition intervals (need >= 2 observations per subject).",
                     call. = FALSE)

  ip <- blk$priors$intensity %||% list()
  lq0m <- ip$logq0_mean %||% log(0.5); lq0s <- ip$logq0_sd %||% 1.5; bs <- ip$beta_sd %||% 1.0

  res <- run_ctmc_mh(
    n_states = K, x_trans = if (p > 0) as.double(xr) else numeric(0), p_trans = as.integer(p),
    seg_dt = seg_dt, seg_interval = seg_iv, interval_from = ifrom, interval_to = ito,
    allowed_from = as.integer(allowed[, 1]), allowed_to = as.integer(allowed[, 2]),
    prior_logq0_mean = lq0m, prior_logq0_sd = lq0s, prior_beta_mean = 0.0, prior_beta_sd = bs,
    n_iter = as.integer(iter), warmup = as.integer(warmup), chains = as.integer(chains),
    seed = as.integer(seed), init_step = 0.4)

  q0names <- vapply(seq_len(nrow(allowed)), function(i)
    sprintf("q0_%s_%s", stnames[allowed[i, 1] + 1L], stnames[allowed[i, 2] + 1L]), character(1))
  betanames <- character(0)
  if (p > 0) for (i in seq_len(nrow(allowed))) for (cn in covnames)
    betanames <- c(betanames, sprintf("beta_q_%s_%s_%s", stnames[allowed[i, 1] + 1L], stnames[allowed[i, 2] + 1L], cn))
  list(draws = res$draws, varnames = c(q0names, betanames), chains = chains)
}

# Attach the regime transition-intensity draws to a fitted model, merging them
# into fit$draws (valid: independent of the level model under clamped states).
.attach_regime_intensities <- function(fit, cm) {
  blocks <- cm$model$regimes %||% list()
  if (length(blocks) == 0) return(fit)
  chains <- fit$chains; iter <- fit$iter; warmup <- fit$warmup
  seed <- (fit$fit_dots$seed %||% 1L) + 7919L
  reg_names <- character(0)
  for (bi in seq_along(blocks)) {
    ri <- .regime_intensity_fit(blocks[[bi]], cm$model$outcome$data, chains, iter, warmup, seed + bi)
    n_post <- nrow(ri$draws[[1]]); nv <- ncol(ri$draws[[1]])
    arr <- array(NA_real_, dim = c(n_post, chains, nv),
                 dimnames = list(NULL, paste0("chain_", seq_len(chains)), ri$varnames))
    for (ch in seq_len(chains)) arr[, ch, ] <- ri$draws[[ch]]
    fit$draws <- posterior::bind_draws(fit$draws, posterior::as_draws_array(arr), along = "variable")
    reg_names <- c(reg_names, ri$varnames)
  }
  fit$regime_names <- reg_names
  fit$outcome_names <- c(fit$outcome_names, reg_names)   # visible in summary()
  fit
}

# ===========================================================================
# v1b: latent-regime (misclassified indicator) joint FFBS fit.
#
# When the observed indicator is a NOISY emission of the true state
# (obs_model = confusion()), the states are latent and the whole model --
# outcome level, transition intensities, misclassification, initial-state and
# noise -- is fit jointly by a forward-filter/backward-sample Gibbs sampler in
# Rust (run_regime_hmm). This is the FFBS phase; unlike v1a it does NOT clamp
# the path, so the level offsets and the intensities are coupled through the
# sampled states and cannot be desugared into a factor.
# ===========================================================================

# Which regime mode do these blocks require? "v1b" if ANY block emits through a
# misclassification matrix (confusion()); "v0" (known-state, desugarable) if all
# blocks are exact(). Mixing the two in one model is not supported.
.regime_mode <- function(blocks) {
  if (length(blocks) == 0) return(NULL)
  types <- vapply(blocks, function(b) b$obs_model$type %||% "exact", character(1))
  if (all(types == "exact")) return("v0")
  if (all(types == "confusion")) return("v1b")
  stop("regimes(): a model may not mix exact() and confusion() emission blocks.",
       call. = FALSE)
}

# Guard the v1b entry surface: the current FFBS phase fits a single latent-regime
# block with a switching LEVEL and a Gaussian outcome, replacing (not composing
# with) the change-point. Everything else is a later phase (v1c) -- fail early
# with a pointer rather than silently ignoring the request.
.validate_regimes_v1b <- function(model, data) {
  blocks <- model$regimes
  if (length(blocks) > 1L)
    stop("regimes(): the FFBS phase supports a single latent-regime block; ",
         "multiple confusion() blocks land in a later phase.", call. = FALSE)
  blk <- blocks[[1L]]
  fam <- model$outcome$family$family %||% "gaussian"
  if (!fam %in% c("gaussian", "binomial", "negative_binomial"))
    stop(sprintf("regimes(): the latent-regime (confusion) phase supports gaussian, binomial and negative_binomial outcomes; got '%s'.", fam),
         call. = FALSE)
  if (!isTRUE(model$outcome$zero_breakpoint))
    stop("regimes(): a latent regime (confusion()) REPLACES the change-point by ",
         "default, so specify a plain outcome (e.g. outcome(y ~ time)) with no ",
         "b1/deltas/omega/rho. Composing a smoothed change-point with the regime ",
         "process is a later phase (v1c).", call. = FALSE)
  sw_lhs <- if (inherits(blk$switch, "formula") && length(blk$switch) == 3L)
    all.vars(blk$switch[[2L]]) else character(0)
  if (!identical(sw_lhs, "level"))
    stop("regimes(): the current phase switches the level only (`switch = level ~ 1`).",
         call. = FALSE)
  for (nm in c(blk$subject, blk$time_var, blk$obs_state)) {
    if (is.null(nm) || !nm %in% names(data))
      stop(sprintf("regimes('%s'): column '%s' is required for the latent-regime fit but is not in the outcome data.",
                   blk$name %||% "", nm %||% "<NULL>"), call. = FALSE)
  }
  invisible(TRUE)
}

# Fit a v1b latent-regime model jointly by FFBS. Builds the fixed outcome design
# (from the -- change-point-free -- outcome formula), the intercept-free
# transition design, the misclassified indicator and the per-subject observation
# times, ALL sorted by (subject, time) so each subject is a contiguous,
# time-increasing block as run_regime_hmm requires, then wraps the returned draws
# in a `bjlm_regime_fit`.
.regime_hmm_fit <- function(object, priors = NULL, chains = 4L, iter = 5000L,
                            warmup = NULL, seed = NULL, cores = 1L, verbose = TRUE, ...) {
  blk   <- object$model$regimes[[1L]]
  data  <- object$model$outcome$data
  fam   <- object$model$outcome$family
  famname <- fam$family %||% "gaussian"
  fam_code <- switch(famname, gaussian = 0L, binomial = 1L, negative_binomial = 2L,
                     stop("unsupported family for latent regime: ", famname))
  sv <- blk$subject; tv <- blk$time_var; sc <- blk$obs_state
  K  <- blk$n_states

  chains <- as.integer(chains); iter <- as.integer(iter)
  if (is.null(warmup)) warmup <- iter %/% 2L
  warmup <- as.integer(warmup)
  if (warmup >= iter) stop("`warmup` must be less than `iter`.", call. = FALSE)
  if (is.null(seed)) seed <- 1L
  seed <- as.integer(seed)

  # ---- (subject, time) ordering: contiguous, time-increasing per subject ----
  subj_raw <- data[[sv]]; tm <- as.numeric(data[[tv]])
  subj_f   <- factor(subj_raw, levels = unique(as.character(subj_raw)))
  ord      <- order(as.integer(subj_f), tm)
  data <- data[ord, , drop = FALSE]
  subj_f <- subj_f[ord]; tm <- tm[ord]

  # ---- outcome response + fixed design (change-point-free formula) ----
  resp <- all.vars(object$model$outcome$formula)[1L]
  y <- as.numeric(data[[resp]])
  mm_terms <- stats::delete.response(stats::terms(object$model$outcome$formula))
  x_fixed <- stats::model.matrix(mm_terms, data = data)
  fixed_names <- colnames(x_fixed)
  p_fixed <- ncol(x_fixed)

  # ---- observed indicator -> 0-based state codes (ref first), -1 missing ----
  raw_s <- data[[sc]]
  lv <- blk$states %||% sort(unique(as.character(raw_s[!is.na(raw_s)])))
  if (length(lv) > K)
    stop(sprintf("regimes('%s'): observed indicator '%s' has %d distinct values but n_states = %d.",
                 blk$name %||% "", sc, length(lv), K), call. = FALSE)
  ref <- if (is.numeric(blk$ref_state)) lv[blk$ref_state] else as.character(blk$ref_state)
  if (!ref %in% lv)
    stop(sprintf("regimes('%s'): ref_state '%s' is not among the states.", blk$name %||% "", ref), call. = FALSE)
  stnames <- c(ref, setdiff(lv, ref))
  if (length(stnames) < K) stnames <- c(stnames, paste0("S", (length(stnames) + 1L):K))
  fac <- factor(as.character(raw_s), levels = stnames)
  if (any(is.na(fac) & !is.na(raw_s)))
    stop(sprintf("regimes('%s'): indicator '%s' has values outside `states`.", blk$name %||% "", sc), call. = FALSE)
  obs_state <- as.integer(fac) - 1L
  obs_state[is.na(obs_state)] <- -1L                 # missing indicator -> latent-only

  # ---- transition design (intercept-free) ----
  Xt <- stats::model.matrix(blk$transition, data = data)
  Xt <- Xt[, colnames(Xt) != "(Intercept)", drop = FALSE]
  p_trans <- ncol(Xt); covnames <- colnames(Xt)

  # ---- all off-diagonal transitions allowed ----
  allowed <- do.call(rbind, lapply(0:(K - 1L), function(a)
    do.call(rbind, lapply(setdiff(0:(K - 1L), a), function(b) c(a, b)))))
  na_t <- nrow(allowed)

  # ---- family-specific response handling ----
  n_trials <- rep(1, length(y))                        # binomial: Bernoulli (single trial)
  if (fam_code == 1L && !all(y %in% c(0, 1)))
    stop("regimes(): binomial latent-regime outcome must be 0/1 (Bernoulli).", call. = FALSE)
  if (fam_code == 2L && (any(y < 0) || any(abs(y - round(y)) > 1e-8)))
    stop("regimes(): negative-binomial latent-regime outcome must be non-negative counts.", call. = FALSE)

  # ---- priors (with forward-compatible bundle overrides) ----
  lp <- blk$priors$level %||% prior_normal(0, 5)
  prior_b0_sd <- lp$sd %||% 5
  ip <- blk$priors$intensity %||% list()
  lq0m <- ip$logq0_mean %||% log(0.5); lq0s <- ip$logq0_sd %||% 1.5; bqs <- ip$beta_sd %||% 1.0
  ediag <- blk$obs_model$diag %||% 8; eoff <- blk$obs_model$offdiag %||% 1
  # fixed-coefficient, sigma (Gaussian) and r (NB) priors; weakly-informative defaults
  op <- priors$outcome %||% list()
  prior_beta_sd <- (op$b0 %||% list())$sd %||% 10
  sig_pr <- op$sigma %||% NULL
  sigma_shape <- if (!is.null(sig_pr) && identical(sig_pr$family, "invgamma")) sig_pr$shape else 2
  sigma_scale <- if (!is.null(sig_pr) && identical(sig_pr$family, "invgamma")) sig_pr$scale else 1
  r_pr <- op$r %||% NULL                               # prior_gamma stores SCALE; rate = 1/scale
  r_shape <- if (!is.null(r_pr) && identical(r_pr$family, "gamma")) r_pr$shape else 2
  r_scale <- if (!is.null(r_pr) && identical(r_pr$family, "gamma")) r_pr$scale else 5
  r_rate  <- 1 / r_scale
  r_init  <- max(1, r_shape * r_scale)                 # prior mean

  if (isTRUE(verbose))
    message(sprintf("bjlm: fitting latent-regime (FFBS, %s) model -- %d states, %d subjects, %d obs.",
                    famname, K, nlevels(subj_f), length(y)))

  res <- run_regime_hmm(
    n_states = K, n_cat = K, family = fam_code, y = y, n_trials = as.double(n_trials),
    x_fixed = as.double(x_fixed), p_fixed = as.integer(p_fixed),
    x_trans = if (p_trans > 0) as.double(Xt) else numeric(0), p_trans = as.integer(p_trans),
    obs_state = as.integer(obs_state), obs_subj = as.integer(as.integer(subj_f) - 1L),
    obs_time = as.double(tm),
    allowed_from = as.integer(allowed[, 1]), allowed_to = as.integer(allowed[, 2]),
    prior_beta_sd = prior_beta_sd, prior_b0_sd = prior_b0_sd,
    sigma_shape = sigma_shape, sigma_scale = sigma_scale,
    r_init = r_init, r_shape = r_shape, r_rate = r_rate,
    e_diag = ediag, e_offdiag = eoff,
    prior_logq0_mean = lq0m, prior_logq0_sd = lq0s, prior_beta_q_sd = bqs,
    n_iter = iter, warmup = warmup, chains = chains, seed = seed, init_step = 0.4)

  # ---- draw column names (matches run_one_chain's storage order) ----
  q0names <- vapply(seq_len(na_t), function(i)
    sprintf("q0_%s_%s", stnames[allowed[i, 1] + 1L], stnames[allowed[i, 2] + 1L]), character(1))
  bqnames <- character(0)
  if (p_trans > 0) for (i in seq_len(na_t)) for (cn in covnames)
    bqnames <- c(bqnames, sprintf("beta_q_%s_%s_%s", stnames[allowed[i, 1] + 1L], stnames[allowed[i, 2] + 1L], cn))
  enames <- character(0)
  for (a in seq_len(K)) for (b in seq_len(K))
    enames <- c(enames, sprintf("E_%s_%s", stnames[a], stnames[b]))
  pinames <- sprintf("pi_%s", stnames)
  b0names <- sprintf("b0_state_%s", stnames[-1L])         # ref is the corner (0)
  # dispersion column: sigma (Gaussian), r (NB); binomial has none -> a placeholder
  # slot we drop after decoding (Rust always emits one dispersion column).
  dispname <- switch(famname, gaussian = "sigma", negative_binomial = "r", "._binom_disp")
  varnames <- c(paste0("b_", fixed_names), b0names, dispname, q0names, bqnames, enames, pinames)

  n_post <- nrow(res$draws[[1L]]); nv <- ncol(res$draws[[1L]])
  stopifnot(nv == length(varnames))
  arr <- array(NA_real_, dim = c(n_post, chains, nv),
               dimnames = list(NULL, paste0("chain_", seq_len(chains)), varnames))
  for (ch in seq_len(chains)) arr[, ch, ] <- res$draws[[ch]]
  if (identical(dispname, "._binom_disp")) {           # drop the meaningless placeholder
    keep <- setdiff(varnames, "._binom_disp")
    arr <- arr[, , keep, drop = FALSE]; varnames <- keep
  }

  structure(list(
    draws = posterior::as_draws_array(arr),
    chains = chains, iter = iter, warmup = warmup,
    n_states = K, state_names = stnames,
    fixed_names = paste0("b_", fixed_names),
    level_names = b0names, intensity_names = c(q0names, bqnames),
    emission_names = enames, init_names = pinames,
    outcome_names = varnames, regime_block = blk,
    model = object$model, compiled_model = object,
    family = fam, subject_var = sv, time_var = tv,
    fit_dots = list(seed = seed)
  ), class = "bjlm_regime_fit")
}

#' @export
print.bjlm_regime_fit <- function(x, ...) {
  cat("bjlm latent-regime fit (FFBS)\n")
  cat(sprintf("  states: %d (%s)\n", x$n_states, paste(x$state_names, collapse = ", ")))
  cat(sprintf("  chains: %d   iter: %d   warmup: %d\n", x$chains, x$iter, x$warmup))
  cat(sprintf("  parameters: %d\n", length(x$outcome_names)))
  cat("  use summary() for posterior quantities.\n")
  invisible(x)
}

#' @export
summary.bjlm_regime_fit <- function(object, ...) {
  posterior::summarise_draws(object$draws)
}
