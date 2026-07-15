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
#' diagonally-dominant Dirichlet rows. \strong{Not yet active} — recorded by
#' [regimes()] but engaged from the FFBS phase onwards; the current phase
#' requires [exact()].
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
#' \strong{Current phase (v0).} Fits a regime-dependent \emph{level} with the
#' states held fixed at the observed indicator (`obs_model = exact()`).
#' Covariate-dependent transitions (`transition`), the misclassification emission
#' ([confusion()]), and hierarchical subject effects are parsed and validated but
#' warn that they engage in a later phase. See
#' `data-raw/DESIGN_regimes_hmm.md`.
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
    trans_rhs <- if (inherits(blk$transition, "formula")) all.vars(blk$transition) else character(0)
    if (length(trans_rhs) > 0)
      warning(sprintf("regimes('%s'): transition intensities are estimated from a later phase; ",
                      blk$name %||% ""),
              "the `transition` formula is recorded but not yet used (states held fixed).",
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
