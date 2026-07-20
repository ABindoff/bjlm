# Design: Continuous-Time Regime-Switching (Hidden Markov Multistate) Outcome Component

Status: design locked; implementation phased (v0 shipping first). Author: design
synthesized 2026-07 from a five-agent research + planning pass (codebase map,
Bayesian discrete-HMM inference, continuous-time inference spike, health-econ /
epidemiology domain grounding, prior-art/API survey).

## 1. Context & motivation

bjlm already fits a smooth change-point — a *single, monotone, deterministic*
regime transition. This extension generalizes that to a **discrete latent regime
process** with *recurring, stochastic, covariate/treatment-dependent*
transitions: a hidden-Markov / "dependent-mixture" component on the outcome
model.

**Target fields.** Dementia epidemiology and (via a merging centre) health
economics in Multiple Sclerosis. These are the native home of the model:

- **MS** is a textbook discrete-regime process: RRMS / SPMS / PPMS phenotypes,
  EDSS-threshold states, relapse as a *noisily observed* regime signal, DMTs as
  treatments acting on transition rates. The RRMS→SPMS boundary is diagnosed
  late and with error — a latent state with a noisy observed indicator.
- **Dementia** has the same shape: CDR / GDS / MCI→dementia stages, observed
  with substantial inter-rater noise.
- **Health economics** *is* multistate Markov modelling — the Markov cohort /
  cost-effectiveness model is the field's workhorse (`heemod`, `hesim`), with
  `msm` (Jackson 2011) the continuous-time subject-level statistical counterpart.

**What bjlm adds over the standard toolkit** (`msm`/`nhm` + cohort CEA +
`depmixS4`): one coherent uncertainty chain instead of an estimate-then-plug-in
pipeline; subject-level hierarchical heterogeneity; joint estimation with a
smooth GP trend (separating *which* regime from *how fast within* a regime); use
of the observed noisy indicator as an informative emission (Yen 2018 de-biasing);
and — the differentiator no incumbent offers — **SBC-certified calibration** of
weakly-identified misclassification-HMM parameters.

Single most compelling use case: *a Bayesian, subject-level MS
natural-history-and-treatment model where the latent RRMS/SPMS+EDSS regime is
inferred from noisy staging and relapse signals, DMT effects act on transition
intensities, a GP carries the smooth within-regime disability trend, and the
posterior transition matrix feeds a cost-effectiveness analysis with fully
propagated uncertainty.* The same engine serves dementia natural-history work —
one methods object bridges both centres.

## 2. The model

Subjects `i`, observed at irregular times `t_i1<...<t_iT_i`. Latent
**continuous-time Markov jump process** `s_i(t) ∈ {1..K}`, generator `Q` with
**proportional intensities** (msm convention):

```
q_kl(x) = q_kl^0 · exp(x' β_kl)   (l ≠ k),   q_kk = -Σ_{l≠k} q_kl
P(Δ; x) = exp(Q(x) · Δ)           (matrix exponential; interval transition matrix)
```

Two conditionally-independent emissions of the state:

- **Observed regime indicator** `r_it ~ Categorical(E_{s_it,·})`, misclassification
  (confusion) matrix `E` (K×R, Dirichlet rows). `E = I` ⇒ states known exactly.
- **Outcome** `y_it ~ f(μ_it)`, `μ_it = b0_{s_it} + η_it`, where `η_it` is the
  existing bjlm linear predictor (fixed effects + optional latent GP + random
  intercept + *optional* smooth change-point), and `f` is Gaussian / binomial /
  negative-binomial. `b0_k` is the **state-dependent level** (v1 scope: level
  only; slope/variance switching deferred).

Covariates piecewise-constant between observations; hierarchical over subjects.
The smooth change-point is now **opt-in** — regimes replace it by default; it
remains available for a continuous *within-regime* ramp.

## 3. Inference — the settled continuous-time hybrid

Rejected: (a) fully marginalized forward-algorithm (loses the conjugate level /
misclassification the host sampler wants); (b) full-path Rao–Teh augmentation
(heavy machinery; only buys Gamma-conjugate baseline rates, and β is MH anyway).

**Chosen: observation/union-grid FFBS with `exp(QΔ)` interval kernels.** Per
sweep:

1. **Union-grid FFBS** per subject over {outcome ∪ indicator ∪ treatment-change
   times}, segment kernels `exp(Q(x)δ)`, node emission
   `L(k) = f(y|b0_k+η)·E_{k,r}` (or `1` at pure-transition nodes) → exact joint
   draw of the state at every needed time. **Reuses the GP union-grid machinery
   (`GpData::process`) and the `sample_gp_state` per-subject loop**, swapping
   elliptical-slice for forward-filter/backward-sample.
2. **`b0_k`** — conjugate; state-indicator columns folded into the existing
   coefficient Gibbs (`sample_linear_coefs_weighted`), corner `b0_ref = 0`.
3. **`E`** — Dirichlet-conjugate from (state, indicator) counts.
4. **`{q^0, β}`** — adaptive random-walk MH on
   `Σ log[exp(QΔ)]_{s_j,s_{j+1}} + log π + priors`; blocked per `(k,l)`,
   `log q^0` blocked with its `β`. **MH, not HMC** ⇒ no matrix-exp derivatives.
5. **`π`** — Dirichlet-conjugate (or fixed).

Matrix exponential: **scaling-and-squaring Padé** (Higham 2005) on nalgebra LU —
robust to defective/repeated eigenvalues, no new crate; K is small (2–5).

Only non-conjugate block is (4); it reuses the MH pattern of the existing change-
point / NB-dispersion updates. This gives msm-parity in the generator formulation
(the health-econ audience's native object), Bayesian and joint with the GP.

**Escape hatch:** if baseline-rate (`q^0`) mixing is poor (the one real risk —
`log q^0`/`β` are correlated and move together by MH), full-path Rao–Teh
uniformization (2013) restores Gamma-conjugacy for `q^0`. Ship as an opt-in env
flag (like `BJLM_OM_ASIS`/`BJLM_OM_LAPLACE`), not a redesign.

## 4. Composition & identifiability

Three latent time-structures (discrete regime, smooth change-point, GP) compete
for the same features. The **observed indicator anchors the regime** (breaks the
tie the change-point and GP cannot). Rules:

- The smooth change-point is subsumed where an observed regime switch explains a
  feature; keep it only for a mechanistically distinct within-regime ramp.
- The GP stays as the smooth confounder/nuisance (pinned by its own observed
  index; robust per the coverage-sweep result).
- **Gate every composition with COVERAGE-mode SBC** on the regime parameters
  (`b0_k`, `β`, `E`) against nuisance-prior sweeps (GP lengthscale, change-point
  sharpness). Self-consistent uniformity SBC is *blind* to three-latent aliasing
  (an exact sampler is uniform at every prior even when non-identified) — the
  exact lesson from the climate coverage work.

Label switching is dissolved by the diagonally-dominant `E` (anchors labels to
observed categories) plus the corner constraint; an ordered-`b0` constraint is a
fallback only. SBC ranks *parameters*, not latent states.

## 5. Phased implementation plan (each SBC-gated before the next)

- **v0 — plumbing.** `regimes()` API + `confusion()`/`exact()` constructors +
  `compile()` validation + SBC integration. States **hard-clamped** to the
  observed indicator, `E = exact()`, no transitions. Level-switching realized via
  the existing conjugate coefficient machinery (state-indicator columns in the
  `b0` design). No CTMC math, no FFI/Rust changes required. Certifies the R API,
  the level semantics, the draw-naming, and the SBC path in isolation.
  Exit: self-consistent SBC on `b0_state` recovery.
- **v1a — intensities.** New Rust `ctmc.rs`: generator, Padé `expm`, segment
  likelihood; adaptive RW-MH on `{log q^0, β}`; FFI marshalling of the regime
  block + transition design + allowed-transition mask. States still clamped.
  Exit: **Gillespie-simulated** SBC on `q^0`, `β`.
- **v1b — FFBS + misclassification. [SHIPPED]** Standalone Rust `regime_hmm.rs`:
  per-subject FFBS (log-space forward filter, backward sample) over the latent
  CTMC path, conjugate joint draw of the outcome coefficients (fixed design +
  state-indicator offsets), inverse-gamma `sigma`, Dirichlet `E` and initial `pi`,
  adaptive RW-MH on `{log q^0, beta_q}`. Fit jointly (states latent, NOT clamped),
  so the level and the intensities couple through the sampled path; routed via
  `.regime_hmm_fit()` -> `run_regime_hmm` when `obs_model = confusion()`. Requires
  a change-point-free Gaussian outcome. Recovery verified (levels, trend, sigma,
  base intensities, misclassification recover; the transition covariate is
  correctly weakly identified and converges with N). Exit: **draw==fit
  coverage-mode** SBC on intercept/trend, `b0_state`, `sigma`, `q^0`, `E` (opt-in,
  `BJLM_SBC_CERT=1`).
- **v1c-a — non-Gaussian families. [SHIPPED]** Binomial (logit) and
  negative-binomial (log) latent-regime outcomes in the standalone
  `regime_hmm.rs`, kept isolated (no main-loop surgery). FFBS uses the family
  emission log-density per state; the coefficient/level draw is Polya-Gamma
  augmented (bjlm convention: NB augments the log-odds `psi - ln r`, `b = y + r`,
  `kappa = (y - r)/2`; Binomial `b = 1`, `kappa = y - 1/2`), reducing every family
  to the same weighted normal equations. NB dispersion `r` by RW-MH (shared
  `ln_gamma`); Binomial has no dispersion. `run_regime_hmm` gains `family`,
  `n_trials`, `r_init/r_shape/r_rate`. Verified: recovery for both families
  (levels, trend, `r`, `E`); draw==fit NB SBC (opt-in) Bonferroni-PASS.
- **v1c-b — composition. [SHIPPED]** The latent regime is now a Gibbs step
  INSIDE `run_chain_bjlm`, composing with the latent GP confounder and the
  smoothed change-point. Keystone: the state level `b0_state[path[i]]` is a pure
  additive term like the random intercept `u_b0`, so (a) `means()` adds it and
  every consumer inherits it, (b) `LinearCache::build` folds it into `b0_fixed`
  so the omega/rho HMC residual excludes it, (c) `sample_linear_coefs_weighted`
  subtracts it as an offset (Gaussian `y_tilde`, Binomial/NB PG centres). The
  FFBS runs on the state-excluded mean `mu_full - b0_state[path]` (the GP block's
  cache-and-subtract pattern); the level draw mirrors
  `sample_random_effects_weighted` grouped by the path. New Rust module
  `regime_step.rs`; `RegimeData`/`RegimeState` in `ModelData`/`State` (empty =
  strict no-op); FFI `run_bjlm(regimes=)`. R: `bjlm(regimes=)` + draw naming;
  `fit()` auto-selects the in-loop engine when a GP or change-point is present
  (else the faster isolated engine); `.build_regime_list()` marshals the block.
  Built and verified in gated increments: (0) inert scaffolding, non-regime fits
  byte-identical; (1) regime-only in-loop reproduces the SBC-certified isolated
  v1b to ~3 decimals on every parameter; (2) regime + change-point recover
  SEPARATELY (omega/delta vs b0_state, all rhat <=1.02); (3) regime + GP +
  change-point all three recover (GP hypers alpha/rho/sigma_x + loading to 2-3
  decimals with the regime composed on top). **COVERAGE-MODE SBC CERTIFIED**
  (three-latent aliasing): 40 datasets from a FIXED reality (GP confounder +
  change-point + regime all present), fresh latent fields/paths/indicators/noise
  each rep, fit with the composed model. Every parameter covers at ~nominal rate
  -- cov90 in [0.82, 0.97] (SE ~0.047 at 40 reps -> all within noise of 0.90),
  cov50 within noise of 0.50: GP hypers (alpha 0.95, rho 0.90, sigma_x 0.88),
  GP loading (0.88), delta (0.95), b0_state (0.97/0.88), E (0.95/0.95/0.90),
  sigma (0.82). So the three latents are IDENTIFIABLE and none aliases another.
  Sole caveat: omega cov50 0.38 with maxRhat 1.64 -- a MIXING artifact (the
  change-point location mixes slowly with three latents; cov90 still 0.90),
  fixed by more iterations, not a calibration defect. Follow-up: non-Gaussian
  composition (the emission + PG offset already handle families, but
  GP-in-non-Gaussian has its own known interactions to re-verify).

## 6. API (forward-compatible from v0)

```r
bjlm_model() |>
  outcome(y ~ tau, b0 = ~ 1 + (1 | subject), data = d) |>   # change-point OFF by default
  latent_gp(name = "conf", ...) |>                          # optional smooth confounder
  regimes(
    name = "regime", data = d, n_states = 3,
    states     = c("RRMS", "SPMS_early", "SPMS_late"),
    time_var   = "t", subject = "subject",
    switch     = level ~ 1,          # what switches (level in v1); RHS = within-state formula
    obs_state  = "edss_band",        # observed (noisy) regime column
    obs_model  = confusion(diag = 8, offdiag = 1),   # v1b; exact() = known states (v0)
    obs_true   = NULL,               # column marking exactly-known rows
    transition = ~ dmt + age,        # covariate/treatment-dependent intensities (v1a)
    ref_state  = "RRMS",             # corner / multinomial reference
    init       = ~ 1,                # or "stationary"
    hierarchical = TRUE, re = ~ (1 | subject),
    priors     = regime_priors(...)
  ) |>
  compile() |> fit()
```

v0 activates `name/data/n_states/states/time_var/subject/switch=level~1/obs_state/
obs_model=exact()/ref_state` (level-switching, known states); `transition`,
`confusion()`, `hierarchical` are parsed/validated but warn that they engage in
v1a+.

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Baseline-rate `q^0` mixing (RW-MH, correlated with `β`) | block `log q^0` with `β`; Robbins–Monro adaptation; Rao–Teh escape hatch (opt-in) |
| Matrix-exp degeneracy (general non-symmetric Q) | scaling-and-squaring Padé as primary (no eigendecomposition special-casing) |
| Label switching / identifiability | diagonally-dominant `E` + corner `b0_ref=0`; ordered-`b0` fallback; SBC ranks params not states |
| Draw-naming lockstep (only a `message()` guard) | append all new scalars LAST after GP hypers; `b0_state` names ride `colnames(x_b0)`; hard count-assertion in dev |
| Grid bookkeeping (union of indicator/outcome/treatment-change times) | reuse `GpData::process` verbatim incl. 1e-8 dedup; tag node kind; unit-test index maps |
| SBC convention drift (cf. NB postmortem) | simulate the TRUE CTMC forward (Gillespie), not the `exp(QΔ)` surrogate; identical parameterization/sign/orientation in simulator and likelihood |

## 8. Design decisions (resolved with the maintainer)

1. **Change-point vs regime:** replace by default; smooth change-point opt-in.
2. **Time:** continuous-time (hard requirement); discrete-time not pursued.
3. **State-path output:** per-observation smoothed occupancy probabilities
   (Rao-Blackwellized), plus optional modal path — not raw sampled paths.
   \[SHIPPED\] `run_regime_hmm` returns, per chain, the posterior-mean
   forward-backward smoothed marginals `P(s_i = s | y)`; R averages across chains,
   maps back to the original row order, and exposes them via `state_occupancy(fit)`
   (a `p_<state>` column per state + a `modal_state`). Averaging the smoothed
   marginals (not tallying sampled paths) is the Rao-Blackwellization; recovers the
   true latent state ~0.98 at `E`-diag 0.9 (beats the raw indicator).
4. **`π`:** estimated (Dirichlet-conjugate) by default; fixable.
5. **Observed alphabet R ≠ K:** supported (rectangular K×R confusion) — coarser
   observed categories (EDSS bands, CDR) are the norm.
6. **Rao–Teh:** deferred; architected as a drop-in swap behind an opt-in flag.
7. **Conventions frozen:** `ref_state` = reference for both `b0` and `π` corners;
   store all `E`-row entries (named `E_k_r`), matrix-exp = pure Padé for v1.

## 9. Key references

FFBS/HMM: Scott (2002, JASA); Frühwirth-Schnatter (2006). Non-homogeneous /
input-output: Hughes & Guttorp (1994); Bengio & Frasconi (1995). Continuous-time
multistate: Kalbfleisch & Lawless (1985); Jackson (2011, msm, JSS 38/8); Titman &
Sharples (2010); Titman (2011, nhm). Path sampling: Rao & Teh (2013, JMLR);
Hobolth & Stone (2009, AoAS). Matrix exp: Van Loan (1978); Moler & Van Loan
(2003); Higham (2005). Misclassification HMM (Bayesian, covariate, measurement-
error): Jackson & Sharples (2003); Yen et al. (2018, Stat. Med.). PG multinomial
(discrete-time reference, not used in continuous-time v1): Polson-Scott-Windle
(2013); Linderman et al. (2015); Holsclaw et al. (2017). Nonparametric K:
Fox et al. (2011, sticky HDP-HMM). Calibration: Talts et al. (2018, SBC).
Dependent-mixture software: depmixS4 (Visser & Speekenbrink). Causal/multistate:
Robins g-methods / MSMs; marginal structural illness-death models.
