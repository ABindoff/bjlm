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
- **v1b — FFBS + misclassification.** Union-grid FFBS (latent states),
  Dirichlet `E`, initial `π`; widen chain-runner return to a separate `states`
  matrix (like `log_lik`). Exit: **coverage-mode** SBC on all parameters.
- **v1c — composition.** GP + regime + opt-in smooth change-point together.
  Exit: **coverage-mode** SBC on the three-latent aliasing.

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
