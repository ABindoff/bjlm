# Hand-off: debug the Rust conditional-transport for NB + latent GP

## The one job
Make `sample_gp_transport` in `src/rust/src/sampler_bjlm.rs` match the validated R
prototype so NB + latent-GP mixes without runaway, then SBC-certify and ungate.
This is the user's TOP priority (their first real test is CANTAB count data with a
latent GP confounder).

## Current state (all committed, master, commit 72fb490)
- Approach is SOLVED and validated in R. The Rust port COMPILES but DIVERGES in the
  full model: one chain runs `alpha -> 47, rhat 2.0` while the identical-math R
  prototype is stable.
- It is therefore OPT-IN behind `BJLM_GP_TRANSPORT=1`. NB DEFAULTS to the whitened
  path, which is CORRECT and verified (recovers alpha 1.03 / sigma_x 0.41, no
  runaway; only downside is slow GP-hyper ESS ~5). Gaussian/Binomial collapse
  untouched. So nothing is broken today — this is a mixing upgrade, not a fix for a
  regression.

## The golden reference (trust this, diff against it)
`data-raw/prototype_transport_nb_gp.R` (a.k.a. prototype_transport.R in scratchpad).
Recovers alpha 0.96 / rho 3.08 / sigma_x 0.40 / b0_gp 0.50 (truth 1/3/0.4/0.5) in the
HIGH-COUNT regime where the collapse blows up. WITHOUT the sub-marginal it goes to
alpha 1.73 — i.e. the `N(X_obs; 0, alpha^2 R + sigma_x^2 I)` sub-marginal in the
theta-acceptance is REQUIRED and its presence is the discriminating test.

## Target math (from DESIGN_nb_gp_collapse.md)
    f = mu_f(theta) + L_f(theta) z,  z ~ N(0, I)
    mu_f = K (K + sigma_x^2 I)^{-1} X_obs,   Sigma_f = K - K(K+sigma_x^2 I)^{-1}K,
    L_f = chol(Sigma_f + 1e-9 I),   K = alpha^2 R(rho)
theta-MH target per subject = sub_marginal(X_obs) + NB_ll(f) + log_prior; z by ESS.

## Suspect list (in priority order)
1. **theta-MH adaptation reuse.** `sample_gp_transport` reuses the COLLAPSE's
   `adapts[gp_idx].joint_da.epsilon` / `inv_mass`. Those were tuned for a different
   target; a too-large step + the runaway basin = divergence. Try FIXED RW steps
   matching the prototype (0.08 log-alpha, 0.10 log-rho, 0.08 log-sigma_x) FIRST to
   isolate whether adaptation is the culprit.
2. **transport_pieces returning None at large alpha.** Cholesky PD-failure ->
   subject SKIPPED -> its sub-marginal penalty silently dropped -> the very penalty
   that pins alpha disappears exactly when alpha is drifting up = positive feedback.
   Add jitter / log-and-count skips; a skipped subject must NOT drop the penalty.
3. **Interaction with the other samplers** (change-point, b0_gp, r). The prototype
   has none of these. ISOLATE on the prototype's simple no-change-point,
   fixed-r DGP first; only reintroduce once transport is stable alone.

## Method
1. Add an alpha-trace debug print in `sample_gp_transport`.
2. Build a Rust test on the prototype's exact DGP (no change-point, r fixed). Get it
   matching the R trace step-for-step.
3. Fix, then reintroduce full-model features one at a time.
4. SBC (NB + latent GP, no IPW) in the HIGH-COUNT regime specifically — Fable's
   warning is that a wrong fix looks calibrated on low counts.
5. If calibrated: flip `gp_use_transport` default-on (gp_collapsible already excludes
   NB), remove the env gate, drop the whitened NB fallback to a `BJLM_GP_NO_TRANSPORT`
   escape hatch.

## Build/run notes
- R IS available via PowerShell (not Bash). Run script FILES, not inline `-e`.
- Build/test with `devtools::load_all()`, NOT `cargo build` (the "document" bin fails
  to link R.lib; the LIB itself compiles clean).
- Commit messages via `git commit -F <file>` (PowerShell here-strings mangle quotes;
  Out-File adds a BOM — use the Write tool for the message file).
- Reference docs: `data-raw/DESIGN_nb_gp_collapse.md` (full diagnosis + Fable critique),
  memory `project_bjlm.md`.
