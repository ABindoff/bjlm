# Hand-off: debug the Rust conditional-transport for NB + latent GP

## STATUS: RESOLVED (2026-07-05) — see POSTMORTEM at the bottom

Root cause was an NB parameterization mismatch, not adaptation or the transport
design. Transport is now the NB+GP default (`BJLM_GP_NO_TRANSPORT=1` = whitened
escape hatch); A/B agreement with whitened |z| <= 0.09 on all params with 3-9x
GP-hyper ESS; SBC in the high-count regime via data-raw/sbc_nb_gp_transport.R.

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

## POSTMORTEM (2026-07-05)

**Root cause: NB parameterization mismatch between kernels.** bjlm's NB convention
is `psi = ln(mean)` everywhere (the coef PG-Gibbs augments at `psi - ln r` and shifts
back; `sample_r_weighted`, `compute_pointwise_log_lik`, and the whitened
`compute_ll_noncentered` all agree). The transport port copied the R prototype's
convention, `eta = ln(mean/r)` (mean = `r*exp(eta)`), so its theta-MH and z-ESS
targeted a DIFFERENT joint than the coef/r kernels. A composition of kernels with no
common invariant law has no stationary distribution: the chain ratcheted up the
`(alpha, sigma_x) -> c*(alpha, sigma_x)` ridge (~2.5 nats/sweep of computed-target
loss funded by z re-fits), reaching alpha ~ 1e4-1e5 on the prototype's own DGP. The
symptom chain: `r` dragged to ~3.2 (truth 10), `b0_gp` collapsed toward 0 (suppressing
the mis-calibrated field term), theta wandering the scale ridge. Fix: evaluate the NB
field log-likelihood at `eta = psi - ln(r)` (one line, `field_ll` in
`sample_gp_transport`); the collapse's NB PG row got the matching `ln r` shift for
coherence (it is dormant: `gp_collapsible` excludes NB).

**How it was found** (the suspect list was wrong, instructively so):
1. Suspect 2 (silent penalty drop on Cholesky failure) was REAL but secondary: a
   proposal whose `transport_pieces` all failed was accepted at acc=1.000 (target =
   prior only beats any honest target), then trapped the chain (alpha 47-153, the
   originally reported symptom). Fixed: a numerically failed subject with data now
   returns -inf for the whole theta (a failure VETOES the move); `Sigma_f` jitter now
   scales with alpha^2 (the subtraction cancels alpha^2-sized terms, so an absolute
   1e-9 goes indefinite at large alpha) with a x100 retry ladder.
2. Suspect 1 (adaptation) was a red herring: fixed prototype steps (0.08/0.10/0.08,
   `BJLM_GP_FIXED_STEP=1`, kept as a debug switch) still diverged.
3. Component-level tracing (`BJLM_GP_DEBUG=1`: prior/sub-marginal/field-ll split,
   re-whitening round-trip check) showed the sub-marginal EXACTLY matched the R
   formulas and the state was consistent, while the chain sat 3700+ nats below its
   own start -- impossible for target-preserving kernels, hence a no-common-target
   bug. The 30k-iteration R prototype run (stable, even with sampled r) exonerated
   the transport design and pinned the port.

**Also added: centered theta interweave (ASIS-style).** After the z-ESS, a second
theta-MH holds f FIXED against `p(theta) N(f; 0, K) N(X_obs; f, sx^2 I)` (outcome
terms cancel). Cheap, exactly invariant, and it seals the inflation ridge mechanically
(with f fixed there is no z re-fit to hide behind), on top of contributing to the
final ESS win.

**The same bug class, pre-existing, in the CHANGE-POINT kernels (bigger deal).**
The first SBC smoke failed hard on omega/b0/delta/r (omega ranks all 0/1: omega
pinned to a prior boundary in EVERY NB fit, both GP paths; r rank 1.000 in 16/16).
Cause: `hmc_step_om_weighted`, `hmc_step_rho_weighted`, `sample_om_laplace_weighted`,
and `sample_re_om_ancillary_weighted` all evaluated the NB log-likelihood (and the
HMC gradients) in the NB-logit convention at psi -- convention B again, predating
the transport work. Invisible in earlier NB testing because the two conventions
COINCIDE at r = 1 (ln r = 0) and earlier NB checks used small r / low counts; at
r ~ 10 the omega kernel sees a model whose implied mean is 10x the data and flees
to the boundary, and the unbent change-point structure is soaked up as fake
overdispersion (r dragged low). All four sites now evaluate at psi - ln r (the
shift is constant in mu, so the gradient/curvature forms are unchanged). After the
fix, 9/10 SBC params pass on the smoke; the r flag that remained was an SBC-script
prior mismatch (`prior_gamma(shape, SCALE)`, not rate -- documented in the script).

AUDIT RULE going forward: bjlm's NB convention is psi = ln(mean), enforced at every
likelihood/gradient site via the NB-logit `psi - ln r`. Legacy `run_chain` /
`run_chain_re` (sampler.rs / sampler_re.rs, non-IPW entry points) still carry an
unshifted PG site each -- audit for INTERNAL consistency before touching (fixing one
site inside an internally-B-consistent sampler would create this same bug).

**One more bug caught by the SBC run itself: PG integer-b hang.**
`sample_pg(b, c)` summed b exact PG(1, c) draws whenever b was an integer with
|c| <= 5. The NB dispersion initialises to exactly r = 1.0, so the FIRST sweep
calls it with integer b = y + 1 -- a prior-tail SBC dataset with counts ~7e9 burned
16 CPU-hours inside one call. Fixed: exact summation only for b <= 50; above that
the moment-matched Gamma (CLT-accurate, and already the path for every non-integer
b, i.e. every post-init sweep). The SBC script also skips degenerate prior-tail
datasets (max y > 1e5): conditioning on a y-measurable event leaves p(theta | y),
and hence SBC rank uniformity, unchanged.

**Certification:** prototype-DGP A/B (transport vs whitened): all params agree to
2-3 decimals, GP-hyper ESS 16-27x whitened. Full model (change-point + RE + sampled
r): |z| <= 0.20 agreement, GP-hyper ESS 22x/11x/5.7x (alpha/rho/sigma_x); omega now
recovers ~4.4 (truth 5) in BOTH paths where it previously pinned at the boundary.

SBC certificate (data-raw/sbc_nb_gp_transport.R): NB outcome + latent GP + one
change-point, high-count regime (b0 ~ N(2.5, 0.5), r ~ gamma(2, scale 5), counts
regularly in the hundreds), no RE, no IPW, transport default path. 128 reps drawn,
14 skipped as degenerate prior-tail sims (max y > 1e5), 0 fit failures ->
114 usable reps, 10-bin chi-square, Bonferroni pass p > 0.0050:

    b0          chisq   5.12  p 0.8235  PASS
    b0_gp       chisq   8.28  p 0.5061  PASS
    b1          chisq   8.28  p 0.5061  PASS
    delta       chisq   3.54  p 0.9388  PASS
    omega       chisq  16.18  p 0.0633  PASS
    rho         chisq  12.32  p 0.1961  PASS
    r           chisq   3.54  p 0.9388  PASS
    gp_alpha    chisq   8.46  p 0.4889  PASS
    gp_rho      chisq  11.96  p 0.2153  PASS
    gp_sigma_x  chisq  10.04  p 0.3477  PASS

Ranks: data-raw/sbc_nb_gp_transport_ranks.rds (local artifact, .rds is gitignored;
regenerate with the script above).
