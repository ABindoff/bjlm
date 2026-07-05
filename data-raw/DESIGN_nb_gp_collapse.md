# Design: a stable, exact collapse for latent-GP + non-Gaussian (NB) outcome

## Problem
The Rao-Blackwellised GP-hyperparameter collapse marginalises the latent field `f`
exactly. Extended to non-Gaussian outcomes via Polya-Gamma (PG), it works for Binomial
(bounded PG weights) but RUNS AWAY for Negative Binomial: on moderate counts the
hyperparameters diverge to an extreme (either `rho -> short, sigma_x -> 0`, or with a
lengthscale floor `rho -> 1e4, alpha -> 90, sigma_x -> 16`). The whitened fallback is
correct but the GP hyperparameters mix at ESS ~5.

## Root cause (diagnosed)
The GP enters the OUTCOME as the product `b0_gp * f`, but `X_obs` measures `f` directly.
The collapse marginalises `f` while CONDITIONING on `b0_gp`. When the outcome carries a
lot of information (NB counts -> large PG weights) and `b0_gp` is momentarily off, the
marginal for `alpha` is pulled by the outcome to compensate, and `X_obs` cannot hold it
-> `alpha` runs away. Binomial is safe only because its PG weights are bounded (< 0.25),
so `X_obs` always dominates `f`. The rho-floor experiment merely flipped the runaway to
the other extreme -> confirms a two-sided conditioning pathology, NOT a sub-resolution
over-fit and NOT non-identifiability (X_obs identifies alpha/rho/sigma_x; the outcome
identifies b0_gp; all parameters are identified).

## Fix (primary): jointly collapse the confounded pair (theta, b0_gp)
The fibre `f` and its outcome loading `b0_gp` are confounded (only `b0_gp*f` is seen by
the outcome). Handle them TOGETHER: marginalise `f` and MH-update the GP hyperparameters
AND `b0_gp` jointly from the exact marginal

    v ~ N(0,  A(b0_gp) K(alpha,rho) A(b0_gp)^T  +  P^{-1}),

where the outcome pseudo-observation rows carry coefficient `c = b0_gp` (assuming the GP
is on the intercept, `p_b1_idx < 0`; fall back otherwise). With `b0_gp` free, `alpha`
stays pinned by `X_obs` (no compensation is required), so the runaway cannot occur.
Exact: it is the marginal of the PG-augmented Gaussian model over `(theta, b0_gp)`.

Implementation delta over the current collapse:
- pass `&Priors` in; use the b0 prior (normal) for `b0_gp`'s log-prior.
- mark outcome pseudo-obs rows; in `marg_ll`, scale their coefficient by the proposed
  `b0_gp` (X_obs rows c=1 and propensity rows c=beta_prop are `b0_gp`-independent).
- extend the joint proposal with a normal RW on `b0_gp`; on accept, write it back to
  `state.beta_b0[p_b0_idx]`; redraw `f` with the accepted `b0_gp`.
- keep the linear-coefficient sampler's `b0_gp` update too (redundant but valid).

## Alternative (if the primary is insufficient): fibr door-1 on the X_obs-conditional
Reparameterise the fibre against its CLEAN (Gaussian) sub-likelihood rather than the
prior: `f = mu_f(theta, X_obs) + L_f(theta) z`, the GP-regression posterior given
`X_obs`. HMC over `(z, theta, b0_gp)` with the IFT gradients of `mu_f, L_f`. This absorbs
`X_obs` into the reparameterisation, leaving only the (weaker) outcome coupling. More
general (any non-Gaussian outcome) but a much larger build. This is the "proper" fibr
door-1: the exact conditional-score map applied to the clean part of the fibre.

## Validity note (partial collapse is NOT the naive version)
Marginalising `f` against `X_obs` ONLY for the hyperparameter step is BIASED: it targets
`p(theta | X_obs)`, dropping `y`'s (indirect) information about `theta`. The correct
marginal is `p(theta, b0_gp | X_obs, y-PG)`, which the primary fix above computes.

## Validation plan
1. Implement the primary fix; keep NB gated behind `BJLM_GP_COLLAPSE_NB=1`.
2. NB-GP A/B vs whitened: require (a) NO runaway, (b) recovery agreement with a LONG
   whitened reference run, (c) ESS improvement on the GP hyperparameters.
3. If it agrees and is stable, ungate NB.
4. SBC (NB outcome + latent GP, no IPW) for calibration, as for the Gaussian case.
