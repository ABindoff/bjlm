# H0: BJLM pipeline map and fibr swap-in seams

Phase H0 deliverable for `PLAN_bjlm_mechanisms.md`. Maps the compiled model's
per-iteration sampler, rates each block's mixing risk, and identifies the drop-in
seams for the fibr mechanisms. Read alongside `design.md` (architecture) and
`PAPER_fibr_sampler.md` (the mechanisms and their validation bar).

Source read: `src/rust/src/sampler_bjlm.rs` (orchestrator + weighted updates),
`src/rust/src/sampler_re.rs` (the un-weighted smoothbp sampler these mirror),
plus `design.md`, `README.md`, `piped_api.R`.

## Per-iteration order (run_chain_bjlm, sampler_bjlm.rs:1897)

```
1. PROPENSITY (cut)     sample_propensity            PG-augmented Gibbs, exact
2. WEIGHTS              compute_weights + expand     deterministic, one-way alpha -> outcome
3. GP latents           sample_gp_state              elliptical slice, whitened latents
4. GP hyperparameters   sample_gp_hyperparameters    adaptive RW-MH (log scale), non-centred
5. LINEAR COEFS         sample_linear_coefs_weighted conjugate Gibbs (Gaussian) / PG (Bin, NB)
6. RANDOM EFFECTS       sample_random_effects_weighted   CENTRED Gibbs on u_b0
7. CHANGE-POINTS        hmc_step_om/rho_weighted     adaptive HMC (mass matrix, dual-avg, jittered L)
8. LINEAR COEFS (2nd)   sample_linear_coefs_weighted
9. SIGMA / SIGMA_U / R / GAMMA   conjugate or RW-MH; spike-slab gamma
```

## Block-by-block: current update, mixing risk, fibr opportunity

| Block | Current update | Mixing risk | fibr opportunity |
|-------|----------------|-------------|------------------|
| Propensity alpha | PG-augmented Gibbs, perfect acceptance | LOW (exact) | none needed; this is already the ideal |
| Linear coefs b0,b1,delta | conjugate Gibbs (Gaussian) / PG-weighted (Bin,NB) | LOW-MED | fine; the coupling to random effects is the issue, not the coefs |
| **Random effects u_b0 + sigma_u** | **CENTRED Gibbs**: draw u\|rest (post_mean,post_sd), then sigma_u\|u conjugate | **HIGH (funnel)** | **primary H1 target.** Exact score non-centring; Gaussian/PG makes it closed-form |
| Change-point loc/sharpness beta_om, beta_rho | adaptive HMC (HmcAdapt: mass matrix, dual-avg, jittered L) | MED | already geometric; watch for multimodality (H4). If beta_om has random effects (sigma_re_om), that variance is a second centred funnel |
| GP latents Z | elliptical slice on whitened latents | LOW-MED | ESS is solid; the whitening already non-centres w.r.t. hyperparameters |
| GP hyperparameters alpha,rho,sigma_x | joint + componentwise RW-MH, dual-avg, Welford mass | MED-HIGH | H3 target: the length-scale funnel; the holonomy metric applies to the base-dependent GP covariance |
| sigma, sigma_u, sigma_re_om | conjugate Gibbs | LOW alone, HIGH via the funnel with their effects | the variance side of the centred funnels above |
| r (NB dispersion) | RW-MH log scale, simple adaptive step | MED | minor; could share the base geometric sampler |
| spike-slab gamma | Kuo-Mallick Gibbs | discrete, separate concern | leave; the cut is preserved inside the IPW block |

## The bottleneck (Gate H0)

**The centred random-effects funnel (u_b0, sigma_u) is the clear primary
bottleneck, and it has a clean swap seam.** `sample_random_effects_weighted`
draws each u_j from its conjugate Gaussian (post_mean, post_sd), then
`sample_sigma_u_weighted` draws sigma_u from sum(u_j^2). This is the textbook
centred parameterisation: u and sigma_u are updated in alternation, which mixes
slowly precisely when the data are sparse per group or sigma_u is small (the
funnel neck). This is the most likely source of "not every parameter mixes well",
especially for sigma_u and the group effects.

Why it is the ideal fibr target: the fibr obstacle in the paper (a Laplace /
Newton solve, because the logistic fibre conditional is non-Gaussian) DOES NOT
ARISE here. Under the Gaussian outcome, u_b0 | rest is exactly Gaussian; under
Binomial/NB the same code path already draws PG latents so the conditional is
Gaussian too. The conjugate post_mean m_j and post_sd s_j are ALREADY COMPUTED in
`sample_random_effects_weighted`. So the exact score coordinate
z_j = (u_j - m_j) / s_j is closed-form and z_j ~ N(0, 1) by construction. The
non-centring is essentially free.

Gate H0: PASS. A funnel-prone block (centred u_b0/sigma_u) is identified, it is a
genuine bottleneck, and its interface (`sample_random_effects_weighted` +
`sample_sigma_u_weighted`) is a clean drop-in seam.

## Recommended H1 target and shape of the fix

Replace the centred (u, sigma_u) alternation with an interweaved / horizontally
non-centred update, using the machinery already present:

1. Keep the conjugate draw of u (it gives m_j, s_j for free).
2. Add the ancillary (non-centred) update: hold z_j = (u_j - m_j)/s_j fixed and
   update sigma_u (and any hyperparameters that move m_j, s_j) jointly. This is
   the ASIS / interweaving move (Yu-Meng 2011), which the fibr exact score map
   makes principled; here it is closed-form.
3. Optionally, put the variance block (log sigma_u, and log sigma_re_om) under the
   constant Fisher-adapted geometric base sampler from fibr
   (`adapted_base_hmc`-style, base_step_scale decoupled) if the interweave alone
   does not flatten the neck.

The swap is local: a new `sample_random_effects_asis` (or a modification of
`sample_random_effects_weighted` to interleave centred and non-centred passes),
plus the corresponding sigma_u update conditioning on z rather than u. No change
to the pipe contract, the cut, or the weight flow.

## Validation gates before any benchmark (inherited, non-negotiable)

The repo currently is NOT calibrated (user). So H1 must, in order:
1. FD-check any new gradient (if the geometric base sampler is used).
2. SBC the modified sampler on a small instance (the recovery.R / simulate.R
   infrastructure is the natural harness; see below).
3. Compare ESS/gradient (and, since Rust is compiled, ESS/second is a fair
   comparison here too) vs the current centred update, replicated + rhat-filtered.

Harness note: `R/simulate.R` and `R/recovery.R` already exist for
data generation and parameter recovery; these are the seed of the SBC and
benchmark harness. Confirm they cover the random-effects variance recovery, which
is exactly what the funnel fix must not break.

## Interactions to keep in view (H4, flagged now)

- Change-point beta_om can be multimodal; the HMC already there may be the right
  tool, and the geometric base sampler assumes unimodality, so do NOT redirect the
  change-point block to a constant-metric sampler without checking.
- The Bayesian cut is a one-way weight flow (alpha -> weights -> outcome). The
  random-effects fix lives entirely inside the outcome block, downstream of the
  cut, so the cut is preserved trivially.
- Weighted likelihood: the fibr score map for u must carry the observation
  weights w_i into the conjugate m_j, s_j (the current code already does, via the
  weighted sufficient statistics). Preserve that.

## Next action (H1)

Implement the interweaved random-effects update on a Gaussian-outcome instance
first (closed-form, simplest), FD/SBC it, benchmark vs the centred update on a
funnel-inducing simulation (small groups / small sigma_u). Then extend to the
Binomial and NB families (same PG code path). Gate H1: exact (SBC) and better
ESS/gradient on the funnel block.
