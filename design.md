# bayesian_ipw: Joint Bayesian Inverse Probability Weighting

## The Problem

In observational longitudinal studies (e.g., THBP), treatment/exposure
assignment is non-random. Inverse probability weighting (IPW) creates a
pseudo-population where confounders are balanced, enabling causal inference.

The standard workflow is:

1. **Estimate propensity scores** — fit a logistic model: P(T=1 | X) = e(X)
2. **Compute weights** — w_i = 1/e(X_i) for treated, 1/(1-e(X_i)) for control
3. **Fit weighted outcome model** — treating weights as fixed

**The fundamental problem:** Step 3 treats the weights as known, ignoring
estimation uncertainty from Step 1. This produces:
- Confidence intervals that are too narrow
- No valid Bayesian interpretation (the posterior is conditional on point
  estimates of weights, not marginalised over their uncertainty)

### Why existing solutions are inadequate

| Approach | Problem |
|----------|---------|
| **brms / Stan** (two-step, posterior draws of weights) | Requires fitting the outcome model N_draws times, or passing weight draws through Stan's data block. Computationally prohibitive for MCMC on longitudinal data. |
| **Frequentist sandwich estimator** | Corrects SE for weight estimation, but no Bayesian posterior. |
| **Joint Stan model** | Can be written, but Stan's NUTS sampler treats the full joint parameter space as one HMC target — very slow for the propensity + outcome block structure. |
| **bcf** (Bayesian Causal Forests) | BART-based; excellent for heterogeneous effects but not designed for longitudinal trajectory models with breakpoints. |

### The opportunity

A custom Rust sampler that exploits the **block structure** of the joint
likelihood can be orders of magnitude faster:

- The propensity model (logistic) and outcome model (Gaussian/GLM with
  piecewise trajectory) share covariates but have **separate parameter blocks**
- Gibbs sampling alternates between blocks, conditioning on the other
- The propensity block can use **Pólya-Gamma augmentation** to get exact
  conjugate Gibbs updates (no MH, no tuning)
- The outcome block already exists in smoothbp's Rust sampler

This architecture naturally propagates weight uncertainty into the causal
effect posterior — without fitting the outcome model N times.

---

## Mathematical Framework

### The Joint Model

Let:
- $Y_i$ = outcome (cognitive score) for subject $i$ at time $t$
- $T_i$ = binary treatment indicator (1 = experimental, 0 = control)
- $\mathbf{X}_i$ = baseline confounders (age, education, sex, etc.)
- $\boldsymbol{\alpha}$ = propensity model parameters
- $\boldsymbol{\theta}$ = outcome model parameters

**Propensity model (treatment assignment):**
$$
T_i | \mathbf{X}_i, \boldsymbol{\alpha} \sim \text{Bernoulli}(\pi_i), \quad
\pi_i = \text{logistic}(\mathbf{X}_i^\top \boldsymbol{\alpha})
$$

**IPW weights:**
$$
w_i(\boldsymbol{\alpha}) = \frac{T_i}{\pi_i(\boldsymbol{\alpha})} +
\frac{1 - T_i}{1 - \pi_i(\boldsymbol{\alpha})}
$$

**Weighted outcome model (marginal structural model):**
$$
Y_{it} | \boldsymbol{\theta}, w_i \sim \mathcal{N}(\mu_{it}(\boldsymbol{\theta}),
\sigma^2 / w_i)
$$

where $\mu_{it}$ follows the smoothbp piecewise trajectory (or any GLM).

**Joint posterior:**
$$
p(\boldsymbol{\alpha}, \boldsymbol{\theta} | \mathbf{Y}, \mathbf{T}, \mathbf{X})
\propto \underbrace{p(\mathbf{Y} | \boldsymbol{\theta}, w(\boldsymbol{\alpha}))}_{\text{weighted outcome likelihood}}
\cdot \underbrace{p(\mathbf{T} | \mathbf{X}, \boldsymbol{\alpha})}_{\text{propensity likelihood}}
\cdot p(\boldsymbol{\alpha}) \cdot p(\boldsymbol{\theta})
$$

### Weight Variants

| Weight type | Formula | Use case |
|-------------|---------|----------|
| **ATE** (average treatment effect) | $w_i = T_i/\pi_i + (1-T_i)/(1-\pi_i)$ | Population-level causal effect |
| **ATT** (effect on the treated) | $w_i = T_i + (1-T_i)\pi_i/(1-\pi_i)$ | Effect among those who received treatment |
| **Stabilised ATE** | $w_i = T_i \cdot P(T=1)/\pi_i + (1-T_i) \cdot P(T=0)/(1-\pi_i)$ | Reduces variance; recommended for MSMs |
| **Trimmed** | $w_i = \min(w_i, c)$ for some threshold $c$ | Prevents extreme weights from dominating |

Stabilised weights are recommended for longitudinal models (Robins, Hernán,
Brumback 2000). The stabilisation constant $P(T=1)$ is estimated from the
data and has its own posterior — the joint sampler handles this automatically.

---

## The Pólya-Gamma Trick

The key to making the joint sampler efficient is **Pólya-Gamma augmentation**
(Polson, Scott, Windle 2013) for the propensity model.

### The problem with logistic likelihood

The logistic likelihood is not conjugate with a Gaussian prior on $\alpha$:
$$
p(\mathbf{T} | \boldsymbol{\alpha}) = \prod_i \pi_i^{T_i} (1-\pi_i)^{1-T_i}
$$

This requires MH proposals with tuning — slow and fragile.

### Pólya-Gamma augmentation

Introduce latent variables $\omega_i \sim PG(1, \mathbf{X}_i^\top \boldsymbol{\alpha})$.
Conditional on $\omega_i$, the logistic regression becomes a **weighted linear
regression** with known closed-form Gibbs updates:

**Step 1: Sample PG latent variables**
$$
\omega_i | \boldsymbol{\alpha} \sim PG(1, \mathbf{X}_i^\top \boldsymbol{\alpha})
$$

**Step 2: Sample propensity coefficients (exact Gibbs)**
$$
\boldsymbol{\alpha} | \boldsymbol{\omega}, \mathbf{T} \sim
\mathcal{N}(\mathbf{m}_\omega, \boldsymbol{\Sigma}_\omega)
$$

where:
$$
\boldsymbol{\Sigma}_\omega = (\mathbf{X}^\top \boldsymbol{\Omega} \mathbf{X} + \mathbf{B}^{-1})^{-1}, \quad
\mathbf{m}_\omega = \boldsymbol{\Sigma}_\omega (\mathbf{X}^\top (\mathbf{T} - \tfrac{1}{2}) + \mathbf{B}^{-1} \mathbf{b})
$$

with $\boldsymbol{\Omega} = \text{diag}(\omega_1, \ldots, \omega_N)$ and
prior $\boldsymbol{\alpha} \sim \mathcal{N}(\mathbf{b}, \mathbf{B})$.

This is **exact** — no MH acceptance ratio, no tuning, perfect acceptance.
The Cholesky solve is the same infrastructure smoothbp already uses for the
Gaussian linear coefficient block.

### Sampling PG random variables

The Pólya-Gamma distribution $PG(b, c)$ with $b=1$ can be sampled efficiently
via the method of Devroye (2009) or the alternating series method. For $b=1$:

$$
PG(1, c) = \frac{1}{2c} \cdot J^*(1, c/2)
$$

where $J^*$ is a Jacobi distribution. There are efficient truncated series
algorithms. Windle, Carvalho, Scott (2014) provide Rust-friendly
pseudocode.

**Rust implementation strategy:** Port the `rpg_devroye()` sampler from the
BayesLogit R package. This is ~50 lines of C that translate directly to Rust.

---

## Gibbs Sampler Architecture

### Per-iteration structure

```
for iter in 1..n_iter:
    # --- Propensity block (PG-augmented) ---
    1. Sample ω_i | α          ~ PG(1, X_i'α)         for all i
    2. Sample α | ω, T         ~ N(m_ω, Σ_ω)          [Cholesky]
    3. Compute π_i(α) and w_i(α)

    # --- Outcome block (smoothbp) ---
    4. Sample β₀, β₁, δ | ω, w    [weighted Gibbs/IWLS]
    5. Sample u | w                [weighted random effects]
    6. Sample ω_k, ρ_k | w        [weighted HMC]
    7. Sample σ | w                [weighted conjugate]
    8. (if spike-slab) Sample γ | w

    # --- Store draws ---
    9. Store (α, π, w, θ, σ)
```

### What changes in the outcome block

The only change is that the Gaussian likelihood becomes **weighted**:

$$
\ell(\boldsymbol{\theta}) = -\frac{1}{2\sigma^2} \sum_i w_i (Y_i - \mu_i)^2
$$

This affects:

| Component | Change |
|-----------|--------|
| `sample_linear_coefs` | $P = X'WX/\sigma^2 + P_\text{prior}$ instead of $X'X/\sigma^2$ |
| `sample_sigma` | Shape: $\alpha + \sum w_i / 2$; Scale: $\beta + \sum w_i r_i^2 / 2$ |
| `sample_random_effects` | Weighted sufficient statistics |
| `hmc_step_om/rho` | Energy: $-\frac{1}{2\sigma^2} \sum w_i r_i^2$; gradient scaled by $w_i$ |
| `sample_gamma` | Log-likelihood contributions scaled by $w_i$ |

All of these are **trivial modifications** — multiply residual contributions
by $w_i$. The Cholesky infrastructure is unchanged.

### Block diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Joint Posterior                        │
│                                                          │
│  ┌──────────────────────┐   ┌─────────────────────────┐ │
│  │  Propensity Block    │   │   Outcome Block         │ │
│  │                      │   │                         │ │
│  │  PG augmentation     │──▶│  Weighted smoothbp      │ │
│  │  α ~ N(m_ω, Σ_ω)    │   │  β, δ, ω, ρ, σ, u      │ │
│  │  ω_i ~ PG(1, X'α)   │   │                         │ │
│  │                      │   │  w_i = f(π_i(α))        │ │
│  │  No MH needed        │   │  enters as observation  │ │
│  │  Perfect acceptance   │   │  weights only           │ │
│  └──────────────────────┘   └─────────────────────────┘ │
│                                                          │
│  Coupling: w_i(α) flows from propensity → outcome        │
│  No feedback: outcome does NOT influence α               │
└─────────────────────────────────────────────────────────┘
```

---

## The Feedback Problem and How to Handle It

### What is feedback?

In a fully joint likelihood, the outcome $Y$ can influence the posterior of
$\alpha$ (propensity parameters) through the weighted likelihood term. This
is problematic because:

1. The propensity score should reflect the **treatment assignment mechanism**,
   not the outcome
2. Feedback can produce propensity scores that are "too good" — they overfit
   to the outcome, biasing the causal effect

### Solution: Cut feedback (modular Bayes)

Use the **cut posterior** approach (Plummer 2015; Jacob et al. 2017):

$$
p_{\text{cut}}(\boldsymbol{\alpha}, \boldsymbol{\theta} | \mathbf{Y}, \mathbf{T})
\propto p(\mathbf{Y} | \boldsymbol{\theta}, w(\boldsymbol{\alpha}))
\cdot p(\mathbf{T} | \boldsymbol{\alpha})
\cdot p(\boldsymbol{\alpha}) \cdot p(\boldsymbol{\theta})
$$

**but** the update for $\alpha$ conditions only on $(\mathbf{T}, \mathbf{X})$,
not on $\mathbf{Y}$. Operationally, this means:

- Step 2 (sample α) uses **only** the propensity likelihood $p(T|X,α)$
- Step 4-8 (sample θ) uses **both** the outcome likelihood with current $w_i(α)$

This is trivially implemented in the blocked Gibbs: the propensity block
simply ignores the outcome data. The blocks share only $w_i$ as a one-way
data flow.

### Why this is elegant

The cut posterior has a clean interpretation:
- $\alpha$ has a valid posterior from the propensity model alone
- $\theta$ has a valid posterior that **integrates over** the uncertainty in
  $\alpha$ (via the varying $w_i$ across MCMC iterations)
- No feedback, no bias, proper uncertainty propagation

---

## Practical Design Decisions

### Weight stabilisation

Unstabilised weights can be extreme (e.g., $w_i > 100$ when $\pi_i \approx 0$).
This inflates variance and can cause numerical issues.

**Stabilised weights:**
$$
sw_i = \frac{T_i \cdot \hat{p}}{e(\mathbf{X}_i)} +
\frac{(1 - T_i) \cdot (1 - \hat{p})}{1 - e(\mathbf{X}_i)}
$$

where $\hat{p} = P(T=1) = \bar{T}$ (marginal treatment probability).

In the Gibbs sampler, $\hat{p}$ is a constant (proportion treated in the
sample). The weights are recomputed from $\pi_i(\alpha)$ at each iteration.

**Optional weight trimming:**
```rust
let w_trimmed = w_i.min(max_weight);  // e.g., max_weight = 20
```

This can be user-configurable with a sensible default.

### Positivity diagnostics

At each iteration, check $\pi_i$ values. Flag:
- $\pi_i < 0.01$ or $\pi_i > 0.99$ (near-violations of positivity)
- Report the proportion of iterations where any $\pi_i$ is extreme
- Store $\min(\pi_i)$ and $\max(\pi_i)$ across iterations for diagnostics

### Covariate selection for propensity model

Not all covariates that predict the outcome should be in the propensity model.
Variables that predict treatment assignment but NOT the outcome can increase
variance without reducing bias ("instruments").

**Options:**
1. User specifies propensity covariates via a formula
2. Spike-and-slab on $\alpha$ (select treatment predictors automatically)
3. Both: user formula + optional spike-and-slab

The PG-augmented Gibbs naturally supports spike-and-slab on $\alpha$ using
the same Kuo-Mallick formulation as smoothbp_ss.

---

## R-Side API Design

### Basic usage

```r
library(bayesian_ipw)

fit <- bipw(
  # Outcome model (smoothbp piecewise trajectory)
  outcome = score ~ tau,
  b0      = ~ 1 + Group + age_s + education_s + (1 | subject),
  b1      = ~ 1 + Group,
  deltas  = list(~ 1 + Group),
  omega   = list(~ 1 + Group),
  rho     = list(~ 1),

  # Propensity model
  propensity = Group ~ age_s + education_s + sex,

  # Weight specification
  weights = "stabilised_ate",  # or "ate", "att", "stabilised_att"
  max_weight = 20,             # trimming threshold

  # Data
  data   = thbp_long,

  # Priors
  outcome_priors = smoothbp_priors(
    b0 = prior_normal(0, 2),
    b1 = prior_normal(0, 2),
    ...
  ),
  propensity_priors = list(
    alpha = prior_normal(0, 2.5)  # weakly informative for logistic
  ),

  # MCMC
  chains = 4, iter = 5000, warmup = 3000,
  seed = 42
)
```

### Output

```r
# Treatment model summary (propensity)
summary(fit, model = "propensity")

# Outcome model summary (weighted smoothbp)
summary(fit, model = "outcome")

# Causal effect summary (ATE / ATT)
causal_effect(fit)

# Weight diagnostics
weight_diagnostics(fit)
# → plot of weight distribution, positivity checks, ESS of weights

# Fitted values (on response scale, weighted)
fitted(fit)

# Model comparison via spike-and-slab (if enabled)
pip(fit)
```

### Key functions

| Function | Description |
|----------|-------------|
| `bipw()` | Fit joint propensity + outcome model |
| `summary.bipw_fit()` | Posterior summaries for either model |
| `causal_effect()` | Marginal causal effect with CrI |
| `weight_diagnostics()` | Weight distribution, positivity, ESS |
| `fitted.bipw_fit()` | Fitted values with uncertainty |
| `pp_check.bipw_fit()` | Posterior predictive checks |
| `pip.bipw_fit()` | PIPs for spike-and-slab parameters |
| `tab_bipw()` | Formatted results table |
| `plot.bipw_fit()` | Trajectory plot with causal contrasts |

---

## Rust Sampler Implementation

### Module structure

```
src/rust/src/
├── lib.rs              # R interface
├── model.rs            # Data structures (shared with smoothbp?)
├── polya_gamma.rs      # PG(1, c) sampler
├── propensity.rs       # PG-augmented propensity block
├── outcome.rs          # Weighted outcome block (adapted smoothbp sampler)
├── weights.rs          # Weight computation (ATE/ATT/stabilised)
└── sampler.rs          # Main chain loop orchestrating both blocks
```

### PG(1, c) sampler in Rust

The Devroye method for PG(1, c) with b=1:

```rust
/// Sample from PG(1, c) using the method of Devroye (2009)
/// as described in Windle et al. (2014).
pub fn sample_pg1(c: f64, rng: &mut StdRng) -> f64 {
    let c_abs = c.abs();
    if c_abs < 1e-12 {
        // PG(1, 0) ~ Ga(1, 1) / (4 * pi^2) approximately
        // Actually PG(1, 0) = sum_{k=0}^inf Ga(1,1) / ((k+0.5)^2 * 4*pi^2)
        // Use the truncated series or J*(1, 0) method
        return sample_pg1_small_c(rng);
    }

    // For |c| > 0, use alternating series / Devroye method
    let t = 0.64;
    let k = c_abs * c_abs / 2.0 + std::f64::consts::FRAC_PI_2.powi(2) / 2.0;

    loop {
        // Proposal from mixture of truncated inverse Gaussian and exponential
        let (x, accepted) = proposal_pg(c_abs, t, k, rng);
        if accepted {
            return x;
        }
    }
}
```

The full implementation (~60 lines) follows Windle's BayesLogit C code.

### Propensity block

```rust
fn sample_propensity(
    treatment: &[f64],      // T_i (0/1)
    x_prop: &DMatrix<f64>,  // propensity design matrix
    alpha: &mut DVector<f64>,
    omega_pg: &mut Vec<f64>,
    prior_mean: &DVector<f64>,
    prior_prec: &DMatrix<f64>,  // B^{-1}
    rng: &mut StdRng,
) {
    let n = treatment.len();
    let p = x_prop.ncols();

    // Step 1: Sample PG latent variables
    for i in 0..n {
        let psi = x_prop.row(i).dot(alpha);
        omega_pg[i] = sample_pg1(psi, rng);
    }

    // Step 2: Construct weighted regression
    // Ω = diag(ω)
    // Σ = (X'ΩX + B⁻¹)⁻¹
    // m = Σ(X'(T - 0.5) + B⁻¹b)
    let mut xtox = DMatrix::zeros(p, p);
    let mut xty = DVector::zeros(p);
    for i in 0..n {
        let xi = x_prop.row(i);
        let wi = omega_pg[i];
        let ki = treatment[i] - 0.5;
        for j in 0..p {
            xty[j] += xi[j] * ki;
            for l in j..p {
                let v = xi[j] * wi * xi[l];
                xtox[(j, l)] += v;
                if l != j { xtox[(l, j)] += v; }
            }
        }
    }

    let precision = xtox + prior_prec;
    let chol = precision.cholesky().expect("Propensity precision not PD");
    let rhs = xty + prior_prec * prior_mean;
    let mean = chol.solve(&rhs);

    // Sample from N(mean, precision^{-1})
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z = DVector::from_iterator(p, (0..p).map(|_| normal.sample(rng)));
    let y = chol.l().transpose()
        .solve_upper_triangular(&z)
        .expect("Failed triangular solve");
    *alpha = mean + y;
}
```

### Weight computation

```rust
fn compute_weights(
    treatment: &[f64],
    x_prop: &DMatrix<f64>,
    alpha: &DVector<f64>,
    weight_type: WeightType,
    max_weight: f64,
) -> Vec<f64> {
    let n = treatment.len();
    let p_marginal = treatment.iter().sum::<f64>() / n as f64;
    let mut weights = vec![0.0; n];

    for i in 0..n {
        let eta = x_prop.row(i).dot(alpha);
        let pi_i = sigmoid(eta);
        let pi_i = pi_i.clamp(1e-6, 1.0 - 1e-6);  // prevent extreme weights

        weights[i] = match weight_type {
            WeightType::Ate => {
                if treatment[i] > 0.5 { 1.0 / pi_i } else { 1.0 / (1.0 - pi_i) }
            }
            WeightType::Att => {
                if treatment[i] > 0.5 { 1.0 } else { pi_i / (1.0 - pi_i) }
            }
            WeightType::StabilisedAte => {
                if treatment[i] > 0.5 { p_marginal / pi_i }
                else { (1.0 - p_marginal) / (1.0 - pi_i) }
            }
            WeightType::StabilisedAtt => {
                if treatment[i] > 0.5 { 1.0 }
                else { p_marginal * pi_i / ((1.0 - p_marginal) * (1.0 - pi_i)) }
            }
        };

        weights[i] = weights[i].min(max_weight);
    }
    weights
}
```

### Weighted outcome modifications

The changes to the existing smoothbp sampler are minimal. Here's the
weighted version of `sample_sigma`:

```rust
fn sample_sigma_weighted(
    data: &ModelData, priors: &Priors, state: &mut State,
    weights: &[f64], rng: &mut StdRng
) {
    let mu = state.means(data);
    let mut wss = 0.0;   // weighted sum of squares
    let mut wn = 0.0;    // effective sample size
    for i in 0..data.n {
        let r = data.y[i] - mu[i];
        wss += weights[i] * r * r;
        wn += weights[i];
    }
    let shape = priors.sigma_shape + wn * 0.5;
    let scale = priors.sigma_scale + wss * 0.5;
    let gamma_dist = Gamma::new(shape, 1.0 / scale).unwrap();
    state.sigma = 1.0 / gamma_dist.sample(rng).sqrt();
}
```

And the weighted `sample_linear_coefs` — the only change is $X'X \to X'WX$
and $X'y \to X'Wy$:

```rust
// In sample_linear_coefs_weighted:
// Replace:  let mut precision = &xt * &x_full / sigma2;
// With:
let mut w_x = x_full.clone();
for i in 0..n {
    let mut row = w_x.row_mut(i);
    for j in 0..p_total { row[j] *= weights[i]; }
}
let mut precision = &xt * &w_x / sigma2;

// Replace:  let xty = &xt * &y_tilde / sigma2;
// With:
let mut w_y = y_tilde.clone();
for i in 0..n { w_y[i] *= weights[i]; }
let xty = &xt * &w_y / sigma2;
```

---

## Implementation Roadmap

### Phase 1: Core infrastructure

1. Set up R package skeleton (`bayesian_ipw/`)
2. Implement PG(1, c) sampler in Rust
3. Test PG sampler against BayesLogit R package (compare distributions)
4. Implement propensity block (PG-augmented Gibbs)
5. Test: recover known logistic regression coefficients

### Phase 2: Weighted outcome model (Gaussian)

1. Fork smoothbp's `sampler.rs`, add weight vector
2. Implement `sample_linear_coefs_weighted`
3. Implement `sample_sigma_weighted`
4. Implement weighted HMC for ω, ρ
5. Test: simulate confounded data, verify ATE recovery

### Phase 3: Joint sampler

1. Wire up propensity + outcome blocks in `sampler.rs`
2. Implement weight computation (ATE, ATT, stabilised)
3. Implement cut-feedback architecture
4. R-side: `bipw()` function, prior specification, output assembly
5. Test: full simulation study (confounded treatment, known ATE)

### Phase 4: Diagnostics and post-processing

1. Weight diagnostics (distribution, ESS, positivity)
2. `summary()`, `print()`, `fitted()`, `pp_check()`
3. Causal effect extraction with CrI
4. Trajectory plots with counterfactual contrasts
5. `tab_bipw()` formatted table

### Phase 5: Extensions

1. Spike-and-slab on α (propensity variable selection)
2. GLM families for outcome (connect to smoothbp family extension)
3. Time-varying confounders (longitudinal propensity scores)
4. Marginal structural models with time-varying treatment

---

## Simulation Study Design

### Simulation 1: Basic ATE recovery

```r
# Generate confounded data
set.seed(42)
n <- 500
X1 <- rnorm(n)                           # confounder: age
X2 <- rnorm(n)                           # confounder: education
pi <- plogis(0.5 + 0.8*X1 - 0.3*X2)     # true propensity
T  <- rbinom(n, 1, pi)                    # treatment

# Outcome (true ATE = 2.0)
Y <- 5 + 2.0*T + 1.5*X1 + 0.5*X2 + rnorm(n)

# Naive estimate (biased): lm(Y ~ T) → β_T ≈ 2.0 + confounding bias
# IPW estimate (unbiased): bipw(...) → ATE ≈ 2.0
```

### Simulation 2: Piecewise trajectory with confounding

```r
# Longitudinal data with breakpoint
# Treatment affects the post-breakpoint slope
# Age confounds both treatment assignment and cognitive trajectory
# → IPW needed to estimate the causal effect of treatment on
#   the slope-change parameter δ
```

### Simulation 3: Coverage and calibration

- Repeat 500 times
- Compare: bipw (joint), bipw (two-step), brms (weighted), naive
- Metrics: bias, RMSE, 95% CI coverage, CI width

---

## Comparison with Existing Software

| Feature | bayesian_ipw | brms (two-step) | Stan (custom) | WeightIt + lm |
|---------|-------------|-----------------|---------------|---------------|
| Joint posterior | ✓ | ✗ | ✓ | ✗ |
| Weight uncertainty propagation | ✓ (automatic) | Partial (manual) | ✓ (slow) | ✗ |
| PG augmentation (tuning-free) | ✓ | ✗ | ✗ (NUTS) | ✗ |
| Piecewise trajectories | ✓ (built-in) | ✗ (manual) | ✗ (manual) | ✗ |
| Spike-and-slab (breakpoint selection) | ✓ | ✗ | ✗ | ✗ |
| Cut feedback | ✓ | N/A | Manual | N/A |
| Speed (estimated) | Fast (Rust) | Slow | Very slow | Fast (no posterior) |
| Weight diagnostics | ✓ | Manual | Manual | ✓ (WeightIt) |

---

## Key References

1. **Polson, Scott, Windle (2013).** Bayesian Inference for Logistic Models
   Using Pólya-Gamma Latent Variables. *JASA* 108(504): 1339–1349.
   — The PG augmentation method.

2. **Robins, Hernán, Brumback (2000).** Marginal Structural Models and
   Causal Inference in Epidemiology. *Epidemiology* 11(5): 550–560.
   — Foundational MSM paper; stabilised weights.

3. **Plummer (2015).** Cuts in Bayesian graphical models.
   *Statistics and Computing* 25: 37–43.
   — Cut feedback methodology.

4. **Saarela, Stephens, Moodie, Klein (2015).** On Bayesian estimation of
   marginal structural models. *Biometrics* 71(2): 279–288.
   — Joint Bayesian MSM with discussion of feedback.

5. **Zigler (2016).** The Central Role of Bayes' Theorem for Joint Estimation
   of Causal Effects and Propensity Scores. *The American Statistician*
   70(1): 47–54. — Arguments for/against joint estimation.

6. **Windle, Carvalho, Scott (2014).** Sampling Pólya-Gamma Random Variates:
   Alternate and Approximate Techniques. — Efficient PG sampling algorithms.

7. **Liao, Zigler (2020).** Uncertainty in the design stage of two-stage
   Bayesian propensity score analysis. *Statistics in Medicine* 39(17):
   2267–2290. — Propagating PS uncertainty.

---

## Dependencies

### Rust crates
- `nalgebra` — linear algebra (already used by smoothbp)
- `rand`, `rand_distr` — random number generation (already used)
- No new crate dependencies needed; PG sampler is self-contained

### R packages
- `posterior` — draws manipulation (already used by smoothbp)
- `bayesplot` — pp_check plots
- `gt` — formatted tables
- `ggplot2` — diagnostics plots

### Relationship to smoothbp

`bayesian_ipw` could either:
1. **Import smoothbp** and use its sampler components via Rust linkage
2. **Fork the sampler code** with weight-aware modifications

Option 2 is simpler initially. Long-term, the weighted sampler could be
upstreamed into smoothbp as a `weights` argument, and bayesian_ipw would
provide the propensity model infrastructure on top.
