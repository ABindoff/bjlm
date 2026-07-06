// Joint latent-GP time-varying confounder + Negative-Binomial change-point
// outcome model, matched to bjlm's conditional-transport NB+GP sampler
// (see vignettes/gp-nb-changepoint.Rmd and data-raw/sbc_nb_gp_transport.R,
// which this model's priors and DGP mirror exactly).
//
// mu = b0 + b0_gp * f(tau) + b1*(tau-omega) + delta*(tau-omega)*sigmoid(rho*(tau-omega))
// y  ~ NegBinomial2Log(mu, r)
// f  ~ GP(0, alpha^2 * SE(rho_gp)), one draw per subject on a shared time grid
// X_obs = f + noise (sigma_x)
//
// This is the full joint posterior over the latent GP field AND the
// nonlinear change-point simultaneously -- the combination bjlm's fibr-derived
// conditional-transport reparameterisation was built to handle, and which
// plain non-centered HMC (this file) is expected to struggle with via the
// GP-hyperparameter funnel interacting with the change-point's own geometry.

data {
  int<lower=1> N;                  // total observations (N_subj * T_subj)
  int<lower=1> N_subj;
  int<lower=1> T_subj;
  array[N] int<lower=0> Y;
  vector[N] X_obs;
  vector[N] tau;                   // per-row time (repeats time_grid per subject)
  vector[T_subj] time_grid;        // shared time grid
  real rho_loc;                    // resolution-aware lognormal location for gp_rho
  real rho_scale;                  // resolution-aware lognormal scale for gp_rho
}

parameters {
  real b0;
  real b0_gp;
  real b1;
  real delta;
  real<lower=0.5, upper=9.5> omega;
  real<lower=1, upper=10> rho_cp;
  real<lower=0> r;                 // NB dispersion ("size"), bjlm convention

  real<lower=0> gp_alpha;
  real<lower=0> gp_rho;
  real<lower=0> gp_sigma_x;

  vector[N] z;                     // non-centered GP innovations
}

transformed parameters {
  vector[N] f;
  {
    matrix[T_subj, T_subj] K;
    matrix[T_subj, T_subj] L_K;

    for (i in 1:(T_subj - 1)) {
      K[i, i] = gp_alpha^2 + 1e-8;
      for (j in (i + 1):T_subj) {
        K[i, j] = gp_alpha^2 * exp(-0.5 * square((time_grid[i] - time_grid[j]) / gp_rho));
        K[j, i] = K[i, j];
      }
    }
    K[T_subj, T_subj] = gp_alpha^2 + 1e-8;
    L_K = cholesky_decompose(K);

    for (s in 1:N_subj) {
      int lo = (s - 1) * T_subj + 1;
      int hi = s * T_subj;
      f[lo:hi] = L_K * z[lo:hi];
    }
  }
}

model {
  // Priors matched to bjlm's smoothbp_priors()/gp_priors() defaults used in
  // data-raw/sbc_nb_gp_transport.R (the certified NB+GP high-count regime).
  b0     ~ normal(2.5, 0.5);
  b0_gp  ~ normal(0, 1);
  b1     ~ normal(0, 0.3);
  delta  ~ normal(0, 0.8);
  omega  ~ normal(5, 1.5);         // truncated to [0.5, 9.5] by the declared bounds
  rho_cp ~ normal(4, 1.5);         // truncated to [1, 10] by the declared bounds
  r      ~ gamma(2, 0.2);          // shape 2, rate 0.2 == R's prior_gamma(2, scale = 5)

  gp_alpha   ~ lognormal(0, 1);
  gp_rho     ~ lognormal(rho_loc, rho_scale);  // resolution-aware, see R driver
  gp_sigma_x ~ lognormal(-1, 1);

  z ~ std_normal();
  X_obs ~ normal(f, gp_sigma_x);

  {
    vector[N] d = tau - omega;
    vector[N] mu = b0 + b0_gp * f + b1 * d + delta * (d .* inv_logit(rho_cp * d));
    Y ~ neg_binomial_2_log(mu, r);
  }
}
