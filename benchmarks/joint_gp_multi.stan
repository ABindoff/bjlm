data {
  int<lower=1> N;                  
  int<lower=1> N_subj;
  int<lower=1> T_subj;
  vector[N] Y;                     
  vector[N] X_obs;                 
  vector[T_subj] time_grid;
  vector[N] w; // IPW weights
}

parameters {
  // Global effects
  real b0_int;
  real b0_x;
  real<lower=0> sigma_y;
  
  // GP Hyperparameters
  real<lower=0> alpha;
  real<lower=0> rho;
  
  // Noisy covariate error
  real<lower=0> sigma_x;
  
  // GP latent states (standard normal for non-centered parameterization)
  vector[N] z;
}

transformed parameters {
  vector[N] X_true;
  {
    matrix[T_subj, T_subj] L_K;
    matrix[T_subj, T_subj] K;
    
    // RBF Kernel on the shared time grid
    for (i in 1:(T_subj-1)) {
      K[i, i] = alpha^2 + 1e-9;
      for (j in (i+1):T_subj) {
        K[i, j] = alpha^2 * exp(-0.5 * square((time_grid[i] - time_grid[j]) / rho));
        K[j, i] = K[i, j];
      }
    }
    K[T_subj, T_subj] = alpha^2 + 1e-9;
    
    L_K = cholesky_decompose(K);
    
    // Non-centered parameterization per subject
    for (s in 1:N_subj) {
      int start_idx = (s - 1) * T_subj + 1;
      int end_idx = s * T_subj;
      X_true[start_idx:end_idx] = L_K * z[start_idx:end_idx];
    }
  }
}

model {
  // Priors (matched to bjlm)
  b0_int ~ normal(0, 10);
  b0_x ~ normal(0, 10);
  sigma_y ~ inv_gamma(1.0, 1.0);
  
  alpha ~ lognormal(0.0, 1.0);
  rho ~ lognormal(0.0, 1.0);
  sigma_x ~ lognormal(-1.0, 1.0);
  
  z ~ std_normal();
  
  // Covariate model (unweighted GP observations)
  X_obs ~ normal(X_true, sigma_x);
  
  // Outcome model (weighted)
  for (n in 1:N) {
    target += w[n] * normal_lpdf(Y[n] | b0_int + b0_x * X_true[n], sigma_y);
  }
}
