data {
  int<lower=1> N;                  
  vector[N] Y;                     
  vector[N] X_obs;                 
  vector[N] time;                  
  vector[N] w; // IPW weights
}

parameters {
  // Global effects
  real b0_int;
  real b0_x;
  real<lower=0> sigma_y_sq;
  
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
    matrix[N, N] L_K;
    matrix[N, N] K;
    
    // RBF Kernel
    for (i in 1:(N-1)) {
      K[i, i] = alpha^2 + 1e-9;
      for (j in (i+1):N) {
        K[i, j] = alpha^2 * exp(-0.5 * square((time[i] - time[j]) / rho));
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = alpha^2 + 1e-9;
    
    L_K = cholesky_decompose(K);
    X_true = L_K * z;
  }
}

model {
  // Priors (matched to bjlm)
  b0_int ~ normal(0, 10);
  b0_x ~ normal(0, 10);
  sigma_y_sq ~ inv_gamma(1.0, 1.0);
  
  alpha ~ lognormal(0.0, 1.0);
  rho ~ lognormal(0.0, 1.0);
  sigma_x ~ lognormal(-1.0, 1.0);
  
  z ~ std_normal();
  
  // Covariate model (unweighted GP observations)
  X_obs ~ normal(X_true, sigma_x);
  
  // Outcome model (weighted)
  real sigma_y = sqrt(sigma_y_sq);
  for (n in 1:N) {
    target += w[n] * normal_lpdf(Y[n] | b0_int + b0_x * X_true[n], sigma_y);
  }
}
