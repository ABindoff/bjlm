library(dplyr)
devtools::load_all('.')
dat <- data.frame(time = 1:10, X_obs = rnorm(10), Y = rnorm(10), dummy_trt = rbinom(10, 1, 0.5), id = rep(1, 10))
fit <- bjlm_model() |> 
  propensity(dummy_trt ~ 1, data = dat) |> 
  outcome(Y ~ time, b0 = ~ 1 + X_obs, data = dat, family = 'gaussian') |> 
  compile() |> 
  fit(iter = 10)
