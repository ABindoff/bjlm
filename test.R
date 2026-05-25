library(dplyr)
devtools::load_all('.')
set.seed(42)
dat <- expand.grid(tau=1:10, id=1:10) |> mutate(X_obs=rnorm(100), Trt=rbinom(100, 1, 0.5), Y=rnbinom(100, 5, mu=10))
fit <- bjlm_model() |> 
  propensity(Trt~1, data=dat) |> 
  outcome(Y~tau, b0=~1+Trt+X_obs+(1|id), b1=~1, data=dat, family='negative_binomial') |> 
  compile() |> 
  fit(iter=10, warmup=5)
print(dimnames(fit$draws))
