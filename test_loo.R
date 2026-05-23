tryCatch({
library(loo)
x1 <- list(
  estimates = matrix(c(10, 2), nrow=1, dimnames=list('elpd_loo', c('Estimate', 'SE'))),
  pointwise = matrix(rnorm(5), 5, 1, dimnames=list(NULL, 'elpd_loo'))
)
class(x1) <- c('bjlm_lfo', 'loo')
x2 <- list(
  estimates = matrix(c(12, 2.5), nrow=1, dimnames=list('elpd_loo', c('Estimate', 'SE'))),
  pointwise = matrix(rnorm(5), 5, 1, dimnames=list(NULL, 'elpd_loo'))
)
class(x2) <- c('bjlm_lfo', 'loo')
print(loo_compare(list(m1=x1, m2=x2)))
}, error = function(e) cat('ERROR:', e$message, '\n'))
