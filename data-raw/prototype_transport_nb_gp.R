# R prototype: conditional-transport sampler for latent GP + NB outcome.
# f = mu_f(theta; X_obs) + L_f(theta) z ; update (theta, b0, b0_gp, z).
# Validates: recovery + stability in the HIGH-COUNT regime, and that the X_obs
# sub-marginal is required (Fable's critical point).
set.seed(11)
log1pexp <- function(x) ifelse(x > 30, x, log1p(exp(x)))

## ---- simulate: high-count NB + GP ----
ns <- 30L; nt <- 8L; tau <- seq(0, 10, length.out = nt)
alpha_t <- 1.0; rho_t <- 3.0; sx_t <- 0.4; b0_t <- 0.4; b0gp_t <- 0.5; r_t <- 10
D <- as.matrix(dist(tau))
Rmat <- function(rho) exp(-0.5 * (D / rho)^2)
Kf_true <- alpha_t^2 * Rmat(rho_t) + diag(1e-8, nt)
Lt <- t(chol(Kf_true))
Xobs <- matrix(0, ns, nt); Y <- matrix(0L, ns, nt); Ftrue <- matrix(0, ns, nt)
for (j in 1:ns) {
  f <- as.numeric(Lt %*% rnorm(nt)); Ftrue[j, ] <- f
  Xobs[j, ] <- f + rnorm(nt, 0, sx_t)
  eta <- b0_t + b0gp_t * f
  Y[j, ] <- rnbinom(nt, size = r_t, mu = r_t * exp(eta))
}
cat(sprintf("counts: range [%d,%d] mean %.1f\n", min(Y), max(Y), mean(Y)))

## ---- resolution-aware rho prior (matches Rust) ----
gap <- median(diff(tau)); rng <- diff(range(tau))
rloc <- 0.5 * (log(gap) + log(rng)); rscale <- max(0.35, 0.25 * log(rng / gap))
lprior_theta <- function(a, rho, sx)
  -0.5 * log(a)^2 - 0.5 * ((log(rho) - rloc)/rscale)^2 - 0.5 * (log(sx) + 1)^2

## ---- per-subject GP-regression pieces given theta ----
gp_pieces <- function(a, rho, sx) {
  K <- a^2 * Rmat(rho) + diag(1e-8, nt); Kfull <- K + diag(sx^2, nt)
  cK <- chol(Kfull)                              # for sub-marginal + mu_f
  Kinv_full <- chol2inv(cK)
  muW <- K %*% Kinv_full                          # mu_f = muW %*% Xobs_j
  Sigf <- K - K %*% Kinv_full %*% K
  Lf <- t(chol(Sigf + diag(1e-9, nt)))
  logdet <- 2 * sum(log(diag(cK)))
  list(Kinv_full = Kinv_full, muW = muW, Lf = Lf, logdet = logdet)
}
submarg <- function(pc, xrow) as.numeric(-0.5 * (t(xrow) %*% pc$Kinv_full %*% xrow + pc$logdet + nt*log(2*pi)))
nb_ll <- function(f, yrow, b0, b0gp) { eta <- b0 + b0gp * f; sum(yrow*eta - (yrow + r_t)*log1pexp(eta)) }

## ---- sampler ----
run_transport <- function(niter = 1500, use_submarginal = TRUE) {
  a <- 1; rho <- 3; sx <- 0.5; b0 <- 0; b0gp <- 0.3
  Z <- matrix(rnorm(ns*nt), ns, nt)
  pc <- gp_pieces(a, rho, sx)
  keep <- matrix(0, niter, 4, dimnames = list(NULL, c("alpha","rho","sigma_x","b0_gp")))
  fmat <- function(pc) t(sapply(1:ns, function(j) pc$muW %*% Xobs[j,] + pc$Lf %*% Z[j,]))
  for (it in 1:niter) {
    ## theta MH (joint log-scale RW)
    ap <- exp(log(a)   + 0.08*rnorm(1)); rp <- exp(log(rho) + 0.10*rnorm(1)); sp <- exp(log(sx) + 0.08*rnorm(1))
    pcp <- tryCatch(gp_pieces(ap, rp, sp), error = function(e) NULL)
    if (!is.null(pcp)) {
      Fc <- fmat(pc); Fp <- fmat(pcp)
      lt_c <- lprior_theta(a, rho, sx); lt_p <- lprior_theta(ap, rp, sp)
      for (j in 1:ns) {
        lt_c <- lt_c + nb_ll(Fc[j,], Y[j,], b0, b0gp)
        lt_p <- lt_p + nb_ll(Fp[j,], Y[j,], b0, b0gp)
        if (use_submarginal) { lt_c <- lt_c + submarg(pc, Xobs[j,]); lt_p <- lt_p + submarg(pcp, Xobs[j,]) }
      }
      if (log(runif(1)) < lt_p - lt_c) { a<-ap; rho<-rp; sx<-sp; pc<-pcp }
    }
    ## z ESS per subject (prior N(0,I), NB likelihood at transported f)
    for (j in 1:ns) {
      muf <- pc$muW %*% Xobs[j,]
      ll <- function(z) nb_ll(as.numeric(muf + pc$Lf %*% z), Y[j,], b0, b0gp)
      z <- Z[j,]; nu <- rnorm(nt); logy <- ll(z) + log(runif(1))
      th <- runif(1, 0, 2*pi); tmn <- th - 2*pi; tmx <- th
      repeat { zp <- z*cos(th) + nu*sin(th)
        if (ll(zp) > logy) { Z[j,] <- zp; break }
        if (th < 0) tmn <- th else tmx <- th; th <- runif(1, tmn, tmx) }
    }
    ## b0, b0_gp MH (normal RW; b0_gp crosses zero)
    Fc <- fmat(pc)
    for (par in 1:2) {
      b0p <- b0; bgp <- b0gp
      if (par==1) b0p <- b0 + 0.05*rnorm(1) else bgp <- b0gp + 0.05*rnorm(1)
      lc <- -0.5*(b0/5)^2 - 0.5*(b0gp/2)^2; lp <- -0.5*(b0p/5)^2 - 0.5*(bgp/2)^2
      for (j in 1:ns) { lc <- lc + nb_ll(Fc[j,], Y[j,], b0, b0gp); lp <- lp + nb_ll(Fc[j,], Y[j,], b0p, bgp) }
      if (log(runif(1)) < lp - lc) { b0 <- b0p; b0gp <- bgp }
    }
    keep[it,] <- c(a, rho, sx, b0gp)
  }
  keep[(niter/2+1):niter, , drop = FALSE]
}

cat("\n== WITH sub-marginal (correct) ==\n")
post <- run_transport(1500, TRUE)
cat(sprintf("true: alpha=1.0 rho=3.0 sigma_x=0.40 b0_gp=%.2f\n", b0gp_t))
print(round(colMeans(post), 3))
cat("(no runaway if these are near truth)\n")

cat("\n== WITHOUT sub-marginal (Fable says: wrong) ==\n")
post2 <- run_transport(1500, FALSE)
print(round(colMeans(post2), 3))
