# regimes(): phase v0 — known-state level switching (desugars to a b0 factor).

test_that("regime emission/prior constructors build the right objects", {
  expect_s3_class(exact(), "regime_obs_model")
  expect_equal(exact()$type, "exact")
  expect_s3_class(confusion(diag = 8, offdiag = 1), "regime_obs_model")
  expect_equal(confusion()$type, "confusion")
  expect_error(confusion(diag = 0))
  expect_s3_class(regime_priors(), "regime_priors")
})

test_that("regimes() validates its arguments and appends a block", {
  m <- bjlm_model()
  expect_error(regimes(list(), name = "r", data = data.frame(s = 1), n_states = 2, obs_state = "s"),
               "bjlm_model")
  expect_error(regimes(m, name = "r", n_states = 2, obs_state = "s"), "data")
  d <- data.frame(s = c("A", "B"))
  expect_error(regimes(m, name = "r", data = d, n_states = 2), "obs_state")
  expect_error(regimes(m, name = "r", data = d, n_states = 2, obs_state = "s"), "time_var")
  expect_error(regimes(m, name = "r", data = d, n_states = 1, obs_state = "s",
                       time_var = "t", subject = "id"), "n_states")
  expect_error(regimes(m, name = "r", data = d, n_states = 2, obs_state = "s", obs_model = 1,
                       time_var = "t", subject = "id"), "obs_model")

  m2 <- regimes(m, name = "regime", data = d, n_states = 2, states = c("A", "B"),
                obs_state = "s", obs_model = exact(), ref_state = "A",
                time_var = "t", subject = "id")
  expect_length(m2$regimes, 1L)
  expect_equal(m2$regimes[[1]]$name, "regime")
})

.regime_data <- function(seed = 1, ns = 25L, nt = 6L) {
  set.seed(seed)
  subj <- rep(seq_len(ns), each = nt)
  time <- rep(seq(0, 5, length.out = nt), ns)
  state <- sample(c("A", "B", "C"), ns * nt, replace = TRUE)
  lev <- c(A = 0, B = 1.0, C = -0.5)[state]
  y <- 2 + 0.1 * time + lev + rnorm(ns * nt, 0, 0.4)
  data.frame(y = y, time = time, state = state, subj = factor(subj))
}

test_that("compile() desugars a known-state regime into a b0 factor", {
  dat <- .regime_data()
  cm <- bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, data = dat) |>
    regimes(name = "regime", data = dat, n_states = 3, states = c("A", "B", "C"),
            time_var = "time", subject = "subj", obs_state = "state",
            obs_model = exact(), ref_state = "A") |>
    compile()
  expect_true("state" %in% all.vars(cm$b0_formula))
  # the observed-state column is now a factor with the reference level first
  expect_true(is.factor(cm$model$outcome$data$state))
  expect_equal(levels(cm$model$outcome$data$state)[1], "A")
})

test_that("v0 capability gates fire", {
  dat <- .regime_data()
  reg <- function(...) bjlm_model() |> outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, data = dat) |>
    regimes(name = "r", data = dat, time_var = "time", subject = "subj", ...) |> compile()
  # confusion() with a b0/b1 outcome is now VALID (composes in-loop, v1c-b):
  # compiles to the latent-regime mode rather than erroring.
  expect_identical(reg(n_states = 3, states = c("A","B","C"), obs_state = "state",
                       obs_model = confusion(), ref_state = "A")$regime_mode, "v1b")
  # only the level switches in this phase
  expect_error(reg(n_states = 3, states = c("A","B","C"), obs_state = "state",
                   obs_model = exact(), switch = scale ~ 1, ref_state = "A"), "level only")
  # missing observed-state column
  expect_error(reg(n_states = 3, obs_state = "nope", obs_model = exact()), "not found")
  # bad reference state
  expect_error(reg(n_states = 3, states = c("A","B","C"), obs_state = "state",
                   obs_model = exact(), ref_state = "Z"), "ref_state")
})

test_that("a known-state regime fit recovers the level offsets", {
  skip_on_cran()
  dat <- .regime_data(seed = 2, ns = 30L)
  fit <- suppressMessages(
    bjlm_model() |>
      outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, data = dat) |>
      regimes(name = "regime", data = dat, n_states = 3, states = c("A","B","C"),
              time_var = "time", subject = "subj",
              obs_state = "state", obs_model = exact(), ref_state = "A") |>
      compile() |>
      fit(chains = 2L, iter = 800L, warmup = 400L, seed = 2L, verbose = FALSE))
  sm <- posterior::summarise_draws(fit$draws, "mean")
  gm <- function(v) sm$mean[sm$variable == v]
  expect_true("b0_stateB" %in% sm$variable && "b0_stateC" %in% sm$variable)
  expect_gt(gm("b0_stateB"), 0.4)     # truth +1.0
  expect_lt(gm("b0_stateC"), -0.1)    # truth -0.5
})

test_that("SBC certifies regime level recovery via custom functionals", {
  skip_on_cran()
  dat <- .regime_data(seed = 3, ns = 20L)
  cm <- bjlm_model() |>
    outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, data = dat) |>
    regimes(name = "regime", data = dat, n_states = 3, states = c("A","B","C"),
            time_var = "time", subject = "subj",
            obs_state = "state", obs_model = exact(), ref_state = "A") |>
    compile()
  res <- suppressMessages(sbc(cm, reps = 4L, iter = 200L, warmup = 100L, chains = 1L,
                              seed = 5L, rhat_threshold = 5, verbose = FALSE,
                              functionals = list(
                                stateB = function(p) p[["b0_stateB"]],
                                stateC = function(p) p[["b0_stateC"]])))
  expect_true(all(c("stateB", "stateC") %in% colnames(res$ranks)))
  expect_true(all(res$ranks >= 0 & res$ranks <= 1, na.rm = TRUE))
})

# Gillespie forward simulation of a homogeneous CTMC; state at obs_times.
.sim_ctmc_path <- function(q0, allowed, k, obs_times, s0 = 0L) {
  outr <- vector("list", k)
  for (i in seq_len(nrow(allowed))) {
    fr <- allowed[i, 1] + 1L; outr[[fr]] <- rbind(outr[[fr]], c(allowed[i, 2], q0[i]))
  }
  t <- 0; s <- s0; oi <- 1L; states <- integer(length(obs_times))
  repeat {
    r <- outr[[s + 1L]]; total <- if (is.null(r)) 0 else sum(r[, 2])
    t_next <- if (total <= 0) Inf else t + stats::rexp(1, total)
    while (oi <= length(obs_times) && obs_times[oi] < t_next) { states[oi] <- s; oi <- oi + 1L }
    if (oi > length(obs_times)) break
    if (!is.finite(t_next)) { while (oi <= length(obs_times)) { states[oi] <- s; oi <- oi + 1L }; break }
    s <- if (nrow(r) == 1) r[1, 1] else sample(r[, 1], 1, prob = r[, 2] / total); t <- t_next
  }
  states
}

test_that("regime transition intensities are estimated and merged into the fit (v1a)", {
  skip_on_cran()
  set.seed(4)
  k <- 3L; allowed <- rbind(c(0, 1), c(1, 0), c(1, 2), c(2, 1)); q0t <- c(0.4, 0.2, 0.3, 0.15)
  ns <- 50L; nt <- 8L; ot <- seq(0, 12, length.out = nt); sn <- c("S1", "S2", "S3")
  rows <- lapply(seq_len(ns), function(s) {
    st <- .sim_ctmc_path(q0t, allowed, k, ot, 0L)
    data.frame(y = 2 + c(0, 1, -0.5)[st + 1L] + rnorm(nt, 0, 0.4), time = ot,
               state = sn[st + 1L], subj = s)
  })
  dat <- do.call(rbind, rows); dat$subj <- factor(dat$subj)

  fit <- suppressMessages(
    bjlm_model() |>
      outcome(y ~ time, b0 = ~ 1, b1 = ~ 1, data = dat) |>
      regimes(name = "r", data = dat, n_states = 3, states = sn, time_var = "time",
              subject = "subj", obs_state = "state", obs_model = exact(),
              transition = ~ 1, ref_state = "S1") |>
      compile() |>
      fit(chains = 2L, iter = 1000L, warmup = 500L, seed = 4L, verbose = FALSE))

  vn <- posterior::variables(fit$draws)
  # intensity draws are merged alongside the level draws
  expect_true(all(c("q0_S1_S2", "q0_S2_S3", "q0_S1_S3", "b0_stateS2") %in% vn))
  expect_true(!is.null(fit$regime_names))
  sm <- posterior::summarise_draws(fit$draws, "mean")
  gm <- function(v) sm$mean[sm$variable == v]
  # real transitions carry rate; the never-simulated S1<->S3 shrink below them
  expect_gt(gm("q0_S1_S2"), 0.15)
  expect_lt(gm("q0_S1_S3"), gm("q0_S1_S2"))
})

test_that("CTMC intensity SBC certification (opt-in, slow)", {
  skip_on_cran()
  skip_if(!nzchar(Sys.getenv("BJLM_SBC_CERT")), "set BJLM_SBC_CERT=1 to run the CTMC SBC cert")
  frank <- function(dl, col, truth) {                       # ESS-thinned fractional rank
    np <- nrow(dl[[1]]); nc <- length(dl); m <- sapply(dl, function(d) d[, col])
    da <- posterior::as_draws_array(array(m, dim = c(np, nc, 1)))
    ess <- suppressWarnings(min(posterior::ess_bulk(da), posterior::ess_tail(da), na.rm = TRUE))
    v <- as.vector(m); if (!is.finite(ess) || ess < 2) ess <- length(v)
    mean(v[seq(1, length(v), by = max(1L, floor(length(v) / ess)))] < truth)
  }
  K <- 3L
  allowed <- do.call(rbind, lapply(0:(K - 1L), function(a)
    do.call(rbind, lapply(setdiff(0:(K - 1L), a), function(b) c(a, b)))))
  na <- nrow(allowed); lq0m <- log(0.5); lq0s <- 1.5; bs <- 1.0
  REPS <- 30L; ns <- 40L; nt <- 7L; ot <- seq(0, 10, length.out = nt)
  ranks <- matrix(NA_real_, REPS, 2 * na)
  for (rep in seq_len(REPS)) {
    set.seed(1000L + rep)
    q0 <- exp(rnorm(na, lq0m, lq0s)); beta <- rnorm(na, 0, bs)
    seg_dt <- c(); seg_iv <- c(); ifrom <- c(); ito <- c(); xr <- c(); iv <- 0L
    for (s in seq_len(ns)) {
      trt <- rbinom(1, 1, 0.5); q0e <- q0 * exp(beta * trt)
      st <- .sim_ctmc_path(q0e, allowed, K, ot, 0L)
      for (j in seq_len(nt - 1L)) {
        seg_dt <- c(seg_dt, ot[j+1] - ot[j]); seg_iv <- c(seg_iv, iv)
        ifrom <- c(ifrom, st[j]); ito <- c(ito, st[j+1]); xr <- c(xr, trt); iv <- iv + 1L
      }
    }
    res <- run_ctmc_mh(n_states = K, x_trans = as.double(xr), p_trans = 1L,
      seg_dt = seg_dt, seg_interval = as.integer(seg_iv), interval_from = as.integer(ifrom),
      interval_to = as.integer(ito), allowed_from = as.integer(allowed[,1]),
      allowed_to = as.integer(allowed[,2]), prior_logq0_mean = lq0m, prior_logq0_sd = lq0s,
      prior_beta_mean = 0, prior_beta_sd = bs, n_iter = 1500L, warmup = 750L, chains = 2L,
      seed = 1000L + rep, init_step = 0.4)
    for (i in seq_len(na)) {
      ranks[rep, i] <- frank(res$draws, i, q0[i])
      ranks[rep, na + i] <- frank(res$draws, na + i, beta[i])
    }
  }
  mr <- colMeans(ranks, na.rm = TRUE)
  expect_true(all(mr > 0.3 & mr < 0.7))          # ranks centred near uniform
  B <- 8L
  pv <- apply(ranks, 2, function(x) {
    x <- x[is.finite(x)]; h <- as.numeric(table(cut(x, seq(0, 1, length.out = B+1), include.lowest = TRUE)))
    stats::pchisq(sum((h - length(x)/B)^2 / (length(x)/B)), B - 1L, lower.tail = FALSE)
  })
  expect_gte(sum(pv > 0.05 / (2 * na)), 2 * na - 2)   # allow <=2 borderline flags of 12
})

# ===========================================================================
# v1b: latent-regime (misclassified indicator) joint FFBS fit
# ===========================================================================

# Simulate a latent CTMC path + misclassified indicator + Gaussian outcome.
.sim_hmm_data <- function(seed, ns = 60L, nt = 8L) {
  set.seed(seed)
  K <- 3L; allowed <- rbind(c(0,1), c(1,0), c(1,2), c(2,1))
  q0t <- c(0.4, 0.2, 0.3, 0.15); bqt <- c(0.8, 0, 0, 0)
  b0s <- c(0, 1.2, -0.7); intercept <- 2.0; trend <- 0.1; sig <- 0.4
  E <- matrix(0.075, K, K); diag(E) <- 0.85
  ot <- seq(0, 10, length.out = nt)
  rows <- lapply(seq_len(ns), function(s) {
    trt <- stats::rbinom(1, 1, 0.5)
    st  <- .sim_ctmc_path(q0t * exp(bqt * trt), allowed, K, ot, 0L)
    r   <- vapply(st, function(z) sample(0:(K-1), 1, prob = E[z+1, ]), integer(1))
    data.frame(id = s, time = ot, y = intercept + trend * ot + b0s[st+1] + rnorm(nt, 0, sig),
               r_obs = r, trt = trt)
  })
  do.call(rbind, rows)
}

test_that("v1b entry gates fire", {
  dat <- .sim_hmm_data(seed = 1, ns = 5L)
  reg_v1b <- function(oc) suppressMessages(
    oc(bjlm_model()) |>
      regimes(name = "r", data = dat, n_states = 3L, time_var = "time", subject = "id",
              obs_state = "r_obs", obs_model = confusion(), transition = ~ trt) |>
      compile())
  # a change-point may COMPOSE with the regime (v1c-b, fit in-loop): valid compile
  expect_identical(reg_v1b(function(m) outcome(m, y ~ time, b0 = ~1, b1 = ~1, data = dat))$regime_mode, "v1b")
  # gaussian / binomial / negative_binomial all route to the v1b latent-regime mode
  for (fm in list(gaussian(), binomial(), "negbin")) {
    cm <- reg_v1b(function(m) outcome(m, y ~ time, data = dat, family = fm))
    expect_identical(cm$regime_mode, "v1b")
  }
  # multiple confusion blocks not yet supported
  expect_error(suppressMessages(
    bjlm_model() |> outcome(y ~ time, data = dat) |>
      regimes(name = "a", data = dat, n_states = 3L, time_var = "time", subject = "id",
              obs_state = "r_obs", obs_model = confusion()) |>
      regimes(name = "b", data = dat, n_states = 3L, time_var = "time", subject = "id",
              obs_state = "r_obs", obs_model = confusion()) |>
      compile()), "single")
  # exact() and confusion() cannot be mixed
  expect_error(suppressMessages(
    bjlm_model() |> outcome(y ~ time, data = dat) |>
      regimes(name = "a", data = dat, n_states = 3L, time_var = "time", subject = "id",
              obs_state = "r_obs", obs_model = exact()) |>
      regimes(name = "b", data = dat, n_states = 3L, time_var = "time", subject = "id",
              obs_state = "r_obs", obs_model = confusion()) |>
      compile()), "mix")
})

test_that("latent-regime (confusion) fit recovers levels, intensities and misclassification (v1b)", {
  skip_on_cran()
  dat <- .sim_hmm_data(seed = 11, ns = 120L)
  dat <- dat[sample(nrow(dat)), ]                     # prove (subject,time) re-ordering
  fit <- suppressMessages(
    bjlm_model() |>
      outcome(y ~ time, data = dat, family = gaussian()) |>
      regimes(name = "regime", data = dat, n_states = 3L, time_var = "time", subject = "id",
              obs_state = "r_obs", obs_model = confusion(diag = 8, offdiag = 1),
              transition = ~ trt, priors = regime_priors(level = prior_normal(0, 5))) |>
      compile() |>
      fit(chains = 2L, iter = 1200L, warmup = 600L, seed = 5L, verbose = FALSE))

  expect_s3_class(fit, "bjlm_regime_fit")
  sm <- posterior::summarise_draws(fit$draws, "mean")
  gm <- function(v) sm$mean[sm$variable == v]
  # outcome level model
  expect_equal(gm("b_(Intercept)"), 2.0, tolerance = 0.3)
  expect_equal(gm("b_time"), 0.1, tolerance = 0.05)
  expect_equal(gm("b0_state_1"), 1.2, tolerance = 0.4)   # state names inferred: "0","1","2"
  expect_equal(gm("b0_state_2"), -0.7, tolerance = 0.4)
  expect_equal(gm("sigma"), 0.4, tolerance = 0.15)
  # base intensities recover; misclassification diagonal is dominant
  expect_gt(gm("q0_0_1"), 0.2)
  expect_gt(gm("E_0_0"), 0.7); expect_gt(gm("E_1_1"), 0.7); expect_gt(gm("E_2_2"), 0.7)
})

test_that("latent-regime FFBS SBC certification (opt-in, slow)", {
  skip_on_cran()
  skip_if(!nzchar(Sys.getenv("BJLM_SBC_CERT")), "set BJLM_SBC_CERT=1 to run the FFBS SBC cert")
  rdir <- function(a) { g <- stats::rgamma(length(a), a, 1); g / sum(g) }
  frank <- function(dl, col, truth) {
    np <- nrow(dl[[1]]); nc <- length(dl); m <- sapply(dl, function(d) d[, col])
    da <- posterior::as_draws_array(array(m, dim = c(np, nc, 1)))
    ess <- suppressWarnings(min(posterior::ess_bulk(da), posterior::ess_tail(da), na.rm = TRUE))
    v <- as.vector(m); if (!is.finite(ess) || ess < 2) ess <- length(v)
    mean(v[seq(1, length(v), by = max(1L, floor(length(v) / ess)))] < truth)
  }
  K <- 3L; allowed <- rbind(c(0,1), c(1,0), c(1,2), c(2,1)); na <- nrow(allowed)
  pb_sd <- 3; pb0_sd <- 3; a_sig <- 3; b_sig <- 1
  lq0m <- log(0.4); lq0s <- 0.8; bq_sd <- 0.6; ed <- 12; eo <- 1
  # ranked functionals: intercept, trend, b0_1, b0_2, sigma, q0[4], E_00,E_11,E_22
  REPS <- 40L; ns <- 60L; nt <- 8L; ot <- seq(0, 10, length.out = nt)
  nf <- 5L + na + 3L; ranks <- matrix(NA_real_, REPS, nf)
  for (rep in seq_len(REPS)) {
    set.seed(7000L + rep)
    beta  <- rnorm(2, 0, pb_sd)                       # intercept, trend
    b0f   <- rnorm(2, 0, pb0_sd)                      # state 1,2 offsets (state 0 = 0)
    sigma <- 1 / sqrt(stats::rgamma(1, a_sig, b_sig))
    q0    <- exp(rnorm(na, lq0m, lq0s)); bq <- rnorm(na, 0, bq_sd)
    Erows <- t(vapply(seq_len(K), function(k) rdir(ifelse(seq_len(K) == k, ed, eo)), numeric(K)))
    pi_t  <- rdir(rep(1, K))                          # draw the initial distribution too
    b0s   <- c(0, b0f)
    Y <- c(); XF <- c(); XT <- c(); OS <- c(); OSUB <- c(); OT <- c()
    for (s in seq_len(ns)) {
      trt <- rbinom(1, 1, 0.5); s0 <- sample(0:(K-1), 1, prob = pi_t)
      st <- .sim_ctmc_path(q0 * exp(bq * trt), allowed, K, ot, s0)
      r <- vapply(st, function(z) sample(0:(K-1), 1, prob = Erows[z+1, ]), integer(1))
      Y <- c(Y, beta[1] + beta[2]*ot + b0s[st+1] + rnorm(nt, 0, sigma))
      XF <- rbind(XF, cbind(1, ot)); XT <- c(XT, rep(trt, nt))
      OS <- c(OS, r); OSUB <- c(OSUB, rep(s-1L, nt)); OT <- c(OT, ot)
    }
    res <- run_regime_hmm(n_states = K, n_cat = K, family = 0L, y = as.double(Y),
      n_trials = as.double(rep(1, length(Y))),
      x_fixed = as.double(XF), p_fixed = 2L, x_trans = as.double(XT), p_trans = 1L,
      obs_state = as.integer(OS), obs_subj = as.integer(OSUB), obs_time = as.double(OT),
      allowed_from = as.integer(allowed[,1]), allowed_to = as.integer(allowed[,2]),
      prior_beta_sd = pb_sd, prior_b0_sd = pb0_sd, sigma_shape = a_sig, sigma_scale = b_sig,
      r_init = 8, r_shape = 2, r_rate = 0.2,
      e_diag = ed, e_offdiag = eo, prior_logq0_mean = lq0m, prior_logq0_sd = lq0s,
      prior_beta_q_sd = bq_sd, n_iter = 1500L, warmup = 750L, chains = 2L,
      seed = 7000L + rep, init_step = 0.4)
    truths <- c(beta, b0f, sigma, q0, diag(Erows))
    # draw column order: beta[2] b0[K-1] sigma q0[na] beta_q[na] E[K*K row-major] pi[K]
    ecol0  <- 5L + na + na                                        # last col before E block
    ecols  <- c(ecol0 + 1L, ecol0 + K + 2L, ecol0 + 2L*K + 3L)    # E_00, E_11, E_22
    cols   <- c(1, 2, 3, 4, 5, 6:(5+na), ecols)
    for (i in seq_len(nf)) ranks[rep, i] <- frank(res$draws, cols[i], truths[i])
  }
  mr <- colMeans(ranks, na.rm = TRUE)
  expect_true(all(mr > 0.25 & mr < 0.75))
  B <- 8L
  pv <- apply(ranks, 2, function(x) {
    x <- x[is.finite(x)]; h <- as.numeric(table(cut(x, seq(0, 1, length.out = B+1), include.lowest = TRUE)))
    stats::pchisq(sum((h - length(x)/B)^2 / (length(x)/B)), B - 1L, lower.tail = FALSE)
  })
  expect_gte(sum(pv > 0.05 / nf), nf - 2L)             # allow <=2 borderline flags
})

# ===========================================================================
# v1c-a: non-Gaussian latent-regime outcomes (Binomial / NegBin) via PG-FFBS
# ===========================================================================

test_that("latent-regime fit recovers non-Gaussian outcomes (binomial, negbin)", {
  skip_on_cran()
  K <- 3L; allowed <- rbind(c(0,1), c(1,0), c(1,2), c(2,1)); q0t <- c(0.4,0.2,0.3,0.15)
  Et <- matrix(0.05, K, K); diag(Et) <- 0.9; nt <- 10L; ot <- seq(0, 10, length.out = nt)
  mk <- function(seed, ns, gen) {
    set.seed(seed)
    do.call(rbind, lapply(seq_len(ns), function(s) {
      trt <- rbinom(1, 1, 0.5); st <- .sim_ctmc_path(q0t, allowed, K, ot, 0L)
      r <- vapply(st, function(z) sample(0:(K-1), 1, prob = Et[z+1, ]), integer(1))
      data.frame(id = s, time = ot, r_obs = r, trt = trt, y = gen(st, ot))
    }))
  }

  # --- Negative binomial (log link): recover levels, trend and dispersion r ---
  nd <- mk(21, 200L, function(st, ot) rnbinom(length(st), size = 8, mu = exp(1.5 + 0.05*ot + c(0,0.8,-0.6)[st+1])))
  nf <- suppressMessages(
    bjlm_model() |> outcome(y ~ time, data = nd, family = "negbin") |>
      regimes(name = "r", data = nd, n_states = 3L, time_var = "time", subject = "id",
              obs_state = "r_obs", obs_model = confusion(), transition = ~ trt) |>
      compile() |> fit(chains = 2L, iter = 1200L, warmup = 600L, seed = 3L, verbose = FALSE))
  sm <- posterior::summarise_draws(nf$draws, "mean"); gm <- function(v) sm$mean[sm$variable == v]
  expect_true("r" %in% sm$variable)                     # NB dispersion named "r"
  expect_equal(gm("b_(Intercept)"), 1.5, tolerance = 0.3)
  expect_equal(gm("b0_state_1"), 0.8, tolerance = 0.4)
  expect_equal(gm("b0_state_2"), -0.6, tolerance = 0.4)
  expect_gt(gm("r"), 3); expect_lt(gm("r"), 20)

  # --- Binomial (logit link): recover the state-dependent log-odds offsets ---
  bd <- mk(22, 260L, function(st, ot) rbinom(length(st), 1, plogis(0.2 + c(0,1.5,-1.5)[st+1])))
  bf <- suppressMessages(
    bjlm_model() |> outcome(y ~ 1, data = bd, family = binomial()) |>
      regimes(name = "r", data = bd, n_states = 3L, time_var = "time", subject = "id",
              obs_state = "r_obs", obs_model = confusion()) |>
      compile() |> fit(chains = 2L, iter = 1200L, warmup = 600L, seed = 4L, verbose = FALSE))
  sm <- posterior::summarise_draws(bf$draws, "mean"); gm <- function(v) sm$mean[sm$variable == v]
  expect_false("sigma" %in% sm$variable)                # no dispersion for binomial
  expect_gt(gm("b0_state_1"), 0.6)                      # truth +1.5 (logit-scale attenuation ok)
  expect_lt(gm("b0_state_2"), -0.6)                     # truth -1.5
})

test_that("in-loop regime engine (v1c-b) reproduces the isolated v1b fit", {
  skip_on_cran()
  dat <- .sim_hmm_data(seed = 11, ns = 120L)
  mk <- function() bjlm_model() |>
    outcome(y ~ time, data = dat, family = gaussian()) |>
    regimes(name = "regime", data = dat, n_states = 3L, time_var = "time", subject = "id",
            obs_state = "r_obs", obs_model = confusion(diag = 8, offdiag = 1), transition = ~ trt)
  gm <- function(fit) { sm <- posterior::summarise_draws(fit$draws, "mean"); function(v) sm$mean[sm$variable == v] }

  fi <- suppressMessages(mk() |> compile() |> fit(chains = 2L, iter = 1000L, warmup = 500L, seed = 5L, verbose = FALSE))
  gi <- gm(fi)
  fc <- withr::with_envvar(c(BJLM_REGIME_INLOOP = "1"),
    suppressMessages(mk() |> compile() |> fit(chains = 2L, iter = 1000L, warmup = 500L, seed = 5L, verbose = FALSE)))
  gc <- gm(fc)

  expect_s3_class(fc, "bjlm_fit")                       # in-loop returns a full bjlm_fit
  vn <- posterior::variables(fc$draws)
  expect_true(all(c("b0_state_1","b0_state_2","q0_0_1","E_0_0","pi_0") %in% vn))
  # the two engines agree (same FFBS math, different orchestration)
  for (v in c("b0_state_1","b0_state_2","E_0_0","E_1_1","E_2_2")) {
    expect_equal(gc(v), gi(v), tolerance = 0.12,
                 info = sprintf("%s: in-loop %.3f vs isolated %.3f", v, gc(v), gi(v)))
  }
  # recovery of the levels (truth 1.2 / -0.7)
  expect_equal(gc("b0_state_1"), 1.2, tolerance = 0.4)
  expect_equal(gc("b0_state_2"), -0.7, tolerance = 0.4)
})

test_that("in-loop regime COMPOSES with a smoothed change-point (v1c-b)", {
  skip_on_cran()
  sig <- function(x) 1 / (1 + exp(-x))
  K <- 3L; allowed <- rbind(c(0,1), c(1,0), c(1,2), c(2,1)); q0t <- c(0.4,0.2,0.3,0.15)
  b0s <- c(0, 1.0, -0.8); Et <- matrix(0.05, K, K); diag(Et) <- 0.9
  b0 <- 2; b1 <- 0.2; delta <- -0.5; om <- 5; rho <- 2.0; sg <- 0.4
  nt <- 8L; ot <- seq(0, 10, length.out = nt)
  set.seed(41); rows <- list()
  for (s in seq_len(130L)) {
    st <- .sim_ctmc_path(q0t, allowed, K, ot, 0L)
    r  <- vapply(st, function(z) sample(0:(K-1), 1, prob = Et[z+1, ]), integer(1))
    d  <- ot - om
    rows[[s]] <- data.frame(subject = s, tau = ot,
                            y = b0 + b1*d + delta*d*sig(rho*d) + b0s[st+1] + rnorm(nt, 0, sg), r_obs = r)
  }
  dat <- do.call(rbind, rows)
  fit <- suppressMessages(
    bjlm_model() |>
      outcome(y ~ tau, b0 = ~ 1, b1 = ~ 1, deltas = list(~ 1), omega = list(~ 1), rho = list(~ 1),
              data = dat, family = gaussian()) |>
      regimes(name = "r", data = dat, n_states = 3L, time_var = "tau", subject = "subject",
              obs_state = "r_obs", obs_model = confusion(diag = 8, offdiag = 1)) |>
      compile() |> fit(chains = 2L, iter = 1200L, warmup = 600L, seed = 5L, verbose = FALSE))

  expect_s3_class(fit, "bjlm_fit")
  sm <- posterior::summarise_draws(fit$draws, "mean"); gm <- function(v) sm$mean[sm$variable == v]
  # change-point recovered (location + slope-change) alongside the regime
  expect_equal(gm("omega1_(Intercept)"), 5.0, tolerance = 1.0)
  expect_equal(gm("delta1_(Intercept)"), -0.5, tolerance = 0.3)
  # regime levels + misclassification NOT absorbed by the change-point
  expect_equal(gm("b0_state_1"), 1.0, tolerance = 0.4)
  expect_equal(gm("b0_state_2"), -0.8, tolerance = 0.4)
  expect_gt(gm("E_0_0"), 0.75); expect_gt(gm("E_1_1"), 0.75)
})

test_that("in-loop regime COMPOSES with a latent GP confounder (v1c-b)", {
  skip_on_cran()
  dat <- simulate_bjlm(n_subj = 60L, n_obs = 8L, b0 = 2, b0_trt = 0, b1 = -0.3,
    omegas = c(5), rhos = c(4), deltas_int = c(-0.4), deltas_trt = 0, sigma = 0.4,
    sigma_u = 0, gp_confounder = TRUE, gp_alpha = 1.0, gp_rho = 3.0, gp_sigma_x = 0.5,
    b0_gp = 1.2, trt_gp = 0, seed = 51)
  K <- 3L; allowed <- rbind(c(0,1), c(1,0), c(1,2), c(2,1)); q0t <- c(0.4,0.2,0.3,0.15)
  b0s <- c(0, 1.5, -1.2); Et <- matrix(0.05, K, K); diag(Et) <- 0.9
  set.seed(7); dat$r_obs <- NA_integer_
  for (sj in unique(dat$subject)) {
    idx <- which(dat$subject == sj); ot <- dat$tau[idx]
    st <- .sim_ctmc_path(q0t, allowed, K, ot, 0L)
    dat$y[idx] <- dat$y[idx] + b0s[st + 1]
    dat$r_obs[idx] <- vapply(st, function(z) sample(0:(K-1), 1, prob = Et[z+1, ]), integer(1))
  }
  fit <- suppressMessages(
    bjlm_model() |>
      outcome(y ~ tau, b0 = ~ 1 + X_obs, b1 = ~ 1, deltas = list(~ 1), omega = list(~ 1),
              rho = list(~ 1), data = dat, family = gaussian()) |>
      latent_gp(name = "X_obs", data = dat, time_var = "tau", obs_var = "X_obs",
                subject = "subject", time_out_var = "tau", time_trt_var = "tau") |>
      regimes(name = "regime", data = dat, n_states = 3L, time_var = "tau", subject = "subject",
              obs_state = "r_obs", obs_model = confusion(diag = 8, offdiag = 1)) |>
      compile() |> fit(chains = 2L, iter = 1200L, warmup = 600L, seed = 5L, verbose = FALSE))

  expect_s3_class(fit, "bjlm_fit")
  sm <- posterior::summarise_draws(fit$draws, "mean"); gm <- function(v) sm$mean[sm$variable == v]
  vn <- posterior::variables(fit$draws)
  expect_true(all(c("X_obs_alpha","X_obs_rho","X_obs_sigma_x","b0_X_obs","b0_state_1") %in% vn))
  # GP hypers + loading recover with the regime composed on top
  expect_equal(gm("b0_X_obs"), 1.2, tolerance = 0.4)
  expect_equal(gm("X_obs_rho"), 3.0, tolerance = 1.5)
  expect_equal(gm("X_obs_sigma_x"), 0.5, tolerance = 0.25)
  # regime levels + misclassification recover, not absorbed by the GP
  expect_equal(gm("b0_state_1"), 1.5, tolerance = 0.5)
  expect_equal(gm("b0_state_2"), -1.2, tolerance = 0.5)
  expect_gt(gm("E_0_0"), 0.75)
})

test_that("in-loop regime composes with a latent GP + NON-Gaussian outcome (v1c-b)", {
  skip_on_cran()
  sig <- function(x) 1/(1+exp(-x))
  K <- 3L; allowed <- rbind(c(0,1), c(1,0), c(1,2), c(2,1)); q0t <- c(0.4,0.2,0.3,0.15)
  Et <- matrix(0.05, K, K); diag(Et) <- 0.9; nt <- 8L; ot <- seq(0,10,length.out=nt)
  gen <- function(seed, ns, family, b0, b0_gp, b0s, rr = 8) {
    set.seed(seed); rows <- list()
    for (s in seq_len(ns)) {
      D <- as.matrix(dist(ot)); Kmat <- 1.0*exp(-0.5*(D/3.0)^2) + diag(1e-8, nt)
      f <- as.numeric(t(chol(Kmat)) %*% rnorm(nt)); X_obs <- f + rnorm(nt, 0, 0.5)
      st <- .sim_ctmc_path(q0t, allowed, K, ot, 0L)
      r_obs <- vapply(st, function(z) sample(0:(K-1), 1, prob = Et[z+1, ]), integer(1))
      eta <- b0 + b0_gp*f + b0s[st+1]
      y <- if (family == "negbin") rnbinom(nt, size=rr, mu=exp(eta)) else rbinom(nt, 1, sig(eta))
      rows[[s]] <- data.frame(subject=s, tau=ot, y=y, X_obs=X_obs, r_obs=r_obs)
    }
    do.call(rbind, rows)
  }
  fitit <- function(df, family) suppressMessages(
    bjlm_model() |>
      outcome(y ~ tau, b0 = ~ 1 + X_obs, b1 = ~ 1, data = df, family = family) |>
      latent_gp(name = "X_obs", data = df, time_var = "tau", obs_var = "X_obs",
                subject = "subject", time_out_var = "tau", time_trt_var = "tau") |>
      regimes(name = "r", data = df, n_states = 3L, time_var = "tau", subject = "subject",
              obs_state = "r_obs", obs_model = confusion(diag = 8, offdiag = 1)) |>
      compile() |> fit(chains = 2L, iter = 1000L, warmup = 500L, seed = 5L, cores = 2L, verbose = FALSE))
  gm <- function(fit) { sm <- posterior::summarise_draws(fit$draws, "mean"); function(v) sm$mean[sm$variable == v] }

  # Negative binomial: GP hypers + NB dispersion + regime all recover together
  fn <- fitit(gen(21, 200L, "negbin", b0 = 1.6, b0_gp = 0.6, b0s = c(0,0.8,-0.7)), "negbin")
  g <- gm(fn); vn <- posterior::variables(fn$draws)
  expect_s3_class(fn, "bjlm_fit"); expect_true("r" %in% vn)
  expect_equal(g("X_obs_rho"), 3.0, tolerance = 1.5)
  expect_equal(g("X_obs_sigma_x"), 0.5, tolerance = 0.25)
  expect_equal(g("b0_X_obs"), 0.6, tolerance = 0.4)
  expect_gt(g("r"), 3); expect_lt(g("r"), 20)
  expect_equal(g("b0_state_1"), 0.8, tolerance = 0.5)
  expect_gt(g("E_0_0"), 0.75)

  # Binomial: GP hypers + regime recover on the logit scale
  fb <- fitit(gen(22, 280L, "binomial", b0 = 0, b0_gp = 1.0, b0s = c(0,1.6,-1.6)), "binomial")
  g <- gm(fb)
  expect_equal(g("X_obs_rho"), 3.0, tolerance = 1.5)
  expect_equal(g("X_obs_sigma_x"), 0.5, tolerance = 0.25)
  expect_gt(g("b0_state_1"), 0.7); expect_lt(g("b0_state_2"), -0.7)
})

test_that("composed three-latent (regime + GP + change-point) coverage SBC (opt-in, slow)", {
  skip_on_cran()
  skip_if(!nzchar(Sys.getenv("BJLM_SBC_CERT")), "set BJLM_SBC_CERT=1 to run the coverage cert")
  K <- 3L; allowed <- rbind(c(0,1), c(1,0), c(1,2), c(2,1)); q0t <- c(0.4,0.2,0.3,0.15)
  b0s <- c(0, 1.5, -1.2); Et <- matrix(0.05, K, K); diag(Et) <- 0.9
  TR <- c(`delta1_(Intercept)` = -0.4, b0_X_obs = 1.2, X_obs_alpha = 1.0, X_obs_rho = 3.0,
          X_obs_sigma_x = 0.5, b0_state_1 = 1.5, b0_state_2 = -1.2, E_0_0 = 0.9, E_1_1 = 0.9,
          E_2_2 = 0.9, sigma = 0.4)                       # omega excluded (mixing, not aliasing)
  fp <- bjlm_priors(outcome = smoothbp_priors(
    b0 = prior_normal(0, 3), b1 = prior_normal(0, 1), deltas = prior_normal(0, 1),
    omega = prior_normal(5, 2, lb = 0.5, ub = 9.5), rho = prior_normal(4, 2, lb = 1, ub = 10),
    sigma = prior_invgamma(3, 1)))
  REPS <- 24L; hit90 <- matrix(NA, REPS, length(TR), dimnames = list(NULL, names(TR)))
  for (rep in seq_len(REPS)) {
    dat <- tryCatch(simulate_bjlm(n_subj = 45L, n_obs = 7L, b0 = 2, b0_trt = 0, b1 = -0.3,
      omegas = c(5), rhos = c(4), deltas_int = c(-0.4), deltas_trt = 0, sigma = 0.4, sigma_u = 0,
      gp_confounder = TRUE, gp_alpha = 1.0, gp_rho = 3.0, gp_sigma_x = 0.5, b0_gp = 1.2,
      trt_gp = 0, seed = 3000 + rep), error = function(e) NULL)
    if (is.null(dat)) next
    set.seed(9000 + rep); dat$r_obs <- NA_integer_
    for (sj in unique(dat$subject)) {
      idx <- which(dat$subject == sj); ot <- dat$tau[idx]
      st <- .sim_ctmc_path(q0t, allowed, K, ot, 0L)
      dat$y[idx] <- dat$y[idx] + b0s[st + 1]
      dat$r_obs[idx] <- vapply(st, function(z) sample(0:(K-1), 1, prob = Et[z+1, ]), integer(1))
    }
    fit <- tryCatch(suppressMessages(
      bjlm_model() |>
        outcome(y ~ tau, b0 = ~ 1 + X_obs, b1 = ~ 1, deltas = list(~ 1), omega = list(~ 1),
                rho = list(~ 1), data = dat, family = gaussian()) |>
        latent_gp(name = "X_obs", data = dat, time_var = "tau", obs_var = "X_obs",
                  subject = "subject", time_out_var = "tau", time_trt_var = "tau") |>
        regimes(name = "r", data = dat, n_states = 3L, time_var = "tau", subject = "subject",
                obs_state = "r_obs", obs_model = confusion(diag = 8, offdiag = 1)) |>
        compile() |>
        fit(priors = fp, chains = 2L, iter = 1000L, warmup = 500L, seed = 5 + rep, cores = 2L, verbose = FALSE)),
      error = function(e) NULL)
    if (is.null(fit)) next
    q <- posterior::summarise_draws(fit$draws, ~quantile(.x, c(0.05, 0.95)))
    for (v in names(TR)) {
      r <- q[q$variable == v, ]
      if (nrow(r)) hit90[rep, v] <- TR[[v]] >= r$`5%` && TR[[v]] <= r$`95%`
    }
  }
  cov90 <- colMeans(hit90, na.rm = TRUE)
  # at ~24 reps the 90%-coverage SE is ~0.06; require all identifiability params
  # within ~2.5 SE of nominal (allow one borderline for sampling noise)
  expect_gte(sum(cov90 >= 0.75), length(TR) - 1L)
})

test_that("state_occupancy() returns Rao-Blackwellized smoothed state probabilities", {
  skip_on_cran()
  K <- 3L; allowed <- rbind(c(0,1), c(1,0), c(1,2), c(2,1)); q0t <- c(0.4,0.2,0.3,0.15)
  b0s <- c(0, 1.5, -1.2); Et <- matrix(0.05, K, K); diag(Et) <- 0.9
  nt <- 8L; ot <- seq(0, 10, length.out = nt)
  set.seed(31); rows <- list(); truth <- c()
  for (s in seq_len(70L)) {
    st <- .sim_ctmc_path(q0t, allowed, K, ot, 0L)
    r  <- vapply(st, function(z) sample(0:(K-1), 1, prob = Et[z+1, ]), integer(1))
    rows[[s]] <- data.frame(id = s, time = ot, y = 2 + 0.1*ot + b0s[st+1] + rnorm(nt, 0, 0.4),
                            r_obs = r, truth = st)
  }
  dat <- do.call(rbind, rows); dat <- dat[sample(nrow(dat)), ]     # shuffle -> tests re-order map
  fit <- suppressMessages(
    bjlm_model() |> outcome(y ~ time, data = dat, family = gaussian()) |>
      regimes(name = "r", data = dat, n_states = 3L, time_var = "time", subject = "id",
              obs_state = "r_obs", obs_model = confusion()) |>
      compile() |> fit(chains = 2L, iter = 1000L, warmup = 500L, seed = 5L, verbose = FALSE))

  occ <- state_occupancy(fit)
  pc <- paste0("p_", c("0","1","2"))
  expect_true(all(c("id","time",pc,"modal_state") %in% names(occ)))
  expect_equal(nrow(occ), nrow(dat))
  expect_equal(unname(rowSums(as.matrix(occ[, pc]))), rep(1, nrow(dat)), tolerance = 1e-8)
  # occupancy is in ORIGINAL (shuffled) data order -> aligns with dat$truth
  modal_idx <- max.col(as.matrix(occ[, pc]), ties.method = "first") - 1L
  acc <- mean(modal_idx == dat$truth)
  expect_gt(acc, 0.9)                       # smoothing beats the ~0.9 raw indicator
})

test_that("negative-binomial FFBS SBC certification (opt-in, slow)", {
  skip_on_cran()
  skip_if(!nzchar(Sys.getenv("BJLM_SBC_CERT")), "set BJLM_SBC_CERT=1 to run the NB FFBS SBC cert")
  rdir <- function(a) { g <- stats::rgamma(length(a), a, 1); g / sum(g) }
  frank <- function(dl, col, truth) {
    np <- nrow(dl[[1]]); nc <- length(dl); m <- sapply(dl, function(d) d[, col])
    da <- posterior::as_draws_array(array(m, dim = c(np, nc, 1)))
    ess <- suppressWarnings(min(posterior::ess_bulk(da), posterior::ess_tail(da), na.rm = TRUE))
    v <- as.vector(m); if (!is.finite(ess) || ess < 2) ess <- length(v)
    mean(v[seq(1, length(v), by = max(1L, floor(length(v) / ess)))] < truth)
  }
  K <- 3L; allowed <- rbind(c(0,1), c(1,0), c(1,2), c(2,1)); na <- nrow(allowed)
  # tighter priors keep the log-mean (hence counts) in a sane range for draw==fit
  pb_sd <- 0.5; pb0_sd <- 0.7; r_shape <- 4; r_rate <- 0.5
  lq0m <- log(0.4); lq0s <- 0.8; bq_sd <- 0.6; ed <- 12; eo <- 1
  REPS <- 40L; ns <- 80L; nt <- 8L; ot <- seq(0, 10, length.out = nt)
  # functionals: intercept, b0_1, b0_2, r, q0[na], E_00,E_11,E_22
  nf <- 4L + na + 3L; ranks <- matrix(NA_real_, REPS, nf)
  for (rep in seq_len(REPS)) {
    set.seed(9000L + rep)
    b0    <- rnorm(1, 0, pb_sd)                         # intercept only (no trend)
    b0f   <- rnorm(2, 0, pb0_sd)
    r_t   <- stats::rgamma(1, r_shape, rate = r_rate)
    q0    <- exp(rnorm(na, lq0m, lq0s)); bq <- rnorm(na, 0, bq_sd)
    Erows <- t(vapply(seq_len(K), function(k) rdir(ifelse(seq_len(K) == k, ed, eo)), numeric(K)))
    pi_t  <- rdir(rep(1, K))                          # draw the initial distribution too
    b0s   <- c(0, b0f)
    Y <- c(); XF <- c(); XT <- c(); OS <- c(); OSUB <- c(); OT <- c()
    for (s in seq_len(ns)) {
      trt <- rbinom(1, 1, 0.5); s0 <- sample(0:(K-1), 1, prob = pi_t)
      st <- .sim_ctmc_path(q0 * exp(bq * trt), allowed, K, ot, s0)
      r <- vapply(st, function(z) sample(0:(K-1), 1, prob = Erows[z+1, ]), integer(1))
      Y <- c(Y, rnbinom(nt, size = r_t, mu = exp(b0 + b0s[st+1])))
      XF <- rbind(XF, matrix(1, nt, 1)); XT <- c(XT, rep(trt, nt))
      OS <- c(OS, r); OSUB <- c(OSUB, rep(s-1L, nt)); OT <- c(OT, ot)
    }
    res <- run_regime_hmm(n_states = K, n_cat = K, family = 2L, y = as.double(Y),
      n_trials = as.double(rep(1, length(Y))),
      x_fixed = as.double(XF), p_fixed = 1L, x_trans = as.double(XT), p_trans = 1L,
      obs_state = as.integer(OS), obs_subj = as.integer(OSUB), obs_time = as.double(OT),
      allowed_from = as.integer(allowed[,1]), allowed_to = as.integer(allowed[,2]),
      prior_beta_sd = pb_sd, prior_b0_sd = pb0_sd, sigma_shape = 2, sigma_scale = 1,
      r_init = r_shape / r_rate, r_shape = r_shape, r_rate = r_rate,
      e_diag = ed, e_offdiag = eo, prior_logq0_mean = lq0m, prior_logq0_sd = lq0s,
      prior_beta_q_sd = bq_sd, n_iter = 1500L, warmup = 750L, chains = 2L,
      seed = 9000L + rep, init_step = 0.4)
    truths <- c(b0, b0f, r_t, q0, diag(Erows))
    # p_fixed=1 col order: b0(1) b0_state(2) r(1) q0(na) beta_q(na) E(K*K) pi(K)
    # cols before the E block = 1 + (K-1) + 1(disp) + na + na = 4 + 2*na
    ecol0 <- 4L + na + na
    ecols <- c(ecol0 + 1L, ecol0 + K + 2L, ecol0 + 2L*K + 3L)   # E_00,E_11,E_22
    cols  <- c(1, 2, 3, 4, 5:(4+na), ecols)
    for (i in seq_len(nf)) ranks[rep, i] <- frank(res$draws, cols[i], truths[i])
  }
  mr <- colMeans(ranks, na.rm = TRUE)
  expect_true(all(mr > 0.25 & mr < 0.75))
  B <- 8L
  pv <- apply(ranks, 2, function(x) {
    x <- x[is.finite(x)]; h <- as.numeric(table(cut(x, seq(0, 1, length.out = B+1), include.lowest = TRUE)))
    stats::pchisq(sum((h - length(x)/B)^2 / (length(x)/B)), B - 1L, lower.tail = FALSE)
  })
  expect_gte(sum(pv > 0.05 / nf), nf - 2L)
})
