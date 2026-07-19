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
  # misclassification (v1b) replaces the change-point: erroring here is about the
  # change-point still being present, not about confusion() being inactive.
  expect_error(reg(n_states = 3, states = c("A","B","C"), obs_state = "state",
                   obs_model = confusion(), ref_state = "A"), "change-point")
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
  # confusion() requires a change-point-free (zero-breakpoint) outcome
  expect_error(reg_v1b(function(m) outcome(m, y ~ time, b0 = ~1, b1 = ~1, data = dat)),
               "change-point")
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
