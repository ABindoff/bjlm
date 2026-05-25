txt <- readLines("src/rust/src/sampler_bjlm.rs", encoding="UTF-8")
full_text <- paste(txt, collapse = "\n")

target1 <- "fn hmc_step_om_weighted("
replacement1 <- "fn softplus(x: f64) -> f64 {\n    if x > 20.0 { x } else { x.exp().ln_1p() }\n}\n\nfn hmc_step_om_weighted("
full_text <- sub(target1, replacement1, full_text, fixed = TRUE)

target_om <- '        let r = &data.y - &mu;
        // WEIGHTED log-likelihood
        let mut ll = 0.0;
        for i in 0..data.n {
            ll += -0.5 * weights[i] * r[i] * r[i] / (sigma * sigma);
        }
        let lp = log_truncated_normal_prior(
            q.as_slice(), &priors.om_mean[k], &priors.om_sd[k],
            &priors.om_lb[k], &priors.om_ub[k],
        );

        let inv_s2 = 1.0 / (sigma * sigma);
        let mut grad = DVector::<f64>::zeros(p);
        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            let ri = rho_k[i];
            let bi = cache.delta_vals[k][i];
            let mut dmu_dom = -(bi * si + di * ri * si * (1.0 - si) * bi);
            if is_om1 { dmu_dom -= cache.b1_vals[i]; }
            // WEIGHTED gradient
            let factor = weights[i] * r[i] * inv_s2 * dmu_dom;
            for j in 0..p {
                grad[j] -= factor * data.x_om[k][(i, j)];
            }
        }'

new_om <- '        let mut ll = 0.0;
        let mut grad_ll_mu = DVector::<f64>::zeros(data.n);

        match data.outcome_family {
            crate::model::OutcomeFamily::Gaussian => {
                let inv_s2 = 1.0 / (sigma * sigma);
                for i in 0..data.n {
                    let ri = data.y[i] - mu[i];
                    ll += -0.5 * weights[i] * ri * ri * inv_s2;
                    grad_ll_mu[i] = weights[i] * ri * inv_s2;
                }
            },
            crate::model::OutcomeFamily::Binomial => {
                for i in 0..data.n {
                    let expit_mu = sigmoid(mu[i]);
                    ll += weights[i] * (data.y[i] * mu[i] - softplus(mu[i]));
                    grad_ll_mu[i] = weights[i] * (data.y[i] - expit_mu);
                }
            },
            crate::model::OutcomeFamily::NegativeBinomial => {
                let r_param = state.r;
                for i in 0..data.n {
                    let expit_mu = sigmoid(mu[i]);
                    ll += weights[i] * (data.y[i] * mu[i] - (data.y[i] + r_param) * softplus(mu[i]));
                    grad_ll_mu[i] = weights[i] * (data.y[i] - (data.y[i] + r_param) * expit_mu);
                }
            }
        }

        let lp = log_truncated_normal_prior(
            q.as_slice(), &priors.om_mean[k], &priors.om_sd[k],
            &priors.om_lb[k], &priors.om_ub[k],
        );

        let mut grad = DVector::<f64>::zeros(p);
        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            let ri = rho_k[i];
            let bi = cache.delta_vals[k][i];
            let mut dmu_dom = -(bi * si + di * ri * si * (1.0 - si) * bi);
            if is_om1 { dmu_dom -= cache.b1_vals[i]; }
            let factor = grad_ll_mu[i] * dmu_dom;
            for j in 0..p {
                grad[j] -= factor * data.x_om[k][(i, j)];
            }
        }'

full_text <- sub(target_om, new_om, full_text, fixed = TRUE)

target_rho <- '        let r = &data.y - &mu;
        let mut ll = 0.0;
        for i in 0..data.n {
            ll += -0.5 * weights[i] * r[i] * r[i] / (sigma * sigma);
        }
        let lp = log_truncated_normal_prior(
            q.as_slice(), &priors.rho_mean[k], &priors.rho_sd[k],
            &priors.rho_lb[k], &priors.rho_ub[k],
        );

        let inv_s2 = 1.0 / (sigma * sigma);
        let mut grad = DVector::<f64>::zeros(p);
        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            let bi = cache.delta_vals[k][i];
            let dmu_drho = di * di * si * (1.0 - si) * bi;
            let factor = weights[i] * r[i] * inv_s2 * dmu_drho;
            for j in 0..p {
                grad[j] -= factor * data.x_rho[k][(i, j)];
            }
        }'

new_rho <- '        let mut ll = 0.0;
        let mut grad_ll_mu = DVector::<f64>::zeros(data.n);

        match data.outcome_family {
            crate::model::OutcomeFamily::Gaussian => {
                let inv_s2 = 1.0 / (sigma * sigma);
                for i in 0..data.n {
                    let ri = data.y[i] - mu[i];
                    ll += -0.5 * weights[i] * ri * ri * inv_s2;
                    grad_ll_mu[i] = weights[i] * ri * inv_s2;
                }
            },
            crate::model::OutcomeFamily::Binomial => {
                for i in 0..data.n {
                    let expit_mu = sigmoid(mu[i]);
                    ll += weights[i] * (data.y[i] * mu[i] - softplus(mu[i]));
                    grad_ll_mu[i] = weights[i] * (data.y[i] - expit_mu);
                }
            },
            crate::model::OutcomeFamily::NegativeBinomial => {
                let r_param = state.r;
                for i in 0..data.n {
                    let expit_mu = sigmoid(mu[i]);
                    ll += weights[i] * (data.y[i] * mu[i] - (data.y[i] + r_param) * softplus(mu[i]));
                    grad_ll_mu[i] = weights[i] * (data.y[i] - (data.y[i] + r_param) * expit_mu);
                }
            }
        }

        let lp = log_truncated_normal_prior(
            q.as_slice(), &priors.rho_mean[k], &priors.rho_sd[k],
            &priors.rho_lb[k], &priors.rho_ub[k],
        );

        let mut grad = DVector::<f64>::zeros(p);
        for i in 0..data.n {
            let di = data.tau[i] - om_k[i];
            let si = sigmoid(di * rho_k[i]);
            let bi = cache.delta_vals[k][i];
            let dmu_drho = di * di * si * (1.0 - si) * bi;
            let factor = grad_ll_mu[i] * dmu_drho;
            for j in 0..p {
                grad[j] -= factor * data.x_rho[k][(i, j)];
            }
        }'

full_text <- sub(target_rho, new_rho, full_text, fixed = TRUE)

writeLines(full_text, "src/rust/src/sampler_bjlm.rs", useBytes=TRUE)
