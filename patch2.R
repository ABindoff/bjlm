txt <- readLines("src/rust/src/sampler_bjlm.rs", encoding="UTF-8")

# 1. Add softplus right before hmc_step_om_weighted
idx_om <- grep("fn hmc_step_om_weighted", txt)
if (length(idx_om) > 0) {
  txt <- append(txt, c(
    "fn softplus(x: f64) -> f64 {",
    "    if x > 20.0 { x } else { x.exp().ln_1p() }",
    "}",
    ""
  ), after = idx_om[1] - 1)
}

# Find block of ll and grad for om
start_om <- grep("let mut ll = 0.0;", txt)
end_om <- grep("let lp = log_truncated_normal_prior", txt)
# Wait, this might be fragile. 
writeLines(txt, "src/rust/src/sampler_bjlm.rs", useBytes=TRUE)
