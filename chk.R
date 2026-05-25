d <- readRDS('benchmarks/bench_nb_breakpoint.rds')
print(posterior::summarise_draws(d$bjlm$draws)[100:115,])
