library(cmdstanr)
library(posterior)
fit <- read_cmdstan_csv(list.files("/home/DCCS5/cerebellum_project/results", pattern="fit_vopt_n30-.*\\.csv", full.names=TRUE)[1])$post_warmup_draws
vars <- dimnames(fit)$variable
print(vars[grep("a_base", vars)])
print(vars[grep("v_ctx", vars)])
