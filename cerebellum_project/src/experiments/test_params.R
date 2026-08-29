
pacman::p_load(cmdstanr, posterior)
fit_bvk <- readRDS('../../results/fit_bvk_complete.rds')
p_fit <- fit_bvk[['metadata']]()[['model_params']]
cat('Params in fit:', length(p_fit), '\n')
print(p_fit[!grepl('\\[', p_fit)]) # print scalars

