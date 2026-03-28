# libs ----
pacman::p_load(
    tidyverse,
    cmdstanr,
    posterior,
    tidybayes,
    ggdist,
    patchwork,
    GGally,
    survival,
    survminer,
    emmeans,
    lme4,
    entropy,
    glmmTMB,
    robustlmm,
    performance,
    ggdist
)

# set source as path ----
setwd(this.path::here())

# load data ----
switch_data <- read_rds("../proc_datasets/switch_data.rds")
posteriors <- read_rds("../proc_datasets/random_effects.rds") %>%
    mutate(
        ID = case_when(
            ID == 1 ~ 574,
            ID == 2 ~ 575,
            ID == 3 ~ 576,
            ID == 4 ~ 577,
            ID == 5 ~ 578,
            ID == 6 ~ 579,
            ID == 7 ~ 580,
            ID == 8 ~ 581
        )
    ) %>%
    mutate(ID = as.factor(ID)) %>%
    mutate(
        context = case_when(
            context == 1 ~ "low",
            context == 2 ~ "mid",
            context == 3 ~ "high",
        ),
        context = factor(context,
            levels = c("low", "mid", "high")
        ),
        droga = case_when(
            droga == 2 ~ "veh",
            droga == 3 ~ "tcs",
            TRUE ~ "baseline"
        )
    ) %>%
    filter(droga != "baseline")
posteriors

switch_params <- switch_p %>%
    mutate(
        droga = as.factor(droga),
        context = as.factor(context)
    ) %>%
    ungroup() %>%
    group_by(ID, droga, context) %>%
    summarise(
        mean_switch = mean(switch_prob)
    ) %>%
    left_join(posteriors %>%
        mutate(ID = as.factor(ID)), by = c("ID", "droga", "context"))
switch_params

switch_mdl_0 <- glmmTMB(
    data = switch_params,
    mean_switch ~ 1 + (1 | ID),
    family = beta_family(link = "logit")
)
summary(switch_mdl_0)

switch_mdl_1 <- glmmTMB(
    data = switch_params,
    mean_switch ~ droga * context + (1 | ID),
    family = beta_family(link = "logit")
)
summary(switch_mdl_1)

switch_mdl_2 <- glmmTMB(
    data = switch_params,
    mean_switch ~ droga * context + beta + kappa + phi + side + beta_slope + (1 | ID),
    family = beta_family(link = "logit")
)
summary(switch_mdl)

# model params explain our target raw behavioral metric
test_likelihoodratio(
    switch_mdl_0,
    switch_mdl_1,
    switch_mdl_2
)

mdl_cor <- switch_params %>%
    ungroup() %>%
    mutate(.pred = predict(switch_mdl_2, type = "response"))

mdl_cor %>%
    ggplot(aes(
        .pred, mean_switch,
        color = interaction(droga, context)
    )) +
    geom_abline(
        intercept = 0,
        slope = 1,
        color = "black",
        linetype = "dashed"
    ) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE) +
    ggpubr::theme_pubr() +
    scale_x_continuous(breaks = seq(0, 0.25, 0.05), limits = c(0, 0.25)) +
    scale_y_continuous(breaks = seq(0, 0.25, 0.05), limits = c(0, 0.25)) +
    coord_fixed(ratio = 1)
