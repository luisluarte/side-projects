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

# load behavioral data ----
raw_data <- read_rds(file = "../data/lickometer_data.rds")

# data for bayesian ppc ----
discrete_data <- read_rds("../data/processed/stan_behavior_data.rds")


# switch behavior -----
behavior <- raw_data %>%
    ungroup() %>%
    # this is the baseline
    filter(droga != "na_na_na_na") %>%
    # context recoding
    mutate(
        context = case_when(
            true_context %in% c("C_T") ~ "low",
            true_context %in% c("C_S2a", "C_S2b") ~ "mid",
            true_context %in% c("C_S3a", "C_S3b") ~ "high"
        ),
        action = case_when(
            # for now sensor 0 is set arbitrarily as the high_ev spout
            context == "low" & sensor == 0 ~ "high_ev",
            context == "mid" & tipo_recompensa == "cond100prob" ~ "high_ev",
            context == "high" & tipo_recompensa == "cond50prob" ~ "high_ev",
            TRUE ~ "low_ev"
        ),
        switch = if_else(sensor != lag(sensor, n = 1), 1, 0)
    ) %>%
    ungroup() %>%
    mutate(
        droga = factor(as.factor(droga), levels = c(
            "veh_na_na_na", "tcs_na_na_na"
        )),
        context = factor(as.factor(context),
            levels = c("low", "mid", "high")
        )
    ) %>%
    ungroup()
behavior

switch_p <- ev_behavior %>%
    drop_na(switch) %>%
    select(ID, switch, droga, context) %>%
    mutate(
        droga = if_else(droga == "veh_na_na_na", "veh", "tcs")
    ) %>%
    group_by(ID, droga, context) %>%
    mutate(
        cummulative_switch = cumsum(switch),
        step = row_number(),
        cum_switch_norm = scales::rescale(cummulative_switch, to = c(0, 1)),
        switch_prob = sum(switch) / n(),
        step_norm = scales::rescale(step, to = c(0, 1))
    )
switch_p

## stats ----
switch_mdl <- glmer(
    data = switch_p,
    family = binomial(link = "logit"),
    switch ~ droga * context + (1 | ID)
)
summary(switch_mdl)

switch_emm <- emmeans(
    switch_mdl,
    specs = ~ droga | context,
    type = "response"
)
switch_emm

switch_emm_df <- pairs(switch_emm) %>%
    broom::tidy(., conf.int = TRUE) %>%
    mutate(
        context = factor(as.factor(context),
            levels = c("low", "mid", "high")
        )
    )
switch_emm_df

switch_total_mdl <- glmmTMB(
    data = switch_p,
    cummulative_switch ~ droga * context + step_norm + (1 | ID),
    family = "nbinom2"
)
summary(switch_total_mdl)

switch_total_df <- emmeans(
    switch_total_mdl,
    specs = ~ droga + step_norm | context,
    type = "response"
) %>%
    pairs() %>%
    broom::tidy(., conf.int = TRUE) %>%
    mutate(
        context = factor(as.factor(context),
            levels = c("low", "mid", "high")
        )
    )
switch_total_df

## plots ----

# probability of switch
switch_p_p2 <- switch_p %>%
    ungroup() %>%
    group_by(ID, droga, context) %>%
    filter(step_norm == 1) %>%
    ggplot(aes(
        droga, switch_prob,
        color = droga
    )) +
    geom_boxplot(
        outlier.shape = NA,
        width = 0.5,
        aes(color = droga)
    ) +
    geom_line(aes(group = ID), color = "gray") +
    geom_point() +
    facet_wrap(~context) +
    ggpubr::theme_pubr() +
    theme(legend.position = "none") +
    scale_color_viridis(discrete = TRUE)
switch_p_p2

switch_p_p3 <- switch_emm_df %>%
    ggplot(aes(
        context, odds.ratio,
        ymin = conf.low,
        ymax = conf.high
    )) +
    geom_pointinterval() +
    geom_hline(
        yintercept = 1,
        linetype = "dashed"
    ) +
    ggpubr::theme_pubr()
switch_p_p3


# cummulative switching over normalized time
switch_p_p1 <- switch_p %>%
    ggplot(aes(
        step_norm, cummulative_switch,
        color = droga
    )) +
    geom_smooth(
        method = "gam",
        aes(group = droga),
        linewidth = 1
    ) +
    geom_line(aes(group = interaction(ID, droga)),
        alpha = 0.25
    ) +
    facet_wrap(~context) +
    ggpubr::theme_pubr() +
    scale_y_continuous(
        transform = scales::pseudo_log_trans()
    ) +
    scale_color_viridis(discrete = TRUE)
switch_p_p1

switch_p_p4 <- switch_total_df %>%
    ggplot(aes(
        context, ratio,
        ymin = conf.low,
        ymax = conf.high
    )) +
    geom_pointinterval() +
    geom_hline(
        yintercept = 1,
        linetype = "dashed"
    ) +
    ylab("tcs / veh")
switch_p_p4
