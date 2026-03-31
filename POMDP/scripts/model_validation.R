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
    ggdist,
    ggokabeito
)

# set source as path ----
setwd(this.path::here())

raw_data <- read_rds("../data/lickometer_data.rds")

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

mean_lick <- behavior %>%
    group_by(ID, context) %>%
    summarise(
        mean_licks = mean(n())
    )
mean_lick


# frequentist stats ----
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

# prior predictive checks ----

prior_sim_func <- function(behavior_dat, n_sims, context) {
    behavior_dat_sel <- behavior_dat %>%
        filter(context == context) %>%
        group_by(ID, context) %>%
        summarise(
            mean_licks = mean(n())
        )
    mod_sim <- cmdstan_model("prior_simulator.stan")
    if (context == "low") {
        p_reward_1 <- 1.0
        p_reward_2 <- 1.0
    } else if (context == "mid") {
        p_reward_1 <- 1.0
        p_reward_2 <- 0.5
    } else {
        p_reward_1 <- 0.5
        p_reward_2 <- 0.25
    }
    sim_data <- list(
        N_steps = 144000,
        p_reward_1 = p_reward_1,
        p_reward_2 = p_reward_2
    )
    fit_sim <- mod_sim$sample(
        data = sim_data,
        fixed_param = TRUE,
        iter_sampling = n_sims,
        chains = 1,
        seed = 42
    )
    prior_animals <- fit_sim$draws(format = "df") %>%
        as_tibble() %>%
        mutate(
            total_licks = total_licks_1 + total_licks_2
        )
    real_mean <- mean(behavior_dat_sel %>% pull(mean_licks))
    sim_mean <- mean(prior_animals$total_licks)
    p_B <- mean(prior_animals$total_licks >= real_mean)
    return(
        list(
            sim_data = prior_animals,
            real_data = behavior_dat_sel,
            real_mean = real_mean,
            sim_mean = sim_mean,
            p_B <- p_B,
            context = context
        )
    )
}

if (file.exists("../results/prior_sim_data.rds")) {
    sim_data <- read_rds(file = "../results/prior_sim_data.rds")
} else {
    sim_data <- c("low", "mid", "high") %>%
        map(., function(context) {
            prior_sim_func(
                behavior_dat = behavior,
                n_sims = 1000,
                context = context
            )
        })
}

prior_check_plots_0 <- sim_data %>%
    map(., function(sim_data) {
        sim_data$sim_data %>%
            ggplot(aes(
                total_licks
            )) +
            geom_density(aes(
                fill = after_stat(x) >= sim_data$real_mean
            ), alpha = 0.5) +
            geom_vline(
                xintercept = sim_data$real_mean,
                linetype = "dashed"
            ) +
            scale_fill_manual(
                values = c(
                    "FALSE" = "gray70",
                    "TRUE" = palette_okabe_ito(3)
                ),
                labels = c(
                    "FALSE" = "simulated < real",
                    "TRUE" = "simulated >= real"
                ),
                guide = guide_legend(nrow = 2),
                name = ""
            ) +
            ggtitle(sim_data$context) +
            ggpubr::theme_pubr() +
            coord_cartesian(xlim = c(0, 20000))
    })

prior_check_plots_1 <- sim_data %>%
    map(., function(sim_data) {
        sim_data$sim_data %>%
            ggplot(aes(
                total_licks
            )) +
            geom_density(
                alpha = 0.5,
                aes(fill = "sim_data")
            ) +
            geom_density(
                data = sim_data$real_data,
                aes(
                    mean_licks,
                    fill = "obs_data"
                ),
                alpha = 0.5
            ) +
            scale_fill_manual(
                values = c(
                    "obs_data" = palette_okabe_ito(1),
                    "sim_data" = palette_okabe_ito(2)
                ),
                labels = c(
                    "obs_data" = "Observed data",
                    "sim_data" = "Simulated data"
                ),
                guide = guide_legend(nrow = 2),
                name = ""
            ) +
            ggtitle(sim_data$context) +
            ggpubr::theme_pubr() +
            coord_cartesian(xlim = c(0, 20000))
    })

p1 <- wrap_plots(
    prior_check_plots_0,
    ncol = 1,
    nrow = 3,
    guides = "collect"
)

p2 <- wrap_plots(
    prior_check_plots_1,
    ncol = 1,
    nrow = 3,
    guides = "collect"
)

wrap_plots(p1, p2)

# parameter recovery ----
