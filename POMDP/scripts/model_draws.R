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
    viridis
)

# set source as path ----
setwd(this.path::here())

# load fit ----
fit <- read_rds("../results/beta_bernoulli_model_prior_checked.rds")

# main effects draws ----
main_effects_draws <- fit %>%
    gather_draws(
        mu_beta[drug, context],
        mu_kappa[drug, context],
        mu_phi[drug, context],
        mu_side[drug, context],
        mu_beta_slope[drug, context]
    ) %>%
    pivot_wider(
        names_from = .variable,
        values_from = .value
    )
main_effects_draws
write_rds(x = main_effects_draws, file = "../posterior_draws/main_effects_draws.rds")


## main effects plots -----
main_effects_dat <- main_effects_draws %>%
    pivot_longer(
        cols = contains("mu_"),
        names_to = ".variable", values_to = ".value"
    ) %>%
    filter(drug %in% c(2, 3)) %>%
    mutate(
        drug = ifelse(drug == 2, "veh", "tcs"),
        param = str_replace(.variable, "mu_", "")
    )
main_effects_dat

params <- main_effects_dat %>%
    group_by(param) %>%
    group_split()

main_effects_plot <- params %>%
    map(., function(param_dat) {
        param_dat %>%
            ggplot(aes(x = drug, y = .value, color = drug, fill = drug)) +
            stat_halfeye(
                alpha = 0.7,
                .width = c(0.95),
                point_interval = median_hdci,
                position = position_dodge(width = 0.6)
            ) +
            ggpubr::theme_pubr() +
            theme(
                legend.position = "none"
            ) +
            facet_wrap(~context) +
            ggtitle(label = param_dat$param[1]) +
            scale_color_viridis(discrete = TRUE) +
            scale_fill_viridis(discrete = TRUE)
    })

wrap_plots(
    main_effects_plot,
    ncol = 3, nrow = 3
)

# main diffs draws -----
delta_params <- c(
    "drug_delta_beta", "drug_delta_kappa", "drug_delta_phi",
    "drug_delta_side", "drug_delta_beta_slope"
)
delta_draws <- as_draws_df(fit$draws(variables = delta_params)) %>%
    as_tibble() %>%
    select(-.chain, -.iteration, -.draw) %>%
    pivot_longer(everything(), names_to = ".variable", values_to = ".value") %>%
    mutate(
        param = str_remove(.variable, "drug_delta_"),
        param = str_remove(param, "\\[\\d+\\]"),
        context = as.integer(str_extract(.variable, "\\d+"))
    )
delta_draws

delta_dat <- delta_draws %>%
    group_by(param) %>%
    group_split()
delta_dat

# diffs plot ----
main_diffs_plot <- delta_dat %>%
    map(., function(param_dat) {
        param_dat %>%
            ggplot(aes(
                x = context,
                y = .value
            )) +
            stat_halfeye(
                alpha = 0.7,
                .width = c(0.95),
                point_interval = median_hdci
            ) +
            geom_hline(
                yintercept = 0,
                linetype = "dashed"
            ) +
            ggpubr::theme_pubr() +
            theme(
                legend.position = "none"
            ) +
            ggtitle(label = param_dat$param[1]) +
            scale_color_viridis(discrete = TRUE) +
            scale_fill_viridis(discrete = TRUE) +
            scale_x_continuous(breaks = c(1, 2, 3))
    })

wrap_plots(
    main_diffs_plot,
    ncol = 3, nrow = 3
)
