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
    ggokabeito,
    SBC,
    future
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
                n_sims = 100,
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

sim_model <- cmdstan_model("prior_simulator_hierarchical.stan")
main_model <- cmdstan_model("beta_bernoulli_model_prior_checked.stan")

format_and_compress_session <- function(sim_data, IDs) {
    n <- length(sim_data$action)
    act <- sim_data$action
    st <- sim_data$state
    rew <- sim_data$reward

    # pre-allocate the vector
    next_st <- rep(IDs$ID_IDLE, n)
    # fill with actual states
    if (n > 1) {
        next_st[1:(n - 1)] <- st[2:n]
    }

    # get if lick
    is_lick <- (act == IDs$ID_LICK1) | (act == IDs$ID_LICK2)
    # determine next step
    # is lick is a vector of TRUE / FALSE for licking behavior
    # rew is generated at the top and notes if a particular licking
    # behavior was associated with a reward or not
    # here we just fill resulting rewarded or non rewarded states
    next_st[is_lick & rew == 1] <- IDs$ID_REWARD_STATE
    next_st[is_lick & rew == 0] <- IDs$ID_NOREWARD_STATE

    # compression
    comp_act <- integer(n)
    comp_st <- integer(n)
    comp_nxt <- integer(n)
    comp_wt <- integer(n)

    idx <- 1
    i <- 1

    while (i <= n) {
        current_act <- act[i]
        current_st <- st[i]

        # if animal starts waiting start compression
        if (current_act == IDs$ID_WAIT) {
            weight <- 1
            while ((i + weight <= n) &&
                (act[i + weight] == IDs$ID_WAIT) &&
                (st[i + weight] == current_st)) {
                weight <- weight + 1
            }
            comp_act[idx] <- current_act
            comp_st[idx] <- current_st
            comp_wt[idx] <- weight
            comp_nxt[idx] <- next_st[i + weight - 1]
            i <- i + weight
        } else {
            comp_act[idx] <- current_act
            comp_st[idx] <- current_st
            comp_wt[idx] <- 1
            comp_nxt[idx] <- next_st[i]
            i <- i + 1
        }
        idx <- idx + 1
    }
    idx <- idx - 1
    return(list(
        action = comp_act[1:idx],
        state = comp_st[1:idx],
        next_state = comp_nxt[1:idx],
        weight = comp_wt[1:idx]
    ))
}

recovery_generator <- function(n_steps = 144000, n_animals = 8) {
    sim_model <- cmdstan_model("prior_simulator_hierarchical.stan")
    IDs <- list(
        ID_IDLE = 1, ID_WAIT = 1,
        ID_LICK1 = 5, ID_LICK2 = 6,
        ID_REWARD_STATE = 99, ID_NOREWARD_STATE = 100
    )

    contexts <- list(
        c(1.0, 1.0),
        c(1.0, 0.5),
        c(0.5, 0.25)
    )

    # global true parameters
    global_beta <- rnorm(1, 10.0, 1.5)
    global_beta_slope <- rnorm(1, 0, 1.5)
    global_kappa <- abs(rnorm(1, 0, 1.5))
    global_phi <- abs(rnorm(1, 0, 1.5))
    global_side <- rnorm(1, 0, 1.5)
    global_eps <- rbeta(1, 1, 10)

    # animal to animal spread
    sigma_beta_trait <- abs(rnorm(1, 0, 0.5))
    sigma_beta_slope_trait <- abs(rnorm(1, 0, 0.5))
    sigma_kappa_trait <- abs(rnorm(1, 0, 0.5))
    sigma_phi_trait <- abs(rnorm(1, 0, 0.5))

    n_sessions_total <- n_animals * 3

    # storage arrays
    all_action_list <- vector("list", n_sessions_total)
    all_state_list <- vector("list", n_sessions_total)
    all_next_list <- vector("list", n_sessions_total)
    all_weight_list <- vector("list", n_sessions_total)

    session_cog_ctx <- integer(n_sessions_total)
    start_idx <- integer(n_sessions_total)
    end_idx <- integer(n_sessions_total)

    sessions_per_animal_start <- integer(n_animals)
    sessions_per_animal_end <- integer(n_animals)

    session_counter <- 1
    current_row_idx <- 1

    # sim hierarchical fit
    for (a in 1:n_animals) {
        anim_beta <- global_beta + rnorm(1, 0, sigma_beta_trait)
        anim_beta_slope <- global_beta_slope + rnorm(1, 0, sigma_beta_slope_trait)
        anim_kappa <- global_kappa + rnorm(1, 0, sigma_kappa_trait)
        anim_phi <- global_phi + rnorm(1, 0, sigma_phi_trait)

        sessions_per_animal_start[a] <- session_counter

        # loop 3 contexts
        for (c in 1:3) {
            fit_sim <- sim_model$sample(
                data = list(
                    N_steps = n_steps,
                    p_reward_1 = contexts[[c]][1],
                    p_reward_2 = contexts[[c]][2],
                    prior_beta = anim_beta,
                    prior_beta_slope = anim_beta_slope,
                    prior_kappa = anim_kappa,
                    prior_phi = anim_phi,
                    prior_side = global_side,
                    epsilon = global_eps
                ),
                fixed_param = TRUE, iter_warmup = 0, iter_sampling = 1,
                chains = 1, show_messages = FALSE
            )

            raw_data <- list(
                action = as.vector(fit_sim$draws("action_history")),
                state = as.vector(fit_sim$draws("state_history")),
                reward = as.vector(fit_sim$draws("reward_history"))
            )

            comp_data <- format_and_compress_session(raw_data, IDs)

            all_action_list[[session_counter]] <- comp_data$action
            all_state_list[[session_counter]] <- comp_data$state
            all_next_list[[session_counter]] <- comp_data$next_state
            all_weight_list[[session_counter]] <- comp_data$weight

            len <- length(comp_data$action)

            session_cog_ctx[session_counter] <- c
            start_idx[session_counter] <- current_row_idx
            end_idx[session_counter] <- current_row_idx + len - 1

            current_row_idx <- current_row_idx + len
            session_counter <- session_counter + 1
        }
        sessions_per_animal_end[a] <- session_counter - 1
    }


    all_action <- unlist(all_action_list, use.names = FALSE)
    all_state <- unlist(all_state_list, use.names = FALSE)
    all_next <- unlist(all_next_list, use.names = FALSE)
    all_weight <- unlist(all_weight_list, use.names = FALSE)

    stan_data <- list(
        N_animals = n_animals,
        N_sessions_total = n_sessions_total,
        N_compressed_steps = length(all_action),
        N_drugs = 3,
        N_physics_contexts = 1,
        N_cognitive_contexts = 3,
        sessions_per_animal_start = sessions_per_animal_start,
        sessions_per_animal_end = sessions_per_animal_end,
        session_physics_context = rep(1, n_sessions_total),
        session_cognitive_context = session_cog_ctx,
        session_drug = rep(1, n_sessions_total),
        state_id = all_state,
        action_id = all_action,
        next_state_id = all_next,
        weight = all_weight,
        start_idx = start_idx,
        end_idx = end_idx,
        N_actions = 6,
        N_states = 100,
        ID_IDLE = IDs$ID_IDLE, ID_WAIT = IDs$ID_WAIT,
        ID_LICK1 = IDs$ID_LICK1, ID_LICK2 = IDs$ID_LICK2,
        ID_REWARD_STATE = IDs$ID_REWARD_STATE, ID_NOREWARD_STATE = IDs$ID_NOREWARD_STATE,
        grainsize = ceiling(n_animals / 6)
    )
    true_params_df <- data.frame(
        base_beta = global_beta,
        base_beta_slope = global_beta_slope,
        base_side = global_side,
        base_kappa = global_kappa,
        base_phi = global_phi,
        epsilon = global_eps,
        sigma_beta_trait = sigma_beta_trait,
        sigma_beta_slope_trait = sigma_beta_slope_trait,
        sigma_kappa_trait = sigma_kappa_trait,
        sigma_phi_trait = sigma_phi_trait
    )

    true_params_matrix <- posterior::as_draws_matrix(true_params_df)
    return(SBC_datasets(
        variables = true_params_matrix,
        generated = list(stan_data)
    ))
}

sbc_generator <- SBC_generator_function(recovery_generator)
datasets_test <- generate_datasets(
    sbc_generator,
    n_sims = 10
)
