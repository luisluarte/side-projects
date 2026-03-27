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

# load fit ----
fit <- read_rds("../results/fit_optimal_final_v6.rds")
fit <- read_rds("../results/pathfinder_model.rds")

# data for bayesian ppc ----
discrete_data <- read_rds("../data/processed/stan_behavior_data.rds")

# helper functions ----
softmax <- function(x) {
    exp_x <- exp(x - max(x))
    return(exp_x / sum(exp_x))
}

# posterior medians -----

## pop means ----
mu_draws <- fit %>%
    gather_draws(
        mu_beta[droga, context],
        mu_kappa[droga, context],
        mu_phi[droga, context],
        mu_side[droga, context],
        mu_beta_slope[droga, context]
    ) %>%
    pivot_wider(
        names_from = .variable,
        values_from = .value
    )
mu_draws

## pop deltas ----
delta_params <- c(
    "drug_delta_beta", "drug_delta_kappa", "drug_delta_phi",
    "drug_delta_side", "drug_delta_beta_slope"
)
delta_draws <- as_draws_df(fit$draws(variables = delta_params)) %>%
    as_tibble() %>%
    select(-.chain, -.iteration, -.draw) %>%
    pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
    mutate(
        param = str_remove(variable, "drug_delta_"),
        param = str_remove(param, "\\[\\d+\\]"),
        context_id = as.integer(str_extract(variable, "\\d+")),
        context_id = case_when(
            context_id == 1 ~ "low",
            context_id == 2 ~ "mid",
            context_id == 3 ~ "high"
        ),
        context_id = factor(as.factor(context_id), levels = c("low", "mid", "high"))
    )
delta_draws

delta_draw_summ <- delta_draws %>%
    group_by(param, context_id) %>%
    summarise(
        Median = median(value),
        Lower_95 = quantile(value, 0.025),
        Upper_95 = quantile(value, 0.975),
        Prob_Greater_Zero = mean(value > 0),
        .groups = "drop"
    ) %>%
    arrange(context_id, param)
delta_draw_summ

### plots -----
delta_param_p1 <- delta_draws %>%
    ggplot(aes(
        param, value
    )) +
    stat_halfeye(aes(color = param)) +
    facet_wrap(~ as.factor(context_id)) +
    geom_hline(
        yintercept = 0,
        linetype = "dashed",
        color = "black"
    ) +
    ggpubr::theme_pubr() +
    theme(
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        legend.position = "none"
    ) +
    ylab("Estimate (TCS - VEH)") +
    xlab("")
delta_param_p1

kappa_p1 <- delta_draws %>%
    filter(param == "kappa") %>%
    ggplot(aes(
        context_id, value
    )) +
    stat_halfeye(
        point_interval = median_hdci
    ) +
    geom_hline(
        yintercept = 0,
        linetype = "dashed",
        color = "black"
    ) +
    ggpubr::theme_pubr() +
    ylab("Estimate (tcs - veh)") +
    xlab("")
kappa_p1

phi_p1 <- delta_draws %>%
    filter(param == "phi") %>%
    ggplot(aes(
        context_id, value
    )) +
    stat_halfeye(
        point_interval = median_hdci
    ) +
    geom_hline(
        yintercept = 0,
        linetype = "dashed",
        color = "black"
    ) +
    ggpubr::theme_pubr() +
    xlab("") +
    ylab("Estimate (TCS - VEH)")
phi_p1

beta_p1 <- delta_draws %>%
    filter(param == "beta") %>%
    ggplot(aes(
        context_id, value
    )) +
    stat_halfeye(
        point_interval = median_hdci
    ) +
    geom_hline(
        yintercept = 0,
        linetype = "dashed",
        color = "black"
    ) +
    ggpubr::theme_pubr() +
    xlab("") +
    ylab("Estimate (TCS - VEH)")
beta_p1

beta_slope_p1 <- delta_draws %>%
    filter(param == "beta_slope") %>%
    ggplot(aes(
        context_id, value
    )) +
    stat_halfeye(
        point_interval = median_hdci
    ) +
    geom_hline(
        yintercept = 0,
        linetype = "dashed",
        color = "black"
    ) +
    ggpubr::theme_pubr() +
    xlab("") +
    ylab("Estimate (TCS - VEH)")
beta_slope_p1

side_p1 <- delta_draws %>%
    filter(param == "side") %>%
    ggplot(aes(
        context_id, value
    )) +
    stat_halfeye(
        point_interval = median_hdci
    ) +
    geom_hline(
        yintercept = 0,
        linetype = "dashed",
        color = "black"
    ) +
    ggpubr::theme_pubr() +
    xlab("") +
    ylab("Estimate (TCS - VEH)")
side_p1

## ind level ----
trait_draws <- fit %>%
    gather_draws(
        r_animal_beta[ID, dim2],
        r_animal_kappa[ID, dim2],
        r_animal_phi[ID, dim2],
        r_animal_beta_slope[ID, dim2],
    ) %>%
    ungroup() %>%
    select(-dim2) %>%
    pivot_wider(
        names_from = .variable,
        values_from = .value
    )
trait_draws

## per condition draws -----
absolute_draws <- fit %>%
    gather_draws(
        mu_kappa[drug_id, ctx_id],
        mu_phi[drug_id, ctx_id],
        mu_side[drug_id, ctx_id],
        mu_beta[drug_id, ctx_id],
        mu_beta_slope[drug_id, ctx_id]
    ) %>%
    filter(drug_id %in% c(2, 3)) %>%
    mutate(
        drug = ifelse(drug_id == 2, "Vehicle", "TCS"),
        param = str_replace(.variable, "mu_", "")
    )

### plots -----
absolute_p1 <- absolute_draws %>%
    ggplot(aes(x = drug, y = .value, color = drug, fill = drug)) +
    stat_halfeye(
        alpha = 0.7,
        .width = c(0.89, 0.95),
        point_interval = median_hdci,
        position = position_dodge(width = 0.6)
    ) +
    ggpubr::theme_pubr() +
    facet_wrap(~ ctx_id * param, scales = "free")
absolute_p1

absolute_kappa_p1 <- absolute_draws %>%
    filter(param == "kappa") %>%
    ggplot(aes(x = drug, y = .value, color = drug, fill = drug)) +
    stat_halfeye(
        alpha = 0.7,
        position = position_dodge(width = 0.6)
    ) +
    ggpubr::theme_pubr() +
    facet_wrap(~ ctx_id * param) +
    theme(legend.position = "none")
absolute_kappa_p1

absolute_phi_p1 <- absolute_draws %>%
    filter(param == "phi") %>%
    ggplot(aes(x = drug, y = .value, color = drug, fill = drug)) +
    stat_halfeye(
        alpha = 0.7,
        position = position_dodge(width = 0.6)
    ) +
    ggpubr::theme_pubr() +
    theme(legend.position = "none") +
    facet_wrap(~ ctx_id * param) +
    xlab("")
absolute_phi_p1

absolute_beta_p1 <- absolute_draws %>%
    filter(param == "beta") %>%
    ggplot(aes(x = drug, y = .value, color = drug, fill = drug)) +
    stat_halfeye(
        alpha = 0.7,
        position = position_dodge(width = 0.6)
    ) +
    ggpubr::theme_pubr() +
    theme(legend.position = "none") +
    facet_wrap(~ ctx_id * param) +
    xlab("")
absolute_beta_p1

absolute_beta_slope_p1 <- absolute_draws %>%
    filter(param == "beta_slope") %>%
    ggplot(aes(x = drug, y = .value, color = drug, fill = drug)) +
    stat_halfeye(
        alpha = 0.7,
        position = position_dodge(width = 0.6)
    ) +
    ggpubr::theme_pubr() +
    theme(legend.position = "none") +
    facet_wrap(~ ctx_id * param) +
    xlab("")
absolute_beta_slope_p1

absolute_side_p1 <- absolute_draws %>%
    filter(param == "side") %>%
    ggplot(aes(x = drug, y = .value, color = drug, fill = drug)) +
    stat_halfeye(
        alpha = 0.7,
        position = position_dodge(width = 0.6)
    ) +
    ggpubr::theme_pubr() +
    theme(legend.position = "none") +
    facet_wrap(~ ctx_id * param) +
    xlab("")
absolute_side_p1

## posterior medians compilate -----
posterior_medians <- mu_draws %>%
    left_join(trait_draws, by = c(".chain", ".iteration", ".draw")) %>%
    mutate(
        mouse_beta = mu_beta + r_animal_beta,
        mouse_kappa = mu_kappa + r_animal_kappa,
        mouse_phi = mu_phi + r_animal_phi,
        mouse_side = mu_side,
        mouse_beta_slope = mu_beta_slope + r_animal_beta_slope
    ) %>%
    group_by(ID, droga, context) %>%
    summarise(
        beta = median(mouse_beta),
        kappa = median(mouse_kappa),
        phi = median(mouse_phi),
        side = median(mouse_side),
        beta_slope = median(mouse_beta_slope),
        .groups = "drop"
    )
posterior_medians

# one step posterior predictive check ----
run_onestep_ppc <- function(stan_data, posterior_medians) {
    N_sessions <- stan_data$N_sessions_total
    N_physics_contexts <- stan_data$N_physics_contexts
    dt <- 0.025

    animal_mapping <- integer(N_sessions)
    for (a in 1:stan_data$N_animals) {
        animal_mapping[stan_data$sessions_per_animal_start[a]:stan_data$sessions_per_animal_end[a]] <- a
    }
    true_p_matrix <- stan_data$context_probs
    results <- list()
    result_idx <- 1

    for (s in 1:N_sessions) {
        ID <- animal_mapping[s]
        context <- stan_data$session_cognitive_context[s]
        droga <- stan_data$session_drug[s]
        mouse_params <- posterior_medians %>%
            filter(
                ID == ID,
                droga == droga,
                context == context
            )
        b_s <- mouse_params$beta[1]
        k_s <- mouse_params$kappa[1]
        phi_s <- mouse_params$phi[1]
        side_s <- mouse_params$side[1]
        b_slope <- mouse_params$beta_slope[1]
        ep <- exp(mouse_params$eta_pos[1])
        en <- exp(mouse_params$eta_neg[1])
        diffusion <- mouse_params$belief_diffusion[1]

        # init session
        uniform_prior <- rep(1 / N_physics_contexts, N_physics_contexts)
        belief <- uniform_prior

        current_wait_time <- 0
        prev_act <- 0
        last_lick_spout <- NA

        lik_reward <- matrix(0, nrow = N_physics_contexts, ncol = 2)
        lik_noreward <- matrix(0, nrow = N_physics_contexts, ncol = 2)
        for (c in 1:N_physics_contexts) {
            lik_reward[c, 1] <- true_p_matrix[c, 1]^ep
            lik_reward[c, 2] <- true_p_matrix[c, 2]^ep
            lik_noreward[c, 1] <- (1.0 - true_p_matrix[c, 1])^en
            lik_noreward[c, 2] <- (1.0 - true_p_matrix[c, 2])^en
        }

        # apply belief difussion
        belief <- (1.0 - diffusion) * belief + diffusion * uniform_prior

        H <- 0
        for (c in 1:N_physics_contexts) {
            safe_b <- belief[c] + 1e-12
            H <- H - (safe_b * log(safe_b))
        }

        start_t <- stan_data$start_idx[s]
        end_t <- stan_data$end_idx[s]

        # replay
        for (t in start_t:end_t) {
            st <- stan_data$state_id[t]
            act <- stan_data$action_id[t]
            next_st <- stan_data$next_state_id[t]
            w <- stan_data$weight[t]

            Q_values <- numeric(stan_data$N_actions)
            for (a in 1:stan_data$N_actions) {
                acc <- 0
                for (c in 1:N_physics_contexts) {
                    acc <- acc + (belief[c] * stan_data$Q_star[ID, st, a, c])
                }
                Q_values[a] <- acc
            }

            if (st == stan_data$ID_IDLE) {
                b_eff <- b_s + b_slope * log1p(current_wait_time)
                Q_values[stan_data$ID_WAIT] <- Q_values[stan_data$ID_WAIT] + (b_eff * dt)
            }

            Q_values[stan_data$ID_LICK1] <- Q_values[stan_data$ID_LICK1] + (k_s * H) + side_s
            Q_values[stan_data$ID_LICK2] <- Q_values[stan_data$ID_LICK2] + (k_s * H)

            if (!is.na(last_lick_spout)) {
                Q_values[last_lick_spout] <- Q_values[last_lick_spout] + phi_s
            }

            # save lick predictions
            if (act == stan_data$ID_LICK1 || act == stan_data$ID_LICK2) {
                probs <- softmax(Q_values)

                prob_lick1 <- probs[stan_data$ID_LICK1] / (probs[stan_data$ID_LICK1] + probs[stan_data$ID_LICK2])
                prob_lick2 <- probs[stan_data$ID_LICK2] / (probs[stan_data$ID_LICK1] + probs[stan_data$ID_LICK2])

                p_switch <- 0
                actual_switch <- 0

                if (!is.na(last_lick_spout)) {
                    if (last_lick_spout == stan_data$ID_LICK1) {
                        p_switch <- prob_lick2
                        if (act == stan_data$ID_LICK2) actual_switch <- 1
                    } else {
                        p_switch <- prob_lick1
                        if (act == stan_data$ID_LICK1) actual_switch <- 1
                    }
                }

                results[[result_idx]] <- data.frame(
                    ID = ID,
                    session = s,
                    droga = droga,
                    context = context,
                    step = t,
                    is_switch = actual_switch,
                    model_p_switch = p_switch
                )
                result_idx <- result_idx + 1

                last_lick_spout <- act
            }

            if (act == stan_data$ID_WAIT) {
                current_wait_time <- current_wait_time + (w * dt)
            } else {
                current_wait_time <- 0
            }

            prev_act <- act

            if (next_st == stan_data$ID_REWARD_STATE ||
                next_st == stan_data$ID_NOREWARD_STATE) {
                spout_idx <- ifelse(act == stan_data$ID_LICK1, 1, 2)

                if (next_st == stan_data$ID_REWARD_STATE) {
                    likelihoods <- lik_reward[, spout_idx]
                } else {
                    likelihoods <- lik_noreward[, spout_idx]
                }

                belief <- belief * likelihoods
                belief <- belief + 1e-30
                sum_b <- sum(belief)

                if (sum_b > 1e-9) {
                    belief <- belief / sum_b
                } else {
                    belief <- uniform_prior
                }

                H <- 0
                for (c in 1:N_physics_contexts) {
                    safe_b <- max(belief[c], 1e-9)
                    H <- H - (safe_b * log(safe_b))
                }
            }
        }
    }

    bind_rows(results) %>%
        group_by(session) %>%
        mutate(lick_number = row_number()) %>%
        ungroup()
}

ppc_data <- run_onestep_ppc(
    discrete_data,
    posterior_medians
)


# high-ev, low-ev behavior -----
ev_behavior <- raw_data %>%
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

## plots ----
ev_behavior %>%
    group_by(
        ID, droga, context
    ) %>%
    summarise(
        high_ev = sum(action == "high_ev"),
        low_ev = sum(action == "low_ev")
    ) %>%
    ungroup() %>%
    mutate(
        high_low_ratio = low_ev / (low_ev + high_ev)
    ) %>%
    ggplot(aes(
        droga, high_low_ratio
    )) +
    geom_boxplot(outlier.shape = NA) +
    geom_point() +
    geom_line(aes(group = ID)) +
    facet_wrap(~context, scales = "free")


# behavior and model validation ----

## probability of switching raw ----
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

### switch stats ----
# check the number of total switches
switch_total_mdl <- lmer(
    data = switch_p,
    cummulative_switch ~ droga * context * step_norm + (1 | ID)
)
summary(switch_total_mdl)

switch_total_emm <- emmeans(
    switch_total_mdl,
    specs = ~ droga | context | step_norm,
    at = list(step_norm = c(1)),
    type = "response"
)
pairs(switch_total_emm, reverse = TRUE)

switch_total_emm_df <- pairs(switch_total_emm) %>%
    broom::tidy(., conf.int = TRUE) %>%
    mutate(
        context = factor(as.factor(context),
            levels = c("low", "mid", "high")
        )
    )
switch_total_emm_df

switch_total_raw <- switch_p %>%
    filter(step_norm == 1) %>%
    ungroup() %>%
    group_by(ID, context) %>%
    mutate(
        switch_diff = cummulative_switch[droga == "tcs"] - cummulative_switch[droga == "veh"]
    ) %>%
    filter(droga == "tcs") %>%
    mutate(contrast = "tcs - veh")
switch_total_raw

# does tcs reduce the p of switching spouts
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
pairs(switch_emm)
switch_emm_df <- switch_emm %>%
    broom::tidy(., conf.int = TRUE) %>%
    mutate(
        context = factor(as.factor(context),
            levels = c("low", "mid", "high")
        )
    )
# vehicle show greater probability of switch in mid and high contexts
switch_diffs <- pairs(switch_emm %>% regrid(),
    type = "response"
) %>%
    broom::tidy(., conf.int = TRUE) %>%
    mutate(
        context = factor(as.factor(context),
            levels = c("low", "mid", "high")
        )
    )
switch_diffs

### plots ----
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
    facet_wrap(~context)
switch_p_p1

switch_p_p2 <- switch_p %>%
    ungroup() %>%
    group_by(ID, droga, context) %>%
    filter(step_norm == 1) %>%
    ggplot(aes(
        droga, switch_prob,
        color = droga
    )) +
    geom_point() +
    geom_pointrange(
        data = switch_emm_df,
        aes(droga, prob,
            ymin = conf.low,
            ymax = conf.high
        )
    ) +
    geom_line(aes(group = ID), color = "gray") +
    facet_wrap(~context) +
    ggpubr::theme_pubr() +
    theme(legend.position = "none")
switch_p_p2

switch_p_p3 <- switch_diffs %>%
    ggplot(aes(
        interaction(context, contrast), estimate
    )) +
    geom_pointrange(
        aes(
            ymin = conf.low,
            ymax = conf.high
        )
    ) +
    geom_hline(
        yintercept = 0,
        linetype = "dashed"
    ) +
    ggpubr::theme_pubr() +
    ylab("Estimate (tcs - veh)")
switch_p_p3

switch_p_p4 <- switch_total_emm_df %>%
    ggplot(aes(
        contrast, estimate
    )) +
    geom_pointrange(
        aes(
            ymin = conf.low,
            ymax = conf.high
        )
    ) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    facet_wrap(~context)
switch_p_p4

# validation of model latent cognitive mechanisms ----
ppc_clean <- ppc_data %>%
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
    filter(droga %in% c("veh", "tcs")) %>%
    group_by(ID, droga, context) %>%
    mutate(
        step_norm = scales::rescale(step, to = c(0, 1))
    ) %>%
    ungroup()

## empirical switch rate ----
emp_summary <- switch_p %>%
    group_by(ID, droga, context) %>%
    summarise(emp_switch_rate = max(cummulative_switch, na.rm = TRUE), .groups = "drop") %>%
    mutate(
        ID = case_when(
            ID == 574 ~ 1,
            ID == 575 ~ 2,
            ID == 576 ~ 3,
            ID == 577 ~ 4,
            ID == 578 ~ 5,
            ID == 579 ~ 6,
            ID == 580 ~ 7,
            ID == 581 ~ 8
        )
    ) %>%
    mutate(ID = as.factor(ID))

## model switch rate -----
mod_summary <- ppc_clean %>%
    group_by(ID, droga, context) %>%
    summarise(mod_switch_rate = mean(model_p_switch, na.rm = TRUE), .groups = "drop") %>%
    mutate(ID = as.factor(ID))

## merge switch rates ----
switch_rates_merged <- emp_summary %>%
    left_join(mod_summary, by = c("ID", "droga", "context"))
switch_rates_merged

lookback <- 10
lookforward <- 0
peri_switch_data <- ppc_data %>%
    filter(droga %in% c(2, 3)) %>%
    group_by(session) %>%
    mutate(
        switch_step = if_else(is_switch == 1, step, NA_real_)
    ) %>%
    fill(switch_step, .direction = "up") %>%
    mutate(
        time_to_switch = step - switch_step
    ) %>%
    ungroup() %>%
    filter(time_to_switch >= -lookback & time_to_switch <= lookforward) %>%
    filter(!is.na(time_to_switch))

peri_switch_summary <- peri_switch_data %>%
    group_by(droga, context, time_to_switch) %>%
    summarise(
        avg_model_prob = mean(model_p_switch, na.rm = TRUE),
        se_model_prob = sd(model_p_switch, na.rm = TRUE) / sqrt(n()),
        .groups = "drop"
    ) %>%
    mutate(
        droga = as.factor(droga)
    )
peri_switch_summary

peri_switch_mdl <- glmmTMB(
    data = peri_switch_data %>%
        mutate(
            droga = as.factor(droga),
            context = as.factor(context)
        ),
    model_p_switch ~ time_to_switch * droga * context + (1 | ID),
    family = ordbeta,
)
summary(peri_switch_mdl)

emmeans(
    peri_switch_mdl,
    specs = ~ time_to_switch * droga * context,
    at = list(time_to_switch = seq(-100, 0, 10)),
    type = "response"
) %>%
    broom::tidy() %>%
    ggplot(aes(
        time_to_switch, response,
        color = droga
    )) +
    geom_line() +
    facet_wrap(~context)

peri_switch_data %>%
    ggplot(aes(
        time_to_switch, (model_p_switch)
    )) +
    geom_point(aes(color = as.factor(droga))) +
    facet_wrap(~context)

delta_data <- switch_rates_merged %>%
    select(
        ID, context, droga,
        emp_switch_rate, mod_switch_rate
    ) %>%
    pivot_wider(
        names_from = droga,
        values_from = c(emp_switch_rate, mod_switch_rate)
    ) %>%
    mutate(
        emp_delta = emp_switch_rate_tcs - emp_switch_rate_veh,
        mod_delta = mod_switch_rate_tcs - mod_switch_rate_veh
    )
delta_data

delta_data %>%
    ggplot(aes(
        emp_delta, mod_delta,
        color = context
    )) +
    geom_abline(
        slope = 1, intercept = 0,
        linetype = "dashed",
        color = "gray"
    ) +
    geom_vline(xintercept = 0, color = "black") +
    geom_hline(yintercept = 0, color = "black") +
    geom_point() +
    geom_smooth(
        method = "lm", se = FALSE,
        aes(group = 1),
        color = "tan1"
    )

cor.test(
    delta_data$emp_delta,
    delta_data$mod_delta
)


## mdl parameters and switch rate ----
posterior_medians_aligned <- posterior_medians %>%
    mutate(
        context = case_when(
            context == 1 ~ "low",
            context == 2 ~ "mid",
            context == 3 ~ "high"
        ),
        droga = case_when(
            droga == 2 ~ "veh",
            droga == 3 ~ "tcs",
            TRUE ~ "baseline"
        )
    ) %>%
    filter(droga != "baseline")

switch_params <- switch_p %>%
    mutate(
        ID = as.factor(as.numeric(ID)),
        droga = as.factor(droga),
        context = as.factor(context)
    ) %>%
    left_join(posterior_medians_aligned %>%
        mutate(ID = as.factor(ID)), by = c("ID", "droga", "context"))
switch_params

## stats ----

switch_param <- switch_params %>%
    ungroup() %>%
    drop_na()

null_mdl <- lmer(
    data = switch_params,
    switch_prob ~ 1 + step_norm + side + (1 | ID)
)

context_mdl <- lmer(
    data = switch_params,
    switch_prob ~ droga + context + side + step_norm + (1 | ID)
)

param_mdl <- lmer(
    data = switch_params,
    switch_prob ~ context + droga +
        kappa + phi + side +
        beta + beta_slope + step_norm + (1 | ID)
)
summary(param_mdl)

emmeans(
    param_mdl,
    specs = ~ context + droga + kappa + phi + side + beta + beta_slope + step_norm,
    type = "respose",
    at = list(kappa = seq(-15, 0, 1))
) %>%
    broom::tidy(., conf.int = TRUE) %>%
    ggplot(aes(
        kappa, estimate,
        color = droga
    )) +
    geom_line() +
    facet_wrap(~context)

test_likelihoodratio(null_mdl, context_mdl, param_mdl)
anova(null_mdl, context_mdl, param_mdl)

# plots ----


# TODO: parameter values drug versus vehicle

# TODO: parameter values per drug

# TODO: parameter interaction with context

# TODO: posterior of baseline wait value across drugs

# TODO: posterior of wait slope across drugs

# TODO: cognitive dominance ratio

# TODO: shannon entropy calculation
