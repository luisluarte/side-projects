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
    clipr
)

# set source as path ----
setwd(this.path::here())

# raw data ----
raw_data <- read_rds(file = "../data/lickometer_data.rds")

## kaplan curve ----
bout_data <- raw_data %>%
    ungroup() %>%
    filter(droga != "na_na_na_na") %>%
    mutate(
        context = case_when(
            true_context %in% c("C_T") ~ "low",
            true_context %in% c("C_S2a", "C_S2b") ~ "mid",
            true_context %in% c("C_S3a", "C_S3b") ~ "high"
        ),
        action = case_when(
            context == "low" & sensor == 0 ~ "high_ev",
            context == "mid" & tipo_recompensa == "cond100prob" ~ "high_ev",
            context == "high" & tipo_recompensa == "cond50prob" ~ "high_ev",
            TRUE ~ "low_ev"
        )
    ) %>%
    group_by(ID, context, droga) %>%
    summarise(
        entropy = entropy(table(sensor), unit = "log2")
    ) %>%
    ungroup() %>%
    mutate(
        droga = factor(as.factor(droga), levels = c(
            "veh_na_na_na", "tcs_na_na_na"
        )),
        context = factor(as.factor(context),
            levels = c("low", "mid", "high")
        )
    )
bout_data



## stats ----
bout_mdl <- lmer(
    data = bout_data,
    entropy ~ droga * context + (context | ID)
)
summary(bout_mdl)

bout_emm <- emmeans(
    bout_mdl,
    specs = ~ droga | context,
    type = "response"
)
bout_emm

bout_pairs <- pairs(bout_emm, reverse = TRUE)
bout_pairs

## plots ----
p1 <- bout_data %>%
    ggplot(aes(
        droga, entropy,
        color = droga
    )) +
    geom_boxplot() +
    geom_point() +
    geom_line(color = "gray70", aes(group = ID)) +
    ggpubr::theme_pubr() +
    facet_wrap(~context) +
    scale_y_continuous(
        breaks = seq(0, 1, 0.25),
        limits = c(0, 1)
    ) +
    theme(legend.position = "none")
p1

# model description ----

## cognitive architecture ----

# the model is a hierarchical partially observable markov decision process (POMDP)
# this model captures how animals infer hidden environmental state to make decisions
# in a setting of motivated behavior (lickometer), and how TCS alters this cofnitive process

# belief updating and entropy
# the model assumes that animals do not know the true "experimental context", that is,
# the reward probability of the spouts is unknown. However, they do maintain a "belief state",
# which is simply a probability vector over possible contexts (low, mid, high entropy).
# bayesian updating: after receiving a reward or non-reward, the animal updates its belief using bayes' rule

# learning rates eta_pos eta_neg: we fit separate parameters for how much weight an animal gives
# to positive feedback (licking is rewarded) vs. negative feedback (no reward)

# belief diffusion: between sessions, beliefs decay slightly toward a uniform distribution,
# which is how the model represent not knowing the context or "forgetting" the context

# entropy (H): computed form the belief state, high entropy means the animal is confused/exploring;
# low entropy means the animal is confident/exploiting

## value function ----

# the probability of taking an action is proportional to its value (Q), evaluated
# via softmax function.
# Q-values are constructed from two main components:
# extrinsic values: the dot product of the current belief state and the optimal policy values (Q*).
# Q* is pre-computed via value iteration based on the objective mechanics of the lickometer.
# intrinsic value:
# kappa (information seeking): scales the entropy H. Determines if the animal is driven
# to lick when uncertain.
# phi (persistence): a bonus added if the animal repeats its last action
# side (spatial bias): inherent preference for spout1 over spout2
# beta and beta_slope (impulsivity/waiting): the baseline value of waiting, and how that
# value decays the longer the animal has waited.

## hierarchical structure ---

# to separate the acute drug effects from underlying animal traits, parameters
# are structures hierarchically:
# population level (mu): the base effect of a cognitive context, plus the acute delta
# effect of TCS.
# trait level (animal random effects): stable, animal-specific offsets. modeled
# via non-centered parametrization (z-scores * sigma_trat) to prevent divergences
# state level (session random effects): fluctuations for a specific animal a specific day
# Flattened into a global 1D vector to prevent out-of-bounds indexing error and optimize CPU cache
# during HMC sampling

## computational optimizations ----

# reduce-sum: log-likelihood is parallelized across animals
# vectorized session: session are indexed globally
# entropy memoization: entropy is only recomputed when the belief state actually changes
# that is, after feedback

# 1. LOAD MODEL & DATA ---------------------------------------------------------
message("Loading posterior samples...")
fit <- readRDS("../results/fit_optimal_final_v7.rds")

# 2. HMC HEALTH DIAGNOSTICS ----------------------------------------------------
message("\n--- HMC Diagnostics ---")
diag_summary <- fit$diagnostic_summary()
print(diag_summary)

# Check for R-hat (Convergence) and ESS (Effective Sample Size)
fit_summary <- fit$summary()
bad_rhat <- fit_summary %>% filter(rhat > 1.05)
if (nrow(bad_rhat) > 0) {
    warning("Some parameters have R-hat > 1.05. Model may not have fully converged.")
    print(bad_rhat %>% select(variable, rhat, ess_bulk))
} else {
    message("All R-hat values look excellent (< 1.05).")
}

# detect the troublesome ones ----
fit_summary %>%
    filter(rhat > 1.05) %>%
    select(variable, median, rhat, ess_bulk) %>%
    ungroup() %>%
    arrange(desc(rhat)) %>%
    view()

nrow(bad_rhat) / nrow(fit_summary)

# 3. TRAIT CORRELATION ANALYSIS (The "Funnel/Ridge" Check) ---------------------
# 3. TRAIT CORRELATION ANALYSIS (The "Funnel/Ridge" Check) ---------------------
message("\n--- Analyzing Baseline Trait Correlations ---")

# Extract the individual animal trait offsets (medians)
traits_df <- fit$summary(
    variables = c("r_animal_beta", "r_animal_kappa", "r_animal_phi"),
    median
) %>%
    mutate(
        # Extract EXACTLY what is inside the brackets (e.g., "1,1" or "1")
        # This guarantees every single row has a unique identifier
        unique_index = str_extract(variable, "(?<=\\[).*(?=\\])"),

        # Extract the parameter name
        param = str_extract(variable, "(beta|kappa|phi)")
    ) %>%
    # Keep only the columns we need to spread
    select(unique_index, param, median) %>%
    # Pivot wider (modern equivalent of spread)
    pivot_wider(names_from = param, values_from = median)

print("Median Baseline Traits per Index:")
print(traits_df)

# Plot the correlation matrix (ignoring the index column for the math)
p_corr <- ggpairs(
    traits_df %>% select(-unique_index),
    title = "Posterior Correlation of Baseline Traits"
)
print(p_corr)
ggsave("../results/trait_correlations.png", p_corr, width = 6, height = 6)

# issue with overparametrization -----
fit %>%
    gather_draws(sigma_beta_trait, sigma_kappa_trait, sigma_phi_trait, sigma_beta_slope_trait) %>%
    ggplot(aes(x = .value, y = .variable, fill = .variable)) +
    stat_halfeye(alpha = 0.7) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
    labs(
        title = "Are Subject-Level Traits Necessary?",
        subtitle = "If a distribution is piled up against 0, the trait can be safely removed",
        x = "Estimated Variance (Sigma)",
        y = "Parameter"
    ) +
    theme_classic(base_size = 14) +
    theme(legend.position = "none")

fit %>%
    gather_draws(sigma_beta_trait, sigma_kappa_trait, sigma_phi_trait, sigma_beta_slope_trait) %>%
    group_by(.variable) %>%
    summarise(
        median_val = median(.value),
        standard_dev = sd(.value)
    )

# 4. DRUG EFFECT HYPOTHESIS TESTING --------------------------------------------
message("\n--- Drug vs. Vehicle Deltas (Within-Context) ---")

# The parameters we care about: the explicit differences in each context
delta_params <- c(
    "drug_delta_beta", "drug_delta_kappa", "drug_delta_phi",
    "drug_delta_side", "drug_delta_beta_slope"
)

# Extract draws
draws_df <- as_draws_df(fit$draws(variables = delta_params))

# Calculate Statistics: Median, 95% HDI, and Probability of Direction (Pd)
# Pd = the percentage of the posterior distribution that has the same sign as the median
results_df <- draws_df %>%
    select(-.chain, -.iteration, -.draw) %>%
    pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
    mutate(
        # Parse the variable name and context index
        param = str_remove(variable, "drug_delta_"),
        param = str_remove(param, "\\[\\d+\\]"),
        context_id = as.integer(str_extract(variable, "\\d+"))
    ) %>%
    group_by(param, context_id) %>%
    summarise(
        Median = median(value),
        Lower_95 = quantile(value, 0.025),
        Upper_95 = quantile(value, 0.975),
        # Bayesian P-value equivalent (Probability the effect is > 0)
        Prob_Greater_Zero = mean(value > 0),
        .groups = "drop"
    ) %>%
    arrange(context_id, param)

print("Target Hypothesis Tests (Drug Deltas):")
print(results_df, n = Inf)

write_csv(results_df, "../results/drug_delta_statistics.csv")

# 5. VISUALIZATION OF DRUG DELTAS ----------------------------------------------
# Forest plot for the three contexts
p_forest <- results_df %>%
    mutate(Context = paste("Context", context_id)) %>%
    ggplot(aes(x = Median, y = param, color = Context)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black", alpha = 0.6) +
    geom_pointrange(aes(xmin = Lower_95, xmax = Upper_95), position = position_dodge(width = 0.5)) +
    facet_wrap(~Context, ncol = 1) +
    ggpubr::theme_pubr() +
    theme(legend.position = "none")
p_forest

print(p_forest)
ggsave("../results/drug_deltas_forest.png", p_forest, width = 8, height = 10)

message("\nAnalysis complete. Results and plots saved to ../results/")

# absolute values ----
# 1. Extract the pre-calculated absolute values directly from the mu_ matrices
absolute_draws <- fit %>%
    gather_draws(
        mu_kappa[drug_id, ctx_id],
        mu_phi[drug_id, ctx_id],
        mu_side[drug_id, ctx_id],
        mu_beta_slope[drug_id, ctx_id]
    ) %>%
    # Keep only Vehicle (ID 2) and Active Drug (ID 3)
    filter(drug_id %in% c(2, 3)) %>%
    mutate(
        Condition = ifelse(drug_id == 2, "Vehicle", "TCS"),
        # Clean up the variable names for the plot (e.g., "mu_phi" -> "phi")
        parameter = str_replace(.variable, "mu_", "")
    )

# 2. Calculate Summary Statistics
absolute_summary <- absolute_draws %>%
    group_by(parameter, Condition, ctx_id) %>%
    summarise(
        Median = median(.value),
        Lower_95 = quantile(.value, 0.025),
        Upper_95 = quantile(.value, 0.975),
        .groups = "drop"
    )
write_csv(x = absolute_summary, file = "../results/parameter_summary.csv")

# 3. Visualization: Side-by-Side Posterior Densities
p_absolute <- absolute_draws %>%
    ggplot(aes(x = Condition, y = .value, color = Condition, fill = Condition)) +
    stat_halfeye(
        alpha = 0.7,
        .width = c(0.89, 0.95),
        point_interval = median_hdci,
        position = position_dodge(width = 0.6)
    ) +
    ggpubr::theme_pubr() +
    facet_wrap(~ ctx_id * parameter, scales = "free")
p_absolute

# model sims ----

# population means
animal_parameters <- fit %>%
    spread_draws(
        mu_phi[drug_id, ctx_id],
        mu_kappa[drug_id, ctx_id],
        mu_side[drug_id, ctx_id],
        mu_beta[drug_id, ctx_id],
        mu_beta_slope[drug_id, ctx_id],
        mu_eta_pos[drug_id, ctx_id],
        mu_eta_neg[drug_id, ctx_id]
    )
animal_parameters

# animal traits
trait_parameters <- fit %>%
    spread_draws(
        r_animal_phi[animal_id, ctx_id],
        r_animal_kappa[animal_id, ctx_id],
        r_animal_beta[animal_id, ctx_id]
    )
trait_parameters

# complete estimation
final_parameters <- animal_parameters %>%
    inner_join(
        trait_parameters,
        by = c(".chain", ".iteration", ".draw")
    ) %>%
    mutate(
        phi_s = mu_phi + r_animal_phi,
        kappa_s = mu_kappa + r_animal_kappa,
        side_s = mu_side,
        beta_s = mu_beta + r_animal_beta,
        beta_slope_s = mu_beta_slope,
        eta_pos_s = mu_eta_pos,
        eta_neg_s = mu_eta_neg
    ) %>%
    group_by(animal_id, drug_id, ctx_id.x) %>%
    median_qi(
        phi_s, kappa_s, side_s,
        beta_s, beta_slope_s,
        eta_pos_s, eta_neg_s
    )
final_parameters

p1 <- animal_parameters %>%
    mutate(
        drug_id = as.factor(drug_id),
        ctx_id = as.factor(ctx_id)
    ) %>%
    filter(drug_id != 1) %>%
    ggplot(aes(ctx_id, mu_phi, color = drug_id)) +
    geom_violin() +
    ggpubr::theme_pubr()
p1

## POMDP simulation -----
sim_pomdp_session <- function(animal_id, drug_cond, context_cond,
                              phi_s, k_s, side_s, beta_s, beta_slope_s,
                              eta_pos_s, eta_neg_s, # NEW: Subjective Learning Parameters
                              n_trials = 300) {
    dt <- 0.025
    ID_WAIT <- 1
    ID_LICK1 <- 2
    ID_LICK2 <- 3

    # Exponentiate etas exactly like Stan lines 20-21
    # We use pmin to prevent computational overflow just in case
    ep <- exp(min(eta_pos_s, 5.0))
    en <- exp(min(eta_neg_s, 5.0))

    true_probs <- case_when(
        context_cond == 1 ~ c(0.99, 0.99),
        context_cond == 2 ~ c(0.99, 0.50),
        context_cond == 3 ~ c(0.50, 0.25)
    )

    # These are the objective probabilities of the 3 contexts
    # context 1: 99/99 | context 2: 99/50 | context 3: 50/25
    context_probs <- matrix(c(
        0.99, 0.99,
        0.99, 0.50,
        0.50, 0.25
    ), nrow = 3, byrow = TRUE)

    # Myopic Q_star (Assuming ID_WAIT has 0 external value)
    Q_star <- cbind(c(0, 0, 0), context_probs)

    belief <- c(0.333, 0.333, 0.334)
    prev_act <- 0
    current_wait_time <- 0

    history_action <- numeric(n_trials)

    for (t in 1:n_trials) {
        # 1. Expected Value (Q_ext)
        Q_ext <- c(
            sum(belief * Q_star[, 1]),
            sum(belief * Q_star[, 2]),
            sum(belief * Q_star[, 3])
        )

        # 2. Curiosity / Information Gain
        H <- -sum(belief * log(belief + 1e-9))

        Q_total <- numeric(3)

        # 3. Add Heuristics (Exactly matching Stan lines 29-31)
        b_eff <- beta_s + beta_slope_s * log1p(current_wait_time)
        Q_total[ID_WAIT] <- Q_ext[ID_WAIT] + (b_eff * dt)

        Q_total[ID_LICK1] <- Q_ext[ID_LICK1] + (k_s * H) + side_s
        Q_total[ID_LICK2] <- Q_ext[ID_LICK2] + (k_s * H)

        if (prev_act == ID_LICK1) Q_total[ID_LICK1] <- Q_total[ID_LICK1] + phi_s
        if (prev_act == ID_LICK2) Q_total[ID_LICK2] <- Q_total[ID_LICK2] + phi_s

        # 4. Softmax Action
        Q_total <- pmin(pmax(Q_total, -50), 50)
        prob_action <- exp(Q_total) / sum(exp(Q_total))
        act <- sample(1:3, size = 1, prob = prob_action)

        # 5. Universe Logic & Subjective Bayesian Update
        if (act == ID_WAIT) {
            current_wait_time <- current_wait_time + dt
        } else {
            current_wait_time <- 0

            # Map action to spout (Lick1 -> Spout 1, Lick2 -> Spout 2)
            spout_idx <- ifelse(act == ID_LICK1, 1, 2)
            reward <- rbinom(1, 1, true_probs[spout_idx])

            # ==========================================
            # SUBJECTIVE LIKELIHOODS (Stan lines 22-24 & 35-36)
            # ==========================================
            if (reward == 1) {
                # Probability of reward raised to ep
                likelihoods <- context_probs[, spout_idx]^ep
            } else {
                # Probability of NO reward raised to en
                likelihoods <- (1 - context_probs[, spout_idx])^en
            }

            # Update Belief
            belief <- (belief * likelihoods) / sum(belief * likelihoods)
        }

        history_action[t] <- act
        prev_act <- act
    }

    return(tibble(
        animal_id = animal_id,
        drug_cond = drug_cond,
        context_cond = context_cond,
        trial = 1:n_trials,
        action = history_action
    ))
}

full_sim <- pmap_dfr(list(
    animal_id = final_parameters$animal_id,
    drug_cond = final_parameters$drug_id,
    context_cond = final_parameters$ctx_id.x,
    phi_s = final_parameters$phi_s,
    k_s = final_parameters$kappa_s,
    side_s = final_parameters$side_s,
    beta_s = final_parameters$beta_s,
    beta_slope_s = final_parameters$beta_slope_s,
    eta_pos_s = final_parameters$eta_pos_s,
    eta_neg_s = final_parameters$eta_neg_s
), ~ sim_pomdp_session(..1, ..2, ..3,
    ..4, ..5, ..6, ..7, ..8, ..9, ..10,
    n_trials = 144000
))

full_sim_entropy <- full_sim %>%
    filter(action %in% c(2, 3)) %>%
    mutate(action = if_else(action == 2, 1, 0)) %>%
    group_by(animal_id, drug_cond, context_cond) %>%
    mutate(
        entropy = entropy(table(action), unit = "log2")
    )

full_sim_entropy %>%
    filter(drug_cond != 1) %>%
    ggplot(aes(
        interaction(drug_cond, context_cond), entropy,
        color = as.factor(drug_cond)
    )) +
    geom_boxplot() +
    geom_point()

expected_entropy <- function(phi, kappa, side,
                             beta, beta_slope,
                             eta_pos, eta_neg, ctx_id, drug) {
    # 1. Subjective Probability Distortion (Stan lines 21-24)
    true_p <- case_when(
        ctx_id == 1 ~ c(0.99, 0.99),
        ctx_id == 2 ~ c(0.99, 0.50),
        ctx_id == 3 ~ c(0.50, 0.25)
    )

    ep <- exp(eta_pos)
    en <- exp(eta_neg)

    # Subjective Expected Value (Q_ext)
    # Distorts how the rat "perceives" the 0.99 vs 0.50
    q_sub <- (true_p^ep) / (true_p^ep + (1 - true_p)^en)

    # 2. Information Gain (Entropy of Belief)
    # Assuming a mid-session stable belief state for H
    H_belief <- 0.5 # Approximation of H from Stan line 17

    # 3. Transition Probabilities (3-Way Softmax)
    # We calculate the probability of picking Spout 1, Spout 2, or Waiting
    # given that the rat was previously at Spout 1

    # Current Wait Time = 0 (assuming we just finished a lick)
    b_eff <- (beta + beta_slope * log1p(0)) * 0.025 # Stan line 29-30

    # Values if Prev Action was Spout 1
    v_wait1 <- b_eff
    v_s1_1 <- q_sub[1] + (kappa * H_belief) + side + phi
    v_s2_1 <- q_sub[2] + (kappa * H_belief)

    denom1 <- exp(v_wait1) + exp(v_s1_1) + exp(v_s2_1)
    p_stay_s1 <- exp(v_s1_1) / denom1
    p_switch_s1_to_s2 <- exp(v_s2_1) / denom1

    # Values if Prev Action was Spout 2
    v_wait2 <- b_eff
    v_s1_2 <- q_sub[1] + (kappa * H_belief) + side
    v_s2_2 <- q_sub[2] + (kappa * H_belief) + phi

    denom2 <- exp(v_wait2) + exp(v_s1_2) + exp(v_s2_2)
    p_stay_s2 <- exp(v_s2_2) / denom2
    p_switch_s2_to_s1 <- exp(v_s1_2) / denom2

    # 4. Stationary Distribution for Spout Choice
    # Removing the 'Wait' time to look only at the Choice Entropy
    # This finds the long-term ratio of Spout 1 vs Spout 2
    p_steady_s1 <- p_switch_s2_to_s1 / (p_switch_s2_to_s1 + p_switch_s1_to_s2)

    # 5. Shannon Entropy
    entropy <- if_else(p_steady_s1 <= 0 | p_steady_s1 >= 1, 0,
        -p_steady_s1 * log2(p_steady_s1) - (1 - p_steady_s1) * log2(1 - p_steady_s1)
    )

    return(entropy)
}

derive_3state_stationary_H <- function(phi, side, kappa, beta, beta_slope, eta_pos, eta_neg, ctx_id, drug) {
    # 1. Subjective Reward perceived at steady-state
    true_p <- case_when(ctx_id == 1 ~ c(0.99, 0.99), ctx_id == 2 ~ c(0.99, 0.50), ctx_id == 3 ~ c(0.50, 0.25))
    ep <- exp(eta_pos)
    en <- exp(eta_neg)
    q_sub <- (true_p^ep) / (true_p^ep + (1 - true_p)^en) # Subjective Q values

    # 2. Parameters
    dt <- 0.025
    b_eff <- (beta + beta_slope * log1p(0)) * dt # Value of waiting
    kH <- kappa * 0.5 # Constant curiosity bias

    # 3. Transition Matrix (States: 1=Lick1, 2=Lick2, 3=Wait)
    # Row i is 'Previous Action', Column j is 'Current Action'
    tm <- matrix(0, nrow = 3, ncol = 3)

    for (prev in 1:3) {
        v <- numeric(3)
        v[3] <- b_eff # Wait value is constant
        v[1] <- q_sub[1] + kH + side + (if (prev == 1) phi else 0) # Phi only if prev was Lick1
        v[2] <- q_sub[2] + kH + (if (prev == 2) phi else 0) # Phi only if prev was Lick2

        probs <- exp(v) / sum(exp(v))
        tm[prev, ] <- probs
    }

    # 4. Find Stationary Distribution (pi * tm = pi)
    # Solving the linear system for the steady-state occupancy of each state
    evals <- eigen(t(tm))$vectors[, 1]
    pi_steady <- as.numeric(evals / sum(evals))

    # 5. Extract Choice Probability (Spout 1 vs Spout 2, ignoring Wait)
    p_s1 <- pi_steady[1] / (pi_steady[1] + pi_steady[2])

    # 6. Choice Entropy
    H <- if_else(p_s1 <= 0 | p_s1 >= 1, 0, -p_s1 * log2(p_s1) - (1 - p_s1) * log2(1 - p_s1))
    return(H)
}

analytical_results <- final_parameters %>%
    mutate(
        expected_entropy = pmap_dbl(
            list(
                phi_s,
                side_s,
                kappa_s,
                beta_s,
                beta_slope_s,
                eta_pos_s,
                eta_neg_s,
                ctx_id.x,
                drug_id
            ),
            ~ derive_3state_stationary_H(..1, ..2, ..3, ..4, ..5, ..6, ..7, ..8, ..9)
        )
    )
analytical_results

analytic_summary <- analytical_results %>%
    group_by(drug_id, ctx_id.x) %>%
    summarise(
        mean_expected_H = mean(expected_entropy)
    ) %>%
    filter(drug_id != 1)
analytic_summary

analytical_results %>%
    filter(drug_id != 1) %>%
    ggplot(aes(
        interaction(drug_id, ctx_id.x), expected_entropy
    )) +
    geom_point()


# posterior phi across contexts ----
posterior_phi <- fit %>%
    spread_draws(mu_phi[drug_idx, context_idx]) %>%
    mutate(
        drug_cond = case_when(
            drug_idx == 1 ~ "baseline",
            drug_idx == 2 ~ "vehicle",
            drug_idx == 3 ~ "tcs"
        ),
        context_cond = case_when(
            context_idx == 1 ~ "100_100",
            context_idx == 2 ~ "100_50",
            context_idx == 3 ~ "50_25"
        )
    ) %>%
    ungroup() %>%
    select(-drug_idx, -context_idx) %>%
    pivot_wider(
        names_from = c(drug_cond, context_cond), values_from = mu_phi
    ) %>%
    mutate(
        effect_c1_low = vehicle_100_100 - tcs_100_100,
        effect_c2_mid = vehicle_100_50 - tcs_100_50,
        effect_c3_high = vehicle_50_25 - tcs_50_25
    ) %>%
    mutate(
        mid_vs_low = effect_c2_mid - effect_c1_low,
        high_vs_mid = effect_c3_high - effect_c2_mid,
        high_vs_low = effect_c3_high - effect_c1_low
    ) %>%
    summarise(
        # high versus low
        median_high_vs_low = median(high_vs_low),
        ci_lower_hvl = quantile(high_vs_low, 0.025),
        ci_upper_hvl = quantile(high_vs_low, 0.975),
        pd_high_vs_low = mean(high_vs_low > 0),

        # high versus mid
        median_high_vs_mid = median(high_vs_mid),
        ci_lower_hvm = quantile(high_vs_mid, 0.025),
        ci_upper_hvm = quantile(high_vs_mid, 0.975),
        pd_high_vs_mid = mean(high_vs_mid > 0)
    )
posterior_phi
glimpse(posterior_phi)

## plots -----
posterior_phi_table <- tibble(
    comparison = c("high_vs_low", "high_vs_mid"),
    medians = c(posterior_phi$median_high_vs_low, posterior_phi$median_high_vs_mid),
    ci_upper = c(posterior_phi$ci_upper_hvl, posterior_phi$ci_upper_hvm),
    ci_lower = c(posterior_phi$ci_lower_hvl, posterior_phi$ci_lower_hvm)
)
posterior_phi_table

p2 <- posterior_phi_table %>%
    ggplot(aes(
        comparison, medians,
        ymin = ci_lower, ymax = ci_upper
    )) +
    geom_pointrange() +
    geom_hline(yintercept = 0, color = "gray70", linetype = "dashed") +
    ggpubr::theme_pubr() +
    ylab("phi_median")
p2
