# libs ----
pacman::p_load(
    tidyverse,
    cmdstanr,
    posterior,
    bayesplot,
    parallel,
    data.table
)

# set working directory
setwd(this.path::here())

# load data ----
d <- readRDS("../data/processed/discrete_data.rds")

# coding A and S ----
# this is necessary for bayesian fit
coded_d <- d %>%
    mutate(
        A = case_when(
            # wait
            A == "a_W" ~ 1,
            # nose into poke
            A == "a_P" ~ 2,
            # stay in the poke
            A == "a_SP" ~ 3,
            # leave the poke
            A == "a_LP" ~ 4,
            # lick in spout 0
            A == "a_L1" ~ 5,
            # lick in spout 1
            A == "a_L2" ~ 6
        ),
        S = case_when(
            # idle
            S == "S_I" ~ 1,
            # poke transient
            S == "S_P1" ~ 2,
            # poke valid
            S == "S_P2" ~ 3,
            # the task is armed but nothing is happening
            S == "S_Armed" ~ 4,
            # left chain of FR5
            S == "S_1_0" ~ 5,
            S == "S_2_0" ~ 6,
            S == "S_3_0" ~ 7,
            S == "S_4_0" ~ 8,
            # right chain of FR5
            S == "S_0_1" ~ 9,
            S == "S_0_2" ~ 10,
            S == "S_0_3" ~ 11,
            S == "S_0_4" ~ 12,
            # consuming reward
            S == "S_CR" ~ 99,
            # consuming no-reward
            S == "S_CN" ~ 100
        )
    ) %>%
    filter(!is.na(A), !is.na(S)) %>%
    mutate(next_S = lead(S, default = 1))

# COMPRESSION LOGIC (Morphism Aggregation)
# We collapse rows where the belief is stationary (between outcomes)
compressed_d <- coded_d %>%
    mutate(
        outcome_event = next_S %in% c(99, 100),
        run_id = rleid(ID, n_sesion, S, A, next_S, outcome_event)
    ) %>%
    group_by(run_id) %>%
    summarise(
        ID = first(ID),
        n_sesion = first(n_sesion),
        true_context = first(true_context),
        S = first(S),
        A = first(A),
        next_S = first(next_S),
        weight = n(),
        .groups = "drop"
    )

# physics calibration -----
# the model uses 'ideal' conditions and then determines deviations from it
# the most relevant mechanic in the task is to get a reward
# to do so you require some time in nosepoke + travel to spout + FR5
# in order to get good baseline level we use training data from last 3 session
# and determine this time cost as follows:
# nosepoke = 50 ms (this is a constant, set by the task)
# travel time = the 10th percentile of the distribution over the lengths a_LP -> first lick
# FR5 = the mean ILI within FR5

## lick costs ----
fr5_cost <- coded_d %>%
    filter(droga == "na_na_na_na", is_datum == TRUE, n_sesion %in% c(15, 16, 17)) %>%
    group_by(ID) %>%
    mutate(
        ILI = timestamp - lag(timestamp)
    ) %>%
    filter(
        # exclude the first one because it computes time between non-lick
        # and an actual lick from FR5
        S %in% 5:12,
        # exclude clear disengagement
        ILI <= 1000
    ) %>%
    summarise(
        mean_ili = mean(ILI, na.rm = TRUE),
        # turn into the sample rate
        ili_steps = round(mean_ili / 25),
        .groups = "drop"
    )

## travel costs ----
travel_cost <- coded_d %>%
    filter(droga == "na_na_na_na", A %in% c(4, 5, 6), n_sesion %in% c(15, 16, 17)) %>%
    group_by(ID) %>%
    mutate(
        is_leave = A == 4,
        is_lick = A %in% c(5, 6),
        travel_group = cumsum(is_leave)
    ) %>%
    filter(is_leave | is_lick) %>%
    ungroup() %>%
    group_by(ID, travel_group) %>%
    mutate(
        travel_time = if_else(n() >= 2, min(timestamp[A != 4], na.rm = TRUE) - timestamp[1], NA)
    ) %>%
    slice_head(., n = 1) %>%
    filter(travel_time > 0) %>%
    ungroup() %>%
    group_by(ID) %>%
    summarise(
        fast_travel = quantile(x = travel_time, 0.10, na.rm = TRUE),
        # turn into the sample rate
        fast_travel_steps = round(fast_travel / 25),
        .groups = "drop"
    )

## merge travel and licks costs ----
animal_physics <- full_join(
    select(fr5_cost, ID, ili_steps),
    select(travel_cost, ID, fast_travel_steps),
    by = "ID"
) %>%
    mutate(
        ili_steps = replace_na(ili_steps, 6),
        fast_travel_steps = replace_na(fast_travel_steps, 20),
        poke_cost_steps = 2
    )

# Constants ----
GAMMA <- 0.99
N_LICK_CHAIN <- 5
N_CONTEXTS <- 5
N_STATES <- 100
N_ACTIONS <- 6
N_ANIMALS <- nrow(animal_physics)

CTX_PROBS <- matrix(c(
    0.99, 0.99,
    0.50, 0.99,
    0.99, 0.50,
    0.25, 0.50,
    0.50, 0.25
), ncol = 2, byrow = TRUE)

# Q* calculation ----
calc_chain_value <- function(p_reward, k, lick_cost) {
    v_outcome <- p_reward
    licks_needed <- N_LICK_CHAIN - k
    if (licks_needed <= 0) {
        return(v_outcome)
    }
    return((GAMMA^(lick_cost * licks_needed)) * v_outcome)
}

Q_STAR <- array(0, dim = c(N_ANIMALS, N_CONTEXTS, N_STATES, N_ACTIONS))

for (a in 1:N_ANIMALS) {
    a_params <- animal_physics[a, ]
    L_COST <- a_params$ili_steps
    T_COST <- a_params$fast_travel_steps
    P_COST <- a_params$poke_cost_steps

    for (c in 1:N_CONTEXTS) {
        p1 <- CTX_PROBS[c, 1]
        p2 <- CTX_PROBS[c, 2]

        for (k in 0:4) {
            sid_left <- if (k == 0) 4 else (4 + k)
            if (sid_left != 4) {
                Q_STAR[a, c, sid_left, 5] <- (GAMMA^L_COST) * calc_chain_value(p1, k + 1, L_COST)
                Q_STAR[a, c, sid_left, 6] <- (GAMMA^(T_COST + L_COST)) * calc_chain_value(p2, 1, L_COST)
            }
            sid_right <- if (k == 0) 4 else (8 + k)
            if (sid_right != 4) {
                Q_STAR[a, c, sid_right, 6] <- (GAMMA^L_COST) * calc_chain_value(p2, k + 1, L_COST)
                Q_STAR[a, c, sid_right, 5] <- (GAMMA^(T_COST + L_COST)) * calc_chain_value(p1, 1, L_COST)
            }
        }
        Q_STAR[a, c, 4, 5] <- (GAMMA^(T_COST + L_COST)) * calc_chain_value(p1, 1, L_COST)
        Q_STAR[a, c, 4, 6] <- (GAMMA^(T_COST + L_COST)) * calc_chain_value(p2, 1, L_COST)
        v_armed <- max(Q_STAR[a, c, 4, 5], Q_STAR[a, c, 4, 6])
        Q_STAR[a, c, 3, 4] <- v_armed * GAMMA
        v_valid <- Q_STAR[a, c, 3, 4]
        Q_STAR[a, c, 2, 3] <- v_valid * GAMMA
        v_transient <- Q_STAR[a, c, 2, 3]
        Q_STAR[a, c, 1, 2] <- v_transient * (GAMMA^P_COST)
    }
}

# Stan Data Prep ----
animal_map_df <- tibble(ID = animal_physics$ID) %>% mutate(animal_idx = row_number())
compressed_d <- left_join(compressed_d, animal_map_df, by = "ID") %>%
    mutate(context_id = case_when(
        true_context == "C_T" ~ 1, true_context == "C_S2a" ~ 2,
        true_context == "C_S2b" ~ 3, true_context == "C_S3a" ~ 4,
        true_context == "C_S3b" ~ 5
    )) %>%
    arrange(animal_idx, n_sesion)

sessions_df <- compressed_d %>%
    group_by(animal_idx, n_sesion) %>%
    summarise(
        start_idx = min(row_number()), end_idx = max(row_number()),
        true_context = first(context_id), .groups = "drop"
    )

Q_star_permuted <- aperm(Q_STAR, c(2, 1, 3, 4))

# Stan Data ----
stan_data <- list(
    N_animals = N_ANIMALS, N_sessions_total = nrow(sessions_df),
    N_compressed_steps = nrow(compressed_d), animal_idx = compressed_d$animal_idx,
    state_id = compressed_d$S, action_id = compressed_d$A,
    next_state_id = compressed_d$next_S, weight = compressed_d$weight,
    animal_map = sessions_df$animal_idx, start_idx = sessions_df$start_idx,
    end_idx = sessions_df$end_idx, true_context = sessions_df$true_context,
    N_contexts = N_CONTEXTS, N_actions = N_ACTIONS, N_states = 100,
    Q_star = Q_star_permuted, context_probs = CTX_PROBS,
    ID_IDLE = 1, ID_WAIT = 1, ID_LICK1 = 5, ID_LICK2 = 6,
    ID_REWARD_STATE = 99, ID_NOREWARD_STATE = 100
)

N_CORES <- parallel::detectCores(logical = FALSE)
mod <- cmdstan_model("POMDP_model.stan")

start_inference <- Sys.time()

fit <- mod$sample(
    data = stan_data,
    chains = 4,
    parallel_chains = min(4, N_CORES),
    iter_warmup = 1000,
    iter_sampling = 1000,
    # High verbosity refresh rate (every 10 iterations) for granular progress tracking
    refresh = 10,
    init = 0,
    show_messages = TRUE,
    show_exceptions = TRUE
)

end_inference <- Sys.time()
message("Sampling complete. Total Inference Duration: ", round(difftime(end_inference, start_inference, units = "mins"), 2), " minutes.")

# ==============================================================================
# OUTPUT PERSISTENCE (Morphism into Storage)
# ==============================================================================

# Create results directory if it doesn't exist
results_dir <- "../results/model_fits"
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)

# 1. Save the full CmdStanMCMC object
# This ensures all samples, metadata, and diagnostics are preserved.
# This is the 'raw' belief state.
fit$save_object(file = file.path(results_dir, "pomdp_fit_full.rds"))

# 2. Extract and save the posterior summary
# A compressed representation of the distribution (Means, SDs, R-hat)
fit_summary <- fit$summary()
write_csv(fit_summary, file.path(results_dir, "pomdp_summary.csv"))

# 3. Save subject-specific parameters (The individual functors)
# This extracts beta, kappa, and tau for each animal
animal_params <- fit$draws(variables = c("beta", "kappa", "tau"), format = "df") %>%
    as_tibble() %>%
    # Use subject metadata to map indices back to IDs if needed
    mutate(animal_id_map = list(animal_map_df))

saveRDS(animal_params, file.path(results_dir, "animal_posteriors.rds"))

message("Model output successfully persisted to ", results_dir)
