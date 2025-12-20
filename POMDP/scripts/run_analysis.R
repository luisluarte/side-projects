# libs ----
pacman::p_load(
    tidyverse,
    cmdstanr,
    posterior,
    bayesplot
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
    )

# physics calibration -----
# the model uses 'ideal' conditions and then determines deviations from it
# the most relevant mechanic in the task is to get a reward
# to do so you require some time in nosepoke + travel to spout + FR5
# in order to get good baseline level we use training data from last 3 session
# and determine this time cost as follows:
# nosepoke = 50 ms (this is a constant, set by the task)
# travel time = the 10th percentile of the distribution over the lengths a_LP -> first lick
# FR5 = the 10th percetile of the ILI within FR5

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
        S %in% c(5, 6, 7, 9, 10, 10),
        # exclude clear disengagement
        ILI <= 1000
    ) %>%
    summarise(
        mean_ili = mean(ILI),
        # turn into the sample rate
        ili_steps = round(mean_ili / 25)
    )
fr5_cost

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
        fast_travel_steps = round(fast_travel / 25)
    )
travel_cost

## merge travel and licks costs ----
complete_physics <- full_join(
    select(fr5_cost, ID, ili_steps),
    select(travel_cost, ID, fast_travel_steps)
) %>%
    mutate(poke_cost_steps = 2) # this is constant
complete_physics

# Constants ----
GAMMA <- 0.99

N_LICK_CHAIN <- 5

CTX_PROBS <- matrix(c(
    0.99, 0.99,
    0.50, 0.99,
    0.99, 0.50,
    0.25, 0.50,
    0.50, 0.25
), ncol = 2, byrow = TRUE)

N_CONTEXTS <- dim(CTX_PROBS)[1]

N_STATES <- 100

N_ACTIONS <- 6

N_ANIMALS <- nrow(complete_physics)


# Q* calculation ----

calc_chain_value <- function(p_reward, k, lick_cost) {
    v_outcome <- p_reward # Utility: Reward=1, No-Reward=0
    licks_needed <- N_LICK_CHAIN - k
    if (licks_needed <= 0) {
        return(v_outcome)
    }
    return((GAMMA^(lick_cost * licks_needed)) * v_outcome)
}

Q_STAR <- array(0, dim = c(N_ANIMALS, N_CONTEXTS, N_STATES, N_ACTIONS))

# fill the array
for (s in 1:N_ANIMALS) {
    s_params <- complete_physics[s, ]
    # this is in sample space (each step = 25 ms.)
    L_COST <- s_params$ili_steps
    T_COST <- s_params$fast_travel_steps
    P_COST <- s_params$poke_cost_steps

    for (c in 1:N_CONTEXTS) {
        p1 <- CTX_PROBS[c, 1]
        p2 <- CTX_PROBS[c, 2]

        # lick chain states
        for (k in 0:4) {
            # left spout
            sid_left <- if (k == 0) 4 else (4 + k)
            if (sid_left != 4) { # already in chain
                # for animal s in context c, filling the left spout
                Q_STAR[s, c, sid_left, 5] <- (GAMMA^L_COST) * calc_chain_value(p1, k + 1, L_COST)
                # Switching includes travel + the first lick execution
                Q_STAR[s, c, sid_left, 6] <- (GAMMA^(T_COST + L_COST)) * calc_chain_value(p2, 1, L_COST)
            }

            # right spout
            sid_right <- if (k == 0) 4 else (8 + k)
            if (sid_right != 4) {
                Q_STAR[s, c, sid_right, 6] <- (GAMMA^L_COST) * calc_chain_value(p2, k + 1, L_COST)
                Q_STAR[s, c, sid_right, 5] <- (GAMMA^(T_COST + L_COST)) * calc_chain_value(p1, 1, L_COST)
            }
        }

        # considering thing after the nosepoke port
        # Action Lick requires traveling to spout
        Q_STAR[s, c, 4, 5] <- (GAMMA^(T_COST + L_COST)) * calc_chain_value(p1, 1, L_COST)
        Q_STAR[s, c, 4, 6] <- (GAMMA^(T_COST + L_COST)) * calc_chain_value(p2, 1, L_COST)

        # poke and idle states
        v_armed <- max(Q_STAR[s, c, 4, 5], Q_STAR[s, c, 4, 6])
        Q_STAR[s, c, 3, 4] <- v_armed * GAMMA # s_P2 -> a_LP -> Armed

        v_valid <- Q_STAR[s, c, 3, 4]
        Q_STAR[s, c, 2, 3] <- v_valid * GAMMA # s_P1 -> a_SP -> s_P2

        v_transient <- Q_STAR[s, c, 2, 3]
        Q_STAR[s, c, 1, 2] <- v_transient * (GAMMA^P_COST) # s_I -> a_P -> s_P1 (Delayed by hold)
    }
}


# stan data prep ----

# animals
animal_map_df <- tibble(ID = complete_physics$ID) %>%
    mutate(
        animal_idx = row_number()
    )
coded_d <- left_join(
    coded_d, animal_map_df,
    by = "ID"
)

# context string to id mapping
coded_d <- coded_d %>%
    filter(!is.na(true_context)) %>%
    mutate(
        context_id = case_when(
            true_context == "C_T" ~ 1,
            true_context == "C_S2a" ~ 2,
            true_context == "C_S2b" ~ 3,
            true_context == "C_S3a" ~ 4,
            true_context == "C_S3b" ~ 5
        )
    )

# sort data
coded_d <- coded_d %>%
    arrange(
        animal_idx, n_sesion, timestamp_discrete
    )

# metadata for looping and carrying learning between sessions
sessions_df <- coded_d %>%
    group_by(animal_idx, n_sesion) %>%
    summarise(
        start_idx = min(row_number()),
        end_idx = max(row_number()),
        true_context = first(context_id)
    ) %>%
    ungroup() %>%
    arrange(animal_idx, n_sesion)

# stan data ----
stan_data <- list(
    N_animals = N_ANIMALS,
    N_sessions_total = nrow(sessions_df),
    N_timesteps_total = nrow(coded_d),
    animal_idx = coded_d$animal_idx,
    animal_map = sessions_df$animal_idx,
    state_id = coded_d$S,
    action_id = coded_d$A,
    next_state_id = lead(coded_d$S, default = 1),
    start_idx = sessions_df$start_idx,
    end_idx = sessions_df$end_idx,
    true_context = sessions_df$true_context,
    N_contexts = N_CONTEXTS,
    N_actions = N_ACTIONS,
    N_states = 100,
    Q_star = Q_STAR,
    context_probs = CTX_PROBS,
    # IDLE and WAIT are the source for the exploration bonus
    ID_IDLE = 1,
    ID_WAIT = 1,
    # LICK1 and LICK2 are for the information gain
    ID_LICK1 = 5,
    ID_LICK2 = 6,
    # REWARD and NOREWARD state, this trigger the belief update
    # also called terminal states
    ID_REWARD_STATE = 99,
    ID_NOREWARD_STATE = 100
)
