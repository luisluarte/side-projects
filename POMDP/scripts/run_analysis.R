# ==============================================================================
# HIERARCHICAL POMDP ANALYSIS: ROBUST ZEN 5 OPTIMIZATION
# ==============================================================================
# 1. Threading: 3 Chains x 2 Threads (Physical Core Saturation)
# 2. Safety: PRIMITIVE ARRAYS + NO HEAP ALLOCS inside loop.
# ==============================================================================

pacman::p_load(
    tidyverse, cmdstanr, posterior, parallel, data.table, this.path
)

setwd(this.path::here())

# 1. DATA PREPARATION & DISCRETIZATION ----
d <- readRDS("../data/processed/discrete_data.rds")

# Map actions and states to numeric IDs
coded_d <- d %>%
    mutate(
        A = case_when(
            A == "a_W" ~ 1, A == "a_P" ~ 2, A == "a_SP" ~ 3,
            A == "a_LP" ~ 4, A == "a_L1" ~ 5, A == "a_L2" ~ 6
        ),
        S = case_when(
            S == "S_I" ~ 1, S == "S_P1" ~ 2, S == "S_P2" ~ 3, S == "S_Armed" ~ 4,
            S %in% paste0("S_", 1:4, "_0") ~ 5, S %in% paste0("S_0_", 1:4) ~ 6,
            S == "S_CR" ~ 99, S == "S_CN" ~ 100,
            .default = 1
        )
    ) %>%
    filter(!is.na(A), !is.na(S)) %>%
    mutate(next_S = lead(S, default = 1))

# 2. SUBJECT-SPECIFIC PHYSICS CALIBRATION ----
message("Calibrating individual motor physics...")

animal_physics <- coded_d %>%
    filter(true_context == "C_T") %>%
    group_by(ID) %>%
    summarise(
        lick_cost_steps = round(mean(diff(timestamp[A %in% c(5, 6)]) %/% 25, na.rm = T)),
        .groups = "drop"
    ) %>%
    mutate(
        lick_cost_steps = ifelse(is.na(lick_cost_steps) | lick_cost_steps < 4, 6, lick_cost_steps),
        travel_cost_steps = 15,
        poke_cost_steps = 2
    )

# 3. VALUE ITERATION: GENERATING Q_STAR_FINAL ----
GAMMA <- 0.99
N_LICK_CHAIN <- 5
N_PHYS_CONTEXTS <- 5
N_STATES <- 100
N_ACTIONS <- 6
N_ANIMALS <- nrow(animal_physics)
CTX_PROBS <- matrix(c(0.99, 0.99, 0.50, 0.99, 0.99, 0.50, 0.25, 0.50, 0.50, 0.25), ncol = 2, byrow = T)

calc_chain_value <- function(p_reward, k, lick_cost) {
    v_outcome <- p_reward
    licks_needed <- N_LICK_CHAIN - k
    if (licks_needed <= 0) {
        return(v_outcome)
    }
    return((GAMMA^(lick_cost * licks_needed)) * v_outcome)
}

Q_raw <- array(0, dim = c(N_ANIMALS, N_PHYS_CONTEXTS, N_STATES, N_ACTIONS))

for (a in 1:N_ANIMALS) {
    p_a <- animal_physics[a, ]
    L_C <- p_a$lick_cost_steps
    T_C <- p_a$travel_cost_steps
    P_C <- p_a$poke_cost_steps
    for (c in 1:N_PHYS_CONTEXTS) {
        p1 <- CTX_PROBS[c, 1]
        p2 <- CTX_PROBS[c, 2]
        for (k in 0:4) {
            sid <- if (k == 0) 4 else (4 + k)
            Q_raw[a, c, sid, 5] <- (GAMMA^(if (k == 0) T_C else L_C)) * calc_chain_value(p1, k + 1, L_C)
        }
        for (k in 0:4) {
            sid <- if (k == 0) 4 else (8 + k)
            Q_raw[a, c, sid, 6] <- (GAMMA^(if (k == 0) T_C else L_C)) * calc_chain_value(p2, k + 1, L_C)
        }
        v_a <- max(Q_raw[a, c, 4, 5], Q_raw[a, c, 4, 6])
        Q_raw[a, c, 3, 4] <- v_a * GAMMA
        Q_raw[a, c, 2, 3] <- Q_raw[a, c, 3, 4] * GAMMA
        Q_raw[a, c, 1, 2] <- Q_raw[a, c, 2, 3] * (GAMMA^P_C)
    }
}

# RESHAPE: [Animal, Context, State, Action] -> [Context, Animal, State, Action]
# This matches the new array[,,,] declaration in Stan
Q_star_final <- aperm(Q_raw, c(2, 1, 3, 4))

# 4. DATA MAPPING & SESSION INDEXING ----
animal_map <- tibble(ID = animal_physics$ID) %>% mutate(animal_idx = row_number())

sessions_df <- coded_d %>%
    left_join(animal_map, by = "ID") %>%
    mutate(
        phys_ctx_id = case_when(
            true_context == "C_T" ~ 1, true_context == "C_S2a" ~ 2,
            true_context == "C_S2b" ~ 3, true_context == "C_S3a" ~ 4, TRUE ~ 5
        ),
        droga_mapped = case_when(
            droga %in% c("na_na_na_na", "veh_na_na_na") ~ "Vehicle",
            TRUE ~ droga
        )
    ) %>%
    group_by(animal_idx, n_sesion) %>%
    summarise(
        phys_ctx = first(phys_ctx_id),
        cog_ctx = case_when(first(droga_mapped) == "Vehicle" & phys_ctx == 1 ~ 1, phys_ctx %in% 2:3 ~ 2, TRUE ~ 3),
        drug_name = first(droga_mapped),
        .groups = "drop"
    )

drug_map <- tibble(drug_name = unique(sessions_df$drug_name)) %>% mutate(drug_id = row_number())
sessions_df <- sessions_df %>% left_join(drug_map, by = "drug_name")

compressed_steps <- coded_d %>%
    left_join(animal_map, by = "ID") %>%
    mutate(run_id = rleid(animal_idx, n_sesion, S, A)) %>%
    group_by(run_id) %>%
    summarise(
        animal_idx = first(animal_idx), n_sesion = first(n_sesion),
        S = first(S), A = first(A), next_S = first(next_S), weight = n(),
        .groups = "drop"
    )

session_pointers <- compressed_steps %>%
    group_by(animal_idx, n_sesion) %>%
    summarise(start_idx = min(row_number()), end_idx = max(row_number()), .groups = "drop")

animal_pointers <- session_pointers %>%
    group_by(animal_idx) %>%
    summarise(s_start = min(row_number()), s_end = max(row_number()), .groups = "drop")

# 5. EXECUTION (SAFE MODE) ----

stan_data <- list(
    N_animals = nrow(animal_map),
    N_sessions_total = nrow(session_pointers),
    N_compressed_steps = nrow(compressed_steps),
    N_drugs = nrow(drug_map),
    N_physics_contexts = 5,
    N_cognitive_contexts = 3,
    sessions_per_animal_start = animal_pointers$s_start,
    sessions_per_animal_end = animal_pointers$s_end,
    session_physics_context = sessions_df$phys_ctx,
    session_cognitive_context = sessions_df$cog_ctx,
    session_drug = sessions_df$drug_id,
    state_id = compressed_steps$S,
    action_id = compressed_steps$A,
    next_state_id = compressed_steps$next_S,
    weight = compressed_steps$weight,
    start_idx = session_pointers$start_idx,
    end_idx = session_pointers$end_idx,
    N_actions = 6, N_states = 100,
    # PASSING ARRAY DIRECTLY
    Q_star = Q_star_final,
    context_probs = CTX_PROBS,
    ID_IDLE = 1, ID_WAIT = 1, ID_LICK1 = 5, ID_LICK2 = 6,
    ID_REWARD_STATE = 99, ID_NOREWARD_STATE = 100,
    grainsize = max(1, as.integer(nrow(animal_map) / 6))
)

# SAFETY UPDATE: Removed -march=native
# Arch Linux + Zen 5 + AVX512 + reduce_sum is creating alignment crashes.
# -O3 is stable and fast enough.
mod <- cmdstan_model(
    "pomdp_model.stan",
    cpp_options = list(
        stan_threads = TRUE,
        "CXXFLAGS += -O3"
    )
)

# Threading: 3 Chains x 2 Threads
fit <- mod$sample(
    data = stan_data,
    chains = 3,
    parallel_chains = 3,
    threads_per_chain = 2,
    iter_warmup = 1000,
    iter_sampling = 1000,
    max_treedepth = 12,
    adapt_delta = 0.90,
    init = 0.1,
    refresh = 10
)

dir.create("../results", showWarnings = FALSE)
fit$save_object("../results/fit_optimal_final.rds")
