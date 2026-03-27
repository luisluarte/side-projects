# ==============================================================================
# HIERARCHICAL POMDP ANALYSIS: ROBUST ZEN 5 OPTIMIZATION
# ==============================================================================
# 1. Threading: 3 Chains x 2 Threads (Physical Core Saturation)
# 2. Safety: PRIMITIVE ARRAYS + NO HEAP ALLOCS inside loop.
# ==============================================================================
message("starting fit...")

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

            # UNROLL THE SPOUT 1 CHAIN (Matches sid = 4 + k)
            S == "S_1_0" ~ 5,
            S == "S_2_0" ~ 6,
            S == "S_3_0" ~ 7,
            S == "S_4_0" ~ 8,

            # UNROLL THE SPOUT 2 CHAIN (Matches sid = 8 + k)
            S == "S_0_1" ~ 9,
            S == "S_0_2" ~ 10,
            S == "S_0_3" ~ 11,
            S == "S_0_4" ~ 12,
            S == "S_CR" ~ 99, S == "S_CN" ~ 100,
            .default = 1
        )
    ) %>%
    filter(!is.na(A), !is.na(S)) %>%
    mutate(next_S = lead(S, default = 1))

message("trimming baseline sessions per animal...")

coded_d <- coded_d %>%
    group_by(ID) %>%
    mutate(
        is_baseline = (droga == "na_na_na_na"),
        baseline_rank = if_else(
            is_baseline,
            dense_rank(n_sesion),
            NA_integer_
        ),
        max_baseline = max(baseline_rank, na.rm = TRUE)
    ) %>%
    filter(!is_baseline | baseline_rank >= (max_baseline - 2)) %>%
    ungroup() %>%
    select(-is_baseline, -baseline_rank, -max_baseline)

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

# 4. DATA MAPPING & SESSION INDEXING ----
animal_map <- tibble(ID = animal_physics$ID) %>% mutate(animal_idx = row_number())

sessions_df <- coded_d %>%
    left_join(animal_map, by = "ID") %>%
    mutate(
        phys_ctx_id = case_when(
            true_context == "C_T" ~ 1,
            true_context == "C_S2a" ~ 2,
            true_context == "C_S2b" ~ 3,
            true_context == "C_S3a" ~ 4,
            TRUE ~ 5
        ),
        treatment_category = case_when(
            droga == "na_na_na_na" ~ "Baseline_NoInj",
            droga == "veh_na_na_na" ~ "Vehicle",
            TRUE ~ "Active_Drug"
        )
    ) %>%
    group_by(animal_idx, n_sesion) %>%
    summarise(
        phys_ctx = first(phys_ctx_id),
        cog_ctx = case_when(
            phys_ctx == 1 ~ 1, # context 1 (C_T)
            phys_ctx %in% 2:3 ~ 2, # context 2 (C_S2a/b)
            TRUE ~ 3
        ),
        treatment_name = first(treatment_category),
        actual_drug = first(droga),
        .groups = "drop"
    )

treatment_levels <- c("Baseline_NoInj", "Vehicle", "Active_Drug")
sessions_df <- sessions_df %>%
    mutate(
        drug_id = match(treatment_name, treatment_levels)
    )

drug_map <- tibble(
    drug_name = treatment_levels,
    drug_id = 1:3
)

compressed_steps <- coded_d %>%
    left_join(animal_map, by = "ID") %>%
    mutate(
        # 1. Identify the standard consecutive blocks
        base_run_id = data.table::rleid(animal_idx, n_sesion, S, A)
    ) %>%
    group_by(base_run_id) %>%
    summarise(
        animal_idx = first(animal_idx),
        n_sesion = first(n_sesion),
        S = first(S),
        A = first(A),
        next_S = last(next_S),
        weight = n(),
        .groups = "drop"
    ) %>%
    # mutate(weight = if_else(weight > 400, 400, weight)) %>%
    arrange(base_run_id)

session_pointers <- compressed_steps %>%
    ungroup() %>%
    mutate(global_idx = row_number()) %>% # Assigns 1 to 90303
    group_by(animal_idx, n_sesion) %>%
    summarise(
        start_idx = min(global_idx),
        end_idx = max(global_idx),
        .groups = "drop"
    )

animal_pointers <- session_pointers %>%
    ungroup() %>%
    mutate(global_session_idx = row_number()) %>% # Assigns 1 to 184
    group_by(animal_idx) %>%
    summarise(
        s_start = min(global_session_idx),
        s_end = max(global_session_idx),
        .groups = "drop"
    )

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
    ID_IDLE = 1, ID_WAIT = 1, ID_LICK1 = 5, ID_LICK2 = 6,
    ID_REWARD_STATE = 99, ID_NOREWARD_STATE = 100,
    grainsize = ceiling(nrow(animal_map) / (3 * 2))
)
write_rds(stan_data, "../data/processed/stan_behavior_data.rds")

message("loading model...")
# SAFETY UPDATE: Removed -march=native
# Arch Linux + Zen 5 + AVX512 + reduce_sum is creating alignment crashes.
# -O3 is stable and fast enough.
mod <- cmdstan_model(
    "beta_bernoulli_model.stan",
    force_recompile = TRUE,
    cpp_options = list(
        stan_threads = TRUE
    )
)


# # Threading: 3 Chains x 2 Threads
# fit <- mod$sample(
#     data = stan_data,
#     chains = 4,
#     parallel_chains = 4,
#     threads_per_chain = 3,
#     iter_warmup = 1000,
#     iter_sampling = 1000,
#     max_treedepth = 12,
#     adapt_delta = 0.90,
#     refresh = 1,
#     init = 0.1
# )

fit <- mod$pathfinder(
    data = stan_data,
    num_paths = 4,
    single_path_draws = 1000,
    num_threads = 4
)

dir.create("../results", showWarnings = FALSE)
fit$save_object("../results/pathfinder_model.rds")
print(fit$summary("mu_kappa"))
print("---------------------")
print(fit$summary(variable = "epsilon"))
