pacman::p_load(
    tidyverse, cmdstanr, posterior, parallel, data.table, this.path
)

setwd(this.path::here())

# data preparation ----
# loads raw data and maps behavioral events into a discrete state-action category.
d <- readRDS("../data/processed/discrete_data.rds")

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

compressed_d <- coded_d %>%
    mutate(
        outcome_event = next_S %in% c(99, 100),
        run_id = rleid(ID, n_sesion, S, A, next_S, outcome_event)
    ) %>%
    group_by(run_id) %>%
    summarise(
        ID = first(ID), n_sesion = first(n_sesion), droga = first(droga),
        true_context = first(true_context), S = first(S), A = first(A),
        next_S = first(next_S), weight = n(), .groups = "drop"
    )

# mapping and design constraints ----
# identifies the experimental support and restricts naive to the baseline context.
animal_map <- tibble(ID = unique(compressed_d$ID)) %>% mutate(animal_idx = row_number())

compressed_d <- compressed_d %>%
    mutate(droga_mapped = case_when(
        droga == "na_na_na_na" ~ "Naive",
        droga == "veh_na_na_na" ~ "Vehicle",
        TRUE ~ droga
    ))

drug_map <- tibble(droga = unique(compressed_d$droga_mapped)) %>%
    mutate(is_naive = droga == "Naive") %>%
    arrange(desc(is_naive)) %>%
    mutate(drug_id = row_number()) %>%
    select(-is_naive)

compressed_d <- compressed_d %>%
    left_join(animal_map, by = "ID") %>%
    mutate(
        phys_ctx_id = case_when(
            true_context == "C_T" ~ 1, true_context == "C_S2a" ~ 2,
            true_context == "C_S2b" ~ 3, true_context == "C_S3a" ~ 4, TRUE ~ 5
        ),
        cog_ctx_id = case_when(
            droga_mapped == "Naive" ~ 1,
            phys_ctx_id == 1 ~ 1,
            phys_ctx_id %in% 2:3 ~ 2,
            TRUE ~ 3
        )
    )

# session indexing ----
# constructs a fiber bundle of sessions nested within subject traits.
sessions_df <- compressed_d %>%
    group_by(animal_idx, n_sesion) %>%
    summarise(
        start_idx = min(row_number()), end_idx = max(row_number()),
        phys_ctx = first(phys_ctx_id), cog_ctx = first(cog_ctx_id),
        drug_id = drug_map$drug_id[drug_map$droga == first(droga_mapped)],
        .groups = "drop"
    ) %>%
    group_by(animal_idx) %>%
    mutate(local_session_idx = row_number()) %>%
    ungroup()

max_sessions <- max(sessions_df$local_session_idx)
animal_sessions <- sessions_df %>%
    group_by(animal_idx) %>%
    summarise(s_start = min(row_number()), s_end = max(row_number()))

# publication ready execution ----
# pathfinder initialization followed by high-resolution mcmc sampling.
N_ANIMALS <- length(unique(compressed_d$ID))
N_DRUGS <- nrow(drug_map)
N_CORES <- detectCores()
threads_per_chain <- floor(N_CORES / 4)
opt_grainsize <- floor(N_ANIMALS / threads_per_chain)

stan_data <- list(
    N_animals = N_ANIMALS,
    N_sessions_total = nrow(sessions_df),
    N_max_sessions_per_animal = max_sessions,
    N_compressed_steps = nrow(compressed_d),
    N_drugs = N_DRUGS,
    N_physics_contexts = 5,
    N_cognitive_contexts = 3,
    sessions_per_animal_start = animal_sessions$s_start,
    sessions_per_animal_end = animal_sessions$s_end,
    session_physics_context = sessions_df$phys_ctx,
    session_cognitive_context = sessions_df$cog_ctx,
    session_drug = sessions_df$drug_id,
    state_id = compressed_d$S,
    action_id = compressed_d$A,
    next_state_id = compressed_d$next_S,
    weight = compressed_d$weight,
    start_idx = sessions_df$start_idx,
    end_idx = sessions_df$end_idx,
    N_actions = 6,
    N_states = 100,
    Q_star = array(0, dim = c(5, N_ANIMALS, 100, 6)),
    context_probs = matrix(c(0.99, 0.99, 0.50, 0.99, 0.99, 0.50, 0.25, 0.50, 0.50, 0.25), ncol = 2, byrow = T),
    ID_IDLE = 1, ID_WAIT = 1, ID_LICK1 = 5, ID_LICK2 = 6,
    ID_REWARD_STATE = 99, ID_NOREWARD_STATE = 100, grainsize = opt_grainsize
)

mod <- cmdstan_model("pomdp_model.stan", cpp_options = list(stan_threads = TRUE))

# high-resolution sampling ----
# executes nuts algorithm to map the stationary posterior manifold.
fit <- mod$sample(
    data = stan_data,
    chains = 4, parallel_chains = 4, threads_per_chain = threads_per_chain,
    init = 0,
    iter_warmup = 2000, iter_sampling = 1000, adapt_delta = 0.95,
    max_treedepth = 12, refresh = 1
)

# final persistence ----
# internalizes the stan draws and saves the full object to disk.
dir.create("../results", showWarnings = FALSE)
fit$save_object("../results/fit_full_volatility_final.rds")
