# libs ----
pacman::p_load(
    tidyverse,
    ggplot2,
    furrr
)
# source lickometer library
devtools::source_url("https://github.com/lab-cpl/lickometer-library/blob/main/src/lickometer_functions_compilate.R?raw=TRUE")

# set working directory
setwd(this.path::here())

# load data ----
data_path <- "../data/lickometer_raw/"
metadata <- "../metadata/lickometer_metadata.csv"

d <- load_experiment(metadataFileName = metadata, data_directory_path = data_path) %>%
    mutate(
        true_context = paste(
            str_extract(estimulo_spout_1, pattern = "[0-9]+prob"),
            str_extract(estimulo_spout_2, pattern = "[0-9]+prob"),
            sep = "-"
        ),
        true_context = case_when(
            true_context == "100prob-100prob" ~ "C_T",
            true_context == "50prob-100prob" ~ "C_S2a",
            true_context == "100prob-50prob" ~ "C_S2b",
            true_context == "25prob-50prob" ~ "C_S3a",
            true_context == "50prob-25prob" ~ "C_S3b"
        )
    )
d

# create data ----
# set timestamp relative for each animal
TIME_STEP <- 25
CORES <- parallel::detectCores()
plan(multisession, workers = CORES)
create_data <- d %>%
    group_by(ID, n_sesion) %>%
    group_split() %>%
    map(., function(dat) {
        ## status ----
        # first part to set the nosepoke time
        # creates groups for nosepoking instances
        nosepoke <- dat %>%
            mutate(
                timestamp = tiempo - min(tiempo),
                timestamp_discrete = trunc(timestamp / TIME_STEP) * TIME_STEP,
                nosepoke = if_else(actividad == -1, TRUE, FALSE),
                evento = if_else(nosepoke == TRUE, "-1", evento),
                nosepoke_time = if_else(nosepoke, timestamp, 0),
                nosepoke_time = if_else(lag(nosepoke) == FALSE, 0,
                    nosepoke_time - lag(nosepoke_time, n = 1)
                ) %>% replace_na(., 0) %>% pmax(., 0),
                nosepoke_group = data.table::rleid(nosepoke)
            ) %>%
            ungroup() %>%
            group_by(nosepoke_group) %>%
            # only nosepoke instances are non-zero timings
            # this is to determine when the nosepoke was valid
            mutate(nosepoke_time = cumsum(nosepoke_time)) %>%
            ungroup() %>%
            group_by(sensor) %>%
            # first part to determine if task is armed
            # task is not armed when an event is triggered
            # task is armed when animal is on nosepoke > 50 ms
            # for now everything else in undetermined
            mutate(
                task_armed = case_when(
                    evento != lag(evento) & nosepoke == FALSE ~ FALSE,
                    nosepoke_time >= 50 ~ TRUE
                ),
                # this is to detect if animal is idle
                # assumption is : animal is not licking for 250 ms or more
                # the its idle
                idle = if_else(actividad != lag(actividad) & # just in case of log error
                    nosepoke == FALSE & # do not consider time in nosepoke as idle
                    timestamp - lag(timestamp) >= TIME_STEP * 100, # threshold for being idle
                TRUE, FALSE
                ) %>% replace_na(., TRUE), # idle is the default state
                # this is to detect if licking activity is related to a reward
                rewarded = case_when(
                    evento != lag(evento) & exito != lag(exito) & nosepoke == FALSE & lag(nosepoke == FALSE) ~ TRUE,
                    evento != lag(evento) & exito == lag(exito) & nosepoke == FALSE & lag(nosepoke == FALSE) ~ FALSE,
                    nosepoke == TRUE ~ FALSE
                ) %>% zoo::na.locf(., na.rm = FALSE) %>% replace_na(., FALSE)
            ) %>%
            ungroup() %>%
            group_by(sensor, evento) %>%
            mutate(
                # this is to detect when the animal is doing the FR5 part
                licks_rel = actividad - min(actividad) + 1,
                accum = if_else(licks_rel <= 4 &
                    nosepoke == FALSE, TRUE, FALSE)
            ) %>%
            ungroup() %>%
            mutate(
                # fill down
                # when task becomes not armed it will stay that way
                # until it becomes armed again, it will stay armed
                # until it becomes not armed again ...
                task_armed = zoo::na.locf(task_armed, na.rm = FALSE) %>%
                    replace_na(., TRUE) # task armed by default
            ) %>%
            ungroup()
        ## sampling ----
        sampling_data <- tibble(timestamp_discrete = seq(0, max(nosepoke$timestamp),
            by = TIME_STEP
        ))
        discrete_data <- sampling_data %>%
            left_join(., nosepoke, by = c("timestamp_discrete")) %>%
            ungroup() %>%
            mutate(
                # datum is a record, non datum are derived records
                is_datum = if_else(is.na(nosepoke), FALSE, TRUE),
                task_armed = zoo::na.locf(task_armed, na.rm = FALSE),
                idle = zoo::na.locf(idle, na.rm = FALSE, fromLast = TRUE),
                rewarded = zoo::na.locf(rewarded, na.rm = FALSE),
                nosepoke = zoo::na.locf(nosepoke, na.rm = FALSE, fromLast = TRUE),
                ID = zoo::na.locf(ID, na.rm = FALSE),
                sensor = zoo::na.locf(sensor, na.rm = FALSE),
                accum = zoo::na.locf(accum, na.rm = FALSE, fromLast = TRUE),
                licks_rel = zoo::na.locf(licks_rel, na.rm = FALSE, fromLast = TRUE),
                n_sesion = zoo::na.locf(n_sesion, na.rm = FALSE),
                droga = zoo::na.locf(droga, na.rm = FALSE),
                dosis = zoo::na.locf(dosis, na.rm = FALSE),
                tipo_recompensa = zoo::na.locf(tipo_recompensa, na.rm = FALSE),
                true_context = zoo::na.locf(true_context, na.rm = FALSE)
            ) %>%
            select(
                ID, n_sesion, droga, dosis, true_context, tipo_recompensa,
                is_datum, timestamp_discrete, nosepoke, task_armed,
                idle, rewarded, accum, sensor, evento, licks_rel,
                timestamp, is_datum
            ) %>%
            ## states and action ----
            mutate(
                # this is the state definitions
                S = case_when(
                    idle == TRUE & nosepoke == FALSE ~ "S_I",
                    nosepoke == TRUE & task_armed == FALSE ~ "S_P1",
                    nosepoke == TRUE & task_armed == TRUE ~ "S_P2",
                    nosepoke == FALSE & task_armed == TRUE & sensor == 0 & accum == TRUE ~ paste("S", licks_rel, "0", sep = "_"),
                    nosepoke == FALSE & task_armed == TRUE & sensor == 1 & accum == TRUE ~ paste("S", "0", licks_rel, sep = "_"),
                    accum == FALSE & nosepoke == FALSE & rewarded == TRUE ~ "S_CR",
                    accum == FALSE & nosepoke == FALSE & rewarded == FALSE ~ "S_CN",
                    nosepoke == FALSE & task_armed == TRUE ~ "S_Armed",
                    .default = "S_I"
                ),
                # this is the action definitions
                A = case_when(
                    lead(S) == "S_P1" ~ "a_P",
                    lead(S) == "S_P2" ~ "a_SP",
                    S %in% c("S_P1", "S_P2") & !(lead(S) %in% c("S_P1", "S_P2")) ~ "a_LP",
                    grepl(pattern = "S_[0-9]_[0-9]", x = lead(S)) & lead(S) != S & sensor == 0 ~ "a_L1",
                    grepl(pattern = "S_[0-9]_[0-9]", x = lead(S)) & lead(S) != S & sensor == 1 ~ "a_L2",
                    lead(S) %in% c("S_CR", "S_CN") & sensor == 0 ~ "a_L1",
                    lead(S) %in% c("S_CR", "S_CN") & sensor == 1 ~ "a_L2",
                    lead(S) == "S_I" ~ "a_W"
                ),
                # this is to fill lick related data
                # a sequence a_L1 -> a_W -> a_L1, is representing a lick 25ms.
                # then another lick an so on , number of a_W between a_L1
                # is the interlick interval discretized approximation
                A = if_else(is.na(A) & grepl(pattern = "S_[0-9]_[0-9]|S_CR|S_CN|S_Armed", x = S), "a_W", A)
            ) %>%
            select(
                ID, is_datum, timestamp, timestamp_discrete, n_sesion,
                droga, dosis, true_context, tipo_recompensa, A, S
            )
        return(discrete_data)
    })

# save data ----
saveRDS(object = bind_rows(create_data), file = "../data/processed/discrete_data.rds")
