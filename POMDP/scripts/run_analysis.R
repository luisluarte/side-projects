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

# Physics calibration -----
# the model uses 'ideal' conditions and then determines deviations from it
# the most relevant mechanic in the task is to get a reward
# to do so you require some time in nosepoke + travel to spout + FR5
# in order to get good baseline level we use training data from last 3 session
# and determine this time cost as follows:
# nosepoke = 50 ms (this is a constant, set by the task)
# travel time = the 10th percentile of the distribution over the lengths a_LP -> first lick
# FR5 = the 10th percetile of the ILI within FR5

# lick costs ----
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
        mean_ili = mean(ILI)
    )
fr5_cost

# travel costs ----
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
        fast_travel = quantile(x = travel_time, 0.10, na.rm = TRUE)
    )
travel_cost
