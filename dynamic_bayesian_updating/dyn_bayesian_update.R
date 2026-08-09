# libs ----
pacman::p_load(
    tidyverse
)

setwd(this.path::here())
palette("Okabe-Ito")

# data ----
btc_data <- crypto2::crypto_list() %>%
    filter(slug == "bitcoin") %>%
    crypto2::crypto_history()

# metrics ----
data_ema <- btc_data %>%
    select(timestamp, close) %>%
    mutate(
        log_close = log(close),
        log_close_7 = lead(log_close, n = 30) - log_close,
        ema_100 = log_close - zoo::rollmean(log_close, k = 100, fill = NA, align = "right"),
        ema_50 = log_close - zoo::rollmean(log_close, k = 50, fill = NA, align = "right"),
        ema_10 = log_close - zoo::rollmean(log_close, k = 10, fill = NA, align = "right"),
        M = lubridate::month(timestamp)
    )
data_ema

mdl_data <- data_ema %>%
    drop_na()
mdl_data

mdl_0 <- lm(
    data = mdl_data,
    log_close_7 ~ 1
)

mdl_1 <- lm(
    data = mdl_data,
    log_close_7 ~ ema_50
)

mdl_2 <- lm(
    data = mdl_data,
    log_close_7 ~ ema_10 * ema_50
)

mdl_3 <- lm(
    data = mdl_data,
    log_close_7 ~ ema_10 * ema_50 * ema_100 * M
)

performance::test_likelihoodratio(
    mdl_0,
    mdl_1,
    mdl_2,
    mdl_3
)

summary(mdl_3)

sigma_sq <- summary(mdl_3)$sigma^2
pred_dat <- mdl_data %>%
    mutate(
        .preds = predict(mdl_3),
        corr_pred = exp(.preds + (sigma_sq / 2))
    )


pred_dat %>%
    ggplot(aes(
        timestamp, .preds
    )) +
    geom_point() +
    geom_point(aes(timestamp, log_close_7), color = "red")
