# libs ----
pacman::p_load(
    tidyverse,
    rayshader,
    ggplot2
)

setwd(this.path::here())

# helper functions ----
plot_embedding <- function(embedding) {
    frame <- map_dfr(embedding, ~ as_tibble(t(.x)) %>%
        set_names(c("x", "y", "z")))
    gg_plot <- frame %>%
        ggplot(aes(x = x, y = y, fill = z)) +
        stat_density_2d(
            geom = "polygon",
            aes(fill = after_stat(level)),
            n = 200,
            bins = 50
        ) +
        coord_equal() +
        ggpubr::theme_classic2() +
        theme(legend.position = "none") +
        viridis::scale_fill_viridis()
    three_dim_p <- plot_gg(
        ggobj = gg_plot,
        multicore = TRUE,
        raytrace = TRUE
    )
    return(gg_plot)
}
plot_embedding(generate_embedding(runif(n = 100, min = 0, max = 1), 3, 3))

# main algorithm ----

# take a number series
generate_embedding <- function(number_series, tau, m) {
    burn_in_period <- 1 + (m - 1) * tau
    if (length(number_series) < burn_in_period) {
        return(list())
    }
    valid_idx <- burn_in_period:length(number_series)
    embedding <- map(valid_idx, function(t) {
        offsets <- (0:(m - 1)) * tau
        indices <- t - offsets
        number_series[indices]
    })

    return(embedding)
}


# test ----
