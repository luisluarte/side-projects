# libs ----
pacman::p_load(
    tidyverse,
    rayshader,
    ggplot2,
    fpc,
    numbers
)

setwd(this.path::here())

# helper functions ----
plot_embedding <- function(embedding) {
    frame <- map_dfr(embedding, ~ as_tibble(t(.x)) %>%
        set_names(c("x", "y")))
    gg_plot <- frame %>%
        ggplot(aes(x = x, y = y)) +
        geom_path(color = "gray70") +
        geom_point() +
        coord_equal() +
        ggpubr::theme_classic2() +
        theme(legend.position = "none") +
        viridis::scale_fill_viridis()
    return(gg_plot)
}

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

generate_connectivity <- function(embedding, at_field) {
    to_mat <- map_dfr(embedding, ~ as_tibble(t(.x)) %>%
        set_names(c("x", "y"))) %>%
        mutate(
            original_id = row_number()
        )
    embedding_clust <- pamk(
        select(to_mat, x, y),
        krange = 1:10,
        usepam = TRUE
    )
    distances <- to_mat %>%
        as_tibble() %>%
        mutate(
            clust_id = embedding_clust$pamobject$clustering
        ) %>%
        group_by(clust_id) %>%
        group_split() %>%
        map_dfr(., function(dat) {
            d <- select(dat, c(x, y))
            dist_mat <- apply(as.matrix(dist(as.matrix(d), method = "euclidean")),
                MARGIN = 2, FUN = mean
            )
            out <- d %>%
                ungroup() %>%
                mutate(
                    norm_mean_distance = if_else(rep(at_field, nrow(d)) >= 0,
                        (dist_mat / max(dist_mat)) * at_field,
                        (1 - (dist_mat / max(dist_mat))) * abs(at_field)
                    ),
                    original_id = dat$original_id
                ) %>%
                arrange(original_id)
            return(out)
        })
    return(distances)
}

convolution_kernel <- function(old_embedding, new_embedding) {
    n_old <- nrow(old_embedding)
    n_new <- nrow(new_embedding)
    m <- ncol(old_embedding)
    t_old <- seq(0, 1, length.out = n_old)
    t_new <- seq(0, 1, length.out = n_new)
    bandwidth <- 1.5 * (1 / (n_old - 1))
    apply_kernel <- function(y_values) {
        smoothed <- ksmooth(
            x = t_old,
            y = y_values,
            kernel = "normal",
            bandwidth = bandwidth,
            x.points = t_new
        )

        # Handle edge case: if bandwidth is tiny, ksmooth might return NA at boundaries.
        # We fill NAs with the nearest valid value (0 or 1 edge).
        y_out <- smoothed$y
        y_out[is.na(y_out)] <- mean(y_values, na.rm = TRUE) # Fallback
        return(y_out)
    }
    # A. Resample Position (The "Ghost" of the old body)
    # apply(..., 2, func) applies the kernel to each column (dimension) separately
    old_embedding_resampled <- apply(old_embedding, 2, apply_kernel)

    # B. Resample Rigidity (The "Connectivity Field")
    rigidity_resampled <- apply_kernel(old_embedding$norm_mean_distance)

    # Clamp rigidity to [0, 1] just in case the kernel overshoots slightly
    rigidity_resampled <- pmin(pmax(rigidity_resampled, 0), 1)

    # 4. The Physics Update (Viscoelastic Deformation)
    # Force = Target (New) - Current (Old Resampled)
    # Displacement = Force * Plasticity (1 - Rigidity)

    force_vector <- new_embedding - old_embedding_resampled
    plasticity <- 1 - rigidity_resampled

    # Note: R handles matrix * vector multiplication by column, so we replicate
    # plasticity for each dimension to ensure correct element-wise math.
    plasticity_mat <- matrix(plasticity, nrow = n_new, ncol = m, byrow = FALSE)

    merged_embedding <- as_tibble(old_embedding_resampled + (plasticity_mat * force_vector))

    return(merged_embedding)
}

old_pattern <- rnorm(n = 900, mean = 0, sd = 1)
new_pattern <- 6 * sin((2 * pi * 1:900) / 100)
old_emb <- generate_connectivity(generate_embedding(old_pattern, 1, 2), 1)
new_emb <- generate_connectivity(generate_embedding(new_pattern, 25, 2), 1)

old_emb_seq <- rev(seq(-1, 1, 0.1)) %>%
    map_dfr(., function(at_field) {
        old <- generate_connectivity(generate_embedding(old_pattern, 1, 2), at_field)
        new <- generate_connectivity(generate_embedding(new_pattern, 25, 2), 1)
        convolution_kernel(old, new) %>%
            mutate(at_field = at_field)
    })

old_emb_seq %>%
    ggplot(aes(
        x, y
    )) +
    geom_path(linewidth = 1, color = "gold") +
    geom_path(data = old_emb, color = "red", alpha = 0.5) +
    geom_path(data = new_emb, color = "blue", alpha = 0.5) +
    coord_equal() +
    viridis::scale_color_viridis() +
    theme_void() +
    theme(legend.position = "none") +
    facet_wrap(~at_field)
