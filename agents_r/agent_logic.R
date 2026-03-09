# libs ----
pacman::p_load(
    tidyverse
)

# agent ----

generate_random_markov <- function(size) {
    m <- matrix(
        data = runif(size * size, min = 0, max = 1),
        nrow = size, ncol = size
    )
    m / apply(m, MARGIN = 1, sum)
}

walk_markov <- function(markov, steps) {
    node_path <- c()
    init_idx <- sample(1:dim(markov)[1], size = 1, replace = TRUE)
    row_probs <- markov[init_idx, ]
    for (i in 1:steps) {
        next_node <- sample(
            1:dim(markov)[1],
            size = 1,
            replace = TRUE,
            prob = row_probs
        )
        row_probs <- markov[init_idx, ]
        node_path[i] <- next_node
    }
    return(node_path)
}

walk_markov(t, 100)
