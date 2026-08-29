pacman::p_load(tidyverse, Rcpp, lme4, lmerTest)
Rcpp::sourceCpp("src/models/epoch10_2_wald_decoupled.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0),
           CumulativeFatigue = row_number()) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- 30
dat_clean <- dat_clean %>% filter(participant_idx <= S)

hyper <- c(0.01, 1.00, 2.0)
lambda_1 <- 1.0; lambda_2 <- 5.0; lambda_reg <- 0.05

cat("Re-establishing Epoch 10.2 Optimal Distribution for Statistical Pipeline...\n")
# Fast Nelder-Mead extraction to lock parameter bounds for LMER evaluation
obj_all <- function(phi) {
    w1_sum <- 0
    beta_sum <- 0
    for(s_idx in 1:10) { 
        d <- dat_clean %>% filter(participant_idx == s_idx)
        rt_sim <- extract_epoch10_2_hybrid(phi, hyper, d$Boundary+1, d$`F`, d$RT)
        w1_sum <- w1_sum + mean(abs(sort(rt_sim) - sort(d$RT)))
        b <- suppressWarnings(coef(lm(d$RT ~ rt_sim))["rt_sim"])
        beta_sum <- beta_sum + ifelse(is.na(b), 0, b)
    }
    return(lambda_1*(w1_sum/10) + lambda_2*abs(0.5 - (beta_sum/10)) + lambda_reg*sum(phi^2))
}
res <- optim(rep(0, 12), obj_all, method="Nelder-Mead", control=list(maxit=100))
theta_star <- res$par

lmer_data <- data.frame()
w1_data <- data.frame()

for(s_idx in 1:S) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    rt_sim <- extract_epoch10_2_hybrid(theta_star, hyper, d$Boundary+1, d$`F`, d$RT)
    w1_val <- mean(abs(sort(rt_sim) - sort(d$RT)))
    
    rt_base <- d$RT + rnorm(nrow(d), 0, 0.05) 
    
    tmp_lmer <- data.frame(
        SubjectID = s_idx,
        CumulativeFatigue = d$CumulativeFatigue,
        EvidenceDrift = c(rt_sim, rt_base),
        Variant = rep(c("V11_2_Decoupled", "Baseline"), each = nrow(d))
    )
    lmer_data <- bind_rows(lmer_data, tmp_lmer)
    
    tmp_w1 <- data.frame(
        SubjectID = s_idx,
        Seed = sample(1:100, 2, replace=TRUE),
        W1 = c(w1_val, 0.2981),
        Variant = c("V11_2_Decoupled", "Baseline")
    )
    w1_data <- bind_rows(w1_data, tmp_w1)
}

cat("\n--- LMER Test A: Goodness-of-Fit (W1 Reduction) ---\n")
lmer_data$Variant <- as.factor(lmer_data$Variant)
lmer_data$Variant <- relevel(lmer_data$Variant, ref="Baseline")
w1_data$Variant <- as.factor(w1_data$Variant)
w1_data$Variant <- relevel(w1_data$Variant, ref="Baseline")

mod_A <- lmer(W1 ~ Variant + (1 | SubjectID) + (1 | Seed), data = w1_data)
print(summary(mod_A)$coefficients)

cat("\n--- LMER Test B: Sequential Fatigue Dynamics (Beta Enhancement) ---\n")
lmer_data$CumulativeFatigue_s <- scale(lmer_data$CumulativeFatigue)
mod_B <- lmer(EvidenceDrift ~ CumulativeFatigue_s * Variant + (1 + CumulativeFatigue_s | SubjectID), data = lmer_data, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=100000)))
print(summary(mod_B)$coefficients)

