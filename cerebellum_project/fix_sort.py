import sys
with open('src/r/stan_model_fit.R', 'r') as f:
    content = f.read()
content = content.replace('dat_raw <- read_csv("../../data/raw/behavioral_compilate.csv") %>%', 'dat_raw <- read_csv("../../data/raw/behavioral_compilate.csv") %>%\n  arrange(participant_id, nt) %>%')
with open('src/r/stan_model_fit.R', 'w') as f:
    f.write(content)
