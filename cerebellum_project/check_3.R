df <- read.csv('data/raw/behavioral_compilate.csv')
subset_3 <- df[!is.na(df$Resp) & df$Resp == 3, ]
print(head(subset_3))
