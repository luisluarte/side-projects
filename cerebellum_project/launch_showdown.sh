#!/bin/bash
sudo Rscript -e "install.packages(c('pROC', 'PRROC', 'lme4'), repos='https://cloud.r-project.org')"
Rscript /home/DCCS5/cerebellum_project/run_n30_showdown.R > /home/DCCS5/cerebellum_project/showdown_N30_final.out 2>&1
