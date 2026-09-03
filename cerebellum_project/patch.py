import sys
import re

content = open('src/stan/m012_ss3.stan').read()
content = content.replace('vector[4]', 'vector[32]')
content = content.replace('matrix[N_subj, 4]', 'matrix[N_subj, 32]')
content = content.replace('for (i in 1:4)', 'for (i in 1:32)')
content = content.replace('/ 3.0)', '/ 31.0)')
content = content.replace('[1:2]', '[1:16]')
content = content.replace('[3:4]', '[17:32]')

open('src/stan/m012_ss3.stan', 'w').write(content)
