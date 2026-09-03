import sys

with open('src/r/stan_model_fit.R', 'r') as f:
    content = f.read()

content = content.replace('iter_warmup = 300', 'iter_warmup = 3')
content = content.replace('iter_sampling = 300', 'iter_sampling = 3')

with open('src/r/test_fit.R', 'w') as f:
    f.write(content)
