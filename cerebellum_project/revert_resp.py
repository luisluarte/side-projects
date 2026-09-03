import sys

content = open('src/r/stan_model_fit.R').read()
old_code = """# stay = 1; switch = 2
resp <- dat %>%
  pull(stay_switch) %>%
  {
    ifelse(. == "stay", 1, 2)
  }"""
new_code = """# stay = 1; switch = 2
resp <- dat %>%
  pull(Resp) %>%
  {
	  ifelse(. %in% c(1, 2), ., -999)
  }"""

if old_code in content:
    content = content.replace(old_code, new_code)
    open('src/r/stan_model_fit.R', 'w').write(content)
    print("Reverted successfully")
else:
    print("Not found")
