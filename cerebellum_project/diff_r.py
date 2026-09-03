import sys
import re

with open(r'C:\Users\DCCS5\.gemini\antigravity\brain\1d8f9958-fd49-4502-b57b-97a7887eb7ad\scratch\run_final_showdown_vopt.R') as f:
    r1 = f.read()
with open(r'src\r\stan_model_fit.R') as f:
    r2 = f.read()

def get_stan_data(script_content):
    match = re.search(r'stan_data_m012\s*<-\s*list\((.*?)\)', script_content, re.DOTALL)
    if not match:
        match = re.search(r'stan_data\s*<-\s*list\((.*?)\)', script_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return 'NOT FOUND'

print('--- SHOWDOWN STAN DATA ---')
print(get_stan_data(r1))
print('--- CURRENT STAN DATA ---')
print(get_stan_data(r2))
