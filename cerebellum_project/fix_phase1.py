import re

def patch(file):
    with open(file, 'r', encoding='ascii') as f:
        text = f.read()

    # Move eta_mli_s and lambda_mli_s to the top of the loop
    s_idx = '[s]' if '_gq' not in file else '[s]'
    
    loop_start = '      for (t in start_t:end_t) {'
    new_loop_start = f'      for (t in start_t:end_t) {{\n        real eta_mli_s = eta_gc{s_idx};\n        real lambda_mli_s = lambda_gc{s_idx} * 1.5;'

    text = text.replace(loop_start, new_loop_start)
    text = text.replace('lambda_mli[s]', 'lambda_mli_s')
    text = text.replace('lambda_mli_s = lambda_gc[s]', f'lambda_mli_s_redundant = lambda_gc{s_idx}')
    
    with open(file, 'w', encoding='ascii', newline='\n') as f:
        f.write(text)

patch('bvk_continuous.stan')
patch('bvk_continuous_gq.stan')
