text = open('bvk_continuous.stan', 'r').read()
text = text.replace('lambda_mli[s]', '(lambda_gc[s] * 1.5)')
text = text.replace('eta_mli[s]', 'eta_gc[s]')
open('bvk_continuous.stan', 'w', newline='\n').write(text)

text = open('bvk_continuous_gq.stan', 'r').read()
text = text.replace('lambda_mli[s]', '(lambda_gc[s] * 1.5)')
text = text.replace('eta_mli[s]', 'eta_gc[s]')
open('bvk_continuous_gq.stan', 'w', newline='\n').write(text)
