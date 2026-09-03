import json
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

log_path = r'C:\Users\DCCS5\.gemini\antigravity\brain\1d8f9958-fd49-4502-b57b-97a7887eb7ad\.system_generated\logs\transcript.jsonl'
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        if 'elpd_diff' in line:
            data = json.loads(line)
            content = data.get('content', '')
            if 'elpd_diff' in content:
                print(f'--- {data.get("created_at")} ---')
                lines = content.split('\n')
                for i, l in enumerate(lines):
                    if 'elpd_diff' in l:
                        print('\n'.join(lines[i:i+3]))
                        break
