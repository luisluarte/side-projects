import json

log_path = r"C:\Users\DCCS5\.gemini\antigravity\brain\1d8f9958-fd49-4502-b57b-97a7887eb7ad\.system_generated\logs\transcript.jsonl"
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        if "241" in line:
            data = json.loads(line)
            content = data.get("content", "")
            if "241" in content and "elpd" in content.lower():
                print(f"--- {data.get('created_at')} ---")
                lines = content.split('\n')
                for l in lines:
                    if "241" in l or "elpd" in l.lower() or "M012" in l:
                        print(l)
