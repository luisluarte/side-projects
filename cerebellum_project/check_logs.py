import json

log_path = r"C:\Users\DCCS5\.gemini\antigravity\brain\1d8f9958-fd49-4502-b57b-97a7887eb7ad\.system_generated\logs\transcript.jsonl"

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        if "2026-08-30" in line:
            if "W_exp" in line or "df_n30$Boundary" in line or "confusion" in line.lower() or "stan_data" in line:
                data = json.loads(line)
                created_at = data.get("created_at", "")
                if "17:" in created_at or "21:" in created_at or "22:" in created_at or "16:" in created_at:
                    content = data.get("content", "")
                    if "W_exp" in content or "resp" in content:
                        print(f"--- {created_at} ---")
                        # print only snippets containing W_exp or resp
                        lines = content.split('\n')
                        for i, l in enumerate(lines):
                            if "W_exp" in l or "resp" in l:
                                start = max(0, i-2)
                                end = min(len(lines), i+3)
                                print("\n".join(lines[start:end]))
                                print("...")
