import os
import re
import subprocess
import pandas as pd

# Define the grid
grid_configs = [
    (10, 50),
    (5, 25)
]

cpp_file = "src/cpp/reservoir_loocv_cmaes.cpp"
extract_script = "src/R/manifold_extraction.R"
tournament_script = "src/R/hierarchical_loocv_tournament.R"
results_file = "results/tables/hbm_manifold_loocv_results.csv"
grid_log = "results/tables/grid_search_results_small.csv"

# First, modify R scripts to set n=10 and use set.seed(42) to ensure consistent subjects
for script in [extract_script, tournament_script]:
    with open(script, "r") as f:
        content = f.read()
    content = re.sub(r"sample_participants <- sample\(participants, \d+\)", "sample_participants <- sample(participants, 10)", content)
    # Ensure set.seed is present before sampling
    if "set.seed(42)" not in content:
        content = content.replace("sample_participants <-", "set.seed(42)\nsample_participants <-")
    
    if script == tournament_script:
        content = re.sub(r"for \(fold in 1:\d+\)", "for (fold in 1:10)", content)
        
    with open(script, "w") as f:
        f.write(content)

# Initialize grid log
with open(grid_log, "w") as f:
    f.write("N_MF,N_GC,M3_Mean_NLL,M2_Mean_NLL,M1_Mean_NLL,M3_Mean_PRAUC\n")

for n_mf, n_gc in grid_configs:
    print(f"\\n--- Running Grid: N_MF={n_mf}, N_GC={n_gc} ---")
    
    # 1. Edit C++ file
    with open(cpp_file, "r") as f:
        cpp_code = f.read()
    cpp_code = re.sub(r"int N_GC = \d+;", f"int N_GC = {n_gc};", cpp_code)
    cpp_code = re.sub(r"int N_MF = \d+;", f"int N_MF = {n_mf};", cpp_code)
    with open(cpp_file, "w") as f:
        f.write(cpp_code)
    
    # 2. Run extraction
    subprocess.run(["Rscript", extract_script], check=True)
    
    # 3. Run tournament
    subprocess.run(["Rscript", tournament_script], check=True)
    
    # 4. Extract metrics
    df = pd.read_csv(results_file)
    m3_nll = df['M3_Topo_NLL'].mean()
    m2_nll = df['M2_RWCF_NLL'].mean()
    m1_nll = df['M1_WSLS_NLL'].mean()
    m3_prauc = df['M3_Topo_PRAUC'].mean()
    
    res_str = f"{n_mf},{n_gc},{m3_nll:.2f},{m2_nll:.2f},{m1_nll:.2f},{m3_prauc:.4f}\\n"
    print(f"Result: {res_str.strip()}")
    
    with open(grid_log, "a") as f:
        f.write(res_str)

print("\\nGrid Search Complete!")
