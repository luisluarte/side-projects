import os
import re
import subprocess
import pandas as pd

# Define the grid
k_golgi_list = [1, 2, 4, 8]

cpp_file = "src/cpp/reservoir_loocv_cmaes.cpp"
extract_script = "src/R/manifold_extraction.R"
tournament_script = "src/R/hierarchical_loocv_tournament.R"
results_file = "results/tables/hbm_manifold_loocv_results.csv"
grid_log = "results/tables/grid_search_golgi.csv"

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
    f.write("K_Golgi,M3_Mean_NLL,M2_Mean_NLL,M1_Mean_NLL,M3_Mean_PRAUC\n")

# Set N_MF=20, N_GC=100
with open(cpp_file, "r") as f:
    cpp_code = f.read()
cpp_code = re.sub(r"int N_GC = \d+;", f"int N_GC = 100;", cpp_code)
cpp_code = re.sub(r"int N_MF = \d+;", f"int N_MF = 20;", cpp_code)
with open(cpp_file, "w") as f:
    f.write(cpp_code)

for k in k_golgi_list:
    print(f"\\n--- Running Grid: K_Golgi={k} ---")
    
    # 1. Edit C++ file to inject k
    with open(cpp_file, "r") as f:
        cpp_code = f.read()
        
    cpp_code = re.sub(r"std::vector<int>\(4\)", f"std::vector<int>({k})", cpp_code)
    cpp_code = re.sub(r"std::vector<double>\(4, 0\.0\)", f"std::vector<double>({k}, 0.0)", cpp_code)
    cpp_code = re.sub(r"for \(int k = 0; k < \d+; \+\+k\) \{", f"for (int k = 0; k < {k}; ++k) {{", cpp_code)
    cpp_code = re.sub(r"for \(int k = 0; k < \d+; \+\+k\) in_sum", f"for (int k = 0; k < {k}; ++k) in_sum", cpp_code)

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
    
    res_str = f"{k},{m3_nll:.2f},{m2_nll:.2f},{m1_nll:.2f},{m3_prauc:.4f}\\n"
    print(f"Result: {res_str.strip()}")
    
    with open(grid_log, "a") as f:
        f.write(res_str)

print("\\nGrid Search Complete!")
