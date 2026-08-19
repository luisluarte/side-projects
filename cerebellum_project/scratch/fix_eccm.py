import re

file_path = "src/cpp/reservoir_loocv_cmaes.cpp"
with open(file_path, "r") as f:
    content = f.read()

# We need to replace the IO_error and weight update logic in both functions.
# The logic starts around line: double IO_error = ...

replacement = """
    // Fix: IO_error is the physiological Climbing Fiber spike rate (1 = Punishment, 0 = Reward)
    double IO_error = 1.0 - (double)out;
    rpe_abs_prev = IO_error; // For DDM boundary modulation if needed

    double L1_norm1 = 0.0, L1_norm2 = 0.0;
    for (int i = 0; i < N_GC; ++i) {
        L1_norm1 += std::abs(w_PF1[i]);
        L1_norm2 += std::abs(w_PF2[i]);
    }
    double eps = 1e-4;
    
    for (int i = 0; i < N_GC; ++i) {
        if (ch == 1) {
            double P_i = std::abs(w_PF1[i]) / (L1_norm1 + eps);
            double delta_w = -alpha_LTD * IO_error * z_GC_prev[i] * P_i + alpha_LTP * z_GC_prev[i];
            w_PF1[i] = std::max(0.0, w_PF1[i] + delta_w);
        } else {
            double P_i = std::abs(w_PF2[i]) / (L1_norm2 + eps);
            double delta_w = -alpha_LTD * IO_error * z_GC_prev[i] * P_i + alpha_LTP * z_GC_prev[i];
            w_PF2[i] = std::max(0.0, w_PF2[i] + delta_w);
        }
        z_GC_prev[i] = z_GC_curr[i];
    }
"""

# Regex to find the old block and replace it.
# We match from "double IO_error =" down to "z_GC_prev[i] = z_GC_curr[i];\n    }"
pattern = re.compile(r"double IO_error =.*?z_GC_prev\[i\] = z_GC_curr\[i\];\n    \}", re.DOTALL)
content = pattern.sub(replacement.strip(), content)

with open(file_path, "w") as f:
    f.write(content)
print("Fixed IO_error logic in C++.")
