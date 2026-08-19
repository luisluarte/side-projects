import re

file_path = "src/cpp/reservoir_loocv_cmaes.cpp"
with open(file_path, "r") as f:
    content = f.read()

# Replace alpha_LTD and alpha_LTP with eta and lambda
content = content.replace("double alpha_LTD   = phi_15d[10];", "double eta         = phi_15d[10];")
content = content.replace("double alpha_LTP   = phi_15d[11];", "double lambda      = phi_15d[11];")

# We need to replace the IO_error and weight update logic in both functions.
# The logic starts around line: double IO_error = ...

replacement = """
    // Fix: Standard Signed Error for Ridge Regression Readout
    double IO_error = ((double)out - 0.5) * 2.0 - ((ch == 1) ? y_PC1 : y_PC2); // mapped out to [-1, 1] for symmetry
    rpe_abs_prev = std::abs(IO_error);

    for (int i = 0; i < N_GC; ++i) {
        if (ch == 1) {
            double delta_w = (eta / N_GC) * IO_error * z_GC_prev[i] - lambda * w_PF1[i];
            w_PF1[i] += delta_w;
        } else {
            double delta_w = (eta / N_GC) * IO_error * z_GC_prev[i] - lambda * w_PF2[i];
            w_PF2[i] += delta_w;
        }
        z_GC_prev[i] = z_GC_curr[i];
    }
"""

# Regex to find the old block and replace it.
pattern = re.compile(r"double IO_error =.*?z_GC_prev\[i\] = z_GC_curr\[i\];\n    \}", re.DOTALL)
content = pattern.sub(replacement.strip(), content)

with open(file_path, "w") as f:
    f.write(content)
print("Applied Ridge Regression L2 logic in C++.")
