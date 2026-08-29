# ==============================================================================
# EXACT-R: Gaussian Process Surrogate Modeling (Phase 3 & Phase 4)
# ARD Matérn 5/2 GPR, Manifold Extraction & Contour Visualization
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from scipy.optimize import minimize
import os

# 1. Load LHS Dataset
dataset_path = "gp_lhs_dataset.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"{dataset_path} not found. Ensure Phase 1 & 2 completed.")

df = pd.read_csv(dataset_path)
print(f"Loaded {len(df)} samples from {dataset_path}")

param_cols = [
    "rho_base_mean", "tau_log_mean", "d_in", "d_fb", "d_inh", "lambda_fb",
    "f_base", "A_load", "delta_phi", "sigma_noise"
]

X_raw = df[param_cols].values
y_raw = df["Fitness_J"].values

# Parameter bounds (min, max) for scaling
bounds_dict = {
    "rho_base_mean": (0.01, 0.50),
    "tau_log_mean":  (0.50, 3.00),
    "d_in":          (0.01, 0.20),
    "d_fb":          (0.01, 0.10),
    "d_inh":         (0.05, 0.30),
    "lambda_fb":     (0.70, 0.99),
    "f_base":        (0.10, 5.00),
    "A_load":        (1.00, 10.00),
    "delta_phi":     (0.00, np.pi),
    "sigma_noise":   (0.01, 1.00)
}

lower_bounds = np.array([bounds_dict[c][0] for c in param_cols])
upper_bounds = np.array([bounds_dict[c][1] for c in param_cols])
ranges = upper_bounds - lower_bounds

# Min-Max Scale X to [0, 1]^10
X_scaled = (X_raw - lower_bounds) / ranges

# Standardize Target y to Zero Mean, Unit Variance
y_mean = np.mean(y_raw)
y_std_dev = np.std(y_raw) + 1e-12
y_scaled = (y_raw - y_mean) / y_std_dev

# 2. Gaussian Process Regression with ARD Matérn 5/2 Kernel
# Matérn 5/2 with ARD lengthscales (10 lengthscales for 10 dimensions)
matern_kernel = Matern(
    length_scale=np.ones(10),
    length_scale_bounds=(1e-2, 10.0),
    nu=2.5
)
kernel = C(1.0, (1e-3, 1e3)) * matern_kernel + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))

gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    n_restarts_optimizer=10,
    random_state=42
)

print("\nFitting ARD Matérn 5/2 Gaussian Process Regression...")
gpr.fit(X_scaled, y_scaled)
print("GPR fitting completed.")

opt_kernel = gpr.kernel_
print("\nOptimized Kernel Structure:")
print(opt_kernel)

# Extract lengthscales in scaled space [0, 1]
matern_part = opt_kernel.k1.k2
l_scaled = matern_part.length_scale
l_physical = l_scaled * ranges

# Calculate Sensitivity Score (Inverse of Scaled Lengthscale)
sensitivity_scores = 1.0 / (l_scaled ** 2)

# Create ARD Lengthscale Ledger DataFrame
ard_ledger = pd.DataFrame({
    "Dimension": range(1, 11),
    "Parameter": param_cols,
    "Physical_Range": [f"[{lower_bounds[i]:.2f}, {upper_bounds[i]:.2f}]" for i in range(10)],
    "Lengthscale_Scaled": l_scaled,
    "Lengthscale_Physical": l_physical,
    "Sensitivity_Score": sensitivity_scores
}).sort_values(by="Sensitivity_Score", ascending=False).reset_index(drop=True)

print("\n==============================================================================")
print("THE ARD LENGTHSCALE LEDGER (Parameter Sensitivity Ranking)")
print("==============================================================================")
print(ard_ledger.to_string(index=False))

ard_ledger.to_csv("ard_lengthscale_ledger.csv", index=False)
print("Saved ARD Lengthscale Ledger to ard_lengthscale_ledger.csv")

# 3. Manifold Extraction & Optimization
# Define surrogate objective function to maximize: \hat{\mathcal{J}}(\Psi) in physical domain units
def predict_fitness(psi_physical):
    psi_scaled = (psi_physical - lower_bounds) / ranges
    mu_std, sigma_std = gpr.predict(psi_scaled.reshape(1, -1), return_std=True)
    mu_phys = mu_std[0] * y_std_dev + y_mean
    var_phys = (sigma_std[0] * y_std_dev) ** 2
    return mu_phys, var_phys

def neg_predict_fitness_structural(theta, phi_fixed):
    psi = np.concatenate([theta, phi_fixed])
    mu, _ = predict_fitness(psi)
    return -mu

# Subspace indices
theta_indices = list(range(6))
phi_indices = list(range(6, 10))

bounds_theta = [(lower_bounds[i], upper_bounds[i]) for i in theta_indices]

# Regime A: Slow, heavy movement (f_base = 0.2 Hz, A_load = 8.0)
phi_regime_A = np.array([0.2, 8.0, 0.5 * np.pi, 0.1])
theta_init = 0.5 * (lower_bounds[:6] + upper_bounds[:6])

res_A = minimize(neg_predict_fitness_structural, theta_init, args=(phi_regime_A,), method="L-BFGS-B", bounds=bounds_theta)
theta_opt_A = res_A.x
fitness_opt_A = -res_A.fun

# Regime B: Fast, ballistic movement (f_base = 3.0 Hz, A_load = 2.0)
phi_regime_B = np.array([3.0, 2.0, 0.5 * np.pi, 0.1])
res_B = minimize(neg_predict_fitness_structural, theta_init, args=(phi_regime_B,), method="L-BFGS-B", bounds=bounds_theta)
theta_opt_B = res_B.x
fitness_opt_B = -res_B.fun

print("\n==============================================================================")
print("CONDITIONAL MANIFOLD EXTRACTED REGIME ADAPTATIONS")
print("==============================================================================")
print(f"Regime A (Slow, Heavy: f_base = 0.2 Hz, A_load = 8.0):")
for name, val in zip(param_cols[:6], theta_opt_A):
    print(f"  {name:15s} = {val:.4f}")
print(f"  Predicted Fitness J = {fitness_opt_A:.2f}\n")

print(f"Regime B (Fast, Ballistic: f_base = 3.0 Hz, A_load = 2.0):")
for name, val in zip(param_cols[:6], theta_opt_B):
    print(f"  {name:15s} = {val:.4f}")
print(f"  Predicted Fitness J = {fitness_opt_B:.2f}\n")

# Compute frequency transition continuum across f_base \in [0.2, 3.0]
f_grid = np.linspace(0.2, 3.0, 15)
tau_opt_list = []
lambda_opt_list = []

for f in f_grid:
    phi_f = np.array([f, 5.0 - 1.0 * f, 0.5 * np.pi, 0.1])
    res_f = minimize(neg_predict_fitness_structural, theta_init, args=(phi_f,), method="L-BFGS-B", bounds=bounds_theta)
    tau_opt_list.append(res_f.x[1])    # tau_log_mean
    lambda_opt_list.append(res_f.x[5]) # lambda_fb

# Fit conditional linear manifold relationship
poly_tau = np.polyfit(f_grid, tau_opt_list, deg=1)
poly_lambda = np.polyfit(f_grid, lambda_opt_list, deg=1)

print("Conditional Manifold Equations:")
print(f"  tau_log_mean(f_base)  = {poly_tau[0]:+.4f} * f_base + {poly_tau[1]:.4f}")
print(f"  lambda_fb(f_base)     = {poly_lambda[0]:+.4f} * f_base + {poly_lambda[1]:.4f}")

manifold_df = pd.DataFrame({
    "f_base": f_grid,
    "tau_log_mean_opt": tau_opt_list,
    "lambda_fb_opt": lambda_opt_list
})
manifold_df.to_csv("conditional_manifold_results.csv", index=False)

# 4. Surrogate Variance & Mean 2D Contour Plot
# Identify top 2 most sensitive parameters from ARD ledger
top_2_params = ard_ledger["Parameter"].values[:2]
top_2_idx = [param_cols.index(p) for p in top_2_params]

p1_name, p2_name = top_2_params
idx1, idx2 = top_2_idx

print(f"\nTop 2 Sensitive Parameters for Contour Plot: {p1_name} vs {p2_name}")

# Create 2D Grid over top 2 parameters
grid_res = 100
p1_grid = np.linspace(lower_bounds[idx1], upper_bounds[idx1], grid_res)
p2_grid = np.linspace(lower_bounds[idx2], upper_bounds[idx2], grid_res)
P1, P2 = np.meshgrid(p1_grid, p2_grid)

# Nominal base point (midpoint of remaining 8 parameters)
nominal_psi = 0.5 * (lower_bounds + upper_bounds)
# Set fixed parameters to optimal values where applicable
for i, name in enumerate(param_cols[:6]):
    if i not in top_2_idx:
        nominal_psi[i] = theta_opt_A[i]

MU_grid = np.zeros((grid_res, grid_res))
VAR_grid = np.zeros((grid_res, grid_res))

for i in range(grid_res):
    for j in range(grid_res):
        eval_psi = nominal_psi.copy()
        eval_psi[idx1] = P1[i, j]
        eval_psi[idx2] = P2[i, j]
        mu, var = predict_fitness(eval_psi)
        MU_grid[i, j] = mu
        VAR_grid[i, j] = var

# Plotting Panel
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: GP Predictive Mean \mu_*
c1 = axes[0].contourf(P1, P2, MU_grid, levels=25, cmap="viridis")
fig.colorbar(c1, ax=axes[0], label="Predictive Mean Fitness $\hat{\mathcal{J}}(\mathbf{\Psi})$")
axes[0].set_title(f"GP Predictive Mean $\mu_*$ ({p1_name} vs {p2_name})", fontsize=12, fontweight="bold")
axes[0].set_xlabel(p1_name, fontsize=11)
axes[0].set_ylabel(p2_name, fontsize=11)
axes[0].grid(True, linestyle="--", alpha=0.3)

# Panel 2: GP Predictive Variance \sigma_*^2
c2 = axes[1].contourf(P1, P2, VAR_grid, levels=25, cmap="magma")
fig.colorbar(c2, ax=axes[1], label="Predictive Variance $\sigma_*^2(\mathbf{\Psi})$")
axes[1].set_title(f"GP Predictive Variance $\sigma_*^2$ ({p1_name} vs {p2_name})", fontsize=12, fontweight="bold")
axes[1].set_xlabel(p1_name, fontsize=11)
axes[1].set_ylabel(p2_name, fontsize=11)
axes[1].grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()

# Save plot to workspace & artifact directory
fig_path_local = "gp_surrogate_contours.png"
fig_path_artifact = os.path.join(r"C:\Users\DCCS5\.gemini\antigravity\brain\1d8f9958-fd49-4502-b57b-97a7887eb7ad", "gp_surrogate_contours.png")

plt.savefig(fig_path_local, dpi=300, bbox_inches="tight")
plt.savefig(fig_path_artifact, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nRendered and saved 2D contour plot to:\n  - {fig_path_local}\n  - {fig_path_artifact}")
print("\nPhase 3 & 4 GPR analysis complete!")
