import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

repo_root = "." if os.path.exists("results/cohort_comparison_metrics.csv") else "../.."
results_dir = os.path.join(repo_root, "results")
figures_dir = os.path.join(repo_root, "figures")
os.makedirs(figures_dir, exist_ok=True)

df_comp = pd.read_csv(os.path.join(results_dir, "cohort_comparison_metrics.csv"))
df_params = pd.read_csv(os.path.join(results_dir, "m006_parameter_distributions.csv"))

# -------------------------------------------------------------
# 1. PARAMETER DISTRIBUTIONS (9-Panel Multi-Plot)
# -------------------------------------------------------------
params_info = [
    ('a_base', r'Baseline Boundary $a_{base}$', '#2980b9', (0, 4)),
    ('tnd', r'Non-Decision Time $t_{nd}$ (s)', '#27ae60', (0, 0.8)),
    ('v_ctx', r'Cortical Drift Scale $v_{ctx}$', '#8e44ad', (0, 5)),
    ('alpha_ctx', r'Cortical Learning Rate $\alpha_{ctx}$', '#e67e22', (0, 1)),
    ('alpha_pc', r'Purkinje Plasticity Rate $\alpha_{pc}$', '#d35400', (0, 1)),
    ('gamma', r'Cerebellar Gain $\gamma$', '#c0392b', (0, 6)),
    ('lambda_sa_temp', r'Annealing Temperature $\lambda_{temp}$', '#16a085', (0, 10)),
    ('tau_decay', r'Physical ITI Decay $\tau_{decay}$ (s)', '#34495e', (0, 10)),
    ('w_u', r'Epistemic Doubt Weight $w_u$', '#e74c3c', (0, 10))
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, (col, title, color, xlim) in enumerate(params_info):
    vals = df_params[col].clip(upper=xlim[1]*1.5)
    sns.kdeplot(vals, ax=axes[i], color=color, fill=True, alpha=0.3, linewidth=2)
    sns.rugplot(vals, ax=axes[i], color=color, alpha=0.6, height=0.08)
    med = df_params[col].median()
    q25 = df_params[col].quantile(0.25)
    q75 = df_params[col].quantile(0.75)
    
    axes[i].axvline(med, color='black', linestyle='--', linewidth=1.5, label=f'Median: {med:.2f}')
    axes[i].axvspan(q25, q75, color=color, alpha=0.1, label=f'IQR: [{q25:.2f}, {q75:.2f}]')
    axes[i].set_title(title, fontsize=13, fontweight='bold', pad=10)
    axes[i].set_xlim(xlim)
    axes[i].set_xlabel('Parameter Value', fontsize=10)
    axes[i].set_ylabel('Density', fontsize=10)
    axes[i].legend(loc='upper right', frameon=True, fontsize=9)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

plt.suptitle(r'M006 Biologically Bounded Parameter Distributions Across Cohort ($N = 128$)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "m006_parameter_distributions.png"), dpi=200, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 2. STATISTICAL COMPARISON (M006 vs Base)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: NLL Distribution & Paired Difference
sns.kdeplot(df_comp['NLL_Base'], ax=axes[0], color='#7f8c8d', fill=True, alpha=0.3, label=f"Baseline (Mean: {df_comp['NLL_Base'].mean():.1f})")
sns.kdeplot(df_comp['NLL_M006'], ax=axes[0], color='#2980b9', fill=True, alpha=0.3, label=f"M006 (Mean: {df_comp['NLL_M006'].mean():.1f})")
axes[0].set_title(r'Joint Defective Density ($\mathcal{L}_{DDM}$)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Negative Log-Likelihood (Lower is Superior)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].legend(frameon=True, loc='upper right')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Panel B: Delta Likelihood (Forest/Scatter)
delta_sorted = df_comp['Delta_NLL'].sort_values().reset_index(drop=True)
axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Equivalence')
axes[1].scatter(delta_sorted, range(1, len(delta_sorted)+1), color='#2980b9', alpha=0.8, s=20)
axes[1].hlines(range(1, len(delta_sorted)+1), xmin=0, xmax=delta_sorted, color='#2980b9', alpha=0.3, linewidth=0.8)
axes[1].set_title(r'Subject-Level Advantage ($\Delta \mathcal{L} = \mathcal{L}_{Base} - \mathcal{L}_{M006}$)', fontsize=13, fontweight='bold')
axes[1].set_xlabel(r'$\Delta \mathcal{L}$ (Positive = M006 Superior)', fontsize=11)
axes[1].set_ylabel('Subject Index (Ranked)', fontsize=11)
axes[1].legend(frameon=True, loc='lower right')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Panel C: Model Calibration & Brier Score Comparison
brier_df = pd.DataFrame({
    'Model': ['Baseline']*len(df_comp) + ['M006']*len(df_comp),
    'Brier': list(df_comp['Brier_Base']) + list(df_comp['Brier_M006'])
})
sns.boxplot(x='Model', y='Brier', data=brier_df, ax=axes[2], palette={'Baseline': '#7f8c8d', 'M006': '#2980b9'}, boxprops=dict(alpha=0.4), showfliers=False)
sns.stripplot(x='Model', y='Brier', data=brier_df, ax=axes[2], palette={'Baseline': '#7f8c8d', 'M006': '#2980b9'}, alpha=0.7, jitter=True, size=5)
axes[2].set_title('Choice Probability Calibration (Brier Score)', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Brier Score (Lower is Superior)', fontsize=11)
axes[2].set_xlabel('')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.suptitle(r'Full Cohort Empirical Statistical Benchmark: $M_{006}$ (Bounded) vs. $M_{base}$ ($N=128$)', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "m006_vs_base_cohort_benchmark.png"), dpi=200, bbox_inches='tight')
plt.close()

print("Figures successfully generated in figures/")
