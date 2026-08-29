import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

repo_root = "." if os.path.exists("results/cohort_comparison_metrics_3way.csv") else "../.."
results_dir = os.path.join(repo_root, "results")
figures_dir = os.path.join(repo_root, "figures")
os.makedirs(figures_dir, exist_ok=True)

df_comp = pd.read_csv(os.path.join(results_dir, "cohort_comparison_metrics_3way.csv"))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: NLL Distribution
sns.kdeplot(df_comp['NLL_Base'], ax=axes[0], color='#7f8c8d', fill=True, alpha=0.3, label=f"Baseline (Mean: {df_comp['NLL_Base'].mean():.1f})")
sns.kdeplot(df_comp['NLL_M006_Unclamped'], ax=axes[0], color='#e67e22', fill=True, alpha=0.3, label=f"M006 Unclamped (Mean: {df_comp['NLL_M006_Unclamped'].mean():.1f})")
sns.kdeplot(df_comp['NLL_M006_Clamped'], ax=axes[0], color='#27ae60', fill=True, alpha=0.3, label=f"M006 Clamped (Mean: {df_comp['NLL_M006_Clamped'].mean():.1f})")

axes[0].set_title(r'Joint Defective Density ($\mathcal{L}_{DDM}$)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Negative Log-Likelihood (Lower is Superior)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].legend(frameon=True, loc='upper right')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Panel B: Delta Likelihood (Clamp vs Base)
delta_sorted = df_comp['Delta_NLL_Clamp'].sort_values().reset_index(drop=True)
axes[1].axvline(0, color='black', linestyle='--', linewidth=1.5, label='Equivalence')
axes[1].scatter(delta_sorted, range(1, len(delta_sorted)+1), color='#27ae60', alpha=0.8, s=20)
axes[1].hlines(range(1, len(delta_sorted)+1), xmin=0, xmax=delta_sorted, color='#27ae60', alpha=0.3, linewidth=0.8)
axes[1].set_title(r'Clamped vs Base ($\Delta \mathcal{L}_{Clamp} = \mathcal{L}_{Base} - \mathcal{L}_{Clamp}$)', fontsize=13, fontweight='bold')
axes[1].set_xlabel(r'$\Delta \mathcal{L}$ (Positive = Clamped Superior)', fontsize=11)
axes[1].set_ylabel('Subject Index (Ranked)', fontsize=11)
axes[1].legend(frameon=True, loc='lower right')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Panel C: Delta Likelihood (Clamp vs Unclamp)
delta_unc_sorted = df_comp['Delta_NLL_Clamp_vs_Unc'].sort_values().reset_index(drop=True)
axes[2].axvline(0, color='black', linestyle='--', linewidth=1.5, label='Equivalence')
axes[2].scatter(delta_unc_sorted, range(1, len(delta_unc_sorted)+1), color='#e67e22', alpha=0.8, s=20)
axes[2].hlines(range(1, len(delta_unc_sorted)+1), xmin=0, xmax=delta_unc_sorted, color='#e67e22', alpha=0.3, linewidth=0.8)
axes[2].set_title(r'Clamped vs Unclamped ($\Delta \mathcal{L} = \mathcal{L}_{Unc} - \mathcal{L}_{Clamp}$)', fontsize=13, fontweight='bold')
axes[2].set_xlabel(r'$\Delta \mathcal{L}$ (Positive = Clamped Superior)', fontsize=11)
axes[2].set_ylabel('')
axes[2].legend(frameon=True, loc='lower right')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.suptitle(r'Full Cohort Structural Ablation Benchmark: Baseline vs. Unclamped $M_{006}$ vs. Clamped $M_{006}$ ($N=128$)', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "m006_3way_cohort_benchmark.png"), dpi=200, bbox_inches='tight')
plt.close()

print("3-way figures successfully generated in figures/")
