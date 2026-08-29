# Layer Information Theory Metrics: Predictive Power on Switches

We ran a brute-force analysis extracting the Shannon Entropy, Hoyer's Sparsity, and L2 Norm (Energy) for every physical layer in the Cerebellar network (Mossy Fibers, Granule Cells, Molecular Layer Interneurons) using the optimized Cortical RPE model for each participant.

We then computed the Delta ($\Delta = t_{switch} - t_{pre-switch}$) and ran mixed-effects logistic regression models to determine which information metric held the most predictive power for an incoming switch.

## Results (Ranked by Predictive Power / AIC)
| Layer Metric | Coefficient (Scaled) | P-Value | AIC |
| :--- | :--- | :--- | :--- |
| **d_MF_L2** | 0.2302 | 1.36e-06 | 2518.8 |
| **d_GC_Ent** | -0.2200 | 1.37e-05 | 2522.9 |
| **d_GC_L2** | 0.1912 | 5.00e-05 | 2525.9 |
| **d_MF_Spa** | -0.1778 | 2.71e-04 | 2529.0 |
| **d_MLI_L2** | 0.0915 | 5.87e-02 | 2538.5 |
| **d_GC_Spa** | -0.0514 | 2.53e-01 | 2540.8 |
| **d_MLI_Ent** | -0.0541 | 2.71e-01 | 2540.9 |
| **d_MLI_Spa** | -0.0363 | 4.31e-01 | 2541.5 |
| **d_MF_Ent** | -0.0291 | 5.48e-01 | 2541.7 |
