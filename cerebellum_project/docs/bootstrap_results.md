# Bootstrapped NLL Differences (N=128)

I have performed a non-parametric bootstrap ($B=10,000$ iterations) over the participant-wise Negative Log-Likelihood (NLL) differences derived from the 70/30 cross-validation. This gives us robust 95% Confidence Intervals for the relative model performances.

## Statistical Summary

- **M1 (WSLS) vs ECCM:**
  - Mean Difference: **-5.981**
  - 95% CI: **[-6.943, -5.004]**
  - *Interpretation:* The WSLS heuristic strictly outperforms the ECCM by an average of 5.98 NLL points per participant. The 95% CI completely excludes zero, indicating massive, robust superiority.

- **M2 (RWCF) vs ECCM:**
  - Mean Difference: **6.138**
  - 95% CI: **[4.603, 7.715]**
  - *Interpretation:* The ECCM strictly outperforms the classical Random Walk Cognitive Filter by an average of 6.13 NLL points per participant.

## Distribution Density

![Bootstrapped Differences Density](file:///C:/Users/DCCS5/.gemini/antigravity/brain/1d8f9958-fd49-4502-b57b-97a7887eb7ad/bootstrapped_differences_density.png)

> [!NOTE]
> The density plot visualizes the complete separation of the two distributions. The WSLS baseline remains the most robust descriptor for this generalized N=128 human population.
