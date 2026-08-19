# Mathematical Proof of Topological Collapse in ECCM

The execution of the strict 150-iteration tournament completed, but the ECCM model performed massively worse than the M1 WSLS baseline ($p < 0.001$). 

By extracting the optimal parameters chosen by the CMA-ES and mathematically analyzing the thermodynamic equilibrium equation, I have discovered a fundamental algebraic flaw in the proposed formulation that mathematically destroys the topological manifold.

## The Flaw: Algebraic Factoring of the Spatial Topology

Your proposed thermodynamic equilibrium equation is:
$$ \Delta w_{PF, i} = - \alpha_{LTD} \cdot \delta_{IO} \cdot z_{GC, i} \cdot P_i + \alpha_{LTP} \cdot z_{GC, i} $$

The candidate manifold achieves thermodynamic equilibrium when $\Delta w_{PF, i} = 0$. Let's solve for the final steady-state weights at equilibrium:

$$ 0 = - \alpha_{LTD} \cdot \delta_{IO} \cdot z_{GC, i} \cdot P_i + \alpha_{LTP} \cdot z_{GC, i} $$

Because both the Pruning term (LTD) and the Expansion drift (LTP) are multiplied by the same instantaneous spatial state $z_{GC, i}$, we can divide both sides by $z_{GC, i}$ (assuming the synapse is active, $z_{GC, i} \neq 0$):

$$ 0 = - \alpha_{LTD} \cdot \delta_{IO} \cdot P_i + \alpha_{LTP} $$

Solving for the normalized weight penalty $P_i$:

$$ P_i = \frac{\alpha_{LTP}}{\alpha_{LTD} \cdot \delta_{IO}} $$

### The Consequence: Complete Loss of Linear Separability
Notice what is completely missing from the final equilibrium solution: **$z_{GC, i}$**. 

The equation mathematically dictates that at equilibrium, every single active synapse $i$ must converge to the **exact same normalized proportional weight** ($P_i$), entirely independent of its spatial topography or temporal fading memory trace. 

Because every active synapse is forced to have the exact same weight, the Purkinje linear readout $y_{PC} = \mathbf{w}_{PF}^\top \mathbf{z}_{GC}$ algebraically collapses into:
$$ y_{PC} = w_{avg} \sum_{i=1}^{N_{GC}} z_{GC, i} $$

The entire 500-dimensional, heterogeneous candidate manifold is squashed into a 1-dimensional, unweighted scalar sum of background activity. The model learns absolutely no spatial patterns because the proportional pruning prunes everything equally.

---

> [!WARNING]
> ## User Review Required
> To restore the linear separability of the manifold, we must break the symmetry between the LTP and LTD terms so that $z_{GC}$ does not factor out. 
>
> **Option A: The Oja/BCM formulation**
> Make the pruning proportional to the *unnormalized* weight and keep LTP strictly driven by reward spikes rather than a constant basal drift:
> $\Delta w_i = \alpha_{LTP} \cdot \delta_{reward} \cdot z_{GC, i} - \alpha_{LTD} \cdot w_i \cdot z_{GC, i}$
> 
> **Option B: The Standard Delta Rule**
> Abandon proportional pruning and use the biologically standard supervised error signal:
> $\Delta w_i = \eta \cdot (\text{Outcome} - y_{PC}) \cdot z_{GC, i}$
>
> **Option C: Re-formulate the Proportional Pruning**
> If you wish to keep the $L_1$ proportional pruning, we must alter the LTP expansion term so it does not scale perfectly with $z_{GC, i}$.
>
> Which topological correction would you like to apply to the Phase 4 plasticity mechanism?
