# Formal Model Specifications

This document provides a line-by-line breakdown of the computational models used to map Cortico-Cerebellar non-stationary sequence learning.

## 1. The Win-Stay/Lose-Shift (WSLS) Baseline Model
**File:** [`src/models/wsls.cpp`](file:///C:/Users/DCCS5/Documents/GitHub/side-projects/cerebellum_project/src/models/wsls.cpp)

### Intuition
The WSLS model represents a low-dimensional, cortically-driven heuristic state. It tracks the outcome of the immediate previous trial. If the outcome was positive (a win), the model repeats the action. If the outcome was negative (a loss), the model shifts to the alternative action.

### Mathematical Formulation
The Drift Rate $v^{(t)}$ of the Drift-Diffusion Model is strictly dictated by the previous outcome $R^{(t-1)}$:

$$ v^{(t)} = \beta_v \cdot \begin{cases} 
+1 & \text{if } Ch^{(t-1)} = 1 \text{ and } R^{(t-1)} = +1 \\
+1 & \text{if } Ch^{(t-1)} = 2 \text{ and } R^{(t-1)} = -1 \\
-1 & \text{otherwise}
\end{cases} $$

Where $\beta_v$ is the subjective drift scaling parameter.

### Line-By-Line Code Mapping
```cpp
// Lines 8-16 in wsls.cpp
int last_ch = -1, last_out = -1;
for (int t=0; t<resp.size(); ++t) {
    double v = 0.0;
    if (last_ch != -1) {
        // Evaluate the heuristic shift
        int pred_ch = (last_out == 1) ? last_ch : (last_ch == 1 ? 2 : 1);
        
        // Assign the strict drift magnitude beta_v
        v = (pred_ch == 1) ? beta_v : -beta_v;
    }
    // ... DDM Evaluation ...
    last_ch = resp[t]; last_out = out[t];
}
```

---

## 2. Intact Expansion-Compression Cerebellar Manifold (ECCM)
**File:** [`src/models/eccm_intact.cpp`](file:///C:/Users/DCCS5/Documents/GitHub/side-projects/cerebellum_project/src/models/eccm_intact.cpp)

### Intuition
Unlike WSLS, the intact cerebellum extracts the latent chronological history of the environment ($k=80$ past trials). It expands this sequence via random Mossy Fiber-to-Granule Cell projections into a massive high-dimensional manifold, which is then dynamically regularized by Molecular Layer Interneurons (MLIs) and compressed back into a low-dimensional Purkinje readout. This geometric readout is multiplicatively injected into the Cortical Q-value estimate via the Thalamus.

### Mathematical Formulation
**1. The Spatial Expansion (Mossy Fibers to Granule Cells)**
Let $\mathbf{mf}^{(t)} \in \mathbb{R}^{160}$ be the concatenated shift-register of past actions and rewards.
The Granule Cell layer $\mathbf{gc}^{(t)} \in \mathbb{R}^{1024}$ extracts high-dimensional non-linear features:
$$ \mathbf{gc}^{(t)} = \tanh\left( \mathbf{W}_{MF \to GC} \cdot \mathbf{mf}^{(t)} \right) $$

**2. The Recurrent Regularization (Granule Cells to MLIs)**
The MLI layer $\mathbf{mli}^{(t)} \in \mathbb{R}^{256}$ provides recurrent inhibition:
$$ \mathbf{mli}^{(t)} = \tanh\left( \mathbf{W}_{GC \to MLI} \cdot \mathbf{gc}^{(t)} \right) $$

**3. Purkinje Cell Compression (Q-Value Readout)**
The Cerebellar Q-value is the linear combination of the excitatory GC and inhibitory MLI signals:
$$ Q_{CB}^{(t)} = \mathbf{w}_{PF} \cdot \mathbf{gc}^{(t)} - \mathbf{w}_{MLI} \cdot \mathbf{mli}^{(t)} $$

**4. Thalamic Multiplicative Modulation**
The ultimate DDM drift rate $v^{(t)}$ scales the Cortical Estimate ($Q_{CTX}$) by the Cerebellar prediction:
$$ v^{(t)} = \beta_v \left[ Q_{CTX, 1}^{(t)} \big(1 + w_{cb} Q_{CB, 1}^{(t)}\big) - Q_{CTX, 2}^{(t)} \big(1 + w_{cb} Q_{CB, 2}^{(t)}\big) \right] $$

### Line-By-Line Code Mapping
```cpp
// Lines 34-40 in eccm_intact.cpp: High-Dimensional Expansion
for (int i=0; i<N_GC; ++i) {
    double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
    gc[i] = std::tanh(act);
}
for (int i=0; i<N_MLI; ++i) {
    double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
    mli[i] = std::tanh(act);
}

// Lines 42-43: Purkinje Cell Compression
for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }

// Line 45: Thalamic Multiplicative Modulation
double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );

// Lines 63-64: Shift Register chronological updating
for(int j=k-2; j>=0; --j) { mf[j+1] = mf[j]; mf[k+j+1] = mf[k+j]; }
mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw;
```

---

## 3. Lesioned Expansion-Compression Cerebellar Manifold
**File:** [`src/models/eccm_lesioned.cpp`](file:///C:/Users/DCCS5/Documents/GitHub/side-projects/cerebellum_project/src/models/eccm_lesioned.cpp)

### Intuition
To prove that the high-dimensional spatial expansion is necessary for predictive precision, the Lesioned model structurally deletes the $1024$ Granule Cells and $256$ MLIs. The Purkinje cells must now linearly read directly from the $160$ Mossy Fibers. The geometric curvature of the manifold is destroyed.

### Mathematical Formulation
The Cerebellar Q-value is stripped of all non-linear basis transformations:
$$ Q_{CB}^{(t)} = \text{clip}\left( \mathbf{w}_{PF} \cdot \mathbf{mf}^{(t)}, -1, 1 \right) $$

### Line-By-Line Code Mapping
```cpp
// Lines 21-23 in eccm_lesioned.cpp: Direct Linear Readout (Lesion)
for (int i=0; i<N_MF; ++i) { Q1_CB += W_PF1[i]*mf[i]; Q2_CB += W_PF2[i]*mf[i]; }
Q1_CB = std::max(-1.0, std::min(1.0, Q1_CB));
Q2_CB = std::max(-1.0, std::min(1.0, Q2_CB));
```
