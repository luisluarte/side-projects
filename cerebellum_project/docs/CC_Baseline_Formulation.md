# Formal Axiomatization of Baseline Cortical Model ($M_{base}$)

This document formalizes the canonical baseline model ($M_{base}$) against which the bio-thermodynamic cerebellar architectures ($M_{005}$, $M_{006}$) are evaluated.

---

## I. Cortical Value Updating (Q-Learning)

The cortical value representation maintains expected reward values $Q_c(t) \in [0, 1]$ for discrete choices $c \in \{0, 1\}$. 

On each trial $t$, the chosen action $c^{(t)}$ yields empirical reward $R^{(t)} \in \{0, 1\}$, producing a standard scalar prediction error:
\[ \delta^{(t)} = R^{(t)} - Q_{c^{(t)}}(t) \]

Values update according to the cortical learning rate $\alpha_{ctx} \in (0, 1)$:
\[ Q_{c^{(t)}}(t+1) = Q_{c^{(t)}}(t) + \alpha_{ctx} \cdot \delta^{(t)} \]
\[ Q_{1 - c^{(t)}}(t+1) = Q_{1 - c^{(t)}}(t) \]

---

## II. Evidence Accumulation Velocity (Drift Rate)

The drift rate $v^{(t)}$ into the Wiener process is linearly determined by the cortical value discrepancy:
\[ v^{(t)} = v_{ctx} \cdot (Q_1(t) - Q_0(t)) \]

Where $v_{ctx}$ scales the sensitivity of accumulation velocity to cortical value differences.

---

## III. Static Decision Boundary & Non-Decision Time

The baseline model employs a stationary, trial-invariant decision threshold $a$ and non-decision time $t_{nd}$:
\[ a^{(t)} = a_{base} = \text{constant} \]
\[ t_{nd}^{(t)} = t_{nd} = \text{constant} \]

---

## IV. Wiener First-Passage Time Joint Likelihood ($\mathcal{L}_{DDM}$)

Choice $c \in \{1, 2\}$ (mapping to index $0, 1$) and continuous reaction time $RT$ are jointly evaluated under the Navarro-Fuss (2009) defective probability density:
\[ \mathcal{L}_{DDM}(M_{base}) = -\sum_{t=1}^T \log f_{WFPT}\left(RT_t, c_t \mid v^{(t)}, a_{base}, t_{nd}, w=0.5\right) \]
