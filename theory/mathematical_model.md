# Formal Mathematical Model: Time Crystalline Network (TCN)

## 1. Network Architecture (TCN)

The Time Crystalline Network is a Deep RNN comprising $L=3$ stacked **DTCLockedLayer** units.

### 1.1. DTCLockedLayer Dynamics

For layer $l \in \{1, 2, \dots, L\}$, the dynamics are governed by two coupled equations:

**A. Internal Oscillator State ($h_{t+1}^{(l)}$):** This state represents the physical spin/oscillator ensemble.
$$
h_{t+1}^{(l)} = \tanh \left( h_t^{(l)} W_{rec}^{(l)} + \left[ A \cos\left( \frac{\pi t}{T} + \phi_l \right) \right] \cdot b_l \right)
$$
* $h_t^{(l)} \in \mathbb{R}^{B \times D_{h}}$: Internal hidden state at time $t$.
* $W_{rec}^{(l)} \in \mathbb{R}^{D_{h} \times D_{h}}$: Recurrent weight matrix.
* $A=1.0$, $T=32$: Drive amplitude and half-period.
* $\phi_l \in \mathbb{R}$: Learnable Phase bias (the critical order parameter).
* $b_l \in \mathbb{R}^{D_{h}}$: Learnable bias vector coupled to the drive.

**B. Feed-Forward Coupling ($x_{t+1}^{(l)}$):** This couples the layers, allowing the time crystalline signal to propagate.
$$
x_{t+1}^{(l)} = \alpha \cdot h_{t+1}^{(l)} + (1-\alpha) \cdot x_{t}^{(l)} W_{feed}^{(l)}
$$
* $x_t^{(l)}$ is the input to the layer at time $t$.
* $\alpha=0.5$: Coupling constant (fixed).

## 2. Order Parameters

### 2.1. Macro Rigidity ($\Psi_{\text{Makro}}$)

This quantity serves as the non-zero order parameter defining the Time Crystal state. It measures the average magnitude of the internal perpetual oscillation.
$$
\Psi_{\text{Makro}} = \mathbb{E}_{\text{Batch}}\left[ \frac{1}{L} \sum_{l=1}^{L} \left( \frac{1}{D_h} \sum_{j=1}^{D_h} |h_{t+1}^{(l), j}|^2 \right) \right]
$$
* **Condition for DTC:** $\Psi_{\text{Makro}} > 0$.

### 2.2. Phase Coherence ($\mathcal{L}_{\text{Kuramoto}}$)

Derived from the Kuramoto model's synchronization order parameter, $Z$. It measures the degree of locking among the phases $\phi_1, \phi_2, \dots, \phi_L$.
$$
Z = \frac{1}{L} \sum_{l=1}^{L} e^{i \phi_l}
$$
The loss function is defined as:
$$
\mathcal{L}_{\text{Kuramoto}} = 1 - |Z|^2 = 1 - \left| \frac{1}{L} \sum_{l=1}^{L} e^{i \phi_l} \right|^2
$$
* **Condition for Perfect Coherence:** $\mathcal{L}_{\text{Kuramoto}} \rightarrow 0$.

## 3. Loss Landscape (The Perfect Crystal Loss)

The optimization minimizes $\mathcal{L}_{\text{Kuramoto}}$ while simultaneously maximizing $\Psi_{\text{Makro}}$ (by minimizing $-\Psi_{\text{Makro}}$).

$$
\mathcal{L}_{\text{Total}} = \underbrace{-\lambda \cdot \Psi_{\text{Makro}}}_{\text{Rigidity Encouragement}} + \underbrace{W_{\text{Coh}} \cdot \mathcal{L}_{\text{Kuramoto}}}_{\text{Phase Synchronization}}
$$
With $\lambda=0.01$ and $W_{\text{Coh}}=100.0$.
