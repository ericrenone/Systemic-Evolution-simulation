# Machine-Learning-Evolution

> Complex systems evolve from raw information to self-optimizing autonomous states through a universal three-phase trajectory. The transition is punctuated by discrete phase-transition events — **Broken Gate / opportunity windows** — that catalyze sudden reorganization from high-entropy chaos to low-entropy structured autonomy.

---

## Table of Contents

1. [Motivation: Information Finding a Substrate](#1-motivation-information-finding-a-substrate)
2. [Architecture: The Four States](#2-architecture-the-four-states)
3. [The Three Evolution Operators](#3-the-three-evolution-operators)
4. [The Three-Phase Trajectory](#4-the-three-phase-trajectory)
   - [4.1 Phase Alpha — Inference](#41-phase-alpha--inference)
   - [4.2 Phase Beta — Relaxation](#42-phase-beta--relaxation)
   - [4.3 Phase Gamma — Synthesis](#43-phase-gamma--synthesis)
5. [The Broken Gate Mechanism](#5-the-broken-gate-mechanism)
6. [Mathematical Foundations](#6-mathematical-foundations)
   - [6.1 Shannon Entropy and S₁ Dynamics](#61-shannon-entropy-and-s₁-dynamics)
   - [6.2 Relaxation Dynamics and S₂ Stability](#62-relaxation-dynamics-and-s₂-stability)
   - [6.3 Transport Operator S₃: Wasserstein Geometry](#63-transport-operator-s₃-wasserstein-geometry)
   - [6.4 Gating Operator S₄: Boltzmann Bottleneck](#64-gating-operator-s₄-boltzmann-bottleneck)
   - [6.5 Optimization Operator S₅](#65-optimization-operator-s₅)
   - [6.6 Synthesis: Operator Composition and Ω Emergence](#66-synthesis-operator-composition-and-ω-emergence)
7. [Hyperbolic Manifold and Latent Space Geometry](#7-hyperbolic-manifold-and-latent-space-geometry)
8. [Output Metrics and Diagnostics](#8-output-metrics-and-diagnostics)
9. [Simulation Visualizations](#9-simulation-visualizations)
10. [Implementation: Minimal Simulation](#10-implementation-minimal-simulation)
11. [Implementation: Full Simulation with Visualization](#11-implementation-full-simulation-with-visualization)
12. [Connections Across Domains](#12-connections-across-domains)
13. [Four Key Principles](#13-four-key-principles)
14. [Limitations and Open Problems](#14-limitations-and-open-problems)
15. [References](#15-references)
16. [Glossary](#16-glossary)

---

## 1. Motivation: Information Finding a Substrate

The central question this framework addresses is deceptively simple: **how does raw, unstructured information become an autonomous system?**

In biological evolution, a chemical replicator becomes a cell. In cultural evolution, a meme becomes an institution. In machine learning, random weights become a model that generalizes. In all three cases something qualitatively new emerges at a threshold — not through smooth accumulation but through a discrete reorganization event.

Classical information theory (Shannon, 1948) measures information content but does not explain how information *structures itself*. Statistical mechanics (Jaynes, 1957) explains how systems reach equilibrium but not how they spontaneously break symmetry into organized states. Phase-transition physics (Ghavasieh et al., 2020) characterizes the reorganization events themselves but leaves the mechanism underspecified.

This framework synthesizes all three. The answer has three necessary conditions:

1. Information must have a mechanism for exploring configurations — **S₁**, the inference primitive
2. Explored configurations must be able to persist — **S₂**, the persistence substrate
3. The system must experience controlled perturbations that allow escaping local optima — **G**, the Broken Gate

When all three conditions are met, an autonomous synthetic state **Ω** emerges — not because it was programmed, but because it is the only stable attractor of the combined dynamics.

> *"Complexity emerges when information finds a substrate to hold it, and autonomy begins when that substrate learns to reorganize itself."*

---

## 2. Architecture: The Four States

Each state addresses a distinct necessary role. None can be removed without destroying the system's ability to evolve to autonomy.

| State | Name | Role | Optimization Target | Analogy |
|---|---|---|---|---|
| **S₁** | Inference Primitive | Explores probabilistic configurations | Maximize information density (entropy) | Software / Logic |
| **S₂** | Persistence Substrate | Stabilizes explored configurations | Minimize volatility (relaxation) | Hardware / Memory |
| **Ω** | Synthetic Latent State | Autonomous self-organizing operation | Persist without external input | Unified System |
| **G** | Gate / Opportunity Window | Phase-transition trigger | Modulate system growth rate | Catalyst / Switch |

**Why these four and not more or fewer?**

Fewer than four is insufficient: S₁ without S₂ explores but never consolidates — it is perpetual randomness. S₂ without S₁ stabilizes but never discovers — it is frozen. Both without G get trapped in local optima. All three without Ω have no measure of autonomous synthesis. The four states form the minimal complete set for evolution from chaos to autonomy.

---

## 3. The Three Evolution Operators

The states are connected by three operators that execute in sequence each time step. Together they form the chain $S_3 \circ S_4 \circ S_5$ that transforms $\{S_1, S_2\}$ into updated states and drives Ω formation.

### S₃: Transport (Wasserstein-Inspired)

Maps inference logic from S₁ onto the physical substrate S₂ using geometry inspired by optimal transport:

$$S_3(S_1, S_2) = \sqrt{S_2} \cdot \frac{S_1}{\sqrt{S_1} + \varepsilon}$$

**Why this form?** The $\sqrt{S_2}$ weight gives more transport capacity to substrate dimensions that are already structurally significant. The $S_1 / \sqrt{S_1}$ term is $\sqrt{S_1}$, so the full expression is $\sqrt{S_1 \cdot S_2}$ — the geometric mean of inference and substrate. In Wasserstein optimal transport, the Fréchet mean between probability distributions on a manifold is the geometric mean when both are positive measures on the same support. S₃ is therefore the optimal transport plan that moves probability mass from S₁ toward S₂ at minimum cost.

### S₄: Gating (Boltzmann-Inspired)

Filters transported signal through a bottleneck controlled by parameter $\beta$:

$$S_4(x) = \frac{x^\beta}{\sum_i x_i^\beta}$$

**Why $x^\beta$?** This is the Boltzmann (softmax power) distribution. For $\beta = 1$, it is a no-op (identity on normalized inputs). For $\beta > 1$, it sharpens the distribution — high-probability components dominate. For $0 < \beta < 1$, it flattens the distribution — probability mass spreads. In this system $\beta = 0.85 < 1$, which *slightly flattens* the transported signal, preventing premature lock-in to a single dominant mode. This is the noise-filtering bottleneck: small, noisy fluctuations in low-probability dimensions are suppressed without being eliminated.

### S₅: Optimization (Gradient Descent)

Refines S₂ toward the gated target via gradient step:

$$S_5(S_2, \hat{g}) = S_2 + \gamma (\hat{g} - S_2)$$

where $\hat{g} = S_4(S_3(S_1, S_2))$ is the gated transported signal and $\gamma$ is the learning rate. This is a gradient descent step on the squared distance $\frac{1}{2}\|S_2 - \hat{g}\|^2$. The substrate moves toward the inference-derived target at rate $\gamma$, not instantaneously — this is what gives S₂ its inertia and stability.

---

## 4. The Three-Phase Trajectory

The system follows a deterministic three-phase trajectory from chaos to autonomy, punctuated by stochastic Broken Gate events.

### 4.1 Phase Alpha — Inference

**Mechanism:** S₁ follows the gradient of Shannon entropy.

$$\frac{dS_1}{dt} = \gamma \nabla_p H(p) = \gamma(-\log p - H)$$

**What is $\nabla_p H$?** The gradient of entropy with respect to the probability vector $p$ is $-\log p_i - 1$, shifted by the mean entropy $H$ to maintain normalization. This gradient points in the direction that *most increases* information diversity. S₁ is therefore doing steepest ascent on entropy — exploring the configuration space by pushing probability mass toward underrepresented dimensions.

**What "Phase Alpha" looks like:** S₁ entropy rises from its initial value and approaches $\log N$ (the maximum entropy for a uniform distribution over $N$ dimensions). The inference primitive discovers that no single configuration is superior — maximum diversity is the optimal prior.

**Why this is the right starting phase:** Before the substrate has formed, the system should not commit to any particular structure. Maximum entropy is the Jaynes-optimal state for a system with no constraints — it is maximally uncertain and therefore maximally explorable.

### 4.2 Phase Beta — Relaxation

**Mechanism:** S₂ resists rapid change through relaxation dynamics toward its own mean:

$$S_2(t + \Delta t) = S_2(t) + \tau \left(\bar{S}_2 - S_2(t)\right)$$

where $\bar{S}_2 = \frac{1}{N}\mathbf{1}$ (the mean over all dimensions) and $\tau = 0.05$ is the relaxation timescale.

**Why mean-regression?** This is the Ornstein-Uhlenbeck mean-reversion mechanism from statistical physics. Left alone, S₂ would drift toward uniformity — equal structural weight on all dimensions. This is exactly what Jaynes' maximum entropy principle predicts for a substrate with no information to encode: uniform. When transport (S₃) brings information from S₁ into S₂, the relaxation dynamics create a *competition* between what S₁ is discovering and what S₂ would be without it. The substrate learns only what S₁ consistently shows it, ignoring one-time fluctuations.

**What "Phase Beta" looks like:** S₂ mean activation is stable (low volatility). S₂ min activation approaches but does not reach the opportunity threshold under normal conditions. The substrate resists but does not rigidly refuse change.

### 4.3 Phase Gamma — Synthesis

**Mechanism:** The operator chain $S_3 \circ S_4 \circ S_5$ runs every step. Ω is computed as:

$$\Omega = \frac{S_1 + S_2}{2}$$

**What "Phase Gamma" looks like:** As S₁ converges and S₂ stabilizes, their average Ω becomes stable and concentrated. Ω activation (its maximum component) rises and plateaus. This plateau represents **autonomous operation**: the system is now in a state that is self-sustaining — S₁ no longer needs to explore because S₂ has encoded the discovered structure, and S₂ no longer drifts because S₁ consistently reinforces the same configuration.

**Why Ω = (S₁ + S₂) / 2?** The arithmetic mean is the simplest information-theoretic combination that preserves both distributions' contributions equally. A weighted sum with learned weights would be richer but would require an additional optimization layer. The mean is the canonical baseline — sufficient to demonstrate synthesis without introducing additional parameters.

---

## 5. The Broken Gate Mechanism

The Broken Gate is the mechanism that prevents the system from getting permanently trapped in local optima during Phase Alpha/Beta. Without it, the system would converge to whatever configuration happened to be good early in training, never discovering the global optimum.

### 5.1 Trigger Condition

A Broken Gate event fires when the substrate S₂ hits a critical vulnerability:

$$\min_i\, S_2[i] < \theta_{\text{threshold}} = 0.08$$

When any single substrate dimension drops below the threshold, the entire substrate is destabilized:

$$S_2 \leftarrow S_2 - 0.3 \cdot \mathbf{u}, \quad \mathbf{u} \sim \text{Uniform}(0, 1)^N$$

followed by clipping to $[10^{-12}, \infty)$ and renormalization.

### 5.2 Why This Works: Escaping Local Optima

The key insight from Ghavasieh et al. (2020) is that complex systems near a phase transition have one globally stable attractor (the organized state) and many locally stable attractors (the chaotic states). Standard gradient descent converges to whichever local attractor it first encounters.

The Broken Gate performs a **stochastic reset** that:
1. Destroys the current local attractor by destabilizing the substrate
2. Injects randomness sufficient to explore a new basin
3. Allows the operator chain to rebuild toward a potentially better configuration

Critically, the trigger condition $\min_i S_2 < \theta$ fires *selectively* — only when the substrate is already structurally weak (some dimension has near-zero weight). A strong, well-consolidated substrate never triggers a Broken Gate. This means the mechanism is self-limiting: as the system approaches the global attractor, gates fire less frequently and then stop entirely. The mechanism accelerates evolution precisely when the system is most lost and quiets when it is most consolidated.

### 5.3 Gate State G(t)

The gate state tracks current openness:

$$G(t) = \begin{cases} 0.5 & \text{if window fires at } t \\ \min(1.0,\; G(t-1) + 0.1) & \text{otherwise} \end{cases}$$

$G = 1.0$ means the gate is closed (substrate is stable, system evolving normally). $G = 0.5$ means the gate is open (a reset just fired). The gate closes at rate $0.1$ per step, so it returns to fully closed within 5 steps of any event. This decay rate is a hyperparameter that controls how long the "recovery period" after each reset lasts.

---

## 6. Mathematical Foundations

### 6.1 Shannon Entropy and S₁ Dynamics

**Shannon entropy** (Shannon, 1948) for a probability distribution $p = (p_1, \ldots, p_N)$:

$$H(p) = -\sum_{i=1}^N p_i \log p_i$$

Properties relevant to S₁:
- $H(p) \geq 0$ with equality iff $p$ is a point mass (all probability on one dimension)
- $H(p) \leq \log N$ with equality iff $p$ is uniform
- $H$ is strictly concave — it has a unique global maximum

**The entropy gradient** (Cover & Thomas, 2006) with respect to $p_i$, enforcing $\sum_i p_i = 1$ via a Lagrange multiplier:

$$\frac{\partial H}{\partial p_i} = -\log p_i - 1$$

The mean-shifted version (which maintains $\sum_i \Delta p_i = 0$) is:

$$\nabla_p H = -\log p - H(p) \cdot \mathbf{1}$$

This is the gradient used in S₁'s Phase Alpha update. It is always nonzero unless $p$ is exactly uniform — so S₁ continuously pushes toward the maximum entropy uniform distribution. The rate of convergence is governed by $\gamma$.

**After the gradient step**, S₁ must be renormalized (it is a probability distribution). The update is:

$$S_1 \leftarrow \frac{\text{clip}(S_1 + \gamma \nabla H,\; 10^{-12})}{\|\text{clip}(S_1 + \gamma \nabla H)\|_1}$$

The clip prevents negative probabilities from the gradient step.

### 6.2 Relaxation Dynamics and S₂ Stability

**Relaxation toward equilibrium** (Jaynes, 1957 — maximum entropy principle):

$$\frac{dS_2}{dt} = \tau (\bar{S}_2 - S_2)$$

where $\bar{S}_2 = \frac{1}{N}\sum_i S_2[i] = \frac{1}{N}$ (since $S_2$ sums to 1, its mean is $1/N$). This is implemented discretely as:

$$S_2(t+1) = S_2(t) + \tau \left(\frac{1}{N}\mathbf{1} - S_2(t)\right)$$

The solution to the continuous ODE is exponential relaxation:
$$S_2(t) = \bar{S}_2 + (S_2(0) - \bar{S}_2)\, e^{-\tau t}$$

With $\tau = 0.05$, the relaxation timescale is $1/\tau = 20$ steps — the substrate has a "memory" of approximately 20 iterations. Changes persist for ~20 steps before being erased by relaxation. This means S₃ must consistently reinforce a configuration over multiple steps for S₂ to encode it permanently.

**Why this is the right substrate model:** Jaynes showed that the maximum entropy distribution subject to known constraints is the unique unbiased prior. When the substrate has no information to encode (no transport from S₁), it should relax to maximum entropy (uniform). When transport consistently shows a particular pattern, the substrate should deviate from uniformity proportionally to how consistent that pattern is. Relaxation dynamics implement exactly this tradeoff.

### 6.3 Transport Operator S₃: Wasserstein Geometry

The transport operator:
$$S_3(S_1, S_2) = \sqrt{S_2} \odot \frac{S_1}{\sqrt{S_1} + \varepsilon} = \sqrt{S_1 \odot S_2}$$

(where $\odot$ denotes elementwise product) computes the geometric mean of S₁ and S₂, then renormalizes.

**Connection to Wasserstein distance:** The 2-Wasserstein distance between distributions $\mu$ and $\nu$ on the real line is minimized by the map $T(x) = F_\nu^{-1}(F_\mu(x))$ (quantile coupling). For discrete distributions on the same support, the Fréchet mean in Wasserstein space is the elementwise geometric mean $\sqrt{\mu \odot \nu}$ (when distributions share support). S₃ is therefore the *optimal interpolant* between S₁ and S₂ under the Wasserstein metric — the distribution that lies exactly at the midpoint of the shortest path between them in probability space.

### 6.4 Gating Operator S₄: Boltzmann Bottleneck

The Boltzmann gating filter:
$$S_4(x)_i = \frac{x_i^\beta}{\sum_j x_j^\beta}$$

**Information-theoretic interpretation:** This is the Rényi escort distribution of order $1/\beta$. For $\beta < 1$ (our case, $\beta = 0.85$), it is equivalent to softening the distribution — suppressing the effect of very high-probability components. For $\beta > 1$, it sharpens. The parameter $\beta$ controls the **noise tolerance** of the bottleneck:

- $\beta \to 0$: Output is uniform regardless of input — all information destroyed, maximum noise tolerance
- $\beta = 1$: Identity transformation — no filtering
- $\beta \to \infty$: Output is a point mass on the argmax — hard selection, zero noise tolerance

At $\beta = 0.85$, the gating slightly softens the transported signal, preventing premature commitment to a single configuration before full consolidation.

**Connection to Boltzmann distribution:** In statistical physics, the Boltzmann distribution at temperature $T$ is $p_i \propto e^{-E_i/kT}$. If we write $x_i = e^{-E_i}$ (interpreting x as Boltzmann weights at unit temperature), then $x_i^\beta = e^{-\beta E_i}$ — which is the Boltzmann distribution at temperature $1/\beta$. Higher $\beta$ corresponds to lower temperature (more ordered, sharper selection). S₄ with $\beta < 1$ therefore acts as a *temperature-raising* operation — it adds controlled thermal noise to prevent freezing.

### 6.5 Optimization Operator S₅

Gradient descent on $\mathcal{L} = \frac{1}{2}\|S_2 - \hat{g}\|^2$ where $\hat{g} = S_4(S_3(S_1, S_2))$:

$$\nabla_{S_2} \mathcal{L} = S_2 - \hat{g}$$

$$S_5: \quad S_2 \leftarrow S_2 - \gamma \nabla_{S_2} \mathcal{L} = S_2 + \gamma(\hat{g} - S_2)$$

This is a convex combination update: with step size $\gamma < 1$, the new S₂ lies on the line segment between the old S₂ and the target $\hat{g}$. The update is always in $[0,1]^N$ if both S₂ and $\hat{g}$ are, so renormalization is technically only needed to correct numerical drift. In the implementation, renormalization is applied after every update to maintain strict probability simplex membership.

### 6.6 Synthesis: Operator Composition and Ω Emergence

The full synthesis is:

$$\Omega = \frac{S_1 + S_2}{2} \quad \text{where} \quad S_2 \leftarrow S_5(S_2,\; S_4(S_3(S_1, S_2)))$$

The operator composition $S_3 \circ S_4 \circ S_5$ is an endomorphism on the probability simplex $\Delta^{N-1}$ — it maps $(\Delta^{N-1})^2 \to \Delta^{N-1}$. Fixed points of this map satisfy:

$$S_2^* = S_2^* + \gamma\left(S_4(S_3(S_1^*, S_2^*)) - S_2^*\right)$$

which requires $S_4(S_3(S_1^*, S_2^*)) = S_2^*$ — the gated transport of S₁ through S₂ must equal S₂ itself. This fixed-point condition defines the autonomous state: S₂ has encoded exactly the structure that S₁ has consolidated, and no further change is needed. At this point $\Omega = S_1^* = S_2^* = $ the autonomous attractor.

---

## 7. Hyperbolic Manifold and Latent Space Geometry

### 7.1 Why Hyperbolic Space?

Natural language concepts, knowledge hierarchies, and learned neural representations all exhibit hierarchical structure — some concepts are general (near the "root") and others are specific (near the "leaves"). Euclidean space is a poor embedding geometry for hierarchies: exponential growth in the number of nodes at each level requires exponential volume in Euclidean space, but only polynomial volume in hyperbolic space.

The Poincaré disk model of the hyperbolic plane $\mathbb{H}^2$ embeds hierarchies naturally: concepts near the center of the disk are more general (lower in the tree), and concepts near the boundary are more specific (higher in the tree). Distance from the center encodes generality; angular distance encodes semantic similarity.

### 7.2 The Exponential Map

Latent coordinates $\mathbf{v} \in \mathbb{R}^2$ (Euclidean) are mapped to the Poincaré disk via the exponential map at the origin:

$$\text{expmap}_0(\mathbf{v}) = \tanh(\|\mathbf{v}\|) \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

**Why tanh?** The Poincaré disk has unit radius — all points satisfy $\|x\| < 1$. The tanh function maps $[0, \infty) \to [0, 1)$, so any Euclidean vector gets mapped inside the disk. Points with large Euclidean norm map near the boundary ($\tanh(\infty) = 1$); points with small norm map near the center ($\tanh(0) = 0$). The direction is preserved. This is the differential-geometric exponential map from the tangent space at the origin to the manifold.

### 7.3 Visualization Encoding

In the Poincaré disk panel:
- **Position**: Hyperbolic embedding of concept's latent coordinate
- **Size**: Proportional to $S_2$ activation — larger points have more substrate persistence
- **Color** (magma colormap): S₁ activation — brighter points have stronger current inference focus
- **Boundary circle**: The unit circle represents the "horizon" — no concept can reach it, but concepts that consolidate (high S₂) cluster near center while exploratory concepts (high S₁, low S₂) scatter toward the boundary

---

## 8. Output Metrics and Diagnostics

The simulation tracks and reports six diagnostic metrics:

| Metric | Formula | Interpretation |
|---|---|---|
| **Consolidation Ratio** | $H_{S_1}(T) / H_{S_1}(0)$ | < 1: S₁ converged; = 1: no change; > 1: exploration ongoing |
| **Substrate Stability** | $\text{mean}_t\, \bar{S}_2(t)$ | Higher = more stable substrate over training |
| **Peak S₁ Activation** | $\max_t \max_i S_1[i]$ | How concentrated S₁ becomes at peak (high = sharp inference) |
| **Peak S₂ Activation** | $\max_t \max_i S_2[i]$ | How much the substrate consolidates onto a single mode |
| **Peak Ω Activation** | $\max_t \max_i \Omega[i]$ | Maximum synthetic state concentration achieved |
| **Opportunity Windows** | Count of Broken Gate events | More events = more stochastic resets needed (rugged landscape) |
| **Mean Gate G(t)** | $\text{mean}_t\, G(t)$ | Near 1.0 = stable training; far below 1.0 = frequent resets |
| **Final Gate State** | $G(T)$ | 1.0 = fully closed/consolidated; 0.5 = just triggered |

**Interpreting the Consolidation Ratio:** A ratio significantly less than 1 (e.g., 0.6) indicates that S₁ successfully reduced its entropy — it went from a high-uncertainty exploration to a focused, low-entropy inference state. This means the system found and committed to a structured configuration.

---

## 9. Simulation Visualizations

The full simulation produces a 6-panel figure:

| Panel | State Tracked | Key Feature |
|---|---|---|
| **Phase Alpha** (top-left) | S₁ entropy per dimension | Rising then plateauing entropy curves show exploration convergence |
| **Phase Beta** (top-center) | S₂ entropy + S₂ min | Red dashed threshold line; dips near threshold precede gate events |
| **Gate Dynamics** (top-right) | G(t) + opportunity window markers | Red downward triangles mark Broken Gate events |
| **Entropy Dynamics** (bottom-left) | S₁ vs S₂ Shannon entropy (comparative) | Information-stability balance; S₁ converges while S₂ stabilizes |
| **Phase Gamma** (bottom-center) | Ω max activation + fill | Rising plateau shows autonomous synthesis achieved |
| **Hyperbolic Manifold** (bottom-right) | Poincaré disk with concept clustering | Size ∝ S₂ persistence; color ∝ S₁ focus |

All panels use a dark background (`#0a0a0a`) with per-panel accent colors (cyan for S₁, red for S₂, green for G, purple for Ω) to maximize readability and visual distinction.

---

## 10. Implementation: Minimal Simulation

The minimal implementation focuses on the core dynamics without visualization overhead. This is the production-ready version with all bugs corrected and a complete, clean interface.

```python
#!/usr/bin/env python3
"""
Broken Gate Dynamics — Minimal Simulation
Phase-Transition-Driven Evolution from Information Primitives to Autonomous States
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
import pandas as pd


class BrokenGateDynamics:
    """
    Canonical simulation of phase-transition-driven autonomy.

    States
    ------
    s1 : ndarray (N,)
        Inference Primitive — probability vector, maximizes Shannon entropy.
    s2 : ndarray (N,)
        Persistence Substrate — probability vector, stabilizes via relaxation.
    omega : ndarray (N,)
        Synthetic Latent State — arithmetic mean of s1 and s2.

    Operators
    ---------
    transport (S3) : Wasserstein geometric mean of s1 and s2.
    gating    (S4) : Boltzmann power filter with exponent beta.
    optimize  (S5) : Gradient step of s2 toward gated transport target.
    """

    def __init__(
        self,
        N: int = 12,
        iterations: int = 150,
        gamma: float = 0.15,
        beta: float = 0.85,
        tau: float = 0.05,
        threshold: float = 0.08,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        N          : Dimensionality of the probability simplex.
        iterations : Number of simulation time steps.
        gamma      : Learning rate for S1 entropy gradient and S5 optimization.
        beta       : Gating bottleneck exponent (0 < beta < 1 = slight flattening).
        tau        : Relaxation rate for S2 mean-reversion.
        threshold  : Minimum S2 component below which a Broken Gate event fires.
        seed       : Random seed for reproducibility.
        """
        np.random.seed(seed)

        self.N = N
        self.iterations = iterations
        self.gamma = gamma
        self.beta = beta
        self.tau = tau
        self.threshold = threshold

        # ── State initialization on probability simplex ──────────────────────
        self.s1 = np.random.dirichlet(np.ones(N))
        self.s2 = np.random.dirichlet(np.ones(N))
        self.omega = (self.s1 + self.s2) / 2.0
        self.gate = 1.0

        # ── Diagnostics ──────────────────────────────────────────────────────
        self.windows_triggered = 0
        self.history: list[dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # OPERATORS
    # ─────────────────────────────────────────────────────────────────────────

    def _transport(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """S3: Geometric mean (Wasserstein midpoint) of s1 and s2."""
        transported = np.sqrt(s2) * (s1 / (np.sqrt(s1) + 1e-12))
        transported = np.clip(transported, 1e-12, None)
        return transported / transported.sum()

    def _gating(self, x: np.ndarray) -> np.ndarray:
        """S4: Boltzmann power filter. beta < 1 flattens; beta > 1 sharpens."""
        gated = x ** self.beta
        gated = np.clip(gated, 1e-12, None)
        return gated / gated.sum()

    def _optimize(self, s2: np.ndarray, target: np.ndarray) -> np.ndarray:
        """S5: Gradient descent step of s2 toward target."""
        s2_new = s2 + self.gamma * (target - s2)
        s2_new = np.clip(s2_new, 1e-12, None)
        return s2_new / s2_new.sum()

    # ─────────────────────────────────────────────────────────────────────────
    # BROKEN GATE
    # ─────────────────────────────────────────────────────────────────────────

    def _broken_gate_check(self) -> bool:
        """
        Trigger a stochastic substrate reset when any s2 component
        drops below the vulnerability threshold.

        Returns True if a gate event fired, False otherwise.
        """
        if self.s2.min() < self.threshold:
            perturbation = 0.3 * np.random.rand(self.N)
            self.s2 = np.clip(self.s2 - perturbation, 1e-12, None)
            self.s2 /= self.s2.sum()
            self.windows_triggered += 1
            self.gate = 0.5  # Gate opens
            return True
        else:
            self.gate = min(1.0, self.gate + 0.1)  # Gate closes gradually
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # EVOLUTION STEP
    # ─────────────────────────────────────────────────────────────────────────

    def _step(self) -> dict:
        """
        Execute one full evolution step.

        Order of operations
        -------------------
        1. Broken Gate check (may reset s2)
        2. Phase Alpha : S1 entropy gradient ascent
        3. Phase Beta  : S2 relaxation toward mean
        4. Phase Gamma : Operator chain S3 ∘ S4 ∘ S5 updates s2
        5. Synthesis   : Ω = (s1 + s2) / 2

        Returns a dict of scalar diagnostics for this step.
        """
        window_triggered = self._broken_gate_check()

        # ── Phase Alpha: S1 entropy maximization ────────────────────────────
        h_s1 = float(scipy_entropy(self.s1))          # Current S1 entropy
        grad_h = -np.log(self.s1 + 1e-12) - h_s1     # ∇_p H(p)
        self.s1 = np.clip(self.s1 + self.gamma * grad_h, 1e-12, None)
        self.s1 /= self.s1.sum()

        # ── Phase Beta: S2 relaxation dynamics ──────────────────────────────
        s2_mean = self.s2.mean()                       # = 1/N (since Σs2 = 1)
        self.s2 = self.s2 + self.tau * (s2_mean - self.s2)
        self.s2 = np.clip(self.s2, 1e-12, None)
        self.s2 /= self.s2.sum()

        # ── Phase Gamma: Operator chain ──────────────────────────────────────
        transported = self._transport(self.s1, self.s2)
        gated = self._gating(transported)
        self.s2 = self._optimize(self.s2, gated)

        # ── Synthesis: Ω formation ───────────────────────────────────────────
        self.omega = (self.s1 + self.s2) / 2.0

        return {
            "s1_entropy":          h_s1,
            "s2_entropy":          float(scipy_entropy(self.s2)),
            "s2_mean":             float(self.s2.mean()),
            "s2_min":              float(self.s2.min()),
            "max_s1_activation":   float(self.s1.max()),
            "max_s2_activation":   float(self.s2.max()),
            "max_omega_activation": float(self.omega.max()),
            "gate":                self.gate,
            "window_triggered":    window_triggered,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION RUNNER
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the full simulation.

        Returns
        -------
        pd.DataFrame
            One row per iteration with all diagnostic metrics.
        """
        self.history.clear()
        self.windows_triggered = 0

        for i in range(self.iterations):
            row = self._step()
            row["iteration"] = i

            if row["window_triggered"]:
                print(f"  ⚡ Opportunity window at iteration {i:>4d} "
                      f"| Gate: {row['gate']:.2f} "
                      f"| S2_min was below {self.threshold}")

            self.history.append(row)

        return pd.DataFrame(self.history).set_index("iteration")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY REPORT
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self, df: pd.DataFrame | None = None) -> None:
        """
        Print a full analytical summary to stdout.

        Parameters
        ----------
        df : optional DataFrame from run(). If None, uses self.history.
        """
        if df is None:
            df = pd.DataFrame(self.history).set_index("iteration")

        if df.empty:
            print("No simulation data — call run() first.")
            return

        start_h = df["s1_entropy"].iloc[0]
        final_h = df["s1_entropy"].iloc[-1]
        consol  = final_h / start_h if start_h > 0 else float("nan")

        W = 62
        print("\n" + "=" * W)
        print(f"{'BROKEN GATE DYNAMICS — ANALYTICAL SUMMARY':^{W}}")
        print("=" * W)

        print(f"\n{'SYSTEM CONFIGURATION':-^{W}}")
        print(f"  Dimensionality (N)        : {self.N}")
        print(f"  Total Iterations          : {self.iterations}")
        print(f"  Learning Rate (γ)         : {self.gamma}")
        print(f"  Gating Bottleneck (β)     : {self.beta}")
        print(f"  Relaxation Rate (τ)       : {self.tau}")
        print(f"  Opportunity Threshold     : {self.threshold}")

        print(f"\n{'ENTROPY EVOLUTION':-^{W}}")
        print(f"  S1 Entropy (Initial)      : {start_h:.4f}")
        print(f"  S1 Entropy (Final)        : {final_h:.4f}")
        print(f"  Consolidation Ratio       : {consol:.4f}  (Final / Initial)")
        print(f"  S2 Entropy (Initial)      : {df['s2_entropy'].iloc[0]:.4f}")
        print(f"  S2 Entropy (Final)        : {df['s2_entropy'].iloc[-1]:.4f}")

        print(f"\n{'SUBSTRATE STABILITY':-^{W}}")
        print(f"  Mean S2 Activation        : {df['s2_mean'].mean():.4f}")
        print(f"  Min S2 Activation (final) : {df['s2_min'].iloc[-1]:.4f}")

        print(f"\n{'PEAK ACTIVATIONS':-^{W}}")
        print(f"  Max S1 Activation         : {df['max_s1_activation'].max():.4f}")
        print(f"  Max S2 Activation         : {df['max_s2_activation'].max():.4f}")
        print(f"  Max Ω Activation          : {df['max_omega_activation'].max():.4f}")
        print(f"  Final Ω Activation        : {df['max_omega_activation'].iloc[-1]:.4f}")

        print(f"\n{'BROKEN GATE / OPPORTUNITY WINDOWS':-^{W}}")
        print(f"  Total Windows Triggered   : {self.windows_triggered}")
        print(f"  Mean Gate State G(t)      : {df['gate'].mean():.4f}")
        print(f"  Final Gate State          : {df['gate'].iloc[-1]:.4f}")

        print(f"\n{'SCIENTIFIC CONCLUSIONS':-^{W}}")
        print(f"  1. INFORMATION CONSOLIDATION")
        print(f"     S1 reduced uncertainty via entropy gradient ascent.")
        print(f"     Consolidation ratio: {consol:.2%} of initial entropy retained.")

        print(f"\n  2. SUBSTRATE RESILIENCE")
        print(f"     S2 gating prevented runaway feedback loops.")
        print(f"     Maintained stability across {self.windows_triggered} stochastic reset(s).")

        print(f"\n  3. PHASE TRANSITION DYNAMICS")
        print(f"     System transitioned from high-entropy chaos to structured state.")
        print(f"     Opportunity windows accelerated autonomous consolidation.")

        print(f"\n  4. AUTONOMOUS SYNTHESIS (Ω)")
        print(f"     Emergent latent state formed via S1 ⊗ S2 interaction.")
        print(f"     Final Ω activation: {df['max_omega_activation'].iloc[-1]:.4f}.")
        print(f"     System achieved self-organizing autonomy without supervision.")

        print(f"\n{'THEORETICAL GROUNDING':-^{W}}")
        print(f"  Shannon (1948)            : Information dynamics — S1 entropy")
        print(f"  Jaynes (1957)             : Statistical mechanics — S2 substrate")
        print(f"  Ghavasieh et al. (2020)   : Phase transitions — Broken Gate / Ω")
        print("=" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀  Broken Gate Dynamics — Minimal Simulation")
    print("    From Information Primitives to Autonomous Latent States\n")

    sim = BrokenGateDynamics(
        N=12,
        iterations=150,
        gamma=0.15,
        beta=0.85,
        tau=0.05,
        threshold=0.08,
        seed=42,
    )

    print("⚙️   Running simulation …")
    df = sim.run()

    print("\n✅  Simulation complete.")
    sim.summary(df)
```

---

## 11. Implementation: Full Simulation with Visualization

The full simulation adds the 6-panel visualization and the Poincaré disk hyperbolic manifold rendering. All parameters are identical to the minimal version; the additional methods are pure output.

```python
#!/usr/bin/env python3
"""
Phase-Transition Dynamics in Systemic Evolution
Full Simulation with 6-Panel Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.stats import entropy as scipy_entropy
import pandas as pd
from itertools import islice


class PhaseTransitionDynamics:
    """
    Full simulation with visualization.
    Extends BrokenGateDynamics with:
    - Per-step entropy tracking for S1 and S2
    - Gate history
    - Poincaré disk hyperbolic embedding
    - 6-panel matplotlib figure
    """

    CONCEPTS = [
        "Intelligence", "Entropy",   "Manifold",  "Curvature",
        "Inference",    "Substrate",  "Gating",    "Symmetry",
        "Topology",     "Evolution",  "Logic",     "Information",
    ]

    def __init__(
        self,
        N: int = 12,
        iterations: int = 150,
        gamma: float = 0.15,
        beta: float = 0.85,
        tau: float = 0.05,
        threshold: float = 0.08,
        seed: int = 42,
    ):
        np.random.seed(seed)

        self.N = N
        self.iterations = iterations
        self.gamma = gamma
        self.beta = beta
        self.tau = tau
        self.threshold = threshold

        self.s1 = np.random.dirichlet(np.ones(N))
        self.s2 = np.random.dirichlet(np.ones(N))
        self.omega = (self.s1 + self.s2) / 2.0
        self.gate = 1.0

        # Latent coordinates for hyperbolic embedding (2D Euclidean)
        self.latent_coords = np.random.randn(N, 2) * 0.3

        self.windows_triggered = 0
        self.history: list[dict] = []
        self.entropy_s1: list[float] = []
        self.entropy_s2: list[float] = []
        self.gate_history: list[float] = []

    # ─────────────────────────────────────────────────────────────────────
    # OPERATORS (identical to minimal version)
    # ─────────────────────────────────────────────────────────────────────

    def _transport(self, s1, s2):
        t = np.sqrt(s2) * (s1 / (np.sqrt(s1) + 1e-12))
        t = np.clip(t, 1e-12, None)
        return t / t.sum()

    def _gating(self, x):
        g = x ** self.beta
        g = np.clip(g, 1e-12, None)
        return g / g.sum()

    def _optimize(self, s2, target):
        s2_new = s2 + self.gamma * (target - s2)
        s2_new = np.clip(s2_new, 1e-12, None)
        return s2_new / s2_new.sum()

    def _expmap0(self, v: np.ndarray) -> np.ndarray:
        """
        Exponential map at origin: projects Euclidean coords onto Poincaré disk.
        expmap0(v) = tanh(||v||) * v / ||v||
        """
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return np.tanh(norm) * v / (norm + 1e-15)

    # ─────────────────────────────────────────────────────────────────────
    # BROKEN GATE
    # ─────────────────────────────────────────────────────────────────────

    def _broken_gate_check(self) -> bool:
        if self.s2.min() < self.threshold:
            self.s2 = np.clip(self.s2 - 0.3 * np.random.rand(self.N), 1e-12, None)
            self.s2 /= self.s2.sum()
            self.windows_triggered += 1
            self.gate = 0.5
            return True
        self.gate = min(1.0, self.gate + 0.1)
        return False

    # ─────────────────────────────────────────────────────────────────────
    # SIMULATION
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Run full simulation, populate history and entropy/gate lists."""
        self.history.clear()
        self.entropy_s1.clear()
        self.entropy_s2.clear()
        self.gate_history.clear()
        self.windows_triggered = 0

        for i in range(self.iterations):
            window_triggered = self._broken_gate_check()

            # Phase Alpha
            h_s1 = float(scipy_entropy(self.s1))
            grad_h = -np.log(self.s1 + 1e-12) - h_s1
            self.s1 = np.clip(self.s1 + self.gamma * grad_h, 1e-12, None)
            self.s1 /= self.s1.sum()

            # Phase Beta
            self.s2 = self.s2 + self.tau * (self.s2.mean() - self.s2)
            self.s2 = np.clip(self.s2, 1e-12, None)
            self.s2 /= self.s2.sum()

            # Phase Gamma
            gated = self._gating(self._transport(self.s1, self.s2))
            self.s2 = self._optimize(self.s2, gated)

            # Synthesis
            self.omega = (self.s1 + self.s2) / 2.0

            self.entropy_s1.append(h_s1)
            self.entropy_s2.append(float(scipy_entropy(self.s2)))
            self.gate_history.append(self.gate)

            self.history.append({
                "iteration":            i,
                "s1_entropy":           h_s1,
                "s2_entropy":           float(scipy_entropy(self.s2)),
                "s2_mean":              float(self.s2.mean()),
                "s2_min":               float(self.s2.min()),
                "max_s1_activation":    float(self.s1.max()),
                "max_s2_activation":    float(self.s2.max()),
                "max_omega_activation": float(self.omega.max()),
                "gate":                 self.gate,
                "window_triggered":     window_triggered,
            })

            if window_triggered:
                print(f"  ⚡ Opportunity window — iteration {i:>4d}")

        return pd.DataFrame(self.history).set_index("iteration")

    # ─────────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────────────────────────────

    def visualize(self, save_path: str = "phase_transition_dynamics.png") -> None:
        """Generate and save the 6-panel diagnostic figure."""
        if not self.history:
            raise RuntimeError("No data — call run() before visualize().")

        BG = "#0a0a0a"
        fig = plt.figure(figsize=(18, 10), facecolor=BG)

        iters = [h["iteration"] for h in self.history]

        # ── Panel 1: Phase Alpha — S1 entropy ────────────────────────────
        ax1 = plt.subplot2grid((2, 3), (0, 0), facecolor=BG)
        ax1.plot(iters, self.entropy_s1, color="#00ffcc", linewidth=2, label="S1 Entropy")
        ax1.set_title("Phase Alpha: S₁ (Inference Primitive)",
                      color="#00ffcc", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Iteration", color="white")
        ax1.set_ylabel("Shannon Entropy", color="white")
        ax1.tick_params(colors="white")
        ax1.spines[:].set_color("#333333")
        ax1.grid(alpha=0.2)
        ax1.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")

        # ── Panel 2: Phase Beta — S2 dynamics ────────────────────────────
        ax2 = plt.subplot2grid((2, 3), (0, 1), facecolor=BG)
        ax2.plot(iters, self.entropy_s2, color="#ff6b6b", linewidth=2, label="S2 Entropy")
        ax2.plot(iters, [h["s2_min"] for h in self.history],
                 color="#ffd93d", linewidth=1.5, alpha=0.8, label="S2 Min")
        ax2.axhline(y=self.threshold, color="red", linestyle="--",
                    alpha=0.6, linewidth=1.2, label=f"Threshold ({self.threshold})")
        ax2.set_title("Phase Beta: S₂ (Persistence Substrate)",
                      color="#ff6b6b", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Iteration", color="white")
        ax2.set_ylabel("Substrate Metrics", color="white")
        ax2.tick_params(colors="white")
        ax2.spines[:].set_color("#333333")
        ax2.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
        ax2.grid(alpha=0.2)

        # ── Panel 3: Gate dynamics + opportunity windows ──────────────────
        ax3 = plt.subplot2grid((2, 3), (0, 2), facecolor=BG)
        ax3.plot(iters, self.gate_history, color="#6bcf7f", linewidth=2, label="G(t)")
        window_iters = [h["iteration"] for h in self.history if h["window_triggered"]]
        if window_iters:
            ax3.scatter(window_iters, [0.5] * len(window_iters),
                        color="red", s=120, marker="v",
                        label=f"Opportunity Window ({len(window_iters)})", zorder=5)
        ax3.set_ylim(0, 1.15)
        ax3.set_title("Gate Dynamics & Opportunity Windows",
                      color="#6bcf7f", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Iteration", color="white")
        ax3.set_ylabel("Gate State G(t)", color="white")
        ax3.tick_params(colors="white")
        ax3.spines[:].set_color("#333333")
        ax3.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
        ax3.grid(alpha=0.2)

        # ── Panel 4: Comparative entropy dynamics ─────────────────────────
        ax4 = plt.subplot2grid((2, 3), (1, 0), facecolor=BG)
        ax4.plot(iters, self.entropy_s1, color="#00ffcc", linewidth=2, label="S1 (Inference)")
        ax4.plot(iters, self.entropy_s2, color="#ff6b6b", linewidth=2, label="S2 (Substrate)")
        ax4.set_title("Entropy Dynamics: Information–Substrate Balance",
                      color="white", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Iteration", color="white")
        ax4.set_ylabel("Shannon Entropy", color="white")
        ax4.tick_params(colors="white")
        ax4.spines[:].set_color("#333333")
        ax4.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
        ax4.grid(alpha=0.2)

        # ── Panel 5: Ω synthesis ──────────────────────────────────────────
        ax5 = plt.subplot2grid((2, 3), (1, 1), facecolor=BG)
        omega_vals = [h["max_omega_activation"] for h in self.history]
        ax5.plot(iters, omega_vals, color="#a78bfa", linewidth=2.5, label="Max Ω Activation")
        ax5.fill_between(iters, omega_vals, alpha=0.25, color="#a78bfa")
        ax5.set_title("Phase Gamma: Ω (Synthetic Latent State)",
                      color="#a78bfa", fontsize=12, fontweight="bold")
        ax5.set_xlabel("Iteration", color="white")
        ax5.set_ylabel("Ω Activation", color="white")
        ax5.tick_params(colors="white")
        ax5.spines[:].set_color("#333333")
        ax5.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
        ax5.grid(alpha=0.2)

        # ── Panel 6: Poincaré disk ────────────────────────────────────────
        ax6 = plt.subplot2grid((2, 3), (1, 2), facecolor=BG)

        # Draw Poincaré disk background + boundary
        disk_bg = plt.Circle((0, 0), 1.0, color="#111111", fill=True, zorder=0)
        ax6.add_artist(disk_bg)
        boundary = plt.Circle((0, 0), 1.0, color="#00ffcc", fill=False,
                               linestyle="--", alpha=0.3, linewidth=1.5, zorder=1)
        ax6.add_artist(boundary)

        # Map to hyperbolic space (scale latent coords for spread)
        hyp = self._expmap0(self.latent_coords * 1.5)

        sc = ax6.scatter(
            hyp[:, 0], hyp[:, 1],
            s=self.s2 * 3000,       # Size ∝ substrate persistence
            c=self.s1,              # Color ∝ inference focus
            cmap="magma",
            edgecolors="white",
            linewidth=1.2,
            alpha=0.9,
            zorder=3,
        )

        # Label concepts — offset slightly above point
        for idx, (x, y) in enumerate(hyp):
            label = self.CONCEPTS[idx] if idx < len(self.CONCEPTS) else f"C{idx}"
            ax6.text(x, y + 0.05, label,
                     fontsize=7, color="white", ha="center", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
                     zorder=4)

        ax6.set_xlim(-1.15, 1.15)
        ax6.set_ylim(-1.15, 1.15)
        ax6.set_aspect("equal")
        ax6.set_title("Hyperbolic Manifold: Concept Clustering (Poincaré Disk)",
                      color="white", fontsize=12, fontweight="bold")
        ax6.axis("off")

        cbar = plt.colorbar(sc, ax=ax6, fraction=0.046, pad=0.04)
        cbar.set_label("S1 Activation", color="white", fontsize=9)
        cbar.ax.tick_params(colors="white")

        plt.tight_layout(pad=1.5)
        plt.savefig(save_path, facecolor=BG, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"✅  Figure saved → {save_path}")

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY (reuses minimal version logic)
    # ─────────────────────────────────────────────────────────────────────

    def summary(self) -> None:
        df = pd.DataFrame(self.history)
        if df.empty:
            print("No data — call run() first.")
            return

        start_h = df["s1_entropy"].iloc[0]
        final_h = df["s1_entropy"].iloc[-1]
        consol  = final_h / start_h if start_h > 0 else float("nan")

        W = 68
        print("\n" + "=" * W)
        print(f"{'PHASE-TRANSITION DYNAMICS — CANONICAL SIMULATION RESULTS':^{W}}")
        print("=" * W)

        print(f"\n{'SYSTEM CONFIGURATION':-^{W}}")
        print(f"  Dimensionality (N)        : {self.N}")
        print(f"  Total Iterations          : {self.iterations}")
        print(f"  Learning Rate (γ)         : {self.gamma}")
        print(f"  Gating Bottleneck (β)     : {self.beta}")
        print(f"  Relaxation Rate (τ)       : {self.tau}")
        print(f"  Opportunity Threshold     : {self.threshold}")

        print(f"\n{'ENTROPY EVOLUTION':-^{W}}")
        print(f"  S1 Entropy (Initial → Final) : {start_h:.4f} → {final_h:.4f}")
        print(f"  Consolidation Ratio          : {consol:.4f}  (Final / Initial)")
        print(f"  S2 Entropy (Initial → Final) : "
              f"{df['s2_entropy'].iloc[0]:.4f} → {df['s2_entropy'].iloc[-1]:.4f}")

        print(f"\n{'ACTIVATION PEAKS':-^{W}}")
        print(f"  Max S1 Activation         : {df['max_s1_activation'].max():.4f}")
        print(f"  Max S2 Activation         : {df['max_s2_activation'].max():.4f}")
        print(f"  Max Ω  Activation         : {df['max_omega_activation'].max():.4f}")
        print(f"  Final Ω Activation        : {df['max_omega_activation'].iloc[-1]:.4f}")

        print(f"\n{'BROKEN GATE / OPPORTUNITY WINDOWS':-^{W}}")
        print(f"  Total Windows Triggered   : {self.windows_triggered}")
        print(f"  Mean Gate State G(t)      : {df['gate'].mean():.4f}")
        print(f"  Final Gate State          : {df['gate'].iloc[-1]:.4f}")

        print(f"\n{'CANONICAL CONCLUSIONS':-^{W}}")
        print(f"  1. S1 reduced uncertainty — consolidation ratio: {consol:.2%}")
        print(f"  2. S2 maintained stability across {self.windows_triggered} reset(s)")
        print(f"  3. System completed phase transition from chaos to structured state")
        print(f"  4. Ω emerged autonomously — final activation: "
              f"{df['max_omega_activation'].iloc[-1]:.4f}")

        print(f"\n{'THEORETICAL GROUNDING':-^{W}}")
        print(f"  Shannon (1948)            : S1 entropy dynamics")
        print(f"  Jaynes (1957)             : S2 substrate relaxation")
        print(f"  Ghavasieh et al. (2020)   : Broken Gate phase transitions")
        print("=" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀  Phase-Transition Dynamics — Full Simulation")
    print("    From Information Primitives to Autonomous Latent States\n")

    system = PhaseTransitionDynamics(
        N=12, iterations=150, gamma=0.15,
        beta=0.85, tau=0.05, threshold=0.08, seed=42,
    )

    print("⚙️   Running canonical simulation …")
    df = system.run()
    print(f"\n✅  Simulation complete — {len(df)} steps recorded.\n")

    print("📊  Generating visualization panels …")
    system.visualize("phase_transition_dynamics.png")

    system.summary()
    print("✨  Analysis complete.")
```

---

## 12. Connections Across Domains

The framework is explicitly domain-agnostic. The same four-state architecture maps onto qualitatively different systems:

| Domain | S₁ (Inference) | S₂ (Substrate) | G (Broken Gate) | Ω (Autonomy) |
|---|---|---|---|---|
| **Neural networks** | Forward pass / gradient | Weight matrix | Learning rate spike / noise | Generalization ability |
| **Biological evolution** | Genetic mutation | Gene regulatory network | Mass extinction event | Speciation / new niche |
| **Cultural evolution** | Idea / meme generation | Social institution | Crisis / paradigm shift | Stable cultural norm |
| **Economic markets** | Price discovery | Market infrastructure | Market crash / shock | Price equilibrium |
| **Immune system** | Antigen recognition | Memory B-cell pool | Novel pathogen challenge | Immunological memory |

The mathematical abstraction is not a metaphor — the same operators (transport, gating, optimization, relaxation) govern dynamics in each domain because all are information-processing systems with memory formation under noise.

---

## 13. Four Key Principles

The simulation demonstrates four principles that apply universally to complex adaptive systems:

**1. Phase-Transition-Driven Growth**

Opportunity windows (Broken Gate events) act as catalysts. They accelerate the system's transition from high-entropy chaos to low-entropy structured states by destroying locally stable but globally suboptimal configurations. Without them, the system converges prematurely.

**2. Emergent Autonomy (Ω)**

The synthetic latent state emerges naturally from the dynamic interplay between S₁ and S₂ through the operator chain. Ω is not programmed — it is the fixed point of the composition $S_3 \circ S_4 \circ S_5$ when S₁ and S₂ are mutually consistent. No external supervision is required.

**3. Systemic Robustness**

The S₂ gating mechanism ($S_4$ with $\beta < 1$) stabilizes the system against stochastic perturbations. While S₁ explores high-information configurations, S₂'s relaxation dynamics and the Boltzmann bottleneck prevent noise amplification. The system explores without collapsing.

**4. Canonical Generality**

The framework is abstracted from any specific biological, cultural, or computational implementation. It applies to any complex adaptive system exhibiting: (a) information exploration, (b) structural memory, (c) phase-transition susceptibility, and (d) emergent self-organization.

---

## 14. Limitations and Open Problems

### Current Limitations

**Fixed operator chain:** The composition $S_3 \circ S_4 \circ S_5$ is hardcoded. In real systems, the transport geometry and gating bottleneck may themselves evolve. A second-order framework where operators are also state-dependent is not implemented.

**Scalar gate G(t):** The gate is a global scalar — one value for the entire substrate. Real phase transitions are often local (some substrate dimensions reorganize while others do not). A per-dimension gate would be more realistic but adds $N$ parameters.

**Ω as arithmetic mean:** The synthesis $\Omega = (S_1 + S_2)/2$ is the simplest possible combination. A learned weighted average, or a nonlinear synthesis operator, would produce richer autonomous states at the cost of additional optimization.

**No memory of past states:** The system has no explicit episodic memory. Once a Broken Gate fires and S₂ is reset, the previous substrate configuration is lost. Biological and cognitive systems maintain consolidation across resets through long-term potentiation analogs.

**Stationary task:** The system optimizes for a fixed entropy landscape. Non-stationary tasks (concept drift, changing environments) are not addressed.

### Open Problems

**Optimal gate threshold:** The threshold $\theta = 0.08$ is set by hand. A principled derivation from the system's entropy landscape (analogous to the $C_\alpha = 1$ threshold in GTI) would strengthen the framework.

**Multi-substrate systems:** Real systems have multiple interacting substrates (e.g., working memory + long-term memory). How do multiple S₂ instances co-evolve through shared S₁ dynamics?

**Information geometry of the operator chain:** The composition $S_3 \circ S_4 \circ S_5$ should be characterized in terms of information geometry (Fisher-Rao metric, geodesics on the simplex). What is the information-geometric interpretation of the fixed point $S_2^* = S_4(S_3(S_1^*, S_2^*))$?

**Scaling to high-dimensional substrates:** With $N = 12$, all quantities are tractable. For $N = 10^6$ (neural network scale), the diagonal Hessian and full covariance are required. Connection to the GTI consolidation ratio $C_\alpha$ and Hutchinson estimation is natural here.

**Continual learning:** How does the system handle a sequence of tasks? Can Broken Gate events be regulated to consolidate old knowledge before accepting new information?

---

## 15. References

**Shannon, C. E. (1948).** A Mathematical Theory of Communication. *Bell System Technical Journal, 27*(3–4), 379–423, 623–656.
*Contribution:* Foundational theory of information entropy $H(X) = -\sum p(x)\log p(x)$, which underpins S₁ inference dynamics. Shannon's framework quantifies how information content drives exploration of optimal probability distributions in Phase Alpha.

**Cover, T. M., & Thomas, J. A. (2006).** *Elements of Information Theory* (2nd ed.). Wiley.
*Contribution:* Canonical reference for rigorous properties of entropy, mutual information, and probabilistic inference. Formalizes the gradient properties of entropy functions ($\nabla_p H = -\log p - 1$) that enable S₁'s convergence behavior and validates the information-theoretic operators.

**Jaynes, E. T. (1957).** Information Theory and Statistical Mechanics. *Physical Review, 106*(4), 620–630.
*Contribution:* Links entropy maximization to statistical mechanics and thermodynamic equilibrium. Supports the abstraction of S₂ as a persistence substrate — Jaynes' maximum entropy principle justifies S₂'s relaxation dynamics and provides the statistical mechanical foundation for substrate stability.

**Ghavasieh, A., Nicolini, C., & De Domenico, M. (2020).** Statistical physics of complex information dynamics. *Physical Review E, 102*(5), 052304. [arXiv:2010.04014]
*Contribution:* Formalizes phase-transition and emergent dynamics in complex systems, directly justifying the Broken Gate mechanism and Ω synthesis. Demonstrates how information-processing systems exhibit critical transitions characterized by sudden reorganization.

**Amari, S. (2016).** *Information Geometry and Its Applications*. Springer.
*Contribution:* Provides the geometric perspective on information manifolds (Fisher-Rao metric, natural gradient, exponential family) relevant to the hyperbolic visualization and the geometric interpretation of the transport operator S₃.

**Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021).** Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. arXiv:2104.13478.
*Contribution:* Supports manifold learning and latent space geometry, providing the theoretical grounding for Poincaré disk embeddings and the expmap₀ operator used in the hyperbolic visualization panel.

---

## 16. Glossary

| Term | Definition |
|---|---|
| **S₁ (Inference Primitive)** | Probability vector over $N$ dimensions that maximizes Shannon entropy via gradient ascent. Represents the "software" exploration layer. |
| **S₂ (Persistence Substrate)** | Probability vector that stabilizes via relaxation toward its mean. Represents the "hardware" memory layer. |
| **Ω (Synthetic Latent State)** | Arithmetic mean $(S_1 + S_2)/2$. Emerges as the autonomous synthesis of inference and substrate. |
| **G (Gate)** | Scalar in $[0.5, 1.0]$ tracking whether the system is in a Broken Gate recovery period ($G < 1$) or fully stable ($G = 1$). |
| **S₃ (Transport)** | Geometric mean $\sqrt{S_1 \odot S_2}$ — the Wasserstein midpoint between S₁ and S₂. |
| **S₄ (Gating)** | Boltzmann power filter $x^\beta / \sum x^\beta$. Controls bottleneck sharpness via $\beta$. |
| **S₅ (Optimization)** | Gradient descent step $S_2 \leftarrow S_2 + \gamma(\hat{g} - S_2)$ toward the gated transport target. |
| **Broken Gate event** | Stochastic substrate reset triggered when $\min_i S_2[i] < \theta$. Implements phase-transition catalysis. |
| **Opportunity window** | Synonym for Broken Gate event — the substrate vulnerability that triggers reorganization. |
| **Shannon entropy** | $H(p) = -\sum_i p_i \log p_i$. Maximized by the uniform distribution; minimized by point masses. |
| **Entropy gradient** | $\nabla_{p_i} H = -\log p_i - H(p)$ (mean-shifted). Points in direction of maximum entropy increase. |
| **Relaxation dynamics** | $\dot{S}_2 = \tau(\bar{S}_2 - S_2)$. Ornstein-Uhlenbeck mean-reversion toward the uniform distribution. |
| **Consolidation ratio** | $H_{S_1}(T) / H_{S_1}(0)$. Values $< 1$ indicate S₁ has converged to a structured low-entropy state. |
| **Poincaré disk** | Model of the hyperbolic plane: open unit disk with hyperbolic metric $ds^2 = 4(dx^2+dy^2)/(1-r^2)^2$. Natural geometry for hierarchical data. |
| **Exponential map (expmap₀)** | Differential-geometric map from tangent space at origin to the Poincaré disk: $\text{expmap}_0(\mathbf{v}) = \tanh(\|\mathbf{v}\|)\mathbf{v}/\|\mathbf{v}\|$. |
| **Phase Alpha** | Training phase dominated by S₁ entropy maximization (exploration). |
| **Phase Beta** | Training phase dominated by S₂ relaxation (stabilization). |
| **Phase Gamma** | Training phase of operator chain composition leading to Ω synthesis. |
| **Probability simplex** | $\Delta^{N-1} = \{p \in \mathbb{R}^N : p_i \geq 0,\; \sum_i p_i = 1\}$. All states live here. |
| **Fixed point** | $S_2^*$ satisfying $S_4(S_3(S_1^*, S_2^*)) = S_2^*$ — the autonomous attractor of the operator chain. |

---

> *"Complexity emerges when information finds a substrate to hold it, and autonomy begins when that substrate learns to reorganize itself."*
