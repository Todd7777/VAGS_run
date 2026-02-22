# Exponential Conflict-Aware Guidance for Diffusion-Based Image Editing

This repository contains the research implementation for Exponential Conflict-Aware Guidance (CAG), an advanced modulation framework for diffusion-based image editing using Stable Diffusion 3 and Stable Diffusion 3.5. It resolves the structural-editability tradeoff by dynamically scaling Classifier-Free Guidance (CFG) based on the instantaneous velocity conflict between source and target generation trajectories.

## Project Overview

Standard editing methods utilize a constant guidance scale, which forces a compromise: lower scales preserve the original image structure but fail to apply the edit, while higher scales enforce the edit but destroy the background and structural integrity. 

This project introduces a relative conflict metric $s$ to measure the deviation between the target velocity field $v_{t}$ and the source velocity field $v_{s}$:

$$s = \frac{\| v_{t} - v_{s} \|_2}{\| v_{s} \|_2 + \epsilon}$$

Using this metric, we implement two novel exponential modulation strategies to dynamically adjust the guidance weight $w$ during the reverse diffusion process.

## Implemented Strategies

### 1. Time-Agnostic Exponential Decrease
A purely conflict-driven formulation that penalizes guidance when the velocity conflict is high, preventing structural collapse in high-deviation regions.

* **Mechanism:** The base CFG is multiplied by an exponentially decaying factor controlled by the conflict score, bounded by a hyperbolic tangent to prevent numerical instability.
* **Formulation:** $$w = w_{base} \cdot \exp(-\kappa \tanh(s/m))$$
* **Hyperparameters:** $\kappa = 4.0$ (exponential scale), $m = 3.0$ (conflict sensitivity threshold).

### 2. Time-Aware Exponential CAG (Gated)
A temporal formulation that recognizes the changing role of noise across the diffusion process. It suppresses guidance during early structural formation and boosts it during late semantic refinement.

* **Mechanism:** Integrates a sigmoidal temporal gate $\sigma(t)$ that smoothly transitions from early steps to late steps.
* **Formulation:**
$$w(t) = w_{base} \cdot \exp(\kappa \cdot (2\sigma(t) - 1) \cdot \tanh(s/m))$$
* **Performance:** Surpasses current State-of-the-Art methods on PIE-Bench by maintaining strict background preservation while maximizing text alignment.

### 3. State-of-the-Art Baseline Alignment
The codebase strictly aligns with the exact flow inversion protocols from recent literature to ensure direct comparability:
* Target CFG: 13.5
* Source CFG: 3.5
* Generation Start Time: $t_{start} = 0.66$
