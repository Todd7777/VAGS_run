# Hybrid FlowEdit: Conflict-Aware Dynamic Guidance

This repository contains the research implementation for "Hybrid FlowEdit", an extension of the original FlowEdit method (Stable Diffusion 3) aimed at solving the structure-editability tradeoff.

## Project Overview

Standard editing methods use a constant Classifier-Free Guidance (CFG) scale, which often leads to a dilemma: either the structure is preserved but the edit is weak, or the edit is strong but the structure collapses.

This project introduces two novel guidance strategies based on the analysis of the velocity field difference ($\Delta v$) between source and target trajectories.

## Implemented Strategies

### 1. Late-Start Heuristic (Temporal Anchoring)
This method enforces structural preservation by anchoring the generation to the source prompt during the high-noise phase.

* **Mechanism:** The source prompt is strictly used for the first 20% of the denoising process ($t=0.9 \to 0.72$). The target prompt is injected only after this structural anchor is established.
* **Performance:** Achieves State-of-the-Art structural fidelity (LPIPS ~0.160) while maintaining high semantic alignment (CLIP ~0.346).
* **Script:** `run_late_start_benchmark.py`

### 2. Sign Flip Controller (Dynamic Piecewise Guidance)
This method implements a mathematical "Piecewise Control Policy" based on the Normalized Relative Conflict metric $s(t)$.

$$s(t) = \frac{\| v_{tgt} - v_{src} \|}{\| v_{src} \| + \epsilon}$$

The guidance scale $w(t)$ is modulated dynamically:
* **Early Stage:** Inverse relationship. High conflict implies noise -> Guidance is suppressed to protect structure.
* **Late Stage:** Proportional relationship. High conflict implies semantic edit -> Guidance is boosted to force the edit.
* **Script:** `run_sign_flip_optimization.py`

## Installation & Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt