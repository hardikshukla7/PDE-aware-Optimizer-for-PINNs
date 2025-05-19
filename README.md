# PDE-Aware Optimizer for Physics-Informed Neural Networks (PINNs)

## 📘 Overview

This project introduces a **PDE-Aware Optimizer**—a lightweight, gradient-sensitive optimization algorithm designed to improve the training stability of **Physics-Informed Neural Networks (PINNs)**, especially for stiff and ill-conditioned partial differential equations (PDEs).

Traditional optimizers like Adam often struggle to balance the competing loss terms in PINNs (initial, boundary, and residual conditions). Our approach dynamically scales parameter updates based on the **variance of per-sample residual gradients**, enabling better convergence in regions with sharp gradients or high stiffness—without incurring the computational overhead of second-order optimizers like SOAP.

## 🔬 Research Paper

Check out the full project report [here (PDF)](./PDE_aware_Optimizer_for_Physics_informed_Neural_Networks.pdf)

## 🚀 Features

* Implements a **PDE-aware optimizer** in JAX.
* Benchmarked on 1D **Burgers’, Allen–Cahn, and Korteweg–de Vries (KdV)** equations.
* Outperforms Adam and SOAP in **convergence smoothness and error distribution**.
* **Mitigates gradient misalignment** (Type I & II) by aligning updates with dominant PDE residuals.
* Fully configurable training pipeline with hyperparameter tuning and support for custom PDEs.

## 📁 Project Structure

```
├── optimizer/                      # PDE-aware optimizer implementation
├── models/                         # MLP architecture for PINNs
├── experiments/                    # Training scripts for Burgers, Allen–Cahn, KdV
├── plots/                          # Heatmaps and training curves
├── utils/                          # Collocation sampling, PDE residuals
├── PDE_aware_Optimizer_for_Physics_informed_Neural_Networks.pdf
└── README.md
```

## ⚙️ How It Works

The optimizer modifies Adam’s update rule by conditioning the second-moment term on the **element-wise variance** of per-sample PDE gradients:

```math
w_{t+1} = w_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
```

Where:

* $m_t$: Batch-averaged PDE gradients
* $v_t$: Element-wise variance of per-sample PDE gradients
* $\eta$: Learning rate

This formulation allows the optimizer to adaptively shrink learning rates in stiff regions and expand them where the solution is smoother.

## 🧪 Getting Started

### Prerequisites

* Python 3.8+
* JAX with GPU support (recommended)
* NumPy, Matplotlib

### Run Training Examples

```bash
python main.py --pde burgers --model basic
python main.py --pde allen_cahn --model basic
python main.py --pde kdv --model basic
```

### After Training

Error heatmaps will be saved in the `figures/` directory as:

```
figures/<pde>_<optimizer>_error.png
```
Check the pdf titled "PDE_aware_Optimizer_for_Physics_informed_Neural_Networks" for results.

## 🧠 Authors

* [Hardik Shukla](https://github.com/hardikshukla7)
* [Manurag Khullar](https://github.com/manuragkhullar)
* [Vismay Churiwala](https://github.com/vismaychuriwala)

---
