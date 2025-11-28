# Learning-to-Optimize for Battery Energy Storage System Scheduling

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/L2O_BESS_Scheduling.ipynb](https://colab.research.google.com/drive/1wdD9hdUni19CpRHC5MTEsoZglk6AWMnA#scrollTo=YYvTBPLp-igB))

A **self-supervised Learning-to-Optimize (L2O)** framework for day-ahead scheduling of residential battery energy storage systems (BESS) coupled with solar PV under net metering.

## 📋 Overview

This project implements a neural network policy that learns to schedule battery charging and discharging decisions by directly optimizing the economic objective, without requiring pre-solved optimization labels. The L2O policy achieves near-optimal solutions with **orders of magnitude speedup** compared to traditional mixed-integer programming (MIP) solvers.

### Key Features

- **Constraint satisfaction by construction**
- **Self-supervised training**
- **Near-optimal performance**
- **Real-time capable**
- **Fully differentiable**

## 🏗️ Problem Formulation

The optimization problem minimizes daily electricity costs for a residential prosumer:

$$\min \sum_{t=0}^{T-1} \left( p_t^{\text{import}} \cdot \pi_t^{\text{buy}} - p_t^{\text{export}} \cdot \pi_t^{\text{sell}} \right)$$

Subject to:
- **Mode exclusivity**: Battery operates in exactly one mode (charge/discharge/idle) per timestep
- **Power limits**: Charge and discharge bounded by power rating
- **SOC dynamics**: Energy balance with efficiency losses
- **SOC bounds**: State of charge within $[0, E_{\text{cap}}]$
- **Terminal condition**: Cyclic operation ($\text{SOC}_T = \text{SOC}_0$)
- **Power balance**: Grid import/export balances demand, solar, and battery

### System Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $T$ | 24 | Planning horizon (hours) |
| $E_{\text{cap}}$ | 10 kWh | Battery capacity |
| $P_{\max}$ | 3 kW | Max charge/discharge power |
| $\eta$ | 0.95 | One-way efficiency |
| $\alpha$ | 0.3 | Feed-in tariff ratio |

## 🧠 Model Architecture

```
Input (96-dim)                    # 4 features × 24 hours
    │
    ▼
┌─────────────────────────────┐
│   Shared Encoder            │
│   Linear(96→768) + LN + ReLU│
│   Linear(768→768) + LN + ReLU│
└─────────────────────────────┘
    │
    ├──────────────────┐
    ▼                  ▼
┌─────────────┐  ┌─────────────┐
│  Mode Head  │  │ Power Head  │
│  (Gumbel-   │  │ (Sigmoid    │
│  Softmax)   │  │  Scaling)   │
└─────────────┘  └─────────────┘
    │                  │
    └────────┬─────────┘
             ▼
    ┌─────────────────┐
    │ SOC Unrolling   │
    │ (Deterministic) │
    └─────────────────┘
             │
             ▼
      Output (Schedule)
```

### Constraint Handling

| Constraint | Mechanism |
|------------|-----------|
| Mode exclusivity | Gumbel-Softmax with temperature annealing |
| Power limits | Sigmoid activation × $P_{\max}$ |
| SOC dynamics | Deterministic unrolling (exact satisfaction) |
| SOC bounds | ReLU penalty in loss function |
| Terminal condition | SmoothL1 penalty in loss function |
| Power balance | ReLU rectification (exact satisfaction) |

## Training Strategy

1. **Bias initialization**: Mode head bias favors charge/discharge to avoid "idle trap"
2. **Stochastic perturbations**: 5% Gaussian noise on inputs for robustness
3. **Temperature annealing**: Gumbel-Softmax τ: 5.0 → 0.1 over training
4. **OneCycleLR**: Learning rate from 1e-4 → 1e-3 → 1e-5

### Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{econ}} + \lambda_{\text{soc}} \mathcal{L}_{\text{soc}} + \lambda_{\text{term}} \mathcal{L}_{\text{term}}$$

| Term | Weight | Description |
|------|--------|-------------|
| $\mathcal{L}_{\text{econ}}$ | 1.0 | Net electricity cost |
| $\mathcal{L}_{\text{soc}}$ | 20.0 | SOC bound violations (ReLU) |
| $\mathcal{L}_{\text{term}}$ | 10.0 | Terminal SOC mismatch (SmoothL1) |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


