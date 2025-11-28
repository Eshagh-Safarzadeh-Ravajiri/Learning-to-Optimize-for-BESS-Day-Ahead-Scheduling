# Learning-to-Optimize for Battery Energy Storage System Scheduling

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/L2O_BESS_Scheduling.ipynb)

A **self-supervised Learning-to-Optimize (L2O)** framework for day-ahead scheduling of residential battery energy storage systems (BESS) coupled with solar PV under net metering.

## 📋 Overview

This project implements a neural network policy that learns to schedule battery charging and discharging decisions by directly optimizing the economic objective, without requiring pre-solved optimization labels. The L2O policy achieves near-optimal solutions with **orders of magnitude speedup** compared to traditional mixed-integer programming (MIP) solvers.

### Key Features

- **Constraint satisfaction by construction**: Mode exclusivity (Gumbel-Softmax), power limits (Sigmoid scaling), SOC dynamics (deterministic unrolling), power balance (ReLU rectification)
- **Self-supervised training**: No need for pre-computed MIP solutions as labels
- **Near-optimal performance**: Typically <5% optimality gap relative to MIP solver
- **Real-time capable**: 1000x+ speedup over MIP solvers
- **Fully differentiable**: End-to-end trainable with standard deep learning frameworks

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

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CVXPY (for MIP baseline)
- NumPy, Matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/L2O-BESS-Scheduling.git
cd L2O-BESS-Scheduling

# Install dependencies
pip install torch numpy matplotlib cvxpy

# Optional: Install CBC solver for faster MIP solutions
pip install cylp
```

### Quick Start (Google Colab)

Click the "Open in Colab" badge above, or:

1. Upload `L2O_BESS_Scheduling.ipynb` to Google Colab
2. Run all cells sequentially
3. Training takes ~10-20 minutes on CPU, ~5 minutes on GPU

## 🚀 Usage

### Training

```python
from l2o_bess import BatterySystemParams, L2O, generate_training_data_netmeter, train_l2o_model

# Initialize parameters
params = BatterySystemParams()

# Generate training data (5000 synthetic daily scenarios)
train_data = generate_training_data_netmeter(num_samples=5000)

# Create and train model
model = L2O(params)
train_l2o_model(model, train_data, params, n_epochs=1500)
```

### Inference

```python
# Generate test scenario
test_data = generate_training_data_netmeter(num_samples=1)
solar, demand, price_buy, price_sell = test_data

# Get L2O schedule
model.eval()
with torch.no_grad():
    schedule = model.forward(solar, demand, price_buy, price_sell)

# Access results
power_charge = schedule['power_charge']      # (1, 24)
power_discharge = schedule['power_discharge'] # (1, 24)
soc = schedule['soc']                         # (1, 24)
mode = schedule['mode']                       # (1, 24, 3)
```

### Comparison with MIP

```python
from l2o_bess import MIPSolverNetMetering, compare_methods

# Compare on test set
test_data = generate_training_data_netmeter(num_samples=50)
results = compare_methods(model, test_data, params, num_instances=50)

# Results include:
# - results['cost_gaps']: Optimality gap (%) for each instance
# - results['mip_time']: Total MIP solve time
# - results['l2o_time']: Total L2O inference time
```

## 📊 Results

### Performance Summary

| Metric | Value |
|--------|-------|
| Average optimality gap | < 5% |
| Feasibility rate | > 95% |
| Speedup vs MIP | > 1000× |
| Training time | ~15 min (CPU) |

### Sample Output

```
============================================================
COMPARING L2O vs MIP ON 50 TEST INSTANCES
============================================================

Cost Comparison:
  MIP Average Cost:     $1.23
  L2O Average Cost:     $1.28
  Average Gap:          3.42%
  Median Gap:           2.15%

Solve Time:
  MIP Total Time:       45.32s
  L2O Total Time:       0.0312s
  Speedup:              1453x

Feasibility:
  L2O Feasible:         48/50 (96.0%)
```

## 📁 Project Structure

```
L2O-BESS-Scheduling/
├── L2O_BESS_Scheduling.ipynb   # Main notebook (Colab-ready)
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
└── figures/                     # Generated figures
    ├── figure1_soc_power.png
    ├── figure2_inputs.png
    ├── figure3_gap_distribution.png
    ├── figure4_solve_time.png
    └── figure5_cost_scatter.png
```

## 🔬 Methodology

### Training Strategy

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

## 📚 References

- Donti, P., et al. (2017). "Task-based End-to-end Model Learning in Stochastic Optimization." NeurIPS.
- Bengio, Y., et al. (2021). "Machine Learning for Combinatorial Optimization: A Methodological Tour d'Horizon." European Journal of Operational Research.
- Jang, E., et al. (2017). "Categorical Reparameterization with Gumbel-Softmax." ICLR.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**If you find this work useful, please consider giving it a ⭐!**
