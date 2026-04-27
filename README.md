# ioptics

`ioptics` is a PyTorch-based framework for integrated optical neural networks.
It provides optical components, mesh/array layers, differentiators, and a hardware-aware simulator.

## Installation

From source:

```bash
pip install .
```

For development (editable install + dev dependencies):

```bash
pip install -e ".[dev]"
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
import ioptics as iop

# Build a small optical model
model = iop.OpticalNN(
    iop.ClementsMesh(4),
    iop.AbsSquared(),
)

x = torch.randn(8, 4, dtype=torch.complex64)
y = model(x)
print("Model output shape:", y.shape)

# Optional: run hardware-aware simulation
sim = iop.simulator(profile=iop.SimulatorProfile.LAB.value)
y_sim = sim.run(
    x,
    model,
    mode=iop.SimulationMode.STOCHASTIC.value,
    seed=42,
    model_output_domain=iop.ModelOutputDomain.POWER.value,
)
print("Sim output shape:", y_sim.shape)
```

## Run Tests

```bash
pytest tests/
```

## Repository Layout

- `ioptics/` — core package (components, layers, models, differentiators, simulator)
- `tests/` — unit and regression tests
- `examples/` — notebooks demonstrating typical workflows
