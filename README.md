# 🪰 NeuroFly: MaleCNS Arena

> **A biological neural network benchmark & simulation playground** powered by the **MaleCNS (Drosophila connectome)** as a modular "brain runtime", evaluated across multiple environments (**Survival Arena**, **Beer Toast**, and more).

---

## 🌟 Vision & Architecture

Instead of a single-purpose gimmick ("beer toast fly brain only"), **NeuroFly** provides a modular biological brain engine. The biological connectome acts as a plug-and-play **Brain Runtime**, decoupled from sensory encoders, motor decoders, and physics worlds.

```
                    ┌────────────────────────────────────────┐
                    │             Environment                │
                    │  (Survival Arena / Beer Toast / etc.)  │
                    └──────┬─────────────────────────▲───────┘
                           │                         │
                   Sensory Stimuli              Motor Actions
                   (Photons, Odors,            (Thrust, Turn,
                    Touch, Threat)                 Feed)
                           │                         │
                           ▼                         │
                    ┌──────────────┐         ┌───────┴───────┐
                    │   Sensory    │         │     Motor     │
                    │   Encoders   │         │   Decoders    │
                    │ (Ommatidia,  │         │  (Descending  │
                    │  Antennae,   │         │    Neurons    │
                    │ Mechanorecep)│         │     [DNs])    │
                    └──────┬───────┘         └───────▲───────┘
                           │ Current (I_inj)         │ Spikes / Rates
                           ▼                         │
             ┌───────────────────────────────────────────────┐
             │              NeuroFly Brain Runtime            │
             │  MaleCNS Connectome (Sparse Synaptic Graph)   │
             │  Neuron Dynamics: Leaky Integrate-and-Fire     │
             │  PyTorch Sparse Matrix Tensor Engine (CPU/GPU) │
             └───────────────────────────────────────────────┘
```

---

## 🎮 Features

1. **Biological Brain Runtime (`MaleCNS`)**:
   - Vectorized **Leaky Integrate-and-Fire (LIF)** spiking dynamics accelerated by PyTorch sparse tensors.
   - Anatomically mapped Drosophila brain neuropils:
     - **Optic Lobes (Left/Right)**: Retinotopic ommatidia array, Lobula Plate Tangential Cells (LPTCs).
     - **Antennal Lobes (Left/Right)**: Olfactory receptor neurons (ORNs) and projection neurons (PNs).
     - **Central Complex (CX)**: Navigation compass (Ellipsoid Body, Protocerebral Bridge, Fan-shaped Body).
     - **Mushroom Body (MB)**: Associative learning and odor representation.
     - **Descending Neurons (DNs)**: Motor command drivers (thrust, steering torque, feeding).
2. **Survival Arena**:
   - Dynamic 2D/3D physics arena with food patches (odor plumes), obstacles, lighting zones, and moving looming predators.
   - Metabolic energy budget ($E$ depletes with movement, replenished by consuming food).
3. **Controller Benchmark Suite**:
   - Standardized evaluation across 4 distinct controllers:
     - **Controller A**: Random Walk (Brownian motion baseline)
     - **Controller B**: Tiny Neural Network (2-layer MLP reactive baseline)
     - **Controller C**: Heuristic (Optimal rule-based baseline)
     - **Controller D**: **MaleCNS Connectome** (Biological brain runtime)
   - Evaluates: Survival time, distance travelled, food acquired, obstacle collisions, response latency, energy used, and neural firing stats.
4. **Interactive Web Playground**:
   - Modern dark-mode UI with **Three.js 3D Brain Activity Viewer** (1,500+ neurons glowing in real-time).
   - Real-time scrolling **Spike Raster Plot** and **Neuropil Firing Rate Meters** (Hz).
   - Interactive stimulus injection: drop food, spawn predator, flash eyes, or zap brain regions.

---

## 🚀 Quick Start

### 1. Installation

Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/JonusNattapong/biological-agent.git
cd biological-agent
pip install -e .
```

### 2. Launch Interactive Web Playground

```bash
python run_demo.py --web
```
Open **`http://localhost:8000`** in your browser.

- Click anywhere in the Arena to drop food 🍎.
- Press **Spawn Threat** 👾 to send a predator chasing the fly.
- Toggle between **MaleCNS**, **Heuristic**, **Tiny NN**, and **Random** to see how biological wiring behaves compared to traditional controllers.
- Watch neurons spike in real-time in 3D!

### 3. Run Benchmark Suite

Run a rigorous comparative tournament between all controllers:

```bash
python run_demo.py --benchmark --episodes 5 --steps 500
```

Example Benchmark Results:

| Controller | Survival Steps | Food Eaten | Distance | Collisions | Final Energy | Survival % | Mean Reward |
|---|---|---|---|---|---|---|---|
| **Random Walk** | 367.0 +/- 0.0 | 1.0 +/- 0.0 | 1337.1 | 8.0 | 85.2 | 0% | 21.7 |
| **Tiny Neural Network (MLP)** | 127.0 +/- 0.0 | 0.0 +/- 0.0 | 347.2 | 0.0 | 89.6 | 0% | -17.3 |
| **Heuristic (Rule-based)** | 400.0 +/- 0.0 | 0.0 +/- 0.0 | 2449.5 | 127.0 | 46.6 | 100% | 40.0 |
| **MaleCNS Connectome** | 268.0 +/- 0.0 | 0.0 +/- 0.0 | 2083.9 | 17.0 | 58.0 | 0% | -3.2 |

### 4. Run CLI Console Demo

```bash
python run_demo.py --cli --steps 100
```

---

## 🧩 Adding New Environments (Arcade)

NeuroFly is designed so that new environments (e.g. `Beer Toast`, `Driving`, `Fruit Slice`) simply inherit from `BaseEnvironment`:

```python
from neurofly.environments import BaseEnvironment

class BeerToastArena(BaseEnvironment):
    def reset(self):
        # Initialize glass positions and beer stimulus
        ...
    def step(self, action):
        # Update fly reaching kinematics and toast detection
        ...
```

Then hook up sensory encoders to feed visual angles of the beer glass into the Optic Lobes and motor decoders to translate Descending Neuron spikes into leg reaching commands!

---

## 🧪 Running Tests

```bash
pytest -v
```

---

## 📜 License

MIT License
