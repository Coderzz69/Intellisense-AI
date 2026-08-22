# 🚦 IntelliSignal-AI — AI Traffic Management System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![SUMO](https://img.shields.io/badge/SUMO-1.18.0+-orange.svg)](https://sumo.dlr.de/docs/index.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **Decentralized Multi-Agent Reinforcement Learning and Real-time Analysis for Indian Urban Traffic Control.**

---

## 📖 Overview

This repository contains a comprehensive suite for intelligent traffic management, developed for the **SSS Hackathon**. It combines high-fidelity traffic simulation using **SUMO** and **Reinforcement Learning** with a real-time **YOLO-based analysis dashboard**.

### Core Objectives
1.  **Reduce Congestion**: Replace fixed-timer signals with AI policies that adapt to real-time demand.
2.  **Emergency Priority**: Integrated "Global Green Corridor" to ensure zero-wait for ambulances.
3.  **Realistic Simulation**: Modeled after Indian urban traffic with lane weaving and mixed vehicle classes.
4.  **Actionable Insights**: Real-time visualization of traffic density and vehicle distributions.

---

## 🏗️ Repository Structure

-   **`Yolo11n_Implementation/`**: The vision stack — YOLO detection, lane tracking, and a trained DQN that picks the signal phase — behind a React dashboard and a Flask four-feed UI. **This runs today.**
-   **`traffic_rl/`**: The SUMO training environment for the hierarchical MARL system. **Currently incomplete** — see [`traffic_rl/MISSING_FILES.md`](traffic_rl/MISSING_FILES.md).

> ### ⚠️ Status of `traffic_rl/`
>
> `traffic_rl/env/traffic_env.py` and the SUMO checkpoints are absent from this
> repository. A boilerplate `env/` pattern in `traffic_rl/.gitignore` — meant for
> virtualenvs — matched the project's own `env/` **source** package, so it was
> never committed. Every entry point there (`main.py`, `demo.py`, `simulate.py`,
> `scripts/test_env.py`) imports it and therefore cannot run.
>
> The ignore patterns are now fixed, so restoring the directory is enough. The
> metrics below were produced by that environment and cannot be reproduced from
> this repository until it is restored.
>
> The DQN used by the vision pipeline (`Yolo11n_Implementation/traffic/weights/rl_latest.pt`)
> is a **different, 16-dimensional** model. It is present and working.

---

## 📊 Metrics & Performance

The AI controller significantly outperforms traditional fixed-timer baselines across all key metrics.

> Reported from the original SUMO training runs. Not reproducible from this
> repository until `traffic_rl/env/` is restored.

### Comparison: Fixed-Timer Baseline vs. Hierarchical Master AI
| Metric | Fixed-Timer Baseline | Master AI | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Wait Time (s)** | 93,563 | 18,914 | **↓ 79.8%** |
| **Avg Wait Per Vehicle (s)** | 14.7 | 6.2 | **↓ 57.7%** |
| **Avg Emergency Wait (s)** | Context-dependent | ~0 (corridor created) | **↓ ~100%** |
| **Throughput (Vehicles Arrived)** | Baseline | High | **↑ Optimized** |

### Training Progress
Below are the training curves showing the reduction in waiting time and the convergence of the RL policy over 110 episodes.

![Training Curves](traffic_rl/plots/training_curves.png)

---

## 🚀 Quick Start

### 1. Traffic RL (Simulation & AI)
Requires [SUMO](https://sumo.dlr.de/docs/Downloads.php) installed and `SUMO_HOME` environment variable set.

Blocked until `traffic_rl/env/` is restored — see
[`traffic_rl/MISSING_FILES.md`](traffic_rl/MISSING_FILES.md). Once it is:

```bash
cd traffic_rl
pip install -r requirements.txt

# Run the comparison demo (Baseline vs AI)
python demo.py --mode compare --headless

# Open the AI Control Dashboard
python gui.py
```

### 2. Vision Pipeline + Dashboard
Requires Python 3.10+, [Node.js](https://nodejs.org/) 18+, and a Mapbox token.

```bash
cd Yolo11n_Implementation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
npm install
cp .env.example .env                                 # set your Mapbox token

python scripts/make_sample_video.py                  # or set TRAFFIC_DEFAULT_VIDEO_PATH
npm run start:backend                                # analysis feed on :8000
npm start                                            # dashboard on :3000
```

Verify the AI actually loaded:

```bash
curl -s http://127.0.0.1:8000/api/config | python -m json.tool   # see the "ai" block
npm run test:backend
```

Details in [`Yolo11n_Implementation/README.md`](Yolo11n_Implementation/README.md).

---

## 🛠️ Technical Highlights

-   **Hierarchical Control**: A **Master Controller** handles global crises (Emergency/Congestion) while **Local RL Agents** optimize per-junction throughput.
-   **Shared Weight Generalization**: A single DQN model manages multiple intersections, generalized for both 3-way and 4-way junctions.
-   **Indian Traffic Patterns**: Custom SUMO configuration includes erratic braking, lane weaving, and varied vehicle physics.
-   **Hybrid Overrides**: Safety-critical rule-based layers ensure the AI never blocks an emergency vehicle.

---

## 👨‍💻 Developed By

Built as a unified solution for **Society Security Solution (SSS)**.
- **Frameworks**: SUMO · TraCI · PyTorch · YOLOv11 · React
- **Architecture**: Decentralized MARL with Hierarchical Supervision
