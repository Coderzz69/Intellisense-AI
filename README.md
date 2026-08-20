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

-   **`traffic_rl/`**: The core AI engine. Features a Hierarchical MARL system (DQN) trained on SUMO environments.
-   **`Yolo11n_Implementation/`**: An interactive React-based dashboard that visualizes traffic telemetry and performs YOLO-based video analysis.

---

## 📊 Metrics & Performance

The AI controller significantly outperforms traditional fixed-timer baselines across all key metrics.

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

```powershell
cd traffic_rl
pip install -r requirements.txt

# Run the comparison demo (Baseline vs AI)
python demo.py --mode compare --headless

# Open the AI Control Dashboard
python gui.py
```

### 2. YOLO Dashboard (Frontend & Analysis)
Requires [Node.js](https://nodejs.org/) and a Mapbox token.

```powershell
cd Yolo11n_Implementation
npm install
npm run start:backend  # Start analysis feed
npm start              # Open React dashboard
```

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
