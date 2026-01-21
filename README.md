# 🧠 PokeAI — Autonomous Pokémon Yellow Reinforcement Learning Agent

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Development-success" />
  <img src="https://img.shields.io/badge/Python-3.11-blue" />
  <img src="https://img.shields.io/badge/RL-Stable--Baselines3-orange" />
  <img src="https://img.shields.io/badge/Emulator-PyBoy-purple" />
</p>

> **Architecture:** PPO (Proximal Policy Optimization) + Neuro-Symbolic State Decoding  
> **Goal:** Train an Artificial Intelligence agent capable of completing **Pokémon Yellow** from scratch, with *zero prior knowledge* (Tabula Rasa).

---

## 🎯 Project Overview

**PokeAI** is a Deep Reinforcement Learning research project focused on solving **long-horizon RPG environments**. Pokémon Yellow represents a particularly challenging benchmark due to:

- Extremely sparse rewards
- Large state space
- Long-term dependencies (decisions made minutes or hours earlier)
- Partial observability from pixels alone

To overcome these challenges, PokeAI combines **visual perception** with **explicit symbolic game state extraction**, allowing the agent to both *see* and *understand* the game world.

---

## 🧩 Technical Description

This project implements a **Deep Reinforcement Learning (Deep RL)** architecture designed for complex Game Boy–era RPGs.

Unlike purely vision-based agents that rely only on raw pixels, PokeAI uses a **Hybrid Observation Space** composed of:

1. **👁️ Vision (CNN-based)**  
   Screen processing to understand local geometry, obstacles, and transitions.

2. **🧠 Memory (RAM Inspection)**  
   Direct reading of emulator memory to extract global context such as:
   - Player coordinates
   - Current map ID
   - Progress flags (e.g., badges)

This neuro-symbolic approach dramatically improves sample efficiency and stability during training.

---

## ✨ Key Features

- **⚡ Accelerated Emulation**  
  Uses **PyBoy** in headless mode during training, achieving speeds of **1000+ FPS**.

- **👁️ Hybrid Observations**  
  The agent not only sees pixels, but *knows* where it is through RAM-injected state vectors.

- **🗺️ Efficient Exploration**  
  Dense reward shaping based on unique visited coordinates `(x, y)` to mitigate sparse reward issues.

- **🎥 Streamer-Ready Architecture**  
  Asymmetric design allows full-speed training in the background while a cloned instance runs at **60 FPS** for live visualization or streaming.

- **⚙️ Hardware-Aware & Scalable**  
  Default settings are chosen for broad compatibility with consumer hardware.  
  The number of CPU cores and threads used during training can be adjusted via environment variables (e.g. `OMP_NUM_THREADS`) and configuration inside `train.py`, allowing the same codebase to scale from low-end machines to multi-core systems.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|---------|-----------|---------|
| **Language** | Python 3.11 | Core logic |
| **RL Framework** | Stable-Baselines3 | PPO implementation & vectorized environments |
| **Emulator** | PyBoy | Low-level Game Boy emulation |
| **Vision** | OpenCV, NumPy | Frame preprocessing & rendering |
| **Logging** | TensorBoard | Real-time metrics (reward, loss, entropy) |

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.11** (Conda recommended)
- **Pokémon Yellow ROM**  
  Must be named exactly `PokemonYellow.gb` and placed inside the `roms/` directory.

### Step-by-Step Guide

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/OutFerz/PokeAI.git
cd PokeAI
```

#### 2️⃣ Create a virtual environment

```bash
conda create -n pokeai python=3.11
conda activate pokeai
```

#### 3️⃣ Install dependencies

```bash
pip install gymnasium pyboy shimmy stable-baselines3[extra] opencv-python torch-directml
```

#### 4️⃣ Generate the initial save state (Skip Intro)

To prevent the agent from wasting hours navigating menus, create a save state immediately after the intro sequence.

```bash
python src/utils/create_initial_state.py
```

> **Instruction:** Manually play until you gain control of the character in Ash’s room, then close the window.

---

## 🏃 Workflow & Execution

PokeAI is designed to run in **two terminals simultaneously**:

- **🧠 Brain:** High-speed training
- **👀 Eyes:** Real-time visualization

---

### 🧠 1. Training (The Brain)

Runs the PPO training loop in headless mode for maximum performance.

- **CPU / Multithreading Configurable:**  
  Training parallelism is not fixed. You can scale CPU usage by modifying `train.py` and/or environment variables such as `OMP_NUM_THREADS`, enabling efficient execution on both low-end and high-core systems.
- **Checkpoints:** Automatically saved to `experiments/`

```bash
python train.py
```

> Press **Ctrl + C** at any time to trigger a safe emergency checkpoint.

---

### 👀 2. Visualization (The Eyes)

Displays the agent playing at **60 FPS**.

- Automatically detects improved models
- Hot-reloads new checkpoints without restarting

```bash
python watch_continuous.py
```

---

### 📊 3. Monitoring (Analytics)

Visualize reward curves, loss, and entropy in real time:

```bash
tensorboard --logdir experiments/poke_ppo_v1/logs
```

---

## 🧠 Agent Architecture

### Action Space

**Discrete (6 actions):**

```
[DOWN, LEFT, RIGHT, UP, A, B]
```

> `START` and `SELECT` are intentionally disabled to reduce stochastic noise and avoid menu-locking behaviors.

---

### Reward Shaping

The current reward function emphasizes **pure exploration**:

R_t = R_exploration + R_events

- **Exploration Reward:** +1.0 for each unique `(x, y)` coordinate visited per map
- **Inactivity Penalty (Implicit):** No reward for standing still forces movement through optimization pressure

---

## 📂 Project Structure

```
PokeAI/
├── config/                  # Hyperparameters & configs
├── experiments/             # PPO checkpoints & TensorBoard logs
├── roms/                    # Game ROMs (.gb)
├── states/                  # PyBoy save states (.state)
├── src/
│   ├── environment/
│   │   ├── pokemon_env.py   # Gym wrapper (RAM, vision, smooth ticking)
│   │   └── ...
│   ├── utils/
│   │   ├── memory_reader.py # Hex-level RAM extraction
│   │   └── ...
├── train.py                 # Training backend
├── watch_continuous.py      # Streaming / visualization frontend
└── README.md
```

---

## 🔮 Roadmap

- [ ] Integrate **HippoTorch / S4** for long-term memory
- [ ] Add **Vision-Language Model (VLM)** for on-screen dialogue understanding
- [ ] Badge-aware curriculum learning
- [ ] Multi-objective reward decomposition

---

## 📜 Disclaimer

This project is for **research and educational purposes only**. You must legally own a copy of Pokémon Yellow to use the ROM.

---

⭐ *If you find this project interesting, consider giving it a star!*
