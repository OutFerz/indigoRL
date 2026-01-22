# IndigoRL - Pokémon Yellow Deep Reinforcement Learning 🧠🎮

<p align="center">
  <img src="assets/banner.png" width="100%" alt="IndigoRL Banner" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Development-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PyBoy-2.0-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/RL-Recurrent%20PPO-orange?style=for-the-badge" />
  <img src="https://img.shields.io/github/stars/OutFerz/indigoRL?style=for-the-badge" />
</p>

<p align="center">
  <strong>Neuro-Symbolic Vision + RAM Reinforcement Learning Agent</strong><br>
  Autonomous completion of Pokémon Yellow using long-term memory (LSTM) and direct memory access.
</p>

<p align="center">
  <img src="assets/demo.gif" width="600" alt="Agent Gameplay Demo" />
</p>

---

## 📚 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Monitoring & Metrics](#-monitoring--metrics)
- [Agent Architecture](#-agent-architecture)
- [Project Structure](#-project-structure)
- [Credits](#-credits)
- [Disclaimer](#-disclaimer)

---

## 🎯 Project Overview

**IndigoRL** is an autonomous Artificial Intelligence agent designed to complete *Pokémon Yellow* using **Deep Reinforcement Learning**.

Unlike generic agents that randomly press buttons, IndigoRL implements a **Neuro-Symbolic Architecture** combining:

- 🖼️ **Computer Vision:** CNN processing over resized game frames.
- 🧠 **Symbolic State:** Direct RAM memory inspection (event flags, battle state, map data).
- 🔁 **Long-Term Memory:** Recurrent Neural Networks (LSTM via Recurrent PPO).

This allows the agent to reason about **story progression, battles, and exploration** in an extremely sparse, long-horizon RPG environment.

---

## ✨ Key Features

### 🧠 LSTM Brain (Long-Term Memory)
- Uses `RecurrentPPO` (PPO + LSTM).
- Enables maze navigation, backtracking, and objective persistence.
- Solves the "memoryless" limitation of standard RL agents.

### 🧩 Neuro-Symbolic Reward System
- **Story Progress**
  - Reads event flags directly from game RAM.
  - Rewards badges, key items, and narrative milestones.
- **Battle Awareness**
  - Reads enemy HP, player HP, and battle states.
  - Learns combat strategies instead of brute-force button mashing.
- **Exploration**
  - Rewards discovering new Map IDs.
  - Penalizes stagnation and looping behavior.

### ⚡ Extreme Efficiency
- **State Loading**
  - Skips Oak’s intro using a clean save-state.
  - ~20% reduction in compute per episode.
- **Headless Training**
  - SDL disabled during training for maximum FPS.
- **Parallel Training**
  - Supports multiple emulator instances.

---

## 🛠️ Technology Stack

| Component | Technology |
|---------|-----------|
| Language | Python 3.10+ |
| RL Algorithm | Stable-Baselines3 Contrib (Recurrent PPO) |
| Emulator | PyBoy 2.0+ |
| Vision | OpenCV, NumPy, Scikit-Image |
| Logging | TensorBoard |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+ (Conda recommended)
- Pokémon Yellow ROM (`.gb`) — **legally owned**

### Setup

```bash
git clone https://github.com/OutFerz/indigoRL.git
cd indigoRL
conda create -n indigoRL python=3.10
conda activate indigoRL
pip install -r requirements.txt
```

### ROM

Place your ROM at:

```
roms/PokemonYellow.gb
```

---

## 🕹️ Usage

### 1️⃣ Generate Initial Save State (Optional)

```bash
python record_state.py
```

Play the intro manually and close the window once you gain control of the player.

---

### 2️⃣ Train the Agent

```bash
python train_lstm.py
```

Models and logs are saved to:

```
experiments/poke_lstm_v1/
```

---

### 3️⃣ Watch the Agent Play

```bash
python play.py
```

- Real-time 60 FPS playback
- Neural network input overlay
- Live RAM debugging info

---

## 📈 Monitoring & Metrics

Monitor training in real time using TensorBoard:

```bash
tensorboard --logdir experiments/poke_lstm_v1/logs
```

Open your browser at:

```
http://localhost:6006
```

---

## 🧠 Agent Architecture

**Policy:** Multi-Input Recurrent Policy

- **Visual Encoder (CNN)**
  - Grayscale, downsampled game frames
- **Symbolic Encoder (MLP)**
  - RAM vector:
    - X, Y, Map ID
    - Player HP, Enemy HP
    - Party Levels
    - In-Battle Flag
- **Memory Core**
  - LSTM (256 units)
- **Action Head**
  - Discrete GameBoy button actions

---

## 📂 Project Structure

```
indigoRL/
├── assets/                 # README images
├── experiments/            # Models and logs
├── roms/                   # Game ROMs
├── src/
│   └── environment/
│       └── pokemon_env.py  # Gym environment & RAM reader
├── states/                 # Save states
├── train_lstm.py           # Training entry point
├── play.py                 # Visualization script
├── record_state.py         # Save-state utility
└── requirements.txt
```

---

## 🤝 Credits

- PyBoy Emulator
- Stable-Baselines3 Contrib
- pret/pokeyellow disassembly project

---

## 📜 Disclaimer

This project is for **research and educational purposes only**.  
You must legally own a physical or digital copy of *Pokémon Yellow* to use the ROM.  
The authors do not encourage or support piracy.