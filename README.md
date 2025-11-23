# Doodle Jump RL

A reinforcement learning project that trains a Double Deep Q-Network (DQN) agent to play Doodle Jump using Pygame and PyTorch.

## Overview

This project implements a DQN-based agent that learns to play Doodle Jump through trial and error. The game features platforms, obstacles, and boost platforms, with the agent learning to navigate upward while avoiding obstacles.

## Features

- **Deep Q-Network (DQN)** with Double DQN algorithm
- **Experience Replay** buffer for stable training
- **Reward Shaping** with obstacle avoidance and platform approach rewards
- **Toroidal Boundary Handling** using periodic encoding for seamless screen wrapping
- **Model Checkpointing** with automatic best model saving
- **Visual Testing** interface to evaluate trained models

## Requirements

- Python 3.7+
- pygame >= 2.0.0
- numpy >= 1.21.0
- torch >= 1.9.0
- gymnasium >= 0.28.0
- matplotlib >= 3.5.0

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a new DQN agent:
```bash
python train_dqn.py
```

Continue training from a checkpoint:
```bash
python train_dqn_continue.py
```

### Testing

Run the main script to choose between manual play or model testing:
```bash
python main.py
```

Options:
- **1**: Manual play mode (use arrow keys to control)
- **2**: Test trained DQN model (select from saved models in `models/`)

## Project Structure

- `game.py` - Core game logic and physics
- `rl_env.py` - Gymnasium environment wrapper
- `dqn_model.py` - DQN agent implementation
- `train_dqn.py` - Training script
- `main.py` - Main entry point
- `settings.py` - Game configuration
- `sprites.py` - Game sprites (Player, Platform)
- `assets/` - Game assets (images, sounds)
- `models/` - Saved model checkpoints

## Training Details

- **State Space**: 27-dimensional observation (player state, nearest obstacle, nearest platform, 5 platforms above)
- **Action Space**: 3 discrete actions (left, no-op, right)
- **Network Architecture**: 2 hidden layers (128-128) with Batch Normalization
- **Training**: 600 episodes with automatic checkpointing every 100 episodes
