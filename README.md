# AlphaZero Implementation

A simple implementation of the AlphaZero algorithm for learning and understanding the core concepts. This implementation doesn't contain any performance optimizations.

## Overview

This project implements AlphaZero from scratch to support multiple games including:

- Tic-tac-toe
- Connect Four
- Chess

The implementation includes:

- Monte Carlo Tree Search (MCTS)
- Neural network for position evaluation and move prediction
- Self-play training loop
- Evaluation with various players

## Installation

We use `uv` for dependency management. To install:

```bash
# Install dependencies
uv sync
```

(However, directly running the code as described below also takes care of this.)

## Usage

### Training

See all configuration options:

```bash
uv run python training.py --help
```

Train an AlphaZero agent on a game:

```bash

# Train on tictactoe
uv run python training.py --game tictactoe

# Customize training parameters
uv run python training.py --game chess --iterations 200 --self-play-rounds 50
```

Key training parameters:

- `--game`: Game to train on. Any of the games listed in `games/` (Currently tictactoe, connect_four, chess)
- `--iterations`: Number of training iterations (default: 100)
- `--self-play-rounds`: Self-play games per iteration (default: 100)
- `--mcts-time-limit`: MCTS thinking time in seconds (default: 0.5)
- `--epochs`: Neural network training epochs per iteration (default: 10)

### Evaluation

Evaluate trained models or pit different players against each other:

```bash
# Evaluate a trained network against random player
uv run python evaluation.py --game tictactoe --player1 network-mcts --player1-model checkpoints/best_model_tictactoe.pt --player2 random --games 100

# Human vs trained network
uv run python evaluation.py --game connect_four --player1 human --player2 network-mcts --player2-model checkpoints/best_model_connect_four.pt

# See all available options
uv run python evaluation.py --help
```

Player types:

- `random`: Random moves
- `greedy`: Simple greedy strategy
- `human`: Human player (interactive)
- `network-mcts`: Trained network with MCTS
- `network-direct`: Trained network direct policy (no MCTS)
