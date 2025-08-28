# AlphaZero Implementation

A fully functional yet simple implementation of the [AlphaZero algorithm by Silver et al., 2017](https://www.nature.com/articles/nature24270). The focus lies on clarity and understanding of the core algorithm and the implementation doesn't contain any performance optimizations.

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

### Missing features

This package is mainly for (self-)educational purposes and does not implement all features required for optimal efficiency and performance.

Missing:

- experience replay buffer
- parallelization for data generation (self-play)
- data augmentation (e.g. using board symmetries)
- various numerical stability & training optimization measures
- model pass/resignation capabilities

## Installation

### As a Package

You can install this package directly from GitHub using pip or uv:

```bash
# Using pip
pip install git+https://github.com/davhofer/alphazero.git

# Using uv
uv pip install git+https://github.com/davhofer/alphazero.git
```

### For Development

If you want to develop or modify the code:

```bash
# Clone the repository
git clone https://github.com/yourusername/alphazero.git
cd alphazero

# Install dependencies
uv sync
```

## Usage

### Training

Train an AlphaZero agent on a game:

```bash
uv run python -m alphazero train --game tictactoe

# Customize training parameters
uv run python -m alphazero train --game chess --iterations 200 --self-play-rounds 50
```

Key training parameters:

- `--game`: Game to train on. Any of the games listed in `games/` (Currently tictactoe, connect_four, chess)
- `--iterations`: Number of training iterations (default: 100)
- `--self-play-rounds`: Self-play games per iteration (default: 100)
- `--mcts-time-limit`: MCTS thinking time in seconds (default: 0.5)
- `--epochs`: Neural network training epochs per iteration (default: 10)

To see all configuration options:

```bash
uv run python -m alphazero train --help
```

### Evaluation

Evaluate trained models or pit different players against each other:

```bash
# Evaluate a trained network against random player
uv run python -m alphazero eval --game tictactoe --player1 network-mcts --player1-model checkpoints/best_model_tictactoe.pt --player2 random --games 100

# Human vs trained network
uv run python -m alphazero eval --game connect_four --player1 human --player2 network-mcts --player2-model checkpoints/best_model_connect_four.pt

# See all available options
uv run python -m alphazero eval --help
```

Player types:

- `random`: Random moves
- `greedy`: Simple greedy strategy (NOT IMPLEMENTED)
- `human`: Human player (interactive)
- `network-mcts`: Trained network with MCTS
- `network-direct`: Trained network direct policy (no MCTS)

### Plotting

Generate plots from training logs:

```bash
# Generate all plots from training log
uv run python -m alphazero plot logs/training_log.json

# See all available options
uv run python -m alphazero plot --help
```

## Results

TBD
