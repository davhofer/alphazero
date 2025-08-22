"""AlphaZero - A Python implementation of the AlphaZero algorithm."""

__version__ = "0.1.0"

# Core modules are available for import
# Example usage:
#   from alphazero import mcts, network, players, training
#   from alphazero.games import tictactoe
from alphazero.training import training_loop, TrainingConfig
from alphazero.players import (
    RandomPlayer,
    NetworkMCTSPlayer,
    NetworkDirectPlayer,
    HumanPlayer,
    GreedyPlayer,
)
from alphazero.games import load_game_module
from alphazero.evaluation import play_games, play_single_game
from alphazero.plotting import create_all_plots


__all__ = [
    "training_loop",
    "TrainingConfig",
    "RandomPlayer",
    "NetworkDirectPlayer",
    "NetworkMCTSPlayer",
    "HumanPlayer",
    "GreedyPlayer",
    "load_game_module",
    "play_games",
    "play_single_game",
    "create_all_plots",
]
