"""Games package containing implementations of various games for AlphaZero training."""

import importlib

from types import ModuleType
from . import tictactoe
from . import chess
from . import connect_four


def load_game_module(module_name: str) -> ModuleType:
    """Dynamically load a game module."""
    try:
        # Prepend "alphazero.games." to the module name
        full_module_name = f"alphazero.games.{module_name}"
        game_module = importlib.import_module(full_module_name)

        # Validate required classes exist
        required_classes = ["GameState", "Move"]
        for class_name in required_classes:
            if not hasattr(game_module, class_name):
                raise AttributeError(
                    f"Game module '{module_name}' missing required class: {class_name}"
                )

        # Validate required class methods exist
        GameState = getattr(game_module, "GameState")
        Move = getattr(game_module, "Move")

        required_gamestate_methods = [
            "num_possible_moves",
            "encoded_shape",
            "initial_state",
        ]
        for method_name in required_gamestate_methods:
            if not hasattr(GameState, method_name):
                raise AttributeError(
                    f"GameState missing required class method: {method_name}"
                )

        print(f"✅ Loaded and validated game module: {module_name}")
        return game_module

    except ImportError as e:
        raise ImportError(f"Could not import game module '{module_name}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Game module validation failed: {e}")


__all__ = ["tictactoe", "chess", "connect_four", "load_game_module"]

