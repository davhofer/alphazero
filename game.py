"""Game and move wrapper classes."""

import torch


class Move:
    def encode(self) -> int:
        """Maps each move to a unique integer in [0, GameState.num_possible_moves())."""
        pass

    @classmethod
    def decode(cls, encoded: int) -> "Move":
        pass


class GameState:
    @classmethod
    def num_possible_moves(cls) -> int:
        """The total number of possible moves in this game."""
        pass

    @classmethod
    def initial_state(cls) -> "GameState":
        pass

    def get_legal_moves(self) -> list[Move]:
        pass

    def apply_move(self, move: Move) -> "GameState":
        pass

    def is_terminal(self) -> bool:
        pass

    def encode(self) -> torch.Tensor:
        pass

    def get_value(self) -> float | None:
        """If terminal state, returns 1, 0 or -1. Otherwise returns None."""
        pass
