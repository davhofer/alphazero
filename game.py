"""Abstract base classes for games and moves."""

import torch
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class Move(ABC):
    """Abstract base class for game moves."""
    
    @abstractmethod
    def encode(self) -> int:
        """Maps each move to a unique integer in [0, GameState.num_possible_moves())."""
        pass

    @classmethod
    @abstractmethod
    def decode(cls, encoded: int) -> "Move":
        """Decodes an integer back to a Move object."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation of the move."""
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        """Equality comparison for moves."""
        pass


class GameState(ABC):
    """Abstract base class for game states."""
    
    @classmethod
    @abstractmethod
    def num_possible_moves(cls) -> int:
        """The total number of possible moves in this game."""
        pass

    @classmethod
    @abstractmethod
    def encoded_shape(cls) -> Tuple[int, int, int]:
        """Returns (channels, height, width) for neural network input."""
        pass

    @classmethod
    @abstractmethod
    def initial_state(cls) -> "GameState":
        """Returns the initial game state."""
        pass

    @abstractmethod
    def get_legal_moves(self) -> List[Move]:
        """Returns list of legal moves from this state."""
        pass

    @abstractmethod
    def apply_move(self, move: Move) -> "GameState":
        """Returns new state after applying the move."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Returns True if game is over."""
        pass

    @abstractmethod
    def encode(self) -> torch.Tensor:
        """Encodes state as tensor for neural network."""
        pass

    @abstractmethod
    def get_value(self) -> Optional[float]:
        """If terminal state, returns 1, 0 or -1. Otherwise returns None."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation of the game state."""
        pass
