"""TicTacToe game implementation."""

import torch
import numpy as np
from typing import List, Optional, Tuple
from copy import deepcopy

from .game import Move, GameState


class TicTacToeMove(Move):
    """A move in TicTacToe - placing a mark at position (row, col)."""
    
    def __init__(self, row: int, col: int):
        if not (0 <= row < 3 and 0 <= col < 3):
            raise ValueError(f"Invalid position: ({row}, {col})")
        self.row = row
        self.col = col
    
    def encode(self) -> int:
        """Encode move as integer: row * 3 + col (0-8)."""
        return self.row * 3 + self.col
    
    @classmethod
    def decode(cls, encoded: int) -> "TicTacToeMove":
        """Decode integer back to move."""
        if not (0 <= encoded < 9):
            raise ValueError(f"Invalid encoded move: {encoded}")
        row = encoded // 3
        col = encoded % 3
        return cls(row, col)
    
    def __str__(self) -> str:
        return f"({self.row}, {self.col})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TicTacToeMove):
            return False
        return self.row == other.row and self.col == other.col
    
    def __hash__(self) -> int:
        return hash((self.row, self.col))


class TicTacToeState(GameState):
    """TicTacToe game state."""
    
    def __init__(self, board: Optional[np.ndarray] = None, current_player: int = 1):
        """
        Initialize TicTacToe state.
        
        Args:
            board: 3x3 numpy array. 0=empty, 1=player 1 (X), -1=player 2 (O)
            current_player: 1 for player 1 (X), -1 for player 2 (O)
        """
        if board is None:
            self.board = np.zeros((3, 3), dtype=int)
        else:
            self.board = board.copy()
        self.current_player = current_player
    
    @classmethod
    def num_possible_moves(cls) -> int:
        """TicTacToe has 9 possible positions."""
        return 9
    
    @classmethod
    def encoded_shape(cls) -> Tuple[int, int, int]:
        """Returns (channels, height, width) = (3, 3, 3)."""
        return (3, 3, 3)  # 3 channels: current player, opponent, empty squares
    
    @classmethod
    def initial_state(cls) -> "TicTacToeState":
        """Returns empty board with player 1 to move."""
        return cls()
    
    def get_legal_moves(self) -> List[TicTacToeMove]:
        """Returns all moves to empty squares."""
        moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == 0:
                    moves.append(TicTacToeMove(row, col))
        return moves
    
    def apply_move(self, move: TicTacToeMove) -> "TicTacToeState":
        """Returns new state after applying move."""
        if self.board[move.row, move.col] != 0:
            raise ValueError(f"Square ({move.row}, {move.col}) is not empty")
        
        new_board = self.board.copy()
        new_board[move.row, move.col] = self.current_player
        return TicTacToeState(new_board, -self.current_player)
    
    def is_terminal(self) -> bool:
        """Check if game is over (win or draw)."""
        return self._check_winner() is not None or len(self.get_legal_moves()) == 0
    
    def get_value(self) -> Optional[float]:
        """Returns game value from current player's perspective."""
        if not self.is_terminal():
            return None
        
        winner = self._check_winner()
        if winner is None:
            return 0.0  # Draw
        elif winner == 1:
            # Player 1 won - return value from current player's perspective
            return 1.0 if self.current_player == 1 else -1.0
        else:  # winner == -1
            # Player 2 won - return value from current player's perspective  
            return 1.0 if self.current_player == -1 else -1.0
    
    def encode(self) -> torch.Tensor:
        """
        Encode state as 3x3x3 tensor.
        Channel 0: Current player's pieces
        Channel 1: Opponent's pieces
        Channel 2: Empty squares
        """
        tensor = torch.zeros(3, 3, 3)
        
        # Channel 0: Current player's pieces
        tensor[0] = torch.tensor((self.board == self.current_player).astype(float))
        
        # Channel 1: Opponent's pieces
        tensor[1] = torch.tensor((self.board == -self.current_player).astype(float))
        
        # Channel 2: Empty squares
        tensor[2] = torch.tensor((self.board == 0).astype(float))
        
        return tensor
    
    def _check_winner(self) -> Optional[int]:
        """Check if there's a winner. Returns 1, -1, or None."""
        # Check rows
        for row in range(3):
            if abs(self.board[row].sum()) == 3:
                return self.board[row, 0]
        
        # Check columns
        for col in range(3):
            if abs(self.board[:, col].sum()) == 3:
                return self.board[0, col]
        
        # Check diagonals
        if abs(np.trace(self.board)) == 3:
            return self.board[0, 0]
        if abs(np.trace(np.fliplr(self.board))) == 3:
            return self.board[0, 2]
        
        return None
    
    def __str__(self) -> str:
        """String representation of the board."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        lines = []
        for row in range(3):
            line = ' '.join(symbols[self.board[row, col]] for col in range(3))
            lines.append(line)
        
        player_symbol = 'X' if self.current_player == 1 else 'O'
        return f"Board:\n" + "\n".join(lines) + f"\nNext: {player_symbol}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TicTacToeState):
            return False
        return np.array_equal(self.board, other.board) and self.current_player == other.current_player


# Convenience functions for testing
def play_interactive_game():
    """Play an interactive TicTacToe game."""
    state = TicTacToeState.initial_state()
    
    print("TicTacToe Game!")
    print("Positions are (row, col) from 0-2")
    print(state)
    
    while not state.is_terminal():
        try:
            row = int(input("Enter row (0-2): "))
            col = int(input("Enter col (0-2): "))
            move = TicTacToeMove(row, col)
            
            if move not in state.get_legal_moves():
                print("Invalid move! Try again.")
                continue
            
            state = state.apply_move(move)
            print(f"\nAfter move {move}:")
            print(state)
            
        except (ValueError, IndexError):
            print("Invalid input! Enter numbers 0-2.")
    
    # Game over
    value = state.get_value()
    if value == 0:
        print("It's a draw!")
    elif value == 1:
        winner = "X" if state.current_player == 1 else "O"
        print(f"Player {winner} wins!")
    else:
        winner = "O" if state.current_player == -1 else "X"  
        print(f"Player {winner} wins!")


# Aliases for training framework compatibility
GameState = TicTacToeState
Move = TicTacToeMove


if __name__ == "__main__":
    # Quick test
    print("Testing TicTacToe implementation...")
    
    # Test initial state
    state = TicTacToeState.initial_state()
    print(f"Initial state:\n{state}")
    print(f"Legal moves: {len(state.get_legal_moves())}")
    print(f"Encoded shape: {state.encoded_shape()}")
    print(f"Is terminal: {state.is_terminal()}")
    
    # Test a few moves
    move1 = TicTacToeMove(1, 1)  # Center
    state = state.apply_move(move1)
    print(f"\nAfter center move:\n{state}")
    
    move2 = TicTacToeMove(0, 0)  # Corner
    state = state.apply_move(move2)
    print(f"\nAfter corner move:\n{state}")
    
    print("\nEncoded tensor shape:", state.encode().shape)
    print("All tests passed!")
    
    # Uncomment to play interactively:
    # play_interactive_game()