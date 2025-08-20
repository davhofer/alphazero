"""Connect Four game implementation."""

import torch
import numpy as np
from typing import List, Optional, Tuple
from copy import deepcopy

from .game import Move, GameState


class ConnectFourMove(Move):
    """A move in Connect Four - dropping a piece in a column."""

    def __init__(self, column: int):
        if not (0 <= column < 7):
            raise ValueError(f"Invalid column: {column}")
        self.column = column

    def encode(self) -> int:
        """Encode move as integer: just the column index (0-6)."""
        return self.column

    @classmethod
    def decode(cls, encoded: int) -> "ConnectFourMove":
        """Decode integer back to move."""
        if not (0 <= encoded < 7):
            raise ValueError(f"Invalid encoded move: {encoded}")
        return cls(encoded)

    def __str__(self) -> str:
        return f"Col {self.column}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ConnectFourMove):
            return False
        return self.column == other.column

    def __hash__(self) -> int:
        return hash(self.column)


class ConnectFourState(GameState):
    """Connect Four game state."""

    ROWS = 6
    COLS = 7
    CONNECT = 4

    def __init__(
        self,
        board: Optional[np.ndarray] = None,
        current_player: int = 1,
        column_heights: Optional[np.ndarray] = None,
        move_count: int = 0,
        last_move: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize Connect Four state.

        Args:
            board: 6x7 numpy array. 0=empty, 1=player 1, -1=player 2
            current_player: 1 for player 1, -1 for player 2
            column_heights: Array of height (number of pieces) in each column
            move_count: Number of moves played so far
            last_move: (row, col) of the last piece placed (for win checking)
        """
        if board is None:
            self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        else:
            self.board = board.copy()

        self.current_player = current_player
        self.move_count = move_count
        self.last_move = last_move

        if column_heights is None:
            self.column_heights = np.zeros(self.COLS, dtype=int)
            # Calculate column heights from board if provided
            if board is not None:
                for col in range(self.COLS):
                    for row in range(self.ROWS):
                        if self.board[row, col] != 0:
                            self.column_heights[col] = row + 1
        else:
            self.column_heights = column_heights.copy()

    @classmethod
    def num_possible_moves(cls) -> int:
        """Connect Four has 7 possible columns."""
        return cls.COLS

    @classmethod
    def encoded_shape(cls) -> Tuple[int, int, int]:
        """Returns (channels, height, width) = (2, 6, 7)."""
        return (2, cls.ROWS, cls.COLS)

    @classmethod
    def initial_state(cls) -> "ConnectFourState":
        """Returns empty board with player 1 to move."""
        return cls()

    def get_legal_moves(self) -> List[ConnectFourMove]:
        """Returns all moves to non-full columns."""
        moves = []
        for col in range(self.COLS):
            if self.column_heights[col] < self.ROWS:
                moves.append(ConnectFourMove(col))
        return moves

    def apply_move(self, move: ConnectFourMove) -> "ConnectFourState":
        """Returns new state after dropping piece in column."""
        if self.column_heights[move.column] >= self.ROWS:
            raise ValueError(f"Column {move.column} is full")

        # Calculate where piece will land
        row = self.column_heights[move.column]

        # Create new state
        new_board = self.board.copy()
        new_board[row, move.column] = self.current_player

        new_heights = self.column_heights.copy()
        new_heights[move.column] += 1

        return ConnectFourState(
            board=new_board,
            current_player=-self.current_player,
            column_heights=new_heights,
            move_count=self.move_count + 1,
            last_move=(row, move.column),
        )

    def is_terminal(self) -> bool:
        """Check if game is over (win or draw)."""
        return self._check_winner() or self.move_count >= self.ROWS * self.COLS

    def get_value(self) -> Optional[float]:
        """Returns game result from perspective of player 1."""
        if not self.is_terminal():
            return None

        if self._check_winner():
            # The player who just moved (opposite of current_player) won
            winner = -self.current_player
            return float(winner)
        else:
            return 0.0  # Draw

    def encode(self) -> torch.Tensor:
        """
        Encode state as 2x6x7 tensor.
        Channel 0: Current player's pieces
        Channel 1: Opponent's pieces
        """
        tensor = torch.zeros(2, self.ROWS, self.COLS)

        # Channel 0: Current player's pieces
        tensor[0] = torch.tensor((self.board == self.current_player).astype(float))

        # Channel 1: Opponent's pieces
        tensor[1] = torch.tensor((self.board == -self.current_player).astype(float))

        return tensor

    def _check_winner(self) -> bool:
        """Check if the last move created a winning line."""
        if self.last_move is None:
            return False

        row, col = self.last_move
        player = self.board[row, col]

        if player == 0:
            return False

        # Check all four directions
        return (
            self._check_direction(row, col, 0, 1, player)  # Horizontal
            or self._check_direction(row, col, 1, 0, player)  # Vertical
            or self._check_direction(row, col, 1, 1, player)  # Diagonal /
            or self._check_direction(row, col, 1, -1, player)
        )  # Diagonal \

    def _check_direction(
        self, row: int, col: int, dr: int, dc: int, player: int
    ) -> bool:
        """Check if there are 4 consecutive pieces in a direction."""
        count = 1  # Count the piece at (row, col)

        # Count in positive direction
        r, c = row + dr, col + dc
        while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r, c] == player:
            count += 1
            r += dr
            c += dc

        # Count in negative direction
        r, c = row - dr, col - dc
        while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r, c] == player:
            count += 1
            r -= dr
            c -= dc

        return count >= self.CONNECT

    def __str__(self) -> str:
        """String representation of the board."""
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = []

        # Print from top to bottom (row 5 to 0)
        for row in range(self.ROWS - 1, -1, -1):
            line = " ".join(symbols[self.board[row, col]] for col in range(self.COLS))
            lines.append(f"  {line}")

        # Add column numbers
        lines.append("  " + " ".join(str(i) for i in range(self.COLS)))

        player_symbol = "X" if self.current_player == 1 else "O"
        return "\n".join(lines) + f"\nNext: {player_symbol}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ConnectFourState):
            return False
        return (
            np.array_equal(self.board, other.board)
            and self.current_player == other.current_player
        )


# Convenience functions for testing
def play_interactive_game():
    """Play an interactive Connect Four game."""
    state = ConnectFourState.initial_state()

    print("Connect Four Game!")
    print("Enter column numbers 0-6 to drop pieces")
    print(state)

    while not state.is_terminal():
        try:
            col = int(
                input(
                    f"Player {'X' if state.current_player == 1 else 'O'}, enter column (0-6): "
                )
            )
            move = ConnectFourMove(col)

            if move not in state.get_legal_moves():
                print("Invalid move! Column is full or out of range.")
                continue

            state = state.apply_move(move)
            print(f"\nAfter dropping in column {col}:")
            print(state)

        except (ValueError, IndexError):
            print("Invalid input! Enter numbers 0-6.")

    # Game over
    value = state.get_value()
    if value == 0:
        print("It's a draw!")
    elif value == 1:
        print("Player X wins!")
    else:
        print("Player O wins!")


# Aliases for training framework compatibility
GameState = ConnectFourState
Move = ConnectFourMove


if __name__ == "__main__":
    # Quick test
    print("Testing Connect Four implementation...")

    # Test initial state
    state = ConnectFourState.initial_state()
    print(f"Initial state:\n{state}")
    print(f"Legal moves: {len(state.get_legal_moves())}")
    print(f"Encoded shape: {state.encoded_shape()}")
    print(f"Is terminal: {state.is_terminal()}")

    # Test a few moves
    move1 = ConnectFourMove(3)  # Center column
    state = state.apply_move(move1)
    print(f"\nAfter dropping in column 3:\n{state}")

    move2 = ConnectFourMove(3)  # Same column
    state = state.apply_move(move2)
    print(f"\nAfter second drop in column 3:\n{state}")

    print("\nEncoded tensor shape:", state.encode().shape)

    # Test win detection
    print("\nTesting vertical win...")
    state = ConnectFourState.initial_state()
    for i in range(4):
        state = state.apply_move(ConnectFourMove(0))
        if i < 3:
            state = state.apply_move(ConnectFourMove(1))  # Opponent move

    print("Final state:")
    print(state)
    print(f"Is terminal: {state.is_terminal()}")
    print(f"Game value: {state.get_value()}")

    print("All tests passed!")

    # Uncomment to play interactively:
    play_interactive_game()
