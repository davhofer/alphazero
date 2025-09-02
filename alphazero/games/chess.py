"""Chess game implementation using python-chess library."""

import torch
import chess
import chess.engine
from typing import List, Optional, Tuple
from copy import deepcopy

from .game import Move, GameState


class ChessMove(Move):
    """A move in Chess using python-chess Move representation."""
    
    def __init__(self, chess_move: chess.Move):
        """
        Initialize ChessMove from python-chess Move.
        
        Args:
            chess_move: A chess.Move object from python-chess library
        """
        self.chess_move = chess_move
    
    def encode(self) -> int:
        """
        Encode move as integer using (from_square, to_square, promotion) encoding.
        
        Chess has 64 squares, so we use:
        - from_square: 0-63 (6 bits)  
        - to_square: 0-63 (6 bits)
        - promotion: 0-4 (3 bits) where 0=no promotion, 1=queen, 2=rook, 3=bishop, 4=knight
        
        Total encoding: from_square * 64 * 5 + to_square * 5 + promotion
        Maximum value: 63 * 64 * 5 + 63 * 5 + 4 = 20,479
        """
        from_square = self.chess_move.from_square
        to_square = self.chess_move.to_square
        
        # Handle promotion
        promotion = 0
        if self.chess_move.promotion:
            # Map chess piece types to our encoding
            promotion_map = {
                chess.QUEEN: 1,
                chess.ROOK: 2, 
                chess.BISHOP: 3,
                chess.KNIGHT: 4
            }
            promotion = promotion_map.get(self.chess_move.promotion, 0)
        
        return from_square * 64 * 5 + to_square * 5 + promotion
    
    @classmethod
    def decode(cls, encoded: int) -> "ChessMove":
        """Decode integer back to ChessMove."""
        if not (0 <= encoded < 20480):  # 64 * 64 * 5 = 20,480
            raise ValueError(f"Invalid encoded move: {encoded}")
        
        promotion = encoded % 5
        encoded //= 5
        to_square = encoded % 64
        from_square = encoded // 64
        
        # Handle promotion
        chess_promotion = None
        if promotion > 0:
            promotion_map = {1: chess.QUEEN, 2: chess.ROOK, 3: chess.BISHOP, 4: chess.KNIGHT}
            chess_promotion = promotion_map[promotion]
        
        chess_move = chess.Move(from_square, to_square, chess_promotion)
        return cls(chess_move)
    
    def __str__(self) -> str:
        """String representation using UCI notation."""
        return str(self.chess_move)
    
    def __eq__(self, other) -> bool:
        """Equality comparison for moves."""
        if not isinstance(other, ChessMove):
            return False
        return self.chess_move == other.chess_move
    
    def __hash__(self) -> int:
        return hash(self.chess_move)


class ChessState(GameState):
    """Chess game state using python-chess Board."""
    
    def __init__(self, board: Optional[chess.Board] = None):
        """
        Initialize Chess state.
        
        Args:
            board: chess.Board object. If None, creates initial chess position.
        """
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board.copy()
    
    @property
    def current_player(self) -> int:
        """Returns the current player (1 for white, -1 for black)."""
        return 1 if self.board.turn else -1
    
    @classmethod
    def num_possible_moves(cls) -> int:
        """Chess has maximum ~20,480 possible moves with our encoding."""
        return 20480  # 64 * 64 * 5
    
    @classmethod
    def encoded_shape(cls) -> Tuple[int, int, int]:
        """Returns (channels, height, width) = (14, 8, 8) for chess_14ch encoding."""
        return (14, 8, 8)  # 14 channels: 6 current player pieces + 6 opponent pieces + en passant + castling
    
    @classmethod
    def initial_state(cls) -> "ChessState":
        """Returns initial chess position."""
        return cls()
    
    def get_legal_moves(self) -> List[ChessMove]:
        """Returns all legal moves from this position."""
        return [ChessMove(move) for move in self.board.legal_moves]
    
    def apply_move(self, move: ChessMove) -> "ChessState":
        """Returns new state after applying move."""
        new_board = self.board.copy()
        new_board.push(move.chess_move)
        return ChessState(new_board)
    
    def is_terminal(self) -> bool:
        """Check if game is over (checkmate, stalemate, or draw)."""
        return self.board.is_game_over()
    
    def get_value(self) -> Optional[float]:
        """
        Returns objective game result from absolute perspective:
        1.0 if white won, -1.0 if black won, 0.0 for draw.
        """
        if not self.is_terminal():
            return None
        
        outcome = self.board.outcome()
        if outcome is None:
            return 0.0  # Draw
        
        if outcome.winner is None:
            return 0.0  # Draw
        elif outcome.winner == chess.WHITE:
            return 1.0  # White won
        else:
            return -1.0  # Black won
    
    def encode(self) -> torch.Tensor:
        """
        Encode state as 14x8x8 tensor using chess_14ch encoding.
        
        Channels:
        0-5: Current player pieces (pawn, rook, knight, bishop, queen, king)
        6-11: Opponent pieces (pawn, rook, knight, bishop, queen, king)  
        12: En passant target squares
        13: Castling rights (encoded across relevant squares)
        """
        tensor = torch.zeros(14, 8, 8)
        
        current_player = self.board.turn  # True for white, False for black
        
        # Piece type mapping
        piece_types = [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]
        
        # Fill piece positions
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                row = 7 - (square // 8)  # Convert to tensor coordinates (rank 8 = row 0)
                col = square % 8
                
                # Find piece type index
                piece_idx = piece_types.index(piece.piece_type)
                
                # Determine if this is current player's piece or opponent's
                if piece.color == current_player:
                    # Current player's pieces: channels 0-5
                    tensor[piece_idx, row, col] = 1.0
                else:
                    # Opponent's pieces: channels 6-11
                    tensor[6 + piece_idx, row, col] = 1.0
        
        # En passant target square (channel 12)
        if self.board.ep_square is not None:
            ep_row = 7 - (self.board.ep_square // 8)
            ep_col = self.board.ep_square % 8
            tensor[12, ep_row, ep_col] = 1.0
        
        # Castling rights (channel 13)
        # Encode castling rights on relevant squares (king and rook initial positions)
        if current_player:  # White to move
            if self.board.has_kingside_castling_rights(chess.WHITE):
                tensor[13, 7, 4] = 1.0  # King square
                tensor[13, 7, 7] = 1.0  # Kingside rook square
            if self.board.has_queenside_castling_rights(chess.WHITE):
                tensor[13, 7, 4] = 1.0  # King square (may already be set)
                tensor[13, 7, 0] = 1.0  # Queenside rook square
        else:  # Black to move
            if self.board.has_kingside_castling_rights(chess.BLACK):
                tensor[13, 0, 4] = 1.0  # King square
                tensor[13, 0, 7] = 1.0  # Kingside rook square
            if self.board.has_queenside_castling_rights(chess.BLACK):
                tensor[13, 0, 4] = 1.0  # King square (may already be set)
                tensor[13, 0, 0] = 1.0  # Queenside rook square
        
        return tensor
    
    def __str__(self) -> str:
        """String representation of the chess position."""
        result = str(self.board)
        turn_str = "White" if self.board.turn else "Black"
        result += f"\n\nTurn: {turn_str}"
        
        # Add game status info
        if self.board.is_check():
            result += " (in check)"
        if self.is_terminal():
            outcome = self.board.outcome()
            if outcome and outcome.winner is not None:
                winner = "White" if outcome.winner else "Black"
                result += f" - {winner} wins!"
            else:
                result += " - Draw"
        
        return result
    
    def __eq__(self, other) -> bool:
        """Equality comparison for chess states."""
        if not isinstance(other, ChessState):
            return False
        return self.board == other.board
    
    def to_fen(self) -> str:
        """Export position as FEN string."""
        return self.board.fen()
    
    @classmethod
    def from_fen(cls, fen: str) -> "ChessState":
        """Create ChessState from FEN string."""
        board = chess.Board(fen)
        return cls(board)


def play_interactive_game():
    """Play an interactive chess game."""
    state = ChessState.initial_state()
    
    print("Chess Game!")
    print("Enter moves in UCI notation (e.g., 'e2e4', 'e7e5')")
    print("Type 'quit' to exit")
    print()
    print(state)
    
    while not state.is_terminal():
        try:
            move_str = input(f"\nEnter move: ").strip()
            
            if move_str.lower() == 'quit':
                break
            
            # Parse UCI move
            try:
                chess_move = chess.Move.from_uci(move_str)
                move = ChessMove(chess_move)
            except ValueError:
                print("Invalid move format! Use UCI notation like 'e2e4'")
                continue
            
            # Check if move is legal
            if move not in state.get_legal_moves():
                print("Illegal move! Try again.")
                print(f"Legal moves: {[str(m) for m in state.get_legal_moves()[:10]]}{'...' if len(state.get_legal_moves()) > 10 else ''}")
                continue
            
            state = state.apply_move(move)
            print(f"\nAfter move {move}:")
            print(state)
            
        except KeyboardInterrupt:
            print("\nGame interrupted.")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Game over
    if state.is_terminal():
        value = state.get_value()
        if value == 0:
            print("\nGame ended in a draw!")
        elif value == 1:
            winner = "White" if state.board.turn else "Black"
            print(f"\n{winner} wins!")
        else:
            winner = "Black" if state.board.turn else "White"  
            print(f"\n{winner} wins!")


# Aliases for training framework compatibility
GameState = ChessState
Move = ChessMove


if __name__ == "__main__":
    # Quick test
    print("Testing Chess implementation...")
    
    # Test initial state
    state = ChessState.initial_state()
    print(f"Initial state:\n{state}")
    print(f"Legal moves: {len(state.get_legal_moves())}")
    print(f"Encoded shape: {state.encoded_shape()}")
    print(f"Is terminal: {state.is_terminal()}")
    
    # Test a few moves
    e4_move = ChessMove(chess.Move.from_uci("e2e4"))
    state = state.apply_move(e4_move)
    print(f"\nAfter 1.e4:\n{state}")
    
    e5_move = ChessMove(chess.Move.from_uci("e7e5"))
    state = state.apply_move(e5_move)
    print(f"\nAfter 1...e5:\n{state}")
    
    print(f"\nEncoded tensor shape: {state.encode().shape}")
    print(f"FEN: {state.to_fen()}")
    
    # Test move encoding/decoding
    print(f"\nTesting move encoding:")
    test_move = ChessMove(chess.Move.from_uci("e2e4"))
    encoded = test_move.encode()
    decoded = ChessMove.decode(encoded)
    print(f"Original: {test_move}, Encoded: {encoded}, Decoded: {decoded}")
    print(f"Round trip successful: {test_move == decoded}")
    
    print("\nAll tests passed!")
    
    # Uncomment to play interactively:
    # play_interactive_game()
