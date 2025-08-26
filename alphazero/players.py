"""Different player implementations for game evaluation."""

import random
import torch
from abc import ABC, abstractmethod
from typing import Optional

from .games import game
from . import network
from . import mcts
from . import training


class Player(ABC):
    """Base class for all players."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_move(self, state: game.GameState) -> game.Move:
        """Select a move given the current game state."""
        pass

    def __str__(self) -> str:
        return self.name


class RandomPlayer(Player):
    """Player that selects moves uniformly at random from legal moves."""

    def __init__(self):
        super().__init__("Random")

    def select_move(self, state: game.GameState) -> game.Move:
        legal_moves = state.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available")
        return random.choice(legal_moves)


class NetworkMCTSPlayer(Player):
    """Player using neural network with MCTS (full AlphaZero player)."""

    def __init__(
        self,
        model: network.Model,
        mcts_time_limit: float = 1.0,
        name: str = "NetworkMCTS",
        temperature: float = 0.0,  # Default to deterministic play for evaluation
    ):
        super().__init__(name)
        self.model = model
        self.mcts_time_limit = mcts_time_limit
        self.temperature = temperature
        self.model.eval()

    def select_move(self, state: game.GameState) -> game.Move:
        # Run MCTS to get improved policy
        root_node = mcts.Node(None, state, 1.0, None)
        policy = mcts.run_mcts(root_node, self.model, time_limit=self.mcts_time_limit)

        # Get legal moves
        legal_moves = state.get_legal_moves()

        if self.temperature == 0:
            # Deterministic: pick the move with highest MCTS probability
            best_move = None
            best_prob = -1
            for move in legal_moves:
                prob = policy[move.encode()].item()
                if prob > best_prob:
                    best_prob = prob
                    best_move = move
            return best_move if best_move else random.choice(legal_moves)
        else:
            # Stochastic: sample from MCTS policy (with optional temperature)
            policy_probs = policy.numpy()

            # Filter policy to only legal moves
            legal_probs = []
            legal_move_list = []
            for move in legal_moves:
                encoded_move = move.encode()
                legal_probs.append(policy_probs[encoded_move])
                legal_move_list.append(move)

            # Normalize probabilities
            total_prob = sum(legal_probs)
            if total_prob > 0:
                legal_probs = [p / total_prob for p in legal_probs]

                # Apply temperature (1.0 returns unchanged, so always safe to call)
                temp_policy = training.apply_temperature(
                    torch.tensor(legal_probs), self.temperature
                )
                legal_probs = temp_policy.numpy().tolist()

                selected_move = random.choices(legal_move_list, weights=legal_probs)[0]
            else:
                # Fallback to random if all probabilities are zero
                selected_move = random.choice(legal_move_list)

            return selected_move


class NetworkDirectPlayer(Player):
    """Player using neural network policy directly (no MCTS)."""

    def __init__(self, model: network.Model, name: str = "NetworkDirect"):
        super().__init__(name)
        self.model = model
        self.model.eval()

    def select_move(self, state: game.GameState) -> game.Move:
        # Get network policy directly
        with torch.no_grad():
            state_tensor = state.encode().unsqueeze(0)  # Add batch dimension
            # Move tensor to same device as model
            device = next(self.model.parameters()).device
            state_tensor = state_tensor.to(device)
            policy_logits, _ = self.model(state_tensor)
            policy_logits = policy_logits[0]  # Remove batch dimension

        # Get legal moves
        legal_moves = state.get_legal_moves()
        
        # Create mask for legal moves
        legal_mask = torch.zeros_like(policy_logits)
        for move in legal_moves:
            legal_mask[move.encode()] = 1.0
        
        # Mask illegal moves by setting their logits to -inf
        masked_logits = policy_logits.clone()
        masked_logits[legal_mask == 0] = -float('inf')
        
        # Apply softmax to get probabilities
        policy = torch.softmax(masked_logits, dim=0)
        policy_probs = policy.cpu().numpy()

        legal_probs = []
        legal_move_list = []
        for move in legal_moves:
            encoded_move = move.encode()
            legal_probs.append(policy_probs[encoded_move])
            legal_move_list.append(move)

        # Normalize and sample (should already be normalized from softmax)
        total_prob = sum(legal_probs)
        if total_prob > 0:
            selected_move = random.choices(legal_move_list, weights=legal_probs)[0]
        else:
            selected_move = random.choice(legal_move_list)

        return selected_move


class GreedyPlayer(Player):
    """Player that uses simple heuristics to make greedy moves."""

    def __init__(self, name: str = "Greedy"):
        super().__init__(name)

    def select_move(self, state: game.GameState) -> game.Move:
        legal_moves = state.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available")

        # Simple heuristic: prefer moves that form mills or capture pieces
        best_moves = []
        best_score = float("-inf")

        for move in legal_moves:
            next_state = state.apply_move(move)
            score = self._evaluate_state(next_state, state)

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)

    def _evaluate_state(
        self, new_state: game.GameState, old_state: game.GameState
    ) -> float:
        """Simple evaluation function. Override for game-specific heuristics."""
        # This is a placeholder - should be implemented based on your game
        # For now, just prefer non-terminal states over terminal losing states
        if new_state.is_terminal():
            value = new_state.get_value()
            return value if value is not None else 0
        return 0


class HumanPlayer(Player):
    """Interactive human player."""

    def __init__(self, name: str = "Human"):
        super().__init__(name)

    def select_move(self, state: game.GameState) -> game.Move:
        legal_moves = state.get_legal_moves()

        print(f"\nCurrent state: {state}")  # This would need game-specific display
        print("Legal moves:")
        for i, move in enumerate(legal_moves):
            print(f"{i}: {move}")

        while True:
            try:
                choice = int(input(f"Select move (0-{len(legal_moves) - 1}): "))
                if 0 <= choice < len(legal_moves):
                    return legal_moves[choice]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
