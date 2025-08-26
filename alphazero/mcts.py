from __future__ import annotations

import numpy as np
import time
import torch
from typing import Optional, List

from .games import game
from . import network


class Node:
    parent: "Node" | None
    # internal state representation
    state: game.GameState
    children: list["Node"]
    player: int  # 1 or -1
    visit_count: float
    value: float
    prior: float  # prior probability assigned to it by parent
    move: game.Move | None  # move that lead to this node

    def __init__(
        self,
        parent: "Node" | None,
        state: game.GameState,
        prior: float,
        move: game.Move | None = None,
    ):
        self.parent = parent
        self.state = state
        self.prior = prior
        self.visit_count = 1.0 if parent is None else 0.0
        self.value = 0.0
        self.children = []
        self.move = move

        self.player = 1 if parent is None else -parent.player

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def alphazero_ucb(self, c_puct: float = 1.0) -> float:
        """
        AlphaZero UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Where:
        - Q(s,a) = average value of this node from parent's perspective
        - P(s,a) = prior probability from neural network policy
        - N(s) = parent visit count
        - N(s,a) = this node's visit count
        - c_puct = exploration constant (typically 1.0 for AlphaZero)
        """
        if self.parent is None:
            return 0.0

        if self.visit_count == 0:
            q_value = 0.0
        else:
            # self.value is accumulated from this node's player perspective
            # Negate to get value from parent's perspective (who is selecting the move)
            # Also normalize between 0 and 1
            q_value = 1 - ((self.value / self.visit_count) + 1) / 2

        # Exploration term: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        exploration = (
            c_puct
            * self.prior
            * np.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        return q_value + exploration

    def best_child_ucb(self) -> "Node":
        """Select best child using AlphaZero UCB formula."""
        if not self.children:
            return self

        ucbs = [node.alphazero_ucb() for node in self.children]
        best_idx = np.argmax(ucbs)
        return self.children[best_idx]


def select(node: Node) -> Node:
    """Walk down until leaf node, picking best child using AlphaZero UCB."""
    while node.children:
        node = node.best_child_ucb()
    return node


def expand(node: Node, model: network.Model) -> float:
    """Expand all child nodes of current (leaf) node and compute its value."""
    assert not node.children

    if node.is_terminal():
        objective_value = node.state.get_value()
        # Convert objective result to current player's perspective
        # Objective value: +1 if player 1 won, -1 if player -1 won, 0 for draw
        # Return value from current player's perspective
        return objective_value * node.player

    # Get tensor and add batch dimension for network
    state_tensor = node.state.encode().unsqueeze(0)  # (1, channels, height, width)
    # Move tensor to same device as model
    device = next(model.parameters()).device
    state_tensor = state_tensor.to(device)
    policy_logits, value = model.forward(state_tensor)

    # Extract single results from batch
    policy_logits = policy_logits[0]  # Remove batch dimension
    value = value[0, 0].item()  # Extract scalar value

    legal_moves = node.state.get_legal_moves()

    # Create mask for legal moves
    legal_mask = torch.zeros_like(policy_logits)
    for move in legal_moves:
        legal_mask[move.encode()] = 1.0

    # Mask illegal moves by setting their logits to -inf
    masked_logits = policy_logits.clone()
    masked_logits[legal_mask == 0] = -float('inf')
    
    # Apply softmax to get probabilities
    policy = torch.softmax(masked_logits, dim=0)
    
    # Handle edge case where all moves are illegal (shouldn't happen normally)
    if torch.isnan(policy).any() or policy.sum() == 0:
        # If all legal moves have zero probability, use uniform distribution
        policy = torch.zeros_like(policy_logits)
        uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
        for move in legal_moves:
            policy[move.encode()] = uniform_prob

    # Create child nodes with normalized probabilities
    for move in legal_moves:
        encoded_move = move.encode()
        p = policy[encoded_move].detach().item()
        new_state = node.state.apply_move(move)
        child = Node(node, new_state, p, move)
        node.children.append(child)

    return value


def backpropagate(node: Node, reward: float) -> None:
    """Update statistics up the tree."""
    current_node: Node | None = node
    current_reward = reward

    while current_node is not None:
        # Add reward from current player's perspective
        current_node.value += current_reward
        current_node.visit_count += 1

        current_node = current_node.parent
        # Flip reward for next level up (alternating players)
        current_reward = -current_reward


@torch.no_grad()
def run_mcts(
    root: Node,
    model: network.Model,
    time_limit: float = 0.5,
    training: bool = False,
    dirichlet_epsilon: float = 0.25,
    dirichlet_alpha: float = 0.3,
) -> torch.Tensor:
    """Run MCTS simulations from root node.

    Args:
        root: Root node to start search from
        model: Neural network model for evaluation
        time_limit: Time limit for search in seconds
        training: If True, add Dirichlet noise to root for exploration during self-play
        dirichlet_epsilon: Mixing parameter for noise (typically 0.25)
        dirichlet_alpha: Dirichlet distribution parameter (0.3 for chess, 0.03 for Go)
    """
    # Pre-expand root node if needed
    if not root.children and not root.is_terminal():
        _ = expand(root, model)
        
        # Add Dirichlet noise for exploration during training
        if training and root.children:
            noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children):
                child.prior = (
                    1 - dirichlet_epsilon
                ) * child.prior + dirichlet_epsilon * noise[i]
    
    start_time = time.time()

    while time.time() - start_time < time_limit:
        # TODO: should we calculate avg. iteration time and take it into account?
        node = select(root)
        reward = expand(node, model)
        backpropagate(node, reward)

    # Create policy based on visit counts
    policy = [0.0] * root.state.__class__.num_possible_moves()

    # Calculate total visits to children
    total_child_visits = sum(child.visit_count for child in root.children)

    # If no child visits, return uniform policy over legal moves
    if total_child_visits == 0:
        legal_moves = root.state.get_legal_moves()
        if legal_moves:
            uniform_prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                encoded_move = move.encode()
                policy[encoded_move] = uniform_prob
        return torch.tensor(policy, dtype=torch.float32)

    # Calculate policy based on child visit ratios
    for child in root.children:
        if child.move is not None:  # Safety check
            encoded_move = child.move.encode()
            policy[encoded_move] = child.visit_count / total_child_visits

    return torch.tensor(policy, dtype=torch.float32)
