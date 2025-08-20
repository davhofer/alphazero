from __future__ import annotations

import numpy as np
import time
import torch
from typing import Optional, List

from games import game
import network


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
        self.visit_count = 0.0
        self.value = 0.0
        self.children = []
        self.move = move

        self.player = 1 if parent is None else -parent.player

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def alphazero_ucb(self, c_puct: float = 1.0) -> float:
        """
        AlphaZero UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        With:        Where:
        - Q(s,a) = average value of this node (exploitation)
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
            q_value = self.value / self.visit_count

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
    policy, value = model.forward(state_tensor)

    # Extract single results from batch
    policy = policy[0]  # Remove batch dimension
    value = value[0, 0].item()  # Extract scalar value

    legal_moves = node.state.get_legal_moves()

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


def run_mcts(root: Node, model: network.Model, time_limit: float = 0.5) -> torch.Tensor:
    start_time = time.time()
    iterations = 0

    while time.time() - start_time < time_limit:
        # TODO: should we calculate avg. iteration time and take it into account?
        node = select(root)
        reward = expand(node, model)
        backpropagate(node, reward)
        iterations += 1

    # If root was expanded but no children visited, fall back to uniform over legal moves
    if root.children and all(child.visit_count == 0 for child in root.children):
        legal_moves = root.state.get_legal_moves()
        policy = [0.0] * root.state.__class__.num_possible_moves()
        if legal_moves:
            uniform_prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                encoded_move = move.encode()
                policy[encoded_move] = uniform_prob
        return torch.tensor(policy, dtype=torch.float32)

    # Create policy based on visit counts
    policy = [0.0] * root.state.__class__.num_possible_moves()

    # Avoid division by zero - if no visits, return uniform policy over legal moves
    if root.visit_count == 0:
        legal_moves = root.state.get_legal_moves()
        if legal_moves:
            uniform_prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                encoded_move = move.encode()
                policy[encoded_move] = uniform_prob
        return torch.tensor(policy, dtype=torch.float32)

    for child in root.children:
        if child.move is not None:  # Safety check
            encoded_move = child.move.encode()
            policy[encoded_move] = child.visit_count / root.visit_count

    return torch.tensor(policy, dtype=torch.float32)
