# self-play
# record games/positions
# train network

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from . import mcts
from . import network
from . import evaluation
from .games import load_game_module


@dataclass
class TrainingConfig:
    """Configuration for AlphaZero training."""

    # Game selection
    game_module: str = "tictactoe"  # Module name to import (e.g., "tictactoe", "mills")

    # (Outer) training loop parameters
    iterations: int = 100
    checkpoint_frequency: int = 10

    # Self-play parameters
    self_play_rounds: int = 100
    mcts_time_limit: float = 0.5
    temperature_threshold: int = 30  # Use temperature=1 for first N moves, then 0
    temperature_exploration: float = 1.0  # Temperature for exploration phase
    temperature_exploitation: float = 0.0  # Temperature for exploitation phase
    dirichlet_epsilon: float = 0.25  # Mixing parameter for Dirichlet noise
    dirichlet_alpha: float = (
        0.3  # Dirichlet distribution parameter (0.3 for chess, 0.03 for Go)
    )

    # Network training parameters
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # Evaluation parameters
    eval_frequency: int = 20  # Evaluate every N iterations
    eval_games: int = 50  # Number of games for evaluation

    # Network architecture
    num_residual_blocks: int = 4
    num_filters: int = 128

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Device configuration
    device: str | None = None  # "cpu", "cuda", or None for auto-detect

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            return cls(**json.load(f))


def apply_temperature(policy: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature to policy probabilities to control exploration vs exploitation.

    Args:
        policy: Probability distribution over moves
        temperature: Temperature parameter
            - 0: Deterministic (pick best move)
            - 1: Use raw probabilities (unchanged)
            - >1: More exploration (flatter distribution)
            - <1: Less exploration (sharper distribution)

    Returns:
        Modified policy distribution
    """
    if temperature == 0:
        # Deterministic: pick the best move
        best_move = torch.argmax(policy)
        new_policy = torch.zeros_like(policy)
        new_policy[best_move] = 1.0
        return new_policy
    elif temperature == 1.0:
        # No change needed - use original distribution
        return policy
    else:
        # Apply temperature scaling
        # Use log probabilities for numerical stability
        log_policy = torch.log(policy + 1e-8)
        log_policy = log_policy / temperature

        # Subtract max for numerical stability
        log_policy = log_policy - torch.max(log_policy)

        # Exponentiate and normalize
        new_policy = torch.exp(log_policy)
        new_policy = new_policy / torch.sum(new_policy)

        return new_policy


def self_play(model: network.Model, config: TrainingConfig, game_module) -> list[dict]:
    """Let the model play games against itself and record encountered positions and outcomes."""
    training_examples = []

    for _ in tqdm(range(config.self_play_rounds), desc="Self-play games", leave=False):
        state = game_module.GameState.initial_state()
        game_history = []  # (state, policy, current_player) tuples for this game
        move_count = 0  # Track move number for temperature threshold

        # Play one complete game
        while not state.is_terminal():
            root_node = mcts.Node(None, state, 1.0, None)
            policy = mcts.run_mcts(
                root_node,
                model,
                time_limit=config.mcts_time_limit,
                training=True,
                dirichlet_epsilon=config.dirichlet_epsilon,
                dirichlet_alpha=config.dirichlet_alpha,
            )

            # Store state, MCTS policy, and current player for training
            # CRITICAL: We must track which player is to move at this position
            game_history.append((state.encode(), policy, state.current_player))

            # Select temperature based on move count
            temperature = (
                config.temperature_exploration
                if move_count < config.temperature_threshold
                else config.temperature_exploitation
            )

            # Apply temperature to policy (τ=1.0 returns unchanged)
            temperature_policy = apply_temperature(policy, temperature)

            # Sample move from temperature-adjusted policy
            policy_probs = temperature_policy.numpy()
            encoded_move = random.choices(
                range(len(policy_probs)), weights=policy_probs
            )[0]

            # Decode move and apply it
            move = game_module.Move.decode(encoded_move)
            state = state.apply_move(move)
            move_count += 1

        # Game is finished, get final outcome
        final_value = (
            state.get_value()
        )  # +1 if player 1 won, -1 if player 2 won, 0 for draw
        assert final_value is not None

        # Create training examples with final outcome as ground truth
        for state_encoding, mcts_policy, current_player in game_history:
            # Convert objective result to the perspective of the player who was to move
            # final_value is from player 1's perspective (1 = P1 wins, -1 = P2 wins)
            # current_player is who was to move (1 or -1)
            # The encoding already shows the position from current_player's perspective
            player_value = final_value * current_player

            training_examples.append(
                {
                    "state": state_encoding,  # Input tensor (from current player's view)
                    "policy": mcts_policy,  # Target policy
                    "value": player_value,  # Target value (from current player's view)
                }
            )

    return training_examples


def train_network(
    model: network.Model,
    training_samples: list[dict],
    config: TrainingConfig,
) -> dict:
    """Train the neural network on self-play data."""

    if not training_samples:
        print("No training samples provided!")
        return {"avg_loss": 0, "avg_policy_loss": 0, "avg_value_loss": 0}

    # Get device from model
    device = next(model.parameters()).device

    # Prepare training data
    states = torch.stack([sample["state"] for sample in training_samples])
    target_policies = torch.stack([sample["policy"] for sample in training_samples])
    target_values = torch.tensor(
        [sample["value"] for sample in training_samples], dtype=torch.float32
    ).unsqueeze(1)

    # Move data to same device as model
    states = states.to(device)
    target_policies = target_policies.to(device)
    target_values = target_values.to(device)

    # Create dataset and dataloader
    dataset = TensorDataset(states, target_policies, target_values)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Set up optimizer and loss functions
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Loss functions following AlphaZero paper:
    # l = (z-v)^2 - π^T * log p + c * ||θ||^2
    value_loss_fn = nn.MSELoss()  # (z-v)^2

    model.train()  # Set model to training mode

    epoch_losses = []

    for epoch in range(config.epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        # Progress bar for batches
        batch_pbar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}", leave=False
        )

        for batch_states, batch_target_policies, batch_target_values in batch_pbar:
            optimizer.zero_grad()

            # Forward pass
            pred_policy_logits, pred_values = model(batch_states)

            # Calculate losses following AlphaZero paper: l = (z-v)^2 - π^T * log p + c * ||θ||^2

            # Value loss: (z-v)^2
            value_loss = value_loss_fn(pred_values, batch_target_values)

            # Policy loss: cross-entropy between target π and predicted logits
            # F.cross_entropy expects logits and will apply log_softmax internally
            # We use manual calculation since our targets are probabilities, not class indices
            log_probs = torch.log_softmax(pred_policy_logits, dim=1)
            policy_loss = -torch.sum(batch_target_policies * log_probs, dim=1).mean()

            # Combined loss (L2 regularization is handled by optimizer weight_decay)
            total_batch_loss = value_loss + policy_loss

            # Backward pass
            total_batch_loss.backward()
            optimizer.step()

            # Track losses
            total_loss += total_batch_loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

            # Update progress bar with current loss
            batch_pbar.set_postfix(
                {
                    "Total": f"{total_batch_loss.item():.4f}",
                    "Policy": f"{policy_loss.item():.4f}",
                    "Value": f"{value_loss.item():.4f}",
                }
            )

        # Calculate epoch averages
        avg_loss = total_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches

        epoch_losses.append(
            {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "avg_policy_loss": avg_policy_loss,
                "avg_value_loss": avg_value_loss,
            }
        )

    model.eval()  # Set model back to evaluation mode

    # Return summary statistics
    final_stats = (
        epoch_losses[-1]
        if epoch_losses
        else {"avg_loss": 0, "avg_policy_loss": 0, "avg_value_loss": 0}
    )
    return final_stats


def save_model(
    model: network.Model,
    optimizer,
    iteration: int,
    config: TrainingConfig,
    training_stats: list,
    timestamp: str | None = None,
) -> None:
    """Save model checkpoint with training statistics."""
    Path(config.checkpoint_dir).mkdir(exist_ok=True)

    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "training_stats": training_stats,
    }

    # Save model with timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = (
        Path(config.checkpoint_dir) / f"model_{config.game_module}_{timestamp}.pt"
    )
    torch.save(checkpoint, model_path)

    if training_stats and len(training_stats) > 0:
        latest_loss = training_stats[-1]["avg_loss"]
        print(f"💾 Model saved: {model_path.name} (Loss: {latest_loss:.4f})")
    else:
        print(f"💾 Model saved: {model_path.name}")


def load_checkpoint(
    model: network.Model, checkpoint_path: str, device: str | None = None
):
    """Load model from checkpoint.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model onto (e.g., 'cuda', 'cpu').
                If None, auto-detects (uses CUDA if available)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def training_loop(config: TrainingConfig) -> network.Model:
    """Main AlphaZero training loop with monitoring and checkpoints."""

    # Load game module
    game_module = load_game_module(config.game_module)

    # Setup directories
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    Path(config.log_dir).mkdir(exist_ok=True)

    # Compute timestamp for run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup device (GPU/CPU)
    if config.device:
        # Use specified device
        device = torch.device(config.device)
    else:
        # Auto-detect: use CUDA if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Initialize model
    (input_channels, board_height, board_width) = game_module.GameState.encoded_shape()
    num_possible_moves = game_module.GameState.num_possible_moves()

    model = network.Model(
        input_channels,
        board_height,
        board_width,
        num_possible_moves,
        num_filters=config.num_filters,
        num_residual_blocks=config.num_residual_blocks,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Training statistics tracking
    training_stats = []
    eval_stats = []

    print(f"\n🚀 Training for {config.iterations} iterations")
    print(f"📊 Self-play: {config.self_play_rounds} games per iteration")
    print(
        f"🎯 Evaluation: Every {config.eval_frequency} iterations ({config.eval_games} games)"
    )
    print(f"🧠 Network: {sum(p.numel() for p in model.parameters())} parameters")

    # Initial evaluation before training starts
    print("\n🎯 Running initial evaluation vs random baseline...")
    initial_eval_result = evaluation.evaluate_model_vs_random(
        model,
        game_module,
        num_games=config.eval_games,
        mcts_time_limit=config.mcts_time_limit,
    )

    eval_stats.append({"iteration": 0, **initial_eval_result})

    print(
        f"🏆 Initial evaluation result: {initial_eval_result['win_rate']:.1%} win rate "
        f"({initial_eval_result['wins']}-{initial_eval_result['draws']}-{initial_eval_result['losses']})"
    )

    # Main training loop with progress bar
    for iteration in tqdm(range(1, config.iterations + 1), desc="Training iterations"):
        # Self-play phase
        tqdm.write(f"\n🎮 Iteration {iteration}: Generating self-play data...")
        training_samples = self_play(model, config, game_module)
        tqdm.write(f"📚 Generated {len(training_samples)} training examples")

        # Training phase
        tqdm.write("🏋️ Training network...")
        train_stats = train_network(model, training_samples, config)

        # Record training statistics
        iteration_stats = {
            "iteration": iteration,
            "num_training_samples": len(training_samples),
            **train_stats,
        }
        training_stats.append(iteration_stats)

        # Display training progress
        tqdm.write(
            f"📈 Iteration {iteration} complete - "
            f"Loss: {train_stats['avg_loss']:.4f} "
            f"(Policy: {train_stats['avg_policy_loss']:.4f}, "
            f"Value: {train_stats['avg_value_loss']:.4f})"
        )

        # Evaluation phase
        if iteration % config.eval_frequency == 0:
            tqdm.write("\n🎯 Evaluating model vs random baseline...")
            eval_result = evaluation.evaluate_model_vs_random(
                model,
                game_module,
                num_games=config.eval_games,
                mcts_time_limit=config.mcts_time_limit,
            )

            eval_stats.append({"iteration": iteration, **eval_result})

            tqdm.write(
                f"🏆 Evaluation result: {eval_result['win_rate']:.1%} win rate "
                f"({eval_result['wins']}-{eval_result['draws']}-{eval_result['losses']})"
            )

        # Save model
        if iteration % config.checkpoint_frequency == 0:
            save_model(
                model, optimizer, iteration, config, training_stats, timestamp=timestamp
            )

        # Save logs (structured for easy plotting)
        logs = {
            "training": training_stats,
            "evaluation": eval_stats,
            "config": config.__dict__,
        }
        with open(
            Path(config.log_dir)
            / f"training_log_{config.game_module}_{timestamp}.json",
            "w",
        ) as f:
            json.dump(logs, f, indent=2)

    # Final model save
    save_model(
        model, optimizer, config.iterations, config, training_stats, timestamp=timestamp
    )
    print("\n✅ Training complete!")
    print(f"📁 Models saved in: {config.checkpoint_dir}")
    print(f"📊 Logs saved in: {config.log_dir}")

    return model


def main():
    """Entry point for training with command line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Train AlphaZero on various games")

    # Game selection
    parser.add_argument(
        "--game",
        required=True,
        help="Game module to train on (any of the games listed in games/)",
    )

    # Training loop parameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Save checkpoint every N iterations (default: 10)",
    )

    # Self-play parameters
    parser.add_argument(
        "--self-play-rounds",
        type=int,
        default=100,
        help="Number of self-play games per iteration (default: 100)",
    )
    parser.add_argument(
        "--mcts-time-limit",
        type=float,
        default=0.5,
        help="MCTS time limit in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--temperature-threshold",
        type=int,
        default=30,
        help="Number of moves to use exploration temperature (default: 30)",
    )
    parser.add_argument(
        "--temperature-exploration",
        type=float,
        default=1.0,
        help="Temperature for exploration phase (default: 1.0)",
    )
    parser.add_argument(
        "--temperature-exploitation",
        type=float,
        default=0.0,
        help="Temperature for exploitation phase (default: 0.0)",
    )

    # Network training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per iteration (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 weight decay (default: 1e-4)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=20,
        help="Evaluate every N iterations (default: 20)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=50,
        help="Number of evaluation games (default: 50)",
    )

    # Network architecture
    parser.add_argument(
        "--residual-blocks",
        type=int,
        default=4,
        help="Number of residual blocks (default: 4)",
    )
    parser.add_argument(
        "--num-filters",
        type=int,
        default=128,
        help="Number of convolutional filters (default: 128)",
    )

    # Paths
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints)",
    )
    parser.add_argument(
        "--log-dir", default="logs", help="Log directory (default: logs)"
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device to use for training (cpu/cuda). If not specified, uses CUDA if available.",
    )

    # Configuration file
    parser.add_argument("--config", type=str, help="Load configuration from JSON file")

    args = parser.parse_args()

    # Load from config file if specified
    if args.config:
        print(f"📝 Loading configuration from: {args.config}")
        config = TrainingConfig.load(args.config)
    else:
        # Create config from command line arguments
        config = TrainingConfig(
            game_module=args.game,
            iterations=args.iterations,
            checkpoint_frequency=args.checkpoint_freq,
            self_play_rounds=args.self_play_rounds,
            mcts_time_limit=args.mcts_time_limit,
            temperature_threshold=args.temperature_threshold,
            temperature_exploration=args.temperature_exploration,
            temperature_exploitation=args.temperature_exploitation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_frequency=args.eval_freq,
            eval_games=args.eval_games,
            num_residual_blocks=args.residual_blocks,
            num_filters=args.num_filters,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            device=args.device,
        )

    print(f"🎮 Training AlphaZero on: {config.game_module}")

    training_loop(config)


if __name__ == "__main__":
    main()
