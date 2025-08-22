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


# TODO: replace game with the actual game we're learning
# TODO: how to do that elegantly?


def self_play(model: network.Model, config: TrainingConfig, game_module) -> list[dict]:
    """Let the model play games against itself and record encountered positions and outcomes."""
    training_examples = []

    for _ in tqdm(range(config.self_play_rounds), desc="Self-play games", leave=False):
        state = game_module.GameState.initial_state()
        game_history = []  # (state, policy) pairs for this game

        # Play one complete game
        while not state.is_terminal():
            root_node = mcts.Node(None, state, 1.0, None)
            policy = mcts.run_mcts(root_node, model, time_limit=config.mcts_time_limit)

            # Store state and MCTS policy for training
            game_history.append((state.encode(), policy))

            # Sample move from MCTS policy (with some randomness)
            policy_probs = policy.numpy()
            encoded_move = random.choices(
                range(len(policy_probs)), weights=policy_probs
            )[0]

            # Decode move and apply it
            move = game_module.Move.decode(encoded_move)
            state = state.apply_move(move)

        # Game is finished, get final outcome
        final_value = (
            state.get_value()
        )  # +1 if player 1 won, -1 if player 2 won, 0 for draw
        assert final_value is not None

        # Create training examples with final outcome as ground truth
        for i, (state_encoding, mcts_policy) in enumerate(game_history):
            # Convert objective result to current player's perspective at this position
            player_value = final_value if i % 2 == 0 else -final_value

            training_examples.append(
                {
                    "state": state_encoding,  # Input tensor
                    "policy": mcts_policy,  # Target policy
                    "value": player_value,  # Target value
                }
            )

    return training_examples


def train_network(
    model: network.Model, training_samples: list[dict], config: TrainingConfig
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
            pred_policies, pred_values = model(batch_states)

            # Calculate losses following AlphaZero paper: l = (z-v)^2 - π^T * log p + c * ||θ||^2

            # Value loss: (z-v)^2
            value_loss = value_loss_fn(pred_values, batch_target_values)

            # Policy loss: -π^T * log p (cross-entropy between target π and predicted p)
            # Add small epsilon to prevent log(0)
            log_pred_policies = torch.log(pred_policies + 1e-8)
            policy_loss = -torch.sum(
                batch_target_policies * log_pred_policies, dim=1
            ).mean()

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


def save_best_model(
    model: network.Model,
    optimizer,
    iteration: int,
    config: TrainingConfig,
    training_stats: list,
) -> None:
    """Save best model checkpoint with training statistics."""
    Path(config.checkpoint_dir).mkdir(exist_ok=True)

    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "training_stats": training_stats,
    }

    # Save best checkpoint (based on lowest loss)
    if training_stats and len(training_stats) > 0:
        latest_loss = training_stats[-1]["avg_loss"]
        best_path = Path(config.checkpoint_dir) / f"best_model_{config.game_module}.pt"

        # Check if this is the best model so far
        if not best_path.exists():
            torch.save(checkpoint, best_path)
            print(f"💾 First model saved! Loss: {latest_loss:.4f}")
        else:
            best_checkpoint = torch.load(best_path)
            best_stats = best_checkpoint["training_stats"]
            if best_stats and latest_loss < min(
                stat["avg_loss"] for stat in best_stats[-10:]
            ):
                torch.save(checkpoint, best_path)
                print(f"💾 New best model saved! Loss: {latest_loss:.4f}")


def load_checkpoint(model: network.Model, checkpoint_path: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def training_loop(config: TrainingConfig) -> None:
    """Main AlphaZero training loop with monitoring and checkpoints."""

    # Load game module
    game_module = load_game_module(config.game_module)

    # Setup directories
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    Path(config.log_dir).mkdir(exist_ok=True)

    # Save configuration
    config.save(Path(config.log_dir) / "config.json")

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

        # Save best model
        if iteration % config.checkpoint_frequency == 0:
            save_best_model(model, optimizer, iteration, config, training_stats)

        # Save logs (structured for easy plotting)
        logs = {
            "training": training_stats,
            "evaluation": eval_stats,
            "config": config.__dict__,
        }
        with open(
            Path(config.log_dir) / f"training_log_{config.game_module}.json", "w"
        ) as f:
            json.dump(logs, f, indent=2)

    # Final best model save
    save_best_model(model, optimizer, config.iterations, config, training_stats)
    print("\n✅ Training complete!")
    print(f"📁 Best model saved in: {config.checkpoint_dir}")
    print(f"📊 Logs saved in: {config.log_dir}")


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
