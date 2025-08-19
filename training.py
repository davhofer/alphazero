# self-play
# record games/positions
# train network

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import mcts
import network
import game
import evaluation


@dataclass
class TrainingConfig:
    """Configuration for AlphaZero training."""

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

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            return cls(**json.load(f))


# TODO: replace game with the actual game we're learning
# TODO: how to do that elegantly?


def self_play(model: network.Model, config: TrainingConfig) -> list[dict]:
    """Let the model play games against itself and record encountered positions and outcomes."""
    training_examples = []

    for _ in tqdm(range(config.self_play_rounds), desc="Self-play games", leave=False):
        state = game.GameState.initial_state()
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
            move = game.Move.decode(encoded_move)
            state = state.apply_move(move)

        # Game is finished, get final outcome
        final_value = state.get_value()  # +1, 0, or -1
        assert final_value is not None

        # Create training examples with final outcome as ground truth
        for i, (state_encoding, mcts_policy) in enumerate(game_history):
            # Value from current player's perspective at this position
            # If it's an odd move number, flip the value (different player)
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

    # Prepare training data
    states = torch.stack([sample["state"] for sample in training_samples])
    target_policies = torch.stack([sample["policy"] for sample in training_samples])
    target_values = torch.tensor(
        [sample["value"] for sample in training_samples], dtype=torch.float32
    ).unsqueeze(1)

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


def save_checkpoint(
    model: network.Model,
    optimizer,
    iteration: int,
    config: TrainingConfig,
    training_stats: list,
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

    # Save latest checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_iter_{iteration}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint (based on lowest loss)
    if training_stats and len(training_stats) > 0:
        latest_loss = training_stats[-1]["avg_loss"]
        best_path = Path(config.checkpoint_dir) / "best_model.pt"

        # Check if this is the best model so far
        if not best_path.exists():
            torch.save(checkpoint, best_path)
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

    # Setup directories
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    Path(config.log_dir).mkdir(exist_ok=True)

    # Save configuration
    config.save(Path(config.log_dir) / "config.json")

    # Initialize model
    (input_channels, board_height, board_width) = game.GameState.encoded_shape()
    num_possible_moves = game.GameState.num_possible_moves()

    model = network.Model(
        input_channels,
        board_height,
        board_width,
        num_possible_moves,
        num_filters=config.num_filters,
        num_residual_blocks=config.num_residual_blocks,
    )

    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Training statistics tracking
    training_stats = []
    eval_stats = []

    print(f"🚀 Starting AlphaZero training for {config.iterations} iterations")
    print(f"📊 Self-play: {config.self_play_rounds} games per iteration")
    print(
        f"🎯 Evaluation: Every {config.eval_frequency} iterations ({config.eval_games} games)"
    )
    print(f"🧠 Network: {sum(p.numel() for p in model.parameters())} parameters")

    # Main training loop with progress bar
    for iteration in tqdm(range(1, config.iterations + 1), desc="Training iterations"):
        # Self-play phase
        tqdm.write(f"🎮 Iteration {iteration}: Generating self-play data...")
        training_samples = self_play(model, config)
        tqdm.write(f"📚 Generated {len(training_samples)} training examples")

        # Training phase
        tqdm.write("🏋️  Training network...")
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
            tqdm.write("🎯 Evaluating model vs random baseline...")
            eval_result = evaluation.evaluate_model_vs_random(
                model,
                num_games=config.eval_games,
                mcts_time_limit=config.mcts_time_limit,
            )

            eval_stats.append({"iteration": iteration, **eval_result})

            tqdm.write(
                f"🏆 Evaluation result: {eval_result['win_rate']:.1%} win rate "
                f"({eval_result['wins']}-{eval_result['draws']}-{eval_result['losses']})"
            )

        # Save checkpoints
        if iteration % config.checkpoint_frequency == 0:
            save_checkpoint(model, optimizer, iteration, config, training_stats)
            tqdm.write(f"💾 Checkpoint saved at iteration {iteration}")

        # Save logs (structured for easy plotting)
        logs = {
            "training": training_stats,
            "evaluation": eval_stats,
            "config": config.__dict__,
        }
        with open(Path(config.log_dir) / "training_log.json", "w") as f:
            json.dump(logs, f, indent=2)

    # Final checkpoint
    save_checkpoint(model, optimizer, config.iterations, config, training_stats)
    print("✅ Training complete!")
    print(f"📁 Checkpoints saved in: {config.checkpoint_dir}")
    print(f"📊 Logs saved in: {config.log_dir}")


def main():
    """Entry point for training."""
    config = TrainingConfig(
        iterations=100,
        checkpoint_frequency=10,
        self_play_rounds=100,
        mcts_time_limit=0.4,
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=0.0001,
        eval_frequency=20,
        eval_games=50,
        num_residual_blocks=4,
        num_filters=128,
    )

    # You can modify config here or load from file:
    # config = TrainingConfig.load("custom_config.json")

    training_loop(config)


if __name__ == "__main__":
    main()
