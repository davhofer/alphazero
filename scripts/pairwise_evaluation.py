#!/usr/bin/env python3
"""
Pairwise evaluation script for AlphaZero models.

This script loads multiple model checkpoints from a directory and evaluates them
against each other in a tournament-style comparison, both with and without MCTS.
Results are visualized with training parameters and heatmaps.
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import seaborn as sns

from alphazero.evaluation import load_model_from_checkpoint, play_games
from alphazero.players import NetworkMCTSPlayer, NetworkDirectPlayer
from alphazero.games import load_game_module


def load_checkpoint_info(checkpoint_path: Path) -> Dict:
    """Load checkpoint and extract training configuration."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    info = {
        "path": checkpoint_path,
        "name": checkpoint_path.stem,
        "iteration": checkpoint.get("iteration", 0),
        "config": checkpoint.get("config", {}),
    }

    return info


def run_pairwise_evaluation(
    models_info: List[Dict],
    game_module,
    num_games: int = 100,
    mcts_time_limit: float = 0.5,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run pairwise games between all models.

    Returns:
        Tuple of (results_without_mcts, results_with_mcts) as NxN matrices
        where element (i,j) is the win rate of model i against model j.
    """
    n_models = len(models_info)
    results_without_mcts = np.zeros((n_models, n_models))
    results_with_mcts = np.zeros((n_models, n_models))

    # Load all models
    models = []
    for info in models_info:
        model = load_model_from_checkpoint(str(info["path"]), game_module, device)
        models.append(model)

    print(f"\n🎮 Running pairwise evaluation with {n_models} models...")
    print(f"   Each pair plays {num_games} games")
    print(f"   MCTS time limit: {mcts_time_limit}s")

    # Run pairwise games
    for i in range(n_models):
        for j in range(i, n_models):  # Start from i to avoid redundant matchups
            if i == j:
                # Model vs itself - set to 0.5 (draw)
                results_without_mcts[i, j] = 0.5
                results_with_mcts[i, j] = 0.5
                continue

            model_i = models[i]
            model_j = models[j]

            # Games without MCTS (direct network policy)
            player_i = NetworkDirectPlayer(model_i, name=f"Model_{i}")
            player_j = NetworkDirectPlayer(model_j, name=f"Model_{j}")

            print(
                f"\n📊 Evaluating {models_info[i]['name']} vs {models_info[j]['name']} (Direct)"
            )
            results = play_games(player_i, player_j, game_module, num_games)
            results_without_mcts[i, j] = results["player1_win_rate"]
            results_without_mcts[j, i] = results[
                "player2_win_rate"
            ]  # Fill in the symmetric result

            # Games with MCTS
            player_i_mcts = NetworkMCTSPlayer(
                model_i, mcts_time_limit=mcts_time_limit, name=f"Model_{i}_MCTS"
            )
            player_j_mcts = NetworkMCTSPlayer(
                model_j, mcts_time_limit=mcts_time_limit, name=f"Model_{j}_MCTS"
            )

            print(
                f"📊 Evaluating {models_info[i]['name']} vs {models_info[j]['name']} (MCTS)"
            )
            results = play_games(player_i_mcts, player_j_mcts, game_module, num_games)
            results_with_mcts[i, j] = results["player1_win_rate"]
            results_with_mcts[j, i] = results[
                "player2_win_rate"
            ]  # Fill in the symmetric result

    return results_without_mcts, results_with_mcts


def create_visualization(
    models_info: List[Dict],
    results_without_mcts: np.ndarray,
    results_with_mcts: np.ndarray,
    output_path: Path,
):
    """Create comprehensive visualization with training parameters and result heatmaps."""

    n_models = len(models_info)

    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle("AlphaZero Model Pairwise Evaluation", fontsize=16, fontweight="bold")

    # 1. Training Parameters Table (top, spanning 3 columns)
    ax_params = fig.add_subplot(gs[0, :])
    ax_params.axis("off")

    # Extract key training parameters
    param_keys = [
        ("iterations", "Total Iterations"),
        ("self_play_rounds", "Self-Play Games"),
        ("mcts_time_limit", "MCTS Time (s)"),
        ("epochs", "Training Epochs"),
        ("batch_size", "Batch Size"),
        ("learning_rate", "Learning Rate"),
        ("num_filters", "Network Filters"),
        ("num_residual_blocks", "Residual Blocks"),
        ("eval_frequency", "Eval Frequency"),
        ("eval_games", "Eval Games"),
        ("temperature_threshold", "Temp Threshold"),
    ]

    # Create parameter table
    table_data = []
    n_params_to_show = 8  # Number of parameters to show in the table
    header = ["Model"] + [
        display for _, display in param_keys[:n_params_to_show]
    ]  # Show first n_params_to_show params in main table

    for i, info in enumerate(models_info):
        row = [f"Model {i}\n{info['name'][:20]}"]
        config = info["config"]
        for key, _ in param_keys[:n_params_to_show]:  # Match the header
            value = config.get(key, "N/A")
            if isinstance(value, float):
                row.append(f"{value:.4f}" if value < 1 else f"{value:.1f}")
            else:
                row.append(str(value))
        table_data.append(row)

    # Create table
    table = ax_params.table(
        cellText=table_data,
        colLabels=header,
        cellLoc="center",
        loc="center",
        bbox=[0.05, 0.3, 0.9, 0.6],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style the table - use actual table dimensions
    n_cols = len(header)
    n_rows = len(table_data)
    
    # Style header row
    for j in range(n_cols):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(weight="bold", color="white")

    # Style data rows with alternating colors
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#E7E6E6")

    # Add additional parameters as text below the table
    additional_text = "Additional Parameters:\n"
    for i, info in enumerate(models_info):
        config = info["config"]
        additional_text += f"Model {i}: "
        additional_params = []
        for key, display in param_keys[n_params_to_show:]:  # Show params not in the table
            if key in config:
                value = config[key]
                if isinstance(value, float):
                    formatted = f"{value:.4f}" if value < 1 else f"{value:.1f}"
                else:
                    formatted = str(value)
                additional_params.append(f"{display}={formatted}")
        additional_text += ", ".join(additional_params) + "\n"

    ax_params.text(
        0.5,
        0.1,
        additional_text,
        transform=ax_params.transAxes,
        fontsize=8,
        ha="center",
        va="top",
    )

    ax_params.set_title("Training Configuration Parameters", fontweight="bold", pad=20)

    # 2. Heatmap without MCTS (bottom left)
    ax_heatmap1 = fig.add_subplot(gs[1, 0])

    # Create labels
    model_labels = [f"Model {i}" for i in range(n_models)]

    # Plot heatmap
    sns.heatmap(
        results_without_mcts,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        xticklabels=model_labels,
        yticklabels=model_labels,
        vmin=0,
        vmax=1,
        ax=ax_heatmap1,
        cbar_kws={"label": "Win Rate"},
    )
    ax_heatmap1.set_title(
        "Results WITHOUT MCTS (Direct Network Policy)", fontweight="bold"
    )
    ax_heatmap1.set_xlabel("Opponent")
    ax_heatmap1.set_ylabel("Player")

    # 3. Heatmap with MCTS (bottom middle)
    ax_heatmap2 = fig.add_subplot(gs[1, 1])

    sns.heatmap(
        results_with_mcts,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        xticklabels=model_labels,
        yticklabels=model_labels,
        vmin=0,
        vmax=1,
        ax=ax_heatmap2,
        cbar_kws={"label": "Win Rate"},
    )
    ax_heatmap2.set_title("Results WITH MCTS", fontweight="bold")
    ax_heatmap2.set_xlabel("Opponent")
    ax_heatmap2.set_ylabel("Player")

    # 4. Summary statistics (bottom right)
    ax_summary = fig.add_subplot(gs[1, 2])
    ax_summary.axis("off")

    # Calculate average win rates for each model
    avg_without_mcts = []
    avg_with_mcts = []
    for i in range(n_models):
        # Calculate average excluding self-play (diagonal)
        mask = np.ones(n_models, dtype=bool)
        mask[i] = False
        avg_without_mcts.append(np.mean(results_without_mcts[i, mask]))
        avg_with_mcts.append(np.mean(results_with_mcts[i, mask]))

    summary_text = "Performance Summary:\n" + "=" * 30 + "\n\n"
    summary_text += "Average Win Rates:\n"
    summary_text += "(excluding self-play)\n\n"
    summary_text += "Without MCTS:\n"
    for i, (info, avg) in enumerate(zip(models_info, avg_without_mcts)):
        summary_text += f"  Model {i}: {avg:.2%}\n"

    summary_text += "\nWith MCTS:\n"
    for i, (info, avg) in enumerate(zip(models_info, avg_with_mcts)):
        summary_text += f"  Model {i}: {avg:.2%}\n"

    # Add improvement from MCTS
    summary_text += "\nMCTS Improvement:\n"
    for i in range(n_models):
        improvement = avg_with_mcts[i] - avg_without_mcts[i]
        summary_text += f"  Model {i}: {improvement:+.2%}\n"

    # Rank models
    summary_text += "\n" + "=" * 30 + "\n"
    summary_text += "Rankings (with MCTS):\n"
    rankings = sorted(enumerate(avg_with_mcts), key=lambda x: x[1], reverse=True)
    for rank, (i, score) in enumerate(rankings, 1):
        summary_text += f"  {rank}. Model {i}: {score:.2%}\n"

    ax_summary.text(
        0.05,
        0.95,
        summary_text,
        transform=ax_summary.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Visualization saved to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Run pairwise evaluation between multiple AlphaZero models"
    )

    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Directory containing model checkpoints to evaluate",
    )
    parser.add_argument(
        "--game",
        required=True,
        help="Game module to use (e.g., tictactoe, connect_four, chess)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games per model pair (default: 100)",
    )
    parser.add_argument(
        "--mcts-time-limit",
        type=float,
        default=0.5,
        help="MCTS time limit in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for computation (default: auto)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pt",
        help="Pattern for checkpoint files (default: *.pt)",
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"🖥️  Using device: {device}")

    # Find checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return

    checkpoint_files = sorted(checkpoint_dir.glob(args.pattern))
    if not checkpoint_files:
        print(f"❌ No checkpoint files found matching pattern: {args.pattern}")
        return

    print(f"\n📁 Found {len(checkpoint_files)} checkpoint files:")
    for cp in checkpoint_files:
        print(f"   - {cp.name}")

    # Load game module
    game_module = load_game_module(args.game)

    # Load checkpoint information
    models_info = []
    for cp_path in checkpoint_files:
        info = load_checkpoint_info(cp_path)
        models_info.append(info)
        print(f"\n📋 Model: {info['name']}")
        print(f"   Iteration: {info['iteration']}")
        if info["config"]:
            print(f"   Config keys: {', '.join(info['config'].keys())}")

    # Run pairwise evaluation
    results_without_mcts, results_with_mcts = run_pairwise_evaluation(
        models_info,
        game_module,
        num_games=args.num_games,
        mcts_time_limit=args.mcts_time_limit,
        device=device,
    )

    # Create visualization
    output_path = Path(args.checkpoint_dir) / "pairwise_evaluation.png"
    fig = create_visualization(
        models_info, results_without_mcts, results_with_mcts, output_path
    )

    # Show plot
    plt.show()

    print("\n✅ Pairwise evaluation complete!")


if __name__ == "__main__":
    main()

