"""Plotting utilities for AlphaZero training results."""

import json
import matplotlib.pyplot as plt
import matplotlib.style as style
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


def load_training_data(log_path: str) -> Dict[str, Any]:
    """Load training log data from JSON file."""
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_training_loss(data: Dict[str, Any], save_path: Optional[str] = None, show: bool = True):
    """Plot training loss curves."""
    training = data["training"]
    
    if not training:
        print("No training data found!")
        return
    
    # Extract data
    iterations = [x["iteration"] for x in training]
    total_loss = [x["avg_loss"] for x in training]
    policy_loss = [x["avg_policy_loss"] for x in training]
    value_loss = [x["avg_value_loss"] for x in training]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('AlphaZero Training Loss', fontsize=16, fontweight='bold')
    
    # Plot total loss
    ax1.plot(iterations, total_loss, 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot component losses
    ax2.plot(iterations, policy_loss, 'r-', linewidth=2, label='Policy Loss')
    ax2.plot(iterations, value_loss, 'g-', linewidth=2, label='Value Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Component Losses')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training loss plot saved to: {save_path}")
    
    if show:
        plt.show()


def plot_evaluation_results(data: Dict[str, Any], save_path: Optional[str] = None, show: bool = True):
    """Plot evaluation win rate over training."""
    evaluation = data["evaluation"]
    
    if not evaluation:
        print("No evaluation data found!")
        return
    
    # Extract data
    iterations = [x["iteration"] for x in evaluation]
    win_rates = [x["win_rate"] for x in evaluation]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot win rate
    plt.plot(iterations, win_rates, 'o-', linewidth=2, markersize=6, 
             color='darkgreen', label='Win Rate vs Random')
    
    # Add horizontal reference lines
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    plt.axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='Perfect (100%)')
    
    plt.xlabel('Training Iteration')
    plt.ylabel('Win Rate')
    plt.title('AlphaZero Evaluation: Win Rate vs Random Player', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis limits
    plt.ylim(0, 1.05)
    
    # Add percentage labels on y-axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🏆 Evaluation plot saved to: {save_path}")
    
    if show:
        plt.show()


def plot_training_summary(data: Dict[str, Any], save_path: Optional[str] = None, show: bool = True):
    """Create a comprehensive training summary plot."""
    training = data["training"]
    evaluation = data["evaluation"]
    config = data.get("config", {})
    
    if not training:
        print("No training data found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('AlphaZero Training Summary', fontsize=16, fontweight='bold')
    
    # Training data
    iterations = [x["iteration"] for x in training]
    total_loss = [x["avg_loss"] for x in training]
    policy_loss = [x["avg_policy_loss"] for x in training]
    value_loss = [x["avg_value_loss"] for x in training]
    
    # Plot 1: Total Loss
    axes[0, 0].plot(iterations, total_loss, 'b-', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Component Losses
    axes[0, 1].plot(iterations, policy_loss, 'r-', linewidth=2, label='Policy')
    axes[0, 1].plot(iterations, value_loss, 'g-', linewidth=2, label='Value')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Win Rate (if evaluation data exists)
    if evaluation:
        eval_iterations = [x["iteration"] for x in evaluation]
        win_rates = [x["win_rate"] for x in evaluation]
        
        axes[1, 0].plot(eval_iterations, win_rates, 'o-', linewidth=2, markersize=6, color='darkgreen')
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Win Rate vs Random')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    else:
        axes[1, 0].text(0.5, 0.5, 'No evaluation data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Win Rate vs Random')
    
    # Plot 4: Training Configuration Summary
    axes[1, 1].axis('off')
    config_text = "Training Configuration:\n\n"
    
    key_configs = {
        'iterations': 'Total Iterations',
        'self_play_rounds': 'Self-play Games/Iter',
        'epochs': 'Training Epochs',
        'batch_size': 'Batch Size',
        'learning_rate': 'Learning Rate',
        'eval_frequency': 'Eval Frequency',
        'eval_games': 'Eval Games'
    }
    
    for key, display_name in key_configs.items():
        if key in config:
            config_text += f"{display_name}: {config[key]}\n"
    
    axes[1, 1].text(0.05, 0.95, config_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('Configuration')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training summary plot saved to: {save_path}")
    
    if show:
        plt.show()


def create_all_plots(log_path: str, output_dir: str = "plots", show: bool = True):
    """Create all training plots and save them."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data
    print(f"📊 Loading training data from: {log_path}")
    data = load_training_data(log_path)
    
    # Generate plots
    plot_training_loss(
        data, 
        save_path=Path(output_dir) / "training_loss.png",
        show=show
    )
    
    plot_evaluation_results(
        data,
        save_path=Path(output_dir) / "evaluation_results.png", 
        show=show
    )
    
    plot_training_summary(
        data,
        save_path=Path(output_dir) / "training_summary.png",
        show=show
    )
    
    print(f"✅ All plots saved to: {output_dir}")


def main():
    """Command line interface for plotting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create training plots from AlphaZero logs")
    parser.add_argument("log_path", help="Path to training_log.json file")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots interactively")
    
    args = parser.parse_args()
    
    create_all_plots(args.log_path, args.output_dir, show=not args.no_show)


if __name__ == "__main__":
    main()