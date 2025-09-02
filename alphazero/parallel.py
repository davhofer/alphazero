"""Parallel processing for AlphaZero self-play and evaluation."""

import multiprocessing as mp
from multiprocessing import Pool, Queue
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import traceback
import random
import os
import resource

from . import mcts
from . import network
from . import training
from .players import Player, RandomPlayer, NetworkMCTSPlayer, NetworkDirectPlayer
from .games import load_game_module

# Configure multiprocessing for robustness
# This avoids "received 0 items of ancdata" errors
os.environ["TORCH_MULTIPROCESSING_SET_SHARING_STRATEGY"] = "file_system"

# Try to increase file descriptor limit
try:
    # Get current limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Try to set to a higher value (up to hard limit)
    desired = min(8192, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
except:
    # May fail if not permitted, that's okay
    print("Unable to set resource limits for parallel processing...")
    pass


@dataclass
class WorkerConfig:
    """Configuration passed to worker processes."""

    model_state_dict: dict  # Model weights
    model_args: dict  # Model architecture parameters
    game_module_name: str  # Game module to import
    device: str = "cpu"  # Device for inference (usually CPU for workers)
    seed: Optional[int] = None  # Random seed for reproducibility


def init_worker(worker_config: WorkerConfig, worker_id: int):
    """Initialize worker process with model and game module."""
    global _worker_model, _worker_game_module, _worker_id

    # Limit number of threads per worker to prevent oversubscription
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # Get actual worker ID from multiprocessing

    process = mp.current_process()
    actual_worker_id = (
        int(process.name.split("-")[-1]) if "-" in process.name else worker_id
    )

    # Set random seed for reproducibility with unique seed per worker
    if worker_config.seed is not None:
        seed = worker_config.seed + actual_worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        # Use a random seed if not specified
        import os

        seed = os.getpid() + actual_worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Import game module

    _worker_game_module = load_game_module(worker_config.game_module_name)

    # Create and load model
    _worker_model = network.Model(**worker_config.model_args)
    _worker_model.load_state_dict(worker_config.model_state_dict)
    _worker_model.to(worker_config.device)
    _worker_model.eval()

    _worker_id = worker_id


def self_play_single_game(config_dict: dict) -> List[dict]:
    """
    Play a single self-play game. Runs in worker process.

    Args:
        config_dict: Dictionary with training configuration

    Returns:
        List of training examples from the game
    """
    try:
        # Access global worker resources
        global _worker_model, _worker_game_module

        state = _worker_game_module.GameState.initial_state()
        game_history = []  # (state, policy, current_player) tuples
        move_count = 0

        # Play one complete game
        while not state.is_terminal():
            root_node = mcts.Node(None, state, 1.0, None)

            # Run MCTS with training mode
            policy = mcts.run_mcts(
                root_node,
                _worker_model,
                time_limit=config_dict["mcts_time_limit"],
                training=True,
                dirichlet_epsilon=config_dict["dirichlet_epsilon"],
                dirichlet_alpha=config_dict["dirichlet_alpha"],
            )

            # Store state, MCTS policy, and current player
            game_history.append((state.encode(), policy, state.current_player))

            # Select temperature based on move count
            temperature = (
                config_dict["temperature_exploration"]
                if move_count < config_dict["temperature_threshold"]
                else config_dict["temperature_exploitation"]
            )

            # Apply temperature and sample move
            if temperature == 0:
                # Deterministic selection
                legal_moves = state.get_legal_moves()
                best_move = None
                best_prob = -1
                for move in legal_moves:
                    prob = policy[move.encode()].item()
                    if prob > best_prob:
                        best_prob = prob
                        best_move = move
                selected_move = best_move
            else:
                # Stochastic selection with temperature
                temp_policy = training.apply_temperature(policy, temperature)
                legal_moves = state.get_legal_moves()

                # Sample from temperature-adjusted policy
                legal_probs = []
                for move in legal_moves:
                    legal_probs.append(temp_policy[move.encode()].item())

                # Normalize and sample
                total_prob = sum(legal_probs)
                if total_prob > 0:
                    legal_probs = [p / total_prob for p in legal_probs]
                    selected_move = np.random.choice(legal_moves, p=legal_probs)
                else:
                    selected_move = random.choice(legal_moves)

            state = state.apply_move(selected_move)
            move_count += 1

        # Get final game value
        final_value = state.get_value()

        # Create training examples with value from each position's perspective
        training_examples = []
        for state_tensor, policy, player in game_history:
            # Value is from the perspective of the player to move
            value_from_perspective = final_value * player
            # Ensure tensors are detached and on CPU to avoid multiprocessing issues
            training_examples.append(
                {
                    "state": state_tensor.cpu()
                    if isinstance(state_tensor, torch.Tensor)
                    else state_tensor,
                    "policy": policy.cpu()
                    if isinstance(policy, torch.Tensor)
                    else policy,
                    "value": value_from_perspective,
                }
            )

        return training_examples

    except Exception as e:
        print(f"Error in self-play worker: {e}")
        traceback.print_exc()
        return []


def evaluation_game(args: Tuple[str, str, dict]) -> int:
    """
    Play a single evaluation game. Runs in worker process.

    Args:
        args: Tuple of (player1_type, player2_type, game_config)

    Returns:
        Game result: 1 for player1 win, -1 for player2 win, 0 for draw
    """
    try:
        global _worker_model, _worker_game_module

        player1_type, player2_type, game_config = args

        # Create players based on type
        if player1_type == "network_mcts":
            player1 = NetworkMCTSPlayer(
                _worker_model,
                mcts_time_limit=game_config["mcts_time_limit"],
                name="AlphaZero",
                temperature=0.0,  # Deterministic for evaluation
            )
        elif player1_type == "network_direct":
            player1 = NetworkDirectPlayer(_worker_model, name="NetworkDirect")
        elif player1_type == "random":
            player1 = RandomPlayer()
        else:
            raise ValueError(f"Unknown player type: {player1_type}")

        if player2_type == "network_mcts":
            player2 = NetworkMCTSPlayer(
                _worker_model,
                mcts_time_limit=game_config["mcts_time_limit"],
                name="AlphaZero2",
                temperature=0.0,
            )
        elif player2_type == "network_direct":
            player2 = NetworkDirectPlayer(_worker_model, name="NetworkDirect2")
        elif player2_type == "random":
            player2 = RandomPlayer()
        else:
            raise ValueError(f"Unknown player type: {player2_type}")

        # Alternate who goes first based on game number
        if game_config.get("game_num", 0) % 2 == 0:
            first_player, second_player = player1, player2
            first_is_p1 = True
        else:
            first_player, second_player = player2, player1
            first_is_p1 = False

        # Play the game
        state = _worker_game_module.GameState.initial_state()
        current_player = first_player
        other_player = second_player

        while not state.is_terminal():
            move = current_player.select_move(state)
            state = state.apply_move(move)
            current_player, other_player = other_player, current_player

        # Get game result
        final_value = state.get_value()

        if final_value is None or final_value == 0:
            return 0  # Draw
        elif final_value > 0:
            # Player 1 (first to move in initial state) wins
            return 1 if first_is_p1 else -1
        else:
            # Player 2 wins
            return -1 if first_is_p1 else 1

    except Exception as e:
        print(f"Error in evaluation worker: {e}")
        traceback.print_exc()
        return 0  # Treat errors as draws


def parallel_self_play(
    model: network.Model,
    config: "training.TrainingConfig",
    game_module,
    num_workers: int = 4,
) -> List[dict]:
    """
    Parallel self-play data generation.

    Args:
        model: Neural network model
        config: Training configuration
        game_module: Game module
        num_workers: Number of parallel workers

    Returns:
        List of training examples from all games
    """
    # Prepare worker configuration
    # IMPORTANT: Move model weights to CPU before passing to workers
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    worker_config = WorkerConfig(
        model_state_dict=model_state_dict,
        model_args={
            "input_channels": game_module.GameState.encoded_shape()[0],
            "board_height": game_module.GameState.encoded_shape()[1],
            "board_width": game_module.GameState.encoded_shape()[2],
            "num_possible_moves": game_module.GameState.num_possible_moves(),
            "num_filters": config.num_filters,
            "num_residual_blocks": config.num_residual_blocks,
        },
        game_module_name=config.game_module,
        device="cpu",  # Workers use CPU
        seed=42,  # For reproducibility
    )

    # Create config dict for workers
    config_dict = {
        "mcts_time_limit": config.mcts_time_limit,
        "temperature_threshold": config.temperature_threshold,
        "temperature_exploration": config.temperature_exploration,
        "temperature_exploitation": config.temperature_exploitation,
        "dirichlet_epsilon": config.dirichlet_epsilon,
        "dirichlet_alpha": config.dirichlet_alpha,
    }

    # Create pool of workers
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(worker_config, 0),  # Worker ID will be set by the pool
    ) as pool:
        # Generate games in parallel
        config_dicts = [config_dict for _ in range(config.self_play_rounds)]

        # Use imap_unordered for better progress tracking
        training_examples = []
        with tqdm(
            total=config.self_play_rounds, desc="Self-play games", leave=False
        ) as pbar:
            for game_examples in pool.imap_unordered(
                self_play_single_game, config_dicts
            ):
                training_examples.extend(game_examples)
                pbar.update(1)

    return training_examples


def parallel_evaluate(
    model: network.Model,
    game_module,
    num_games: int,
    opponent_type: str = "random",
    mcts_time_limit: float = 0.5,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Parallel evaluation of model against an opponent.

    Args:
        model: Neural network model to evaluate
        game_module: Game module
        num_games: Number of games to play
        opponent_type: Type of opponent ("random", "network_direct")
        mcts_time_limit: Time limit for MCTS
        num_workers: Number of parallel workers

    Returns:
        Dictionary with evaluation results
    """
    # Prepare worker configuration
    # IMPORTANT: Move model weights to CPU before passing to workers
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    worker_config = WorkerConfig(
        model_state_dict=model_state_dict,
        model_args={
            "input_channels": game_module.GameState.encoded_shape()[0],
            "board_height": game_module.GameState.encoded_shape()[1],
            "board_width": game_module.GameState.encoded_shape()[2],
            "num_possible_moves": game_module.GameState.num_possible_moves(),
            "num_filters": model.num_filters,
            "num_residual_blocks": model.num_residual_blocks,
        },
        game_module_name=game_module.__name__.split(".")[-1],  # Extract module name
        device="cpu",
        seed=42,
    )

    # Prepare game configurations
    game_configs = []
    for i in range(num_games):
        game_configs.append(
            (
                "network_mcts",  # Player 1 type
                opponent_type,  # Player 2 type
                {"mcts_time_limit": mcts_time_limit, "game_num": i},
            )
        )

    # Run games in parallel
    with Pool(
        processes=num_workers, initializer=init_worker, initargs=(worker_config, 0)
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(evaluation_game, game_configs),
                total=num_games,
                desc=f"Evaluation vs {opponent_type}",
                leave=False,
            )
        )

    # Count results
    wins = sum(1 for r in results if r == 1)
    losses = sum(1 for r in results if r == -1)
    draws = sum(1 for r in results if r == 0)

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_games": num_games,
        "win_rate": wins / num_games if num_games > 0 else 0,
        "loss_rate": losses / num_games if num_games > 0 else 0,
        "draw_rate": draws / num_games if num_games > 0 else 0,
    }

