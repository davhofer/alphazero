"""Simple evaluation system for AlphaZero training."""

import argparse
import importlib
import time
import torch
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

import network
from players import Player, RandomPlayer, NetworkMCTSPlayer, NetworkDirectPlayer, GreedyPlayer, HumanPlayer


def play_games(player1: Player, player2: Player, game_module, num_games: int) -> Dict[str, int]:
    """Play n games between two players and return win counts."""

    wins_p1 = 0
    wins_p2 = 0
    draws = 0

    for game_num in tqdm(
        range(num_games), desc=f"{player1.name} vs {player2.name}", leave=False
    ):
        # Alternate who goes first
        if game_num % 2 == 0:
            first_player, second_player = player1, player2
        else:
            first_player, second_player = player2, player1

        # Play one game
        winner = play_single_game(first_player, second_player, game_module)

        # Count results (always from player1's perspective)
        if winner == player1.name:
            wins_p1 += 1
        elif winner == player2.name:
            wins_p2 += 1
        else:
            draws += 1

    return {
        "player1_wins": wins_p1,
        "player2_wins": wins_p2,
        "draws": draws,
        "total_games": num_games,
        "player1_win_rate": wins_p1 / num_games,
        "player2_win_rate": wins_p2 / num_games,
        "draw_rate": draws / num_games,
    }


def play_single_game(player1: Player, player2: Player, game_module) -> Optional[str]:
    """Play a single game and return winner name (or None for draw)."""

    state = game_module.GameState.initial_state()
    current_player = player1
    other_player = player2

    while not state.is_terminal():
        try:
            move = current_player.select_move(state)
            state = state.apply_move(move)

            # Switch players
            current_player, other_player = other_player, current_player

        except Exception:
            # If a player makes an invalid move, they lose
            print(
                f"ERROR: game lost due to illegal move by player {current_player.name}"
            )
            return other_player.name

    # Determine winner from final state
    final_value = state.get_value()
    if final_value is None or final_value == 0:
        return None  # Draw
    elif final_value > 0:
        return player1.name  # Player 1 wins
    else:
        return player2.name  # Player 2 wins


def evaluate_model_vs_random(
    model, game_module, num_games: int = 100, mcts_time_limit=0.5
) -> Dict[str, float]:
    """Evaluate trained model against random player."""

    alphazero_player = NetworkMCTSPlayer(
        model, mcts_time_limit=mcts_time_limit, name="AlphaZero"
    )
    random_player = RandomPlayer()

    results = play_games(alphazero_player, random_player, game_module, num_games)

    return {
        "win_rate": results["player1_win_rate"],
        "wins": results["player1_wins"],
        "losses": results["player2_wins"],
        "draws": results["draws"],
        "total_games": results["total_games"],
    }


def load_model_from_checkpoint(checkpoint_path: str, game_module) -> network.Model:
    """Load a model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model parameters from checkpoint or use defaults
    input_channels, board_height, board_width = game_module.GameState.encoded_shape()
    num_possible_moves = game_module.GameState.num_possible_moves()
    
    # Try to get architecture params from checkpoint, with fallbacks
    num_filters = checkpoint.get('num_filters', 128)
    num_residual_blocks = checkpoint.get('num_residual_blocks', 4)
    
    model = network.Model(
        input_channels=input_channels,
        board_height=board_height, 
        board_width=board_width,
        num_possible_moves=num_possible_moves,
        num_filters=num_filters,
        num_residual_blocks=num_residual_blocks
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def create_player(player_type: str, game_module, model_path: Optional[str] = None, 
                  mcts_time_limit: float = 1.0, name: Optional[str] = None) -> Player:
    """Create a player of the specified type."""
    
    if player_type == "random":
        player = RandomPlayer()
        if name:
            player.name = name
        return player
    
    elif player_type == "greedy":
        return GreedyPlayer(name=name or "Greedy")
    
    elif player_type == "human":
        return HumanPlayer(name=name or "Human")
    
    elif player_type == "network-mcts":
        if not model_path:
            raise ValueError("network-mcts player requires --model-path")
        model = load_model_from_checkpoint(model_path, game_module)
        return NetworkMCTSPlayer(model, mcts_time_limit=mcts_time_limit, name=name or f"MCTS({Path(model_path).stem})")
    
    elif player_type == "network-direct":
        if not model_path:
            raise ValueError("network-direct player requires --model-path")
        model = load_model_from_checkpoint(model_path, game_module)
        return NetworkDirectPlayer(model, name=name or f"Direct({Path(model_path).stem})")
    
    else:
        raise ValueError(f"Unknown player type: {player_type}. Available: random, greedy, human, network-mcts, network-direct")


def main():
    parser = argparse.ArgumentParser(description="Evaluate players against each other")
    
    # Game selection
    parser.add_argument("--game", default="tictactoe", help="Game module to use (default: tictactoe)")
    
    # Player configuration
    parser.add_argument("--player1", required=True, choices=["random", "greedy", "human", "network-mcts", "network-direct"],
                       help="Type of player 1")
    parser.add_argument("--player2", required=True, choices=["random", "greedy", "human", "network-mcts", "network-direct"],
                       help="Type of player 2")
    parser.add_argument("--player1-model", help="Model checkpoint path for player 1 (if network player)")
    parser.add_argument("--player2-model", help="Model checkpoint path for player 2 (if network player)")
    parser.add_argument("--player1-name", help="Custom name for player 1")
    parser.add_argument("--player2-name", help="Custom name for player 2")
    
    # Game configuration
    parser.add_argument("--games", type=int, default=100, help="Number of games to play (default: 100)")
    parser.add_argument("--mcts-time-limit", type=float, default=1.0, 
                       help="MCTS time limit in seconds (default: 1.0)")
    
    args = parser.parse_args()
    
    # Load game module
    try:
        # Prepend "games." to the module name
        full_module_name = f"games.{args.game}"
        game_module = importlib.import_module(full_module_name)
        print(f"🎮 Loaded game module: {args.game}")
    except ImportError as e:
        print(f"❌ Failed to import game module '{args.game}': {e}")
        return
    
    # Create players
    try:
        player1 = create_player(args.player1, game_module, args.player1_model, 
                               args.mcts_time_limit, args.player1_name or f"Player1({args.player1})")
        player2 = create_player(args.player2, game_module, args.player2_model,
                               args.mcts_time_limit, args.player2_name or f"Player2({args.player2})")
        
        print(f"👤 Player 1: {player1.name} ({args.player1})")
        print(f"👤 Player 2: {player2.name} ({args.player2})")
        print(f"🎯 Playing {args.games} games...")
        
    except Exception as e:
        print(f"❌ Error creating players: {e}")
        return
    
    # Play games
    try:
        results = play_games(player1, player2, game_module, args.games)
        
        # Print results
        print(f"\n📊 Results after {results['total_games']} games:")
        print(f"  {player1.name}: {results['player1_wins']} wins ({results['player1_win_rate']:.1%})")
        print(f"  {player2.name}: {results['player2_wins']} wins ({results['player2_win_rate']:.1%})")
        print(f"  Draws: {results['draws']} ({results['draw_rate']:.1%})")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return


if __name__ == "__main__":
    main()

