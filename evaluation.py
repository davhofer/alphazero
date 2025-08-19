"""Simple evaluation system for AlphaZero training."""

import time
from typing import Dict, Optional
from tqdm import tqdm

import game
from players import Player, RandomPlayer, NetworkMCTSPlayer


def play_games(player1: Player, player2: Player, num_games: int) -> Dict[str, int]:
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
        winner = play_single_game(first_player, second_player)

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


def play_single_game(player1: Player, player2: Player) -> Optional[str]:
    """Play a single game and return winner name (or None for draw)."""

    state = game.GameState.initial_state()
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
    model, num_games: int = 100, mcts_time_limit=0.5
) -> Dict[str, float]:
    """Evaluate trained model against random player."""

    alphazero_player = NetworkMCTSPlayer(
        model, mcts_time_limit=mcts_time_limit, name="AlphaZero"
    )
    random_player = RandomPlayer()

    results = play_games(alphazero_player, random_player, num_games)

    return {
        "win_rate": results["player1_win_rate"],
        "wins": results["player1_wins"],
        "losses": results["player2_wins"],
        "draws": results["draws"],
        "total_games": results["total_games"],
    }

