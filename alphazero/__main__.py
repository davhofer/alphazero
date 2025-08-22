"""Main entry point for the alphazero package."""

import sys


def main():
    # Simple dispatcher without argument parsing to allow -h to pass through
    if len(sys.argv) < 2:
        print("AlphaZero - A Python implementation of the AlphaZero algorithm")
        print("\nUsage: python -m alphazero <command> [options]")
        print("\nAvailable commands:")
        print("  train    Train an AlphaZero model")
        print("  eval     Evaluate trained models")
        print("  plot     Plot training results")
        print("\nFor help on a specific command:")
        print("  python -m alphazero <command> -h")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Update sys.argv to remove the command, leaving just the script name and args
    sys.argv = [f"alphazero-{command}"] + sys.argv[2:]
    
    if command == "train":
        from .training import main as train_main
        train_main()
    elif command == "eval":
        from .evaluation import main as eval_main
        eval_main()
    elif command == "plot":
        from .plotting import main as plot_main
        plot_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, eval, plot")
        sys.exit(1)


if __name__ == "__main__":
    main()