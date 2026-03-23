"""
CLI entrypoint for the training pipeline.

Usage:
    python -m src --config configs/config_5x5.json
    python -m src --config configs/config_9x9.json
    python -m src --config configs/config_5x5.json --checkpoint-dir checkpoints/run_01
"""

import argparse
import logging

from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Quoridor training")
    parser.add_argument(
        "--config", default="configs/config_5x5.json",
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Directory for training checkpoints",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start training from scratch (ignore existing checkpoints)",
    )
    parser.add_argument(
        "--log-dir", default="logs",
        help="Directory for CSV / JSON metrics logs",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb remote logging",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)

    # Lazy imports so --help is fast and avoids heavy torch import
    from src.env.quoridor_env import QuoridorEnv, compute_action_space_size
    from src.model.network import QuoridorModel
    from src.mcts.self_play import training_loop

    env = QuoridorEnv(is_poc=cfg.is_poc,
                      max_walls_per_player=cfg.max_walls_per_player)
    net_cfg = cfg.network_config()

    model = QuoridorModel(
        board_size=cfg.board_size,
        action_space_size=compute_action_space_size(cfg.board_size),
        num_channels=net_cfg.get("num_channels", 64),
        num_res_blocks=net_cfg.get("num_res_blocks", 4),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    training_loop(
        env=env,
        model=model,
        config=cfg.training_config(),
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
        log_dir=args.log_dir,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
