#!/usr/bin/env python
"""
Main entry point for training the Hybrid Quantum GNN on MatBench exfoliation energy.

Usage:
    python experiments/run_training.py --config configs/default.yaml
    python experiments/run_training.py --config configs/default.yaml --fold 0
"""
import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import get_config
from src.utils.seed import set_seed
from src.training.matbench_runner import MatBenchRunner


def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid Quantum GNN for exfoliation energy prediction"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Run a single fold (0-4) instead of full 5-fold evaluation"
    )
    parser.add_argument(
        "--model-type", type=str, default=None, choices=["hybrid", "classical"],
        help="Override model type from config"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Load config
    config = get_config(args.config)

    # Apply CLI overrides
    if args.model_type is not None:
        config["model_type"] = args.model_type

    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Model type: {config.get('model_type', 'hybrid')}")

    # Set seed
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    # Run
    runner = MatBenchRunner(config)

    if args.fold is not None:
        logger.info(f"Running single fold: {args.fold}")
        results = runner.run_single_fold(args.fold)
    else:
        logger.info("Running full 5-fold MatBench evaluation")
        results = runner.run()

    logger.info(f"Results: {results}")
    return results


if __name__ == "__main__":
    main()
