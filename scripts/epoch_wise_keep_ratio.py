#!/usr/bin/env python3
import argparse
import subprocess
from experiment import (
    get_train_manager,
    get_arg_parser,
    get_device_config,
    get_relabel_utils,
)

# This script is used for epoch-wise cleansing experiments, the main process includes:
# 1. Train the model (seed can be specified)
# 2. Calculate the influence score for each epoch
# 3. Conduct data cleansing experiments based on influence scores


def main():
    parser = argparse.ArgumentParser(description="Epoch-wise cleansing experiment")
    parser.add_argument("--target", type=str, help="Target dataset")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--save_dir", type=str, help="Save directory")
    parser.add_argument("--relabel", type=int, help="Relabel parameter")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, default 0")
    parser.add_argument(
        "--decay", type=str, default="False", help="Decay parameter, default False"
    )
    parser.add_argument(
        "--keep_ratio", type=int, default=90, help="Percentage of highest scores to keep, default 90"
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate, default 0.0005")
    parser.add_argument("--n_tr", type=int, help="Number of training samples")
    parser.add_argument("--n_val", type=int, help="Number of validation samples")
    parser.add_argument("--type", type=str, default="tim_all_epochs",
                       choices=["tim_all_epochs", "sgd"],
                       help="Influence calculation type: tim_all_epochs or sgd")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use, default 0")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Log level")
    args = parser.parse_args()

    # Step 1: Train the model, do not perform leave-one-out, save the model to the specified directory
    train_cmd = [
        "python",
        "-m",
        "experiment.train",
        "--target",
        args.target,
        "--model",
        args.model,
        "--no-loo",
        "--save_dir",
        args.save_dir,
        "--relabel",
        str(args.relabel),
        "--seed",
        str(args.seed),
        "--decay",
        str(args.decay),
        "--lr",
        str(args.lr),
        "--gpu",
        str(args.gpu),
        "--log_level",
        args.log_level,
    ]

    # Add custom training parameters
    if args.n_tr is not None:
        train_cmd.extend(["--n_tr", str(args.n_tr)])
    if args.n_val is not None:
        train_cmd.extend(["--n_val", str(args.n_val)])

    subprocess.run(train_cmd, check=True)

    # Step 2: Calculate the influence score for all epochs, results are saved in the specified directory
    infl_cmd = [
        "python",
        "-m",
        "experiment.infl",
        "--target",
        args.target,
        "--model",
        args.model,
        "--type",
        args.type,
        "--save_dir",
        args.save_dir,
        "--relabel",
        str(args.relabel),
        "--seed",
        str(args.seed),
        "--gpu",
        str(args.gpu),
        "--log_level",
        args.log_level,
    ]

    subprocess.run(infl_cmd, check=True)

    # Step 3: Conduct cleansing experiments based on influence scores
    cleansing_cmd = [
        "python",
        "-m",
        "experiment.exp_tim_cleansing",
        "--target",
        args.target,
        "--model",
        args.model,
        "--save_dir",
        args.save_dir,
        "--relabel",
        str(args.relabel),
        "--keep_ratio",
        str(args.keep_ratio),
        "--seed",
        str(args.seed),
        "--decay",
        str(args.decay),
        "--lr",
        str(args.lr),
        "--type",
        args.type,
        "--gpu",
        str(args.gpu),
        "--log_level",
        args.log_level,
    ]

    # Add custom training parameters
    if args.n_tr is not None:
        cleansing_cmd.extend(["--n_tr", str(args.n_tr)])
    if args.n_val is not None:
        cleansing_cmd.extend(["--n_val", str(args.n_val)])

    subprocess.run(cleansing_cmd, check=True)


if __name__ == "__main__":
    main()
