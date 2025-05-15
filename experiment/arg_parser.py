import argparse

def parse_arguments():
    """
    Parse command-line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Train Models & Save")
    parser.add_argument("--target", default="adult", type=str, help="target data")
    parser.add_argument("--model", default="logreg", type=str, help="model type")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--gpu", default=0, type=int, help="gpu index")
    parser.add_argument("--n_tr", type=int, help="number of training samples")
    parser.add_argument("--n_val", type=int, help="number of validation samples")
    parser.add_argument("--n_test", type=int, help="number of test samples")
    parser.add_argument("--num_epoch", type=int, help="number of epochs")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument(
        "--save_dir", type=str, help="directory to save models and results"
    )
    parser.add_argument(
        "--no-loo",
        action="store_false",
        dest="compute_counterfactual",
        help="Disable the computation of counterfactual models (leave-one-out).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    parser.add_argument(
        "--relabel", type=int, help="percentage of training data to relabel"
    )
    parser.add_argument(
        "--relabel_csv",
        type=str,
        help="CSV file containing indices of samples to relabel",
    )
    parser.add_argument(
        "--init_model",
        type=str,
        help="Path to the initialization model file (will use the last model in the list)",
    )
    parser.add_argument(
        "--no-recording",
        action="store_false",
        dest="save_recording",
        help="Disable saving of full model recordings (.dat file). Only save metrics CSV.",
    )
    parser.add_argument(
        "--steps-only",
        action="store_true",
        dest="steps_only",
        help="Only record steps without recording epochs and the overall model.",
    )
    parser.add_argument(
        "--decay",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="Enable learning rate decay: True or False (default: False)",
    )

    parser.set_defaults(compute_counterfactual=True, save_recording=True, steps_only=False)

    return parser.parse_args() 
