"""
Sec71 package for training models and analyzing influence.
"""


def get_train_manager():
    from .train import TrainManager

    return TrainManager


def get_arg_parser():
    from .arg_parser import parse_arguments

    return parse_arguments


def get_device_config():
    from .device_config import get_device, get_real_gpu_index

    return get_device, get_real_gpu_index


def get_relabel_utils():
    from .relabel_utils import (
        handle_relabeling,
        generate_relabel_indices,
        save_relabel_indices,
        load_relabel_indices,
    )

    return (
        handle_relabeling,
        generate_relabel_indices,
        save_relabel_indices,
        load_relabel_indices,
    )


__all__ = [
    "get_train_manager",
    "get_arg_parser",
    "get_device_config",
    "get_relabel_utils",
]
