# TIM: Time-Varying Influence Measurement Framework

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

## Introduction

TIM (Time-Varying Influence Measurement) is a comprehensive framework for analyzing how the influence of training examples on model performance evolves throughout the training process. Unlike traditional methods that only consider the final model state, TIM tracks the variation of influence throughout the training process, enabling more accurate identification of harmful or mislabeled samples.

This repository contains the official implementation of the paper: "TIM: Time-Varying Influence Measurement of Training Data".

## Key Features

- **Dynamic Influence Tracking**: Measure data influence at any point during training
- **Multi-dimensional Analysis**: Project parameter influences onto task-relevant dimensions (loss, predictions, feature importance)
- **Multiple Influence Methods**: Support for SGD Influence, TracIn, and our novel TIM approach
- **Data Cleansing**: Identify and remove harmful training samples to improve model performance
- **Training Dynamics**: Observe how different samples' influence evolves throughout training
- **Comprehensive Toolset**: Pre-configured datasets, models, and analysis pipelines

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Time-Varying-Influence-Measurement/TIM.git
cd TIM

# Create and activate virtual environment with pixi
pixi install
pixi shell

# Train the model and calculate influences
pixi run python -m experiment.train --target mnist --model dnn --save_dir result/example --seed 0 --no-loo
pixi run python -m experiment.infl --target mnist --model dnn --type tim_all_epochs --save_dir result/example
pixi run python -m experiment.exp_tim_cleansing --target mnist --model dnn --save_dir result/example
```

## Usage Examples

### Identifying Mislabeled Examples

```bash
# Calculate influences with TIM
pixi run python -m experiment.train --target mnist --model dnn --save_dir result/harmful_samples --relabel 10 --no-loo
pixi run python -m experiment.infl --target mnist --model dnn --type tim_all_epochs --save_dir result/harmful_samples

# Visualize results
pixi run python -m scripts.plot_tim_cleansing_epoch_wise --single_folder result/harmful_samples --relabel_value 10
```

### Comparing Influence Methods

```bash
# Calculate influences with different methods
pixi run python -m experiment.train --target mnist --model dnn --save_dir result/compare_methods --relabel 30 --no-loo

# SGD Influence
pixi run python -m experiment.infl --target mnist --model dnn --type sgd --save_dir result/compare_methods

# TracIn Influence
pixi run python -m experiment.infl --target mnist --model dnn --type tracin --save_dir result/compare_methods

# TIM Influence
pixi run python -m experiment.infl --target mnist --model dnn --type tim_all_epochs --save_dir result/compare_methods

# Compare cleansing effects with different keep_ratios
pixi run python -m scripts.epoch_wise_keep_ratio --target mnist --model dnn --save_dir result/compare_methods --decay True --relabel 30 --keep_ratio 70 --type tim_all_epochs
pixi run python -m scripts.epoch_wise_keep_ratio --target mnist --model dnn --save_dir result/compare_methods --decay True --relabel 30 --keep_ratio 60 --type sgd

# Visualize comparison results
pixi run python -m scripts.plot_tim_cleansing_epoch_wise --single_folder result/compare_methods --relabel_value 30 --keep_ratio 70
pixi run python -m scripts.plot_tim_cleansing_final_acc_bar
```

### Batch Experiments and Learning Rate Optimization

```bash
# Run batch experiments (cleansing effects with different keep_ratios)
pixi run bash scripts/epoch_wise_cleansing.sh mnist dnn epoch_wise_cleansing_keep_ratio_030 0 0 30
pixi run bash scripts/epoch_wise_cleansing.sh mnist dnn epoch_wise_cleansing_keep_ratio_040 0 0 40
pixi run bash scripts/epoch_wise_cleansing.sh mnist dnn epoch_wise_cleansing_keep_ratio_050 0 0 50

# Learning rate experiments
pixi run python -m scripts.epoch_wise_keep_ratio --target mnist --model dnn --save_dir result/lr_experiment --decay True --relabel 30 --keep_ratio 70 --lr 0.015 --type tim_all_epochs
```

## System Architecture

TIM framework consists of five major subsystems:

1. **Data Module System**: Handles dataset loading, preprocessing, and management
   - Supports MNIST (binary), 20 Newsgroups, Adult, CIFAR, EMNIST

2. **Network Module System**: Manages neural network architectures and model creation
   - Supports logistic regression, DNN, CNN, ResNet, ViT, MobileNetV2

3. **Training System**: Handles model training procedures and checkpoint saving

4. **Influence Calculation System**: Core of TIM framework, calculates training sample influences
   - Supports SGD Influence, TracIn, TIM-Last, TIM methods

5. **Data Cleansing System**: Identifies and removes harmful training samples

## Output Files

The framework generates various output files in the specified save directory:

```bash
/save_dir/
├── records/
│   ├── init_.pt                # Initial model state
│   ├── epoch_.pt               # Per-epoch model states
│   ├── step_.pt                # Per-step model states
│   └── permutation_epoch_.npy  # Permutations used in training
├── global_info_.json           # Training configuration
├── infl_.csv                   # Influence scores (CSV)
├── infl_.dat                   # Influence scores (binary)
├── kept_indices_.csv           # Indices of kept samples
├── relabel_overlap_.csv        # Overlap with relabeled data
└── cleansed_.csv               # Cleansing results
```

## Configuration

| Parameter          | Description                   | Default        |
|--------------------|-------------------------------|----------------|
| key                | Dataset keyword               | Required       |
| model_type         | Model type                    | Required       |
| seed               | Random seed                   | 0              |
| gpu                | GPU device index              | 0              |
| save_dir           | Save directory                | ./results      |
| relabel_percentage | Relabeled data percentage     | None           |
| keep_ratio         | Cleansing keep ratio          | 90             |
| infl_type          | Influence measurement method  | 'tim_all_epochs'|

## Command Line Arguments

### Training Parameters (experiment.train)

| Parameter | Description | Default |
|-------------------|------------------------------------------|---------------|
| --target | Target dataset (mnist, adult, etc.) | Required |
| --model | Model type (dnn, logreg, cnn, etc.) | Required |
| --seed | Random seed | 0 |
| --save_dir | Directory to save results | Required |
| --csv_path | Custom CSV data path | None |
| --n_tr | Number of training samples | As specified in config |
| --n_val | Number of validation samples | As specified in config |
| --n_test | Number of test samples | As specified in config |
| --num_epoch | Number of training epochs | As specified in config |
| --batch_size | Batch size | As specified in config |
| --lr | Learning rate | As specified in config |
| --decay | Whether to use learning rate decay | True |
| --compute_counterfactual | Whether to compute counterfactual model | False |
| --init_model | Initial model path | None |
| --save_recording | Whether to save model records | True |
| --steps_only | Only save step checkpoints | False |
| --relabel_csv | Relabel index file | None |
| --relabel | Percentage of relabeled data | None |
| --alpha | L2 regularization strength | Model-specific |
| --no-loo | Disable leave-one-out (must specify to avoid counterfactual computation) | False |

### Influence Calculation Parameters (experiment.infl)

| Parameter | Description | Default |
|-------------------|------------------------------------------|---------------|
| --target | Target dataset (same as training) | adult |
| --model | Model type (same as training) | logreg |
| --type | Influence calculation type | sgd |
| --seed | Random seed (≥0 for single run, <0 for looping 0-99) | 0 |
| --gpu | GPU index | 0 |
| --save_dir | Output directory path override | None |
| --log_level | Log level (DEBUG, INFO, WARNING, ERROR) | INFO |
| --relabel | Percentage of relabeled data | None |
| --length | Length parameter for tim_last influence | 3 |
| --use_tensorboard | Enable TensorBoard logging | False |

## Supported Influence Calculation Types (--type)

| Type | Description |
|------------------|------------------------------------------------------------------|
| sgd | Compute influence using backward SGD method |
| nohess | Backward SGD but ignore Hessian-vector product term (u fixed) |
| tim_last | TIM influence for only the last 'length' epochs |
| tim_first | Compute TIM influence difference for the first epoch |
| tim_middle | Compute TIM influence difference for the middle epoch |
| true | Compute "true" influence by comparing loss with counterfactual models |
| true_first | Compute "true" influence difference for the first epoch |
| true_middle | Compute "true" influence difference for the middle epoch |
| true_last | Compute "true" influence difference for the last epoch |
| lie | Compute LIE influence state at the end of each epoch |
| icml | Compute influence using ICML'17 method (approximate Hessian inverse by optimization) |
| tracin | Compute TracIn influence scores by summing gradient dot products at checkpoints |
| tim_all_epochs | Compute influence for each epoch using the TIM method |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

TIM framework depends on:

- PyTorch (neural networks and training)
- NumPy, Pandas (data processing)
- Scikit-learn (datasets and evaluation)
