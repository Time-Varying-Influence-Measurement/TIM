import os
import numpy as np
import pandas as pd
import torch
import logging
import random
import torch.backends.cudnn as cudnn
import gc

from .train import TrainManager
from .infl import get_file_paths, load_data, load_global_info
from .logging_utils import setup_logging


def set_all_seeds(seed):
    """Set all possible random seeds to ensure full reproducibility"""
    # Python random number generator
    random.seed(seed)
    # NumPy random number generator
    np.random.seed(seed)
    # PyTorch random number generator
    torch.manual_seed(seed)
    # CUDA random number generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # CuDNN settings
        cudnn.deterministic = True
        cudnn.benchmark = False
        logging.info(
            f"CUDA available. Set CuDNN deterministic=True, benchmark=False for seed {seed}"
        )
    else:
        logging.info(
            f"CUDA not available. Skipping CUDA-specific seed setting for seed {seed}"
        )
    # Set environment variables (less common, but for completeness)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Set PYTHONHASHSEED={seed}")


def run_tim_cleansing_experiment(
    key,
    model_type,
    seed=1,
    gpu=0,
    save_dir=None,
    relabel_percentage=None,
    test_performance_metric="test_accuracy",
    keep_ratio=90,
    decay=False,
    lr=0.0005,
    infl_type="tim_all_epochs",
    log_level="INFO",
):
    # 1. Setup: Seeds, Device, Logger, Paths
    set_all_seeds(seed)
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    # Get base directory using the consistent get_file_paths helper
    # We need the directory where the original training and influence files reside.
    # Assume the influence type passed (`infl_type`) corresponds to a file there.
    dir_name, fallback_dat_path, _ = get_file_paths(
        key, model_type, seed, infl_type, save_dir, relabel_percentage
    )
    # Note: fallback_dat_path might not be strictly needed if global_info.json exists

    # Initialize logger within the experiment directory
    log_name = f"cleansing_{infl_type}_{keep_ratio}_pct_{key}_{model_type}"
    logger = setup_logging(
        name=log_name,
        seed=seed,
        save_dir=dir_name,  # Log within the main experiment folder
        gpu=gpu,
        level=getattr(logging, log_level.upper()),
    )
    logger.info(f"--- Starting Cleansing Experiment ---")
    logger.info(f"Config: key={key}, model={model_type}, seed={seed}, gpu={gpu}")
    logger.info(f"Influence Type for Cleansing: {infl_type}, Keep Ratio: {keep_ratio}%")
    logger.info(f"Relabel Percentage: {relabel_percentage}, Device: {device}")
    logger.info(f"Results Directory: {dir_name}")

    # 2. Load Global Info & Original Data Config
    # Use the fallback .dat path obtained earlier if JSON fails
    global_info = load_global_info(
        dir_name, seed, fallback_dat_path, device, logger, relabel_percentage
    )
    n_tr = global_info["n_tr"]
    n_val = global_info["n_val"]
    n_test = global_info["n_test"]
    original_num_epochs = global_info["num_epoch"]  # Epochs in original training
    original_alpha = global_info.get("alpha", 0.0)  # Regularization from original run

    logger.info(
        f"Loaded Global Info: n_tr={n_tr}, n_val={n_val}, n_test={n_test}, orig_epochs={original_num_epochs}, alpha={original_alpha}"
    )

    # 3. Load Original Training Data (will be subsetted later)
    # Note: load_data handles potential relabeling based on relabel_percentage and files in dn
    x_tr_orig, y_tr_orig, x_val, y_val = load_data(
        key,
        global_info,  # Pass the loaded global_info dict
        seed,
        device,
        logger=logger,
        relabel_percentage=relabel_percentage,
        dn=dir_name,
    )
    logger.info(
        f"Loaded original data: x_tr={x_tr_orig.shape}, y_tr={y_tr_orig.shape}, x_val={x_val.shape}, y_val={y_val.shape}"
    )

    # 4. Load Influence Scores
    # Construct the standardized influence CSV filename
    relabel_prefix_csv = (
        f"relabel_{int(relabel_percentage):03d}_pct_"
        if relabel_percentage is not None
        else ""
    )
    infl_csv_filename = f"infl_{infl_type}_{relabel_prefix_csv}{seed:03d}.csv"
    infl_csv_path = os.path.join(dir_name, infl_csv_filename)

    logger.info(f"Attempting to load influence scores from: {infl_csv_path}")
    try:
        infl_df = pd.read_csv(infl_csv_path)
        logger.info(f"Successfully loaded influence file. Shape: {infl_df.shape}")
        # Basic validation
        if "sample_idx" not in infl_df.columns:
            logger.warning(
                f"Influence CSV {infl_csv_path} is missing 'sample_idx' column."
            )
            # Attempt to use index if shapes match, otherwise raise error
            if len(infl_df) == n_tr:
                infl_df["sample_idx"] = np.arange(n_tr)
                logger.info("Added sample_idx based on DataFrame index.")
            else:
                raise ValueError(
                    f"Influence CSV missing 'sample_idx' and length ({len(infl_df)}) doesn't match n_tr ({n_tr})."
                )

    except FileNotFoundError:
        logger.error(f"Influence file not found: {infl_csv_path}")
        logger.error(
            "Please ensure the influence calculation script (infl.py with factory pattern) "
            f"was run successfully for influence type '{infl_type}' "
            f"with seed={seed} and relabel={relabel_percentage}."
        )
        raise  # Stop execution

    # 5. Process Influence Scores and Determine Cleansing Strategy
    use_static_influence = False
    num_cleansing_epochs = original_num_epochs  # Default to original number of epochs

    if infl_type == "tim_all_epochs":
        # Expect columns: sample_idx, influence_segment_0, influence_segment_1, ...
        segment_cols = [
            col for col in infl_df.columns if col.startswith("influence_segment_")
        ]
        if not segment_cols:
            raise ValueError(
                f"Influence file {infl_csv_path} for type 'tim_all_epochs' "
                "does not contain expected 'influence_segment_X' columns."
            )
        # Sort columns just in case they are out of order
        segment_cols.sort(key=lambda name: int(name.split("_")[-1]))
        # Determine epochs based on segments found; should match original_num_epochs
        num_segments = len(segment_cols)
        if num_segments != original_num_epochs:
            logger.warning(
                f"Number of influence segments ({num_segments}) in {infl_csv_path} "
                f"does not match original num_epochs ({original_num_epochs}). "
                f"Using {num_segments} epochs for cleansing loop."
            )
            num_cleansing_epochs = num_segments
        else:
            num_cleansing_epochs = original_num_epochs

        logger.info(
            f"Using TIM per-epoch influence scores. Cleansing loop will run for {num_cleansing_epochs} epochs."
        )

    elif infl_type in [
        "sgd",
        "nohess",
        "tim_last",
        "true",
        "lie",
    ]:  # Add other single-array types here
        # Expect columns: sample_idx, influence, [rank/percentile]
        if "influence" not in infl_df.columns:
            raise ValueError(
                f"Influence file {infl_csv_path} for type '{infl_type}' "
                "does not contain expected 'influence' column."
            )
        if len(infl_df) != n_tr:
            logger.warning(
                f"Influence file length ({len(infl_df)}) does not match n_tr ({n_tr}). This might cause issues."
            )
            # Attempt to proceed if sample_idx allows alignment? Risky. Best to error?
            # Let's try aligning by sample_idx if possible, else error.
            if "sample_idx" in infl_df.columns:
                if not infl_df["sample_idx"].equals(pd.Series(np.arange(n_tr))):
                    raise ValueError(
                        "Influence file length mismatch and sample_idx mismatch."
                    )
            else:
                raise ValueError(
                    "Influence file length mismatch and no sample_idx to align."
                )

        infl_static = (
            infl_df.set_index("sample_idx")["influence"].reindex(np.arange(n_tr)).values
        )
        use_static_influence = True
        num_cleansing_epochs = original_num_epochs  # Use original epoch count for loop
        logger.info(
            f"Using STATIC influence scores from type '{infl_type}'. "
            f"The same cleansing criteria will be applied in each of the {num_cleansing_epochs} retraining epochs."
        )
    else:
        # This case should ideally be caught by argparse choices, but good to have defense
        raise ValueError(
            f"Unsupported influence type for cleansing experiment: {infl_type}"
        )

    # 6. Load Relabeled Indices (for overlap calculation)
    relabel_indices = None
    if relabel_percentage is not None:
        # Standardized name based on refactored infl.py conventions
        relabel_file_prefix = f"relabel_{int(relabel_percentage):03d}_pct_"
        relabel_file_name = f"{relabel_file_prefix}indices_{seed:03d}.csv"
        relabel_file_path = os.path.join(dir_name, relabel_file_name)
        logger.info(f"Looking for relabel indices file at: {relabel_file_path}")
        try:
            relabel_df = pd.read_csv(relabel_file_path)
            # print(f"Read relabel indices file: {relabel_file_path}")
            logger.info(f"Read relabel indices file: {relabel_file_path}")
            # Find column (similar logic to load_data)
            relabel_col = None
            possible_cols = ["relabel_indices", "index", "idx"]
            for col in possible_cols:
                if col in relabel_df.columns:
                    relabel_col = col
                    break
            if relabel_col is None:
                logger.warning(
                    f"Cannot find relabel indices column in {relabel_file_path}. Skipping overlap calculation."
                )
            else:
                relabel_indices = relabel_df[relabel_col].values
                logger.info(
                    f"Loaded {len(relabel_indices)} relabeled indices from {relabel_file_path}. First few: {relabel_indices[:10]}"
                )
        except FileNotFoundError:
            logger.warning(
                f"Relabel indices file not found at {relabel_file_path}. Skipping overlap calculation."
            )
        except Exception as e:
            logger.error(
                f"Error reading relabel indices file {relabel_file_path}: {e}. Skipping overlap calculation."
            )
    else:
        logger.info(
            "No relabel percentage specified, skipping relabel overlap calculation."
        )

    # 7. Setup Training for Cleansed Data
    # Create a subdirectory for this specific cleansing run's records/outputs
    cleansing_run_suffix = f"cleansed_{infl_type}_{keep_ratio:03d}_pct"
    cleansing_records_dir = os.path.join(dir_name, "records")
    cleansing_output_dir = dir_name
    os.makedirs(cleansing_records_dir, exist_ok=True)
    os.makedirs(cleansing_output_dir, exist_ok=True)
    logger.info(f"Saving cleansing run records to: {cleansing_records_dir}")
    logger.info(f"Saving cleansing run performance to: {cleansing_output_dir}")

    # Use training parameters specified for the cleansing run (lr, decay)
    # Keep batch_size maybe from global_info or set fixed? Let's use argument's lr/decay.
    cleansing_training_params = {
        "num_epoch": num_cleansing_epochs,  # Use derived epoch count
        "batch_size": global_info["batch_size"],  # Use original batch size
        "lr": lr,  # Use LR specified for cleansing run
        "decay": decay,  # Use decay specified for cleansing run
    }
    logger.info(f"Cleansing run training parameters: {cleansing_training_params}")

    # Data sizes change per epoch, initialize TM with original sizes for reference?
    tm = TrainManager(
        target=key,
        model=model_type,
        seed=seed,
        # Use the *new* output directory for this cleansing run
        save_dir=cleansing_output_dir,
        n_tr=n_tr,  # Original n_tr for reference
        n_val=n_val,
        n_test=n_test,
        num_epoch=cleansing_training_params["num_epoch"],
        batch_size=cleansing_training_params["batch_size"],
        lr=cleansing_training_params["lr"],
        decay=cleansing_training_params["decay"],
        relabel_percentage=relabel_percentage,  # Keep track if original data was relabeled
        device=device,
        logger=logger,
        # Pass original alpha for consistency if model uses it implicitly?
        alpha=original_alpha,
    )

    # Set training_params to avoid 'NoneType' object is not subscriptable error
    tm.training_params = cleansing_training_params

    # Override specific paths for the cleansing run
    tm.dir_name = cleansing_output_dir  # Main output dir for this run
    tm.records_dir = cleansing_records_dir  # Subdir for records
    # File names should reflect the cleansing run
    tm.relabel_prefix = (
        f"relabel_{int(relabel_percentage):03d}_pct_"
        if relabel_percentage is not None
        else ""
    )
    tm.file_name = os.path.join(
        tm.dir_name,
        f"{tm.relabel_prefix}{cleansing_run_suffix}_model_list_{seed:03d}.dat",
    )  # Less relevant now?
    tm.step_loss_file = os.path.join(
        tm.records_dir,
        f"{tm.relabel_prefix}{cleansing_run_suffix}_step_losses_{seed:03d}.csv",
    )

    # Other necessary TM setup (input_dim uses original data)
    tm.x_tr = x_tr_orig  # Use full dataset
    tm.y_tr = y_tr_orig
    tm.x_val = x_val
    tm.y_val = y_val
    # tm.num_steps will be updated each epoch based on subset size
    tm.input_dim = x_tr_orig.shape[1:]  # Dim from original data
    tm.net_func = lambda: tm.get_model(model_type, tm.input_dim, device)
    tm.list_of_sgd_models = []  # Models saved during this run

    # Save global info specific to this cleansing run
    cleansing_data_sizes = {
        "n_tr": n_tr,
        "n_val": n_val,
        "n_test": n_test,
        "keep_ratio": keep_ratio,
        "infl_type_used": infl_type,
    }
    # Note: n_tr here is the *original* size before cleansing
    tm.save_global_info(
        seed,
        cleansing_data_sizes,
        cleansing_training_params,
        tm.alpha,
        filename_suffix=f"_{cleansing_run_suffix}",
    )

    # 8. Initialize Model and Optimizer for Cleansing Run
    model = tm.net_func()
    # Load the *same initial parameters* as the original run for fair comparison
    init_file_prefix = (
        f"relabel_{int(relabel_percentage):03d}_pct_"
        if relabel_percentage is not None
        else ""
    )
    init_file = os.path.join(
        dir_name, "records", f"{init_file_prefix}init_{seed:03d}.pt"
    )  # Look in original records dir
    init_file_no_relabel = os.path.join(
        dir_name, "records", f"init_{seed:03d}.pt"
    )  # Fallback non-relabel name

    init_loaded = False
    for init_path in [init_file, init_file_no_relabel]:
        if os.path.exists(init_path):
            try:
                # Load state dict carefully (handle potential dict wrapping)
                init_data = torch.load(init_path, map_location=device)
                state_dict = None
                if isinstance(init_data, dict) and "model_state" in init_data:
                    state_dict = init_data["model_state"]
                elif isinstance(init_data, dict) and any(
                    k.endswith(".weight") or k.endswith(".bias")
                    for k in init_data.keys()
                ):
                    state_dict = init_data  # Assume it's the state dict
                if state_dict:
                    model.load_state_dict(state_dict)
                    logger.info(
                        f"Loaded initial model parameters from original run: {init_path}"
                    )
                    init_loaded = True
                    break  # Stop after successful load
                else:
                    logger.warning(
                        f"File {init_path} exists but could not extract state_dict."
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to load initial model parameters from {init_path}: {e}"
                )

    if not init_loaded:
        logger.warning(
            f"Could not load original initial model parameters from {init_file} or {init_file_no_relabel}. "
            "Using NEW random initialization for cleansing run."
        )
        # Re-initialize model just in case previous attempts modified it
        model = tm.net_func()

    # Save the initial state for *this cleansing run*
    tm.save_at_initial(
        model,
        tm.list_of_sgd_models,
        None,  # No counterfactuals needed here
        -1,  # n_run = -1 indicates main run
        False,  # is_counterfactual
        filename_suffix=f"_{cleansing_run_suffix}",  # Add suffix to init file name
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer, is_vit_model, unfreeze_epoch = tm.setup_optimizer(
        model, model_type, cleansing_training_params, -1  # n_run=-1
    )
    lr_n = cleansing_training_params["lr"]
    total_step = 0
    info = []  # Step losses etc.
    results = []  # Performance per epoch

    # Prepare output CSVs for this run
    keep_indices_csv = os.path.join(
        dir_name,
        f"kept_indices_{seed:03d}.csv"
    )
    relabel_overlap_csv = os.path.join(
        dir_name,
        f"relabel_overlap_{seed:03d}.csv"
    )

    with open(keep_indices_csv, "w") as f:
        f.write("epoch,num_kept,kept_indices_preview\n")
    if relabel_indices is not None:
        with open(relabel_overlap_csv, "w") as f:
            f.write("epoch,num_dropped,num_relabelled,num_overlap,overlap_ratio\n")

    # 9. Run Cleansing and Retraining Loop

    # Evaluate initial model performance
    logger.info("Evaluating initial model (epoch -1) before cleansing loop...")
    val_loss_init, test_acc_init, train_loss_init = tm.evaluate_epoch(
        model, loss_fn, 0, -1, -1
    )
    results.append(
        {
            "epoch": -1,
            test_performance_metric: test_acc_init,
            "val_loss": val_loss_init,
            "train_loss": train_loss_init,  # Train loss on original data? Or empty set? Evaluate does full train set.
        }
    )
    logger.info(
        f"Initial model (epoch -1): Val Loss={val_loss_init:.4f}, Test Acc={test_acc_init:.4f}, Train Loss={train_loss_init:.4f}"
    )

    for epoch in range(num_cleansing_epochs):
        logger.info(f"--- Starting Cleansed Epoch {epoch}/{num_cleansing_epochs-1} ---")

        # Temporarily modify tm.relabel_prefix to include suffix
        original_prefix = tm.relabel_prefix
        tm.relabel_prefix = original_prefix + f"{cleansing_run_suffix}_"

        # Select influence scores for this epoch/segment
        if use_static_influence:
            infl_scores = infl_static
            logger.debug(f"Using static influence scores for epoch {epoch}.")
        else:
            # Use tim_all_epochs scores for this segment
            current_segment_col = f"influence_segment_{epoch}"
            if current_segment_col not in infl_df.columns:
                logger.error(
                    f"Missing influence column '{current_segment_col}' for epoch {epoch}. Stopping."
                )
                raise KeyError(f"Missing influence column '{current_segment_col}'")
            # Align scores with original sample indices
            infl_scores = (
                infl_df.set_index("sample_idx")[current_segment_col]
                .reindex(np.arange(n_tr))
                .values
            )
            logger.debug(
                f"Using dynamic influence scores from {current_segment_col} for epoch {epoch}."
            )

        # Determine indices to keep based on influence scores
        keep_num = int(n_tr * keep_ratio / 100)
        # Handle potential NaN values in influence scores (replace with very low value for sorting)
        infl_scores_clean = np.nan_to_num(infl_scores, nan=-np.inf)
        keep_idx = np.argsort(-infl_scores_clean)[
            :keep_num
        ]  # Sort descending, take top indices

        # Calculate the indices of samples to skip (samples not to keep)
        skip_idx = np.setdiff1d(np.arange(n_tr), keep_idx)

        logger.info(
            f"Epoch {epoch}: Keeping top {keep_num}/{n_tr} samples based on influence. Skipping {len(skip_idx)} samples."
        )

        # Save kept indices preview
        with open(keep_indices_csv, "a") as f:
            f.write(
                f"{epoch},{len(keep_idx)}," + ",".join(map(str, keep_idx[:20])) + "\n"
            )  # Save first 20 indices

        # Calculate overlap with relabeled points (if applicable)
        if relabel_indices is not None:
            dropped_idx = skip_idx  # Use skip_idx as dropped_idx
            overlap = np.intersect1d(dropped_idx, relabel_indices)
            num_dropped = len(dropped_idx)
            num_relabelled = len(relabel_indices)
            num_overlap = len(overlap)
            overlap_ratio = num_overlap / num_relabelled if num_relabelled > 0 else 0.0
            logger.info(
                f"Epoch {epoch}: Dropped {num_dropped} samples. Relabeled points: {num_relabelled}. Overlap (dropped & relabeled): {num_overlap}. Overlap Ratio: {overlap_ratio:.4f}"
            )
            # Save overlap stats
            with open(relabel_overlap_csv, "a") as f:
                f.write(
                    f"{epoch},{num_dropped},{num_relabelled},{num_overlap},{overlap_ratio}\n"
                )

        # Update tm.num_steps to reflect the new number of effective training samples
        effective_n_tr = n_tr - len(skip_idx)
        tm.num_steps = int(
            np.ceil(effective_n_tr / cleansing_training_params["batch_size"])
        )
        logger.info(
            f"Effective training size for epoch {epoch}: {effective_n_tr}, num_steps: {tm.num_steps}"
        )

        if effective_n_tr == 0:
            logger.warning(
                f"Epoch {epoch}: No samples kept after cleansing. Skipping training for this epoch."
            )
            # How to handle evaluation? Evaluate with current model state?
            val_loss, test_acc, train_loss = tm.evaluate_epoch(
                model, loss_fn, 0, -1, epoch
            )  # Evaluate current state
            epoch_step_info = []  # No steps taken
            epoch_loss = 0.0
        else:
            # Custom training for a single epoch (because TrainManager's train_epoch does not correctly use the skip parameter)
            model.train()  # Ensure the model is in training mode
            epoch_loss = 0.0
            info_epoch = []  # List of step-level information
            epoch_step_info = []  # Used to save epoch checkpoints

            # Generate permutation using a fixed random seed to ensure it's the same for every run
            np.random.seed(tm.seed + epoch)
            permuted_indices = np.random.permutation(len(tm.x_tr))

            # Save permutation to file for later verification
            perm_file = os.path.join(
                tm.records_dir, f"permutation_epoch_{epoch:03d}.npy"
            )
            np.save(perm_file, permuted_indices)

            # Key modification: Remove indices in skip_idx from the permuted indices
            valid_indices = np.setdiff1d(permuted_indices, skip_idx, assume_unique=True)
            logger.info(
                f"After removing skipped samples, using {len(valid_indices)}/{len(permuted_indices)} samples for training"
            )

            # Split the remaining valid indices into batches
            idx_list = np.array_split(valid_indices, tm.num_steps)

            for i, idx in enumerate(idx_list):
                if len(idx) == 0:
                    continue  # Skip empty batches

                # Get batch data and move to device
                x_batch = tm.x_tr[idx].to(tm.device)
                y_batch = tm.y_tr[idx].to(tm.device)

                # --- Forward propagation ---
                z = model(x_batch)
                loss = loss_fn(z, y_batch)
                batch_loss = loss.item()  # Loss without regularization
                epoch_loss += batch_loss * len(idx)  # Accumulate total loss, weighted by batch size

                # --- Add L2 regularization ---
                l2_reg = 0.0
                if tm.alpha > 0:
                    for p in filter(lambda p: p.requires_grad, model.parameters()):
                        l2_reg += (p * p).sum()
                    loss += 0.5 * tm.alpha * l2_reg

                # --- Backpropagation and optimization ---
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # --- Learning rate update (if decay is enabled) ---
                if cleansing_training_params["decay"]:
                    # Pass the LR of the current optimizer state
                    current_lr_group0 = optimizer.param_groups[0]["lr"]
                    lr_n = tm.update_learning_rate(
                        optimizer,
                        is_vit_model,
                        -1,  # n_run = -1
                        cleansing_training_params,
                        epoch,
                        unfreeze_epoch,
                        total_step,
                        tm.num_steps,
                        current_lr_group0,
                    )

                # --- Record step information ---
                step_info = {
                    "idx": idx.tolist(),
                    "lr": optimizer.param_groups[0]["lr"],
                }  # Record the actually used LR
                info_epoch.append(step_info)
                epoch_step_info.append(step_info)

                # --- Save step checkpoint (only for n=-1 run) ---
                avg_batch_loss = batch_loss  # Use loss without regularization for logging
                tm._save_each_step(
                    model,
                    total_step,
                    idx.tolist(),  # Save indices as a list
                    optimizer.param_groups[0]["lr"],  # Record LR from optimizer
                    step_loss=avg_batch_loss,
                    epoch=epoch,
                )
                if (total_step + 1) % 50 == 0:  # Record every 50 steps
                    logger.info(
                        f"Step {total_step+1}/{cleansing_training_params['num_epoch']*tm.num_steps}, "
                        f"Epoch {epoch+1}/{cleansing_training_params['num_epoch']}, "
                        f"Batch Loss: {avg_batch_loss:.4f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )
                step_info["step_loss"] = avg_batch_loss
                step_info["global_step"] = total_step

                total_step += 1
                del z, loss, x_batch, y_batch  # Free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Calculate average epoch loss
            avg_epoch_loss = (
                epoch_loss / len(valid_indices) if len(valid_indices) > 0 else 0
            )

            # Add step information for this epoch to global info
            info.extend(info_epoch)

            # Evaluate model performance
            val_loss, test_acc, train_loss = tm.evaluate_epoch(
                model,
                loss_fn,
                avg_epoch_loss,
                -1,
                epoch,
            )
            logger.info(
                f"Epoch {epoch} evaluation: Val Loss={val_loss:.4f}, Test Acc={test_acc:.4f}, Subset Train Loss={train_loss:.4f}"
            )

        # The prefix has already been modified here, no need to repeat
        # Save epoch data (model state, step info, performance) for this cleansing run
        tm.save_epoch_data(
            model=model,
            epoch=epoch,
            epoch_step_info=epoch_step_info,
            val_loss=val_loss,
            test_acc=test_acc,
            train_loss=train_loss,
            n=-1,
            seed=seed,
        )

        # Restore the original prefix
        tm.relabel_prefix = original_prefix

        results.append(
            {
                "epoch": epoch,
                test_performance_metric: test_acc,
                "val_loss": val_loss,
                "train_loss": train_loss,
            }
        )
        logger.info(f"--- Finished Cleansed Epoch {epoch} ---")

        # Clean up GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 10. Save Final Results and Statistics
    logger.info("--- Saving Final Cleansing Experiment Results ---")
    try:
        # Save performance results for this run
        # out_csv = os.path.join(cleansing_output_dir, f"performance_{seed:03d}.csv")
        out_csv = os.path.join(cleansing_output_dir, f"{cleansing_run_suffix}_performance_{seed:03d}.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(out_csv, index=False)
        logger.info(f"Saved performance results to {out_csv}")

        # Add final model performance summary to log
        if not results_df.empty:
            final_perf = results_df.iloc[-1]  # Performance after last epoch
            logger.info(
                f"Final performance (Epoch {final_perf['epoch']}): "
                f"{test_performance_metric}={final_perf[test_performance_metric]:.4f}, "
                f"Val Loss={final_perf['val_loss']:.4f}"
            )

        # --- Consolidate Statistics ---
        # (Kept indices and overlap stats were saved per-epoch already)
        # Read back the saved CSVs to calculate overall stats if needed

        # Kept Indices Stats
        try:
            kept_indices_df = pd.read_csv(keep_indices_csv)
            if not kept_indices_df.empty:
                # Convert preview string to list length for num_kept (more robust)
                kept_indices_df["num_kept"] = kept_indices_df[
                    "kept_indices_preview"
                ].apply(
                    lambda x: (
                        len(x.split(",")) if isinstance(x, str) and x.strip() else 0
                    )
                )
                kept_stats = kept_indices_df["num_kept"].describe()
                logger.info(
                    "Kept Indices Statistics (per epoch):\n" + kept_stats.to_string()
                )
                stats_summary_file = os.path.join(
                    cleansing_output_dir, f"stats_summary_{seed:03d}.txt"
                )
                with open(stats_summary_file, "w") as f:
                    f.write("Kept Indices Statistics (per epoch):\n")
                    f.write(kept_stats.to_string())
                    f.write("\n\n")
            else:
                logger.warning(f"Kept indices file {keep_indices_csv} was empty.")
        except Exception as e:
            logger.error(
                f"Error processing kept indices stats from {keep_indices_csv}: {e}"
            )

        # Relabel Overlap Stats
        if relabel_indices is not None:
            try:
                if os.path.exists(relabel_overlap_csv):
                    overlap_df = pd.read_csv(relabel_overlap_csv)
                    if not overlap_df.empty:
                        overlap_stats = overlap_df["overlap_ratio"].describe()
                        logger.info(
                            "Relabel Overlap Ratio Statistics (per epoch):\n"
                            + overlap_stats.to_string()
                        )
                        stats_summary_file = os.path.join(
                            cleansing_output_dir, f"stats_summary_{seed:03d}.txt"
                        )  # Append here
                        with open(stats_summary_file, "a") as f:
                            f.write("Relabel Overlap Ratio Statistics (per epoch):\n")
                            f.write(overlap_stats.to_string())
                            f.write("\n")
                    else:
                        logger.warning(
                            f"Relabel overlap file {relabel_overlap_csv} was empty."
                        )
                else:
                    logger.warning(
                        f"Relabel overlap file {relabel_overlap_csv} not found (expected if no relabeling)."
                    )
            except Exception as e:
                logger.error(
                    f"Error processing relabel overlap stats from {relabel_overlap_csv}: {e}"
                )

        logger.info(f"--- Cleansing Experiment Finished for Seed {seed} ---")

    except Exception as e:
        logger.error(
            f"An error occurred during result saving or final statistics: {e}",
            exc_info=True,
        )
        # Ensure experiment stops gracefully if saving fails critically
        raise


if __name__ == "__main__":
    import argparse

    def str_to_bool(value):
        """Convert common true/false string representations to boolean values, case-insensitive."""
        if isinstance(value, bool):
            return value
        if value.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif value.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            # If the value given on the command line is not one of these, throw an error
            raise argparse.ArgumentTypeError(
                f"Invalid boolean value: '{value}' (should be True/False, etc.)"
            )

    parser = argparse.ArgumentParser(
        description="Run Data Cleansing Experiment using Influence Scores"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target dataset key (e.g., adult, loan)",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model type key (e.g., logreg, dnn)"
    )
    # Allow single seed or comma-separated list
    parser.add_argument(
        "--seed",
        type=str,
        default="0",
        help="Single seed (e.g., 0) or comma-separated list (e.g., 0,1,2)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional base directory override for results",
    )
    parser.add_argument(
        "--relabel",
        type=int,
        default=None,
        help="Percentage of original data that was relabeled (e.g., 10)",
    )
    parser.add_argument(
        "--keep_ratio",
        type=int,
        default=90,
        help="Percentage of samples to KEEP based on influence (0-100, default 90)",
    )
    # Boolean flag for decay
    parser.add_argument(
        "--decay",
        type=str_to_bool,  # Use custom type conversion function
        nargs="?",  # Indicates that the parameter value is optional (0 or 1)
        const=True,  # If --decay is provided but no value is given, it is considered True
        default=False,  # If the --decay flag is completely absent, it defaults to False
        help="Whether to use learning rate decay. Can be specified as --decay (means True) or --decay True/False (default: False)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="Learning rate for cleansing retraining",
    )
    # Updated choices based on implemented factory calculators
    parser.add_argument(
        "--type",
        type=str,
        default="tim_all_epochs",
        choices=[
            "tim_all_epochs",
            "sgd",
            "nohess",
            "tim_last",
            "true",
            "lie",
        ],  # Add more as needed/tested
        help="Influence calculation type whose scores to use for cleansing",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )
    args = parser.parse_args()

    # --- Seed Processing ---
    try:
        if "," in args.seed:
            seed_list = [
                int(s.strip()) for s in args.seed.split(",") if s.strip().isdigit()
            ]
        else:
            seed_list = [int(args.seed)]
        if not seed_list:
            raise ValueError("No valid seeds provided.")
        # Basic validation for seed values (e.g., non-negative)
        if any(s < 0 for s in seed_list):
            print(
                "Warning: Negative seeds are not standard. Proceeding, but ensure this is intended."
            )

    except ValueError as e:
        print(
            f"Error parsing seeds: {e}. Please provide a single integer or comma-separated integers."
        )
        exit(1)

    print(f"Running cleansing experiment for seeds: {seed_list}")
    print(
        f"Config: Dataset='{args.target}', Model='{args.model}', Influence Type='{args.type}', Keep%={args.keep_ratio}"
    )

    all_results_list = []
    base_output_dir = None  # To store the directory for saving mean results

    for seed in seed_list:
        try:
            print(f"\n--- Processing Seed {seed} ---")
            run_tim_cleansing_experiment(
                key=args.target,
                model_type=args.model,
                seed=seed,
                gpu=args.gpu,
                save_dir=args.save_dir,
                relabel_percentage=args.relabel,
                keep_ratio=args.keep_ratio,
                decay=args.decay,
                lr=args.lr,
                infl_type=args.type,
                log_level=args.log_level,
            )
            # Read back the performance result for this seed to aggregate later
            # Construct the path to the cleansing run's output directory
            dn_base, _, _ = get_file_paths(
                args.target, args.model, seed, args.type, args.save_dir, args.relabel
            )
            cleansing_run_suffix = f"cleansed_{args.type}_{args.keep_ratio:03d}_pct"
            cleansing_output_dir = dn_base
            out_csv = os.path.join(cleansing_output_dir, f"{cleansing_run_suffix}_performance_{seed:03d}.csv")

            if (
                base_output_dir is None
            ):  # Store the base directory from the first successful seed
                base_output_dir = dn_base  # Parent directory for the specific seed

            if os.path.exists(out_csv):
                df = pd.read_csv(out_csv)
                df["seed"] = seed
                all_results_list.append(df)
                print(f"--- Seed {seed} completed successfully. ---")
            else:
                print(
                    f"Warning: Performance file not found after running seed {seed}: {out_csv}"
                )

        except Exception as e:
            # Log error (if logger was setup) or print
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR processing seed {seed}: {e}")
            import traceback

            traceback.print_exc()  # Print full traceback for debugging
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # --- Aggregate results if multiple seeds were run ---
    if len(all_results_list) > 1 and base_output_dir:
        print("\n--- Aggregating results across seeds ---")
        try:
            all_df = pd.concat(all_results_list, ignore_index=True)

            # Calculate mean and std deviation across seeds for each epoch
            agg_df = (
                all_df.groupby("epoch")
                .agg(
                    # Dynamically find performance metric columns (besides epoch, seed)
                    **{
                        col + "_mean": pd.NamedAgg(column=col, aggfunc="mean")
                        for col in all_df.columns
                        if col not in ["epoch", "seed"]
                    },
                    **{
                        col + "_std": pd.NamedAgg(column=col, aggfunc="std")
                        for col in all_df.columns
                        if col not in ["epoch", "seed"]
                    },
                    seed_count=(
                        "seed",
                        "count",
                    ),  # Count how many seeds contributed to each epoch
                )
                .reset_index()
            )

            # Construct mean results filename within the base output directory (not seed-specific)
            # Example: place it alongside the seed folders, or inside the folder of seed 0? Let's put it alongside.
            mean_results_filename = f"{cleansing_run_suffix}_performance_aggregated.csv"
            mean_csv_path = os.path.join(base_output_dir, mean_results_filename)

            agg_df.to_csv(mean_csv_path, index=False)
            print(
                f"Saved aggregated (mean/std) performance results to: {mean_csv_path}"
            )

            # Optionally print final aggregated performance
            final_epoch_agg = agg_df[agg_df["epoch"] == agg_df["epoch"].max()]
            if not final_epoch_agg.empty:
                print("\nFinal Aggregated Performance (Mean +/- Std):")
                for col in all_df.columns:
                    if col not in ["epoch", "seed"]:
                        mean_val = final_epoch_agg.iloc[0].get(
                            f"{col}_mean", float("nan")
                        )
                        std_val = final_epoch_agg.iloc[0].get(
                            f"{col}_std", float("nan")
                        )
                        print(f"  {col}: {mean_val:.4f} +/- {std_val:.4f}")

        except Exception as e:
            print(f"Error during results aggregation: {e}")

    elif len(all_results_list) == 1:
        print("\n--- Only one seed processed. No aggregation performed. ---")
    else:
        print(
            "\n--- No successful seeds found or only one seed run. Skipping aggregation. ---"
        )

    print("\n--- Experiment Script Finished ---")
