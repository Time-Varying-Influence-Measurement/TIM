import argparse
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

# ===================== Config & Constants =====================
NETWORKS = ["tinyvit_emnist"]
RELABELS = [2, 4, 6, 8, 10, 15, 20]
WORK_DIR = Path(__file__).resolve().parent.parent
PYTHON_ENV = WORK_DIR / ".venv/bin/python"

# ===================== Logging Config =====================
def setup_logging(log_file: str = None) -> None:
    """Configure logging output"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers
    )

# ===================== Directory & File Operations =====================
def setup_directories(network: str, relabel: int) -> Tuple[Path, Path, Path, Path]:
    """Create required directories for the experiment, return related paths"""
    hessian_dir = WORK_DIR / f"Hessian_8000_{network}"
    dest_dir = WORK_DIR / f"Hessian_8000_{network}_cleansing_plots"
    save_dir = hessian_dir / f"emnist_{network}_dit_last_tracin_relabel_{relabel}_10e-1"
    sec71_save_dir = WORK_DIR / "experiment" / "Sec71" / save_dir.relative_to(WORK_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    sec71_save_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Save directory: {save_dir.resolve()}")
    logging.info(f"Sec71 save directory: {sec71_save_dir.resolve()}")
    return hessian_dir, dest_dir, save_dir, sec71_save_dir

# ===================== Training & Influence Calculation =====================
def run_training(network: str, gpu: int, save_dir: Path, relabel: int) -> bool:
    """Run training process"""
    cmd = [
        str(PYTHON_ENV), "-m", "experiment.Sec71.train",
        "--target", "emnist",
        "--model", network,
        "--gpu", str(gpu),
        "--seed", "0",
        "--save_dir", str(save_dir),
        "--relabel", str(relabel),
        "--no-loo"
    ]
    try:
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": str(WORK_DIR)})
        logging.info("Training completed")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed: {e}")
        return False

def run_dit_last_influence(network: str, gpu: int, save_dir: Path, relabel: int) -> bool:
    """Run dit_last influence calculation"""
    cmd = [
        str(PYTHON_ENV), "-m", "experiment.Sec71.infl",
        "--target", "emnist",
        "--model", network,
        "--gpu", str(gpu),
        "--seed", "0",
        "--save_dir", str(save_dir),
        "--relabel", str(relabel),
        "--type", "dit_last",
        "--length", "1"
    ]
    try:
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": str(WORK_DIR)})
        logging.info("dit_last influence calculation completed")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"dit_last influence calculation failed: {e}")
        return False

def run_tracin_influence(network: str, gpu: int, save_dir: Path, relabel: int) -> bool:
    """Run tracin influence calculation"""
    cmd = [
        str(PYTHON_ENV), "-m", "experiment.Sec71.infl",
        "--target", "emnist",
        "--model", network,
        "--gpu", str(gpu),
        "--seed", "0",
        "--save_dir", str(save_dir),
        "--relabel", str(relabel),
        "--type", "tracin"
    ]
    try:
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": str(WORK_DIR)})
        logging.info("tracin influence calculation completed")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"tracin influence calculation failed: {e}")
        return False

# ===================== Plotting & File Copy =====================
def generate_cleansing_plots(save_dir: Path, sec71_save_dir: Path, relabel: int) -> bool:
    """Generate cleansing plots"""
    cmd = [
        str(PYTHON_ENV), "-m", "experiment.Sec71.cleansing_plot",
        "--save_dir", str(save_dir),
        "--relabel", str(relabel)
    ]
    try:
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": str(WORK_DIR)})
        logging.info(f"Cleansing plot generated successfully: {save_dir}")
        return True
    except subprocess.CalledProcessError:
        # Try Sec71 directory
        cmd[cmd.index(str(save_dir))] = str(sec71_save_dir)
        try:
            subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": str(WORK_DIR)})
            logging.info(f"Cleansing plot generated successfully: {sec71_save_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Cleansing plot generation failed: {e}")
            return False

def copy_plots(hessian_dir: Path, dest_dir: Path, save_dir: Path, sec71_save_dir: Path, relabel: int) -> None:
    """Copy generated plots to target directory"""
    flattened_name = str(save_dir.relative_to(hessian_dir)).replace("/", "_")
    plot_names = [
        f"cleansing_plot_{relabel}_pct.png",
        f"cleansing_plot_{relabel}_pct_decending.png",
        f"cleansing_plot_{relabel}_pct_tracin_vs_dit_last_asc.png",
        f"cleansing_plot_{relabel}_pct_tracin_vs_dit_last_desc.png"
    ]
    for plot_name in plot_names:
        src_path = save_dir / plot_name
        if not src_path.exists():
            src_path = sec71_save_dir / plot_name
        if src_path.exists():
            dst_path = dest_dir / f"{flattened_name}_{plot_name}"
            try:
                dst_path.write_bytes(src_path.read_bytes())
                logging.info(f"Copied {plot_name} to: {dst_path}")
            except Exception as e:
                logging.error(f"Failed to copy {plot_name}: {e}")
        else:
            logging.warning(f"Plot not found: {plot_name} in {save_dir} or {sec71_save_dir}")

# ===================== Single Configuration Execution =====================
def run_configuration(config_idx: int, gpu_ids: List[int]) -> bool:
    """Run the full process for a single configuration, return whether successful"""
    relabels_count = len(RELABELS)
    network_idx = (config_idx - 1) // relabels_count
    relabel_idx = (config_idx - 1) % relabels_count
    network = NETWORKS[network_idx]
    relabel = RELABELS[relabel_idx]
    gpu = gpu_ids[(config_idx - 1) % len(gpu_ids)]
    logging.info(f"Start config {config_idx}: network={network}, relabel={relabel}, GPU={gpu}")
    hessian_dir, dest_dir, save_dir, sec71_save_dir = setup_directories(network, relabel)
    # run_training(network, gpu, save_dir, relabel)
    # run_dit_last_influence(network, gpu, save_dir, relabel)
    # run_tracin_influence(network, gpu, save_dir, relabel)
    generate_cleansing_plots(save_dir, sec71_save_dir, relabel)
    copy_plots(hessian_dir, dest_dir, save_dir, sec71_save_dir, relabel)
    logging.info(f"Config {config_idx} completed")
    return True

# ===================== Main Process =====================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=3, help="Number of parallel workers")
    parser.add_argument("--gpu_ids", type=str, required=True, help="List of GPU IDs, e.g. '0,2,4'")
    parser.add_argument("--log_file", type=str, default=None, help="Log file path")
    args = parser.parse_args()
    setup_logging(args.log_file)
    gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",")]
    total_configs = len(NETWORKS) * len(RELABELS)
    logging.info(f"Total configs: {total_configs}, GPUs: {gpu_ids}")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        config_indices = list(range(1, total_configs + 1))
        futures = [executor.submit(run_configuration, idx, gpu_ids) for idx in config_indices]
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Subprocess exception: {e}")
                results.append(False)
    successful = sum(bool(r) for r in results)
    logging.info(f"Finished: {successful}/{len(results)} configs succeeded.")

if __name__ == "__main__":
    main()
