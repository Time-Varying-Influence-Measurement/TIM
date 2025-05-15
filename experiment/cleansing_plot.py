# %%
import pandas as pd
import argparse
import os
import sys

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, required=True, help='Directory where influence files are stored')
parser.add_argument('--relabel', type=int, default=2, help='Relabel percentage (2, 4, 6, 8, 10)')
args = parser.parse_args()

# Define file paths based on relabel percentage
relabel_pct = args.relabel
tracin_file = os.path.join(args.save_dir, f"infl_tracin_{relabel_pct}_pct_000.csv")
tim_last_file = os.path.join(args.save_dir, f"infl_sgd_{relabel_pct}_pct_000.csv")
relabeled_indices_file = os.path.join(args.save_dir, f"relabeled_indices_{relabel_pct}_pct_000.csv")

# Print absolute paths for debugging
print(f"Looking for files in: {os.path.abspath(args.save_dir)}")
print(f"Tracin file path: {os.path.abspath(tracin_file)}")
print(f"TIM last file path: {os.path.abspath(tim_last_file)}")
print(f"Relabeled indices file path: {os.path.abspath(relabeled_indices_file)}")

# Check if files exist and read them
try:
    if not os.path.exists(tracin_file):
        print(f"Warning: Tracin file not found: {tracin_file}")
        print("Trying alternative file path...")
        # Try with different path formats
        alt_paths = [
            os.path.join(args.save_dir, "infl_tracin_0.csv"),
            os.path.join(args.save_dir, f"infl_tracin_{relabel_pct}.csv"),
            os.path.join(args.save_dir, f"infl_tracin_{relabel_pct}_pct.csv"),
            os.path.join(os.path.dirname(args.save_dir), f"infl_tracin_{relabel_pct}_pct_000.csv"),
            # Try in experiment directory
            os.path.join("experiment", "Sec71", args.save_dir, f"infl_tracin_{relabel_pct}_pct_000.csv")
        ]

        for alt_path in alt_paths:
            print(f"Trying alternative path: {alt_path}")
            if os.path.exists(alt_path):
                print(f"Found alternative tracin file at: {alt_path}")
                tracin_file = alt_path
                break

        if not os.path.exists(tracin_file):
            print(f"Error: Tracin file not found at any alternative path")
            sys.exit(1)

    if not os.path.exists(tim_last_file):
        print(f"Warning: TIM last file not found: {tim_last_file}")
        print("Trying alternative file path...")
        # Try with different path formats
        alt_paths = [
            os.path.join(args.save_dir, "infl_sgd_000.csv"),
            os.path.join(args.save_dir, f"infl_sgd_{relabel_pct}.csv"),
            os.path.join(args.save_dir, f"infl_sgd_{relabel_pct}_pct.csv"),
            os.path.join(os.path.dirname(args.save_dir), f"infl_sgd_{relabel_pct}_pct_000.csv"),
            # Try in experiment directory
            os.path.join("experiment", "Sec71", args.save_dir, f"infl_sgd_{relabel_pct}_pct_000.csv")
        ]

        for alt_path in alt_paths:
            print(f"Trying alternative path: {alt_path}")
            if os.path.exists(alt_path):
                print(f"Found alternative TIM last file at: {alt_path}")
                tim_last_file = alt_path
                break

        if not os.path.exists(tim_last_file):
            print(f"Error: TIM last file not found at any alternative path")
            sys.exit(1)

    if not os.path.exists(relabeled_indices_file):
        print(f"Warning: Relabeled indices file not found: {relabeled_indices_file}")
        print("Trying alternative file path...")
        # Try with different path formats
        alt_paths = [
            os.path.join(args.save_dir, "relabeled_indices_000.csv"),
            os.path.join(args.save_dir, f"relabeled_indices_{relabel_pct}.csv"),
            os.path.join(args.save_dir, f"relabeled_indices_{relabel_pct}_pct.csv"),
            os.path.join(os.path.dirname(args.save_dir), f"relabeled_indices_{relabel_pct}_pct_000.csv"),
            # Try in experiment directory
            os.path.join("experiment", "Sec71", args.save_dir, f"relabeled_indices_{relabel_pct}_pct_000.csv")
        ]

        for alt_path in alt_paths:
            print(f"Trying alternative path: {alt_path}")
            if os.path.exists(alt_path):
                print(f"Found alternative relabeled indices file at: {alt_path}")
                relabeled_indices_file = alt_path
                break

        if not os.path.exists(relabeled_indices_file):
            print(f"Error: Relabeled indices file not found at any alternative path")
            sys.exit(1)

    # Read influence files from the specified save_dir
    tracin_infl = pd.read_csv(tracin_file)
    tim_last_infl = pd.read_csv(tim_last_file)
    relabeled_indices = pd.read_csv(relabeled_indices_file)

    # %%
    # sort tracin descending and tim last ascending
    tracin_infl = tracin_infl.sort_values(by='influence', ascending=True)
    tim_last_infl = tim_last_infl.sort_values(by='influence', ascending=True)

    # %%
    def check_overlap_at_every_sample(influence_df, relabeled_indices, method_name):
        """
        Check overlap between influential samples and relabeled indices at every possible sample count.
        
        Parameters:
        -----------
        influence_df : DataFrame
            DataFrame containing influence scores with 'sample_idx' column
        relabeled_indices : DataFrame or Series
            DataFrame or Series containing the relabeled indices
        method_name : str
            Name of the influence method for display purposes
        
        Returns:
        --------
        DataFrame
            Results containing sample count and overlap count for every possible count
        """
        if isinstance(relabeled_indices, pd.DataFrame) and 'relabeled_indices' in relabeled_indices.columns:
            relabeled_set = set(relabeled_indices['relabeled_indices'])
        else:
            relabeled_set = set(relabeled_indices)

        total_samples = len(influence_df)
        num_relabeled = len(relabeled_set)

        # Initialize results with all zeros
        results = {
            'Method': [method_name] * total_samples,
            'Sample Count': list(range(1, total_samples + 1)),
            'Overlap Count': [0] * total_samples
        }

        # Pre-compute is_relabeled for all samples in influence_df
        is_relabeled = influence_df['sample_idx'].isin(relabeled_set).values

        # Calculate the cumulative sum of relabeled samples
        cumulative_relabeled = is_relabeled.cumsum()

        # Set the overlap counts
        results['Overlap Count'] = cumulative_relabeled.tolist()

        # Calculate overlap percentage
        results['Overlap Percentage'] = [
            (count / idx) * 100 if idx > 0 else 0 
            for idx, count in zip(range(1, total_samples + 1), cumulative_relabeled)
        ]

        return pd.DataFrame(results)

    # Generate data for all possible sample counts
    total_samples = len(tracin_infl)
    num_relabeled = len(relabeled_indices)

    # Use check_overlap_at_every_sample for TracIn method (ascending order)
    tracin_results = check_overlap_at_every_sample(
        tracin_infl.sort_values(by='influence', ascending=True),
        relabeled_indices,
        'TracIn'
    )

    # Use check_overlap_at_every_sample for TIM Last method (ascending order)
    tim_last_results = check_overlap_at_every_sample(
        tim_last_infl.sort_values(by='influence', ascending=True),
        relabeled_indices,
        'TIM Last'
    )

    # Calculate TIM Last descending order
    tim_last_infl_desc = tim_last_infl.sort_values(by='influence', ascending=False)
    tim_last_results_desc = check_overlap_at_every_sample(
        tim_last_infl_desc,
        relabeled_indices,
        'TIM Last'
    )

    # Visualize results - Combined plot with all methods
    import matplotlib.pyplot as plt

    # Set larger global font and font size
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 28  # Larger font
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['xtick.labelsize'] = 26
    plt.rcParams['ytick.labelsize'] = 26
    plt.rcParams['legend.fontsize'] = 26

    # Color list
    color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

    # Create main comparison plot: TracIn, TIM Last, reference line
    plt.figure(figsize=(12, 8))
    # Plotting
    line_tim, = plt.plot(tim_last_results['Sample Count'], tim_last_results['Overlap Count'], '-', label='TIM', color=color_list[1], linewidth=4)
    line_tracin, = plt.plot(tracin_results['Sample Count'], tracin_results['Overlap Count'], '-', label='TracIn', color=color_list[0], linewidth=4)
    plt.plot([0, total_samples], [0, num_relabeled], '--', color='gray', linewidth=3, label='_nolegend_')

    plt.xlabel('Number of Training Data Checked')
    plt.ylabel('Number of Mislabeled Data Identified')
    # Remove title, do not use plt.title()
    # Adjust legend order: TIM first, TracIn second
    plt.legend(handles=[line_tim, line_tracin])
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, 8000)
    plt.ylim(bottom=0)

    # Save figure
    plt.savefig(os.path.join(args.save_dir, f'cleansing_plot_{relabel_pct}_pct.png'), dpi=300)


except Exception as e:
    print(f"Error in cleansing_plot.py: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
