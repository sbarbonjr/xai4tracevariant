import ELExplainer as ELExplainer
import argparse
import glob
import os


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--ocel_path', required=True, help="Path to OCEL log")
    parser.add_argument('--graph_based', action='store_true', help="Enable graph-based explanation")
    args = parser.parse_args()
    
    if (args.graph_based):
        pattern = f"./community_results/{args.ocel_path}*_representative.csv"
    else:
        pattern = f"./cluster_results/{args.ocel_path}*.csv"


    # Find all matching files
    matching_files = glob.glob(pattern)
    matching_files.sort()

    profile_path = f"./results/{args.ocel_path}_profiled.csv"
    for rep_file in matching_files:
        print('rep_file', rep_file)
        if (args.graph_based):
            base_name = os.path.basename(rep_file).replace("_representative.csv", "")
        else:
            base_name = os.path.basename(rep_file)

        # Initialize and run your explainer
        explainer = ELExplainer.ELExplainer(
            profile_df_path=profile_path,
            representatives_df_path=rep_file,
            metric='degree_rep',
            graph_based= args.graph_based
        )
        # Calculate cluster means
        #explainer.plot_time_explanation()
        explainer.plot_explanation(explainer.profile_df, True)