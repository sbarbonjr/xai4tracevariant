import ELExplainer as ELExplainer
import argparse
import glob
import os


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--ocel_path', required=True, help="Path to OCEL log")
    parser.add_argument('--graph_based', action='store_true', help="Enable graph-based explanation")
    parser.add_argument('--case_id', help="Enable graph-based explanation")
    parser.add_argument('--variant_case',help="Enable graph-based explanation")
    args = parser.parse_args()
    

    pattern = f"./community_results/{args.ocel_path}*_representative.csv"

    # Find all matching files
    matching_files = glob.glob(pattern)
    matching_files.sort()

    profile_path = f"./results/{args.ocel_path}_profiled.csv"
    rep_file = f'./community_results/tratime_k21_wa1_wt1_wtime0.5_numcom13_r1.0_representative.csv'
    # Initialize and run your explainer
    explainer = ELExplainer.ELExplainer(
        profile_df_path=profile_path,
        representatives_df_path=rep_file,
        metric='degree_rep',
        graph_based= args.graph_based
    )
    # Calculate cluster means
    #explainer.plot_time_explanation()
    #explainer.plot_explanation(explainer.profile_df, True)
    explainer.local_explanation(case=args.case_id, variant_case=args.variant_case)
        