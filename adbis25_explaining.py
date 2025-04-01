import pandas as pd
import ELExplainer as ELExplainer

# Initialize with your data files
explainer = ELExplainer.ELExplainer(
    profile_df_path="./results/total_newsir_profiled.csv",
    representatives_df_path="./community_results/total_newsir_k3_wa1.0_wt0.5_wtime1.0_representative.csv",
    metric = 'degree_rep'
)

# Calculate cluster means
#explainer.plot_time_explanation()
explainer.explain_variants()