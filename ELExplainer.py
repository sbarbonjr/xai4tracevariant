import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

class ELExplainer:
    def __init__(self, profile_df_path, representatives_df_path, metric, graph_based):
        """
        Initialize the ELExplainer with profile data and representative cases.
        
        Args:
            profile_df_path (str): Path to the profile feature vectors CSV
            representatives_df_path (str): Path to the representatives CSV
        """
        self.metric = metric
        self.graph_based = graph_based
        self.representatives_df = self._load_representatives_data(representatives_df_path)
        self.profile_df = self._load_profile_data(profile_df_path)
        self.dtaset_name = Path(representatives_df_path).stem
        
    def _load_profile_data(self, path):
        """Load and prepare profile feature data, keeping only nodes present in representatives"""
        # Load the profile data
        df = pd.read_csv(path)
        
        # Handle column naming
        if 'case' not in df.columns and 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'case'})
        
        # Set case as index and filter
        df = df.set_index('case')

        # Only keep nodes that exist in representatives
        if self.graph_based:
            if hasattr(self, 'representatives_df') and self.representatives_df is not None:
                valid_nodes = set(self.representatives_df[self.metric].astype(str))
                df = df[df.index.astype(str).isin(valid_nodes)]
        else:
            print("self.representatives_df <<<", self.representatives_df)
            df = df[df.index.astype(str).isin(self.representatives_df['case'].astype(str))]
            print("df.index <<<", df.index)
            print("df <<<", df)

        return df
    
    def _load_representatives_data(self, path):
        """Load and prepare representatives data"""
        df = pd.read_csv(path)
        if self.graph_based:
            required_cols = ['community', self.metric]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Representatives CSV must contain columns: {required_cols}")
                self
        return df
    
    def explain_variants(self):
        #print('representatives_df:', self.profile_df)  
        self.plot_explanation(self.profile_df)

    def plot_explanation(self, df, save=False):
        # Step 1: Extract the first column as row labels
        row_labels = df.iloc[:, 0]
        df_data = df.iloc[:, 1:]

        # Step 2: Calculate the mean of each column
        col_means = df_data.mean(axis=0)

        # Step 3: Compute absolute difference from mean
        df_diff_from_mean = df_data.apply(lambda row: abs(row - col_means), axis=1)

        # Step 4: Reattach row labels
        df_diff_from_mean.insert(0, df.columns[0], row_labels)

        # Step 5: Dynamically find c1 and c2 (based on '->')
        arrow_cols = [i for i, col in enumerate(df_data.columns) if '->' in col]
        if not arrow_cols:
            raise ValueError("No columns with '->' found.")
        c1 = arrow_cols[0]
        c2 = arrow_cols[-1]

        # Step 6: Split segments
        segment_1 = df_diff_from_mean.iloc[:, 1:c1+1]
        segment_2 = df_diff_from_mean.iloc[:, c1+1:c2+1]
        segment_3 = df_diff_from_mean.iloc[:, c2+1:]

        segments = [
            (segment_1, 'Reds', 0),   # 20%
            (segment_2, 'Greens', 1), # 60%
            (segment_3, 'Blues', 2)   # 20%
        ]

        # Step 7: Create custom-sized subplots using GridSpec
        fig = plt.figure(figsize=(24, 10))
        spec = gridspec.GridSpec(ncols=10, nrows=1, figure=fig)
        widths = [2, 6, 2]  # 20%, 60%, 20%

        for idx, (segment, cmap, pos) in enumerate(segments):
            if segment.shape[1] == 0:
                continue
            ax = fig.add_subplot(spec[0, sum(widths[:pos]):sum(widths[:pos+1])])
            sns.heatmap(segment, annot=True, cmap=cmap, cbar=True, fmt=".1f",
                        linewidths=0.5, annot_kws={"size": 8, "rotation": 90}, ax=ax)
            if idx+1==1:
                title_segment = "Activities"
            elif idx+1==2:
                title_segment = "Transitions"
            else:
                title_segment = "Time-Features"
            ax.set_title(title_segment, fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

        plt.suptitle(f"Explaining Variants {self.dtaset_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            plt.savefig(f"./results_img/{self.dtaset_name}_explanation.png")
        else:
            plt.show()

