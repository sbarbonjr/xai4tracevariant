import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

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
            self.representatives_df['case'] = self.representatives_df['case'].astype(str).str.replace(r'\.0$', '', regex=True)
            print("self.representatives_df <<<", self.representatives_df)
            print("self.representatives_df <<<", self.representatives_df['case'])
            df = df[df.index.astype(str).isin(self.representatives_df['case'])]
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
    
    def local_explanation(self, case=None, variant_case=None):
        if case is None:
            case = self.profile_df.index[0]  # Default to the first case if none provided
        
        # Filter the profile dataframe for the specific case
        filtered_df = self.profile_df[self.profile_df.index == case]

        if variant_case is None:
            variant_case = self.profile_df.index[2]  # Default to the first case if none provided
        variant_case = self.profile_df[self.profile_df.index == variant_case]

        print("filtered_df <<<", filtered_df)   
        print("variant_case <<<", variant_case)

        # Plot the explanation for the specific case
        self.plot_local_explanation(filtered_df, variant_case,  save=True)
    
    def explain_variants(self):
        #print('representatives_df:', self.profile_df)  
        self.plot_explanation(self.profile_df)

    def get_custom_cmap(self, base_color):
        return LinearSegmentedColormap.from_list(
            f'white_to_{base_color}',
            ['white', base_color]
        )


    def plot_explanation(self, df, save=False):
        # Step 1: Extract the first column as row labels
        row_labels = df.iloc[:, 0]
        df_data = df.iloc[:, 1:]

        # Step 2: Calculate the mean of each column
        col_means = df_data.mean(axis=0)

        # Step 3: Compute absolute difference from mean
        df_diff_from_mean = df_data.apply(lambda row: (row - col_means), axis=1)

        # Step 4: Reattach row labels
        df_diff_from_mean.insert(0, df.columns[0], row_labels)

        # Step 5: Identify columns with '->'
        arrow_cols = [col for col in df_data.columns if '->' in col]
        if not arrow_cols:
            raise ValueError("No columns with '->' found.")

        # Determine columns with same prefix/suffix around '->'
        same_transition_cols = [col for col in arrow_cols if col.split('->')[0] == col.split('->')[1]]
        different_transition_cols = [col for col in arrow_cols if col not in same_transition_cols]

        all_cols = df_data.columns.tolist()
        idx_c1 = all_cols.index(arrow_cols[0])
        idx_c2 = all_cols.index(arrow_cols[-1])

        # Segment 1: Before any '->'
        segment_1 = df_diff_from_mean.iloc[:, 1:idx_c1+1]

        # Segment 2: Self-loop transitions (e.g., A->A)
        segment_2_cols = [col for col in same_transition_cols]
        segment_2 = df_diff_from_mean[segment_2_cols] if segment_2_cols else df_diff_from_mean.iloc[:, 0:0]

        # Segment 3: Other transitions
        segment_3_cols = [col for col in different_transition_cols]
        segment_3 = df_diff_from_mean[segment_3_cols] if segment_3_cols else df_diff_from_mean.iloc[:, 0:0]

        # Segment 4: Fixed columns right after last transition and tranforming time from seconds to hours
        segment_4 = (df_diff_from_mean.iloc[:, idx_c2+2:idx_c2+4]/3600)

        # Segment 5: The rest
        segment_5 = df_diff_from_mean.iloc[:, idx_c2+5:]

        segments = [
            (segment_1, 0),
            (segment_2, 1),
            (segment_3, 2),
            (segment_4, 3),
            (segment_5, 4)
        ]

        custom_cmaps = [
            self.get_custom_cmap('red'),
            self.get_custom_cmap('green'),
            self.get_custom_cmap('green'),
            'RdYlBu',
            'RdYlBu'
        ]

        labels = ['Difference from Mean (%)', 
                  'Difference from Mean (%)',
                  'Difference from Mean (%)',
                  'Difference from Mean (hours)',
                  'Difference from Mean (%)']

        widths = [2, 2, 3, 1, 2]  # Adjust according to number of columns per segment

        fig = plt.figure(figsize=(28, 10))
        spec = gridspec.GridSpec(ncols=sum(widths), nrows=1, figure=fig)

        for idx, (segment, pos) in enumerate(segments):
            if segment.shape[1] == 0:
                continue
            ax = fig.add_subplot(spec[0, sum(widths[:pos]):sum(widths[:pos+1])])
            
            seg_max = segment.values.max()
            seg_min = segment.values.min()
            if seg_max == 0 and seg_min == 0:
                vmin, vmax = 0, 1e-6  # Force minimal range to preserve white and fix legend
            else:
                vmin, vmax = seg_min, seg_max

            sns.heatmap(
                segment,
                annot=True,
                cmap=custom_cmaps[idx],
                cbar=True,
                fmt=".1f",
                linewidths=0.5,
                annot_kws={"size": 8, "rotation": 90},
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={"label": labels[idx]}
            )

            titles = ["Activities", "Self-loops", "Transitions", "Time (Case)", "Time (Avg. Event)"]
            ax.set_title(titles[idx], fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

        plt.suptitle(f"Explaining Variants {self.dtaset_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            plt.savefig(f"./results_img/{self.dtaset_name}_explanation.png")
        else:
            plt.show()

    def plot_local_explanation(self, df, variant_case, save=False):
        # Step 1: Extract the first column as row labels
        row_labels = df.iloc[:, 0]
        df_data = df.iloc[:, 1:]

        # Step 2: Extract only the data columns from variant_case
        col_variant = variant_case.iloc[:, 1:]

        # Step 3: Compute vectorial difference between df and variant_case
        vectorial_diff = df_data.subtract(col_variant.values, axis=1)

        # Save to instance variable if needed elsewhere
        self.vectorial_difference_df = vectorial_diff.copy()
        self.vectorial_difference_df.insert(0, df.columns[0], row_labels)

        # Step 4: Use this diff for the visualization
        df_diff_from_mean = self.vectorial_difference_df

        print("df_diff_from_mean <<<", df_diff_from_mean)

        # Step 5: Identify columns with '->'
        arrow_cols = [col for col in df_data.columns if '->' in col]
        if not arrow_cols:
            raise ValueError("No columns with '->' found.")

        # Determine columns with same prefix/suffix around '->'
        same_transition_cols = [col for col in arrow_cols if col.split('->')[0] == col.split('->')[1]]
        different_transition_cols = [col for col in arrow_cols if col not in same_transition_cols]

        all_cols = df_data.columns.tolist()
        idx_c1 = all_cols.index(arrow_cols[0])
        idx_c2 = all_cols.index(arrow_cols[-1])

        # Segment 1: Before any '->'
        segment_1 = df_diff_from_mean.iloc[:, 1:idx_c1+1]

        # Segment 2: Self-loop transitions (e.g., A->A)
        segment_2_cols = [col for col in same_transition_cols]
        segment_2 = df_diff_from_mean[segment_2_cols] if segment_2_cols else df_diff_from_mean.iloc[:, 0:0]

        # Segment 3: Other transitions
        segment_3_cols = [col for col in different_transition_cols]
        segment_3 = df_diff_from_mean[segment_3_cols] if segment_3_cols else df_diff_from_mean.iloc[:, 0:0]

        # Segment 4: Fixed columns right after last transition and tranforming time from seconds to hours
        segment_4 = (df_diff_from_mean.iloc[:, idx_c2+2:idx_c2+4]/3600)

        # Segment 5: The rest
        segment_5 = df_diff_from_mean.iloc[:, idx_c2+5:]

        segments = [
            (segment_1, 0),
            (segment_2, 1),
            (segment_3, 2),
            (segment_4, 3),
            (segment_5, 4)
        ]

        custom_cmaps = [
            self.get_custom_cmap('red'),
            self.get_custom_cmap('green'),
            self.get_custom_cmap('green'),
            'RdYlBu',
            'RdYlBu'
        ]

        labels = ['Difference from Mean (%)', 
                  'Difference from Mean (%)',
                  'Difference from Mean (%)',
                  'Difference from Mean (hours)',
                  'Difference from Mean (%)']

        widths = [1, 1, 1, 1, 1]  # Adjust according to number of columns per segment

        fig = plt.figure(figsize=(10, 16))
        spec = gridspec.GridSpec(ncols=1, nrows=sum(widths), figure=fig)

        for idx, (segment, pos) in enumerate(segments):
            if segment.shape[1] == 0:
                continue
            ax = fig.add_subplot(spec[sum(widths[:pos]):sum(widths[:pos+1]), 0])
            
            seg_max = segment.values.max()
            seg_min = segment.values.min()
            if seg_max == 0 and seg_min == 0:
                vmin, vmax = 0, 1e-6  # Force minimal range to preserve white and fix legend
            else:
                vmin, vmax = seg_min, seg_max

            sns.heatmap(
                segment,
                annot=True,
                cmap=custom_cmaps[idx],
                cbar=True,
                fmt=".1f",
                linewidths=0.5,
                annot_kws={"size": 12, "rotation": 0},
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={"label": labels[idx]}
            )
            titles = ["Activities", "Self-loops", "Transitions", "Time (Case)", "Time (Avg. Event)"]
            ax.set_title(titles[idx], fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

        plt.suptitle(f"Local Explanation of Variants {self.dtaset_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            plt.savefig(f"./results_img/{self.dtaset_name}_local_explanation.png")
        else:
            plt.show()

