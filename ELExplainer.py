import pandas as pd
import numpy as np
from pathlib import Path

class ELExplainer:
    def __init__(self, profile_df_path, representatives_df_path, metric):
        """
        Initialize the ELExplainer with profile data and representative cases.
        
        Args:
            profile_df_path (str): Path to the profile feature vectors CSV
            representatives_df_path (str): Path to the representatives CSV
        """
        self.metric = metric
        self.representatives_df = self._load_representatives_data(representatives_df_path)
        self.profile_df = self._load_profile_data(profile_df_path)
        
        
        
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
        if hasattr(self, 'representatives_df') and self.representatives_df is not None:
            valid_nodes = set(self.representatives_df[self.metric].astype(str))
            df = df[df.index.astype(str).isin(valid_nodes)]
        
        return df
    
    def _load_representatives_data(self, path):
        """Load and prepare representatives data"""
        df = pd.read_csv(path)
        required_cols = ['community', self.metric]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Representatives CSV must contain columns: {required_cols}")
        return df
    
    def explain_variants(self):
        print('representatives_df:', self.profile_df)
