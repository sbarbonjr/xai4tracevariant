import pm4py
import pandas as pd
import numpy as np

import pickle

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import euclidean

import numpy as np
from Levenshtein import distance as levenshtein_distance
from scipy.spatial.distance import euclidean
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


import networkx as nx

from skpm.event_feature_extraction import TimestampExtractor
from sklearn.feature_extraction.text import CountVectorizer

import time
from pathlib import Path
from tqdm import tqdm

class EL2GraphTime():
    def __init__(self, ocel_path, ocel_case_notion, config, k=3, plot=False):
        self.ocel_path = ocel_path
        self.ocel_case_notion = ocel_case_notion
        self.plot = plot
        self.graphs = []
        self.config = config
        self.k = k
        self.process()
        

    def process(self):

        database_name = Path(self.ocel_path).stem
        if 'sqlite' in self.ocel_path:
            ocel = pm4py.read_ocel_sqlite(self.ocel_path)
        else:
            ocel = pm4py.read_ocel2_json(self.ocel_path)
        filtered_ocel = pm4py.filter_ocel_object_attribute(ocel, 'ocel:type', [self.ocel_case_notion])
               
        df_filtered_ocel = filtered_ocel.get_extended_table().explode('ocel:type:' + self.ocel_case_notion).drop_duplicates().sort_values(["ocel:timestamp"])

        df_features = df_filtered_ocel.filter(["ocel:eid","ocel:activity","ocel:timestamp"]).iloc[:300,:]
        df_features.columns = ["case:concept:name","concept:name","time:timestamp"]
        df_features["case:concept:name"] = df_features["case:concept:name"].explode()
        
        self.config["vector_size"] = df_features["concept:name"].unique()
        act_features = self.run_onehot(df_features)
        act_features.columns = ["case"] + list(df_features["concept:name"].unique())
        
        tra_features = self.get_trace_transition_matrix(df_features)

        tex = TimestampExtractor()
        tex.fit(df_features.filter(["case:concept:name","time:timestamp"]))
        time_features = tex.transform(df_features.filter(["case:concept:name","time:timestamp"]))
        time_features = time_features.apply(pd.to_numeric, errors='coerce')
        time_features = pd.concat([df_features[["case:concept:name"]],time_features], axis=1)
        time_features = time_features.groupby(["case:concept:name"]).mean().round(4).reset_index() #mean of the time features to represent a case
        time_features.rename(columns={"case:concept:name": "case"}, inplace=True)
        
        # Ensure the "case" column is set as the index for all DataFrames
        print('Act_features:', act_features.shape)
        print('Tra_features:', tra_features.shape)
        print('Time_features:', time_features.shape)
        
        act_features.set_index("case", inplace=True)
        tra_features.set_index("case", inplace=True)
        time_features.set_index("case", inplace=True)

        # Merge the DataFrames using the "case" index for edit distance computation
        df_feature_vector = pd.concat([act_features, tra_features, time_features], axis=1)
        #df_feature_vector.to_csv("feature_matrix.csv")
        
        ########################################################
        # Writing the profiled dataset
        ########################################################
        df_feature_vector.to_csv(f"./results/{database_name}_profiled.csv", index=True)
        
        for wa in [0.5, 1.0]:
            for wt in [0.5, 1.0]:
                for wtime in [0.5, 1.0, 1.5, 2.0]:
                    distance_matrix = self.compute_distance_matrix(
                        act_features * wa,
                        tra_features * wt,
                        time_features * wtime
                    )
                    G = self.extract_knn_graph(distance_matrix, n_neighbors=self.k)
                    G = nx.relabel_nodes(G, {i: case_id for i, case_id in enumerate(df_feature_vector.index)})
                    filename = f"./results/{database_name}_k{self.k}_wa{wa}_wt{wt}_wtime{wtime}.graphml"
                    nx.write_graphml(G, filename)



    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
    
    def get_transitions(self, dataframe):
        dfg, _, _ = pm4py.discover_dfg(dataframe, case_id_key='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
        
        activities = list(set([activity for pair in dfg.keys() for activity in pair]))
        
        # Initialize a matrix with zeros
        dfg_matrix = pd.DataFrame(0, index=activities, columns=activities)
        
        # Fill the matrix with the DFG transitions
        for (activity_from, activity_to), count in dfg.items():
            dfg_matrix.at[activity_from, activity_to] = count
        return dfg_matrix    

    def get_trace_transition_matrix(self, dataframe):
        # Get all unique activities (nodes) in the DataFrame
        activities = list(dataframe['concept:name'].unique())

        # Group the DataFrame by cases (traces)
        grouped = dataframe.groupby("case:concept:name")
        
        transition_matrix_all = []
        case_ids = []  # To store the case names
        
        for case_id, group in grouped:
            transition_matrix = pd.DataFrame(0, index=activities, columns=activities)
            
            # Sort the group by timestamp
            group = group.sort_values(['time:timestamp'])

            # Iterate over the activities in the trace
            for i in range(len(group) - 1):
                activity_from = group.iloc[i]['concept:name']
                activity_to = group.iloc[i + 1]['concept:name']
                transition_matrix.at[activity_from, activity_to] += 1
            # Append the case ID and transition matrix to the lists
            case_ids.append(case_id)
            transition_matrix_all.append(transition_matrix.values.flatten())
        
        # Create a DataFrame with the case IDs as the first column
        transition_matrix_df = pd.DataFrame(transition_matrix_all, columns=[f"{a}->{b}" for a in activities for b in activities])
        transition_matrix_df.insert(0, "case", case_ids)
        
        return transition_matrix_df


    def get_knn_distance_matrix(self, binary_data, n_neighbors=3, metric='euclidean'):
        print("Extracting k-NN Graph...")
        start_time = time.time()
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        binary_data.columns = binary_data.columns.astype(str)
        knn.fit(binary_data)
        # Compute the distance matrix
        distance_matrix = knn.kneighbors_graph(binary_data, mode='distance').toarray()

        # Fill the diagonal with zeros since the distance from a point to itself is zero
        np.fill_diagonal(distance_matrix, 0)

        end_time = time.time() - start_time
        print(f"...k-NN Graph took {round(end_time, 2)} seconds")
        return distance_matrix


    def _pairwise_distances(self, args):
        i, s_act_i, s_tra_i, vec_time_i, act_strings, tra_strings, time_vectors, w_act, w_tra, w_time = args
        row = np.zeros(len(act_strings))
        for j in range(i + 1, len(act_strings)):
            dist_act = levenshtein_distance(s_act_i, act_strings[j])
            dist_tra = levenshtein_distance(s_tra_i, tra_strings[j])
            dist_time = euclidean(vec_time_i, time_vectors[j])
            dist = w_act * dist_act + w_tra * dist_tra + w_time * dist_time
            row[j] = dist
        return i, row

    def compute_distance_matrix(self, act_features, tra_features, time_features, w_act=1.0, w_tra=1.0, w_time=1.0):
        cases = act_features.index
        n_samples = len(cases)
        distance_matrix = np.zeros((n_samples, n_samples))

        # Preprocess for faster access
        act_strings = act_features.astype(str).agg(''.join, axis=1).tolist()
        tra_strings = tra_features.astype(str).agg(''.join, axis=1).tolist()
        time_vectors = time_features.to_numpy()

        args_list = [
            (i, act_strings[i], tra_strings[i], time_vectors[i], act_strings, tra_strings, time_vectors, w_act, w_tra, w_time)
            for i in range(n_samples)
        ]

        with Pool(int(cpu_count()/2)) as pool:
            for i, row in tqdm(pool.imap_unordered(self._pairwise_distances, args_list), total=n_samples, desc="Computing Distance Matrix"):
                distance_matrix[i] = row

        # Fill lower triangle
        i_lower = np.tril_indices(n_samples, -1)
        distance_matrix[i_lower] = distance_matrix.T[i_lower]

        return distance_matrix



    def extract_knn_graph(self, distance_matrix, n_neighbors):
        print("Extracting k-NN graph...")
        start_time = time.time()
        # Create a k-NN graph from the distance matrix
        spmatrix = kneighbors_graph(distance_matrix, n_neighbors=n_neighbors, metric='precomputed', mode='connectivity',)
        G = nx.from_scipy_sparse_array(spmatrix)
        end_time = time.time() - start_time
        print(f"...k-NN took {round(end_time, 2)} seconds")
        return G

    def run_onehot(self, log):
        ids, traces = self.extract_corpus(log)

        start_time = time.time()

        # onehot encoding
        corpus = CountVectorizer(analyzer="word", binary=True).fit_transform(traces)
        encoding = corpus.toarray()
        
        end_time = time.time() - start_time
        print(f"...One-hot encoding took {round(end_time, 2)} seconds")

        # formatting
        encoded_df = pd.DataFrame(encoding, columns=[f"{i}" for i in range(encoding.shape[1])])
        encoded_df.insert(0, "case", ids)

        return encoded_df
    

    def extract_corpus(self, log):
        traces, ids = [], []
        grouped = log.groupby("case:concept:name")

        for case_id, group in tqdm(grouped, desc="Extracting traces", total=log["case:concept:name"].nunique()):
            trace = " ".join(event.replace(" ", "") for event in group["concept:name"])
            ids.append(case_id)
            traces.append(trace)

        return ids, traces