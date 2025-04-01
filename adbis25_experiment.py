
import argparse
import networkx as nx
import numpy as np

from EL2GraphTime import EL2GraphTime

import networkx as nx
from community import community_louvain  # python-louvain package
import pandas as pd
import util as util

from tqdm import tqdm 

import os
from pathlib import Path

PATH = PATH = Path(__file__).resolve().parent

def extract_profiling(ocel_path, k, e, ocel_case_notion, config_input):
    
    print('ocel_path', ocel_path)
    EL2GraphTime(
        ocel_path=ocel_path,
        k=k,
        ocel_case_notion=ocel_case_notion,
        config=config_input,
        plot=True
    )



def community_detection(ocel_path):
    """
    Detect communities in all GraphML files matching the database name using Louvain method.
    
    Args:
        ocel_path (str): Path to the original file (used to determine database name)
        
    Returns:
        dict: A dictionary with file paths as keys and community information as values
    """
    database_name = Path(ocel_path).stem
    database_name_list = list_files_with_prefix('./results/', database_name)
    
    communities_data = {}
    
    for file in database_name_list[:1]:
        combined_data = community_detection_with_centrality(file)
        
    return communities_data

def community_detection_with_centrality(ocel_path, resolution=0.5):
    """
    Enhanced community detection with node centrality analysis
    
    Args:
        ocel_path (str): Path to the original file
        resolution (float): Louvain resolution parameter (higher = more communities)
        
    Returns:
        tuple: (results_dict, combined_df)
    """
    # Setup paths and directories
    database_name = Path(ocel_path).stem
    database_name_list = list_files_with_prefix('./results/', database_name)
    output_dir = Path('./community_results/')
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    all_dfs = []
    
    for file in tqdm(database_name_list, desc="Processing files"):
        try:
            # Load and prepare graph
            G = nx.read_graphml(file)
            if nx.is_directed(G):
                G = G.to_undirected()
            
            # Community detection
            partition = community_louvain.best_partition(G, resolution=resolution)
            modularity = community_louvain.modularity(partition, G)
            num_communities = len(set(partition.values()))
            
            # Calculate all centrality measures
            centrality_measures = {
                'degree': nx.degree_centrality(G),
                'betweenness': nx.betweenness_centrality(G),
                'closeness': nx.closeness_centrality(G),
                'pagerank': nx.pagerank(G)
            }
            
            # Create node data DataFrame
            node_data = []
            for node in G.nodes():
                node_entry = {
                    'node': node,
                    'community': partition[node],
                    **{f'cent_{k}': v[node] for k, v in centrality_measures.items()},
                    **G.nodes[node]  # include original attributes
                }
                node_data.append(node_entry)
            
            df = pd.DataFrame(node_data)
            
            # Find representative nodes for each community
            representatives = {}
            for comm_id in set(partition.values()):
                comm_nodes = df[df['community'] == comm_id]
                representatives[comm_id] = {
                    'community': comm_id,
                    'degree_rep': comm_nodes.loc[comm_nodes['cent_degree'].idxmax()]['node'],
                    'betweenness_rep': comm_nodes.loc[comm_nodes['cent_betweenness'].idxmax()]['node'],
                    'pagerank_rep': comm_nodes.loc[comm_nodes['cent_pagerank'].idxmax()]['node']
                }
            
            # Save results
            csv_path = output_dir / f"{Path(file).stem}_communities.csv"
            df.to_csv(csv_path, index=False)

            csv_path = output_dir / f"{Path(file).stem}_representative.csv"
            pd.DataFrame(representatives).T.to_csv(csv_path, index=False)

            
        except Exception as e:
            print(f"\nError processing {file}: {str(e)}")
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs) if all_dfs else pd.DataFrame()
    
    return combined_df

def list_files_with_prefix(directory, database_name):
    directory = Path(directory)
    return list(directory.glob(f"{database_name}*.graphml"))

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate and plot k-NN graph using OC4LGraph.")
    
    # Define command-line arguments
    parser.add_argument('--ocel_path', type=str, required=True, help='Path to the OCEL JSON file.')
    parser.add_argument('--k', type=int, default=3, help='Number of nearest neighbors for k-NN graph (default: 3).')
    parser.add_argument('--e', type=int, default=2, help='Number of exsternal edges  (default: 2).')
    parser.add_argument('--ocel_case_notion', type=str, required=True, help='The case notion (e.g., "customers").')
    parser.add_argument('--encoding', type=str, default='onehot', help='Encoding type for nodes (default: "onehot").')
    parser.add_argument('--aggregation', type=str, default='average', help='Aggregation method for embeddings (default: "average").')
    parser.add_argument('--embed_from', type=str, default='nodes', help='Where to embed from (default: "nodes").')
    parser.add_argument('--edge_operator', type=str, default='average', help='Edge operator method (default: "average").')
    parser.add_argument('--use_neptune', type=bool, default='True', help='Edge operator method (default: "average").')

    # Parse command-line arguments
    args = parser.parse_args()

    # Create the config dictionary from parsed arguments
    config_input = {
        'encoding': args.encoding,
        'aggregation': args.aggregation,
        'embed_from': args.embed_from,
        'edge_operator': args.edge_operator
    }

    # Call the function with the command-line arguments
    #extract_profiling(
    #    ocel_path= str(PATH) + args.ocel_path,
    #    k=args.k,
    #    e=args.e,
    #    ocel_case_notion=args.ocel_case_notion,
    #    config_input=config_input
    #)

    community_detection(args.ocel_path)

    


#python adbis25_experiment.py --ocel_path /adbis_datasets/total_newsir.sqlite --k 3 --ocel_case_notion patients --ocel_case_notion patients