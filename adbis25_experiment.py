
import argparse
import networkx as nx
import numpy as np

from EL2GraphTime import EL2GraphTime

import networkx as nx
from community import community_louvain

import util as util

import pickle

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
    database_name = Path(ocel_path).stem
    database_name_list = list_files_with_prefix('./results/', database_name)

    for file in database_name_list:
        print('ocel_path>>>>>>>>>>>>', file)
        nx.read_graphml(file)
        #with open(file, 'rb') as f:
        #    G = pickle.load(f)


def list_files_with_prefix(directory, database_name):
    directory = Path(directory)
    return list(directory.glob(f"{database_name}*"))

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
    extract_profiling(
        ocel_path= str(PATH) + args.ocel_path,
        k=args.k,
        e=args.e,
        ocel_case_notion=args.ocel_case_notion,
        config_input=config_input
    )

    community_detection(
        args.ocel_path
    )

    


#python adbis25_experiment.py --ocel_path /datasets/variant_logs/total_newsir.sqlite --k 3 --ocel_case_notion patients --ocel_case_notion patients