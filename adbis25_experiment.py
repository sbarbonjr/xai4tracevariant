
import argparse
import networkx as nx
import numpy as np

from EL2GraphTime import EL2GraphTime

import networkx as nx
from community import community_louvain

import util as util

import neptune

import os
from pathlib import Path

PATH = PATH = Path(__file__).resolve().parent

def create_experiment(ocel_path, k, e, ocel_case_notion, config_input, use_neptune=False):
    
    if use_neptune:
        run = neptune.init_run(
            name=f"E_{k}nn",
            description=f"E_{k}nn",
            project="",
            api_token="",
        )  # your credentials


        run["k"] = k
        run["ocel"] = ocel_path
        run["external"] = e
    
    dataset = EL2GraphTime(
        ocel_path=ocel_path,
        k=k,
        ocel_case_notion=ocel_case_notion,
        config=config_input,
        plot=True
    )
    
    name = os.path.splitext(os.path.basename(ocel_path))[0]

    G = dataset.graphs[0]
    # Plot the k-NN graph
    util.plot_knn_graph(G)

    partition = community_louvain.best_partition(G, resolution=0.5)
    bridge_nodes_stats = util.analyze_bridge_nodes(G, partition)
    bridge_nodes_stats.sort_values("ExternalEdges", ascending=False)

    bridge_nodes_stats["ExternalEdges"] = bridge_nodes_stats["ExternalEdges"].astype(int)
    bridge_nodes_stats["DistinctExternalCommunities"] = bridge_nodes_stats["DistinctExternalCommunities"].astype(int)
    
    meta_graph, meta_node_mapping = util.create_meta_graph(G, partition,bridge_nodes_stats)
    
    # Plot the meta-graph
    fig = util.plot_meta_graph(
        meta_graph, meta_node_mapping, partition, bridge_nodes_stats,
        "Meta-Graph with Bridge Nodes Highlighted"
    )
    
    # #for mean_degree
    # partition_degrees = {part: [] for part in set(partition.values())}
    # for node in G.nodes():
    #     part = partition[node]
    #     partition_degrees[part].append(G.degree(node))
    # mean_degrees = {part: sum(degrees) / len(degrees) for part, degrees in partition_degrees.items()}
    # #for min_degree
    # node_degrees = {node: G.degree(node) for node in G.nodes()}
    # partition_min_degrees = {part: float('inf') for part in set(partition.values())}
    # for node, part in partition.items():
    #     partition_min_degrees[part] = min(partition_min_degrees[part], node_degrees[node])
    # #std degree
    # partition_std_degrees = {part: [] for part in set(partition.values())}
    # for node, part in partition.items():
    #     partition_std_degrees[part].append(node_degrees[node])

    # partition_std_degrees = {part: np.std(degrees) for part, degrees in partition_std_degrees.items()}


    # df_meta_graph = nx.to_pandas_adjacency(meta_graph)
    # df_meta_graph['partition'] = df_meta_graph.index.map(partition)

    # #calcolo la centralità
    # degree_centrality = nx.degree_centrality(G)
    # betweenness_centrality = nx.betweenness_centrality(G)
    # eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    # closeness_centrality = nx.closeness_centrality(G)
    # laplacian = nx.laplacian_centrality(G)
    
    # # Calcolo di un valore sicuro per alpha
    # max_eigenvalue = max(nx.adjacency_spectrum(G).real)
    # alpha_safe = 0.85 / max_eigenvalue  # Assicura la convergenza
    # katz_centrality = nx.katz_centrality(G, alpha=alpha_safe, beta=1.0, max_iter=5000, tol=1e-6)

    # #necessiere per calcolare sulle partizioni
    # information_centrality = {}
    # communicability_centrality = {}
    # second_order_centrality_dict = {}
    # subgraph_centrality_dict = {}
    # subgraph_centrality_exp_dict = {}
    # estrada_index_dict = {}
    # current_flow_betweenness_dict = {}
    # approx_current_flow_betweenness_dict = {}
    # current_flow_betweenness_subset_dict = {}
    # betweenness_centrality_subset_dict = {}
    # group_betweenness_dict = {}
    # group_closeness_dict = {}
    # group_degree_dict = {}
    # for community_id in set(partition.values()):
    #     nodes_in_community = [node for node, comm in partition.items() if comm == community_id]
    #     subgraph = G.subgraph(nodes_in_community)
    #     if nx.is_connected(subgraph):
    #         info_centrality = nx.information_centrality(subgraph)
    #         comm_betweenness = nx.communicability_betweenness_centrality(subgraph)
    #         soc = nx.second_order_centrality(subgraph)
        
    #         subgraph_centrality_dict.update(nx.subgraph_centrality(subgraph))
    #         subgraph_centrality_exp_dict.update(nx.subgraph_centrality_exp(subgraph))
    #         estrada_index_dict[community_id] = nx.estrada_index(subgraph)

    #         information_centrality.update(info_centrality)
    #         communicability_centrality.update(comm_betweenness)
    #         second_order_centrality_dict.update(soc)
    #         current_flow_betweenness_dict.update(nx.current_flow_betweenness_centrality(subgraph))
    #         approx_current_flow_betweenness_dict.update(nx.approximate_current_flow_betweenness_centrality(subgraph))

    #     # Seleziona un sottoinsieme di nodi per il calcolo della subset centrality
    #         if len(nodes_in_community) > 5:
    #             sources = nodes_in_community[:3]  # Prendi i primi 3 nodi
    #             targets = nodes_in_community[3:6]  # Prendi i successivi 3 nodi
    #             current_flow_betweenness_subset_dict.update(
    #                 nx.current_flow_betweenness_centrality_subset(subgraph, sources, targets)
    #             )
    #             betweenness_centrality_subset_dict.update(
    #                 nx.betweenness_centrality_subset(subgraph, sources, targets)
    #             )
    #     # Centralità di gruppo (usa un sottoinsieme di nodi)
    #     if len(nodes_in_community) > 3:
    #         group = nodes_in_community[:4]  # Seleziona 4 nodi per il gruppo
    #         group_betweenness_dict[community_id] = nx.group_betweenness_centrality(subgraph, group)
    #         group_closeness_dict[community_id] = nx.group_closeness_centrality(subgraph, group)
    #         group_degree_dict[community_id] = nx.group_degree_centrality(subgraph, group)

    # # Aggiunta 
    # df_meta_graph['degree'] = df_meta_graph.index.map(lambda node: G.degree[node])
    # df_meta_graph['degree_mean'] = df_meta_graph['partition'].map(mean_degrees)
    # df_meta_graph['degree_min'] = df_meta_graph['partition'].map(partition_min_degrees)
    # df_meta_graph['degree_std'] = df_meta_graph['partition'].map(partition_std_degrees)
    # df_meta_graph['degree_centrality'] = df_meta_graph.index.map(degree_centrality)
    # df_meta_graph['betweenness_centrality'] = df_meta_graph.index.map(betweenness_centrality)
    # df_meta_graph['eigenvector_centrality'] = df_meta_graph.index.map(eigenvector_centrality)
    # df_meta_graph['closeness_centrality'] = df_meta_graph.index.map(closeness_centrality)
    # df_meta_graph['katz_centrality'] = df_meta_graph.index.map(katz_centrality)
    # df_meta_graph['information_centrality'] = df_meta_graph.index.map(information_centrality)
    # df_meta_graph['laplacian_centrality'] = df_meta_graph.index.map(laplacian)
    # df_meta_graph['communicability_betweenness_centrality'] = df_meta_graph.index.map(communicability_centrality)

    # df_meta_graph.to_csv(f"./results/csv/df_TIME_meta_graph_{name}_{k}_nn.csv")
    
    # fig_path = f"./results/figures/TIME_meta_graph{name}_{k}nn_.png"
    # dir_path = os.path.dirname(fig_path)    
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)
    # fig.savefig(fig_path)

    if use_neptune:
        #run["nodes_csv"].track_files(csv_path)
        #run["plot_meta_graph"].upload(fig_path)
        run.stop()


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
    create_experiment(
        ocel_path= str(PATH) + args.ocel_path,
        k=args.k,
        e=args.e,
        ocel_case_notion=args.ocel_case_notion,
        config_input=config_input
    )


#python adbis25_experiment.py --ocel_path /datasets/variant_logs/total_newsir.sqlite --k 3 --ocel_case_notion patients --ocel_case_notion patients