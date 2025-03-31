import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_knn_graph(G):
    """
    Plot the k-NN graph using networkx.

    Parameters:
    G (networkx.Graph): The k-NN graph.
    """
    # Ensure the graph is valid and nodes have labels
    if not G.nodes:
        print("Graph is empty!")
        return

    pos = nx.spring_layout(G, seed=27)  # Use a seed for reproducible layouts
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_size=300, with_labels=True, node_color='skyblue', edge_color='gray', font_size=10, font_weight='bold')
    plt.title("k-NN Graph")
    plt.show()

def plot_graph(graph, partition, title):
    plt.figure(figsize=(8, 6))  # Set the figure size for better visibility
    pos = nx.spring_layout(graph, seed=30)  # Use a fixed layout for consistent visualization
    colors = [partition[node] for node in graph.nodes()]
    nx.draw(
        graph, pos, with_labels=True, node_color=colors, cmap=plt.cm.rainbow,  # Use a color map to differentiate communities
        node_size=500, font_size=10, edge_color='gray'  # Customize node size, font size, and edge color
    )
    plt.title(title, fontsize=16)
    #plt.show()
    return plt

def filter_bridge_nodes(bridge_nodes_stats, min_external_edges= 2, min_external_communities=1, exclude_nodes=None):
    """
    Filter bridge nodes based on criteria:
    - Minimum number of external edges.
    - Minimum number of distinct external communities.
    - Exclude specific nodes manually.
    """
    if exclude_nodes is None:
        exclude_nodes = []

    # # Debugging: Print the input DataFrame
    # print("Bridge Nodes Stats Before Filtering:")
    # print(bridge_nodes_stats)

    # Apply filters
    filtered_nodes_stats = bridge_nodes_stats[
        (bridge_nodes_stats["ExternalEdges"] >= min_external_edges) &
        (bridge_nodes_stats["DistinctExternalCommunities"] >= min_external_communities) &
        (~bridge_nodes_stats["Node"].isin(exclude_nodes))
    ]

    # Debugging: Print the filtered DataFrame
    #print("Filtered Bridge Nodes Stats:")
    #print(filtered_nodes_stats)

    return filtered_nodes_stats    

def split_bridge_nodes_into_communities(graph, partition, bridge_nodes_stats):
    max_community_id = max(partition.values())  # Start new IDs after the existing ones
    new_partition = partition.copy()
    for _, row in bridge_nodes_stats.iterrows():
        node = row["Node"]
        max_community_id += 1  # Assign a new unique community ID
        new_partition[node] = max_community_id
    return new_partition

# Function to analyze bridge nodes and return their stats
def analyze_bridge_nodes(graph, partition):
    bridge_stats = []
    for node in graph.nodes():
        neighbor_communities = {partition[neighbor] for neighbor in graph.neighbors(node)}
        total_edges = graph.degree[node]  # Total number of edges
        internal_edges = sum(1 for neighbor in graph.neighbors(node) if partition[neighbor] == partition[node])
        external_edges = total_edges - internal_edges  # Edges outside the community
        external_communities = len(neighbor_communities - {partition[node]})  # Distinct external communities
        if external_edges > 0:  # Only consider nodes with external edges
            bridge_stats.append({
                "Node": node,
                "TotalEdges": total_edges,
                "InternalEdges": internal_edges,
                "ExternalEdges": external_edges,
                "DistinctExternalCommunities": external_communities
            })
    return pd.DataFrame(bridge_stats)

def create_meta_graph(graph, partition, bridge_nodes_stats):
    """
    Create a meta-graph where each community becomes a single node.
    - The ID of the meta-node is the node with the highest degree in the community.
    - For bridge nodes forming individual communities, their ID remains the same.
    """
    # Group nodes by community
    communities = {}
    for node, community in partition.items():
        communities.setdefault(community, []).append(node)

    # Determine meta-node IDs based on the highest degree node in each community
    meta_node_mapping = {}
    for community_id, nodes in communities.items():
        highest_degree_node = max(nodes, key=lambda node: graph.degree[node])
        meta_node_mapping[community_id] = highest_degree_node

    # Create the meta-graph
    meta_graph = nx.Graph()

    # Add nodes to the meta-graph
    for community_id, meta_node in meta_node_mapping.items():
        meta_graph.add_node(meta_node, community_id=community_id)

    # Add edges between meta-nodes
    seen_edges = set()
    for u, v in graph.edges():
        community_u = partition[u]
        community_v = partition[v]
        if community_u != community_v:  # Only connect different communities
            meta_node_u = meta_node_mapping[community_u]
            meta_node_v = meta_node_mapping[community_v]
            if (meta_node_u, meta_node_v) not in seen_edges and (meta_node_v, meta_node_u) not in seen_edges:
                meta_graph.add_edge(meta_node_u, meta_node_v)
                seen_edges.add((meta_node_u, meta_node_v))

    return meta_graph, meta_node_mapping

def plot_meta_graph(meta_graph, meta_node_mapping, partition, bridge_nodes_stats, title):
    """
    Plot the meta-graph with:
    - Node sizes proportional to the number of nodes in the community.
    - Bridge nodes highlighted in red.
    """
    # Calculate community sizes
    community_sizes = {community: 0 for community in set(partition.values())}
    for node, community in partition.items():
        community_sizes[community] += 1

    # Determine node sizes
    node_sizes = [
        community_sizes[meta_graph.nodes[node]['community_id']] * 100 for node in meta_graph.nodes()
    ]

    # Identify bridge nodes
    bridge_nodes = set(bridge_nodes_stats["Node"])
    node_colors = [
        "red" if node in bridge_nodes else "blue" for node in meta_graph.nodes()
    ]

    # Generate positions for visualization
    pos = nx.spring_layout(meta_graph, seed=42)

    # Plot the meta-graph
    plt.figure(figsize=(10, 8))
    nx.draw(
        meta_graph, pos, with_labels=True,
        node_size=node_sizes, node_color=node_colors, edge_color="gray", font_size=10
    )

    plt.title(title, fontsize=16)
    #plt.show()

    return plt