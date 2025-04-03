import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from Levenshtein import distance as levenshtein_distance
import random

import ELExplainer as ELExplainer
import argparse
import glob
import os

def compute_mixed_distance(a, b, n):
    # Edit distance on first n elements
    edit_dist = sum(levenshtein_distance(str(x), str(y)) for x, y in zip(a[:n], b[:n]))

    # Euclidean distance on the rest
    if n < len(a):
        numeric_a = np.array(a[n:], dtype=np.float64)
        numeric_b = np.array(b[n:], dtype=np.float64)
        eucl_dist = np.linalg.norm(numeric_a - numeric_b)
    else:
        eucl_dist = 0.0

    return edit_dist + eucl_dist


def kmeans_custom_distance(df, k):
    columns = df.columns.tolist()
    n = max(i + 1 for i, col in enumerate(columns) if '->' in col)

    # Convert relevant columns to correct type for distance computation
    str_part = df.iloc[:, :n].astype(str).values.tolist()
    num_part = df.iloc[:, n:].astype(float)
    scaler = StandardScaler()
    num_part_scaled = scaler.fit_transform(num_part)
    data = [s + list(n) for s, n in zip(str_part, num_part_scaled)]

    # Initialize centroids randomly
    centroids = random.sample(data, k)

    max_iters = 200
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]

        for point in data:
            distances = [compute_mixed_distance(point, centroid, n) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)

        new_centroids = []
        for cluster in clusters:
            if not cluster:
                new_centroids.append(random.choice(data))
                continue

            # Compute new centroid
            new_centroid_str = []
            for i in range(n):
                values = [p[i] for p in cluster]
                most_common = max(set(values), key=values.count)
                new_centroid_str.append(most_common)

            new_centroid_num = np.mean([p[n:] for p in cluster], axis=0).tolist()
            new_centroids.append(new_centroid_str + new_centroid_num)

        if new_centroids == centroids:
            break
        centroids = new_centroids

    # Assign clusters
    cluster_labels = []
    for point in data:
        distances = [compute_mixed_distance(point, centroid, n) for centroid in centroids]
        cluster_labels.append(np.argmin(distances))

    df_out = df.copy()
    df_out['cluster'] = cluster_labels
    return df_out

def get_cluster_medoids(df_with_clusters, n, distance_func):
    """
    Given the clustered DataFrame, number of string-based columns (n),
    and a mixed distance function, return the medoid row for each cluster.
    
    Returns a DataFrame of medoids (one per cluster).
    """
    medoids = []

    for cluster_id in sorted(df_with_clusters['cluster'].unique()):
        cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster_id]

        # Prepare data for distance calculation
        str_part = cluster_df.iloc[:, :n].astype(str).values.tolist()
        num_part = cluster_df.iloc[:, n:-1].astype(float).values.tolist()  # exclude cluster column
        points = [s + n for s, n in zip(str_part, num_part)]

        # Apply PAM-like medoid selection
        min_total_dist = float('inf')
        best_idx = None
        for i, p1 in enumerate(points):
            total_dist = sum(distance_func(p1, p2, n) for j, p2 in enumerate(points) if i != j)
            if total_dist < min_total_dist:
                min_total_dist = total_dist
                best_idx = i

        medoid_row = cluster_df.iloc[best_idx]
        medoids.append(medoid_row)

    return pd.DataFrame(medoids)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate and plot k-NN graph using OC4LGraph.")
    parser.add_argument('--file', type=str, required=True, help='File to the OCEL JSON file.')
    parser.add_argument('--k', type=int, required=True, help='k to be used for clustering.')
    args = parser.parse_args()

# Configuration
#file = "tratime"
# = 5
file = args.file
k = args.k
input_path = f"./results/{file}_profiled.csv"
output_path = f"./cluster_results/{file}_medoids_k{k}.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load data
df = pd.read_csv(input_path)

# Cluster
df_clustered = kmeans_custom_distance(df, k)
print(df_clustered)

# Find position of last "->" column
n = max(i + 1 for i, col in enumerate(df_clustered.columns) if '->' in col)

# Get medoids
medoids_df = get_cluster_medoids(df_clustered, n, compute_mixed_distance)
print(medoids_df)

# Filter output
output = medoids_df[["case", "cluster"]]
output = output.rename(columns={"cluster": "community"})

# Save to CSV
output.to_csv(output_path, index=False)
print(f"✅ Medoids saved to {output_path}")

explainer = ELExplainer.ELExplainer(
    profile_df_path=input_path,
    representatives_df_path=output_path,
    metric='degree_rep',
    graph_based=False
)
# Calculate cluster means
#explainer.plot_time_explanation()
explainer.plot_explanation(explainer.profile_df, True)