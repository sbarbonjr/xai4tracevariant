import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import time
import random
import argparse
import numpy as np
import pandas as pd

from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from Levenshtein import distance as levenshtein_distance
from scipy.spatial.distance import pdist, squareform

from datasketch import MinHash
import ELExplainer


def compute_mixed_distance(a, b, n):
    edit_dist = sum(levenshtein_distance(str(x), str(y)) for x, y in zip(a[:n], b[:n]))
    eucl_dist = np.linalg.norm(np.array(a[n:], dtype=np.float64) - np.array(b[n:], dtype=np.float64)) if n < len(a) else 0.0
    return edit_dist + eucl_dist


def compute_minhash_signature(trace, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for event in trace:
        m.update(str(event).encode('utf8'))
    return m


def compute_euclidean_matrix(num_part):
    return squareform(pdist(num_part, metric='euclidean'))


def kmeans_custom_distance_optimized(df, k, n_jobs=-1):
    print("🔄 Running optimized k-means clustering...")
    start_time = time.time()

    columns = df.columns.tolist()
    n = max(i + 1 for i, col in enumerate(columns) if '->' in col)

    str_part = df.iloc[:, :n].astype(str).values.tolist()
    num_part = df.iloc[:, n:].astype(float)
    scaler = StandardScaler()
    num_part_scaled = scaler.fit_transform(num_part)
    data = [s + list(n) for s, n in zip(str_part, num_part_scaled)]

    centroids = random.sample(data, k)
    max_iters = 200

    for iteration in range(max_iters):
        # Assign clusters
        clusters = [[] for _ in range(k)]
        for i, point in enumerate(data):
            distances = [compute_mixed_distance(point, c, n) for c in centroids]
            nearest_centroid = np.argmin(distances)
            clusters[nearest_centroid].append(i)

        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
            else:
                cluster_points = [data[i] for i in cluster]
                str_part_cluster = [p[:n] for p in cluster_points]
                num_part_cluster = np.array([p[n:] for p in cluster_points])
                new_centroid_str = [Counter([p[i] for p in str_part_cluster]).most_common(1)[0][0] for i in range(n)]
                new_centroid_num = num_part_cluster.mean(axis=0)
                new_centroids.append(new_centroid_str + list(new_centroid_num))

        centroids = new_centroids

    # Final assignment
    result = np.zeros(len(data), dtype=int)
    for i, point in enumerate(data):
        distances = [compute_mixed_distance(point, c, n) for c in centroids]
        result[i] = np.argmin(distances)

    elapsed = time.time() - start_time
    print(f"✅ k-means completed in {elapsed:.2f}s")

    return result


def main():
    parser = argparse.ArgumentParser(description="Cluster traces using optimized k-means with mixed distance.")
    parser.add_argument('--file', type=str, required=True, help='CSV filename (prefix, e.g., "BPI2017O")')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters (default: 5)')
    parser.add_argument('--n_cpu', type=int, default=-1, help='Number of CPUs for parallel processing (default: -1 = all)')

    args = parser.parse_args()

    input_file = f"./results/{args.file}_profiled.csv"
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return

    print(f"📂 Loading {input_file}...")
    df = pd.read_csv(input_file, index_col=0)

    print(f"🔄 Clustering with k={args.k}...")
    cluster_labels = kmeans_custom_distance_optimized(df, args.k, n_jobs=args.n_cpu)

    df['cluster'] = cluster_labels
    output_file = f"./cluster_results/{args.file}_k{args.k}_clusters.csv"
    os.makedirs('./cluster_results/', exist_ok=True)
    df.to_csv(output_file)
    print(f"✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()
