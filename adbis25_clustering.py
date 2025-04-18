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
        clusters = [[] for _ in range(k)]
        print(f"🌀 Iteration {iteration + 1}/{max_iters}")

        def assign_cluster(point):
            distances = [compute_mixed_distance(point, centroid, n) for centroid in centroids]
            return np.argmin(distances)

        cluster_ids = Parallel(n_jobs=n_jobs)(
            delayed(assign_cluster)(point) for point in tqdm(data, desc="Assigning clusters")
        )

        for point, cid in zip(data, cluster_ids):
            clusters[cid].append(point)

        new_centroids = []
        for cluster in clusters:
            if not cluster:
                new_centroids.append(random.choice(data))
                continue

            new_str = []
            for i in range(n):
                values = [p[i] for p in cluster]
                most_common = Counter(values).most_common(1)[0][0]
                new_str.append(most_common)

            new_num = np.mean([p[n:] for p in cluster], axis=0).tolist()
            new_centroids.append(new_str + new_num)

        if new_centroids == centroids:
            print("✅ Converged.")
            break
        centroids = new_centroids

    cluster_labels = Parallel(n_jobs=n_jobs)(
        delayed(assign_cluster)(point) for point in tqdm(data, desc="Final cluster assignment")
    )

    df_out = df.copy()
    df_out['cluster'] = cluster_labels
    print(f"✅ Clustering completed in {round(time.time() - start_time, 2)}s.")
    return df_out


def get_cluster_medoids_minhash_per_cluster(df_with_clusters, n, n_jobs=-1, num_perm=128):
    print("📍 Selecting medoids (MinHash on demand, per cluster)...")
    medoids = []

    for cluster_id in sorted(df_with_clusters['cluster'].unique()):
        cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster_id]

        if len(cluster_df) == 1:
            medoids.append(cluster_df.iloc[0])
            continue

        str_part = cluster_df.iloc[:, :n].astype(str).values.tolist()
        num_part = cluster_df.iloc[:, n:-1].astype(float).values

        print(f"🔹 Processing cluster {cluster_id} with {len(str_part)} traces")

        # MinHash signatures
        signatures = Parallel(n_jobs=n_jobs)(
            delayed(compute_minhash_signature)(trace, num_perm) for trace in str_part
        )

        m = len(signatures)
        dist_matrix = np.zeros((m, m), dtype=np.float32)

        def compute_distance(i, j):
            sim = signatures[i].jaccard(signatures[j])
            dist = 1 - sim + np.linalg.norm(num_part[i] - num_part[j])
            return i, j, dist

        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_distance)(i, j) for i, j in tqdm(pairs, desc=f"Cluster {cluster_id}")
        )

        for i, j, d in results:
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

        dist_sums = dist_matrix.sum(axis=1)
        min_idx = np.argmin(dist_sums)
        medoids.append(cluster_df.iloc[min_idx])

    print("✅ Medoid selection completed (memory efficient).")
    return pd.DataFrame(medoids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and plot k-NN graph using OC4LGraph.")
    parser.add_argument('--file', type=str, required=True, help='File name prefix (used to read profiled CSV).')
    parser.add_argument('--k', type=int, required=True, help='Number of clusters (k).')
    parser.add_argument('--n_cpu', type=int, default=-1, help='Number of CPUs for parallel processing (default: use all available)')
    args = parser.parse_args()

    file = args.file
    k = args.k
    n_cpu = args.n_cpu

    input_path = f"./results/{file}_profiled.csv"
    output_path = f"./cluster_for_scoring_results/{file}_k{k}.csv"
    output_path_medoids = f"./cluster_results/{file}_medoids_k{k}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"📂 Loading data from {input_path}")
    df = pd.read_csv(input_path)

    df_clustered = kmeans_custom_distance_optimized(df, k, n_jobs=n_cpu)
    n = max(i + 1 for i, col in enumerate(df_clustered.columns) if '->' in col)
    df_clustered.to_csv(output_path, index=False)

    medoids_df = get_cluster_medoids_minhash_per_cluster(df_clustered, n, n_jobs=n_cpu)

    output = medoids_df[["case", "cluster"]].rename(columns={"cluster": "community"})
    output.to_csv(output_path_medoids, index=False)

    print(f"📂 Medoids saved to {output_path_medoids}")

    explainer = ELExplainer.ELExplainer(
        profile_df_path=input_path,
        representatives_df_path=output_path_medoids,
        metric='degree_rep',
        graph_based=False
    )

    print("📈 Generating explanation plots...")
    explainer.plot_explanation(explainer.profile_df, True)
    print("✅ All done!")
