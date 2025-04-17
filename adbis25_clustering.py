import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from Levenshtein import distance as levenshtein_distance
from joblib import Parallel, delayed
from collections import Counter
from tqdm import tqdm
import random
import time
import argparse
import hashlib
import ELExplainer as ELExplainer
from scipy.spatial.distance import pdist, squareform


def compute_mixed_distance(a, b, n):
    edit_dist = sum(levenshtein_distance(str(x), str(y)) for x, y in zip(a[:n], b[:n]))
    eucl_dist = np.linalg.norm(np.array(a[n:], dtype=np.float64) - np.array(b[n:], dtype=np.float64)) if n < len(a) else 0.0
    return edit_dist + eucl_dist


def fast_hash(trace, length=16):
    joined = "_".join(trace)
    return hashlib.md5(joined.encode()).hexdigest()[:length]


def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def compute_hamming_matrix_hashed(str_part, n_jobs=5):
    print("🧠 Hashing traces...")
    hashed_traces = [fast_hash(trace) for trace in str_part]
    n = len(hashed_traces)
    dist_matrix = np.zeros((n, n))

    def dist_pair(i, j):
        d = hamming_distance(hashed_traces[i], hashed_traces[j])
        return i, j, d

    print("⚡ Computing Hamming distances in parallel...")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(dist_pair)(i, j)
        for i in range(n)
        for j in range(i + 1, n)
    )

    for i, j, d in results:
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

    return dist_matrix


def compute_euclidean_matrix(num_part):
    return squareform(pdist(num_part, metric='euclidean'))


def kmeans_custom_distance_optimized(df, k, n_jobs=5):
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

        cluster_ids = Parallel(n_jobs=n_jobs)(delayed(assign_cluster)(point) for point in tqdm(data, desc="Assigning clusters"))
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

    cluster_labels = Parallel(n_jobs=n_jobs)(delayed(assign_cluster)(point) for point in tqdm(data, desc="Final cluster assignment"))
    df_out = df.copy()
    df_out['cluster'] = cluster_labels
    print(f"✅ Clustering completed in {round(time.time() - start_time, 2)}s.")
    return df_out


def get_cluster_medoids_optimized(df_with_clusters, n):
    print("📍 Selecting medoids with Hamming distance on hashed traces...")
    medoids = []

    str_part_all = df_with_clusters.iloc[:, :n].astype(str).values.tolist()
    num_part_all = df_with_clusters.iloc[:, n:-1].astype(float).values

    hamming_matrix = compute_hamming_matrix_hashed(str_part_all)
    eucl_matrix = compute_euclidean_matrix(num_part_all)
    full_distance_matrix = hamming_matrix + eucl_matrix

    index_map = dict(enumerate(df_with_clusters.index.tolist()))

    for cluster_id in sorted(df_with_clusters['cluster'].unique()):
        cluster_indices = df_with_clusters[df_with_clusters['cluster'] == cluster_id].index.tolist()
        if len(cluster_indices) == 1:
            medoids.append(df_with_clusters.loc[cluster_indices[0]])
            continue

        positions = [list(index_map.values()).index(idx) for idx in cluster_indices]
        submatrix = full_distance_matrix[np.ix_(positions, positions)]

        dist_sums = submatrix.sum(axis=1)
        min_idx = np.argmin(dist_sums)
        best_global_idx = cluster_indices[min_idx]

        medoids.append(df_with_clusters.loc[best_global_idx])

    print("✅ Medoid selection completed.")
    return pd.DataFrame(medoids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and plot k-NN graph using OC4LGraph.")
    parser.add_argument('--file', type=str, required=True, help='File name prefix (used to read profiled CSV).')
    parser.add_argument('--k', type=int, required=True, help='Number of clusters (k).')
    args = parser.parse_args()

    file = args.file
    k = args.k
    input_path = f"./results/{file}_profiled.csv"
    output_path = f"./cluster_results/{file}_medoids_k{k}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"📂 Loading data from {input_path}")
    df = pd.read_csv(input_path)

    df_clustered = kmeans_custom_distance_optimized(df, k)
    n = max(i + 1 for i, col in enumerate(df_clustered.columns) if '->' in col)
    medoids_df = get_cluster_medoids_optimized(df_clustered, n)

    output = medoids_df[["case", "cluster"]].rename(columns={"cluster": "community"})
    output.to_csv(output_path, index=False)
    print(f"💾 Medoids saved to {output_path}")

    explainer = ELExplainer.ELExplainer(
        profile_df_path=input_path,
        representatives_df_path=output_path,
        metric='degree_rep',
        graph_based=False
    )

    print("📈 Generating explanation plots...")
    explainer.plot_explanation(explainer.profile_df, True)
    print("✅ All done!")
