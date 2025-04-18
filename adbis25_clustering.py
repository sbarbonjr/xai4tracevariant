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


def compute_minhash_distance_matrix(traces, num_perm=128, n_jobs=5):
    print(f"🧠 Generating MinHash signatures for {len(traces)} traces...")
    signatures = Parallel(n_jobs=n_jobs)(
        delayed(compute_minhash_signature)(trace, num_perm)
        for trace in tqdm(traces, desc="MinHash signatures")
    )

    n = len(signatures)
    dist_matrix = np.zeros((n, n), dtype=np.float32)

    def compute_distance(i, j):
        sim = signatures[i].jaccard(signatures[j])
        return i, j, 1 - sim  # Convert similarity to distance

    print("⚡ Calculating pairwise MinHash distances...")
    index_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_distance)(i, j)
        for i, j in tqdm(index_pairs, desc="Pairwise distances", total=len(index_pairs))
    )

    for i, j, d in results:
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

    return dist_matrix


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


def get_cluster_medoids_optimized(df_with_clusters, n, n_jobs=-1):
    print("📍 Selecting medoids with MinHash + Euclidean distances...")
    medoids = []

    str_part_all = df_with_clusters.iloc[:, :n].astype(str).values.tolist()
    num_part_all = df_with_clusters.iloc[:, n:-1].astype(float).values

    minhash_matrix = compute_minhash_distance_matrix(str_part_all, n_jobs=n_jobs)
    eucl_matrix = compute_euclidean_matrix(num_part_all)
    full_distance_matrix = minhash_matrix + eucl_matrix

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
    medoids_df = get_cluster_medoids_optimized(df_clustered, n, n_jobs=n_cpu)

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
