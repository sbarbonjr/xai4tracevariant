import argparse
import os
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


def compute_clustering_metrics(df, label_column='cluster'):
    """
    Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz scores.
    """
    if label_column not in df.columns:
        raise ValueError(f"'{label_column}' column not found in DataFrame.")

    X = df.drop(columns=[label_column])
    y = df[label_column]

    if len(set(y)) < 2:
        raise ValueError("At least two clusters are required to compute clustering metrics.")

    X_scaled = StandardScaler().fit_transform(X)

    return {
        "Silhouette Score": silhouette_score(X_scaled, y).round(2),
        "Davies-Bouldin Score": davies_bouldin_score(X_scaled, y).round(2),
        "Calinski-Harabasz Score": calinski_harabasz_score(X_scaled, y).round(2)
    }


def main():
    parser = argparse.ArgumentParser(description="Batch clustering metric comparison for CSVs with similar prefix.")
    parser.add_argument('--prefix', type=str, required=True, help='Filename prefix to filter input files (e.g., BPI2017O)')
    parser.add_argument('--label_column', type=str, default='cluster', help='Name of the column with cluster labels')
    parser.add_argument('--input_dir', type=str, default='./cluster_for_scoring_results/', help='Directory containing input CSVs')

    args = parser.parse_args()
    prefix = args.prefix
    input_dir = args.input_dir
    label_column = args.label_column

    print(f"🔍 Searching for files in {input_dir} starting with '{prefix}'...")
    files = [f for f in os.listdir(input_dir) if f.startswith(prefix) and f.endswith('.csv')]

    if not files:
        print(f"❌ No CSV files found with prefix '{prefix}' in {input_dir}")
        return

    results = []

    for file in files:
        path = os.path.join(input_dir, file)
        try:
            df = pd.read_csv(path, index_col=0)
            metrics = compute_clustering_metrics(df, label_column)
            result = {
                "file": file,
                **metrics
            }
            results.append(result)
            print(f"✅ Processed: {file}")
        except Exception as e:
            print(f"⚠️ Skipping {file} due to error: {e}")

    if results:
        df_results = pd.DataFrame(results)
        output_path = os.path.join(input_dir, f"{prefix}_clustering_metrics_summary.csv")
        df_results.to_csv(output_path, index=False)
        print(f"\n📊 Summary saved to {output_path}")
    else:
        print("⚠️ No valid results to summarize.")


if __name__ == "__main__":
    main()
