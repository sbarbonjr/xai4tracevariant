# ADBIS 2025 Reproducibility

This directory contains scripts to reproduce the experiments from the paper:

> **Explaining Process Model Variants through Explainable AI**  
> Published in: *Advances in Databases and Information Systems (ADBIS 2025)*  
> DOI: [10.1007/978-3-032-05281-0_7](https://link.springer.com/chapter/10.1007/978-3-032-05281-0_7)

## Overview

The ADBIS 2025 paper demonstrates how the **XAI4TraceVariant** framework can explain differences between trace variants in event logs through a combination of:

1. Multi-dimensional trace profiling
2. k-NN graph construction with weighted similarity
3. Community detection via Louvain algorithm
4. Explainable visualizations

## Quick Start

All scripts should be run from the **repository root** (not from this directory):

```bash
cd /path/to/xai4tracevariant  # Navigate to root
```

## Dataset Preparation

Place your event logs in the `experiments/adbis25/datasets/` directory. The framework supports:

- **XES format**: `.xes` files (standard process mining format)
- **OCEL (Object-Centric Event Log)**: SQLite or JSON
  - `*.sqlite` — OCEL SQLite
  - `*.json` — OCEL JSON

Example datasets are included in `adbis_datasets/`:
- `base.sqlite` — Synthetic baseline
- `fullchange.sqlite` — Variant with structural changes
- `tratime.sqlite` — Variant with temporal changes
- `wabo.xes` — Real-world event log

## Workflow

The paper follows a **4-stage pipeline**:

### 1. Profiling (Extract Features)

```bash
python experiments/adbis25/profiler.py \
  --ocel_path /adbis_datasets/base.sqlite \
  --k 3 \
  --ocel_case_notion cases
```

**Outputs:**
- `./results/{dataset}_profiled.csv` — Feature matrix with activities, transitions, time

### 2. Community Detection (Find Variant Groups)

```bash
python experiments/adbis25/experiment.py \
  --ocel_path /adbis_datasets/base.sqlite \
  --k 3 \
  --ocel_case_notion cases \
  --encoding onehot \
  --aggregation average
```

**Outputs:**
- `./results/{dataset}_k{k}_wa{w}_wt{w}_wtime{w}.graphml` — k-NN graph
- `./community_results/{dataset}_modularity{m}_r{r}.csv` — Community assignments & centrality
- `./community_results/{dataset}_numcom{n}_r{r}_representative.csv` — Representative traces

### 3. Global Explanation (Understand Variant Patterns)

```bash
python experiments/adbis25/explaining.py \
  --ocel_path base \
  --graph_based
```

**Outputs:**
- `./results_img/{dataset}_explanation.png` — Heatmap showing how variants differ from mean

### 4. Local Explanation (Compare Specific Traces)

```bash
python experiments/adbis25/local_explaining.py \
  --ocel_path base \
  --case_id trace_123 \
  --variant_case trace_456 \
  --graph_based
```

**Outputs:**
- `./results_img/{dataset}_local_explanation.png` — Pairwise comparison heatmap

## Advanced: Clustering Approach

Alternatively, you can use a custom k-means clustering with mixed distance metrics:

```bash
# Run clustering for a single k value
python experiments/adbis25/clustering.py \
  --file BPI2017O \
  --k 5 \
  --n_cpu 1

# Or test multiple k values
bash experiments/adbis25/run_multiple_k.sh
```

Then evaluate clustering quality:

```bash
python experiments/adbis25/scoring_clusters.py \
  --prefix BPI2017O \
  --input_dir ./cluster_for_scoring_results/
```

## Command Reference

| Script | Purpose | Key Args |
|--------|---------|----------|
| `profiler.py` | Extract trace features | `--ocel_path`, `--k`, `--ocel_case_notion` |
| `experiment.py` | Full pipeline (profile + community detection) | `--ocel_path`, `--k`, `--ocel_case_notion`, `--encoding` |
| `explaining.py` | Generate global variant explanations | `--ocel_path`, `--graph_based` |
| `local_explaining.py` | Compare individual traces | `--ocel_path`, `--case_id`, `--variant_case` |
| `clustering.py` | Alternative: k-means clustering | `--file`, `--k`, `--n_cpu` |
| `scoring_clusters.py` | Evaluate clustering quality | `--prefix`, `--input_dir` |

## Output Structure

After running the full pipeline, expect:

```
.
├── results/
│   ├── {dataset}_profiled.csv
│   └── {dataset}_k{k}_wa{w}_wt{w}_wtime{w}.graphml
├── community_results/
│   ├── {dataset}_modularity{m}_r{r}.csv
│   └── {dataset}_numcom{n}_r{r}_representative.csv
├── results_img/
│   ├── {dataset}_explanation.png
│   └── {dataset}_local_explanation.png
└── cluster_results/
    └── {dataset}_k{k}_clusters.csv  (if using clustering)
```

## Paper Insights

Key findings from ADBIS 2025:

1. **White-box XAI**: The framework avoids black-box models—all explanations are derived from trace features
2. **Scalability**: Parallel distance computation handles large event logs efficiently
3. **Interpretability**: Segmented heatmaps make variant differences actionable for domain experts
4. **Flexibility**: Weighted combinations of activity, transition, and temporal features capture different variant types

## Reproduction Notes

- **Runtime**: Depends on log size. For the included datasets (1K–10K traces), expect minutes to hours
- **Memory**: The distance matrix scales as O(n²), so very large logs (>100K traces) may require subsampling
- **Determinism**: Community detection and k-means use random seeding. Set `np.random.seed()` for reproducibility
- **Paths**: All scripts assume outputs are written to `./results/`, `./community_results/`, etc. at the repo root

## Troubleshooting

**Import errors (EL2GraphTime not found)?**
- Ensure you're running scripts from the repo root: `cd /path/to/xai4tracevariant`
- Scripts in `experiments/adbis25/` automatically add the root to `sys.path`

**File not found errors?**
- Check that input paths like `--ocel_path` are relative to the repo root
- Example: `--ocel_path /adbis_datasets/base.sqlite` refers to `./adbis_datasets/base.sqlite`

**Graph visualization issues?**
- Ensure `graphviz` is installed: `pip install graphviz`
- If exporting to PDF, install system graphviz: `apt-get install graphviz` (Linux) or `brew install graphviz` (macOS)

## Contributing

Found a bug? Have a suggestion for improvement? Open an issue or submit a pull request to the [main repository](https://github.com/your-org/xai4tracevariant).

## Citation

If you use these scripts in your work, please cite the ADBIS 2025 paper:

```bibtex
@inproceedings{barbon2025explaining,
  title={Explaining Process Model Variants through Explainable AI},
  booktitle={Advances in Databases and Information Systems},
  pages={XX--XX},
  year={2025},
  publisher={Springer}
}
```

And reference XAI4TraceVariant:

```bibtex
@software{xai4tracevariant,
  title={XAI4TraceVariant: Explainable AI for Process Mining Trace Variants},
  url={https://github.com/your-org/xai4tracevariant},
  year={2025}
}
```

## License

This work is part of the XAI4TraceVariant project, licensed under the MIT License.

---

**Questions?** See the main [README.md](../../README.md) for framework documentation and usage.
