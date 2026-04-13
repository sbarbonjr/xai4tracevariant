# Migration Guide: Old ADBIS Scripts → New Structure

## What Changed?

The project has been refactored to separate:

1. **Core Framework** (root directory)
   - `EL2GraphTime.py` — Trace profiling & similarity graphs
   - `ELExplainer.py` — Variant explanation visualizations
   - `util.py` — Shared utilities
   - **No changes to these files**

2. **Reproducibility Experiments** (new `experiments/adbis25/` directory)
   - All ADBIS 2025 scripts moved here
   - File names shortened (removed `adbis25_` prefix)
   - Path fixes added to find core modules

## Old → New File Mapping

| Old Location | New Location | Change |
|---|---|---|
| `adbis25_experiment.py` | `experiments/adbis25/experiment.py` | Moved, path fix added |
| `adbis25_profiler.py` | `experiments/adbis25/profiler.py` | Moved, path fix added |
| `adbis25_explaining.py` | `experiments/adbis25/explaining.py` | Moved, path fix added |
| `adbis25_local_explaining.py` | `experiments/adbis25/local_explaining.py` | Moved, path fix added |
| `adbis25_clustering.py` | `experiments/adbis25/clustering.py` | Moved, path fix added |
| `adbis25_scoring_clusters.py` | `experiments/adbis25/scoring_clusters.py` | Moved, path fix added |
| `run_multiple_k.sh` | `experiments/adbis25/run_multiple_k.sh` | Moved, script updated |

## How to Update Your Workflows

### Before Refactoring

```bash
# Run profiler from root
python adbis25_profiler.py --ocel_path ./adbis_datasets/base.sqlite --k 3 --ocel_case_notion cases

# Run experiment from root
python adbis25_experiment.py --ocel_path ./adbis_datasets/base.sqlite --k 3 --ocel_case_notion cases
```

### After Refactoring

```bash
# Same commands work! Just use the new paths
python experiments/adbis25/profiler.py --ocel_path ./adbis_datasets/base.sqlite --k 3 --ocel_case_notion cases

python experiments/adbis25/experiment.py --ocel_path ./adbis_datasets/base.sqlite --k 3 --ocel_case_notion cases
```

**Key**: Scripts still run from the repo **root**, but the Python files are now in `experiments/adbis25/`.

The `sys.path` fixes added to each script ensure they can still find `EL2GraphTime` and `ELExplainer`.

## Shell Script Updates

The `run_multiple_k.sh` script was updated to call the new path:

**Before:**
```bash
python3 adbis25_clustering.py --file "$FILE" --k "$K" --n_cpu 1
```

**After:**
```bash
python3 experiments/adbis25/clustering.py --file "$FILE" --k "$K" --n_cpu 1
```

To use it:
```bash
bash experiments/adbis25/run_multiple_k.sh
```

## Backward Compatibility

The **old files are still in the root directory** for now:
- `adbis25_experiment.py`
- `adbis25_profiler.py`
- `adbis25_explaining.py`
- `adbis25_local_explaining.py`
- `adbis25_clustering.py`
- `adbis25_scoring_clusters.py`
- `run_multiple_k.sh`

You can safely delete these once you've migrated all workflows to use the new structure.

## Documentation

- **Framework docs**: See [README.md](README.md)
- **Reproducibility guide**: See [experiments/adbis25/README.md](experiments/adbis25/README.md)

## Questions?

If you have scripts or workflows that depend on the old file locations:

1. Update import paths to use `experiments/adbis25/`
2. Scripts will still run from the repo root (no need to change working directories)
3. All command-line arguments remain the same

The refactoring keeps functionality intact while organizing code more clearly.
