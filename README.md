# XAllergen

Code for the paper *Residue-Level Attributions in Protein Allergenicity Models Do Not Recover Allergen Epitopes*.

## Repository Layout

```
data/                         Curated input tables, epitope labels, cache files
models/                       Trained baseline and MTL checkpoints
notebooks/                    Reproducible analysis notebooks
results/
  classification/             Protein-level training/evaluation metrics
  insilico_mutagenesis/       Saturation mutagenesis tables and figures
  paper_figures/              Main and supplementary paper figures
  paper_tables/               Main and supplementary paper tables
  probing/
    rows/                     Per-protein residue-localization metrics
    summaries/                Bootstrap summaries and comparison tables
Makefile                      Convenience commands
pyproject.toml                Project metadata and dependencies
replot_probe_figures.py       Figure replotting utility
SETUP.md                      Additional setup notes
src/xallergen/                Shared utilities
uv.lock                       Locked dependency versions
```

## Environment

Requires `uv` and Python 3.13.5.

```bash
uv sync
./.venv/bin/python -m ipykernel install --user --name xallergen2 --display-name "Python (xallergen2)"
```

Utilities in `src/xallergen` are added to `sys.path` automatically via `sitecustomize.py`.

## Workflow

Run notebooks in order to reproduce the full pipeline:

1. `notebooks/01_curate_allergenicity_data.ipynb`
2. `notebooks/02_data_exploration_deepalgpro.ipynb`
3. `notebooks/03_baseline_model_esm2.ipynb`
4. `notebooks/03_deep_plant_allergy_benchmark.ipynb`
5. `notebooks/04_mtl_epitope_supervision.ipynb`
6. `notebooks/05_mtl_top1_unfrozen_epitope_supervision.ipynb`
7. `notebooks/06_generate_probe_rows.ipynb`
8. `notebooks/07_insilico_mutagenesis.ipynb`
9. `notebooks/08_compare_all_model_probes.ipynb`

The retired baseline-top1-unfrozen baseline notebook is kept only for archival reference at `notebooks/Legacy scripts/03_baseline_top1_unfrozen_esm2.ipynb` and is not part of the current analysis workflow.

To regenerate figures without rerunning training or probing:

```bash
./.venv/bin/python replot_probe_figures.py
```

Reads from `results/probing/rows/all_models_probing_rows.csv` and writes figures to `results/figures/`. Use `--rows-csv` or `--output-dir` to override.

## Citation

A BibTeX entry will be added after publication.
