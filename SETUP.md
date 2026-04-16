# Environment Setup

This repository standardizes on Python `3.13.5` and uses `uv` for reproducible local environments.

The environment is intentionally scoped to the active local workflow:

- `01_curate_allergenicity_data.ipynb`
- `02_data_exploration_deepalgpro.ipynb`
- `03_baseline_model_colab.ipynb`
- `04_probing_metrics.ipynb`
- `baseline_notebook_utils.py`

## Prerequisites

- `uv` must be installed.
- `make` must be available on your machine.

If you use `pyenv`, this repo includes a `.python-version` file set to `3.13.5`.

## Create the environment

```bash
make setup
```

That command will:

- create `.venv`
- install the exact dependency set recorded in `uv.lock`

## Register the notebook kernel

If you use Jupyter or VS Code notebooks, register the environment once:

```bash
make kernel
```

That creates a user-level kernel named `xallergen2` with display name `Python (xallergen2)`.

If you prefer VS Code's interpreter picker, you can also select:

```text
.venv/bin/python
```

directly without registering a separate kernelspec.

## Verify the environment

```bash
make doctor
```

This verifies that the core notebook dependencies import correctly from the locked environment.

## Update the lockfile

If you intentionally change dependencies in `pyproject.toml`, refresh the lockfile with:

```bash
make lock
```

## Rebuild from scratch

```bash
make clean
make setup
```

## Notes

- `pyproject.toml` defines the direct notebook dependencies.
- `uv.lock` is the reproducibility artifact and should be committed.
- `ipykernel` is included so notebooks and scripts share the same environment.
- `captum` remains pinned because the probing workflow depends on integrated gradients.
- The local `uv` environment stays on NumPy 1.26 because `captum==0.8.0` still constrains NumPy to `<2`. The Colab notebooks handle their own runtime compatibility separately.
- The notebook code prefers an existing local Hugging Face cache for `facebook/esm2_t6_8M_UR50D` when present, but still falls back to the remote model ID for first-time downloads on machines with network access.
