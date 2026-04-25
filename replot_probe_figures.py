from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT_CANDIDATE = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT_CANDIDATE / "src"
for path in [SRC_DIR, PROJECT_ROOT_CANDIDATE]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from xallergen.baseline_notebook_utils import configure_matplotlib_cache, find_project_root
from xallergen.mtl_epitope_notebook_utils import replot_probe_figures_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate probe figures from saved probe-row CSVs without training or probing."
    )
    parser.add_argument(
        "--rows-csv",
        type=Path,
        default=None,
        help="Combined probe-row CSV. Defaults to results/probing/rows/all_models_probing_rows.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output root. Defaults to the project results directory.",
    )
    args = parser.parse_args()

    project_root = find_project_root(Path.cwd())
    configure_matplotlib_cache(project_root)
    rows_csv = args.rows_csv or project_root / "results" / "probing" / "rows" / "all_models_probing_rows.csv"
    output_dir = args.output_dir or project_root / "results"

    outputs = replot_probe_figures_from_csv(rows_csv, output_dir)
    print("Replotted probe figures from saved rows:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
