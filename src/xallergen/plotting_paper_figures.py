from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil
import tempfile

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)

from .mtl_epitope_notebook_utils import (
    MTLOutputPaths,
    bootstrap_mean_ci,
    ensure_label_variant_column,
    original_label_rows,
    summarize_probe_methods,
)


METHOD_PUBLICATION_LABELS = {
    "random_mean": "Random",
    "attention_weights": "Attention",
    "integrated_gradients": "IG",
    "gradient_x_input": "Grad×Input",
    "smoothgrad_ig": "SmoothGrad-IG",
    "occlusion": "Occlusion",
    "residue_head": "Residue head",
}

METHOD_CATEGORY_LABELS = {
    "random_mean": "Null baseline",
    "attention_weights": "Model-internal signal",
    "integrated_gradients": "Post-hoc attribution",
    "gradient_x_input": "Post-hoc attribution",
    "smoothgrad_ig": "Post-hoc attribution",
    "occlusion": "Perturbation sensitivity",
    "residue_head": "Supervised residue predictor",
}

MAIN_SIGNAL_SPECS = [
    ("random_mean", "Frozen ESM-2", "Random"),
    ("integrated_gradients", "Frozen ESM-2", "Frozen ESM-2 IG"),
    ("occlusion", "Frozen ESM-2", "Frozen ESM-2 occlusion"),
    ("residue_head", "MTL ESM-2", "MTL ESM-2 residue head"),
]

ACTIVE_METHOD_KEYS = (
    "random_mean",
    "attention_weights",
    "integrated_gradients",
    "gradient_x_input",
    "smoothgrad_ig",
    "occlusion",
    "residue_head",
)

ONE_COLUMN_FIGSIZE = (3.35, 2.8)
SHORT_FIGSIZE = (3.35, 2.45)
FONT_AXIS = 8
FONT_TICK = 7
FONT_LEGEND = 7

METRIC_COLOR_MAP = {
    "auroc": "#4C72B0",
    "auprc": "#DD8452",
    "precision_at_k": "#55A868",
}

MODEL_FAMILY_COLOR_MAP = {
    "Frozen ESM-2": "#4C72B0",
    "MTL ESM-2": "#DD8452",
    "MTL ESM-2 top 1": "#55A868",
    "DeepPlantAllergy": "#C44E52",
}

MODEL_FAMILY_DISPLAY_LABELS = {
    "Frozen ESM-2": "Frozen ESM-2",
    "MTL ESM-2": "MTL ESM-2",
    "MTL ESM-2 top 1": "MTL ESM-2\ntop 1",
    "DeepPlantAllergy": "DeepPlantAllergy",
}


def build_output_paths_for_supported_mtl(
    family_key: str,
    display_label: str,
    models_dir: Path,
    results_dir: Path,
    baseline_checkpoint_path: Path,
    baseline_summary_path: Path,
) -> MTLOutputPaths:
    if family_key != "mtl_frozen":
        raise ValueError(f"Unsupported MTL family_key for output path construction: {family_key}")
    prefix = "mtl"
    checkpoint_name = "mtl_frozen_esm2_epitope.pt"
    metrics_name = "mtl_baseline_metrics.json"
    baseline_rows_name = "baseline_probing_rows.csv"
    probe_rows_name = "mtl_probing_rows.csv"
    probe_summary_name = "mtl_probing_summary.csv"
    compare_summary_name = "mtl_vs_baseline_summary.csv"
    figure_prefix = "mtl_vs_baseline"

    return MTLOutputPaths(
        baseline_checkpoint_path=baseline_checkpoint_path,
        checkpoint_path=models_dir / checkpoint_name,
        metrics_path=results_dir / "classification" / metrics_name,
        probe_rows_path=results_dir / "probing" / "rows" / probe_rows_name,
        baseline_probe_rows_path=results_dir / "probing" / "rows" / baseline_rows_name,
        combined_probe_rows_path=None,
        probe_summary_path=results_dir / "probing" / "summaries" / probe_summary_name,
        compare_summary_path=results_dir / "probing" / "summaries" / compare_summary_name,
        combined_violins_png=results_dir / "figures" / "diagnostics" / f"{figure_prefix}_probing_violins.png",
        combined_auroc_density_png=results_dir / "figures" / "diagnostics" / f"{figure_prefix}_probing_auroc_vs_density.png",
        combined_auprc_density_png=results_dir / "figures" / "diagnostics" / f"{figure_prefix}_probing_auprc_vs_density.png",
        baseline_summary_csv=baseline_summary_path,
        mtl_family_label=display_label,
        baseline_family_label="Frozen ESM-2",
    )


def save_registry_probe_summary(
    probe_df: pd.DataFrame,
    summary_path: Path,
    allowed_methods: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    probe_df = ensure_label_variant_column(probe_df)
    available_methods = [
        method for method in allowed_methods if method in set(probe_df["method"].astype(str))
    ]
    if not available_methods:
        summary_df = pd.DataFrame()
    else:
        summary_df = summarize_probe_methods(probe_df, available_methods)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def _style_axes(ax) -> None:
    ax.tick_params(labelsize=FONT_TICK)
    ax.xaxis.label.set_size(FONT_AXIS)
    ax.yaxis.label.set_size(FONT_AXIS)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _format_mean_ci(mean_value: float, ci_low: float, ci_high: float) -> str:
    if pd.isna(mean_value):
        return "NA"
    return f"{mean_value:.3f} [{ci_low:.3f}, {ci_high:.3f}]"


def _write_table_outputs(df: pd.DataFrame, csv_path: Path, tex_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if tex_path is not None:
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        latex = df.to_latex(index=False, escape=False)
        tex_path.write_text(latex, encoding="utf-8")


def _safe_savefig(fig, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".tmp"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        fig.savefig(tmp_path, **kwargs)
        shutil.move(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _legend_below(ax, *, ncol: int = 2, y_offset: float = -0.2) -> None:
    ax.legend(
        frameon=False,
        fontsize=FONT_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        borderaxespad=0.0,
    )


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    ranked = np.asarray(p_values, dtype=float)[order]
    n_tests = len(ranked)
    adjusted = np.empty(n_tests, dtype=float)
    running_min = 1.0
    for idx in range(n_tests - 1, -1, -1):
        rank = idx + 1
        candidate = ranked[idx] * n_tests / rank
        running_min = min(running_min, candidate)
        adjusted[idx] = running_min
    q_values = np.empty(n_tests, dtype=float)
    q_values[order] = np.clip(adjusted, 0.0, 1.0)
    return q_values.tolist()


def _significance_marker(q_value: float) -> str:
    if pd.isna(q_value):
        return "NA"
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "ns"


def _format_q_value(q_value: float) -> str:
    if pd.isna(q_value):
        return "NA"
    if q_value < 0.001:
        return f"{q_value:.1e}"
    return f"{q_value:.3f}"


def _format_vs_random(mean_diff: float, q_value: float) -> str:
    marker = _significance_marker(q_value)
    if marker == "ns":
        return f"ns (q={_format_q_value(q_value)})"
    direction = "higher" if mean_diff > 0 else "lower"
    return f"{direction} {marker} (q={_format_q_value(q_value)})"


def _coalesce_sequence_lengths(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    if "sequence_length" in frame.columns:
        lengths = pd.to_numeric(frame["sequence_length"], errors="coerce")
    else:
        lengths = pd.Series(np.nan, index=frame.index, dtype=float)
    if "length" in frame.columns:
        lengths = lengths.fillna(pd.to_numeric(frame["length"], errors="coerce"))
    if "sequence" in frame.columns:
        lengths = lengths.fillna(frame["sequence"].astype(str).str.len().astype(float))
    return lengths.astype(float)


def _parse_pipe_delimited_ints(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    values: list[int] = []
    for token in str(value).split("|"):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            continue
    return values


def _load_dataset_artifacts(data_dir: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "positives": Path(data_dir) / "positives.csv",
        "positive_metadata": Path(data_dir) / "positives_all_epitope_metadata.csv",
        "positives_split_a": Path(data_dir) / "positives_splitA.csv",
        "positives_split_b": Path(data_dir) / "positives_splitB.csv",
        "negatives_split_a": Path(data_dir) / "negatives_splitA.csv",
        "negatives_split_b": Path(data_dir) / "negatives_splitB.csv",
    }
    artifacts: dict[str, pd.DataFrame] = {}
    for key, path in paths.items():
        artifacts[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return artifacts


def _canonical_model_family_label(label: object) -> str:
    text = str(label).strip()
    aliases = {
        "Frozen ESM-2": "Frozen ESM-2",
        "MTL ESM-2": "MTL ESM-2",
        "MTL ESM-2 top-1": "MTL ESM-2 top 1",
        "MTL ESM-2 top 1": "MTL ESM-2 top 1",
        "DeepPlantAllergy": "DeepPlantAllergy",
    }
    return aliases.get(text, text)


def _load_probe_rows_csv_with_label(rows_csv: Path, model_family_label: str | None = None) -> pd.DataFrame:
    header = pd.read_csv(rows_csv, nrows=0)
    columns = [
        "sequence_id",
        "accession",
        "seq_len",
        "epitope_density",
        "n_epitope_residues",
        "model_family",
        "method",
        "label_variant",
        "auroc",
        "auprc",
        "precision_at_k",
        "source_probe_rows_path",
        "family_key",
    ]
    usecols = [column for column in columns if column in header.columns]
    frame = ensure_label_variant_column(pd.read_csv(rows_csv, usecols=usecols))
    frame = frame.copy()
    if model_family_label is not None:
        frame["model_family"] = model_family_label
    elif "model_family" in frame.columns:
        frame["model_family"] = frame["model_family"].map(_canonical_model_family_label)
    frame["source_probe_rows_path"] = frame.get("source_probe_rows_path", str(rows_csv))
    if "source_probe_rows_path" in frame.columns:
        frame["source_probe_rows_path"] = frame["source_probe_rows_path"].fillna(str(rows_csv))
    return frame


def _augment_combined_probe_rows_from_siblings(combined_df: pd.DataFrame, combined_probe_rows_path: Path) -> pd.DataFrame:
    sibling_map = {
        "baseline_probing_rows.csv": "Frozen ESM-2",
        "mtl_probing_rows.csv": "MTL ESM-2",
        "mtl_top1_unfrozen_probing_rows.csv": "MTL ESM-2 top 1",
        "deep_plant_allergy_benchmark_probing_rows.csv": "DeepPlantAllergy",
    }
    working = ensure_label_variant_column(combined_df.copy())
    if "model_family" in working.columns:
        working["model_family"] = working["model_family"].map(_canonical_model_family_label)
    present_families = set(working.get("model_family", pd.Series(dtype=str)).dropna().astype(str))
    extra_frames: list[pd.DataFrame] = []
    rows_dir = Path(combined_probe_rows_path).parent
    for filename, model_family_label in sibling_map.items():
        probe_rows_path = rows_dir / filename
        if not probe_rows_path.exists():
            continue
        sibling_frame = _load_probe_rows_csv_with_label(probe_rows_path, model_family_label)
        if model_family_label not in present_families:
            print(
                f"Warning: {combined_probe_rows_path.name} is missing {model_family_label}; "
                f"loading rows from {probe_rows_path.name}."
            )
            extra_frames.append(sibling_frame)
            continue
        existing_methods = set(
            working.loc[working["model_family"].astype(str).eq(model_family_label), "method"]
            .dropna()
            .astype(str)
        )
        sibling_methods = set(sibling_frame["method"].dropna().astype(str))
        missing_methods = sorted(
            method for method in sibling_methods if method in ACTIVE_METHOD_KEYS and method not in existing_methods
        )
        if missing_methods:
            print(
                f"Warning: {combined_probe_rows_path.name} is missing {model_family_label} methods "
                f"{missing_methods}; backfilling from {probe_rows_path.name}."
            )
            extra_frames.append(
                sibling_frame.loc[sibling_frame["method"].astype(str).isin(missing_methods)].copy()
            )
    if extra_frames:
        working = ensure_label_variant_column(pd.concat([working, *extra_frames], ignore_index=True))
    return working


def _pick_probe_pair_key(frame: pd.DataFrame) -> str | None:
    for candidate in ("sequence_id", "accession", "entry"):
        if candidate in frame.columns and frame[candidate].notna().any():
            return candidate
    return None


def _metric_axis_limits(summary_df: pd.DataFrame, metric_key: str) -> tuple[float, float]:
    if metric_key == "auroc":
        return 0.0, 1.0
    metric_subset = summary_df.loc[summary_df["metric"] == metric_key].copy()
    if metric_subset.empty:
        return 0.0, 1.0
    max_value = float(
        np.nanmax(
            np.concatenate(
                [
                    metric_subset["ci_high"].to_numpy(dtype=float),
                    metric_subset["mean"].to_numpy(dtype=float),
                    metric_subset["random_mean"].dropna().to_numpy(dtype=float),
                ]
            )
        )
    )
    upper = min(1.0, max(0.25, max_value + 0.08))
    return 0.0, upper


def plot_supplementary_positive_dataset_profile(
    data_dir: Path,
    pdf_path: Path,
    png_path: Path,
) -> bool:
    import matplotlib.pyplot as plt

    artifacts = _load_dataset_artifacts(data_dir)
    positives_df = pd.concat(
        [artifacts["positives_split_a"], artifacts["positives_split_b"]],
        ignore_index=True,
    ).drop_duplicates(subset=["accession", "sequence"], keep="first")
    metadata_df = artifacts["positive_metadata"]
    if positives_df.empty or metadata_df.empty:
        return False

    split_positive_keys = set(
        positives_df[["accession", "sequence"]].astype(str).itertuples(index=False, name=None)
    )
    metadata_filtered = metadata_df.loc[
        metadata_df[["accession", "sequence"]].astype(str).apply(tuple, axis=1).isin(split_positive_keys)
    ].copy()
    coverage = pd.to_numeric(positives_df["epitope_coverage"], errors="coerce").dropna().astype(float)
    retained_epitopes = pd.to_numeric(
        metadata_filtered["n_epitopes_after_relative_filter"],
        errors="coerce",
    ).dropna().astype(float)
    individual_lengths = pd.Series(
        [
            length
            for value in metadata_filtered.get("epitope_lengths_raw", pd.Series(dtype=object))
            for length in _parse_pipe_delimited_ints(value)
        ],
        dtype=float,
    )
    scatter_df = positives_df.copy()
    scatter_df["sequence_length_plot"] = _coalesce_sequence_lengths(scatter_df)
    scatter_df["epitope_coverage_plot"] = pd.to_numeric(
        scatter_df.get("epitope_coverage", pd.Series(dtype=float)),
        errors="coerce",
    )
    scatter_df["n_epitopes_plot"] = pd.to_numeric(
        scatter_df.get("n_epitopes_after_relative_filter", pd.Series(dtype=float)),
        errors="coerce",
    )
    scatter_df = scatter_df.dropna(
        subset=["sequence_length_plot", "epitope_coverage_plot", "n_epitopes_plot"]
    ).copy()
    if coverage.empty or retained_epitopes.empty or individual_lengths.empty or scatter_df.empty:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8))
    ax_coverage, ax_counts, ax_lengths, ax_scatter = axes.ravel()

    ax_coverage.hist(coverage, bins=np.linspace(0.0, 0.75, 16), color="#4C72B0", alpha=0.90)
    ax_coverage.set_xlabel("Protein epitope coverage")
    ax_coverage.set_ylabel("Proteins")
    _style_axes(ax_coverage)

    count_bins = np.arange(0.5, float(retained_epitopes.max()) + 1.5, 1.0)
    ax_counts.hist(retained_epitopes, bins=count_bins, color="#55A868", alpha=0.90)
    ax_counts.set_xlabel("Retained epitopes per protein")
    ax_counts.set_ylabel("Proteins")
    _style_axes(ax_counts)

    ax_lengths.hist(individual_lengths, bins=np.arange(5.5, 46.5, 2.0), color="#DD8452", alpha=0.90)
    ax_lengths.set_xlabel("Individual epitope length (aa)")
    ax_lengths.set_ylabel("Epitopes")
    _style_axes(ax_lengths)

    scatter = ax_scatter.scatter(
        scatter_df["sequence_length_plot"],
        scatter_df["epitope_coverage_plot"],
        c=scatter_df["n_epitopes_plot"].clip(upper=20),
        cmap="viridis",
        s=14,
        alpha=0.85,
        linewidths=0.0,
    )
    ax_scatter.set_xlabel("Sequence length (aa)")
    ax_scatter.set_ylabel("Protein epitope coverage")
    _style_axes(ax_scatter)
    cbar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.04)
    cbar.set_label("Retained epitopes")
    cbar.ax.tick_params(labelsize=FONT_TICK)

    fig.tight_layout()
    _safe_savefig(fig, pdf_path, bbox_inches="tight")
    _safe_savefig(fig, png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _compute_main_alignment_significance(base_df: pd.DataFrame, signal_specs: list[tuple[str, str, str]]) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    metric_keys = ["auroc", "auprc", "precision_at_k"]
    test_rows: list[dict[str, float | str]] = []
    for method_key, family_label, signal_label in signal_specs:
        if method_key == "random_mean":
            continue
        subset = base_df[
            (base_df["model_family"] == family_label)
            & (base_df["method"] == method_key)
        ][["accession", *metric_keys]].copy()
        random_subset = base_df[
            (base_df["model_family"] == family_label)
            & (base_df["method"] == "random_mean")
        ][["accession", *metric_keys]].copy()
        paired_df = subset.merge(random_subset, on="accession", suffixes=("", "_random"))
        if paired_df.empty:
            continue
        for metric_key in metric_keys:
            diffs = paired_df[metric_key] - paired_df[f"{metric_key}_random"]
            nonzero_diffs = diffs.loc[diffs != 0]
            p_value = 1.0
            if not nonzero_diffs.empty:
                p_value = float(
                    wilcoxon(
                        paired_df[metric_key].to_numpy(dtype=float),
                        paired_df[f"{metric_key}_random"].to_numpy(dtype=float),
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )
            test_rows.append(
                {
                    "Signal": signal_label,
                    "metric_key": metric_key,
                    "mean_diff_vs_random": float(diffs.mean()),
                    "p_value_vs_random": p_value,
                }
            )

    significance_df = pd.DataFrame(test_rows)
    if significance_df.empty:
        return significance_df
    significance_df["q_value_vs_random"] = np.nan
    for metric_key, metric_df in significance_df.groupby("metric_key", sort=False):
        q_values = _benjamini_hochberg(metric_df["p_value_vs_random"].tolist())
        significance_df.loc[metric_df.index, "q_value_vs_random"] = q_values
    significance_df["vs_random_summary"] = significance_df.apply(
        lambda row: _format_vs_random(
            float(row["mean_diff_vs_random"]),
            float(row["q_value_vs_random"]),
        ),
        axis=1,
    )
    significance_df["vs_random_marker"] = significance_df["q_value_vs_random"].map(_significance_marker)
    return significance_df


def compute_residue_prevalence(frame: pd.DataFrame) -> float:
    base = original_label_rows(frame)
    if base.empty:
        return float("nan")
    unique_df = base[["accession", "seq_len", "n_epitope_residues"]].drop_duplicates("accession")
    total_residues = float(unique_df["seq_len"].sum())
    total_positive = float(unique_df["n_epitope_residues"].sum())
    return total_positive / total_residues if total_residues > 0 else float("nan")


def summarize_main_residue_alignment_subset(
    all_probe_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    base_df = original_label_rows(all_probe_df)
    prevalence = compute_residue_prevalence(base_df)

    signal_specs = list(MAIN_SIGNAL_SPECS)
    deepplant_ig = base_df[
        (base_df["model_family"] == "DeepPlantAllergy")
        & (base_df["method"] == "integrated_gradients")
    ]
    deepplant_attention = base_df[
        (base_df["model_family"] == "DeepPlantAllergy")
        & (base_df["method"] == "attention_weights")
    ]
    if not deepplant_ig.empty:
        signal_specs.append(("integrated_gradients", "DeepPlantAllergy", "DeepPlantAllergy IG"))
    elif not deepplant_attention.empty:
        signal_specs.append(("attention_weights", "DeepPlantAllergy", "DeepPlantAllergy Attention"))
    else:
        raise ValueError(
            "Notebook 07 requires a DeepPlantAllergy comparison in the main residue-alignment plot, "
            "but no DeepPlantAllergy IG or attention probe rows were found."
        )

    rows = []
    for method_key, family_label, signal_label in signal_specs:
        subset = base_df[
            (base_df["model_family"] == family_label)
            & (base_df["method"] == method_key)
        ].copy()
        if subset.empty:
            continue
        auroc_mean, auroc_ci_low, auroc_ci_high = bootstrap_mean_ci(subset["auroc"])
        auprc_mean, auprc_ci_low, auprc_ci_high = bootstrap_mean_ci(subset["auprc"])
        precision_mean, precision_ci_low, precision_ci_high = bootstrap_mean_ci(subset["precision_at_k"])
        rows.append(
            {
                "Signal": signal_label,
                "Category": METHOD_CATEGORY_LABELS[method_key],
                "model_family": family_label,
                "method_key": method_key,
                "n_proteins": int(subset["accession"].nunique()),
                "auroc_mean": auroc_mean,
                "auroc_ci_low": auroc_ci_low,
                "auroc_ci_high": auroc_ci_high,
                "auprc_mean": auprc_mean,
                "auprc_ci_low": auprc_ci_low,
                "auprc_ci_high": auprc_ci_high,
                "precision_at_k_mean": precision_mean,
                "precision_at_k_ci_low": precision_ci_low,
                "precision_at_k_ci_high": precision_ci_high,
            }
        )
    summary_df = pd.DataFrame(rows)
    significance_df = _compute_main_alignment_significance(base_df, signal_specs)
    if significance_df.empty:
        return summary_df, prevalence
    pivot_source = significance_df[
        ["Signal", "metric_key", "mean_diff_vs_random", "p_value_vs_random", "q_value_vs_random", "vs_random_summary", "vs_random_marker"]
    ].copy()
    pivot_df = pivot_source.pivot(index="Signal", columns="metric_key")
    pivot_df.columns = [f"{metric_key}_{stat_name}" for stat_name, metric_key in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    summary_df = summary_df.merge(pivot_df, on="Signal", how="left")
    return summary_df, prevalence


def write_main_residue_alignment_table(
    summary_df: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
) -> pd.DataFrame:
    table_df = pd.DataFrame(
        {
            "Signal": summary_df["Signal"],
            "Category": summary_df["Category"],
            "n_proteins": summary_df["n_proteins"],
            "AUROC mean with 95% CI": [
                _format_mean_ci(row.auroc_mean, row.auroc_ci_low, row.auroc_ci_high)
                for row in summary_df.itertuples(index=False)
            ],
            "AUPRC mean with 95% CI": [
                _format_mean_ci(row.auprc_mean, row.auprc_ci_low, row.auprc_ci_high)
                for row in summary_df.itertuples(index=False)
            ],
            "Precision@k mean with 95% CI": [
                _format_mean_ci(
                    row.precision_at_k_mean,
                    row.precision_at_k_ci_low,
                    row.precision_at_k_ci_high,
                )
                for row in summary_df.itertuples(index=False)
            ],
            "AUROC vs random": summary_df.get("auroc_vs_random_summary", pd.Series(["NA"] * len(summary_df))),
            "AUPRC vs random": summary_df.get("auprc_vs_random_summary", pd.Series(["NA"] * len(summary_df))),
            "Precision@k vs random": summary_df.get("precision_at_k_vs_random_summary", pd.Series(["NA"] * len(summary_df))),
        }
    )
    _write_table_outputs(table_df, csv_path, tex_path)
    return table_df


def plot_main_residue_alignment_subset(
    summary_df: pd.DataFrame,
    prevalence: float,
    pdf_path: Path,
    png_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    metric_keys = ["auroc", "auprc", "precision_at_k"]
    metric_labels = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "precision_at_k": "Precision@k",
    }
    signal_order = list(summary_df["Signal"])
    y_positions = np.arange(len(signal_order), dtype=float)
    offsets = {"auroc": -0.18, "auprc": 0.0, "precision_at_k": 0.18}
    x_min, x_max = 0.0, 1.0
    annotation_pad = 0.015
    edge_pad = 0.01

    fig, ax = plt.subplots(figsize=(ONE_COLUMN_FIGSIZE[0], 3.35))
    for metric_key in metric_keys:
        means = summary_df[f"{metric_key}_mean"].to_numpy(dtype=float)
        ci_low = summary_df[f"{metric_key}_ci_low"].to_numpy(dtype=float)
        ci_high = summary_df[f"{metric_key}_ci_high"].to_numpy(dtype=float)
        xerr = np.vstack([means - ci_low, ci_high - means])
        ax.errorbar(
            means,
            y_positions + offsets[metric_key],
            xerr=xerr,
            fmt="o",
            ms=4.5,
            linewidth=1.2,
            capsize=2.8,
            color=METRIC_COLOR_MAP[metric_key],
            label=metric_labels[metric_key],
        )
        marker_col = f"{metric_key}_vs_random_marker"
        diff_col = f"{metric_key}_mean_diff_vs_random"
        if marker_col in summary_df.columns and diff_col in summary_df.columns:
            for mean_value, ci_low_value, ci_high_value, y_pos, marker_value, mean_diff in zip(
                means,
                ci_low,
                ci_high,
                y_positions + offsets[metric_key],
                summary_df[marker_col],
                summary_df[diff_col],
            ):
                if pd.isna(marker_value) or str(marker_value) == "ns":
                    continue
                direction = "↑" if float(mean_diff) > 0 else "↓"
                right_x = min(float(ci_high_value) + annotation_pad, x_max - edge_pad)
                left_x = max(float(ci_low_value) - annotation_pad, x_min + edge_pad)
                place_right = float(ci_high_value) + annotation_pad <= x_max - edge_pad
                ax.text(
                    right_x if place_right else left_x,
                    float(y_pos),
                    f"{direction}{marker_value}",
                    color=METRIC_COLOR_MAP[metric_key],
                    fontsize=max(FONT_TICK - 0.3, 6.0),
                    ha="left" if place_right else "right",
                    va="center",
                )

    ax.axvline(0.5, color="#7F7F7F", linestyle="--", linewidth=1.0, label="AUROC random baseline")
    if not pd.isna(prevalence):
        ax.axvline(
            prevalence,
            color="#B07AA1",
            linestyle=":",
            linewidth=1.0,
            label="Residue prevalence baseline",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(signal_order, fontsize=FONT_TICK)
    ax.set_xlabel("Score", labelpad=2)
    ax.set_xlim(x_min, x_max)
    ax.invert_yaxis()
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    handle_map = {label: handle for handle, label in zip(handles, labels)}
    baseline_labels = ["AUROC random baseline", "Residue prevalence baseline"]
    metric_labels_ordered = ["AUROC", "AUPRC", "Precision@k"]
    baseline_handles = [handle_map[label] for label in baseline_labels if label in handle_map]
    baseline_text = [label for label in baseline_labels if label in handle_map]
    metric_handles = [handle_map[label] for label in metric_labels_ordered if label in handle_map]
    metric_text = [label for label in metric_labels_ordered if label in handle_map]
    dummy_handle = Line2D([], [], linestyle="none", linewidth=0, alpha=0)
    ax.legend(
        [*baseline_handles, dummy_handle, *metric_handles],
        [*baseline_text, "", *metric_text],
        frameon=False,
        fontsize=FONT_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.30),
        ncol=2,
        columnspacing=1.8,
        handletextpad=0.8,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.26, 1.0, 1.0))
    _safe_savefig(fig, pdf_path, bbox_inches="tight")
    _safe_savefig(fig, png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_main_protein_performance_table(
    metrics_df: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
) -> pd.DataFrame:
    ordered_models = ["Frozen ESM-2", "MTL ESM-2", "DeepPlantAllergy"]
    if metrics_df.empty:
        table_df = pd.DataFrame(
            columns=["Model", "AUROC", "Precision", "Recall", "F1", "MCC", "Accuracy", "n_test_sequences"]
        )
        _write_table_outputs(table_df, csv_path, tex_path)
        return table_df

    metrics_df = metrics_df.copy()
    metrics_df["_order"] = metrics_df["Model"].map({label: idx for idx, label in enumerate(ordered_models)})
    metrics_df = metrics_df.sort_values("_order").drop(columns="_order")
    for column in ["AUROC", "Precision", "Recall", "F1", "MCC", "Accuracy"]:
        if column in metrics_df.columns:
            metrics_df[column] = metrics_df[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "NA")
    table_df = metrics_df[
        ["Model", "AUROC", "Precision", "Recall", "F1", "MCC", "Accuracy", "n_test_sequences"]
    ].copy()
    _write_table_outputs(table_df, csv_path, tex_path)
    return table_df


def write_supplementary_signal_tables(
    all_probe_df: pd.DataFrame,
    csv_path: Path,
    tex_path: Path | None = None,
) -> pd.DataFrame:
    probe_df = ensure_label_variant_column(all_probe_df)
    probe_df = probe_df.loc[probe_df["method"].isin(ACTIVE_METHOD_KEYS)].copy()
    rows = []
    for group_key, subset in probe_df.groupby(["model_family", "method", "label_variant"], dropna=False):
        model_family, method_key, label_variant = group_key
        auroc_mean, auroc_ci_low, auroc_ci_high = bootstrap_mean_ci(subset["auroc"])
        auprc_mean, auprc_ci_low, auprc_ci_high = bootstrap_mean_ci(subset["auprc"])
        precision_mean, precision_ci_low, precision_ci_high = bootstrap_mean_ci(subset["precision_at_k"])
        prevalence = compute_residue_prevalence(subset)
        rows.append(
            {
                "model_family": model_family,
                "method": METHOD_PUBLICATION_LABELS.get(method_key, method_key),
                "method_key": method_key,
                "method_category": METHOD_CATEGORY_LABELS.get(method_key, "Uncategorized"),
                "label_variant": label_variant,
                "n_proteins": int(subset["accession"].nunique()),
                "AUROC mean": auroc_mean,
                "AUROC 95% CI": _format_mean_ci(auroc_mean, auroc_ci_low, auroc_ci_high),
                "AUPRC mean": auprc_mean,
                "AUPRC 95% CI": _format_mean_ci(auprc_mean, auprc_ci_low, auprc_ci_high),
                "Precision@k mean": precision_mean,
                "Precision@k 95% CI": _format_mean_ci(
                    precision_mean,
                    precision_ci_low,
                    precision_ci_high,
                ),
                "residue epitope prevalence": prevalence,
            }
        )
    table_df = pd.DataFrame(rows)
    if not table_df.empty:
        model_order = {
            "Frozen ESM-2": 0,
            "MTL ESM-2": 1,
            "DeepPlantAllergy": 2,
            "MTL ESM-2 top-1": 3,
        }
        method_order = {label: idx for idx, label in enumerate(METHOD_PUBLICATION_LABELS.values())}
        table_df["_model_order"] = table_df["model_family"].map(model_order).fillna(99)
        table_df["_method_order"] = table_df["method"].map(method_order).fillna(99)
        table_df["_label_order"] = table_df["label_variant"].map({"original": 0, "scrambled": 1}).fillna(2)
        table_df = table_df.sort_values(["_model_order", "_method_order", "_label_order"]).drop(
            columns=["_model_order", "_method_order", "_label_order"]
        )
    _write_table_outputs(table_df, csv_path, tex_path)
    return table_df


def plot_supplementary_all_signals_significance(
    combined_probe_rows_path: Path,
    pdf_path: Path,
    png_path: Path,
    summary_csv_path: Path | None = None,
    summary_tex_path: Path | None = None,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import wilcoxon

    combined_probe_rows_path = Path(combined_probe_rows_path)
    combined_df = _load_probe_rows_csv_with_label(combined_probe_rows_path)
    combined_df = _augment_combined_probe_rows_from_siblings(combined_df, combined_probe_rows_path)
    combined_df = ensure_label_variant_column(combined_df)
    if "label_variant" in combined_df.columns:
        combined_df = combined_df.loc[combined_df["label_variant"].astype(str).eq("original")].copy()
    combined_df = combined_df.loc[combined_df["method"].isin(ACTIVE_METHOD_KEYS)].copy()
    if combined_df.empty:
        print("Warning: no active original-label probe rows were found for supplementary all-signals significance.")
        return pd.DataFrame(
            columns=[
                "model_family",
                "method",
                "method_label",
                "metric",
                "mean",
                "ci_low",
                "ci_high",
                "random_mean",
                "mean_diff_vs_random",
                "n_pairs",
                "p_value",
                "q_value",
                "significance",
            ]
        )

    pair_key = _pick_probe_pair_key(combined_df)
    if pair_key is None:
        print("Warning: no sequence_id/accession/entry column found; skipping paired significance tests.")

    family_order = {
        "Frozen ESM-2": 0,
        "MTL ESM-2": 1,
        "MTL ESM-2 top 1": 2,
        "DeepPlantAllergy": 3,
    }
    method_order = {
        "random_mean": 0,
        "attention_weights": 1,
        "integrated_gradients": 2,
        "gradient_x_input": 3,
        "smoothgrad_ig": 4,
        "occlusion": 5,
        "residue_head": 6,
    }
    metric_keys = ["auroc", "auprc", "precision_at_k"]
    metric_display = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "precision_at_k": "Precision@k",
    }
    method_axis_labels = {
        "random_mean": "Random",
        "attention_weights": "Attention",
        "integrated_gradients": "IG",
        "gradient_x_input": "Grad×Input",
        "smoothgrad_ig": "SmoothGrad-IG",
        "occlusion": "Occlusion",
        "residue_head": "Residue head",
    }

    random_lookup: dict[tuple[str, str], float] = {}
    for model_family, subset in combined_df.groupby("model_family", dropna=False):
        random_subset = subset.loc[subset["method"].astype(str).eq("random_mean")].copy()
        for metric_key in metric_keys:
            if random_subset.empty:
                random_lookup[(str(model_family), metric_key)] = float("nan")
            else:
                random_lookup[(str(model_family), metric_key)] = float(random_subset[metric_key].mean())

    summary_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    available_pairs = (
        combined_df[["model_family", "method"]]
        .drop_duplicates()
        .assign(
            _family_order=lambda frame: frame["model_family"].map(family_order).fillna(99),
            _method_order=lambda frame: frame["method"].map(method_order).fillna(99),
        )
        .sort_values(["_family_order", "_method_order", "model_family", "method"])
        .drop(columns=["_family_order", "_method_order"])
    )

    for pair in available_pairs.itertuples(index=False):
        model_family = str(pair.model_family)
        method_key = str(pair.method)
        subset = combined_df.loc[
            combined_df["model_family"].astype(str).eq(model_family)
            & combined_df["method"].astype(str).eq(method_key)
        ].copy()
        if subset.empty:
            continue
        for metric_key in metric_keys:
            mean_value, ci_low, ci_high = bootstrap_mean_ci(subset[metric_key])
            random_mean = random_lookup.get((model_family, metric_key), float("nan"))
            summary_row = {
                "model_family": model_family,
                "method": method_key,
                "method_label": METHOD_PUBLICATION_LABELS.get(method_key, method_key),
                "metric": metric_display[metric_key],
                "metric_key": metric_key,
                "mean": mean_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "random_mean": random_mean,
                "mean_diff_vs_random": float(mean_value - random_mean) if pd.notna(random_mean) else float("nan"),
                "n_pairs": np.nan,
                "p_value": np.nan,
                "q_value": np.nan,
                "significance": "NA",
            }
            if method_key != "random_mean" and pair_key is not None:
                random_subset = combined_df.loc[
                    combined_df["model_family"].astype(str).eq(model_family)
                    & combined_df["method"].astype(str).eq("random_mean")
                ][[pair_key, metric_key]].copy()
                if random_subset.empty:
                    print(f"Warning: {model_family} lacks random_mean; skipping {metric_display[metric_key]} test for {method_key}.")
                else:
                    method_subset = subset[[pair_key, metric_key]].copy()
                    paired_df = method_subset.merge(
                        random_subset,
                        on=pair_key,
                        suffixes=("", "_random"),
                    ).dropna(subset=[metric_key, f"{metric_key}_random"])
                    n_pairs = int(len(paired_df))
                    summary_row["n_pairs"] = n_pairs
                    if n_pairs < 3:
                        print(
                            f"Warning: only {n_pairs} paired proteins for {model_family} | {method_key} | "
                            f"{metric_display[metric_key]}; skipping Wilcoxon test."
                        )
                    else:
                        diffs = paired_df[metric_key] - paired_df[f"{metric_key}_random"]
                        nonzero_diffs = diffs.loc[diffs != 0]
                        p_value = 1.0
                        if not nonzero_diffs.empty:
                            p_value = float(
                                wilcoxon(
                                    paired_df[metric_key].to_numpy(dtype=float),
                                    paired_df[f"{metric_key}_random"].to_numpy(dtype=float),
                                    alternative="two-sided",
                                    zero_method="wilcox",
                                ).pvalue
                            )
                        summary_row["p_value"] = p_value
                        summary_row["mean_diff_vs_random"] = float(diffs.mean())
                        test_rows.append(
                            {
                                "model_family": model_family,
                                "method": method_key,
                                "metric_key": metric_key,
                                "p_value": p_value,
                            }
                        )
            summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        print("Warning: no supplementary all-signals summary rows could be computed.")
        return summary_df

    if test_rows:
        test_df = pd.DataFrame(test_rows)
        q_value_map: dict[tuple[str, str, str], float] = {}
        for metric_key, metric_tests in test_df.groupby("metric_key", sort=False):
            q_values = _benjamini_hochberg(metric_tests["p_value"].tolist())
            for row, q_value in zip(metric_tests.itertuples(index=False), q_values):
                q_value_map[(str(row.model_family), str(row.method), str(metric_key))] = float(q_value)
        summary_df["q_value"] = [
            q_value_map.get((str(row.model_family), str(row.method), str(row.metric_key)), np.nan)
            for row in summary_df.itertuples(index=False)
        ]
        summary_df["significance"] = [
            _significance_marker(value) if pd.notna(value) else ("NA" if row.method != "random_mean" else "")
            for value, row in zip(summary_df["q_value"], summary_df.itertuples(index=False))
        ]
    else:
        summary_df["significance"] = [""] * len(summary_df)

    summary_df["_family_order"] = summary_df["model_family"].map(family_order).fillna(99)
    summary_df["_method_order"] = summary_df["method"].map(method_order).fillna(99)
    summary_df["_metric_order"] = summary_df["metric_key"].map({key: idx for idx, key in enumerate(metric_keys)})
    summary_df["method_axis_label"] = summary_df["method"].map(lambda value: method_axis_labels.get(str(value), str(value)))
    summary_df["row_label"] = summary_df["method_axis_label"]
    summary_df = summary_df.sort_values(
        ["_family_order", "_method_order", "_metric_order", "model_family", "row_label"]
    ).reset_index(drop=True)

    export_df = summary_df[
        [
            "model_family",
            "method",
            "method_label",
            "metric",
            "mean",
            "ci_low",
            "ci_high",
            "random_mean",
            "mean_diff_vs_random",
            "n_pairs",
            "p_value",
            "q_value",
            "significance",
        ]
    ].copy()
    if summary_csv_path is not None:
        summary_csv_path = Path(summary_csv_path)
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(summary_csv_path, index=False)
    if summary_tex_path is not None:
        tex_df = export_df.copy()
        tex_df["mean [95% CI]"] = [
            _format_mean_ci(row.mean, row.ci_low, row.ci_high)
            for row in tex_df.itertuples(index=False)
        ]
        tex_df["q_value"] = tex_df["q_value"].map(lambda value: _format_q_value(value) if pd.notna(value) else "NA")
        tex_df["mean_diff_vs_random"] = tex_df["mean_diff_vs_random"].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else "NA"
        )
        tex_table = tex_df[
            [
                "model_family",
                "method_label",
                "metric",
                "mean [95% CI]",
                "mean_diff_vs_random",
                "n_pairs",
                "q_value",
                "significance",
            ]
        ].rename(
            columns={
                "model_family": "Model",
                "method_label": "Signal",
                "metric": "Metric",
                "mean_diff_vs_random": "Δ vs random",
                "n_pairs": "n pairs",
                "q_value": "q",
                "significance": "Sig.",
            }
        )
        Path(summary_tex_path).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_tex_path).write_text(tex_table.to_latex(index=False, escape=False), encoding="utf-8")

    row_order = (
        summary_df[["model_family", "row_label", "_family_order", "_method_order"]]
        .drop_duplicates()
        .sort_values(["_family_order", "_method_order", "row_label"])
    )
    ordered_rows = list(row_order.itertuples(index=False))
    row_positions: list[float] = []
    row_method_labels: list[str] = []
    family_blocks: list[dict[str, float | str]] = []
    label_to_y: dict[tuple[str, str], float] = {}
    current_y = 0.0
    block_start = 0.0
    previous_family: str | None = None
    separator_positions: list[float] = []
    for row in ordered_rows:
        family = str(row.model_family)
        method_label = str(row.row_label)
        if previous_family is not None and family != previous_family:
            family_blocks.append(
                {
                    "family": previous_family,
                    "start": block_start,
                    "end": current_y - 1.0,
                    "mid": (block_start + current_y - 1.0) / 2.0,
                }
            )
            separator_positions.append(current_y - 0.5)
            current_y += 0.8
            block_start = current_y
        row_positions.append(current_y)
        row_method_labels.append(method_label)
        label_to_y[(family, method_label)] = current_y
        previous_family = family
        current_y += 1.0
    if previous_family is not None:
        family_blocks.append(
            {
                "family": previous_family,
                "start": block_start,
                "end": current_y - 1.0,
                "mid": (block_start + current_y - 1.0) / 2.0,
            }
        )

    fig_height = max(5.4, 0.31 * current_y + 1.45)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, fig_height), sharey=True)

    for ax, metric_key in zip(axes, metric_keys):
        metric_name = metric_display[metric_key]
        metric_df = summary_df.loc[summary_df["metric_key"] == metric_key].copy()
        metric_df["y"] = [
            label_to_y[(str(row.model_family), str(row.row_label))]
            for row in metric_df.itertuples(index=False)
        ]
        metric_df = metric_df.sort_values("y")
        for separator_y in separator_positions:
            ax.axhline(separator_y, color="#D9D9D9", linewidth=0.8, zorder=0)
        for row in metric_df.itertuples(index=False):
            family_color = MODEL_FAMILY_COLOR_MAP.get(str(row.model_family), "#4C72B0")
            ax.errorbar(
                [float(row.mean)],
                [float(row.y)],
                xerr=[[float(row.mean - row.ci_low)], [float(row.ci_high - row.mean)]],
                fmt="o",
                ms=4.4,
                linewidth=1.2,
                capsize=2.5,
                color=family_color,
                ecolor=family_color,
            )
            if row.method == "random_mean" or pd.isna(row.q_value) or str(row.significance) in {"", "NA", "ns"}:
                continue
            direction = "↑" if float(row.mean_diff_vs_random) > 0 else "↓"
            label = f"{direction}{row.significance}"
            x_low, x_high = _metric_axis_limits(summary_df, metric_key)
            pad = max(0.012, 0.028 * (x_high - x_low))
            text_x = min(float(row.ci_high) + pad, x_high - 0.01 * (x_high - x_low))
            ha = "left"
            if float(row.ci_high) + pad >= x_high - 0.01 * (x_high - x_low):
                text_x = max(float(row.ci_low) - pad, x_low + 0.01 * (x_high - x_low))
                ha = "right"
            ax.text(
                text_x,
                float(row.y),
                label,
                color="#333333",
                fontsize=max(FONT_TICK + 0.2, 7.0),
                ha=ha,
                va="center",
            )
        if metric_key == "auroc":
            ax.axvline(0.5, color="#7F7F7F", linestyle="--", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(metric_name)
        ax.set_title(metric_name, fontsize=FONT_AXIS, pad=12)
        _style_axes(ax)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(current_y - 0.5, -1.5)

    axes[0].set_yticks(row_positions)
    axes[0].set_yticklabels(row_method_labels, fontsize=FONT_TICK)
    axes[0].set_ylabel("")
    for ax in axes[1:]:
        ax.set_yticks(row_positions)
        ax.tick_params(axis="y", labelleft=False)
    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5.5, color=color, label=family)
        for family, color in MODEL_FAMILY_COLOR_MAP.items()
        if family in set(summary_df["model_family"].astype(str))
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(4, max(1, len(legend_handles))),
        frameon=False,
        fontsize=FONT_LEGEND,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.34, right=0.98, top=0.96, bottom=0.10, wspace=0.25)
    _safe_savefig(fig, pdf_path, bbox_inches="tight")
    _safe_savefig(fig, png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return export_df


def plot_supplementary_all_signals_heatmap(
    all_probe_df: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
) -> bool:
    import matplotlib.pyplot as plt
    import seaborn as sns

    probe_df = original_label_rows(all_probe_df)
    probe_df = probe_df.loc[probe_df["method"].isin(ACTIVE_METHOD_KEYS)].copy()
    if probe_df.empty:
        return False
    summary_df = summarize_probe_methods(probe_df, list(ACTIVE_METHOD_KEYS))
    if summary_df.empty:
        return False
    plot_df = summary_df[["model_family", "method", "auprc_mean"]].copy()
    plot_df["method"] = plot_df["method"].map(lambda value: METHOD_PUBLICATION_LABELS.get(value, value))
    heatmap_df = plot_df.pivot(index="method", columns="model_family", values="auprc_mean")
    if heatmap_df.shape[0] < 2 or heatmap_df.shape[1] < 2:
        return False

    fig, ax = plt.subplots(figsize=(3.35, 2.9))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="viridis", cbar=True, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _style_axes(ax)
    fig.tight_layout()
    _safe_savefig(fig, pdf_path, bbox_inches="tight")
    _safe_savefig(fig, png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_supplementary_label_scrambling_sanity_check(
    all_probe_df: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
) -> bool:
    import matplotlib.pyplot as plt

    probe_df = ensure_label_variant_column(all_probe_df)
    probe_df = probe_df.loc[probe_df["method"].isin(ACTIVE_METHOD_KEYS)].copy()
    if "scrambled" not in set(probe_df["label_variant"]):
        return False

    summary_df = summarize_probe_methods(probe_df, list(ACTIVE_METHOD_KEYS))
    original_df = summary_df.loc[summary_df["label_variant"] == "original"].copy()
    scrambled_df = summary_df.loc[summary_df["label_variant"] == "scrambled"].copy()
    merged = original_df.merge(
        scrambled_df,
        on=["model_family", "method"],
        suffixes=("_original", "_scrambled"),
    )
    if merged.empty:
        return False

    merged["delta_auprc"] = merged["auprc_mean_original"] - merged["auprc_mean_scrambled"]
    merged["signal"] = merged["model_family"] + " | " + merged["method"].map(
        lambda value: METHOD_PUBLICATION_LABELS.get(value, value)
    )
    merged = merged.sort_values("delta_auprc", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    y_positions = np.arange(len(merged))
    ax.hlines(y_positions, merged["auprc_mean_scrambled"], merged["auprc_mean_original"], color="#BBBBBB", linewidth=1.2)
    ax.scatter(merged["auprc_mean_original"], y_positions, color="#4C72B0", s=20, label="Original")
    ax.scatter(merged["auprc_mean_scrambled"], y_positions, color="#C44E52", s=20, label="Scrambled")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(merged["signal"], fontsize=FONT_TICK)
    ax.set_xlabel("AUPRC")
    ax.invert_yaxis()
    _style_axes(ax)
    _legend_below(ax, ncol=2, y_offset=-0.18)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    _safe_savefig(fig, pdf_path, bbox_inches="tight")
    _safe_savefig(fig, png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_main_ig_masking_vs_random(
    ig_validation_sweep_csv: Path,
    ig_vs_random_baseline_csv: Path,
    pdf_path: Path,
    png_path: Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sweep_df = pd.read_csv(ig_validation_sweep_csv)
    random_df = pd.read_csv(ig_vs_random_baseline_csv)

    ig_summary_rows = []
    for k_pct, subset in sweep_df.groupby("k_pct", sort=True):
        mean_value, ci_low, ci_high = bootstrap_mean_ci(subset["delta_p"])
        validated_fraction = float(subset["validated"].astype(float).mean()) if "validated" in subset.columns else float("nan")
        ig_summary_rows.append(
            {
                "k_pct": float(k_pct),
                "mean_delta_p": mean_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "pct_validated": validated_fraction,
                "n_proteins": int(subset["sequence_id"].nunique()),
            }
        )
    ig_summary_df = pd.DataFrame(ig_summary_rows).sort_values("k_pct").reset_index(drop=True)
    best_k_row = ig_summary_df.sort_values(
        ["pct_validated", "mean_delta_p", "k_pct"],
        ascending=[False, False, True],
    ).iloc[0]
    best_k_pct = float(best_k_row["k_pct"])

    random_mean, random_ci_low, random_ci_high = bootstrap_mean_ci(random_df["mean_random_delta_p"])
    ig_best_df = sweep_df.loc[sweep_df["k_pct"].eq(best_k_pct)].copy()
    ig_best_mean, ig_best_ci_low, ig_best_ci_high = bootstrap_mean_ci(ig_best_df["delta_p"])
    from scipy.stats import wilcoxon

    paired_df = random_df.dropna(subset=["ig_delta_p", "mean_random_delta_p"]).copy()
    wilcoxon_result = wilcoxon(
        paired_df["ig_delta_p"].to_numpy(dtype=float),
        paired_df["mean_random_delta_p"].to_numpy(dtype=float),
        alternative="two-sided",
    )
    p_value = float(wilcoxon_result.pvalue)
    significance_marker = _significance_marker(p_value)

    fig, ax = plt.subplots(figsize=ONE_COLUMN_FIGSIZE)
    ax.plot(
        ig_summary_df["k_pct"],
        ig_summary_df["mean_delta_p"],
        marker="o",
        markersize=3.8,
        linewidth=1.4,
        color="#4C72B0",
        label="IG-guided masking",
    )
    ax.fill_between(
        ig_summary_df["k_pct"],
        ig_summary_df["ci_low"],
        ig_summary_df["ci_high"],
        color="#4C72B0",
        alpha=0.18,
    )
    ax.errorbar(
        [best_k_pct],
        [random_mean],
        yerr=[[random_mean - random_ci_low], [random_ci_high - random_mean]],
        fmt="D",
        markersize=4.5,
        capsize=3,
        color="#C44E52",
        label="Random masking (selected k)",
    )
    ax.errorbar(
        [best_k_pct],
        [ig_best_mean],
        yerr=[[ig_best_mean - ig_best_ci_low], [ig_best_ci_high - ig_best_mean]],
        fmt="none",
        capsize=3,
        ecolor="#4C72B0",
        elinewidth=1.4,
    )
    ax.scatter(
        [best_k_pct],
        [ig_best_mean],
        s=44,
        facecolors="white",
        edgecolors="#4C72B0",
        linewidths=1.6,
        zorder=5,
    )
    y_top = max(random_ci_high, ig_best_ci_high)
    y_bottom = min(random_mean, ig_best_mean)
    y_range = max(float(ig_summary_df["ci_high"].max()) - float(ig_summary_df["ci_low"].min()), 1e-6)
    bracket_x = best_k_pct + 0.022
    tick_width = 0.012
    ax.plot(
        [bracket_x, bracket_x + tick_width, bracket_x + tick_width, bracket_x],
        [ig_best_mean, ig_best_mean, random_mean, random_mean],
        color="black",
        linewidth=1.1,
    )
    ax.text(
        bracket_x + tick_width * 0.5,
        y_top + 0.03 * y_range,
        significance_marker,
        ha="center",
        va="bottom",
        fontsize=FONT_AXIS + 1,
    )
    ax.set_xlabel("Top-k percentage", labelpad=2)
    ax.set_ylabel("Mean Δp")
    ax.set_xlim(float(ig_summary_df["k_pct"].min()) - 0.025, float(ig_summary_df["k_pct"].max()) + 0.025)
    ax.set_ylim(bottom=min(float(ig_summary_df["ci_low"].min()), y_bottom) - 0.03 * y_range, top=y_top + 0.10 * y_range)
    _style_axes(ax)
    legend_handles = [
        Line2D([], [], color="#4C72B0", marker="o", markersize=3.8, linewidth=1.4, label="IG-guided masking"),
        Line2D(
            [],
            [],
            color="#C44E52",
            marker="D",
            markersize=4.5,
            linewidth=0,
            linestyle="none",
            markerfacecolor="#C44E52",
            markeredgecolor="#C44E52",
            label="Random masking (selected k)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=FONT_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        ncol=2,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
    _safe_savefig(fig, pdf_path, bbox_inches="tight")
    _safe_savefig(fig, png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "best_k_pct": best_k_pct,
        "n_proteins": int(random_df["sequence_id"].nunique()),
        "ig_summary_df": ig_summary_df,
        "random_mean": random_mean,
        "wilcoxon_p": p_value,
    }


def plot_main_saturation_mutagenesis_summary(
    per_protein_deep_dive_csv: Path,
    transition_summary_csv: Path,
    pdf_path: Path,
    png_path: Path,
) -> bool:
    import matplotlib.pyplot as plt

    if not transition_summary_csv.exists():
        return False
    summary_df = pd.read_csv(transition_summary_csv)
    if summary_df.empty or "class" not in summary_df.columns:
        return False
    plot_df = (
        summary_df.groupby("class", as_index=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "weighted_mean_delta_p": np.average(
                        frame["mean_delta_p_reducing"],
                        weights=frame["n_reducing"].clip(lower=1),
                    )
                }
            )
        )
        .reset_index(drop=True)
        .sort_values("weighted_mean_delta_p", ascending=False)
    )
    if plot_df.empty:
        return False

    fig, ax = plt.subplots(figsize=SHORT_FIGSIZE)
    ax.barh(plot_df["class"], plot_df["weighted_mean_delta_p"], color="#4C72B0", alpha=0.9)
    ax.set_xlabel("Mean Δp among reducing substitutions")
    ax.set_ylabel("")
    ax.invert_yaxis()
    _style_axes(ax)
    fig.tight_layout()
    _safe_savefig(fig, pdf_path, bbox_inches="tight")
    _safe_savefig(fig, png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def render_main_mutagenesis_transition_figures(
    results_dir: Path,
    paper_figures_dir: Path,
) -> dict[str, Path]:
    from .plotting_insilico_mutagenesis import (
        CHARGE_POLARITY_CLASSES,
        HYDROPHOBICITY_AROMATICITY_CLASSES,
        _build_normalized_transition_matrix,
        _build_transition_dataframe,
        _plot_transition_class_heatmap,
        _plot_transition_scatter,
        _summarize_transition_residues,
    )

    annotated_csv = Path(results_dir) / "insilico_mutagenesis" / "saturation_mutagenesis_annotated.csv"
    if not annotated_csv.exists():
        return {}

    annotated_df = pd.read_csv(annotated_csv)
    transition_df = _build_transition_dataframe(annotated_df)
    residue_summary_df = _summarize_transition_residues(transition_df)
    charge_matrix = _build_normalized_transition_matrix(transition_df, CHARGE_POLARITY_CLASSES)
    hydrophobicity_matrix = _build_normalized_transition_matrix(transition_df, HYDROPHOBICITY_AROMATICITY_CLASSES)

    outputs = {
        "residue_scatter_pdf": Path(paper_figures_dir) / "main_transition_residue_scatter.pdf",
        "residue_scatter_png": Path(paper_figures_dir) / "main_transition_residue_scatter.png",
        "charge_heatmap_pdf": Path(paper_figures_dir) / "main_transition_charge_polarity_heatmap.pdf",
        "charge_heatmap_png": Path(paper_figures_dir) / "main_transition_charge_polarity_heatmap.png",
        "hydrophobicity_heatmap_pdf": Path(paper_figures_dir) / "main_transition_hydrophobicity_heatmap.pdf",
        "hydrophobicity_heatmap_png": Path(paper_figures_dir) / "main_transition_hydrophobicity_heatmap.png",
    }

    _plot_transition_scatter(residue_summary_df, outputs["residue_scatter_pdf"])
    _plot_transition_scatter(residue_summary_df, outputs["residue_scatter_png"])
    _plot_transition_class_heatmap(charge_matrix, outputs["charge_heatmap_pdf"])
    _plot_transition_class_heatmap(charge_matrix, outputs["charge_heatmap_png"])
    _plot_transition_class_heatmap(hydrophobicity_matrix, outputs["hydrophobicity_heatmap_pdf"])
    _plot_transition_class_heatmap(hydrophobicity_matrix, outputs["hydrophobicity_heatmap_png"])
    return outputs
