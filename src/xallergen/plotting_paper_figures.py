from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def build_output_paths_for_supported_mtl(
    family_key: str,
    display_label: str,
    models_dir: Path,
    results_dir: Path,
    baseline_checkpoint_path: Path,
    baseline_summary_path: Path,
) -> MTLOutputPaths:
    if family_key == "mtl_frozen":
        prefix = "mtl"
        checkpoint_name = "mtl_frozen_esm2_epitope.pt"
        metrics_name = "mtl_baseline_metrics.json"
        baseline_rows_name = "baseline_probing_rows.csv"
    elif family_key == "mtl_top1_unfrozen":
        prefix = "mtl_top1_unfrozen"
        checkpoint_name = "mtl_top1_unfrozen_esm2_epitope.pt"
        metrics_name = "mtl_top1_unfrozen_baseline_metrics.json"
        baseline_rows_name = "baseline_probing_rows_top1_unfrozen.csv"
    else:
        raise ValueError(f"Unsupported MTL family_key for output path construction: {family_key}")

    if family_key == "mtl_frozen":
        probe_rows_name = "mtl_probing_rows.csv"
        probe_summary_name = "mtl_probing_summary.csv"
        combined_rows_name = "mtl_vs_baseline_probing_rows.csv"
        compare_summary_name = "mtl_vs_baseline_summary.csv"
        figure_prefix = "mtl_vs_baseline"
    else:
        suffix = family_key.replace("mtl_", "", 1)
        probe_rows_name = f"mtl_{suffix}_probing_rows.csv"
        probe_summary_name = f"mtl_{suffix}_probing_summary.csv"
        combined_rows_name = f"mtl_{suffix}_vs_baseline_probing_rows.csv"
        compare_summary_name = f"mtl_{suffix}_vs_baseline_summary.csv"
        figure_prefix = f"mtl_{suffix}_vs_baseline"

    return MTLOutputPaths(
        baseline_checkpoint_path=baseline_checkpoint_path,
        checkpoint_path=models_dir / checkpoint_name,
        metrics_path=results_dir / "classification" / metrics_name,
        probe_rows_path=results_dir / "probing" / "rows" / probe_rows_name,
        baseline_probe_rows_path=results_dir / "probing" / "rows" / baseline_rows_name,
        combined_probe_rows_path=results_dir / "probing" / "rows" / combined_rows_name,
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
    return pd.DataFrame(rows), prevalence


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

    metric_keys = ["auroc", "auprc", "precision_at_k"]
    metric_labels = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "precision_at_k": "Precision@k",
    }
    signal_order = list(summary_df["Signal"])
    y_positions = np.arange(len(signal_order), dtype=float)
    offsets = {"auroc": -0.18, "auprc": 0.0, "precision_at_k": 0.18}

    fig, ax = plt.subplots(figsize=ONE_COLUMN_FIGSIZE)
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
    ax.set_xlabel("Score")
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="lower right")
    fig.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_main_protein_performance_table(
    metrics_df: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
) -> pd.DataFrame:
    ordered_models = ["Frozen ESM-2", "MTL ESM-2", "DeepPlantAllergy"]
    if metrics_df.empty:
        table_df = pd.DataFrame(columns=["Model", "AUROC", "AUPRC", "MCC", "Accuracy", "n_test_sequences"])
        _write_table_outputs(table_df, csv_path, tex_path)
        return table_df

    metrics_df = metrics_df.copy()
    metrics_df["_order"] = metrics_df["Model"].map({label: idx for idx, label in enumerate(ordered_models)})
    metrics_df = metrics_df.sort_values("_order").drop(columns="_order")
    for column in ["AUROC", "AUPRC", "MCC", "Accuracy"]:
        if column in metrics_df.columns:
            metrics_df[column] = metrics_df[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "NA")
    table_df = metrics_df[["Model", "AUROC", "AUPRC", "MCC", "Accuracy", "n_test_sequences"]].copy()
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
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
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
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="lower right")
    fig.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_main_ig_masking_vs_random(
    ig_validation_sweep_csv: Path,
    ig_vs_random_baseline_csv: Path,
    pdf_path: Path,
    png_path: Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

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
        fmt="o",
        markersize=4.5,
        capsize=3,
        color="#4C72B0",
    )
    ax.set_xlabel("Top-k percentage")
    ax.set_ylabel("Mean Δp")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="upper left")
    fig.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "best_k_pct": best_k_pct,
        "n_proteins": int(random_df["sequence_id"].nunique()),
        "ig_summary_df": ig_summary_df,
        "random_mean": random_mean,
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
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True
