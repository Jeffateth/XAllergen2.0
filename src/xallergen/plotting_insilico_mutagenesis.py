from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)


ICML_ONE_COLUMN_WIDTH = 4.2
PAPER_SHORT_HEIGHT = 3.0
PAPER_MEDIUM_HEIGHT = 3.6
PAPER_TALL_HEIGHT = 4.4
PAPER_DPI = 300
PAPER_TITLE_FONTSIZE = 13
PAPER_LABEL_FONTSIZE = 11.5
PAPER_TICK_FONTSIZE = 10.5
PAPER_ANNOT_FONTSIZE = 10

MIN_DELTA_P = 0.05
MIN_TRANSITION_SUPPORT = 20
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_RANDOM_STATE = 42

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

PROPERTY_LABELS = {
    "charge_negative": "Acidic",
    "charge_positive": "Basic",
    "polar_uncharged": "Polar uncharged",
    "hydrophobic": "Nonpolar",
    "special": "Small / conformationally special",
}

BAR_CLASS_COLORS = {
    "Acidic": "#d62728",
    "Basic": "#1f77b4",
    "Polar uncharged": "#2ca02c",
    "Nonpolar": "#ffbf00",
    "Small / conformationally special": "#9467bd",
}

CHARGE_POLARITY_CLASSES = {
    "D": "Acidic",
    "E": "Acidic",
    "K": "Basic",
    "R": "Basic",
    "H": "Basic",
    "S": "Polar uncharged",
    "T": "Polar uncharged",
    "N": "Polar uncharged",
    "Q": "Polar uncharged",
    "C": "Polar uncharged",
    "Y": "Polar uncharged",
    "A": "Nonpolar",
    "V": "Nonpolar",
    "L": "Nonpolar",
    "I": "Nonpolar",
    "M": "Nonpolar",
    "F": "Nonpolar",
    "W": "Nonpolar",
    "G": "Small / conformationally special",
    "P": "Small / conformationally special",
}

HYDROPHOBICITY_AROMATICITY_CLASSES = {
    "A": "Strongly hydrophobic aliphatic",
    "V": "Strongly hydrophobic aliphatic",
    "L": "Strongly hydrophobic aliphatic",
    "I": "Strongly hydrophobic aliphatic",
    "M": "Strongly hydrophobic aliphatic",
    "F": "Aromatic",
    "W": "Aromatic",
    "Y": "Aromatic",
    "S": "Polar / H-bonding",
    "T": "Polar / H-bonding",
    "N": "Polar / H-bonding",
    "Q": "Polar / H-bonding",
    "C": "Polar / H-bonding",
    "D": "Charged",
    "E": "Charged",
    "K": "Charged",
    "R": "Charged",
    "H": "Charged",
    "G": "Small / conformationally special",
    "P": "Small / conformationally special",
}

CLASS_LABELS_COMPACT = {
    "Acidic": "Acidic",
    "Basic": "Basic",
    "Polar uncharged": "Polar\nuncharged",
    "Nonpolar": "Nonpolar",
    "Strongly hydrophobic aliphatic": "Hydrophobic\naliphatic",
    "Aromatic": "Aromatic",
    "Polar / H-bonding": "Polar /\nH-bonding",
    "Charged": "Charged",
    "Small / conformationally special": "Small /\nconformational",
}


def bootstrap_ci_mean(
    values,
    n_bootstrap: int = BOOTSTRAP_N_RESAMPLES,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean_value = float(values.mean())
    if values.size == 1:
        return mean_value, mean_value, mean_value
    rng = np.random.default_rng(random_state)
    boot = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        boot[idx] = sample.mean()
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return mean_value, float(ci_low), float(ci_high)


def set_paper_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": PAPER_LABEL_FONTSIZE,
            "axes.titlesize": PAPER_TITLE_FONTSIZE,
            "axes.labelsize": PAPER_LABEL_FONTSIZE,
            "xtick.labelsize": PAPER_TICK_FONTSIZE,
            "ytick.labelsize": PAPER_TICK_FONTSIZE,
            "figure.dpi": PAPER_DPI,
            "savefig.dpi": PAPER_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _select_best_k_pct_from_sweep(frame: pd.DataFrame) -> float:
    summary_rows = []
    for k_pct, group_df in frame.groupby("k_pct", sort=True):
        validated_values = group_df["validated"].astype(float).to_numpy()
        summary_rows.append(
            {
                "k_pct": float(k_pct),
                "pct_validated": float(validated_values.mean()),
                "mean_delta_p": float(group_df["delta_p"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    best_k_row = summary_df.sort_values(
        ["pct_validated", "mean_delta_p", "k_pct"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(best_k_row["k_pct"])


def _build_transition_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "original_aa",
        "mutant_aa",
        "delta_p",
        "reduces_allergenicity",
        "original_property",
        "mutant_property",
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"annotated_mutagenesis_df is missing required columns: {sorted(missing_columns)}"
        )

    df = frame.copy()
    df = df.dropna(
        subset=[
            "original_aa",
            "mutant_aa",
            "delta_p",
            "original_property",
            "mutant_property",
        ]
    ).copy()

    df["original_aa"] = df["original_aa"].astype(str).str.upper()
    df["mutant_aa"] = df["mutant_aa"].astype(str).str.upper()
    df = df.loc[df["original_aa"].isin(AA_ORDER)].copy()
    df["class"] = df["original_property"].map(PROPERTY_LABELS)
    return df.loc[df["class"].notna()].copy()


def _summarize_transition_residues(frame: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for original_aa, group_df in frame.groupby("original_aa", sort=True):
        reducing_mask = group_df["reduces_allergenicity"].fillna(False).astype(bool)
        summary_rows.append(
            {
                "original_aa": str(original_aa),
                "class": str(group_df["class"].mode().iloc[0]),
                "n_total": int(len(group_df)),
                "n_reducing": int(reducing_mask.sum()),
                "frac_reducing": float(reducing_mask.mean()),
                "mean_delta_p_all": float(group_df["delta_p"].mean()),
                "mean_delta_p_reducing": (
                    float(group_df.loc[reducing_mask, "delta_p"].mean())
                    if reducing_mask.any()
                    else 0.0
                ),
            }
        )
    return (
        pd.DataFrame(summary_rows)
        .sort_values(
            ["frac_reducing", "mean_delta_p_all", "original_aa"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def _select_labels_for_scatter(frame: pd.DataFrame) -> set[str]:
    return set(frame["original_aa"].astype(str))


def _summarize_top_supported_transitions(frame: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    reducing_df = frame.loc[frame["reduces_allergenicity"].fillna(False).astype(bool)].copy()
    reducing_df = reducing_df.dropna(subset=["original_aa", "mutant_aa", "delta_p"])
    reducing_df = reducing_df.loc[reducing_df["original_aa"] != reducing_df["mutant_aa"]].copy()
    if reducing_df.empty:
        return pd.DataFrame(
            columns=[
                "original_aa",
                "mutant_aa",
                "n",
                "mean_delta_p",
                "median_delta_p",
                "std_delta_p",
                "sem_delta_p",
                "ci95_low",
                "ci95_high",
                "ci95_half_width",
                "transition_label",
            ]
        )

    summary_df = (
        reducing_df.groupby(["original_aa", "mutant_aa"], as_index=False)["delta_p"]
        .agg(n="count", mean_delta_p="mean", median_delta_p="median", std_delta_p="std")
    )
    summary_df["std_delta_p"] = summary_df["std_delta_p"].fillna(0.0)
    summary_df = summary_df.loc[summary_df["n"] >= int(MIN_TRANSITION_SUPPORT)].copy()
    if summary_df.empty:
        return summary_df

    summary_df["sem_delta_p"] = summary_df["std_delta_p"] / np.sqrt(summary_df["n"].clip(lower=1))
    ci_rows = []
    for row in summary_df.itertuples(index=False):
        group_values = reducing_df.loc[
            (reducing_df["original_aa"] == row.original_aa)
            & (reducing_df["mutant_aa"] == row.mutant_aa),
            "delta_p",
        ].to_numpy(dtype=float)
        _, ci_low, ci_high = bootstrap_ci_mean(group_values)
        ci_rows.append(
            {
                "original_aa": row.original_aa,
                "mutant_aa": row.mutant_aa,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    summary_df = summary_df.merge(pd.DataFrame(ci_rows), on=["original_aa", "mutant_aa"], how="left")
    summary_df["ci95_half_width"] = summary_df["ci95_high"] - summary_df["mean_delta_p"]
    summary_df["transition_label"] = summary_df["original_aa"] + "->" + summary_df["mutant_aa"]
    summary_df = summary_df.sort_values(
        ["mean_delta_p", "n", "median_delta_p", "transition_label"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return summary_df.head(top_n).copy()


def _build_normalized_transition_matrix(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    mapped = frame.copy()
    mapped["original_class"] = mapped["original_aa"].map(mapping)
    mapped["mutant_class"] = mapped["mutant_aa"].map(mapping)
    mapped = mapped.dropna(subset=["original_class", "mutant_class", "delta_p"]).copy()
    matrix = (
        mapped.loc[mapped["reduces_allergenicity"].fillna(False).astype(bool)]
        .groupby(["original_class", "mutant_class"], as_index=False)["delta_p"]
        .mean()
        .pivot(index="original_class", columns="mutant_class", values="delta_p")
    )
    row_order = sorted(matrix.index.tolist())
    col_order = sorted(matrix.columns.tolist())
    return matrix.reindex(index=row_order, columns=col_order).fillna(0.0)


def _plot_stage1_diagnostics(ig_validation_sweep_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_paper_plot_style()
    summary_rows = []
    for k_pct, group_df in ig_validation_sweep_df.groupby("k_pct", sort=True):
        validated_values = group_df["validated"].astype(float).to_numpy()
        pct_validated, ci_low, ci_high = bootstrap_ci_mean(validated_values)
        summary_rows.append(
            {
                "k_pct": float(k_pct),
                "k_absolute_mean": float(group_df["k_absolute"].mean()),
                "n_validated": int(group_df["validated"].sum()),
                "pct_validated": float(pct_validated),
                "pct_validated_ci_low": float(ci_low),
                "pct_validated_ci_high": float(ci_high),
                "n_proteins": int(len(group_df)),
                "mean_delta_p": float(group_df["delta_p"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("k_pct").reset_index(drop=True)

    plot_df = ig_validation_sweep_df.copy()
    plot_df["k_label"] = plot_df["k_pct"].map(lambda value: f"{int(round(value * 100))}%")
    order = [f"{int(round(value * 100))}%" for value in summary_df["k_pct"]]

    delta_plot_path = output_dir / "ig_validation_sweep_delta_p.png"
    fig, ax = plt.subplots(figsize=(ICML_ONE_COLUMN_WIDTH, PAPER_MEDIUM_HEIGHT))
    sns.violinplot(data=plot_df, x="k_label", y="delta_p", order=order, inner=None, cut=0, ax=ax, color="#4C72B0")
    sns.stripplot(data=plot_df, x="k_label", y="delta_p", order=order, ax=ax, color="black", alpha=0.35, size=3.5, jitter=0.15)
    ax.axhline(MIN_DELTA_P, color="#C44E52", linestyle="--", linewidth=1.5)
    ax.set_xlabel("k (% of sequence length)")
    ax.set_ylabel("Delta p")
    fig.tight_layout()
    fig.savefig(delta_plot_path, bbox_inches="tight")
    plt.close(fig)

    fraction_plot_path = output_dir / "ig_validation_sweep_validated_fraction.png"
    x_positions = np.arange(len(summary_df))
    heights = summary_df["pct_validated"].to_numpy()
    yerr = np.vstack(
        [
            heights - summary_df["pct_validated_ci_low"].to_numpy(),
            summary_df["pct_validated_ci_high"].to_numpy() - heights,
        ]
    )
    fig, ax = plt.subplots(figsize=(5.2, PAPER_SHORT_HEIGHT))
    ax.bar(x_positions, heights, color="#55A868", alpha=0.9)
    ax.errorbar(x_positions, heights, yerr=yerr, fmt="none", ecolor="black", capsize=4, linewidth=1.2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(order)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("k (% of sequence length)")
    ax.set_ylabel("Fraction validated")
    fig.tight_layout()
    fig.savefig(fraction_plot_path, bbox_inches="tight")
    plt.close(fig)
    return summary_df


def _plot_ig_vs_random_diagnostics(ig_vs_random_baseline_df: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import wilcoxon

    paired_df = ig_vs_random_baseline_df.dropna(subset=["ig_delta_p", "mean_random_delta_p"]).copy()
    if paired_df.empty:
        raise ValueError("No paired IG-vs-random baseline rows are available for plotting.")

    set_paper_plot_style()
    result = wilcoxon(
        paired_df["ig_delta_p"].to_numpy(dtype=float),
        paired_df["mean_random_delta_p"].to_numpy(dtype=float),
        alternative="two-sided",
    )
    p_value = float(result.pvalue)
    if p_value < 0.001:
        significance_star = "***"
    elif p_value < 0.01:
        significance_star = "**"
    elif p_value < 0.05:
        significance_star = "*"
    else:
        significance_star = "ns"

    plot_df = pd.concat(
        [
            paired_df[["sequence_id", "ig_delta_p"]].rename(columns={"ig_delta_p": "delta_p"}).assign(strategy="IG top-k"),
            paired_df[["sequence_id", "mean_random_delta_p"]].rename(columns={"mean_random_delta_p": "delta_p"}).assign(strategy="Random k"),
        ],
        ignore_index=True,
    )

    dist_plot_path = output_dir / "ig_vs_random_baseline_distribution.png"
    fig, ax = plt.subplots(figsize=(ICML_ONE_COLUMN_WIDTH, PAPER_MEDIUM_HEIGHT))
    for row in paired_df.itertuples(index=False):
        ax.plot([0, 1], [row.ig_delta_p, row.mean_random_delta_p], color="lightgray", alpha=0.25, zorder=0)
    sns.violinplot(data=plot_df, x="strategy", y="delta_p", inner=None, cut=0, color="#B8C7E0", ax=ax)
    sns.stripplot(data=plot_df, x="strategy", y="delta_p", color="black", alpha=0.25, size=3.5, ax=ax)
    for x_pos, strategy in enumerate(["IG top-k", "Random k"]):
        mean_value, ci_low, ci_high = bootstrap_ci_mean(plot_df.loc[plot_df["strategy"] == strategy, "delta_p"].to_numpy())
        ax.errorbar(x_pos, mean_value, yerr=[[mean_value - ci_low], [ci_high - mean_value]], fmt="none", ecolor="black", elinewidth=1.5, capsize=10, capthick=1.5, zorder=6)
        ax.scatter(x_pos, mean_value, color="black", s=28, zorder=7)
    y_values = plot_df["delta_p"].to_numpy(dtype=float)
    y_max = float(np.max(y_values))
    y_min = float(np.min(y_values))
    y_range = max(y_max - y_min, 1e-6)
    bracket_y = y_max * 1.08 if y_max > 0 else y_max + 0.08 * y_range
    tick_drop = 0.03 * y_range
    text_offset = 0.02 * y_range
    ax.plot([0, 0, 1, 1], [bracket_y - tick_drop, bracket_y, bracket_y, bracket_y - tick_drop], color="black", linewidth=1.5)
    ax.text(0.5, bracket_y + text_offset, significance_star, ha="center", va="bottom", fontsize=PAPER_TITLE_FONTSIZE)
    ax.set_ylim(y_min - 0.05 * y_range, bracket_y + 0.12 * y_range)
    ax.set_xlabel("Masking strategy")
    ax.set_ylabel("Delta p")
    fig.tight_layout()
    fig.savefig(dist_plot_path, bbox_inches="tight")
    plt.close(fig)

    advantage_plot_path = output_dir / "ig_vs_random_baseline_advantage.png"
    diff_df = paired_df.assign(delta_difference=paired_df["ig_delta_p"] - paired_df["mean_random_delta_p"]).sort_values(
        "delta_difference",
        ascending=False,
    ).reset_index(drop=True)
    diff_colors = np.where(diff_df["delta_difference"] > 0, "#4C72B0", "#C44E52")
    advantage_height = max(4.0, 0.18 * len(diff_df) + 1.0)
    fig, ax = plt.subplots(figsize=(ICML_ONE_COLUMN_WIDTH, advantage_height))
    ax.barh(diff_df["sequence_id"], diff_df["delta_difference"], color=diff_colors)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.invert_yaxis()
    ax.set_xlabel("Delta p(IG) - Delta p(random)")
    ax.set_ylabel("Sequence")
    ax.tick_params(axis="y", labelsize=max(PAPER_TICK_FONTSIZE - 1, 8.5))
    fig.tight_layout()
    fig.savefig(advantage_plot_path, bbox_inches="tight")
    plt.close(fig)

    return {"wilcoxon_p": p_value, "n_sequences": int(len(paired_df))}


def _plot_delta_p_distribution(ig_validation_sweep_df: pd.DataFrame, best_k_pct: float, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    set_paper_plot_style()
    best_k_all_df = ig_validation_sweep_df.loc[ig_validation_sweep_df["k_pct"] == best_k_pct].copy()
    median_delta_p = float(best_k_all_df["delta_p"].median())
    output_path = output_dir / "delta_p_distribution.png"
    fig, ax = plt.subplots(figsize=(ICML_ONE_COLUMN_WIDTH, PAPER_SHORT_HEIGHT))
    ax.hist(best_k_all_df["delta_p"], bins=20, color="#4C72B0", alpha=0.85)
    ax.axvline(median_delta_p, color="black", linestyle="-", linewidth=1.5)
    ax.axvline(MIN_DELTA_P, color="#C44E52", linestyle="--", linewidth=1.5)
    ax.text(median_delta_p, ax.get_ylim()[1] * 0.95, f"Median = {median_delta_p:.3f}", color="black", ha="left", va="top", fontsize=PAPER_ANNOT_FONTSIZE)
    ax.set_xlabel("Delta p")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_transition_scatter(summary_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    set_paper_plot_style()
    df = summary_df.copy()
    colors = df["class"].map(BAR_CLASS_COLORS)
    if df["n_total"].max() > df["n_total"].min():
        sizes = 45 + 220 * ((df["n_total"] - df["n_total"].min()) / (df["n_total"].max() - df["n_total"].min()))
    else:
        sizes = np.full(len(df), 120.0)
    df["plot_x"] = df["frac_reducing"]
    df["plot_y"] = df["mean_delta_p_all"]

    fig, ax = plt.subplots(figsize=(ICML_ONE_COLUMN_WIDTH, PAPER_SHORT_HEIGHT))
    ax.scatter(df["plot_x"], df["plot_y"], s=sizes, c=colors, alpha=0.78, edgecolors="black", linewidth=0.6, zorder=3)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.45, zorder=1)
    ax.axvline(0.5, color="gray", linewidth=1, linestyle="--", alpha=0.45, zorder=1)
    texts = []
    for _, row in df.iterrows():
        aa = str(row["original_aa"])
        if aa not in _select_labels_for_scatter(df):
            continue
        if aa in {"M", "R"}:
            label_fontsize = max(PAPER_ANNOT_FONTSIZE - 3, 5)
        elif aa in {"G", "C", "K"}:
            label_fontsize = PAPER_ANNOT_FONTSIZE
        elif aa == "A":
            label_fontsize = max(PAPER_ANNOT_FONTSIZE - 7, 4)
        elif aa == "Y":
            label_fontsize = max(PAPER_ANNOT_FONTSIZE - 5, 6)
        else:
            label_fontsize = max(PAPER_ANNOT_FONTSIZE - 5, 6)
        texts.append(
            ax.text(
                row["plot_x"],
                row["plot_y"],
                aa,
                fontsize=label_fontsize,
                ha="center",
                va="center",
                zorder=4,
            )
        )
    if adjust_text is not None and texts:
        adjust_text(texts, ax=ax, expand_points=(1.2, 1.4), expand_text=(1.1, 1.2))
    ax.set_xlabel("Fraction of reducing substitutions")
    ax.set_ylabel("Mean Delta p")
    x_min = max(0.0, float(df["plot_x"].min()) - 0.045)
    x_max = min(1.0, float(df["plot_x"].max()) + 0.045)
    y_span = float(df["plot_y"].max() - df["plot_y"].min())
    y_pad = max(0.002, 0.20 * y_span)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(float(df["plot_y"].min()) - y_pad, float(df["plot_y"].max()) + y_pad)
    legend_handles = [Patch(color=color, label=label) for label, color in BAR_CLASS_COLORS.items() if label in set(df["class"])]
    ax.legend(
        handles=legend_handles,
        title="Residue class",
        loc="lower right",
        bbox_to_anchor=(0.985, 0.03),
        ncol=1,
        fontsize=max(PAPER_ANNOT_FONTSIZE - 3, 5.5),
        title_fontsize=max(PAPER_ANNOT_FONTSIZE - 3, 5.5),
        frameon=True,
        facecolor="white",
        framealpha=0.82,
        edgecolor="#d0d0d0",
        borderpad=0.2,
        labelspacing=0.18,
        handlelength=0.75,
        handletextpad=0.28,
        columnspacing=0.6,
        borderaxespad=0.2,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_top_supported_transitions(summary_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    set_paper_plot_style()
    figure_height = max(PAPER_MEDIUM_HEIGHT, 0.42 * max(len(summary_df), 1) + 0.8)
    fig, ax = plt.subplots(figsize=(ICML_ONE_COLUMN_WIDTH, figure_height))
    if summary_df.empty:
        ax.text(0.5, 0.5, "No supported reducing transitions passed the minimum support threshold.", ha="center", va="center", fontsize=PAPER_LABEL_FONTSIZE)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return
    plot_df = summary_df.sort_values(["mean_delta_p", "n"], ascending=[True, True]).reset_index(drop=True)
    y_positions = np.arange(len(plot_df))
    xerr_low = (plot_df["mean_delta_p"] - plot_df["ci95_low"]).clip(lower=0)
    xerr_high = (plot_df["ci95_high"] - plot_df["mean_delta_p"]).clip(lower=0)
    xerr = np.vstack([xerr_low.to_numpy(dtype=float), xerr_high.to_numpy(dtype=float)])
    ax.barh(
        y_positions,
        plot_df["mean_delta_p"],
        xerr=xerr,
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.8,
        error_kw={"ecolor": "black", "elinewidth": 1.3, "capsize": 3, "capthick": 1.3},
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["transition_label"])
    ax.set_xlabel("Mean Delta p (bootstrap 95% CI)")
    ax.set_ylabel("Mutation transition")
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    ci_right = plot_df["ci95_high"].fillna(plot_df["mean_delta_p"]).to_numpy(dtype=float)
    max_right = float(np.nanmax(ci_right))
    text_offset = max(max_right * 0.035, 0.0015)
    label_padding = max(max_right * 0.22, 0.012)
    ax.set_xlim(left=0, right=max_right + label_padding)
    for y_pos, x_value, n_value in zip(y_positions, ci_right, plot_df["n"]):
        ax.text(float(x_value) + text_offset, y_pos, f"n={int(n_value)}", va="center", ha="left", fontsize=max(PAPER_ANNOT_FONTSIZE - 1, 7), clip_on=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_supplementary_transition_heatmap(transition_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_paper_plot_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    heatmap_df = (
        transition_df.loc[transition_df["reduces_allergenicity"].fillna(False).astype(bool)]
        .groupby(["original_aa", "mutant_aa"], as_index=False)["delta_p"]
        .mean()
        .pivot(index="original_aa", columns="mutant_aa", values="delta_p")
        .reindex(index=AA_ORDER, columns=AA_ORDER)
    )
    row_order = heatmap_df.mean(axis=1).sort_values(ascending=False).index.tolist()
    col_order = heatmap_df.mean(axis=0).sort_values(ascending=False).index.tolist()
    heatmap_df = heatmap_df.loc[row_order, col_order].fillna(0.0)
    sns.heatmap(heatmap_df, cmap="YlOrRd", annot=False, square=True, linewidths=0.25, linecolor="white", ax=ax, cbar_kws={"label": "Mean Delta p"})
    ax.set_xlabel("Mutant amino acid")
    ax.set_ylabel("Original amino acid")
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _compact_labels(labels) -> list[str]:
    return [CLASS_LABELS_COMPACT.get(label, label) for label in labels]


def _plot_transition_class_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig_width = 3.25 * 1.35
    fig_height = max(3.25 * 1.10, 0.55 * matrix.shape[0] + 1.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    finite_values = matrix.to_numpy().ravel()
    finite_values = finite_values[pd.notna(finite_values)]
    vmin = float(finite_values.min()) if finite_values.size else None
    vmax = float(finite_values.max()) if finite_values.size else None
    heatmap = sns.heatmap(
        matrix,
        cmap="YlOrRd",
        annot=False,
        square=True,
        linewidths=0.4,
        linecolor="white",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": "Mean Delta p", "shrink": 0.82, "pad": 0.035},
    )
    ax.set_xticklabels(_compact_labels(matrix.columns), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticklabels(_compact_labels(matrix.index), rotation=0, fontsize=8)
    ax.set_xlabel("Mutant class", fontsize=10, labelpad=8)
    ax.set_ylabel("Original class", fontsize=10, labelpad=8)
    cbar = heatmap.collections[0].colorbar
    cbar.set_label("Mean Delta p", fontsize=9)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def render_insilico_mutagenesis_diagnostics(results_dir: Path) -> dict[str, object]:
    ism_results_dir = Path(results_dir) / "insilico_mutagenesis"
    ig_validation_sweep_csv = ism_results_dir / "ig_validation_sweep.csv"
    ig_vs_random_csv = ism_results_dir / "ig_vs_random_baseline.csv"
    annotated_csv = ism_results_dir / "saturation_mutagenesis_annotated.csv"
    if not ig_validation_sweep_csv.exists():
        raise FileNotFoundError(f"Missing mutagenesis sweep CSV: {ig_validation_sweep_csv}")
    if not ig_vs_random_csv.exists():
        raise FileNotFoundError(f"Missing IG-vs-random CSV: {ig_vs_random_csv}")
    if not annotated_csv.exists():
        raise FileNotFoundError(f"Missing annotated mutagenesis CSV: {annotated_csv}")

    ig_validation_sweep_df = pd.read_csv(ig_validation_sweep_csv)
    ig_vs_random_baseline_df = pd.read_csv(ig_vs_random_csv)
    annotated_mutagenesis_df = pd.read_csv(annotated_csv)

    summary_df = _plot_stage1_diagnostics(ig_validation_sweep_df, ism_results_dir)
    best_k_pct = _select_best_k_pct_from_sweep(ig_validation_sweep_df)
    ig_random_info = _plot_ig_vs_random_diagnostics(ig_vs_random_baseline_df, ism_results_dir)
    _plot_delta_p_distribution(ig_validation_sweep_df, best_k_pct, ism_results_dir)

    transition_df = _build_transition_dataframe(annotated_mutagenesis_df)
    aa_transition_summary_df = _summarize_transition_residues(transition_df)
    aa_transition_summary_csv = ism_results_dir / "transition_panel1_residue_summary.csv"
    aa_transition_summary_df.to_csv(aa_transition_summary_csv, index=False)
    _plot_transition_scatter(aa_transition_summary_df, ism_results_dir / "transition_panel1_scatter.png")

    top_supported_transitions_df = _summarize_top_supported_transitions(transition_df)
    top_supported_transitions_csv = ism_results_dir / "top10_supported_aa_transitions.csv"
    top_supported_transitions_df.to_csv(top_supported_transitions_csv, index=False)
    _plot_top_supported_transitions(top_supported_transitions_df, ism_results_dir / "top10_supported_aa_transitions.png")
    _plot_supplementary_transition_heatmap(transition_df, ism_results_dir / "supplementary_transition_panel2_aa_heatmap.png")

    charge_matrix = _build_normalized_transition_matrix(transition_df, CHARGE_POLARITY_CLASSES)
    hydrophobicity_matrix = _build_normalized_transition_matrix(transition_df, HYDROPHOBICITY_AROMATICITY_CLASSES)
    _plot_transition_class_heatmap(charge_matrix, ism_results_dir / "transition_panel3_charge_polarity_heatmap.png")
    _plot_transition_class_heatmap(hydrophobicity_matrix, ism_results_dir / "transition_panel3_hydrophobicity_heatmap.png")

    return {
        "best_k_pct": best_k_pct,
        "stage1_summary": summary_df,
        "ig_random_info": ig_random_info,
        "transition_summary_csv": aa_transition_summary_csv,
        "top_supported_transitions_csv": top_supported_transitions_csv,
    }
