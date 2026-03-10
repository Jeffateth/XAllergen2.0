# XAllergen2.0

Minimal first experiment for allergen interpretability with frozen `ESM-2 + pretrained InterPLM SAE`.

## Goal

Test whether pretrained sparse autoencoder features from InterPLM separate allergenic from non-allergenic proteins before training any custom SAE.

## Input format

Create a CSV with at least these columns:

```csv
protein_id,sequence,label
allergen_1,MKVLWAALLVTFLAGCQAKVE...,1
non_allergen_1,MSLSTEQMLRDYPRSMQ...,0
```

`label` must be `1` for allergen and `0` for non-allergen.

## Environment

The local virtualenv already has `torch`, `transformers`, `scikit-learn`, and `biopython`.

This repo now vendors the full InterPLM source tree under:

[`external/InterPLM`](/Users/jianzhouyao/Library/Mobile%20Documents/com~apple~CloudDocs/Universität/ETH%20/DL%20in%20Biology/XAllergen2.0/external/InterPLM)

The extraction script prefers that local source tree automatically, because the published `interplm` wheel currently has a packaging issue around `interplm.train`.

## Step 1: extract pretrained SAE features

Start with the smallest supported setup:

- PLM: `esm2-8m`
- Layer: `4`

```bash
./.venv/bin/python scripts/extract_interplm_features.py \
  --input-csv data/allergen_vs_nonallergen.csv \
  --output-npz outputs/interplm_esm2_8m_l4_features.npz \
  --plm-model esm2-8m \
  --plm-layer 4
```

This writes protein-level summaries of each SAE feature:

- `feature_max`: maximum residue activation in the protein
- `feature_mean`: mean residue activation in the protein
- `feature_frac_active`: fraction of residues above the activation threshold

## Step 2: rank allergen-associated features

```bash
./.venv/bin/python scripts/evaluate_feature_associations.py \
  --features-npz outputs/interplm_esm2_8m_l4_features.npz \
  --summary max \
  --output-csv outputs/interplm_esm2_8m_l4_feature_ranking.csv
```

The output CSV ranks features by:

- direction-corrected AUROC
- mean activation difference
- standardized effect size

It also prints a sparse logistic baseline on top-variance SAE features to show whether the pretrained feature space contains usable allergen signal.

## Recommended first run

Keep the first pass small:

1. Use a balanced pilot set like `200 allergens + 200 non-allergens`.
2. Run `esm2-8m`, layer `4`.
3. Inspect the top `20` allergen-enriched and non-allergen-enriched features.
4. Only then scale up to more proteins, more layers, or `esm2-650m`.

## Notes

- InterPLM provides pretrained SAEs for `esm2-8m` layers `1-6` and `esm2-650m` layers `1, 9, 18, 24, 30, 33`.
- This repo currently uses pretrained SAEs only. Training a custom allergen-domain SAE should be the second experiment, not the first.

## Homology-aware split

To stay consistent with AlgPred 2.0, create `subtrain/subval` from the original training set by clustering filtered sequences at `40%` identity with `CD-HIT` and splitting whole clusters:

```bash
./.venv/bin/python scripts/make_homology_split.py \
  --train-csv data/algpred2_train_seq.csv \
  --test-csv data/algpred2_test_seq.csv \
  --out-dir outputs/homology_split \
  --max-seq-len 1022 \
  --identity 0.4 \
  --subval-fraction 0.2
```

Notes:
- This keeps the original test set untouched for final evaluation.
- The optional `--test-csv` argument runs a `cd-hit-2d` leakage audit of filtered test proteins against filtered training proteins.
- `cd-hit` and `cd-hit-2d` must be installed separately and available on `PATH`.
- If you only want the filtered training FASTA/CSV prepared first, add `--skip-cdhit`.
