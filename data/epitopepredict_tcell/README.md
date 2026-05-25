epitopepredict-based MHC-II T-cell epitope outputs for DeepAlgPro live here.

This workflow uses:
- allergen-only DeepAlgPro sequences
- `epitopepredict`
- the built-in `tepitope` predictor
- 15-mer peptides by default

Outputs:
- `deepalgpro_epitopepredict_tcell_manifest.csv`: exported protein manifest
- `deepalgpro_*_tepitope_raw_predictions.csv.gz`: raw scored peptide windows
- `deepalgpro_*_tepitope_binders.csv.gz`: binder-filtered rows using the configured cutoff
- `deepalgpro_*_tepitope_promiscuous_binders.csv.gz`: binders shared across at least `n` alleles
- `deepalgpro_epitopepredict_tcell_summary.json`: run configuration and output counts
