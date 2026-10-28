# Paper-aligned FL reconstruction (under review)

This directory is a **reconstruction**, assembled from the updated `config.py` and `model.py` provided by the authors and the older public repository. It is **not established to be the exact source tree** used to produce the archived paper results. Do not cite it as an independently verified reproduction until its outputs have been compared with the deposited reference JSONs.

## Scope and boundaries

- Synthetic cross-silo experiment only: five logical nodes; 10 common predictors; node-specific bounded Beta distributions, label biases and label corruption.
- Single synchronous aggregation level with FedAvg and exogenous source-weighted FedAvg.
- Separate stratified training, validation, and test partitions, approximately 60/15/25 per node.
- Three threshold-selection policies fitted **only on validation predictions**. Test predictions are never supplied to the threshold selector.
- Logistic local-only, federated and centralized full-data reference; ten configured synthetic seed offsets.
- Optional Keras/TensorFlow reference (128/64/32, Swish, dropout, AdamW, no BN in the active architecture). This is not a ten-seed DNN study.
- Sensitivity to four fixed source-weight profiles; communication is a **scalar-count proxy**, not empirical network traffic.
- No real institutional data, verified event horizon, georeferenced disaster reconstruction, hierarchical deployment, production secure aggregation, privacy guarantee, or poisoning defence.

## Files

- `config.py`: copy of the supplied updated paper-era settings.
- `model.py`: copy of the supplied updated model/training implementation.
- `experiment.py`: reconstructed generator, three-way stratified splitting, single-level federation, validation thresholding, test evaluation, summary export and source-weight sensitivity.
- `archived_results/`: archived JSONs, if present, are copied from provided experimental outputs **not produced by this reconstruction**.

## Running (optional, not required to edit the manuscript)

From this folder:

```bash
python -m pip install numpy
python experiment.py --self-check  # no training; checks split isolation and threshold policy
python experiment.py --all-seeds   # computationally expensive, writes reconstructed_outputs/
```

For the optional DNN reference, install an appropriate TensorFlow version and use `python experiment.py --neural`. `--neural` is exploratory and may execute additional baselines not included in the manuscript's ten-seed main comparison. The reconstruction does not overwrite the historical `outputs/` folder. Supply an explicit output path if desired.

## What must still be verified

1. The exact historical generator, train/validation/test stratification and random-number consumption order.
2. Training/evaluation schedules and initialization semantics in the original missing `main.py`, `data.py`, `federated.py`, and `metrics.py`.
3. Whether a fresh full run matches archived metrics, source-weight sensitivity, prevalence and scalar counts within a documented tolerance. **Not tested or claimed.**
4. Dependency versions and computing environment used in the original run.
5. Only after this audit should the paper describe this repository snapshot as the implementation that produced the published numerical results.

The original `main` branch and its old `outputs/` are preserved for provenance. This folder and branch are a reviewable proposal.
