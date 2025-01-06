---
title: TabPFN's Strong Out-of-Distribution Performance
description: Analysis of TabPFN's generalization capabilities with Drift-Resilient modifications
---

# TabPFN's Out-of-Distribution Excellence

Recent research demonstrates TabPFN's exceptional out-of-distribution (OOD) performance on tabular data, with further improvements through Drift-Resilient modifications.

## Key Performance Metrics

| Model | OOD Accuracy | OOD ROC AUC |
|-------|--------------|-------------|
| TabPFN Base | 0.688 | 0.786       |
| TabPFN + Drift-Resilient | 0.744 | 0.832       |
| XGBoost | 0.664 | 0.754       |
| CatBoost | 0.677 | 0.766       |

## Technical Improvements

The Drift-Resilient modifications introduce:

- 2nd-order structural causal model for temporal adaptation
- Enhanced pattern recognition across distribution shifts 
- Zero hyperparameter tuning requirement
- Inference in seconds on small/medium datasets

## Benchmark

The enhanced model shows robust generalization across:

- 18 diverse datasets (synthetic + real-world)
- Various temporal shift patterns
- Multiple industry applications

For comprehensive documentation and implementation details, visit the [GitHub repository](https://github.com/automl/Drift-Resilient_TabPFN).

## Citation
```bibtex
@inproceedings{
  helli2024driftresilient,
  title={Drift-Resilient Tab{PFN}: In-Context Learning Temporal Distribution Shifts on Tabular Data},
  author={Kai Helli and David Schnurr and Noah Hollmann and Samuel M{\"u}ller and Frank Hutter},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=p3tSEFMwpG}
}
```