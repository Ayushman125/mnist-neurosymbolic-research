# Findings from the MNIST Neuro-Symbolic Addition Project

## Short abstract

This project compares a neuro-symbolic digit-addition model against a dense end-to-end baseline on paired MNIST digits. The key idea is to keep the arithmetic rule fixed and learn only the perception step, then test how far that separation carries under standard evaluation, out-of-distribution digit ranges, and additive noise.

## Experimental map

1. Pre-train a small CNN to recognize single MNIST digits.
2. Wrap that CNN in a symbolic sum layer that computes the distribution over sums from the two digit distributions.
3. Train a dense baseline that predicts the sum directly from concatenated pixels.
4. Evaluate both models on the standard split, on digit-range shift, and under Gaussian noise.

## Full-run results

These are the numbers from running `python nesy_addition.py` in this repository.

| Experiment | Neuro-Symbolic | Baseline |
| --- | ---: | ---: |
| Standard test split | 98.05% | 72.80% |
| OOD digit-range split | 18.40% | 0.00% |
| Noise level 0.0 | 4.00% | 0.00% |
| Noise level 0.3 | 2.55% | 0.00% |
| Noise level 0.6 | 0.70% | 0.00% |
| Noise level 0.9 | 0.20% | 0.00% |

## What the results mean

- The standard split shows that the neuro-symbolic path can solve the task cleanly when the digit recognizer is sufficiently trained.
- The OOD split shows the baseline collapsing completely, which is consistent with a model that learns pixels-to-sum shortcuts rather than an arithmetic rule.
- The noise experiment shows the bottleneck clearly: the symbolic layer is stable, but it cannot recover from a weak perception model.

## Interpretation you can present to someone

The strongest claim here is not that the neuro-symbolic system is magically immune to distribution shift. The defensible claim is narrower and better: once the perception module is good enough, the explicit sum rule gives a cleaner, more interpretable path than a pure neural baseline. The failure mode is equally useful, because it shows exactly where the system breaks when the input channel becomes unreliable.

## Reproducibility

- Main script: [nesy_addition.py](nesy_addition.py)
- Quick smoke test: `python nesy_addition.py --quick`
- Full run: `python nesy_addition.py`
- Generated artifacts: `results/summary.json`, `results/ood_accuracy_chart.png`, `results/noise_robustness_chart.png`

## Practical note

The quick mode is useful for checking that the pipeline still works, but it is not the right run to cite in a paper or presentation. Use the full run if you want the most defensible results from this codebase.