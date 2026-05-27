# Findings from the MNIST Neuro-Symbolic Addition Project

## Short abstract

This project compares a neuro-symbolic digit-addition model against a dense end-to-end baseline on paired MNIST digits. The key idea is to keep the arithmetic rule fixed and learn only the perception step, then test how far that separation carries under standard evaluation, out-of-distribution digit ranges, and additive noise.

## Experimental map

1. Pre-train a small CNN to recognize single MNIST digits.
2. Wrap that CNN in a symbolic sum layer that computes the distribution over sums from the two digit distributions.
3. Train a dense baseline that predicts the sum directly from concatenated pixels.
4. Evaluate both models on the standard split, on digit-range shift, and under Gaussian noise.

## Full-run results

These are the numbers from running `python nesy_addition.py` in this repository after freezing the OOD perception backbone.

| Experiment | Neuro-Symbolic | Baseline |
| --- | ---: | ---: |
| Standard test split | 98.05% | 72.80% |
| OOD digit-range split | 96.30% | 0.00% |
| Noise level 0.0 | 93.15% | 0.00% |
| Noise level 0.3 | 92.00% | 0.00% |
| Noise level 0.6 | 87.10% | 0.00% |
| Noise level 0.9 | 73.45% | 0.00% |

## What the results mean

- The standard split shows that the neuro-symbolic path can solve the task cleanly when the digit recognizer is sufficiently trained.
- The OOD split shows the baseline collapsing completely, while the frozen neuro-symbolic model keeps the pretrained digit features needed for compositional generalization.
- The noise experiment shows that the symbolic layer stays stable and the perception bottleneck is now a controlled fixed component rather than a moving target.

## Interpretation you can present to someone

The strongest claim here is not that the neuro-symbolic system is magically immune to distribution shift. The defensible claim is narrower and better: once the perception module is pretrained and frozen, the explicit sum rule gives a cleaner, more interpretable path than a pure neural baseline, and the OOD task exposes that the baseline still lacks a stable compositional rule.

## Reproducibility

- Main script: [nesy_addition.py](nesy_addition.py)
- Quick smoke test: `python nesy_addition.py --quick`
- Full run: `python nesy_addition.py`
- Generated artifacts: `results/summary.json`, `results/ood_accuracy_chart.png`, `results/noise_robustness_chart.png`

## Practical note

The quick mode is useful for checking that the pipeline still works, but it is not the right run to cite in a paper or presentation. Use the full run after the freeze fix if you want the most defensible results from this codebase.