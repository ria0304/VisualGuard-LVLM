| Method | POPE Random F1 | POPE Popular F1 | POPE Adv F1 | Hallucination (mean, lower better) | POPE worst-setting F1 | MME total |
| --- | --- | --- | --- | --- | --- | --- |
| A. baseline (greedy) | not run | not run | not run | not run | not run | 1776.38 |
| G. full VisualGuard (mme_visualguard_s42) | not run | not run | not run | not run | not run | 1775.76 |
| G. full VisualGuard (sensitivity_lambda_025) | not run | not run | not run | not run | not run | 1776.51 |
| G. full VisualGuard (sensitivity_lambda_075) | not run | not run | not run | not run | not run | 1774.51 |
| G. full VisualGuard (sensitivity_threshold_025) | not run | not run | not run | not run | not run | 1776.38 |
| G. full VisualGuard (sensitivity_threshold_045) | not run | not run | not run | not run | not run | 1770.47 |
| G. full VisualGuard (visualguard_adversarial_n1200) | not run | not run | 0.8129 | 0.1000 | 0.8129 | not run |
| G. full VisualGuard (visualguard_popular_n1200) | not run | 0.8386 | not run | 0.0433 | 0.8386 | not run |
| G. full VisualGuard (visualguard_random_n1000) | 0.8565 | not run | not run | 0.0200 | 0.8565 | not run |

Notes:
- WARNING: rows use different models (['llava-hf/llava-1.5-7b-hf', 'llava-hf/llava-interleave-qwen-0.5b-hf']); they are NOT comparable.
- WARNING: 3 run(s) used --max-samples; those rows are not comparable to a full-dataset run.

Cells marked 'not run' were not measured. They are not zeros.
