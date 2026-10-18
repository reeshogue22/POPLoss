# POPLoss

[![tests](https://github.com/reeshogue22/POPLoss/actions/workflows/tests.yml/badge.svg)](https://github.com/reeshogue22/POPLoss/actions/workflows/tests.yml)

Preference Optimization for Pretraining: cross-entropy plus an
[ORPO](https://arxiv.org/abs/2403.07691)-style odds-ratio term at every token position. The
data token is "chosen"; a token sampled from the model itself (temperature 0.7, no gradient) is
"rejected":

```
loss = CE(x, y) - beta * mean log sigmoid( log odds(y) - log odds(y_sampled) ),   odds(p) = p / (1 - p)
```

The idea: besides raising the target, push the model away from its own plausible-but-wrong
continuations.

```python
from pop import POPLoss
loss = POPLoss(beta=0.1)(logits.flatten(0, 1), targets.flatten())   # logits (N, vocab), targets (N,)
```

## Result: no gain over cross-entropy at this scale

`experiments/char_lm.py`: a 4-layer, 192-wide char-level transformer, 2000 steps, AdamW with
one-cycle LR, trained on the Python standard library's source and evaluated on held-out files.
Seeds, data order, steps and LR are matched across arms; all arms are scored on held-out
cross-entropy (bits per character). Raw rows: `experiments/results.json`.

| beta | held-out bits/char, seeds 0 / 1 / 2 | mean |
|---|---|---|
| 0 (plain CE) | 1.892 / 1.899 / 1.891 | **1.894** |
| 0.1 | 1.891 / 1.908 / 1.893 | 1.897 |
| 0.5 | 1.915 / 1.957 / 1.924 | 1.932 |

beta=0.1 is within seed noise of plain cross-entropy; beta=0.5 is worse on every seed. One
small model and one corpus, so this does not rule out an effect at scale, but there is no
signal here to justify one.

```sh
python experiments/char_lm.py --betas 0 0.1 0.5 --seeds 0 1 2   # ~6 min on an M3 Ultra (MPS)
```

## Tests

```sh
pip install torch pytest && pytest -q
```

**Fixed in 2026:** the original (2024) version *added* `beta * log sigmoid(...)`, so the
preference term pushed the target token down rather than up, and it returned non-finite
values when the model was certain of the target. `tests/test_pop.py` covers both. The
experiment above uses the corrected loss. The fix and the experiment were done with Claude
Code.

MIT licence.
