# Filtering warnings

mlsurv flags methodological risks and suggestions as it works. It does this the
same way scikit-learn does: every warning is a category that subclasses Python's
`UserWarning`, so you control them with the standard library `warnings` module.
The categories are public in `mlsurv.exceptions`.

mlsurv keeps its methodological warnings **loud by default** — it won't hide a
real risk from you. (Suggestions are shown once per issue, and internal
diagnostics are off by default.) The lever is yours to pull.

## The tiers

| Category | Meaning |
|----------|---------|
| `MethodologicalWarning` | A real risk to result validity — a violated proportional-hazards assumption, a tiny or heavily censored cohort, possible leakage. |
| `SuggestionWarning` | A helpful, non-critical nudge (e.g. "consider more CV repeats"). |
| `InternalWarning` | Internal diagnostic detail. Suppressed by default. |
| `IterativeWarning` | Mixin for warnings that repeat across bootstrap / CV / tuning iterations. |

The concrete categories `ClampWarning`, `FitFailureWarning`,
`MetricFailureWarning`, and `SmallSubpopulationWarning` combine a tier with the
`IterativeWarning` mixin, so a filter on either the tier or `IterativeWarning`
catches them.

## Recipes

Silence one tier (keep the rest):

```python
import warnings
from mlsurv.exceptions import SuggestionWarning

warnings.filterwarnings("ignore", category=SuggestionWarning)
```

Silence everything mlsurv emits — all tiers subclass `UserWarning`:

```python
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
```

Silence only inside a block, e.g. a benchmark sweep that rebuilds many learners
and would otherwise repeat the same methodological warning hundreds of times:

```python
import warnings
from mlsurv import SurvivalLearner
from mlsurv.exceptions import MethodologicalWarning, SuggestionWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=MethodologicalWarning)
    warnings.simplefilter("ignore", category=SuggestionWarning)
    for seed in range(30):
        learner = SurvivalLearner(X, y)   # X, y: your survival data
        learner.setup(random_state=seed)
        learner.run(["coxph"])
```

Turn a methodological concern into a hard error (useful in CI):

```python
import warnings
from mlsurv.exceptions import MethodologicalWarning

warnings.filterwarnings("error", category=MethodologicalWarning)
```

Re-enable the internal diagnostic tier while debugging:

```python
from mlsurv.exceptions import enable_internal_warnings

enable_internal_warnings()
```

See [Reporting & Compliance](reporting.md) for what each flag detects and why it
matters.
