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

`FitFailureWarning` is deduplicated on top of the filters. A model fit happens
many times per learner (once per tuning trial, once per CV fold, once for the
final full-train fit, again for calibration), so a single unstable design could
otherwise print the same line dozens of times. The learner keeps track of which
messages it has already shown and skips repeats, so you get one line per
distinct message even where the `IterativeWarning` "once" filter is reset by
the machinery that captures warnings out of worker processes.

Both layers key on the message text, and Python's own "once" filter resets
whenever the filter list changes, which mlsurv does often while capturing
warnings out of worker processes. So a repeat may or may not print. If you are
fitting many learners in a loop (a leave-one-group-out sweep, say) and need to
know which cohorts were affected, don't count lines. Ask each model:

```python
model = learner.models['coxph'].model.named_steps['model']
model.overflow_detected_      # True if numerical instability was detected
model.alpha_                  # the penalty it settled on
model.unstable_fit_kept_      # True if nothing resolved that instability
model.converged_              # False if the optimizer ran out of iterations
model.hr_suppressed_reason_   # why hazard ratios are missing, or None
model.covariance_unavailable_reason_   # why their CIs are missing, or None
```

`overflow_detected_` says instability was *seen*; `unstable_fit_kept_` says
whether the fit you got is still affected by it. They differ whenever the
fallback escalated successfully (seen, then resolved), which is the common
case, so read the second one before deciding whether to trust the survival
probabilities.

`converged_` is tracked separately from `overflow_detected_` on purpose: a fit
can converge and still need a penalty, or hit the iteration limit with no
overflow at all. `None` means the outcome is not known, which is not the same
as "did not converge". Read it with `getattr(model, 'converged_', None)` if the
model might come from a learner saved before this was recorded.

`hr_suppressed_reason_` explains why the hazard-ratio table and forest plot
were withheld, by either route: a penalty you passed, or one the fallback
escalated to *past the first rung*. An escalation to the first rung (0.01)
leaves the table in place, so the reason stays `None` there. The report prints
whatever it returns, so you do not have to keep the fit-time warning to know
why the section is missing.

`covariance_unavailable_reason_` is the separate question of whether the
*intervals around* those hazard ratios could be computed. The two are
independent: the hazard ratios can be perfectly reportable while the observed
information matrix will not invert, which costs every confidence interval and
Wald p-value but leaves the point estimates intact. When it is set,
`coefficients` returns NaN in `hr_lower_95` / `hr_upper_95` / `p_value`,
`plot_hazard_ratios` draws no whiskers, `format_hazard_ratios` puts `'N/A'` in
the CI column, and each of those says so with a `SuggestionWarning` rather than
leaving you to notice. It is rare in practice: on a rank-deficient design the
alpha fallback usually regularizes the fit enough that the inversion succeeds.

`ExportSplitWarning` is a `SuggestionWarning` that fires on every export whose
output is not all where you asked for it — result tables too large for a
worksheet, or files an earlier export left in the same directory — rather than
once per session. Each one describes where a particular run put its files, so
deduplicating them would hide the second export's layout behind the first one's.
Silencing the `SuggestionWarning` tier silences these too; the `_large_tables`
sheet inside the workbook always records the split.

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
