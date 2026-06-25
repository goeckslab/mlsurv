"""Shared collapsible glossary block defining the evaluation metrics.

Inserted in the Model Evaluation and Model Development sections so
readers have one consistent reference for what Harrell C, Uno C, the
integrated Brier score, and Antolini C mean — and which of them are
IPCW-weighted. The order matches the bootstrap metrics table. The
Executive Summary shows only the scoring metric raincloud and does
not need the glossary. Time-dependent AUC(t) has its own dedicated
subsection and is not duplicated here.
"""

_GLOSSARY_ENTRIES = [
    (
        "Harrell's C-index",
        "Rank-based concordance: the fraction of comparable subject pairs "
        "whose predicted risk ordering agrees with the observed event "
        "ordering. **Not IPCW** — ignores the censoring distribution, so "
        "it is biased upward under heavy or informative censoring.",
    ),
    (
        "Uno's C-index",
        "Concordance reweighted by the inverse of the censoring survival "
        "function (IPCW). Unbiased under independent censoring; prefer "
        "over Harrell C when censoring is non-trivial.",
    ),
    (
        "Integrated Brier Score (IBS)",
        "Time-averaged squared difference between the predicted survival "
        "probability S(t|x) and the event-free indicator I(T > t), "
        "integrated over a dense grid of event times within the "
        "censoring-stable window [t_min, tau]. Combines calibration and "
        "discrimination in a single probabilistic score (lower is "
        "better). The practical reference is the IBS of a covariate-free "
        "Kaplan-Meier prediction on the same window — a useful model "
        "should beat that baseline. **IPCW** — censored subjects are "
        "reweighted rather than dropped.",
    ),
    (
        "Antolini's C-index (time-dependent)",
        "Concordance that compares predicted survival probabilities "
        "*at each subject's own event time* rather than a single "
        "marginal risk score (Antolini, Boracchi & Biganzoli 2005). "
        "Sensitive to models whose risk ordering varies over follow-up "
        "(e.g., DeepHit, discrete-time hazards). **Not IPCW** in its "
        "classical form — like Harrell's C, it can be biased under "
        "heavy censoring.",
    ),
]


def metric_glossary_block() -> str:
    """Return a collapsible ``<details>`` block defining the metrics.

    The block mixes an HTML ``<details>`` wrapper with a markdown bullet
    list body — the same pattern used for the BibTeX blocks in
    ``references.py`` and the configuration dump in ``environment.py``.
    Insert the returned string as a single markdown line (followed by an
    empty line) wherever the reader first encounters the metric names.
    """
    lines = [
        "<details>",
        "<summary><strong>About the evaluation metrics</strong> "
        "(click to expand)</summary>",
        "",
    ]
    for name, body in _GLOSSARY_ENTRIES:
        lines.append(f"- **{name}.** {body}")
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)
