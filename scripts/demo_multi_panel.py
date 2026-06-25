"""Demo: Multi-Panel Plot Gallery

Runs a real mlsurv analysis on the GBSG2 breast cancer dataset,
then generates an HTML gallery showing each multi-panel plot alongside
the API call that produced it.

Dataset: GBSG2 (German Breast Cancer Study Group 2)
  - 686 patients, 8 features
  - Endpoint: recurrence-free survival

Models: CoxPH, CoxNet

Usage:
    conda run -n mlsurv python scripts/demo_multi_panel.py
    conda run -n mlsurv python scripts/demo_multi_panel.py --fast
"""

import base64
import io
import textwrap
from pathlib import Path

import pandas as pd
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder

from mlsurv.learner import SurvivalLearner

# ─── HTML template ───────────────────────────────────────────────────

HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multi-Panel Plot Gallery</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    background: #fafafa;
    color: #333;
  }
  h1 { color: #1a1a2e; border-bottom: 3px solid #0072B2; padding-bottom: 0.5rem; }
  h2 { color: #0072B2; margin-top: 2.5rem; }
  .card {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    padding: 1.5rem;
    margin: 1.5rem 0;
  }
  .code-block {
    background: #1a1a2e;
    color: #e0e0e0;
    padding: 1rem 1.2rem;
    border-radius: 6px;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.9rem;
    overflow-x: auto;
    white-space: pre;
    line-height: 1.5;
  }
  .code-block .kw { color: #ff79c6; }
  .code-block .str { color: #f1fa8c; }
  .code-block .cmt { color: #6272a4; }
  .plot-container { margin: 1rem 0; }
  .tag {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
  }
  .tag-point-estimates { background: #d4edda; color: #155724; }
  .tag-metric-time { background: #cce5ff; color: #004085; }
  .setup-block {
    background: #f8f9fa;
    border-left: 4px solid #0072B2;
    padding: 1rem 1.2rem;
    border-radius: 0 6px 6px 0;
    margin: 1rem 0;
  }
  .note {
    background: #e8f4fd;
    border-left: 4px solid #0072B2;
    padding: 0.8rem 1rem;
    border-radius: 0 6px 6px 0;
    margin: 0.5rem 0;
    font-size: 0.9rem;
  }
</style>
</head>
<body>
<h1>Multi-Panel Plot Gallery</h1>
<p>Demonstrates <strong>point estimates in multi-panel plots</strong>
   &amp; <strong>multi-panel <code>plot_metric_time</code></strong>.</p>
"""

HTML_TAIL = """\
</body>
</html>
"""


def fig_to_html_div(fig, div_id):
    """Convert a Plotly figure to an embeddable HTML div."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
    )


def code_block(code: str) -> str:
    """Wrap code in a styled block."""
    code = textwrap.dedent(code).strip()
    # Simple syntax highlighting
    import html as html_mod
    code = html_mod.escape(code)
    return f'<div class="code-block">{code}</div>'


def section(title, tags, code, fig_html, note=None):
    """Build a card with title, tags, code, and plot."""
    tag_html = "".join(
        f'<span class="tag tag-{t.lower()}">{t}</span>' for t in tags
    )
    note_html = f'<div class="note">{note}</div>' if note else ""
    return f"""
    <div class="card">
      <h2>{tag_html} {title}</h2>
      {note_html}
      {code_block(code)}
      <div class="plot-container">{fig_html}</div>
    </div>
    """


# ─── Run analysis ───────────────────────────────────────────────────

def main(output_dir=None, fast=False):
    print("Loading GBSG2 dataset...")
    X, y = load_gbsg2()
    X = OneHotEncoder().fit_transform(X)

    # Rename 'cens' → 'event' for mlsurv compatibility
    import numpy as np
    y = np.array(
        [(bool(e), t) for e, t in y],
        dtype=[("event", bool), ("time", float)],
    )

    print("Setting up learner...")
    learner = SurvivalLearner(X, y)
    learner.setup(
        test_fraction=0.3,
        cv_folds=3,
        scoring="harrell",
        random_state=42,
        time_unit="days",
        verbose=False,
    )

    models = ["coxph", "coxnet"]
    print(f"Training {', '.join(m.upper() for m in models)}...")
    if fast:
        learner.train(models=models, verbose=False)
    else:
        learner.run(models, verbose=False)

    print("Evaluating models...")
    learner.evaluate(verbose=False, n_importance_repeats=2 if fast else "auto")

    n_boot = 2 if fast else 50
    print(f"Running bootstrap ({n_boot} iterations)...")
    learner.evaluate(
        bootstrap=True,
        n_bootstrap=n_boot,
        bootstrap_metrics=["harrell", "ipcw", "integrated_brier"],
        verbose=False,
    )

    # ─── Build HTML ──────────────────────────────────────────────────

    sections = []

    # ── Setup preamble ───────────────────────────────────────────────

    setup_code = """\
    from sksurv.datasets import load_gbsg2
    from sksurv.preprocessing import OneHotEncoder
    from mlsurv.learner import SurvivalLearner

    # Load GBSG2 breast cancer dataset (686 patients, 8 features)
    X, y = load_gbsg2()
    X = OneHotEncoder().fit_transform(X)

    # Setup learner
    learner = SurvivalLearner(X, y)
    learner.setup(
        test_fraction=0.3,
        cv_folds=3,
        scoring="harrell",
        random_state=42,
        time_unit="days",
        verbose=False,
    )

    # Train models
    learner.run(["coxph", "coxnet"], verbose=False)

    # Evaluate with bootstrap
    learner.evaluate(verbose=False)
    learner.evaluate(
        bootstrap=True,
        n_bootstrap=2,
        bootstrap_metrics=["harrell", "ipcw", "integrated_brier"],
        verbose=False,
    )
    """

    sections.append(f"""
    <div class="card">
      <h2>Analysis Setup</h2>
      <p>Dataset: <strong>GBSG2</strong> (German Breast Cancer Study Group 2) —
         686 patients, 8 features, recurrence-free survival endpoint.</p>
      <p>Models: <strong>CoxPH</strong>, <strong>CoxNet</strong></p>
      <p>Bootstrap: 2 iterations with Harrell C-index, IPCW C-index,
         and Integrated Brier Score.</p>
      {code_block(setup_code)}
    </div>
    """)

    # ── Multi-panel raincloud with point_estimates ─────────────

    print("Generating plots...")

    fig1 = learner.plot_metric(
        style="raincloud",
        metric=["harrell", "ipcw"],
        source="bootstrap",
    )
    sections.append(section(
        "Multi-Panel Raincloud with Point Estimates",
        ["point-estimates"],
        """\
        fig = learner.plot_metric(
            style="raincloud",
            metric=["harrell", "ipcw"],
            source="bootstrap",
        )
        """,
        fig_to_html_div(fig1, "fig1"),
        note="Triangle markers show test-set point estimates overlaid on "
             "bootstrap distributions. Each metric gets its own panel. "
             "Point estimates are automatically extracted from the evaluation result.",
    ))

    # ── Multi-panel boxplot with point_estimates ───────────────

    fig2 = learner.plot_metric(
        style="boxplot",
        metric=["harrell", "ipcw"],
        source="bootstrap",
    )
    sections.append(section(
        "Multi-Panel Boxplot with Point Estimates",
        ["point-estimates"],
        """\
        fig = learner.plot_metric(
            style="boxplot",
            metric=["harrell", "ipcw"],
            source="bootstrap",
        )
        """,
        fig_to_html_div(fig2, "fig2"),
        note="Same point_estimates feature with boxplot style. "
             "Triangle markers appear automatically when bootstrap source "
             "is used with multi-panel iteration styles.",
    ))

    # ── Multi-panel forest (3 metrics) ─────────────────────────

    fig3 = learner.plot_metric(
        style="forest",
        metric=["harrell", "ipcw", "integrated_brier"],
        source="bootstrap",
    )
    sections.append(section(
        "Multi-Panel Forest (3 Metrics)",
        ["point-estimates"],
        """\
        fig = learner.plot_metric(
            style="forest",
            metric=["harrell", "ipcw", "integrated_brier"],
            source="bootstrap",
        )
        """,
        fig_to_html_div(fig3, "fig3"),
        note="Three-metric panel grid. Each metric gets a dashed reference "
             "line: 0.5 for concordance metrics, 0.25 for Brier.",
    ))

    # ── Multi-panel plot_metric_time (AUC + Brier) ─────────────

    fig4 = learner.plot_metric_time(
        metric=["auc", "brier"],
    )
    sections.append(section(
        "Multi-Panel Time-Dependent Metrics (AUC + Brier)",
        ["metric-time"],
        """\
        fig = learner.plot_metric_time(
            metric=["auc", "brier"],
        )
        """,
        fig_to_html_div(fig4, "fig4"),
        note="Side-by-side AUC(t) and Brier(t) curves. Each panel gets its "
             "own y-axis range ([0.4, 1.0] for AUC, [0, 0.35] for Brier) and "
             "reference line. Shared x-axis (time). Legend is deduplicated — "
             "each model appears once.",
    ))

    # ── Single-metric time (single panel, for comparison) ──────────

    fig5 = learner.plot_metric_time()
    sections.append(section(
        "Single-Metric Time (Single Panel)",
        ["metric-time"],
        """\
        # Default: single AUC(t) panel (no metric= argument)
        fig = learner.plot_metric_time()
        """,
        fig_to_html_div(fig5, "fig5"),
        note="When metric= is not provided, the function draws a "
             "single-panel AUC(t).",
    ))

    # ── Multi-panel time with custom title ─────────────────────

    fig6 = learner.plot_metric_time(
        metric=["auc", "brier"],
        title="GBSG2: Time-Dependent Discrimination & Calibration",
    )
    sections.append(section(
        "Multi-Panel Time with Custom Title",
        ["metric-time"],
        """\
        fig = learner.plot_metric_time(
            metric=["auc", "brier"],
            title="GBSG2: Time-Dependent Discrimination & Calibration",
        )
        """,
        fig_to_html_div(fig6, "fig6"),
        note="Custom title support. All keyword arguments "
             "(title, time_unit, palette, width, height) work with the "
             "multi-panel mode.",
    ))

    # ── Low-level API: plot_metric_time with metric= ─────────────────

    # Build data manually to show the low-level API
    auc_df = learner.plot_metric_time(return_data=True)
    from mlsurv.viz.metrics import plot_metric_time, compute_brier_curves

    # Compute Brier curves
    fitted = {name: mr.model for name, mr in learner._state.models.items()}
    brier_df = compute_brier_curves(
        fitted,
        learner.X_train, learner.y_train,
        learner.X_test, learner.y_test,
    )

    # Combine into long format
    auc_long = auc_df[["model", "eval_time"]].copy()
    auc_long["value"] = auc_df["auc"].values
    auc_long["metric_name"] = "auc"

    brier_long = brier_df[["model", "eval_time"]].copy()
    brier_long["value"] = brier_df["brier_score"].values
    brier_long["metric_name"] = "brier"

    combined = pd.concat([auc_long, brier_long], ignore_index=True)

    fig7 = plot_metric_time(combined, metric=["auc", "brier"])
    sections.append(section(
        "Low-Level API: plot_metric_time(df, metric=...)",
        ["metric-time"],
        """\
        from mlsurv.viz.metrics import plot_metric_time, compute_brier_curves

        # Get AUC(t) data
        auc_df = learner.plot_metric_time(return_data=True)

        # Compute Brier(t) curves
        fitted = {name: mr.model for name, mr in learner._state.models.items()}
        brier_df = compute_brier_curves(
            fitted,
            learner.X_train, learner.y_train,
            learner.X_test, learner.y_test,
        )

        # Combine into long format with metric_name column
        auc_long = auc_df[["model", "eval_time"]].copy()
        auc_long["value"] = auc_df["auc"].values
        auc_long["metric_name"] = "auc"

        brier_long = brier_df[["model", "eval_time"]].copy()
        brier_long["value"] = brier_df["brier_score"].values
        brier_long["metric_name"] = "brier"

        combined = pd.concat([auc_long, brier_long], ignore_index=True)

        # Plot using low-level API
        fig = plot_metric_time(combined, metric=["auc", "brier"])
        """,
        fig_to_html_div(fig7, "fig7"),
        note="The low-level viz function accepts a long-format DataFrame "
             "with a metric_name column. This gives full control over data "
             "preparation. The learner.plot_metric_time(metric=...) method "
             "builds this DataFrame automatically.",
    ))

    # ─── Write HTML ──────────────────────────────────────────────────

    if output_dir:
        out_path = Path(output_dir) / "demo_multi_panel.html"
    else:
        out_path = Path(__file__).parent / "output" / "demo_multi_panel.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(HTML_HEAD)
        for s in sections:
            f.write(s)
        f.write(HTML_TAIL)

    print(f"\nDemo written to: {out_path}")
    print(f"Open in browser to view {len(sections) - 1} figures.")


if __name__ == "__main__":
    import sys
    main(fast="--fast" in sys.argv)
