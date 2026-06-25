"""Feature Group Coloring Demo — GBSG2 breast-cancer dataset.

Generates 15 interactive Plotly figures showcasing every plot function that
supports feature-group coloring, then compiles them into a single HTML file.
Every figure is produced through the ``learner.plot_*()`` wrappers — the same
API the vignettes and docs use.

Usage
-----
    conda run -n mlsurv python scripts/demo_feature_groups.py
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

from sksurv.datasets import load_gbsg2

# ── mlsurv imports ──────────────────────────────────────────────────────────
from mlsurv import SurvivalLearner
from mlsurv.utils import to_structured_array

# ── Constants ───────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "age": "demographic",
    "menostat": "demographic",
    "tsize": "clinical",
    "tgrade": "clinical",
    "pnodes": "clinical",
    "progrec": "molecular",
    "estrec": "molecular",
    "horTh": "treatment",
}

GROUP_COLORS = {
    "demographic": "#ff7f0e",   # orange
    "clinical": "#1f77b4",      # blue
    "molecular": "#2ca02c",     # green
    "treatment": "#d62728",     # red
}

_DEFAULT_OUTPUT_PATH = Path(__file__).parent / "output" / "feature_groups_demo.html"


# ── Data preparation ────────────────────────────────────────────────────────
def prepare_data():
    """Load GBSG2, train CoxPH + RSF, and evaluate so the plot wrappers have data.

    Each model's evaluation is stored on its own ``ModelResult`` and the
    ``learner.plot_*`` wrappers read it back by model name, so the order of the
    two ``evaluate()`` calls does not matter. CoxPH is evaluated with
    ``include_shap=True`` and ``n_importance_repeats=2`` so its stored result
    carries the SHAP values and per-iteration importance the single-model and
    SHAP figures need; RSF is evaluated without SHAP. Both models keep the
    point-estimate importance the multi-model figures read.
    """
    print("Loading GBSG2 dataset …")
    X, y = load_gbsg2()
    y = to_structured_array(y)

    # Encode categoricals (mlsurv requires numeric input)
    X = X.copy()
    X["horTh"] = (X["horTh"] == "yes").astype(float)
    X["menostat"] = (X["menostat"] == "Post").astype(float)
    X["tgrade"] = X["tgrade"].cat.codes.astype(float)

    learner = SurvivalLearner(X, y)
    learner.setup(
        test_fraction=0.25,
        random_state=42,
        verbose=False,
    )

    print("Training CoxPH + RSF …")
    learner.train(["coxph", "rsf"], verbose=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # SHAP is computed for CoxPH only (the SHAP figures use it); order is irrelevant.
        print("Evaluating RSF …")
        learner.evaluate(
            "rsf",
            n_importance_repeats=2,
            verbose=False,
            show_progress=False,
        )
        print("Evaluating CoxPH (importance iterations + SHAP) …")
        learner.evaluate(
            "coxph",
            n_importance_repeats=2,
            include_shap=True,
            verbose=False,
            show_progress=False,
        )

    return learner


# ── Figure generators ───────────────────────────────────────────────────────
# Each returns (title, api_call_str, figure). All figures go through the
# learner.plot_*() wrappers, which source their data from the stored evaluation.

def fig_01_importance_bar(learner):
    title = "1. plot_importance — bar (single-model)"
    api = textwrap.dedent("""\
        learner.plot_importance(
            models="coxph",
            style="bar",
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
        )""")
    fig = learner.plot_importance(
        models="coxph", style="bar",
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_02_importance_forest(learner):
    title = "2. plot_importance — forest (single-model)"
    api = textwrap.dedent("""\
        learner.plot_importance(
            models="coxph",
            style="forest",
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
        )""")
    fig = learner.plot_importance(
        models="coxph", style="forest",
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_03_importance_raincloud(learner):
    title = "3. plot_importance — raincloud"
    api = textwrap.dedent("""\
        learner.plot_importance(
            models="coxph",
            style="raincloud",   # per-iteration distribution (needs n_importance_repeats>=2)
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
        )""")
    fig = learner.plot_importance(
        models="coxph", style="raincloud",
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_04_importance_boxplot(learner):
    title = "4. plot_importance — boxplot"
    api = textwrap.dedent("""\
        learner.plot_importance(
            models="coxph",
            style="boxplot",
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
        )""")
    fig = learner.plot_importance(
        models="coxph", style="boxplot",
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_05_importance_multi_bar(learner):
    title = "5. plot_importance — multi-model bar"
    api = textwrap.dedent("""\
        learner.plot_importance(
            models=["coxph", "rsf"],
            style="bar",
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
        )""")
    fig = learner.plot_importance(
        models=["coxph", "rsf"], style="bar",
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_06_importance_multi_forest(learner):
    title = "6. plot_importance — multi-model forest"
    api = textwrap.dedent("""\
        learner.plot_importance(
            models=["coxph", "rsf"],
            style="forest",
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
        )""")
    fig = learner.plot_importance(
        models=["coxph", "rsf"], style="forest",
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_07_importance_facet(learner):
    title = "7. plot_importance — facet_by_group"
    api = textwrap.dedent("""\
        learner.plot_importance(
            models="coxph",
            style="bar",
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
            facet_by_group=True,
        )""")
    fig = learner.plot_importance(
        models="coxph", style="bar", facet_by_group=True,
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_08_importance_group_agg_bar(learner):
    title = "8. plot_importance — group_agg bar"
    api = textwrap.dedent("""\
        learner.plot_importance(
            models="coxph",
            style="bar",
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
            group_agg=True,
        )""")
    fig = learner.plot_importance(
        models="coxph", style="bar", group_agg=True,
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_09_importance_group_agg_boxplot(learner):
    title = "9. plot_importance — group_agg boxplot"
    api = textwrap.dedent("""\
        learner.plot_importance(
            models="coxph",
            style="boxplot",
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
            group_agg=True,
        )""")
    fig = learner.plot_importance(
        models="coxph", style="boxplot", group_agg=True,
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_10_heatmap(learner):
    title = "10. plot_importance (heatmap) — group color strip"
    api = textwrap.dedent("""\
        learner.plot_importance(
            models=["coxph", "rsf"],
            style="heatmap",
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
        )""")
    fig = learner.plot_importance(
        models=["coxph", "rsf"], style="heatmap",
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_11_hazard_ratios(learner):
    title = "11. plot_hazard_ratios — group coloring"
    api = textwrap.dedent("""\
        learner.plot_hazard_ratios(
            kind="multivariate",   # hazard ratios from the fitted CoxPH model
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
        )""")
    fig = learner.plot_hazard_ratios(
        kind="multivariate",
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_12_shap_violin(learner):
    title = "12. plot_shap — importance, violin"
    api = textwrap.dedent("""\
        learner.plot_shap(
            models="coxph",
            kind="importance",
            show_violin=True,
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
        )""")
    fig = learner.plot_shap(
        models="coxph", kind="importance", show_violin=True,
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_13_shap_bar(learner):
    title = "13. plot_shap — importance, bar"
    api = textwrap.dedent("""\
        learner.plot_shap(
            models="coxph",
            kind="importance",
            show_violin=False,
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
        )""")
    fig = learner.plot_shap(
        models="coxph", kind="importance", show_violin=False,
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_14_shap_beeswarm(learner):
    title = "14. plot_shap — beeswarm, colored tick labels"
    api = textwrap.dedent("""\
        learner.plot_shap(
            models="coxph",
            kind="beeswarm",
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
        )""")
    fig = learner.plot_shap(
        models="coxph", kind="beeswarm",
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


def fig_15_correlation(learner):
    title = "15. plot_feature_correlation — color strips"
    api = textwrap.dedent("""\
        learner.plot_feature_correlation(
            feature_groups=FEATURE_GROUPS,
            group_colors=GROUP_COLORS,
        )""")
    fig = learner.plot_feature_correlation(
        feature_groups=FEATURE_GROUPS, group_colors=GROUP_COLORS,
    )
    return title, api, fig


# ── HTML builder ────────────────────────────────────────────────────────────
HTML_TEMPLATE = textwrap.dedent("""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Feature Group Coloring Demo — mlsurv</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
      :root {{
        --bg: #fafafa;
        --card: #ffffff;
        --border: #e0e0e0;
        --text: #1a1a2e;
        --accent: #0f3460;
        --code-bg: #f4f4f8;
      }}
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
        background: var(--bg);
        color: var(--text);
        line-height: 1.6;
        padding: 2rem 1rem;
      }}
      .container {{ max-width: 1100px; margin: 0 auto; }}
      h1 {{
        font-size: 1.8rem;
        color: var(--accent);
        margin-bottom: 0.3rem;
      }}
      .subtitle {{
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      }}
      .card h2 {{
        font-size: 1.15rem;
        color: var(--accent);
        margin-bottom: 0.8rem;
      }}
      pre {{
        background: var(--code-bg);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.8rem 1rem;
        overflow-x: auto;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 1rem;
      }}
      code {{ font-family: "Cascadia Code", "Fira Code", "Consolas", monospace; }}
      .legend-bar {{
        display: flex;
        gap: 1.2rem;
        flex-wrap: wrap;
        margin-bottom: 2rem;
        padding: 0.6rem 1rem;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 6px;
      }}
      .legend-item {{
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.9rem;
      }}
      .legend-dot {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
      }}
    </style>
    </head>
    <body>
    <div class="container">
    <h1>Feature Group Coloring Demo</h1>
    <p class="subtitle">
      GBSG2 breast-cancer dataset &mdash; 686 patients, 8 features, 4 groups
    </p>

    <div class="legend-bar">
      <div class="legend-item">
        <span class="legend-dot" style="background:#ff7f0e"></span> demographic
      </div>
      <div class="legend-item">
        <span class="legend-dot" style="background:#1f77b4"></span> clinical
      </div>
      <div class="legend-item">
        <span class="legend-dot" style="background:#2ca02c"></span> molecular
      </div>
      <div class="legend-item">
        <span class="legend-dot" style="background:#d62728"></span> treatment
      </div>
    </div>

    {cards}
    </div>
    </body>
    </html>
""")

CARD_TEMPLATE = textwrap.dedent("""\
    <div class="card">
      <h2>{title}</h2>
      <pre><code>{api}</code></pre>
      {plot_div}
    </div>
""")


def build_html(figures):
    """Compile list of (title, api_str, fig) into a single HTML file."""
    cards = []
    for title, api, fig in figures:
        div = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": True, "responsive": True},
        )
        cards.append(CARD_TEMPLATE.format(
            title=title,
            api=api.replace("<", "&lt;").replace(">", "&gt;"),
            plot_div=div,
        ))
    return HTML_TEMPLATE.format(cards="\n".join(cards))


# ── Main ────────────────────────────────────────────────────────────────────
def main(output_dir=None):
    learner = prepare_data()

    print("Generating 15 figures …")
    figures = [
        fig_01_importance_bar(learner),
        fig_02_importance_forest(learner),
        fig_03_importance_raincloud(learner),
        fig_04_importance_boxplot(learner),
        fig_05_importance_multi_bar(learner),
        fig_06_importance_multi_forest(learner),
        fig_07_importance_facet(learner),
        fig_08_importance_group_agg_bar(learner),
        fig_09_importance_group_agg_boxplot(learner),
        fig_10_heatmap(learner),
        fig_11_hazard_ratios(learner),
        fig_12_shap_violin(learner),
        fig_13_shap_bar(learner),
        fig_14_shap_beeswarm(learner),
        fig_15_correlation(learner),
    ]

    html = build_html(figures)
    if output_dir:
        output_path = Path(output_dir) / "feature_groups_demo.html"
    else:
        output_path = _DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}  ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
