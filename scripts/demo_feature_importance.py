"""Demo: Feature importance plot gallery.

Runs a real survival analysis on GBSG2 breast cancer data and generates
an HTML gallery of the feature-importance plots, each produced through the
learner.plot_*() methods.

Usage:
    conda run -n mlsurv python scripts/demo_feature_importance.py
    conda run -n mlsurv python scripts/demo_feature_importance.py --fast
"""

import html
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.datasets import load_gbsg2

from mlsurv import SurvivalLearner
from mlsurv.reports.sections.feature_analysis._importance import build_feature_row
from mlsurv.utils import to_structured_array

def main(output_dir=None, fast=False):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*convergence.*")
    warnings.filterwarnings("ignore", message=".*deprecated.*")

    # ── Data setup ────────────────────────────────────────────────────────
    print("Loading GBSG2 data and training models...")
    X_raw, y_raw = load_gbsg2()
    X = pd.get_dummies(X_raw, drop_first=True)
    y = to_structured_array(
        times=[row[1] for row in y_raw],
        events=[bool(row[0]) for row in y_raw],
    )

    learner = SurvivalLearner(X, y)
    learner.setup(test_fraction=0.3, random_state=42, cv_folds=3)
    learner.train(models=["coxph", "coxnet"], verbose=False)
    learner.evaluate(verbose=False, n_importance_repeats=2 if fast else "auto")
    print("  Models trained & evaluated: CoxPH, CoxNet")

    # Run PIMP for both models
    print("  Running PIMP for CoxPH...")
    n_perm = 5 if fast else 20
    learner.importance_test(models="coxph", n_permutations=n_perm, verbose=False)
    print("  Running PIMP for CoxNet...")
    learner.importance_test(models="coxnet", n_permutations=n_perm, verbose=False)
    print("  PIMP done for both models")

    # ── Feature groups (clinical domain knowledge) ────────────────────────
    GROUPS = {
        "age": "Demographics",
        "menostat_Post": "Demographics",
        "tsize": "Tumor",
        "pnodes": "Tumor",
        "tgrade_II": "Tumor",
        "tgrade_III": "Tumor",
        "progrec": "Biomarker",
        "estrec": "Biomarker",
        "horTh_yes": "Treatment",
    }
    # group_colors omitted — mlsurv assigns colors automatically

    # ── Figure gallery ────────────────────────────────────────────────────
    figures = []  # (title, code, fig)

    def add(title, code, fig):
        figures.append((title, code, fig))

    print("\nGenerating figures...")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. plot_importance — single model
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    add("learner.plot_importance() — Default sort",
        """\
fig = learner.plot_importance(models="coxph")""",
        learner.plot_importance(models="coxph"))

    add("learner.plot_importance() — Groups visible, default sort",
        """\
fig = learner.plot_importance(
    models="coxph",
    feature_groups=GROUPS,
    sort_by_group=False,   # colors the bars by group, but keeps metric order
)""",
        learner.plot_importance(
            models="coxph",
            feature_groups=GROUPS,
            sort_by_group=False))

    add("learner.plot_importance() — sort_by_group=True",
        """\
fig = learner.plot_importance(
    models="coxph",
    feature_groups=GROUPS,
    sort_by_group=True,    # reorders by group, adds dotted separators
)""",
        learner.plot_importance(
            models="coxph",
            feature_groups=GROUPS,
            sort_by_group=True))

    add("learner.plot_importance() — sort_by_group + style='forest' + CI",
        """\
fig = learner.plot_importance(
    models="coxph",
    feature_groups=GROUPS,

    sort_by_group=True,
    style="forest",
    error_bars="ci", ci_level=0.95,
)""",
        learner.plot_importance(
            models="coxph",
            feature_groups=GROUPS,        sort_by_group=True,
            style="forest", error_bars="ci", ci_level=0.95))

    add("learner.plot_importance() — sort_by_group + show_labels",
        """\
fig = learner.plot_importance(
    models="coxph",
    feature_groups=GROUPS,

    sort_by_group=True,
    show_labels=True,
)""",
        learner.plot_importance(
            models="coxph",
            feature_groups=GROUPS,        sort_by_group=True, show_labels=True))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. plot_importance — multi-model
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    add("learner.plot_importance() — Multi-model, sort_by_group",
        """\
fig = learner.plot_importance(
    models=["coxph", "coxnet"],
    feature_groups=GROUPS,

    sort_by_group=True,
)""",
        learner.plot_importance(
            models=["coxph", "coxnet"],
            feature_groups=GROUPS,        sort_by_group=True))

    add("learner.plot_importance() — Multi-model, default sort",
        """\
fig = learner.plot_importance(models=["coxph", "coxnet"])""",
        learner.plot_importance(models=["coxph", "coxnet"]))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. plot_importance style="heatmap"
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    add("learner.plot_importance(style='heatmap') — sort_by_group",
        """\
fig = learner.plot_importance(
    style="heatmap",
    feature_groups=GROUPS,
    sort_by_group=True,
)""",
        learner.plot_importance(
            style="heatmap",
            feature_groups=GROUPS, sort_by_group=True))

    add("learner.plot_importance(style='heatmap') — Default",
        """\
fig = learner.plot_importance(style="heatmap")""",
        learner.plot_importance(style="heatmap"))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. plot_coefficients
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    add("learner.plot_coefficients() — sort_by_group, forest",
        """\
fig = learner.plot_coefficients(
    feature_groups=GROUPS,

    sort_by_group=True,
    style="forest",
)""",
        learner.plot_coefficients(
            feature_groups=GROUPS,        sort_by_group=True, style="forest"))

    add("learner.plot_coefficients() — sort_by_group, bar style",
        """\
fig = learner.plot_coefficients(
    feature_groups=GROUPS,

    sort_by_group=True,
    style="bar",
)""",
        learner.plot_coefficients(
            feature_groups=GROUPS,        sort_by_group=True, style="bar"))

    add("learner.plot_coefficients() — Default (style='bar')",
        """\
fig = learner.plot_coefficients()
# Default style is 'bar' (horizontal bars). Use style='forest' for point+CI.""",
        learner.plot_coefficients())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. plot_hazard_ratios — univariate and multivariate
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    add("learner.plot_hazard_ratios(kind='univariate') — sort_by_group",
        """\
fig = learner.plot_hazard_ratios(
    kind="univariate",
    feature_groups=GROUPS,

    sort_by_group=True,
)""",
        learner.plot_hazard_ratios(
            kind="univariate",
            feature_groups=GROUPS,        sort_by_group=True))

    add("learner.plot_hazard_ratios(kind='multivariate') — sort_by_group",
        """\
fig = learner.plot_hazard_ratios(
    kind="multivariate",
    feature_groups=GROUPS,

    sort_by_group=True,
)""",
        learner.plot_hazard_ratios(
            kind="multivariate",
            feature_groups=GROUPS,        sort_by_group=True))

    add("learner.plot_hazard_ratios() — Default",
        """\
fig = learner.plot_hazard_ratios()""",
        learner.plot_hazard_ratios())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. plot_hazard_ratios_comparison
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    add("learner.plot_hazard_ratios_comparison() — with groups",
        """\
fig = learner.plot_hazard_ratios_comparison(
    feature_groups=GROUPS,

)
# Note: always sorts by group when feature_groups is provided""",
        learner.plot_hazard_ratios_comparison(
            feature_groups=GROUPS))

    add("learner.plot_hazard_ratios_comparison() — Default",
        """\
fig = learner.plot_hazard_ratios_comparison()""",
        learner.plot_hazard_ratios_comparison())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. plot_feature_correlation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    add("learner.plot_feature_correlation() — sort_by_group",
        """\
fig = learner.plot_feature_correlation(
    feature_groups=GROUPS,

    sort_by_group=True,
)""",
        learner.plot_feature_correlation(
            feature_groups=GROUPS,        sort_by_group=True))

    add("learner.plot_feature_correlation() — Default (hierarchical clustering)",
        """\
fig = learner.plot_feature_correlation()""",
        learner.plot_feature_correlation())


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 2: Multi-model PIMP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pimp_results = learner.pimp_importance_test_results
    pimp_html = "<h2>Multi-Model PIMP Results</h2>\n"
    pimp_html += "<p>PIMP p-values are stored per-model and read back via <code>learner.pimp_importance_test_results</code>.</p>\n"
    pimp_html += f"<p>Models with PIMP results: <b>{', '.join(pimp_results.keys())}</b></p>\n"

    for model_name, result_df in pimp_results.items():
        pimp_html += f"<h3>{model_name}</h3>\n"
        styled = result_df[["feature", "observed_importance", "p_value", "p_adjusted"]].copy()
        styled.columns = ["Feature", "Observed Importance", "p-value", "p (adjusted)"]
        styled = styled.sort_values("p-value")
        pimp_html += styled.to_html(index=False, float_format="%.4f", classes="data-table")
        pimp_html += "\n"

    pimp_html += """
<div class="code-block"><pre><code># Run the test for each model — results stored per-model
learner.importance_test(models="coxph", n_permutations=100)
learner.importance_test(models="coxnet", n_permutations=100)

# Each model's result is read back from the per-model dict
learner.pimp_importance_test_results
# → {"coxph": DataFrame(...), "coxnet": DataFrame(...)}
learner.pimp_importance_test_results["coxph"]   # name the model you want

# The dict is the only public accessor: PIMP is single-model-per-call,
# so it preserves every model (and is what the reports read).</code></pre></div>
"""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 3: Per-feature sample size
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Simulate univar_cox with varying missing data per feature
    n_features = len(learner.X_train.columns)
    rng = np.random.default_rng(42)
    univar_cox = pd.DataFrame({
        "feature": learner.X_train.columns.tolist(),
        "p_value": rng.uniform(0.001, 0.5, n_features),
        "hazard_ratio": rng.uniform(0.5, 2.0, n_features),
        "n_missing": [0, 15, 0, 42, 0, 0, 8, 0, 0][:n_features],
    })
    total_samples = len(learner.X_train)

    importance_df = learner.plot_importance(models="coxph", return_data=True)
    coxph_imp = importance_df[importance_df["model"] == "coxph"].copy()

    # Build feature-summary-style rows: the row builder
    # consumes rows of learner.feature_summary() (uni_* columns), so
    # merge the univariate stats under those names here.
    ucx = univar_cox.set_index("feature")
    coxph_imp["uni_p_value"] = coxph_imp["feature"].map(ucx["p_value"])
    coxph_imp["uni_hazard_ratio"] = coxph_imp["feature"].map(ucx["hazard_ratio"])
    coxph_imp["correlated"] = False

    sample_rows = []
    for _, summary_row in coxph_imp.iterrows():
        row_data = build_feature_row(
            summary_row, "coxph", 0.7,
            univar_cox=univar_cox, total_samples=total_samples,
        )
        sample_rows.append(row_data)

    sample_df = pd.DataFrame(sample_rows)

    sample_html = "<h2>Per-Feature Sample Size</h2>\n"
    sample_html += (
        "<p>When univariate Cox uses per-feature complete-case analysis and some features "
        "have missing data, the <code>n</code> column shows the effective sample size "
        "each feature was fitted on.</p>\n"
    )
    sample_html += (
        f"<p>Total training samples: <b>{total_samples}</b>. "
        f"Features with missing data: <b>"
        + ", ".join(univar_cox.loc[univar_cox["n_missing"] > 0, "feature"].tolist())
        + "</b></p>\n"
    )
    sample_html += sample_df.to_html(index=False, classes="data-table")
    sample_html += """
<div class="code-block"><pre><code># The 'n' column appears automatically in HTML reports when
# univar_cox has any features with n_missing > 0.
#
# It answers: "This p-value was computed on HOW MANY observations?"
# A feature with 90% missing has its p-value on only 10% of data —
# now that's visible in the table.
#
# No API change needed — just run generate_report() as usual.
# The n column appears when the data warrants it.</code></pre></div>
"""


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Build HTML
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print(f"\nBuilding HTML with {len(figures)} figures...")

    html_parts = ["""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Feature Importance Plot Gallery</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 1100px; margin: 0 auto; padding: 20px 40px;
    background: #fafafa; color: #1a1a1a;
  }
  h1 { border-bottom: 2px solid #333; padding-bottom: 12px; margin-top: 40px; }
  h2 { color: #0366d6; margin-top: 50px; border-bottom: 1px solid #e1e4e8; padding-bottom: 8px; }
  h3 { color: #444; }
  .card {
    background: #fff; border: 1px solid #e1e4e8; border-radius: 8px;
    margin: 30px 0; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  .card-title { font-size: 18px; font-weight: 600; margin: 0 0 16px 0; color: #24292e; }
  .code-block {
    background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px;
    padding: 16px; margin: 16px 0; overflow-x: auto;
  }
  .code-block pre { margin: 0; font-family: "SFMono-Regular", Consolas, monospace; font-size: 13px; line-height: 1.5; }
  .code-block code { color: #24292e; }
  .data-table {
    border-collapse: collapse; margin: 16px 0; font-size: 14px; width: 100%;
  }
  .data-table th, .data-table td {
    border: 1px solid #e1e4e8; padding: 8px 12px; text-align: left;
  }
  .data-table th { background: #f6f8fa; font-weight: 600; }
  .data-table tr:nth-child(even) { background: #fafbfc; }
  .toc { background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px 30px; margin: 20px 0; }
  .toc ul { list-style: none; padding-left: 0; }
  .toc li { margin: 6px 0; }
  .toc a { color: #0366d6; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
  .sep { border: none; border-top: 2px solid #0366d6; margin: 60px 0 40px 0; }
</style>
</head>
<body>

<h1>Feature Importance Plot Gallery</h1>
<p>Sort by group, multi-model PIMP, and per-feature sample size &mdash; GBSG2 breast cancer dataset (686 patients, 9 features)</p>
<p>All figures generated via <code>learner.plot_*()</code> methods.</p>

<div class="toc">
<b>Contents</b>
<ul>
  <li><a href="#sort-by-group">1. Sort by Group</a> &mdash; <code>sort_by_group=True</code> on 6 plot functions</li>
  <li><a href="#multi-model-pimp">2. Multi-Model PIMP</a> &mdash; per-model PIMP p-values</li>
  <li><a href="#per-feature-n">3. Per-Feature Sample Size</a> &mdash; <code>n</code> column in importance tables</li>
</ul>
</div>

<hr class="sep">
<h2 id="sort-by-group">1. Sort Features by Group</h2>
<p>The <code>sort_by_group=True</code> parameter groups features by clinical domain,
adds dotted separator lines, and sorts by importance within each group.</p>
<div class="code-block"><pre><code>GROUPS = {
    "age": "Demographics", "menostat_Post": "Demographics",
    "tsize": "Tumor", "pnodes": "Tumor", "tgrade_II": "Tumor", "tgrade_III": "Tumor",
    "progrec": "Biomarker", "estrec": "Biomarker",
    "horTh_yes": "Treatment",
}
# group_colors is optional — mlsurv assigns a default palette automatically</code></pre></div>
"""]

    for title, code, fig in figures:
        fig_html = fig.to_html(
            full_html=False, include_plotlyjs=False,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )

        html_parts.append(f"""
<div class="card">
  <p class="card-title">{html.escape(title)}</p>
  <div class="code-block"><pre><code>{html.escape(code)}</code></pre></div>
  {fig_html}
</div>
""")

    html_parts.append('<hr class="sep">')
    html_parts.append(f'<div id="multi-model-pimp">{pimp_html}</div>')
    html_parts.append('<hr class="sep">')
    html_parts.append(f'<div id="per-feature-n">{sample_html}</div>')

    html_parts.append("""
<hr class="sep">
<p style="color: #666; font-size: 13px; text-align: center;">
  Generated by <code>scripts/demo_feature_importance.py</code>
</p>
</body>
</html>
""")

    out_dir = Path(output_dir) if output_dir else Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "demo_feature_importance.html"
    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"\n  Saved to {output_path}")
    print(f"  Open in browser: file:///{output_path.as_posix()}")


if __name__ == "__main__":
    import sys
    main(fast="--fast" in sys.argv)
