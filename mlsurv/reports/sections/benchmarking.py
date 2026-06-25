"""Benchmarking section for report generation."""

from __future__ import annotations

import html as html_module
from typing import Any

import numpy as np
import pandas as pd

from mlsurv.reports._constants import (
    BENCHMARK_SUBPOPULATION_VISIBLE_ROWS,
    P_STARS,
    P_THRESHOLD_SIGNIFICANT,
    format_pvalue,
)
from mlsurv.reports.sections.base import BaseSection, FigureInfo, ALL_EXCEPTIONS, DATA_EXCEPTIONS

_METRIC_DISPLAY = {
    "delta_c_index": "Delta C-index",
    "nri": "NRI",
    "idi": "IDI",
}

_METRIC_ORDER = ["delta_c_index", "nri", "idi"]


def _format_cell(row: "pd.Series", has_ci: bool) -> str:
    """Format a single metric cell as ``value [CI]`` or ``value``."""
    if pd.isna(row["point"]):
        return "\u2014"
    val = f"{row['point']:.3f}"
    if has_ci and pd.notna(row["ci_lower"]) and pd.notna(row["ci_upper"]):
        val += f" [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
    return val


def _pivot_incremental(
    df: "pd.DataFrame",
    subpopulation_col: str | None = None,
) -> list[dict[str, str]]:
    """Pivot incremental rows into one row per comparison.

    Returns a list of dicts suitable for ``pd.DataFrame(table_data)``.
    Each dict has ``"Model vs Reference"`` (and optionally ``"Subpopulation"``)
    plus one key per metric present (e.g. ``"Delta C-index"``).
    """
    if df.empty:
        return []

    has_ci = df["ci_lower"].notna().any()
    has_pval = "p_value" in df.columns and df["p_value"].notna().any()
    group_cols = ["model", "reference"]
    if subpopulation_col:
        group_cols = [subpopulation_col] + group_cols

    table_data: list[dict[str, str]] = []
    for keys, grp in df.groupby(group_cols, sort=False):
        # keys is always a tuple because group_cols has 2+ elements
        row_dict: dict[str, str] = {}
        if subpopulation_col:
            subpopulation, model, ref = keys
            row_dict["Subpopulation"] = str(subpopulation)
        else:
            model, ref = keys
        row_dict["Model vs Reference"] = f"{model} vs {ref}"

        metric_lookup = {r["metric"]: r for _, r in grp.iterrows()}
        for mk in _METRIC_ORDER:
            col_name = _METRIC_DISPLAY.get(mk, mk)
            row_dict[col_name] = (
                _format_cell(metric_lookup[mk], has_ci)
                if mk in metric_lookup
                else "\u2014"
            )
        if has_pval:
            for mk in _METRIC_ORDER:
                col_name = _METRIC_DISPLAY.get(mk, mk) + " p"
                if mk in metric_lookup:
                    p = metric_lookup[mk].get("p_value", np.nan)
                    row_dict[col_name] = format_pvalue(p, stars=True)
                else:
                    row_dict[col_name] = "\u2014"
        table_data.append(row_dict)

    return table_data


class BenchmarkSection(BaseSection):
    """Report section comparing model predictions against reference scores.

    Shows:
    - Performance comparison (forest plot + table)
    - Time-dependent discrimination (AUC(t) overlay)
    - Incremental value (delta-C, NRI, IDI)
    - Univariate Cox regression (HR + CI)
    - Model-reference correlation

    This section only appears if ``benchmark()`` has been called.
    """

    name = "benchmarking"
    title = "Benchmarking"

    def is_available(self) -> bool:
        return self.learner.benchmark_result is not None

    def generate(self) -> dict[str, Any]:
        md_lines = []
        figures = []
        figure_info = []

        br = self.learner.benchmark_result
        if br is None:
            return {"markdown": "", "figures": [], "figure_info": []}

        md_lines.append(
            "This section compares trained model predictions against "
            "reference scores using the same evaluation metrics. "
            "Positive incremental metrics (Delta C-index, NRI, IDI) indicate "
            "the model provides predictive value beyond the reference alone."
        )
        md_lines.append("")
        md_lines.append(
            f"**Models:** {', '.join(br.model_names)}  \n"
            f"**References:** {', '.join(br.reference_names)}"
        )
        md_lines.append("")

        dataset = getattr(br, 'dataset', 'test')
        if dataset != "test":
            dataset_label = {
                "train": "training", "full": "full",
            }.get(dataset, dataset)
            md_lines.append(
                f"**Note:** Models evaluated on held-out test data; "
                f"references evaluated on **{dataset_label}** data for greater "
                f"statistical power. Incremental metrics and correlations "
                f"use test data for both."
            )
            md_lines.append("")

        # Sample-size note when references were evaluated on filtered subsets
        if br.sample_sizes:
            model_n = max(
                (br.sample_sizes[m] for m in br.model_names
                 if m in br.sample_sizes),
                default=0,
            )
            ref_sizes = {
                r: br.sample_sizes[r]
                for r in br.reference_names
                if r in br.sample_sizes
            }
            # Determine the expected base for NaN-filtered comparison
            base_n = (
                model_n if dataset == "test"
                else max(ref_sizes.values(), default=0)
            )
            filtered = {
                r: n for r, n in ref_sizes.items() if n < base_n
            }
            if filtered:
                lines = [
                    f"- **{r}**: N={n} ({base_n - n} excluded)"
                    for r, n in filtered.items()
                ]
                md_lines.append("> **Note — Different sample sizes**  ")
                md_lines.append(
                    "> References with missing values are evaluated on "
                    "their valid subset:"
                )
                md_lines.extend(f"> {line}" for line in lines)
                md_lines.append("")

        # 1. Performance comparison table
        self._add_performance_table(md_lines, br)

        # 2. Performance comparison figure
        self._add_performance_figure(md_lines, figures, figure_info, br)

        # 3. AUC(t) overlay
        self._add_auct_overlay(md_lines, figures, figure_info, br)

        # 3b. ROC curves
        self._add_roc_overlay(md_lines, figures, figure_info, br)

        # 4. Incremental value
        self._add_incremental_section(md_lines, figures, figure_info, br)

        # 5. Univariate Cox
        self._add_univariate_cox(md_lines, br)

        # 6. Correlation
        self._add_correlation(md_lines, br)

        # 6b. Correlation scatter (per-sample view)
        self._add_correlation_scatter(md_lines, figures, figure_info, br)

        # 7. Per-subpopulation incremental
        self._add_subpopulation_incremental(md_lines, br)

        return {
            "markdown": "\n".join(md_lines),
            "figures": figures,
            "figure_info": figure_info,
        }

    def _add_performance_table(self, md_lines, br):
        """Add concordance comparison table."""
        md_lines.append("### Performance Comparison")
        md_lines.append("")

        try:
            conc = br.evaluation.concordance
            if conc is not None and not conc.empty:
                from mlsurv.constants import (
                    ALL_METRICS, LOWER_IS_BETTER,
                    get_metric_short_name, get_concordance_column,
                )
                table_data = []
                for _, row in conc.iterrows():
                    row_data = {"Entity": row["model_type"]}
                    for mk in ALL_METRICS:
                        col = get_concordance_column(mk)
                        if col in row.index and not pd.isna(row[col]):
                            row_data[get_metric_short_name(mk)] = f"{row[col]:.3f}"
                    table_data.append(row_data)
                if table_data:
                    df = pd.DataFrame(table_data)
                    # Sort by scoring metric (best first)
                    scoring = self.learner.scoring
                    sort_col = get_metric_short_name(scoring)
                    if sort_col in df.columns:
                        df['_sort'] = df[sort_col].str.extract(r'^(-?[\d.]+)')[0].astype(float)
                        df = df.sort_values('_sort', ascending=scoring in LOWER_IS_BETTER)
                        df = df.drop(columns=['_sort']).reset_index(drop=True)
                    # Bold best value in each metric column
                    df = self._bold_best_values(df)
                    md_lines.append(
                        "Each row shows one model or reference with its "
                        "test-set performance metrics. Models with higher "
                        "concordance than a reference suggest they capture "
                        "prognostic information beyond what the reference "
                        "alone provides. However, formal incremental tests "
                        "(below) are needed to confirm statistical "
                        "significance."
                    )
                    md_lines.append("")
                    md_lines.append(self._format_table(df))
                    md_lines.append("")
                    self.data["concordance"] = df
        except DATA_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "benchmarking: performance table",
                warn_user=True,
                user_message=f"Could not generate performance table: {exc}",
            )
            md_lines.append(f"*[Performance table unavailable: {exc}]*")
            md_lines.append("")

    def _add_performance_figure(self, md_lines, figures, figure_info, br):
        """Add forest plot of performance metrics."""
        try:
            fig = self.learner.plot_metric(
                references="all", metric=self.learner.scoring,
            )
            filename = "benchmark_comparison.png"
            figures.append((filename, fig))
            md_lines.append(f"![Benchmark Comparison](images/{filename})")
            md_lines.append("")
            figure_info.append(FigureInfo(
                figure_id="benchmark_comparison",
                title="Performance Comparison",
                data_description="Concordance metrics for models and references",
                data_key="concordance",
                learner_code="fig = learner.plot_metric(references='all')",
            ))
        except (ImportError,) + ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e, "benchmarking: performance figure", warn_user=True,
                user_message=f"Could not generate benchmark comparison figure: {e}",
            )
            md_lines.append(f"*[Benchmark comparison figure unavailable: {e}]*")
            md_lines.append("")

    def _add_auct_overlay(self, md_lines, figures, figure_info, br):
        """Add AUC(t) curves comparing models and references."""
        auc_df = br.evaluation.auc_t
        if auc_df is None or auc_df.empty:
            return

        md_lines.append("### Time-Dependent Discrimination")
        md_lines.append("")

        try:
            fig = self.learner.plot_metric_time(references="all", time_unit=self.time_unit, metric="auc")
            filename = "benchmark_auct.png"
            figures.append((filename, fig))
            md_lines.append(f"![AUC(t) Comparison](images/{filename})")
            md_lines.append("")
            self.data["auct"] = auc_df
            figure_info.append(FigureInfo(
                figure_id="benchmark_auct",
                title="Time-Dependent AUC Comparison",
                data_description="AUC(t) values at each evaluation time",
                data_key="auct",
                learner_code="fig = learner.plot_metric_time(references='all')",
            ))
        except (ImportError,) + ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e, "benchmarking: AUC(t) figure", warn_user=True,
                user_message=f"Could not generate AUC(t) comparison figure: {e}",
            )
            md_lines.append(f"*[AUC(t) figure unavailable: {e}]*")
            md_lines.append("")

    def _add_roc_overlay(self, md_lines, figures, figure_info, br):
        """Add ROC curves comparing models and references with time dropdown."""
        auc_df = br.evaluation.auc_t
        if auc_df is None or auc_df.empty:
            return
        test_set = getattr(br.evaluation, "test_set", None)
        if test_set is None or len(test_set) == 0:
            return

        try:
            from mlsurv.viz._metrics_roc import (
                assemble_roc_curves_with_references,
                bootstrap_roc_curves,
                plot_roc_interactive,
            )
            from mlsurv.reports.sections.evaluation._roc import (
                _extract_risk_scores,
            )
            from mlsurv.reports._helpers import extract_eval_times_from_auc_t

            eval_times = extract_eval_times_from_auc_t(auc_df)
            if not eval_times:
                return

            risk_scores = _extract_risk_scores(test_set)
            if not risk_scores:
                return

            y_train = self.learner.y_train
            y_test = self.learner.y_test
            if y_train is None or y_test is None:
                return

            # All preconditions passed — emit heading
            md_lines.append("### Time-Dependent ROC Curves")
            md_lines.append("")

            # Partition risk_scores into models (full-length y_test) and
            # NaN-bearing references (per-reference NaN-masked y_test), then
            # assemble per-horizon ROC via the shared helper. References like
            # PD-L1 are evaluated by `benchmark()` on their valid subset only,
            # so their rows in `test_set` have length = the non-NaN count;
            # the helper masks each reference's outcomes accordingly.
            res = assemble_roc_curves_with_references(
                y_train, y_test, risk_scores, eval_times,
                reference_names=getattr(br, "reference_names", []) or [],
                reference_arrays=getattr(br, "reference_arrays", None) or {},
                catch=ALL_EXCEPTIONS,
                on_error=lambda exc, kind, name, t: (
                    self._log_recoverable_exception(
                        exc,
                        f"benchmarking: ROC at t={t}"
                        + (f" for reference '{name}'" if name else ""),
                    )
                ),
            )
            roc_data = res.roc_data
            all_roc_dfs = res.all_roc_dfs
            dropped_refs = set(res.dropped_by_ref)

            # Warn once per reference that's structurally missing from
            # `reference_arrays` (bookkeeping bug, not a recoverable data
            # condition). Done after assembly, not per-horizon, to avoid
            # N-fold spam.
            if res.missing_ref_arrays:
                import warnings
                from mlsurv._warnings import InternalWarning
                warnings.warn(
                    f"BenchmarkResult.reference_arrays is missing entries "
                    f"for {sorted(res.missing_ref_arrays)} — ROC for these "
                    f"references cannot be NaN-masked. This indicates a "
                    f"benchmark bookkeeping issue.",
                    InternalWarning, stacklevel=2,
                )

            # Surface dropped reference curves to the user (silent-data-loss
            # guard) — references that never produced an ROC curve at any
            # horizon are missing from the plot entirely.
            never_succeeded = {
                r for r in dropped_refs
                if not any(
                    not df.empty and (df["model"] == r).any()
                    for df in all_roc_dfs
                )
            }
            if never_succeeded:
                md_lines.append(
                    f"*Note: ROC curves could not be computed for "
                    f"reference(s) {sorted(never_succeeded)} at any "
                    f"evaluation horizon (insufficient cases/controls in "
                    f"the valid subset).*"
                )
                md_lines.append("")

            if not roc_data:
                md_lines.append("*[ROC curves unavailable: computation failed]*")
                md_lines.append("")
                return

            # Compute bootstrap CI bands when predictions are available
            ci_data = None
            n_boot_iters = 0
            ci_attempted = False
            boot = getattr(br.evaluation, "bootstrap", None)
            if boot is not None:
                predictions = getattr(boot, "predictions", None)
                if predictions is not None and len(predictions) > 0:
                    ci_attempted = True
                    try:
                        X_test_index = self.learner.X_test.index
                        ci_data = bootstrap_roc_curves(
                            y_train, y_test, predictions,
                            X_test_index, eval_times,
                        )
                        n_boot_iters = int(
                            predictions["iteration"].nunique()
                        )
                    except ALL_EXCEPTIONS as exc:
                        self._log_recoverable_exception(
                            exc,
                            "benchmarking: ROC bootstrap CI bands",
                            warn_user=True,
                            user_message=(
                                f"Could not compute ROC confidence intervals: "
                                f"{exc}. Falling back to plain ROC curves."
                            ),
                        )

            if ci_data:
                md_lines.append(
                    "Shaded bands show 95\u2009% pointwise bootstrap "
                    "confidence intervals."
                )
                if n_boot_iters < 20:
                    md_lines.append(
                        f" *Note: only {n_boot_iters} bootstrap iterations "
                        f"were used \u2014 CI bands may not fully enclose "
                        f"the ROC curves. Increase `n_bootstrap` for "
                        f"tighter bands.*"
                    )
                md_lines.append("")
            elif ci_attempted:
                md_lines.append(
                    "*[Bootstrap CI bands could not be computed because "
                    "per-iteration predictions were not stored on the "
                    "bootstrap result. Re-run with "
                    "`learner.bootstrap(include_predictions=True)` to "
                    "enable ROC CI bands; showing plain ROC curves.]*"
                )
                md_lines.append("")

            # Build palette with reference colors for consistency
            from mlsurv.viz._palettes import build_benchmark_palette
            palette = build_benchmark_palette(br.model_names, br.reference_names)

            fig = plot_roc_interactive(
                roc_data, title="ROC Comparison", palette=palette,
                ci_data=ci_data,
            )
            filename = "benchmark_roc.png"
            figures.append((filename, fig))
            md_lines.append("<!-- MLSURV_RAW_HTML -->")
            md_lines.append(
                f'<div class="figure-placeholder" data-figure="{filename}"></div>'
            )
            md_lines.append("<!-- /MLSURV_RAW_HTML -->")
            md_lines.append("")

            self.data["roc"] = pd.concat(all_roc_dfs, ignore_index=True)
            figure_info.append(FigureInfo(
                figure_id="benchmark_roc",
                title="Time-Dependent ROC Comparison",
                data_description="ROC curve data for models and references",
                data_key="roc",
                learner_code="fig = learner.plot_roc(references='all')",
            ))
        except (ImportError,) + ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e, "benchmarking: ROC figure", warn_user=True,
                user_message=f"Could not generate ROC comparison figure: {e}",
            )
            md_lines.append(f"*[ROC figure unavailable: {e}]*")
            md_lines.append("")

    def _add_incremental_section(self, md_lines, figures, figure_info, br):
        """Add incremental metrics (delta-C, NRI, IDI)."""
        inc = br.incremental
        if inc.empty:
            return

        md_lines.append("### Incremental Value")
        md_lines.append("")
        md_lines.append(
            "Incremental metrics quantify how much additional predictive "
            "value a model provides beyond a reference alone."
        )
        md_lines.append("")

        # Interpretation before table/figure
        try:
            self._add_incremental_interpretation(md_lines, inc)
        except DATA_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "benchmarking: incremental interpretation",
                warn_user=True,
                user_message=f"Could not generate incremental interpretation: {exc}",
            )

        # Pivoted table: one row per (model, reference), one column per metric
        try:
            table_data = _pivot_incremental(inc)
            if table_data:
                md_lines.append(self._format_table(pd.DataFrame(table_data)))
                md_lines.append("")
                self.data["incremental"] = inc.copy()
        except DATA_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "benchmarking: incremental table",
                warn_user=True,
                user_message=f"Could not generate incremental value table: {exc}",
            )
            md_lines.append(f"*[Incremental value table unavailable: {exc}]*")
            md_lines.append("")

        # Forest plot
        try:
            fig = self.learner.plot_benchmark_incremental()
            filename = "benchmark_incremental.png"
            figures.append((filename, fig))
            md_lines.append(f"![Incremental Value](images/{filename})")
            md_lines.append("")
            figure_info.append(FigureInfo(
                figure_id="benchmark_incremental",
                title="Incremental Value Forest Plot",
                data_description="Incremental metrics with bootstrap CIs",
                data_key="incremental",
                learner_code="fig = learner.plot_benchmark_incremental()",
            ))
        except (ImportError,) + ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e, "benchmarking: incremental figure", warn_user=True,
                user_message=f"Could not generate incremental value figure: {e}",
            )
            md_lines.append(f"*[Incremental value figure unavailable: {e}]*")
            md_lines.append("")

    def _add_incremental_interpretation(self, md_lines, inc):
        """Add data-driven interpretation paragraph for incremental metrics."""
        has_ci = inc["ci_lower"].notna().any()

        md_lines.append(
            "**Interpretation:** Delta C-index measures the absolute "
            "concordance improvement when combining the model with the "
            "reference. NRI (Net Reclassification Improvement) quantifies "
            "whether patients are more often correctly reclassified, and IDI "
            "(Integrated Discrimination Improvement) measures the average "
            "improvement in predicted probability separation between events "
            "and non-events."
        )
        md_lines.append("")

        if has_ci:
            # Check whether any CI excludes zero
            has_sig = False
            for _, row in inc.iterrows():
                ci_lo, ci_hi = row["ci_lower"], row["ci_upper"]
                if pd.notna(ci_lo) and pd.notna(ci_hi) and (ci_lo > 0 or ci_hi < 0):
                    has_sig = True
                    break
            if has_sig:
                md_lines.append(
                    "Bootstrap confidence intervals that exclude zero "
                    "indicate statistically significant improvement."
                )
            else:
                md_lines.append(
                    "No metric showed statistically significant improvement "
                    "(all bootstrap confidence intervals include zero)."
                )
            md_lines.append("")

    def _add_univariate_cox(self, md_lines, br):
        """Add univariate Cox regression results."""
        try:
            cox = br.univariate_cox
            if cox.empty:
                return

            md_lines.append("### Univariate Cox Regression")
            md_lines.append("")

            md_lines.append(
                "Each row fits a univariate Cox proportional hazards model "
                "using one reference score as the sole predictor. "
                "HR < 1 means higher reference score is associated with "
                "lower hazard (protective), while HR > 1 indicates higher "
                "score is associated with higher hazard."
            )
            md_lines.append("")

            table_data = []
            for _, row in cox.iterrows():
                hr = row["hazard_ratio"]
                if pd.isna(hr):
                    hr_ci = "N/A"
                    p_str = "N/A"
                else:
                    lo, hi = row["hr_lower_95"], row["hr_upper_95"]
                    if pd.notna(lo) and pd.notna(hi):
                        hr_ci = f"{hr:.3f} [{lo:.3f}, {hi:.3f}]"
                    else:
                        hr_ci = f"{hr:.3f}"
                    p_str = format_pvalue(row["p_value"], stars=True)
                table_data.append({
                    "Reference": row["reference"],
                    "HR [95% CI]": hr_ci,
                    "p-value": p_str,
                })

            md_lines.append(self._format_table(pd.DataFrame(table_data)))
            md_lines.append("")
            self.data["univariate_cox"] = cox.copy()

            n_refs = len(cox)
            n_sig = int(
                (cox["p_value"].notna() & (cox["p_value"] < P_THRESHOLD_SIGNIFICANT)).sum()
            )
            if n_sig == n_refs:
                md_lines.append(
                    f"All {n_refs} references show statistically significant "
                    f"prognostic value (p < {P_THRESHOLD_SIGNIFICANT})."
                )
            elif n_sig > 0:
                md_lines.append(
                    f"Of {n_refs} references, {n_sig} show statistically "
                    f"significant prognostic value (p < {P_THRESHOLD_SIGNIFICANT})."
                )
            else:
                md_lines.append(
                    f"No reference reached statistical significance at the "
                    f"{P_THRESHOLD_SIGNIFICANT} level."
                )
            md_lines.append("")
        except DATA_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "benchmarking: univariate Cox",
                warn_user=True,
                user_message=f"Could not generate univariate Cox table: {exc}",
            )
            md_lines.append(f"*[Univariate Cox unavailable: {exc}]*")
            md_lines.append("")

    def _add_correlation(self, md_lines, br):
        """Add model-reference correlation matrices (Spearman and Pearson).

        Two complementary correlation statistics are rendered, each with its
        conventional symbol: Spearman as ``\u03c1`` (rho, rank correlation)
        and Pearson as lowercase ``r`` (linear correlation). Capital ``R``
        is deliberately avoided because it is reserved for the multiple
        correlation coefficient in regression contexts.
        """
        try:
            corr = br.correlation
            if corr.empty:
                return

            md_lines.append("### Model-Reference Correlation")
            md_lines.append("")

            md_lines.append(
                "Two complementary correlation statistics are reported: "
                "Spearman rank correlation (\u03c1) captures the strength "
                "of any monotonic relationship by correlating ranks, and "
                "Pearson correlation (r) captures the strength of the "
                "linear relationship on raw values. Reporting both lets "
                "you distinguish monotonic-but-nonlinear agreement "
                "(Spearman high, Pearson lower) from linear agreement "
                "(both high). Strong negative values occur when one score "
                "uses higher-is-worse and the other higher-is-better, "
                "meaning both rank patients similarly."
            )
            md_lines.append("")

            md_lines.append("#### Spearman Rank Correlation (\u03c1)")
            md_lines.append("")
            md_lines.append(
                "Rank-based correlation; robust to outliers and "
                "insensitive to monotonic transformations. Use this as "
                "the primary comparison when two scores may track each "
                "other through different functional forms."
            )
            md_lines.append("")
            md_lines.extend(
                self._build_correlation_matrix(corr, "spearman")
            )
            md_lines.append("")

            md_lines.append("#### Pearson Correlation (r)")
            md_lines.append("")
            md_lines.append(
                "Linear correlation on raw values; sensitive to outliers "
                "and assumes roughly linear co-variation. Compare against "
                "Spearman to flag nonlinear or outlier-driven disagreement: "
                "|\u03c1| noticeably larger than |r| suggests either a "
                "nonlinear monotonic transform between the two scores or "
                "that outliers are distorting the linear fit."
            )
            md_lines.append("")
            md_lines.extend(
                self._build_correlation_matrix(corr, "pearson")
            )
            md_lines.append("")

            self.data["correlation"] = corr.copy()
        except DATA_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "benchmarking: correlation table",
                warn_user=True,
                user_message=f"Could not generate correlation table: {exc}",
            )
            md_lines.append(f"*[Correlation table unavailable: {exc}]*")
            md_lines.append("")

    def _add_correlation_scatter(self, md_lines, figures, figure_info, br):
        """Add per-sample scatter of model risk score vs. each reference.

        Complements the Spearman/Pearson matrices above by showing the
        actual shape of each model-vs-reference relationship.
        """
        if not getattr(br, "model_risk_scores", None) or not getattr(
            br, "reference_arrays", None,
        ):
            return
        try:
            fig = self.learner.plot_benchmark_scatter()
            filename = "benchmark_correlation_scatter.png"
            figures.append((filename, fig))
            md_lines.append("#### Scatter: Model vs. Reference")
            md_lines.append("")
            md_lines.append(
                "Per-sample scatter of each model's risk score against "
                "each reference score. Use the dropdown to switch models. "
                "The subplot annotations repeat the matrix ρ/r values; "
                "the markers reveal whether a high correlation reflects a "
                "tight monotonic fit or is driven by a few leverage points "
                "or a nonlinear pattern."
            )
            md_lines.append("")
            md_lines.append(f"![Model vs Reference Scatter](images/{filename})")
            md_lines.append("")

            rows = []
            for model, m_arr in br.model_risk_scores.items():
                for ref, r_arr in br.reference_arrays.items():
                    rows.extend(
                        {
                            "model": model,
                            "reference": ref,
                            "model_score": float(m),
                            "reference_score": float(r),
                        }
                        for m, r in zip(m_arr, r_arr)
                    )
            if rows:
                self.data["correlation_scatter"] = pd.DataFrame(rows)

            figure_info.append(FigureInfo(
                figure_id="benchmark_correlation_scatter",
                title="Model vs. Reference Scatter",
                data_description=(
                    "Per-sample model risk score vs. reference score, "
                    "by reference and model"
                ),
                data_key="correlation_scatter",
                learner_code="fig = learner.plot_benchmark_scatter()",
            ))
        except (ImportError,) + ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e, "benchmarking: correlation scatter figure",
                warn_user=True,
                user_message=(
                    f"Could not generate correlation scatter figure: {e}"
                ),
            )
            md_lines.append(
                f"*[Correlation scatter figure unavailable: {e}]*"
            )
            md_lines.append("")

    @staticmethod
    def _significance_stars(p: float) -> str:
        """Return significance stars for a p-value."""
        if pd.isna(p) or not np.isfinite(p) or p < 0:
            return ""
        for threshold, stars in P_STARS:
            if p < threshold:
                return stars
        return ""

    def _build_correlation_matrix(
        self, corr: "pd.DataFrame", method: str,
    ) -> list[str]:
        """Build an HTML correlation matrix (models x references).

        Parameters
        ----------
        corr : pd.DataFrame
            Correlation rows with ``model``, ``reference``, ``spearman_rho``,
            ``spearman_p``, ``pearson_r``, ``pearson_p`` columns.
        method : {"spearman", "pearson"}
            Which correlation statistic to render. Spearman cells are
            labelled with ``\u03c1`` (rho, rank correlation); Pearson cells
            are labelled with lowercase ``r`` (linear correlation). Capital
            ``R`` is deliberately avoided because it is reserved for the
            multiple correlation coefficient in regression contexts.

        Returns
        -------
        list[str]
            Lines wrapped in ``<!-- MLSURV_RAW_HTML -->`` markers.
        """
        r_col = "spearman_rho" if method == "spearman" else "pearson_r"
        p_col = f"{method}_p"
        symbol = "\u03c1" if method == "spearman" else "r"

        models = corr["model"].unique()
        references = corr["reference"].unique()

        # Build lookup dicts (single pass over rows)
        r_lookup: dict[tuple[str, str], float] = {}
        p_lookup: dict[tuple[str, str], float] = {}
        for _, row in corr.iterrows():
            key = (row["model"], row["reference"])
            r_lookup[key], p_lookup[key] = row[r_col], row[p_col]

        lines = ["<!-- MLSURV_RAW_HTML -->"]
        lines.append('<div class="feature-table-scroll">')
        lines.append('<table class="compact-incremental">')

        # Header row
        lines.append("<thead><tr><th></th>")
        for ref in references:
            lines.append(f"<th>{html_module.escape(str(ref))}</th>")
        lines.append("</tr></thead>")

        # Body rows
        lines.append("<tbody>")
        for model in models:
            lines.append(f"<tr><td><strong>{html_module.escape(str(model))}</strong></td>")
            for ref in references:
                r_val = r_lookup.get((model, ref), np.nan)
                p_val = p_lookup.get((model, ref), np.nan)
                if pd.isna(r_val):
                    cell = html_module.escape("\u2014")
                else:
                    p_str = format_pvalue(p_val, stars=True)
                    cell = (
                        f"{symbol}={html_module.escape(f'{r_val:.3f}')}<br>"
                        f"<small>p={html_module.escape(p_str)}</small>"
                    )
                lines.append(f"<td>{cell}</td>")
            lines.append("</tr>")
        lines.append("</tbody></table></div>")
        lines.append("<!-- /MLSURV_RAW_HTML -->")
        return lines

    def _add_subpopulation_incremental(self, md_lines, br):
        """Add per-subpopulation incremental metrics table with optional toggle."""
        sg_inc = br.subpopulation_incremental
        if sg_inc is None or sg_inc.empty:
            return

        md_lines.append("### Per-Subpopulation Incremental Value")
        md_lines.append("")
        md_lines.append(
            "Incremental metrics computed within each patient subpopulation. "
            "This reveals whether the model adds value over the reference "
            "uniformly or primarily in specific subpopulations."
        )
        md_lines.append("")

        # Interpretation before table
        subpopulations = sg_inc["subpopulation"].unique()
        n_subpopulations = len(subpopulations)
        md_lines.append(
            "Consistent positive metrics across subpopulations indicate the "
            "model adds value for all patient populations. Subpopulations "
            "where metrics are near zero or negative may indicate "
            "populations where the reference alone is sufficient. "
            f"Results are shown for {n_subpopulations} subpopulations."
        )
        md_lines.append("")

        try:
            table_data = _pivot_incremental(sg_inc, subpopulation_col="subpopulation")
            if table_data:
                n_visible = BENCHMARK_SUBPOPULATION_VISIBLE_ROWS
                if len(table_data) > n_visible:
                    md_lines.extend(
                        self._render_subpopulation_table_html(table_data, n_visible)
                    )
                else:
                    md_lines.append(
                        self._format_table(pd.DataFrame(table_data))
                    )
                md_lines.append("")
                self.data["subpopulation_incremental"] = sg_inc.copy()
        except DATA_EXCEPTIONS as exc:
            self._log_recoverable_exception(
                exc, "benchmarking: subpopulation incremental",
                warn_user=True,
                user_message=f"Could not generate subpopulation incremental table: {exc}",
            )
            md_lines.append(f"*[Subpopulation incremental table unavailable: {exc}]*")
            md_lines.append("")

    @staticmethod
    def _render_subpopulation_table_html(
        table_data: list[dict[str, str]], n_visible: int,
    ) -> list[str]:
        """Render subpopulation incremental table as HTML with row toggle.

        Rows beyond *n_visible* are hidden with ``ft-overflow-row`` and a
        toggle button is appended, reusing the feature-table expand
        infrastructure.
        """
        df = pd.DataFrame(table_data)
        columns = list(df.columns)
        n_total = len(df)
        n_remaining = n_total - n_visible

        lines = ["<!-- MLSURV_RAW_HTML -->"]
        lines.append('<div class="feature-table-container">')
        lines.append('<div class="feature-table-scroll">')
        lines.append('<table class="compact-incremental">')

        # Header
        lines.append("<thead><tr>")
        for col in columns:
            lines.append(f"<th>{html_module.escape(col)}</th>")
        lines.append("</tr></thead>")

        # Body
        lines.append("<tbody>")
        for row_idx, row in df.iterrows():
            cls = ' class="ft-overflow-row"' if row_idx >= n_visible else ""
            lines.append(f"<tr{cls}>")
            for col in columns:
                lines.append(
                    f"<td>{html_module.escape(str(row.get(col, '')))}</td>"
                )
            lines.append("</tr>")
        lines.append("</tbody></table></div>")

        # Expand button
        collapsed = f"Show remaining {n_remaining} rows"
        expanded = f"Show first {n_visible} rows"
        lines.append('<div class="feature-table-expand">')
        lines.append(
            f'<button class="ft-expand-btn" '
            f'data-n-visible="{n_visible}" '
            f'data-n-total="{n_total}" '
            f'data-collapsed-text="{html_module.escape(collapsed)}" '
            f'data-expanded-text="{html_module.escape(expanded)}">'
            f'{html_module.escape(collapsed)}</button>'
        )
        lines.append("</div>")
        lines.append("</div>")  # feature-table-container
        lines.append("<!-- /MLSURV_RAW_HTML -->")
        return lines
