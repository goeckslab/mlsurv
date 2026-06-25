"""Paired Model Comparison subsection for evaluation reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsurv.reports._constants import (
    P_STARS,
    P_THRESHOLD_EXCELLENT,
    P_THRESHOLD_GOOD,
    P_THRESHOLD_SIGNIFICANT,
    format_pvalue,
)
from mlsurv.viz._palettes import get_model_display_name

from mlsurv.reports.sections.evaluation._base import _BaseSubsection


def _significance_marker(p: float) -> str:
    """Return significance marker for report table."""
    if pd.isna(p):
        return ""
    for threshold, stars in P_STARS:
        if p < threshold:
            return stars.replace("*", "\\*")
    return "ns"


def _build_comparison_rows(sources, format_row_fn):
    """Collect formatted rows from comparison DataFrames."""
    rows = []
    for df, label in sources:
        if df is not None:
            for _, row in df.iterrows():
                rows.append(format_row_fn(row, label))
    return rows


def _build_interpretation(md_lines, comparison_rows):
    """Append interpretation paragraph based on significant pairs."""
    sig_pairs = [r for r in comparison_rows if r['sig'] not in ("", "ns")]
    if sig_pairs:
        parts = [f"{r['pair']} (p = {r['p_value']})" for r in sig_pairs]
        md_lines.append(
            f"**Interpretation:** Statistically significant differences: "
            f"{'; '.join(parts)}."
        )
    else:
        md_lines.append(
            "**Interpretation:** No statistically significant differences "
            f"were found between any model pairs at the {P_THRESHOLD_SIGNIFICANT} level."
        )
    md_lines.append("")


class _PairedComparisonSubsection(_BaseSubsection):
    """Paired model comparison table for the evaluation report."""

    def is_available(self) -> bool:
        """True when 2+ models and any comparison data is available."""
        models = self._s._get_all_models()
        if len(models) < 2:
            return False
        return (
            bool(self._s.bootstrap_results)
            or self._get_kang_data() is not None
        )

    def add(self, md_lines: list, figures: list, figure_info: list) -> None:
        """Add paired comparison table to report markdown."""
        md_lines.append(
            "Pairwise statistical comparisons test whether observed "
            "performance differences are statistically significant. "
            "Methods include bootstrap permutation tests on discrimination "
            "metrics and the Kang z-test for comparing concordance indices."
        )
        md_lines.append("")

        boot_df = self._compute_bootstrap_comparison() if self._s.bootstrap_results else None
        kang_df = self._get_kang_data()

        comparison_rows = _build_comparison_rows(
            [(boot_df, "Bootstrap"), (kang_df, "Kang z-test")],
            self._format_row,
        )

        if not comparison_rows:
            md_lines.append("*No comparison data available.*")
            md_lines.append("")
            return

        # Build markdown table
        md_lines.append("| Pair | Method | Delta [95% CI] | p-value | Sig. |")
        md_lines.append("|------|--------|----------------|---------|------|")
        for row_dict in comparison_rows:
            md_lines.append(
                f"| {row_dict['pair']} "
                f"| {row_dict['method']} "
                f"| {row_dict['delta_ci']} "
                f"| {row_dict['p_value']} "
                f"| {row_dict['sig']} |"
            )
        md_lines.append("")
        md_lines.append(
            f"Sig.: \\* p < {P_THRESHOLD_SIGNIFICANT}, \\*\\* p < {P_THRESHOLD_GOOD}, \\*\\*\\* p < {P_THRESHOLD_EXCELLENT}"
        )
        md_lines.append("")

        _build_interpretation(md_lines, comparison_rows)

        # Store data for programmatic access
        all_dfs = [df for df in (boot_df, kang_df) if df is not None]
        if all_dfs:
            self._s.data['paired_comparison'] = pd.concat(all_dfs, ignore_index=True)

    def _format_row(self, row: pd.Series, method: str) -> dict:
        """Format a single comparison row for the markdown table."""
        model_a = get_model_display_name(row['model_a'])
        model_b = get_model_display_name(row['model_b'])
        pair = f"{model_a} vs {model_b}"

        delta = row['delta']
        ci_lower = row.get('ci_lower', np.nan)
        ci_upper = row.get('ci_upper', np.nan)

        if pd.notna(delta):
            sign = "+" if delta >= 0 else ""
            delta_ci = f"{sign}{delta:.3f}"
            if pd.notna(ci_lower) and pd.notna(ci_upper):
                delta_ci += f" [{ci_lower:.3f}, {ci_upper:.3f}]"
        else:
            delta_ci = "N/A"

        # Use p_adjusted if available, else p_value
        p_val = row.get('p_adjusted', row.get('p_value', np.nan))
        p_str = format_pvalue(p_val)
        sig = _significance_marker(p_val)

        return {
            'pair': pair,
            'method': method,
            'delta_ci': delta_ci,
            'p_value': p_str,
            'sig': sig,
        }

    def _compute_bootstrap_comparison(self):
        """Merge per-model bootstrap results and run paired comparison."""
        if not self._s.bootstrap_results:
            return None

        try:
            from mlsurv.results.containers import BootstrapResult, _get_model_names
            from mlsurv.evaluation._paired_comparison import compare_bootstrap

            # Merge per-model bootstrap into a single multi-model result
            all_summaries = []
            all_iterations = []
            lower_is_better = None
            ci_level = None
            for boot in self._s.bootstrap_results.values():
                if boot.summary is not None:
                    all_summaries.append(boot.summary)
                if boot.iterations is not None:
                    all_iterations.append(boot.iterations)
                if lower_is_better is None:
                    lower_is_better = boot.lower_is_better
                if ci_level is None:
                    ci_level = boot.ci_level

            if not all_summaries or not all_iterations:
                return None

            merged = BootstrapResult(
                summary=pd.concat(all_summaries, ignore_index=True),
                iterations=pd.concat(all_iterations, ignore_index=True),
                lower_is_better=lower_is_better,
                ci_level=ci_level if ci_level is not None else 0.95,
            )

            if len(_get_model_names(merged.iterations)) < 2:
                return None

            return compare_bootstrap(merged, metric=self._s.metric)
        except (ValueError, KeyError) as e:
            self._log_error(e, "evaluation: bootstrap comparison computation")
            return None

    def _get_kang_data(self):
        """Get the Kang comparison DataFrame if non-empty, otherwise None."""
        df = self._s.learner.kang_comparison_result
        if df is not None and not df.empty:
            return df
        return None

