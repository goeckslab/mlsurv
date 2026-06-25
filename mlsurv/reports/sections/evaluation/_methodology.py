"""Methodology subsection -- bootstrap configuration and interpretation guidance."""

from __future__ import annotations

import html as html_module

from mlsurv.reports.sections.evaluation._base import _BaseSubsection


class _MethodologySubsection(_BaseSubsection):
    """Bootstrap methodology details and general metric interpretation."""

    def is_available(self) -> bool:
        return True

    def add(self, md_lines: list, figures: list, figure_info: list) -> None:
        self._add_bootstrap_methodology_details(md_lines)

        md_lines.append(
            "**Interpretation:** The test set was held out during training and "
            "tuning, so these metrics reflect how the model is expected to "
            "perform on new, unseen patients. The C-index (concordance) measures "
            "the model's ability to correctly rank patients by risk: 0.5 is no "
            "better than chance, and 1.0 is perfect ranking. Brier score "
            "measures the mean squared error between predicted survival "
            "probabilities and observed outcomes (range 0\u20131; lower is better, "
            "0 = perfect)."
        )
        md_lines.append("")
        md_lines.append(
            "*Model selection should be based on cross-validation performance "
            "(see Model Development), not test set performance alone.*"
        )
        md_lines.append("")

    def _add_bootstrap_methodology_details(self, md_lines: list) -> None:
        """Add bootstrap methodology config table."""
        if not self._s.bootstrap_results:
            return

        first_result = next(iter(self._s.bootstrap_results.values()))
        if first_result.iterations is None or first_result.iterations.empty:
            return
        n_bootstrap = first_result.iterations['iteration'].nunique()

        config_data = [
            ("Bootstrap Iterations", str(n_bootstrap)),
            ("CI Method", "Percentile (2.5%, 97.5%)"),
            ("Resampling", f"Stratified by {self._s.outcome_name} status"),
        ]
        header = '<thead><tr><th>Parameter</th><th>Value</th></tr></thead>'
        rows = [
            f'<tr><td>{html_module.escape(param)}</td>'
            f'<td>{html_module.escape(value)}</td></tr>'
            for param, value in config_data
        ]
        table = (
            f'<table class="environment-info-table">'
            f'{header}<tbody>{"".join(rows)}</tbody></table>'
        )
        md_lines.append("<!-- MLSURV_RAW_HTML -->")
        md_lines.append(f'<div class="feature-table-scroll">{table}</div>')
        md_lines.append("<!-- /MLSURV_RAW_HTML -->")
        md_lines.append("")
