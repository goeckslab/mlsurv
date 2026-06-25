"""Calibration analysis per validation cohort."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import pandas as pd

from mlsurv._warnings import SuggestionWarning

from mlsurv.reports._constants import DEFAULT_CALIBRATION_BINS
from mlsurv.reports.sections.base import ALL_EXCEPTIONS, FigureInfo
from mlsurv.reports.sections.validation._base import _BaseValidationSubsection
from mlsurv.reports.sections.validation._helpers import (
    _CAL_COLUMNS,
    collect_calibration_rows,
    empty_calibration_figure,
)


class _CalibrationSubsection(_BaseValidationSubsection):
    """Calibration plots and tables for one external validation cohort.

    Figures are built by :meth:`ValidationView.plot_calibration` so
    users can reproduce them via the learner API. Tidy bin data for the
    per-cohort export is computed inline via the pure viz helpers so the
    section controls the exported column set (:data:`_CAL_COLUMNS`).

    Horizons come from the cohort's canonical eval grid (the same resolution
    the AUC(t)/Brier(t) tables use), sampled down to a few representative
    timepoints for the static side-by-side layout.
    """

    def add(
        self,
        md_lines: list,
        figures: list,
        figure_info: list,
        val_result,
        cohort_name: str,
        cohort_key: str,
    ) -> None:
        from mlsurv.viz.calibration import (
            _compute_multi_model_calibration_data,
            compute_calibration,
        )

        X_val = val_result.X_val
        y_val = val_result.y_val

        models = self._get_calibration_models(val_result, X_val, y_val, md_lines)
        if models is None:
            return

        md_lines.append(
            "Calibration plots compare predicted survival probabilities against observed "
            "(Kaplan-Meier estimated) survival at specific timepoints. Points on the "
            "diagonal indicate perfect calibration."
        )
        md_lines.append("")

        # Source the horizons from the cohort's realized eval grid (the same
        # auc_t grid its AUC(t) table uses), then pick up to 3 representative
        # ones so the static side-by-side layout stays readable.
        eval_times, from_auc_t = _select_report_horizons(val_result, y_val)

        timepoint_strs = [f"t={t:.1f}" for t in eval_times]
        md_lines.append(f"**Evaluation timepoints:** {', '.join(timepoint_strs)}")
        if not from_auc_t:
            md_lines.append("")
            md_lines.append(
                "*Note: the cohort's AUC(t) grid was unavailable, so these "
                "calibration horizons fall back to event-time percentiles and "
                "may differ from the AUC(t)/Brier(t) tables.*"
            )
        md_lines.append("")

        cal_data_key = f'calibration_{cohort_key}'
        self._s.data[cal_data_key] = pd.DataFrame(columns=_CAL_COLUMNS)
        cal_data_rows: List[Dict[str, Any]] = []

        is_multi = len(models) > 1
        view = self._s.learner.validation(cohort_name)

        md_lines.append('<!-- MLSURV_RAW_HTML -->')
        md_lines.append('<div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">')

        for eval_time in eval_times:
            try:
                fig, cal_items = _build_calibration_figure_and_items(
                    view=view,
                    models=models,
                    X_val=X_val,
                    y_val=y_val,
                    eval_time=eval_time,
                    cohort_name=cohort_name,
                    is_multi=is_multi,
                    compute_calibration_fn=compute_calibration,
                    compute_multi_model_fn=_compute_multi_model_calibration_data,
                )

                cal_data_rows.extend(collect_calibration_rows(cal_items))

                t_tag = f'{eval_time:.1f}'.replace('.', 'p')
                suffix = f'_comparison_t{t_tag}' if is_multi else f'_t{t_tag}'
                fig_id = f'validation_cal_{cohort_key}{suffix}'
                fig_filename = f'{fig_id}.png'
                figures.append((fig_filename, fig))
                md_lines.append('<div style="flex: 1; min-width: 300px; max-width: 400px;">')
                md_lines.append(f'<div class="figure-placeholder" data-figure="{fig_filename}"></div>')
                md_lines.append('</div>')

                desc_prefix = 'Multi-model calibration' if is_multi else 'Calibration'
                title_prefix = 'Calibration Comparison' if is_multi else 'Calibration'
                figure_info.append(FigureInfo(
                    figure_id=fig_id,
                    title=f'{cohort_name} {title_prefix} at t={eval_time:.1f}',
                    data_description=f'{desc_prefix} data for {cohort_name}',
                    data_key=cal_data_key,
                    learner_code=_build_calibration_snippet(
                        cohort_name=cohort_name,
                        eval_time=eval_time,
                        is_multi=is_multi,
                        single_model_name=(
                            None if is_multi else next(iter(models.keys()))
                        ),
                    ),
                ))
            except (NotImplementedError, TypeError):
                # Developer bugs should crash the report build, not be
                # silently degraded to a missing-figure warning. The
                # view proxy raises NotImplementedError for unsupported
                # kwargs and TypeError for wrong models= type — both
                # indicate programming errors, not data issues.
                raise
            except ALL_EXCEPTIONS as e:
                self._log_error(
                    e,
                    f"validation: calibration at t={eval_time:.1f} for {cohort_name}",
                    warn_user=True,
                    user_message=f"Could not generate calibration at t={eval_time:.1f} for {cohort_name}: {e}",
                )
                md_lines.append('<div style="flex: 1; min-width: 300px;">')
                md_lines.append(f"*[Calibration unavailable: {e}]*")
                md_lines.append('</div>')

        md_lines.append('</div>')
        md_lines.append('<!-- /MLSURV_RAW_HTML -->')
        md_lines.append("")

        # Store tidy calibration data (update pre-initialized key)
        if cal_data_rows:
            self._s.data[cal_data_key] = pd.DataFrame(cal_data_rows, columns=_CAL_COLUMNS)

    def _get_calibration_models(self, val_result, X_val, y_val, md_lines):
        """Get validated models supporting survival function prediction."""
        if X_val is None or y_val is None:
            md_lines.append("*Calibration unavailable: validation features not stored.*")
            md_lines.append("")
            return None

        # Scope to only the model(s) that were actually validated
        concordance = val_result.concordance
        if concordance is None or concordance.empty:
            md_lines.append("*No validation metrics available for calibration.*")
            md_lines.append("")
            return None

        validated_names = set(concordance['model_type'].unique())
        all_models = self._s._get_all_models()

        models = {
            name: result.model
            for name, result in all_models.items()
            if name in validated_names
            and hasattr(result.model, 'predict_survival_function')
        }
        if not models:
            md_lines.append(
                "*No validated models support survival function prediction "
                "for calibration.*"
            )
            md_lines.append("")
            return None
        return models


def _select_report_horizons(val_result, y_val, k: int = 3):
    """Up to *k* representative horizons from the cohort's realized eval grid.

    Reads the cohort's ``auc_t`` grid (the same horizons its AUC(t) table
    reports, resolved by ``validate()`` through the canonical eval-times
    resolver), then samples up to *k* horizons spread across it (roughly the
    25/50/75 positions) so the static side-by-side calibration layout stays
    readable. Falls back to the cohort's 25/50/75 event-time percentiles only
    when no ``auc_t`` grid is available.

    Returns
    -------
    (list[float], bool)
        The selected horizons and ``from_auc_t`` — ``True`` when they came from
        the cohort's ``auc_t`` grid, ``False`` when the percentile fallback was
        used (so the caller can flag the divergence to the reader).
    """
    from mlsurv.reports._helpers import eval_grid_with_percentile_fallback

    evaluation = getattr(val_result, "evaluation", None)
    grid, from_auc_t = eval_grid_with_percentile_fallback(
        [getattr(evaluation, "auc_t", None)], y_val,
    )
    n = len(grid)
    if n <= k:
        return grid, from_auc_t
    idx = sorted({round(q * (n - 1)) for q in (0.25, 0.50, 0.75)})
    return [grid[i] for i in idx], from_auc_t


def _build_calibration_figure_and_items(
    *,
    view,
    models: dict,
    X_val,
    y_val,
    eval_time: float,
    cohort_name: str,
    is_multi: bool,
    compute_calibration_fn,
    compute_multi_model_fn,
) -> tuple:
    """Compute tidy calibration data, then delegate figure to the view proxy.

    The tidy items are computed first — the same cheap viz helpers run
    inside ``ValidationView.plot_calibration`` — so we can detect the
    empty-bins case and fall back to an empty placeholder figure
    without routing through the proxy. When bins are populated, the
    figure is built by the proxy so users can reproduce it via the
    snippet in ``FigureInfo.learner_code``.

    Returns
    -------
    (figure, cal_items)
        ``cal_items`` is a list of ``(model_name, item_dict)`` pairs
        suitable for :func:`~mlsurv.reports._helpers.collect_calibration_rows`.
    """
    if is_multi:
        title = f'Calibration Comparison at t={eval_time:.1f}'
        cal_by_model = compute_multi_model_fn(
            models=models, X_test=X_val, y_test=y_val,
            eval_time=eval_time, n_bins=DEFAULT_CALIBRATION_BINS,
        )
        if not cal_by_model:
            warnings.warn(
                f"Calibration comparison at t={eval_time:.1f} for "
                f"{cohort_name}: no bins had enough samples "
                f"(n_bins={DEFAULT_CALIBRATION_BINS}).",
                SuggestionWarning, stacklevel=2,
            )
            return empty_calibration_figure(title), []

        fig = view.plot_calibration(
            eval_times=eval_time,
            n_bins=DEFAULT_CALIBRATION_BINS,
            title=title,
        )
        return fig, list(cal_by_model.items())

    model_name, model = next(iter(models.items()))
    title = f'Calibration at t={eval_time:.1f}'
    cal_data = compute_calibration_fn(
        model=model,
        X_test=X_val,
        y_test=y_val,
        eval_times=[eval_time],
        n_bins=DEFAULT_CALIBRATION_BINS,
    )
    if not cal_data:
        warnings.warn(
            f"Calibration at t={eval_time:.1f} for {cohort_name}: no bins had "
            f"enough samples (n_bins={DEFAULT_CALIBRATION_BINS}).",
            SuggestionWarning, stacklevel=2,
        )
        return empty_calibration_figure(title), []

    fig = view.plot_calibration(
        models=model_name,
        eval_times=[eval_time],
        n_bins=DEFAULT_CALIBRATION_BINS,
        title=title,
    )
    return fig, [(model_name, item) for item in cal_data]


def _build_calibration_snippet(
    *,
    cohort_name: str,
    eval_time: float,
    is_multi: bool,
    single_model_name: str | None,
) -> str:
    """Build the ``learner_code`` reproduction snippet for a FigureInfo.

    ``eval_time`` is cast to a native ``float`` so that the repr is a
    plain numeric literal even when the caller passes a numpy scalar
    (the resolved eval grid is ``np.float64``). This keeps the snippet
    executable in a bare namespace — ``test_figure_snippets_execute``
    only binds ``learner``, not ``numpy``.
    """
    eval_time = float(eval_time)
    if is_multi:
        return (
            f"fig = learner.validation({cohort_name!r}).plot_calibration(\n"
            f"    eval_times={eval_time!r},\n"
            f"    n_bins={DEFAULT_CALIBRATION_BINS},\n"
            f")"
        )
    return (
        f"fig = learner.validation({cohort_name!r}).plot_calibration(\n"
        f"    models={single_model_name!r},\n"
        f"    eval_times=[{eval_time!r}],\n"
        f"    n_bins={DEFAULT_CALIBRATION_BINS},\n"
        f")"
    )
