"""Per-family plot dispatch for feature importance (split from ``_plot_impl.py``)."""
from __future__ import annotations

from mlsurv._shared import tidy_model_col as _tidy_model_col
from mlsurv.viz import _data as _vi_feat
from mlsurv.viz._context import PlotContext
from mlsurv.viz._plot_impl_shared import (
    _available_model_names,
    _resolve_models_arg,
)
from mlsurv.viz._resolve import resolve_eval_result


def _plot_importance_impl(
    ctx: PlotContext,
    *,
    models=None,
    n: int = 20,
    style: str = "auto",
    return_data: bool = False,
    **kwargs,
):
    """Shared implementation for importance plots (main context).

    Called by ``SurvivalLearner.plot_importance`` after legacy-path handling.
    """
    from mlsurv.viz.features import (
        plot_importance as _plot_importance,
        plot_importance_heatmap as _plot_heatmap,
    )

    is_heatmap = style == "heatmap"

    model_names = _resolve_models_arg(ctx, models)
    eval_result = resolve_eval_result(ctx.state_ref, model_names)

    if eval_result.importance is None:
        raise ValueError("No importance data. Run evaluate() first.")

    if return_data:
        df = eval_result.importance
        if is_heatmap and 'coefficient' in df.columns:
            df = df.drop(columns='coefficient')
        return _tidy_model_col(df)

    if is_heatmap:
        return _plot_heatmap(
            eval_result.importance, n=n,
            model_col="model_type", **kwargs,
        )

    iterations_df, n_repeats = _vi_feat.resolve_importance_iterations(eval_result)

    fig = _plot_importance(
        eval_result.importance, n=n,
        model_col="model_type",
        style=style,
        iterations_df=iterations_df, n_repeats=n_repeats,
        **kwargs,
    )

    if 'title' not in kwargs:
        title_names = (
            model_names if model_names is not None
            else _available_model_names(ctx)
        )
        _vi_feat.append_single_model_title(fig, title_names)

    return fig
