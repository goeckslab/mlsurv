"""Base section class for report generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, TYPE_CHECKING
import re
import pandas as pd

from mlsurv.constants import METRIC_REGISTRY
from mlsurv.logging_utils import get_logger, log_exception
from mlsurv.reports._constants import (
    DEFAULT_RISK_GROUPS,
    QUALITY_DEFAULT,
    QUALITY_THRESHOLDS,
    RISK_SCORE_PRECISION,
)
from mlsurv.viz._palettes import get_model_display_name

if TYPE_CHECKING:
    from mlsurv.learner import SurvivalLearner

# Shared exception tuples for report section error handling
DATA_EXCEPTIONS = (AttributeError, KeyError, NameError, TypeError, ValueError)
RUNTIME_EXCEPTIONS = (RuntimeError, ZeroDivisionError)
ALL_EXCEPTIONS = DATA_EXCEPTIONS + RUNTIME_EXCEPTIONS
# Extended tuples for sections that interact with the OS or optional imports.
PLATFORM_EXCEPTIONS = ALL_EXCEPTIONS + (OSError,)
IMPORT_EXCEPTIONS = ALL_EXCEPTIONS + (ImportError,)


@dataclass
class FigureInfo:
    """
    Metadata about a figure for the consolidated programmatic access dropdown.

    Attributes
    ----------
    figure_id : str
        Unique identifier matching the figure filename (e.g., 'correlation_heatmap').
    title : str
        Human-readable title for the figure.
    data_description : str
        Brief description of the data structure returned.
    data_key : str, optional
        Key in section.data dict for programmatic access to tidy DataFrame.
    learner_code : str, optional
        Learner method call, e.g. ``learner.plot_feature_correlation()``.
    standalone_import : str, optional
        Standalone import, e.g. ``from mlsurv.viz import plot_feature_correlation``.
    """
    figure_id: str
    title: str
    data_description: str
    data_key: Optional[str] = None
    learner_code: Optional[str] = None
    standalone_import: Optional[str] = None


@dataclass
class DataInfo:
    """Metadata about a standalone data item for the consolidated access dropdown.

    Attributes
    ----------
    data_key : str
        Key in section.data dict (without section prefix).
    description : str
        Brief description.
    learner_code : str, optional
        Learner method call with ``return_data=True``.
    results_code : str, optional
        Direct results access, e.g. ``learner.eval_result.metrics``.
    """
    data_key: str
    description: str
    learner_code: Optional[str] = None
    results_code: Optional[str] = None


class BaseSection(ABC):
    """
    Abstract base class for report sections.

    Each section checks if its required data is available and generates
    content (markdown text and optional figures) when available.

    Attributes
    ----------
    name : str
        Section identifier for configuration.
    title : str
        Display title for the section header.
    """

    name: str = "base"
    title: str = "Base Section"

    def __init__(self, learner: 'SurvivalLearner'):
        """
        Initialize section with learner reference.

        Parameters
        ----------
        learner : SurvivalLearner
            The learner instance containing all analysis data.
        """
        self.learner = learner
        self.data: Dict[str, pd.DataFrame] = {}  # Tidy DataFrames for programmatic access
        self._logger = get_logger(f"reports.sections.{self.name}")

    @property
    def outcome_name(self) -> str:
        """User-facing outcome label (e.g., 'event', 'death')."""
        return self.learner._state.outcome_name

    @property
    def time_unit(self) -> Optional[str]:
        """User-facing time unit (e.g., 'months', 'years'), or None."""
        return self.learner._state.time_unit

    @property
    def feature_groups(self):
        """Feature-to-group mapping for grouped importance plots, or None."""
        return self.learner._state.feature_groups

    @property
    def group_colors(self):
        """Group-to-color mapping for grouped importance plots, or None."""
        return self.learner._state.group_colors

    @property
    def imputer(self):
        """Configured (unfitted) imputer template, or None.

        This is the setup-time template, not the fitted estimator (those live
        in ``result.model.named_steps``). Reports format its params for the
        pipeline/setup tables, so the read stays a report-layer adapter rather
        than a public ``learner`` accessor.
        """
        return self.learner._state.imputer

    @property
    def scaler(self):
        """Configured (unfitted) scaler template, or None.

        Setup-time template (not the fitted estimator); kept as a report-layer
        adapter, see :attr:`imputer`.
        """
        return self.learner._state.scaler

    @property
    def feature_selector(self):
        """Configured (unfitted) feature-selector template, or None.

        Setup-time template (not the fitted estimator); kept as a report-layer
        adapter, see :attr:`imputer`.
        """
        return self.learner._state.feature_selector

    def _group_plot_kwargs(self) -> dict:
        """Build kwargs for feature_groups/group_colors if set.

        When ``categorical_groups`` is also configured, infers the group
        for each categorical group name (e.g. ``"horTh"``) from its member
        columns' groups so that importance-level features get coloured too.
        """
        kw = {}
        if self.feature_groups is not None:
            fg = dict(self.feature_groups)
            cat_groups = getattr(self.learner._state, 'categorical_groups', None)
            if cat_groups:
                for cat_name, cols in cat_groups.items():
                    if cat_name not in fg:
                        member_groups = {fg.get(c) for c in cols if c in fg}
                        member_groups.discard(None)
                        if len(member_groups) == 1:
                            fg[cat_name] = member_groups.pop()
            kw['feature_groups'] = fg
            if self.group_colors is not None:
                kw['group_colors'] = self.group_colors
        return kw

    @property
    def time_label(self) -> str:
        """Axis / column label for time, with unit suffix if set."""
        unit = self.time_unit
        return f"Time ({unit})" if unit else "Time"

    def _has_deep_learning_models(self) -> bool:
        """Check if any deep learning models are present."""
        try:
            models = self.learner.models
            return any(k in {'deepsurv', 'deephit'} for k in models)
        except (AttributeError, TypeError) as exc:
            self._log_recoverable_exception(
                exc, f"{self.name}: deep learning model check"
            )
            return False

    def _has_xgboost_models(self) -> bool:
        """Check if any XGBoost-based models are present."""
        try:
            models = self.learner.models
            return any(k in {'xgbcox', 'xgbaft'} for k in models)
        except (AttributeError, TypeError) as exc:
            self._log_recoverable_exception(
                exc, f"{self.name}: xgboost model check"
            )
            return False

    def _log_recoverable_exception(
        self,
        exception: BaseException,
        context: str,
        *,
        warn_user: bool = False,
        user_message: Optional[str] = None,
    ) -> None:
        """Log a caught exception and optionally surface it as a warning."""
        log_exception(
            self._logger,
            exception,
            context,
            warn_user=warn_user,
            user_message=user_message,
        )

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this section can be generated.

        Returns
        -------
        bool
            True if all required data is available.
        """
        pass

    @abstractmethod
    def generate(self) -> Dict[str, Any]:
        """
        Generate section content.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'markdown': str - Section markdown content
            - 'figures': List[tuple] - List of (filename, plotly.Figure) tuples
            - 'figure_info': List[FigureInfo] - Optional metadata for programmatic access
        """
        pass

    def _format_table(self, df: pd.DataFrame, index: bool = False) -> str:
        """
        Format DataFrame as markdown table.

        Parameters
        ----------
        df : DataFrame
            Data to format.
        index : bool
            Include row index.

        Returns
        -------
        str
            Markdown table string.
        """
        df = self._drop_all_empty_columns(df)
        # Convert all columns to string to ensure consistent formatting
        df_str = df.astype(str)
        # Use disable_numparse to prevent tabulate from trying to parse
        # strings like 'True'/'False' as numbers
        return df_str.to_markdown(index=index, disable_numparse=True)

    @staticmethod
    def _drop_all_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns where every value is missing or blank."""
        if df is None or df.empty:
            return df

        def _is_non_empty(value: Any) -> bool:
            na_check = pd.isna(value)
            if hasattr(na_check, "all"):
                if bool(na_check.all()):
                    return False
            elif bool(na_check):
                return False
            if isinstance(value, str):
                return value.strip() != ""
            return True

        keep_cols = [col for col in df.columns if df[col].apply(_is_non_empty).any()]

        if not keep_cols:
            return df
        return df[keep_cols]

    @staticmethod
    def _build_section_outline(
        entries: list[tuple[str, str]],
        *,
        min_entries: int = 3,
    ) -> list[str]:
        """Build a numbered section outline with anchor links.

        Returns empty list if fewer than *min_entries* are provided.
        """
        if len(entries) < min_entries:
            return []
        lines = ["**Contents:**", ""]
        for i, (label, anchor) in enumerate(entries, 1):
            lines.append(f"{i}. [{label}](#{anchor})")
        lines.append("")
        return lines

    @staticmethod
    def _slugify(value: str) -> str:
        """Build a tab-key-safe slug."""
        slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
        return slug or "model"

    def _model_label(self, key: str) -> str:
        """Display label for a storage *key* — the plain algorithm display name."""
        return get_model_display_name(key)

    def _get_all_models(self) -> Dict[str, Any]:
        """Get all trained models from learner (both tuned and untuned)."""
        return dict(self.learner.models)

    def _get_trained_models(self) -> Dict[str, Any]:
        """Get all trained models from learner (non-tuned models)."""
        return {k: v for k, v in self.learner.models.items() if not v.is_tuned}

    def _get_tuned_models(self) -> Dict[str, Any]:
        """Get all tuned models from learner."""
        return {k: v for k, v in self.learner.models.items() if v.is_tuned}

    def _generate_figure_safe(
        self,
        figure_fn,
        filename: str,
        title: str,
        figure_info_params: Optional[Dict[str, str]] = None,
        error_prefix: str = "figure",
        **kwargs
    ) -> tuple:
        """
        Generate a figure with standard error handling.

        Parameters
        ----------
        figure_fn : callable
            Function that generates the plotly figure.
        filename : str
            Filename for the figure (e.g., 'correlation_heatmap.png').
        title : str
            Human-readable title for the figure.
        figure_info_params : dict, optional
            If provided, dict with keys: figure_id, title, data_description,
            data_key, learner_code, standalone_import. Used to create
            FigureInfo.
        error_prefix : str
            Prefix for error messages (e.g., "correlation heatmap").
        **kwargs
            Additional arguments passed to figure_fn.

        Returns
        -------
        tuple
            (fig, figure_info, error_md) where:
            - fig: the plotly Figure or None if error
            - figure_info: FigureInfo or None
            - error_md: error markdown string or None if success
        """
        try:
            fig = figure_fn(**kwargs)
            if fig is None:
                return None, None, None

            figure_info = None
            if figure_info_params:
                figure_info = FigureInfo(
                    figure_id=figure_info_params.get('figure_id', filename.replace('.png', '')),
                    title=figure_info_params.get('title', title),
                    data_description=figure_info_params.get('data_description', ''),
                    data_key=figure_info_params.get('data_key'),
                    learner_code=figure_info_params.get('learner_code'),
                    standalone_import=figure_info_params.get('standalone_import'),
                )
            return fig, figure_info, None

        except (ImportError,) + ALL_EXCEPTIONS as e:
            self._log_recoverable_exception(
                e,
                f"{self.name}: {error_prefix} generation",
                warn_user=True,
                user_message=f"Could not generate {error_prefix}: {e}",
            )
            error_md = f"*[{title} unavailable: {e}]*"
            return None, None, error_md

    @staticmethod
    def _interpret_quality(score: float) -> str:
        """Map a score (AUC, C-index) to a quality label.

        Thresholds follow Hosmer & Lemeshow (2000), *Applied Logistic
        Regression*, Table 5.8 (AUC discrimination benchmarks):
        >= 0.8 excellent, >= 0.7 good, >= 0.6 moderate, < 0.6 poor.
        """
        for threshold, label in QUALITY_THRESHOLDS:
            if score >= threshold:
                return label
        return QUALITY_DEFAULT

    @staticmethod
    def _create_risk_groups(test_data: pd.DataFrame, n_groups: int = DEFAULT_RISK_GROUPS):
        """Bin risk scores into quantile-based risk groups with fallback.

        Parameters
        ----------
        test_data : DataFrame
            Must contain a ``risk_score`` column.  A *copy* is made internally.
        n_groups : int
            Desired number of risk groups.

        Returns
        -------
        (test_data, actual_groups) or (None, 0)
            The modified DataFrame with a ``risk_group`` column and the number
            of groups actually created.  Returns ``(None, 0)`` when fewer than
            2 unique risk-score values exist.
        """
        test_data = test_data.copy()
        try:
            test_data['risk_group'] = pd.qcut(
                test_data['risk_score'], q=n_groups, labels=False, duplicates='drop'
            )
            return test_data, test_data['risk_group'].nunique()
        except ValueError:
            n_actual = test_data['risk_score'].nunique()
            actual_groups = min(n_groups, n_actual)
            if actual_groups < 2:
                return None, 0
            test_data['risk_group'] = pd.qcut(
                test_data['risk_score'], q=actual_groups, labels=False, duplicates='drop'
            )
            return test_data, actual_groups

    @staticmethod
    def _bold_best_in_column(df, col, lower_is_better=False, ideal_value=None):
        """Bold the best numeric value in *col*, modifying *df* in place.

        When ``ideal_value`` is set, "best" = closest to ideal (absolute distance).
        """
        numeric_vals = df[col].str.extract(r'^(-?[\d.]+)')[0].astype(float)
        if numeric_vals.isna().all():
            return

        if ideal_value is not None:
            # Best = closest to ideal value
            best_idx = (numeric_vals - ideal_value).abs().idxmin()
        else:
            best_idx = numeric_vals.idxmin() if lower_is_better else numeric_vals.idxmax()

        if pd.isna(best_idx):
            return
        df.loc[best_idx, col] = f"**{df.loc[best_idx, col]}**"

    def _bold_best_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bold the best value in each metric column."""
        df = df.copy()
        for col in df.columns:
            if col in ('Model', 'Status'):
                continue
            # Look up metric info by matching display_name/table_name in column header
            is_lower = '*' in col
            ideal = None
            for info in METRIC_REGISTRY.values():
                if info.display_name in col or info.table_name in col:
                    is_lower = info.lower_is_better
                    ideal = info.ideal_value
                    break
            self._bold_best_in_column(df, col, lower_is_better=is_lower, ideal_value=ideal)
        return df

    def _bold_best_auct_values(self, df: pd.DataFrame, n_model_rows: int) -> pd.DataFrame:
        """Bold best AUC in each timepoint column (model rows only)."""
        df = df.copy()
        for col in df.columns:
            if col == '':
                continue
            # Search only model rows but write directly to df (avoids iloc view issue)
            numeric_vals = df[col].iloc[:n_model_rows].str.extract(r'^([\d.]+)')[0].astype(float)
            if numeric_vals.isna().all():
                continue
            best_idx = numeric_vals.idxmax()
            if pd.isna(best_idx):
                continue
            df.loc[best_idx, col] = f"**{df.loc[best_idx, col]}**"
        return df

    @staticmethod
    def _format_prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize prediction rows to report schema."""
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["model", "sample_index", "time", "event", "risk_score"])

        frame = df.copy()
        if "model_type" in frame.columns:
            frame["model"] = frame["model_type"].map(get_model_display_name)
        else:
            frame["model"] = "Model"

        output = pd.DataFrame(
            {
                "model": frame["model"],
                "sample_index": frame["index"] if "index" in frame.columns else pd.RangeIndex(len(frame)),
                "time": pd.to_numeric(frame["time"], errors="coerce") if "time" in frame.columns else pd.NA,
                "event": pd.to_numeric(frame["event"], errors="coerce") if "event" in frame.columns else pd.NA,
                "risk_score": pd.to_numeric(frame["risk_score"], errors="coerce")
                if "risk_score" in frame.columns else pd.NA,
            }
        )
        output["risk_score"] = pd.to_numeric(output["risk_score"], errors="coerce").round(RISK_SCORE_PRECISION)
        return output.sort_values(["model", "sample_index"]).reset_index(drop=True)

    def _build_train_predictions_df(self) -> pd.DataFrame:
        """Build sample-level predictions on the train split."""
        X_train = getattr(self.learner, "X_train", None)
        y_train = getattr(self.learner, "y_train", None)
        models = self._get_all_models()
        if X_train is None or y_train is None or not models:
            return pd.DataFrame(columns=["model", "sample_index", "time", "event", "risk_score"])

        rows = []
        for model_name, result in models.items():
            model = getattr(result, "model", None)
            if model is None or not hasattr(model, "predict"):
                continue
            try:
                risk_scores = pd.to_numeric(pd.Series(model.predict(X_train)), errors="coerce")
                n = min(len(X_train), len(risk_scores))
                if n == 0:
                    continue
                part = pd.DataFrame(
                    {
                        "model": get_model_display_name(model_name),
                        "sample_index": pd.Series(X_train.index).iloc[:n].values,
                        "time": pd.to_numeric(pd.Series(y_train["time"]), errors="coerce").iloc[:n].values,
                        "event": pd.to_numeric(pd.Series(y_train["event"]), errors="coerce").iloc[:n].values,
                        "risk_score": risk_scores.iloc[:n].values,
                    }
                )
                rows.append(part)
            except ALL_EXCEPTIONS as e:
                self._log_recoverable_exception(e, f"{self.name}: train predictions for {model_name}")

        if not rows:
            return pd.DataFrame(columns=["model", "sample_index", "time", "event", "risk_score"])
        df = pd.concat(rows, ignore_index=True)
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").round(RISK_SCORE_PRECISION)
        return df.sort_values(["model", "sample_index"]).reset_index(drop=True)
