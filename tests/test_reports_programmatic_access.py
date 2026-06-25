"""Regression test for Programmatic Access snippets in report sections.

Builds a realistic ``SurvivalLearner`` pipeline (setup + run + tune + evaluate +
bootstrap + benchmark + validate + generate_report), renders each section's
Programmatic Access dropdown exactly as the HTML report does, and executes
every emitted code block in a namespace that contains only ``learner`` — the
report's own promise that the snippets need nothing but the learner.

The point: Programmatic Access dropdowns are built from FigureInfo / DataInfo
objects. If a snippet ever drifts (references the unreachable ``report``
object, wrong method name, wrong key, missing argument), this test reports
*which* block in *which* section failed rather than failing silently during
manual review.

Runs under ``--run-full`` only because the full train + tune + bootstrap
pipeline is slow.
"""

from __future__ import annotations

import html as html_module
import re
import warnings

import pytest

from mlsurv._warnings import (
    InternalWarning,
    MethodologicalWarning,
    SuggestionWarning,
)
from mlsurv.reports.renderers._figures import FigureEmbedder


# An access-item groups the code blocks for one figure/data entry; its blocks
# share a namespace (``fig = ...`` then ``fig.write_image(...)``), so we exec
# them together but isolate each item from the next.
_ACCESS_ITEM_RE = re.compile(r'<div class="access-item">(.*?)</div>', re.DOTALL)
_CODE_BLOCK_RE = re.compile(r'<pre><code>(.*?)</code></pre>', re.DOTALL)

# The unreachable ``report`` object the report must never advertise.
_FORBIDDEN_SNIPPET_TOKENS = (
    'report.figures[',
    'report.export_figure',
    'report.data[',
    'report.export_data',
)


def _iter_access_item_blocks(rendered_html: str):
    """Yield, per access-item, the unescaped code blocks in document order."""
    for item_html in _ACCESS_ITEM_RE.findall(rendered_html):
        blocks = [html_module.unescape(b) for b in _CODE_BLOCK_RE.findall(item_html)]
        if blocks:
            yield blocks


pytestmark = [pytest.mark.integration, pytest.mark.full]


# Exceptions a reasonable user-facing snippet might raise. Deliberately
# narrow per project CONTRIBUTING.md exception-handling rules: we want the test
# to surface real snippet failures, not swallow harness errors like
# MemoryError / RecursionError / KeyboardInterrupt.
_SNIPPET_EXCEPTIONS = (
    NameError,
    AttributeError,
    KeyError,
    IndexError,
    TypeError,
    ValueError,
    RuntimeError,
    ImportError,
    SyntaxError,
)


def _exec_snippet(snippet: str, namespace: dict, *, label: str) -> None:
    """Compile + exec a snippet, raising a descriptive error on failure."""
    try:
        code = compile(snippet, f"<{label}>", "exec")
        exec(code, namespace)
    except _SNIPPET_EXCEPTIONS as exc:
        raise AssertionError(
            f"Programmatic-access snippet failed [{label}]:\n"
            f"--- snippet ---\n{snippet}\n--- error ---\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _silence_expected_noise() -> None:
    """Scope warning suppression to categories the full pipeline is known
    to emit, while letting any *unexpected* warning (e.g. a new silent
    data-loss path per CONTRIBUTING.md) surface in pytest output for triage.

    Deliberately does NOT silence ``RuntimeWarning`` (overflow/divide-by-
    zero in numerical code — mlsurv cares about these) or un-scoped ``UserWarning``
    (sksurv convergence warnings are narrowed below by message).
    """
    # Acceptable per ``CONTRIBUTING.md``:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Narrow: sksurv/lifelines emit this during Cox fitting on the GBSG2
    # subsets used by the fixture.
    warnings.filterwarnings(
        "ignore",
        message=r".*did not converge.*",
        category=UserWarning,
    )
    # mlsurv domain warnings: small dataset + few trials triggers EPV +
    # CV-repeats suggestions on every single fixture call.
    warnings.filterwarnings("ignore", category=MethodologicalWarning)
    warnings.filterwarnings("ignore", category=SuggestionWarning)
    warnings.filterwarnings("ignore", category=InternalWarning)


def _collect_section_results(report):
    """Run each section's ``generate()`` and return its content dict.

    ``ReportGenerator`` calls ``section.generate()`` during report creation
    but does not expose the per-section ``figure_info`` / ``data_info`` list
    afterwards.  Re-invoking ``generate()`` is safe: sections are pure
    functions of learner state.
    """
    section_results = []
    for section in report._sections:
        with warnings.catch_warnings():
            _silence_expected_noise()
            result = section.generate()
        section_results.append(
            (
                section.name,
                result.get("figure_info", []),
                result.get("data_info", []),
                list(section.data.keys()),
            )
        )
    return section_results


@pytest.fixture(scope="module")
def programmatic_access_learner(_gbsg2_data_raw):
    """Full lifecycle learner feeding every report section with figures.

    Covers: CV training (ModelDevelopment, FeatureAnalysis, Evaluation),

    Covers: CV training (ModelDevelopment, FeatureAnalysis, Evaluation),
    explicit tuning (Optuna plots), bootstrap (bootstrap metrics &
    rainclouds), benchmark (Benchmarking), validate (Validation cohort
    subsections), subpopulations (SubpopulationAnalysis).
    """
    import numpy as np

    from mlsurv import SurvivalLearner

    X_raw, y_raw = _gbsg2_data_raw
    X = X_raw.copy()
    # Snapshot menopause labels before they get cat-coded (needed for subpopulations)
    meno_labels = X["menostat"].astype(str).values
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = X[col].cat.codes.astype(float)
    y = np.array(
        [(bool(e), float(t)) for e, t in y_raw],
        dtype=[("event", bool), ("time", float)],
    )

    with warnings.catch_warnings():
        _silence_expected_noise()

        learner = SurvivalLearner(X, y)
        learner.setup(
            random_state=42,
            cv_folds=3,
            test_fraction=0.3,
            feature_selector=None,
            subpopulations={"menostat": meno_labels},
        )

        # Train two models (no auto-tune/bootstrap — driven explicitly below)
        learner.run(
            models=["coxph", "coxnet"],
            tune=False,
            bootstrap=False,
            verbose=False,
        )

        # Explicit small tune so the Optuna plots appear.
        # n_trials=15 gives fANOVA enough signal to produce a non-empty
        # param-importance dict across Optuna versions (5 trials is
        # deterministic under the current TPE seed but not statistically
        # safe). The DataInfo for tuning_param_importance is only added
        # when that data is populated — see model_development.py::generate.
        learner.tune(
            "coxnet",
            n_trials=15,
            evaluate=False,
            bootstrap=False,
            verbose=False,
        )

        # Small bootstrap for bootstrap-related figures (CONTRIBUTING.md fixture
        # convention — minimize runtime).
        learner.bootstrap(n_bootstrap=2, verbose=False)

        # Benchmark against a raw feature as a reference score.
        learner.benchmark(
            references={"baseline_tsize": X["tsize"].values},
            bootstrap=False,
            verbose=False,
        )

        # External validation using two pseudo-cohorts (halves of X). Two
        # cohorts are required for the cross-cohort validation metrics heatmap
        # to render — and thus to exec its
        # ``fig = learner.plot_validation_heatmap()`` snippet. A single
        # cohort falls back to a markdown table and emits no heatmap figure.
        half = len(X) // 2
        learner.validate(
            X.iloc[half:],
            y[half:],
            name="pilot",
            bootstrap=False,
            verbose=False,
        )
        learner.validate(
            X.iloc[:half],
            y[:half],
            name="external",
            bootstrap=False,
            verbose=False,
        )

    return learner


@pytest.fixture(scope="module")
def programmatic_access_report(programmatic_access_learner, tmp_path_factory):
    """Build a ``ReportGenerator`` directly so tests can inspect ``_sections``.

    ``learner.generate_report()`` returns the output file path, not the
    generator instance, so we mirror the same wiring locally.
    """
    from mlsurv.reports import ReportGenerator

    learner = programmatic_access_learner
    out_dir = tmp_path_factory.mktemp("prog_access_report")

    with warnings.catch_warnings():
        _silence_expected_noise()

        eval_result, bootstrap_results = learner._gather_report_data(
            None, None
        )
        generator = ReportGenerator(learner)
        if eval_result is not None:
            generator.set_evaluation_result(eval_result)
        if bootstrap_results is not None:
            generator.set_bootstrap_results(bootstrap_results)

        generator.generate(output_path=str(out_dir / "report.html"))

    return generator


class TestProgrammaticAccessSnippets:
    """Every rendered snippet in every section must execute cleanly."""

    def test_rendered_access_snippets_execute(
        self, programmatic_access_learner, programmatic_access_report,
        tmp_path, monkeypatch,
    ):
        """Exec the *actual rendered* Programmatic Access blocks.

        For every section, render the consolidated-access dropdown exactly as
        the HTML report does, confirm it never references the unreachable
        ``report`` object, then exec each access-item's code blocks in a fresh
        namespace holding only ``learner``. A NameError (e.g. ``report``
        undefined) or wrong method name fails the test with the offending
        snippet — which the old test, that only ran ``learner_code`` directly,
        could never catch.
        """
        learner = programmatic_access_learner
        report = programmatic_access_report

        # Snippets export with fig.write_image(...). Stub the (kaleido-backed)
        # writer so the test stays fast and deterministic: we are verifying
        # that *our* emitted snippet binds ``fig`` and calls write_image with a
        # path, not that kaleido renders (the E2E script covers real rendering).
        import plotly.graph_objects as go

        def _stub_write_image(self, path, *args, **kwargs):
            with open(path, "w"):
                pass

        monkeypatch.setattr(go.Figure, "write_image", _stub_write_image)
        monkeypatch.chdir(tmp_path)  # fig.write_image() writes its file here

        results = _collect_section_results(report)

        n_items = 0
        for section_name, figure_info_list, data_info_list, data_keys in results:
            rendered = FigureEmbedder.render_consolidated_access(
                section_name=section_name,
                figure_info_list=figure_info_list,
                data_keys=list(data_keys),
                data_info_list=data_info_list,
            )
            for token in _FORBIDDEN_SNIPPET_TOKENS:
                assert token not in rendered, (
                    f"Section '{section_name}' still advertises the unreachable "
                    f"report object via '{token}'"
                )

            for blocks in _iter_access_item_blocks(rendered):
                ns = {"learner": learner}
                for block in blocks:
                    with warnings.catch_warnings():
                        _silence_expected_noise()
                        _exec_snippet(
                            block, ns,
                            label=f"{section_name} rendered access block",
                        )
                n_items += 1

        assert n_items > 0, "Expected at least one rendered access-item to exec"

    def test_report_data_figures_dicts_populated(
        self, programmatic_access_report
    ):
        """The report's ``data`` / ``figures`` dicts back every advertised key.

        These dicts still exist as the internal store; this guards that each
        ``FigureInfo`` / ``data_key`` the sections advertise is actually
        present, independent of how the user-facing snippets reach it.
        """
        report = programmatic_access_report
        results = _collect_section_results(report)

        for section_name, figure_info_list, _data_info, data_keys in results:
            for info in figure_info_list:
                full_key = f"{section_name}_{info.figure_id}"
                assert full_key in report.figures, (
                    f"Figure key '{full_key}' advertised by FigureInfo is "
                    f"missing from report.figures"
                )
            for dk in data_keys:
                full_key = f"{section_name}_{dk}"
                assert full_key in report.data, (
                    f"Data key '{full_key}' advertised by section.data is "
                    f"missing from report.data"
                )

    def test_manual_generator_auto_wires_full_report(
        self, programmatic_access_learner, programmatic_access_report,
        tmp_path,
    ):
        """A bare ReportGenerator(learner).generate() must NOT degrade.

        Secondary bug: building a ``ReportGenerator`` by hand and calling
        ``generate()`` without ``set_evaluation_result`` / ``set_bootstrap_results``
        used to skip the evaluation wiring, logging a risk-score error and
        rendering a partial figure set. ``generate()`` now auto-wires from the
        learner's canonical state, so the unwired path must yield the same
        figures as the explicitly-wired reference report.
        """
        from mlsurv.reports import ReportGenerator

        learner = programmatic_access_learner
        wired_figures = set(programmatic_access_report.figures.keys())

        with warnings.catch_warnings():
            _silence_expected_noise()
            auto = ReportGenerator(learner)  # no set_*; relies on auto-wiring
            auto.generate(output_path=str(tmp_path / "auto.html"))

        assert set(auto.figures.keys()) == wired_figures, (
            "Auto-wired report is missing figures the explicitly-wired report "
            "produced — secondary degradation regressed:\n"
            f"missing: {sorted(wired_figures - set(auto.figures.keys()))}"
        )

    def test_manual_generator_auto_wires_missing_slot(
        self, programmatic_access_learner, programmatic_access_report,
        tmp_path,
    ):
        """Partial wiring must still auto-fill the empty slot.

        A caller who sets only bootstrap (and not evaluation) must still get the
        evaluation results auto-wired, rather than silently dropping the
        eval-dependent figures. Guards the per-slot auto-wire in generate().
        """
        from mlsurv.reports import ReportGenerator
        from mlsurv.learner._reports import _split_bootstrap_to_dict

        learner = programmatic_access_learner
        wired_figures = set(programmatic_access_report.figures.keys())

        # Inject bootstrap unconditionally (split from the combined result) so the
        # generator enters generate() with _bootstrap_results set but _eval_result
        # still None — the per-slot branch that must auto-wire evaluation. (Going
        # through _gather_report_data here would return None for bootstrap, since
        # it is embedded in the eval result, and degenerate into the both-None
        # case already covered above.)
        combined = learner.bootstrap_result
        assert combined is not None, "fixture must have bootstrap results"

        with warnings.catch_warnings():
            _silence_expected_noise()
            partial = ReportGenerator(learner)
            partial.set_bootstrap_results(_split_bootstrap_to_dict(combined))
            assert partial._eval_result is None  # eval slot deliberately empty
            partial.generate(output_path=str(tmp_path / "partial.html"))

        assert set(partial.figures.keys()) == wired_figures, (
            "Bootstrap-only wiring failed to auto-fill evaluation; eval-dependent "
            "figures were silently dropped:\n"
            f"missing: {sorted(wired_figures - set(partial.figures.keys()))}"
        )

    def test_all_sections_figure_data_keys_match(
        self, programmatic_access_report
    ):
        """FigureInfo.data_key must be a real key in section.data for every section.

        Regression for the pre-prefixed ``data_key`` bug in benchmarking
        (e.g. ``data_key="benchmarking_auct"`` → ``"auct"``). The matching
        logic in ``_render_data_items`` compares unprefixed
        ``FigureInfo.data_key`` against unprefixed ``section.data`` keys;
        pre-prefixed keys silently fail to match, producing a rendered
        dropdown with no figure↔data link.
        """
        unmatched = []
        for section in programmatic_access_report._sections:
            with warnings.catch_warnings():
                _silence_expected_noise()
                result = section.generate()
            section_keys = set(section.data.keys())
            for info in result.get("figure_info", []):
                if info.data_key is None:
                    continue
                if info.data_key not in section_keys:
                    unmatched.append(
                        f"{section.name}.{info.figure_id}: data_key="
                        f"{info.data_key!r} not in {sorted(section_keys)}"
                    )
        assert not unmatched, (
            "FigureInfo.data_key mismatches (pre-prefix bug):\n"
            + "\n".join(unmatched)
        )

    def test_all_sections_data_info_keys_match(
        self, programmatic_access_report
    ):
        """DataInfo.data_key must also be a real key in section.data.

        Guards the opposite direction from
        ``test_all_sections_figure_data_keys_match``: a DataInfo whose
        ``data_key`` does not exist in ``section.data`` is silently
        dropped by the renderer (``_render_data_items`` iterates
        ``data_keys``, not ``data_info_list``), producing a Programmatic
        Access dropdown that promises a link that never appears.
        """
        unmatched = []
        for section in programmatic_access_report._sections:
            with warnings.catch_warnings():
                _silence_expected_noise()
                result = section.generate()
            section_keys = set(section.data.keys())
            for di in result.get("data_info", []):
                if di.data_key not in section_keys:
                    unmatched.append(
                        f"{section.name}: DataInfo.data_key="
                        f"{di.data_key!r} not in {sorted(section_keys)}"
                    )
        assert not unmatched, (
            "Dangling DataInfo entries (data_key not in section.data):\n"
            + "\n".join(unmatched)
        )

    def test_no_section_pre_prefixes_its_own_data_keys(
        self, programmatic_access_report
    ):
        """Regression guard: sections must not pre-prefix their own data keys.

        ``ReportGenerator.data`` applies ``f"{section.name}_{key}"``
        exactly once when aggregating ``report.data``. Sections must NOT
        pre-prefix their own ``section.data`` / ``FigureInfo.data_key`` /
        ``DataInfo.data_key`` with ``section.name + "_"`` — doing so
        produces double-prefixed user-facing keys like
        ``report.data['validation_validation_km_pilot']``.

        Note: ``FigureInfo.figure_id`` is NOT guarded here because the
        figure-side cleanup is intentionally deferred to a follow-up;
        figure filenames on disk still embed the section prefix.
        """
        offenders = []
        for section in programmatic_access_report._sections:
            prefix = f"{section.name}_"
            with warnings.catch_warnings():
                _silence_expected_noise()
                result = section.generate()

            for key in section.data.keys():
                if key.startswith(prefix):
                    offenders.append(
                        f"{section.name}: section.data key {key!r} "
                        f"starts with {prefix!r}"
                    )
            for info in result.get("figure_info", []):
                if info.data_key and info.data_key.startswith(prefix):
                    offenders.append(
                        f"{section.name}: FigureInfo.data_key "
                        f"{info.data_key!r} starts with {prefix!r}"
                    )
            for di in result.get("data_info", []):
                if di.data_key.startswith(prefix):
                    offenders.append(
                        f"{section.name}: DataInfo.data_key "
                        f"{di.data_key!r} starts with {prefix!r}"
                    )
        assert not offenders, (
            "Sections pre-prefixed their own data keys (produces double "
            "prefix in report.data):\n"
            + "\n".join(offenders)
        )

    def test_expected_data_info_paths_exercised(
        self, programmatic_access_report
    ):
        """Load-bearing guard: assert the *new* DataInfo paths actually fire.

        If the fixture is ever trimmed (e.g., drop validate() for speed),
        this test fails loudly instead of silently losing coverage of the
        new ``_build_cohort_data_info`` + ``tuning_param_importance``
        snippets this PR added.
        """
        results = _collect_section_results(programmatic_access_report)

        exercised = set()
        for section_name, _figure_info, data_info_list, _data_keys in results:
            for di in data_info_list:
                exercised.add(f"{section_name}.{di.data_key}")

        required = {
            # Optuna param_importance DataInfo in
            # model_development.generate().
            "model_development.tuning_param_importance",
            # Validation DataInfo entries produced
            # by _build_cohort_data_info. Keys are
            # unprefixed (``section.name + "_" + key`` is applied
            # once by the generator).
            "validation.auct_pilot",
            "validation.calibration_pilot",
            "validation.km_pilot",
            "validation.median_pilot",
            "validation.rmst_pilot",
        }
        missing = required - exercised
        assert not missing, (
            f"Expected DataInfo paths were not emitted by the current "
            f"fixture: {missing}. Either adjust the fixture to exercise "
            f"them (currently: tune enough trials for fANOVA + call "
            f"validate() with a cohort named 'pilot') or drop stale "
            f"entries from ``required`` with a comment explaining why."
        )

    def test_no_private_state_in_snippet_metadata(
        self, programmatic_access_report
    ):
        """No user-facing snippet string leaks private ``_state``.

        ``test_sections_state_access_discipline.py`` scans section *source* for the literal
        ``._state.``; this is the runtime complement — it walks the
        ``FigureInfo`` / ``DataInfo`` strings the sections actually emit
        (``learner_code``, ``standalone_import``, ``results_code``) and
        asserts none reach into ``_state``. Catches a snippet that builds
        the offending string dynamically, which a source-text grep misses.
        """
        results = _collect_section_results(programmatic_access_report)

        offenders = []
        for section_name, figure_info_list, data_info_list, _keys in results:
            snippets = []
            for info in figure_info_list:
                snippets.append((f"{section_name}.{info.figure_id}.learner_code",
                                 info.learner_code))
                snippets.append((f"{section_name}.{info.figure_id}."
                                 "standalone_import", info.standalone_import))
            for di in data_info_list:
                snippets.append((f"{section_name}.{di.data_key}.results_code",
                                 di.results_code))
                snippets.append((f"{section_name}.{di.data_key}.learner_code",
                                 di.learner_code))
            for label, text in snippets:
                if text and "_state" in text:
                    offenders.append(f"{label}: {text!r}")

        assert not offenders, (
            "User-facing programmatic-access snippet(s) reference private "
            "`_state` — use a public accessor (e.g. "
            "`learner.validation_results`) instead:\n  - "
            + "\n  - ".join(offenders)
        )
