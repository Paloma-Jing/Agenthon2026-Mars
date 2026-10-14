"""A bounded tool-calling controller; no arbitrary Python, shell or direct text access.

The model object is injected by the caller. numeric_agent.house builds the organizer
endpoint client inside the competition image; __main__ builds a local debug client
only outside it. This module reads no credentials and hardcodes no endpoint.
"""
import json
from smolagents import ToolCallingAgent, tool
from .engine import METHODS


def build_agent(session,model,max_steps=8):
    @tool
    def inspect_data() -> str:
        """Inspect numerical task shape, target transformations, coverage and allowed methods."""
        return json.dumps(session.describe())

    @tool
    def diagnose_series() -> str:
        """Compute increment moments, ADF stationarity, autocorrelation, volatility and joint PCA diagnostics."""
        return json.dumps(session.diagnose())

    @tool
    def compute_factors(asset: str) -> str:
        """Construct classic factors and economic metrics for one target asset from the panel.

        Args:
            asset: One target asset id (from inspect_data). Returns family, panel data quality
                   and family-specific factors (term-structure, FX momentum/carry-proxy,
                   factor momentum/vol, macro YoY/MoM, etc.).
        """
        if asset not in session.assets: raise ValueError(f'unknown asset: {asset}')
        return json.dumps(session.factor_report(asset))

    @tool
    def data_quality() -> str:
        """Summarize panel missing rate, cross-asset correlation and PCA variance shares."""
        return json.dumps(session.data_quality_report())

    @tool
    def build_features() -> str:
        """Build and inspect the causal feature matrix actually consumed by the tree: names, missingness and usable columns. Uses the commander's selected families and window."""
        return json.dumps(session.feature_report())

    @tool
    def evaluate_random_walk() -> str:
        """Evaluate the random-walk reference with joint residual paths on historical folds.

        Returns validation metrics; normalized_composite uses reconstructed M0 expected-error scales.
        This is a numerical baseline, not the organizer's M0 implementation or score.
        To write its forecast, diagnose first and call export_forecast with method random_walk.
        """
        return json.dumps(session.evaluate('random_walk'))

    @tool
    def compare_models(methods: list[str]) -> str:
        """Run chronological, training-only rolling validation of selected numerical models.

        Args:
            methods: Select from inspect_data's allowed menu: random_walk, gbm,
                solution_ensemble. random_walk is always included. The ensemble internally
                fits damped trends, relative-price tree features and joint residual calibration.
        """
        if not 1<=len(methods)<=len(METHODS) or not set(methods)<=set(METHODS):raise ValueError('choose random_walk, gbm and/or solution_ensemble')
        ordered = ['random_walk'] + [m for m in methods if m != 'random_walk']
        return json.dumps([session.evaluate(m) for m in dict.fromkeys(ordered)])

    @tool
    def review_model_evidence() -> str:
        """Review cached experiments on identical fold keys and M0 scales. Reports paired wins,
        losses, worst loss increase, failures, leave-one-time-block-out winner stability
        and a mean-loss recommendation. Winner changes indicate sensitivity, not confidence.
        Read-only:
        does not train, tune, blend or establish independent forecasting gains.
        """
        return json.dumps(session.review_evidence())

    @tool
    def export_forecast(method: str, reason: str) -> str:
        """Fit an evaluated model and write coherent joint forecast samples. Requires diagnostics and successful validation; single-candidate exports are flagged as fallback.

        Args:
            method: One successfully evaluated candidate name.
            reason: Concise justification based exclusively on returned numeric diagnostics and validation.
        """
        return json.dumps(session.export(method,reason))

    def callback(step,**kwargs):
        session.trace_step(step)
    return ToolCallingAgent(model=model,tools=[inspect_data,diagnose_series,compute_factors,data_quality,build_features,evaluate_random_walk,compare_models,review_model_evidence,export_forecast],
        max_steps=max_steps,max_tool_threads=1,verbosity_level=0,step_callbacks=[callback],
        instructions='You are the numerical executor after the commander planning stage. Inspect data and diagnose series. Call build_features to inspect usable feature names, missingness and selected context families before comparing the allowed candidates against random_walk. solution_ensemble obeys the commander ensemble_weights exactly, including zero random-walk allocation; null delegates weights to two internal historical-error checks without a minimum random-walk share. It supports damped trends, relative-price features and joint residual KDE. Do not override the commander weights. Its component failures and weights are returned in model_audit; inspect them and do not call it FFORMA or ES-RNN. compute_factors provides economic diagnostics, not an extra model. The market feature profile is experimental and inactive; do not describe its preview as features consumed by the active core tree. The active tree uses an extra maximum-horizon training-to-calibration gap. The evaluate_random_walk tool exposes the numerical reference directly; compare_models also includes it automatically. Compare ALL candidates returned by inspect_data, then call review_model_evidence before exporting ONE successful model. Use its paired fold differences, new failures, worst degradation and leave-one-block-out stability in your justification. Its recommendation follows the existing mean-loss criterion; it is not independent validation. Never search new features or ensemble weights on these selection folds. The approved candidate menu and training window are enforced by the tools. Do not alter the plan. Prefer the lowest normalized_composite: arithmetic mean of losses normalized by public M0 expected-error scales, with failed folds at 8.0 and scores clipped at 8.0. Random_walk is not fixed to 1.0. Compare identical fold keys only. Missing scores are not evidence of superiority. Note only three folds and calibration limitations. You may choose tools adaptively but cannot invent estimates or use background knowledge about assets or past events. No text or event narratives are available. Do not claim official accuracy. Finish only after export_forecast succeeds.',
        final_answer_checks=[lambda answer,memory,agent:session.exported])
