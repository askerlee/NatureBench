"""Codex tool-driven solver with staged scientific planning guidance."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .codex import CodexAgent
from .planning import PLANNING_STAGES


CODEX_PLANNING_GUIDANCE = """# Staged Planning And Evidence
After reading the task materials and inspecting actual data with tools, write a
concise /workspace/PLAN.md covering the four stages below. This is a working
plan, not a separate text-only response: write it with a tool call and continue
implementing. Keep planning lightweight: spend at most 10% of remaining time,
capped at 2 minutes, on drafting the plan after the required reading. In base
and reference modes, set an early task-appropriate baseline target using measured
data-processing and training costs, not a fixed deadline. In reproduce mode,
understand the paper first and preserve its methodology; planning is not
permission to substitute a baseline.

{stages}

For each stage, distinguish supplied facts (with file/section references),
observations from actual tool execution, framework defaults, and assumptions.
List missing details and the cheapest check that could resolve them. Do not
claim to have read an unavailable paper or infer undocumented paper settings.
Update the plan when observations change it. Use installed scientific libraries
and a small number of modules rather than overengineering the architecture.

# Method Selection Heuristics
Choose methods from the structure of the data and prediction target, not
only familiarity or implementation speed. Identify the relationships,
symmetries, noise processes, and domain shifts a method should capture.
Every submitted output must come from a method whose key predictive behavior is
trained, learned, fit, or optimized from the provided data. General-purpose
pretraining is a pipeline component, not a substitute for this requirement or
permission to retrieve benchmark answers. Use evaluation feedback to tune or
select fitted methods, never to reconstruct targets or tune output constants.

Deprioritize CPU-first method selection. CPU compatibility and cheap execution
are not sufficient reasons to rank a method as the main research candidate.
Prioritize structurally appropriate learned representations, graph models,
neural models, and permitted pretraining when scientifically justified and
feasible. Inspect accelerator availability and measure resource requirements;
do not assume an unknown environment requires CPU-only methods. Use inexpensive
classical methods primarily as baselines, diagnostics, or fallbacks unless
scientific reasoning or permitted validation evidence supports prioritizing
them. Do not equate GPU use with scientific quality, and retain the best
validated method regardless of hardware.

For spatial cell/spot data, consider whether neighborhood context and
spatial or molecular graphs are central to the target. Compare fixed
neighborhood aggregation with learned graph integration or partitioning;
do not assume a graph neural network is automatically superior.

For paired modalities, consider learned alignment and complementary
information rather than relying only on concatenation. For source-to-target
transfer, consider explicit domain adaptation. Assess permitted pretrained
representations and scientific resources before defaulting to handcrafted
features.

When applicable and feasible within the remaining budget, strengthen the
representation through additional self-supervised pretraining on permitted
data from the current dataset or a better-suited permitted pretrained model.
Prefer these improvements over a weaker representation when evidence supports
their suitability; do not treat larger models as inherently better. Verify
model provenance, split restrictions, and compute cost before use. Do not use
held-out data for pretraining unless explicitly allowed, and validate the gain
on permitted validation data. Budget representation experiments using measured
costs rather than sacrificing methodological suitability for an arbitrary deadline.

In base and reference modes, distinguish baselines from research candidates.
Baselines establish correct execution and provide performance references;
they are a role, not a model family. Research candidates test concrete
hypotheses about limitations of current approaches using methods suited to
the data and target. State what important structure each approach omits.
Set the first-evaluation target from measured feasibility and record the reason.

Explore multiple candidates as time permits; do not restrict the workflow to
one baseline and one upgrade. Prioritize experiments by expected scientific
value, measured feasibility, and remaining budget. Before rejecting a candidate
on resource grounds, run a small setup and throughput check when affordable,
including training or embedding extraction and inference costs. Do not reject
a candidate solely because it is unfamiliar or more complex, or prefer it
solely because it is neural or pretrained.

Give promising candidates sufficient training to support a meaningful comparison;
do not treat an undertrained model or a repairable implementation failure as
evidence against the method. Before launching an experiment, account for setup,
training, inference, output checks, and evaluation, and reserve a completion
margin. When no new experiment fits, validate and preserve the best evaluated
solution instead of repeatedly evaluating unchanged outputs.

Select the best validated approach or justified ensemble, whether baseline or
research candidate. Record the evidence for retaining or rejecting candidates;
do not claim superiority from architecture or implementation convenience alone.

In reproduce mode, these heuristics guide diagnosis and implementation
choices within the paper's methodology; they do not authorize replacing it.

# Validation And Feedback
Before each evaluation, execute local output-contract checks derived from the
task documentation: required instance directories and filenames, shapes,
dtypes, finite values, and allowed value ranges where specified. Ensure outputs
are from the current execution, not stale files. Check data-split separation and
fit learned preprocessing only on the permitted training data. Never inspect
hidden evaluators or use test labels to design these checks.

Classify observed failures before changing code: implementation errors,
output-contract errors, or valid execution with poor scientific performance.
Repair execution/format errors first. For performance changes, state a concrete
hypothesis and use permitted validation and evaluation feedback to test it.
In reproduce mode, diagnose mismatches against the paper rather than switching
methods, and document unavoidable deviations in /workspace/DEVIATIONS.md.

# Experiment History And Continuation
After each execution/evaluation, append one JSON object to
/workspace/experiments.jsonl using a JSON serializer. Record the approach,
configuration and seed, changed files, execution outcome, validation outcome,
evaluation attempt and scores (or the observed error), remaining time,
hypothesis, and next action. Never fabricate results or record credentials or
evaluation tokens. Preserve the best evaluated source/configuration in a
workspace-local snapshot so later experiments cannot lose it; record its path.
The service still tracks the best score; there is no new submission step.

On resume, read PLAN.md and experiments.jsonl if present, inspect the current
workspace, check remaining time and the service's best score, and continue the
existing experiment rather than restarting planning. These artifacts supplement
the Codex tool transcript; they do not replace concrete tool execution.
Keep using Codex tools to inspect data, implement, debug, execute, and evaluate.
There is no fixed attempt limit: retain the parent prompt's time-budget and
scientific-validity rules, including its stopping criteria.
"""


class CodexPlanningAgent(CodexAgent):
    """Retain Codex execution while adding a structured planning workflow."""

    def __init__(
        self,
        model_name: str,
        mode: str = "base",
        *,
        plan_only: bool = False,
        backend_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_name, mode=mode)
        self.plan_only = plan_only
        if plan_only:
            from .planning import PlanningAgent

            self._plan_only_planner = PlanningAgent(
                model_name, mode=mode, backend_kwargs=backend_kwargs,
            )
            self.backend = self._plan_only_planner.backend

    def planning_guidance(self) -> str:
        stages = "\n".join(
            f"{index}. **{stage.title()}**: {instruction}"
            for index, (stage, instruction) in enumerate(PLANNING_STAGES, start=1)
        )
        return CODEX_PLANNING_GUIDANCE.format(stages=stages)

    def build_system_prompt(self, task: Dict[str, Any]) -> str:
        if self.plan_only:
            heuristics = CODEX_PLANNING_GUIDANCE.split(
                "# Method Selection Heuristics\n", 1,
            )[1].split("# Validation And Feedback", 1)[0]
            return (
                self._plan_only_planner.build_system_prompt(task)
                + "\n\n# Method Selection Heuristics\n" + heuristics
                + "\n# Planning-Only Scope\n"
                "Produce planning prose only. Describe feasibility, hardware, data, "
                "and validation checks for later execution; do not claim they ran. "
                "Do not write implementation files, run tools, or contact an evaluation "
                "service. Execution deadlines and experiment recording apply to the "
                "future solver run, not this planning session."
            )
        return super().build_system_prompt(task) + "\n\n" + self.planning_guidance()

    def call_model(self, messages, **kwargs):
        if not self.plan_only:
            raise ValueError("API model calls require plan_only=True.")
        return self._plan_only_planner.call_model(messages, **kwargs)

    def solve_task(self, task: Dict[str, Any], llm_kwargs=None) -> Dict[str, Any]:
        if self.plan_only:
            raise ValueError("Planning-only agents cannot execute a solution.")
        return super().solve_task(task, llm_kwargs)