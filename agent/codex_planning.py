"""Codex tool-driven solver with staged scientific planning guidance."""
from __future__ import annotations

from typing import Any, Dict

from .codex import CodexAgent
from .planning import PLANNING_STAGES


CODEX_PLANNING_GUIDANCE = """# Staged Planning And Evidence
After reading the task materials and inspecting actual data with tools, write a
concise /workspace/PLAN.md covering the four stages below. This is a working
plan, not a separate text-only response: write it with a tool call and continue
implementing. Keep planning lightweight: spend at most 10% of remaining time,
capped at 2 minutes, on drafting the plan after the required reading. In base
and reference modes, do not delay the first baseline evaluation beyond the
existing five-minute target. In reproduce mode, understand the paper first and
preserve its methodology; planning is not permission to substitute a baseline.

{stages}

For each stage, distinguish supplied facts (with file/section references),
observations from actual tool execution, framework defaults, and assumptions.
List missing details and the cheapest check that could resolve them. Do not
claim to have read an unavailable paper or infer undocumented paper settings.
Update the plan when observations change it. Use installed scientific libraries
and a small number of modules rather than overengineering the architecture.

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


class PlanningCodexAgent(CodexAgent):
    """Retain Codex execution while adding a structured planning workflow."""

    def planning_guidance(self) -> str:
        stages = "\n".join(
            f"{index}. **{stage.title()}**: {instruction}"
            for index, (stage, instruction) in enumerate(PLANNING_STAGES, start=1)
        )
        return CODEX_PLANNING_GUIDANCE.format(stages=stages)

    def build_system_prompt(self, task: Dict[str, Any]) -> str:
        return super().build_system_prompt(task) + "\n\n" + self.planning_guidance()