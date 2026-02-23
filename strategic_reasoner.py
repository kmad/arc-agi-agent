"""Strategic Reasoner: RLM-powered knowledge synthesis for multi-agent orchestration.

Replaces the mechanical KnowledgeMerger with an RLM that reasons over agent reports,
detects conflicts/gaps, generates per-agent directives, and maintains growing reasoning
context across sync cycles.

Architecture:
    Reports → StrategicReasoner.reason() → per-agent directives → each agent
              ↑                                                      ↓
              └── growing ReasoningContext (persists across cycles) ──┘
"""

import json
import traceback
import dspy

from models import (
    KnowledgeEntry, REPLTip, AgentReport, AgentDirective,
    ReasoningContext, StrategicReasoningOutput,
    KnowledgeConflict, KnowledgeGap,
)


# ── DSPy Signature ───────────────────────────────────────────────

class StrategicReasoningSignature(dspy.Signature):
    """You are a strategic coordinator analyzing reports from multiple game-playing agents.

    Your job is to reason over their findings and produce targeted directives for each agent.

    Analyze the agent reports to:
    1. Identify CONFLICTS: Where agents disagree about game mechanics or strategies.
    2. Identify GAPS: What hasn't been explored yet? What hypotheses need testing?
    3. Synthesize CONFIRMED knowledge: What multiple agents agree on.
    4. Generate DIRECTIVES: Give each agent specific, differentiated instructions.
       - Don't send everyone the same thing. Specialize agents.
       - One agent might focus on exploring new actions, another on optimizing a known path.
       - Tell agents what to AVOID (things already proven ineffective).
    5. Track strategy EVOLUTION: How is the overall approach changing across cycles?

    Output directives_json as a JSON array of objects with fields:
    agent_id (str), focus_area (str), avoid (list[str]), try_actions (list[str]),
    updated_instructions (str)."""

    agent_reports: str = dspy.InputField(desc="Summary of all agent reports: knowledge, progress, strategies")
    reasoning_context: str = dspy.InputField(desc="Accumulated context from previous sync cycles")
    available_agents: str = dspy.InputField(desc="List of agent IDs and their current status/progress")

    strategy_notes: str = dspy.OutputField(desc="How the overall strategy is evolving. What's working, what's not.")
    directives_json: str = dspy.OutputField(desc='JSON array of AgentDirective objects, one per agent')


# ── Mechanical Merge (fallback) ──────────────────────────────────

def _mechanical_merge(reports: list[AgentReport]) -> tuple[list[KnowledgeEntry], list[REPLTip]]:
    """Fingerprint-based dedup merge. Used as fallback when RLM fails."""
    best_by_fp: dict[str, KnowledgeEntry] = {}
    for report in reports:
        for entry in report.knowledge:
            if not entry.fingerprint:
                entry.compute_fingerprint()
            fp = entry.fingerprint
            existing = best_by_fp.get(fp)
            if existing is None or entry.confidence > existing.confidence:
                best_by_fp[fp] = entry
    merged_knowledge = sorted(best_by_fp.values(), key=lambda e: e.confidence, reverse=True)

    seen_texts: set[str] = set()
    merged_tips: list[REPLTip] = []
    for report in reports:
        for tip in report.repl_tips:
            key = tip.text.strip().lower()
            if key not in seen_texts:
                seen_texts.add(key)
                merged_tips.append(tip)

    return merged_knowledge, merged_tips


# ── Strategic Reasoner ───────────────────────────────────────────

class StrategicReasoner:
    """RLM-powered strategic reasoning over multi-agent reports.

    Uses DSPy RLM to iteratively reason over agent reports, detect conflicts/gaps,
    and produce per-agent directives. Falls back to mechanical merge on failure.
    """

    def __init__(self, lm, sub_lm=None, max_iterations: int = 8):
        self.lm = lm
        self.sub_lm = sub_lm or lm
        self.context = ReasoningContext()
        self.max_iterations = max_iterations

        # RLM for deep reasoning (iterative REPL-style)
        self.rlm = dspy.RLM(
            StrategicReasoningSignature,
            max_iterations=max_iterations,
            sub_lm=self.sub_lm,
        )
        # Fast fallback (single-pass CoT)
        self.predict = dspy.Predict(StrategicReasoningSignature)

    def reason(self, reports: list[AgentReport], agent_ids: list[str]) -> StrategicReasoningOutput:
        """Run strategic reasoning over all agent reports.

        Returns a StrategicReasoningOutput with merged knowledge, per-agent directives,
        and strategy evolution notes.
        """
        if not reports:
            return StrategicReasoningOutput()

        # Build inputs
        reports_summary = self._build_reports_summary(reports)
        context_text = self._serialize_context()
        agents_text = self._build_agents_summary(reports, agent_ids)

        # Choose reasoning method: RLM for cycles > 1 (when we have context), Predict for first cycle
        use_rlm = self.context.cycle_number > 0

        try:
            if use_rlm:
                with dspy.context(lm=self.lm):
                    result = self.rlm(
                        agent_reports=reports_summary,
                        reasoning_context=context_text,
                        available_agents=agents_text,
                    )
            else:
                with dspy.context(lm=self.lm):
                    result = self.predict(
                        agent_reports=reports_summary,
                        reasoning_context=context_text,
                        available_agents=agents_text,
                    )

            # Parse directives from JSON output
            directives = self._parse_directives(result.directives_json, agent_ids)

            # Merge knowledge (mechanical dedup + RLM-resolved conflicts)
            merged_knowledge, merged_tips = _mechanical_merge(reports)

            # Inject directive-specific knowledge
            for directive in directives:
                # Give each agent the full merged knowledge (they'll dedup on their end)
                directive.knowledge_to_inject = list(merged_knowledge)

            output = StrategicReasoningOutput(
                merged_knowledge=merged_knowledge,
                merged_repl_tips=merged_tips,
                directives=directives,
                strategy_notes=result.strategy_notes,
            )

            # Update reasoning context
            self._update_context(reports, output)

            print(f"  [StrategicReasoner] Cycle {self.context.cycle_number}: "
                  f"{len(directives)} directives, {len(merged_knowledge)} knowledge entries")
            if result.strategy_notes:
                print(f"  [Strategy] {result.strategy_notes[:200]}")

            return output

        except Exception as e:
            print(f"  [StrategicReasoner] RLM failed, falling back to mechanical merge: {e}")
            traceback.print_exc()
            return self._fallback_merge(reports, agent_ids)

    def _fallback_merge(self, reports: list[AgentReport], agent_ids: list[str]) -> StrategicReasoningOutput:
        """Mechanical merge fallback — identical to old KnowledgeMerger behavior."""
        merged_knowledge, merged_tips = _mechanical_merge(reports)

        # No directives — all agents get the same blob (backward compat)
        self.context.cycle_number += 1

        return StrategicReasoningOutput(
            merged_knowledge=merged_knowledge,
            merged_repl_tips=merged_tips,
            directives=[],
            strategy_notes="(fallback: mechanical merge)",
        )

    def _build_reports_summary(self, reports: list[AgentReport]) -> str:
        """Build a text summary of all agent reports for the RLM."""
        sections = []

        for report in reports:
            lines = [
                f"=== Agent: {report.agent_id} ===",
                f"  Game: {report.game_id} | Status: {report.status.value}",
                f"  Steps: {report.steps_taken} | Levels: {report.levels_completed}/{report.total_levels}",
            ]

            # Top knowledge entries (by confidence)
            if report.knowledge:
                sorted_kb = sorted(report.knowledge, key=lambda e: e.confidence, reverse=True)
                lines.append(f"  Knowledge ({len(sorted_kb)} entries):")
                for entry in sorted_kb[:10]:
                    lines.append(f"    [{entry.confidence}%] [{entry.category.value}] {entry.text[:120]}")

            # Instructions snippet
            if report.instructions:
                lines.append(f"  Instructions: {report.instructions[:200]}...")

            sections.append("\n".join(lines))

        # Cross-agent analysis
        cross = ["\n=== Cross-Agent Analysis ==="]

        # Find common knowledge (entries with same fingerprint across agents)
        all_fps: dict[str, list[str]] = {}  # fingerprint -> [agent_ids]
        for report in reports:
            for entry in report.knowledge:
                if not entry.fingerprint:
                    entry.compute_fingerprint()
                fps = all_fps.setdefault(entry.fingerprint, [])
                if report.agent_id not in fps:
                    fps.append(report.agent_id)

        shared = {fp: agents for fp, agents in all_fps.items() if len(agents) > 1}
        if shared:
            cross.append(f"  Shared knowledge (found by {len(shared)} fingerprints across agents)")

        # Find unique knowledge per agent
        for report in reports:
            unique = [e for e in report.knowledge
                      if e.fingerprint and len(all_fps.get(e.fingerprint, [])) == 1]
            if unique:
                cross.append(f"  {report.agent_id}: {len(unique)} unique findings")

        sections.append("\n".join(cross))
        return "\n\n".join(sections)

    def _build_agents_summary(self, reports: list[AgentReport], agent_ids: list[str]) -> str:
        """Build a summary of available agents and their status."""
        lines = []
        report_map = {r.agent_id: r for r in reports}

        for aid in agent_ids:
            report = report_map.get(aid)
            if report:
                lines.append(f"{aid}: steps={report.steps_taken}, "
                           f"levels={report.levels_completed}/{report.total_levels}, "
                           f"status={report.status.value}, "
                           f"kb={len(report.knowledge)}")
            else:
                lines.append(f"{aid}: no report yet")

        return "\n".join(lines)

    def _serialize_context(self) -> str:
        """Serialize the reasoning context for the RLM input."""
        if self.context.cycle_number == 0:
            return "First sync cycle. No previous context."

        lines = [
            f"Cycle: {self.context.cycle_number}",
            f"Strategy evolution ({len(self.context.strategy_evolution)} cycles):",
        ]

        # Show last 3 strategy notes
        for i, note in enumerate(self.context.strategy_evolution[-3:]):
            lines.append(f"  Cycle {self.context.cycle_number - len(self.context.strategy_evolution[-3:]) + i + 1}: {note[:200]}")

        if self.context.confirmed_knowledge:
            lines.append(f"Confirmed knowledge: {len(self.context.confirmed_knowledge)} entries")
            for entry in self.context.confirmed_knowledge[:5]:
                lines.append(f"  [{entry.confidence}%] {entry.text[:100]}")

        if self.context.unresolved_conflicts:
            lines.append(f"Unresolved conflicts: {len(self.context.unresolved_conflicts)}")
            for conflict in self.context.unresolved_conflicts[:3]:
                lines.append(f"  A: {conflict.entry_a.text[:60]} vs B: {conflict.entry_b.text[:60]}")

        if self.context.open_gaps:
            lines.append(f"Open gaps: {len(self.context.open_gaps)}")
            for gap in self.context.open_gaps[:3]:
                lines.append(f"  [{gap.priority}] {gap.description[:80]}")

        if self.context.agent_performance:
            lines.append("Agent performance:")
            for aid, perf in self.context.agent_performance.items():
                lines.append(f"  {aid}: {perf}")

        return "\n".join(lines)

    def _parse_directives(self, directives_json: str, agent_ids: list[str]) -> list[AgentDirective]:
        """Parse directives from the RLM's JSON output."""
        try:
            # Handle common LLM output issues: strip markdown fences
            text = directives_json.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            raw = json.loads(text)
            if not isinstance(raw, list):
                raw = [raw]

            directives = []
            for item in raw:
                if isinstance(item, dict):
                    # Ensure agent_id is present
                    if "agent_id" not in item:
                        continue
                    # Strip knowledge_to_inject from parsed directives (we set it ourselves)
                    item.pop("knowledge_to_inject", None)
                    directives.append(AgentDirective.model_validate(item))

            return directives

        except (json.JSONDecodeError, Exception) as e:
            print(f"  [StrategicReasoner] Failed to parse directives JSON: {e}")
            # Return empty directives for each agent as fallback
            return [
                AgentDirective(agent_id=aid, focus_area="continue current approach")
                for aid in agent_ids
            ]

    def _update_context(self, reports: list[AgentReport], output: StrategicReasoningOutput):
        """Update the persistent reasoning context after a successful cycle."""
        self.context.cycle_number += 1

        # Append strategy notes
        if output.strategy_notes:
            self.context.strategy_evolution.append(output.strategy_notes)

        # Update confirmed knowledge (entries with confidence >= 80)
        high_conf = [e for e in output.merged_knowledge if e.confidence >= 80]
        existing_fps = {e.fingerprint for e in self.context.confirmed_knowledge}
        for entry in high_conf:
            if not entry.fingerprint:
                entry.compute_fingerprint()
            if entry.fingerprint not in existing_fps:
                self.context.confirmed_knowledge.append(entry)
                existing_fps.add(entry.fingerprint)

        # Update agent performance tracking
        for report in reports:
            prev = self.context.agent_performance.get(report.agent_id, {})
            prev_levels = prev.get("levels", 0)
            trend = "improving" if report.levels_completed > prev_levels else "stable"
            if report.levels_completed < prev_levels:
                trend = "regressing"
            self.context.agent_performance[report.agent_id] = {
                "levels": report.levels_completed,
                "steps": report.steps_taken,
                "kb_size": len(report.knowledge),
                "trend": trend,
            }

        # Track conflicts and gaps from output
        if output.conflicts_found:
            self.context.unresolved_conflicts = output.conflicts_found
        if output.gaps_found:
            self.context.open_gaps = output.gaps_found
