"""Custom DSPy optimizer that rewrites solver instructions based on learned knowledge.

This acts as a meta-learning layer: after each batch of moves, the three reflection
agents analyze the game state. This optimizer then synthesizes their findings into
updated instructions for the solver agent."""

import dspy


class InstructionOptimizerSignature(dspy.Signature):
    """You are a meta-learning optimizer for an AI game-playing agent.
    Based on the game knowledge, REPL patterns, and visual observations
    accumulated so far, rewrite the solver agent's instructions to be
    more effective for this specific game.

    The instructions should be specific, actionable, and incorporate
    everything learned about:
    1. Game mechanics (how actions affect the grid)
    2. Level structure and goals
    3. Efficient REPL analysis patterns
    4. Common pitfalls to avoid
    5. Strategies that have worked or failed

    Write instructions as if you're briefing a new agent who has never
    seen this game before but needs to play it optimally."""

    game_knowledge_base: str = dspy.InputField(desc="Full game mechanics knowledge base")
    repl_knowledge_base: str = dspy.InputField(desc="Full REPL strategy knowledge base")
    visual_analysis: str = dspy.InputField(desc="Latest visual observer analysis")
    game_state: str = dspy.InputField(desc="Current game state summary")
    previous_instructions: str = dspy.InputField(desc="Previous solver instructions")
    performance_notes: str = dspy.InputField(desc="Notes on recent performance (levels completed, steps taken, patterns)")

    updated_instructions: str = dspy.OutputField(desc="Updated, improved instructions for the solver agent")
    key_changes: str = dspy.OutputField(desc="What changed in the instructions and why")


class InstructionOptimizer:
    """Rewrites solver instructions based on accumulated knowledge."""

    def __init__(self, lm: dspy.LM):
        self.lm = lm
        self.predict = dspy.ChainOfThought(InstructionOptimizerSignature)
        self.optimization_history: list[dict] = []

    def optimize(self, history, visual_obs: dict) -> str:
        """Generate optimized solver instructions based on current knowledge."""
        # Build performance notes
        perf_notes = self._build_performance_notes(history)

        with dspy.context(lm=self.lm):
            result = self.predict(
                game_knowledge_base=history.get_game_kb_text(),
                repl_knowledge_base=history.get_repl_kb_text(),
                visual_analysis=visual_obs.get("visual_analysis", "No analysis yet."),
                game_state=history.get_state_summary(),
                previous_instructions=history.solver_instructions or "No previous instructions.",
                performance_notes=perf_notes,
            )

        self.optimization_history.append({
            "step": history.current_step,
            "key_changes": result.key_changes,
        })

        return result.updated_instructions

    def _build_performance_notes(self, history) -> str:
        """Analyze recent performance to guide optimization."""
        notes = []
        notes.append(f"Total steps so far: {history.current_step}")
        notes.append(f"Levels completed: {history.levels_completed}/{history.total_levels}")

        if history.current_step > 0:
            # Check for stuck patterns (repeating actions without progress)
            recent = history.get_recent_actions(20)
            if len(set(recent)) <= 2 and len(recent) >= 10:
                notes.append("WARNING: Agent appears stuck in a loop with limited action diversity.")

            # Check level completion rate
            if history.current_step > 50 and history.levels_completed == 0:
                notes.append("WARNING: No levels completed after 50+ steps. May need to rethink strategy entirely.")

            # Track action distribution
            from collections import Counter
            action_counts = Counter(recent)
            notes.append(f"Recent action distribution: {dict(action_counts)}")

        if len(self.optimization_history) > 0:
            notes.append(f"Previous optimizations: {len(self.optimization_history)}")
            last = self.optimization_history[-1]
            notes.append(f"Last optimization changes: {last.get('key_changes', 'N/A')}")

        return "\n".join(notes)
