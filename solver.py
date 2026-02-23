"""Phase-aware solver using RLM for scientific game discovery.

Implements an explore → hypothesize → iterate → execute → reflect loop.
The main LLM synthesizes and plans. The sub-LLM (via RLM) runs parallel
exploratory episodes in the solution space.

Phases:
  EXPLORE    - Try every action, observe effects, build action-effect map
  HYPOTHESIZE - Synthesize observations into game mechanic hypotheses
  ITERATE    - Test hypotheses with targeted experiments
  EXECUTE    - Plan and run goal-directed action sequences
  REFLECT    - (handled externally by agent.py's reflection cycle)
"""

import dspy
import json
import numpy as np
from enum import Enum
from typing import Optional
from game_state import GameHistory, frame_to_text, compute_grid_diff, analyze_frame
from actions import format_action_space, get_valid_action_names
from models import GameHypothesis, ActionEffect, FrameAnalysis, VisualAnalysis


class SolverPhase(Enum):
    EXPLORE = "explore"
    HYPOTHESIZE = "hypothesize"
    ITERATE = "iterate"
    EXECUTE = "execute"


# ── Phase-specific RLM signatures ──────────────────────────────────

class ExploreSignature(dspy.Signature):
    """You are systematically exploring an unknown puzzle game on a 64x64 grid.

    Your ONLY goal right now is to discover what each action does.
    Look at the 'Movement' line in the frame diff to determine direction.
    For each available action, determine:
    - DIRECTION: Does it move something UP, DOWN, LEFT, or RIGHT?
    - MAGNITUDE: How many pixels does it shift?
    - RANGE: Which rows/columns are affected?
    - BLOCKED: Does it produce tiny/no change? (means it hit a wall)

    IMPORTANT: The 'Movement' field in the analysis tells you the DIRECTION
    of the last action's effect. Compare across actions to build a direction map.

    Design an exploration sequence that tests EVERY available action.
    Output a batch of actions as a JSON list."""

    frame_analysis: str = dspy.InputField(desc="Programmatic analysis with Movement direction for last action")
    game_state: str = dspy.InputField(desc="Game state: step count, levels, recent actions")
    action_effects_so_far: str = dspy.InputField(desc="What we know about each action's effect so far, including movement direction")
    available_actions: str = dspy.InputField(desc="All available actions in this game")

    exploration_plan: str = dspy.OutputField(desc="Direction map so far: which action = which direction? What's still unknown?")
    actions: list[str] = dspy.OutputField(desc='List of 4-8 actions. e.g. ["ACTION1","ACTION2","ACTION3","ACTION4"]')


class HypothesizeSignature(dspy.Signature):
    """You are a scientist analyzing experimental results from an unknown puzzle game.

    Given the action-effect observations collected during exploration, form hypotheses:
    1. MOVEMENT: Which actions move an object? Direction and magnitude?
    2. OBJECTS: What are the distinct objects on the grid? (color clusters, shapes)
    3. GOAL: What is the objective? (reach a target, clear blocks, match patterns)
    4. RESOURCES: Is there a timer, move limit, or resource bar?
    5. RULES: Are there walls, boundaries, or conditional behaviors?

    Be specific. Use grid coordinates and color values.
    Rate each hypothesis by confidence (0-100).

    Also output the NEXT set of actions to begin testing your strongest hypothesis."""

    frame_analysis: str = dspy.InputField(desc="Programmatic analysis: objects, positions, sizes, diff from previous frame")
    game_state: str = dspy.InputField(desc="Game state summary")
    exploration_log: str = dspy.InputField(desc="Full log of actions taken and their pixel-change effects")
    available_actions: str = dspy.InputField(desc="Available actions")

    hypotheses: list[GameHypothesis] = dspy.OutputField(desc="List of hypotheses about game mechanics")
    actions: list[str] = dspy.OutputField(desc='List of 4-8 actions to begin testing the top hypothesis')


class IterateSignature(dspy.Signature):
    """You are testing specific hypotheses about an unknown puzzle game.

    You have hypotheses about game mechanics. Design TARGETED experiments:
    - To CONFIRM a hypothesis: predict what should happen, then test it
    - To REFUTE a hypothesis: find a case where the prediction fails
    - To REFINE a hypothesis: test edge cases and boundary conditions

    Compare actual results with predictions. Update confidence levels.

    Once a hypothesis reaches 90%+ confidence, it becomes confirmed knowledge.
    Once enough mechanics are confirmed, recommend switching to EXECUTE phase."""

    frame_analysis: str = dspy.InputField(desc="Programmatic analysis: objects, positions, diffs")
    game_state: str = dspy.InputField(desc="Game state summary")
    hypotheses: str = dspy.InputField(desc="Current hypotheses with confidence levels")
    recent_test_results: str = dspy.InputField(desc="Recent action results vs predictions")
    available_actions: str = dspy.InputField(desc="Available actions")

    updated_hypotheses: list[GameHypothesis] = dspy.OutputField(desc="Updated hypotheses with new confidence levels after testing")
    phase_recommendation: str = dspy.OutputField(desc='Either "keep_iterating" or "ready_to_execute" with reason')
    actions: list[str] = dspy.OutputField(desc='List of 4-8 targeted test actions')


class ExecuteSignature(dspy.Signature):
    """You are playing a puzzle game with CONFIRMED understanding of mechanics.

    You know what the actions do, what the objects are, and what the goal is.
    Now plan and execute efficient action sequences to complete the level.

    CRITICAL RULES:
    1. Look at the 'action_effectiveness' data — if an action is showing tiny/no pixel changes,
       it is BLOCKED and you must stop using it. Switch to a different action.
    2. Different actions may affect different parts of the grid. When one group of objects
       reaches a boundary, only actions that affect the REMAINING objects will make progress.
    3. Check the frame_analysis for object positions. If an object's column is near the
       right edge (>50), it may be close to a boundary or target.
    4. The goal is typically to move a specific object (e.g., a colored bar) to reach
       a target position (e.g., a cyan marker).

    Be EFFICIENT. Every wasted action may cost a resource.
    If an action suddenly produces tiny changes, STOP using it immediately."""

    frame_analysis: str = dspy.InputField(desc="Programmatic analysis: objects, positions, confirmed mechanics context")
    game_state: str = dspy.InputField(desc="Game state with level info and knowledge")
    confirmed_mechanics: str = dspy.InputField(desc="Confirmed game mechanics and rules, including which actions are BLOCKED")
    visual_analysis: str = dspy.InputField(desc="Visual observer's spatial analysis")
    solver_directives: str = dspy.InputField(desc="Dynamic solver directives from optimizer")
    available_actions: str = dspy.InputField(desc="Available actions")

    plan: str = dspy.OutputField(desc="Step-by-step plan: which actions are working, which are blocked, what to do next")
    actions: list[str] = dspy.OutputField(desc='List of 3-8 precise goal-directed actions. NEVER use actions that are BLOCKED.')


# ── Phase-aware Solver ─────────────────────────────────────────────

class Solver:
    """Phase-aware solver: explore → hypothesize → iterate → execute.

    Uses RLM (with sub-LLM) for explore/hypothesize/iterate phases where
    the sub-LLM can explore multiple candidate solutions in parallel.
    Uses fast CoT for the execute phase where we need speed.
    """

    # Phase transition thresholds (counted in BATCHES, not individual actions)
    EXPLORE_MIN_BATCHES = 2      # Minimum batches before leaving EXPLORE
    EXPLORE_MAX_BATCHES = 6      # Force transition to HYPOTHESIZE
    ITERATE_MAX_ROUNDS = 3       # Max hypothesis test rounds before EXECUTE
    HIGH_CONFIDENCE = 80         # Hypothesis confidence to count as "confirmed"
    MECHANICS_TO_EXECUTE = 1     # Confirmed mechanics needed to enter EXECUTE

    def __init__(self, lm: dspy.LM, sub_lm: Optional[dspy.LM] = None,
                 available_actions: Optional[list[str]] = None):
        self.lm = lm
        self.sub_lm = sub_lm or lm
        self.available_actions = available_actions or ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

        # Phase state
        self.phase = SolverPhase.EXPLORE
        self.phase_step_count = 0  # Steps within current phase
        self.iterate_rounds = 0

        # Knowledge accumulated across phases (typed)
        self.action_effects: dict[str, list[str]] = {a: [] for a in self.available_actions}
        self.action_effectiveness: dict[str, dict] = {}  # action -> {productive, tiny, total}
        self.action_directions: dict[str, list[str]] = {}  # action -> observed directions
        self.hypotheses: list[GameHypothesis] = []
        self.confirmed_mechanics: list[GameHypothesis] = []

        # RLM modules for discovery phases (sub-LLM explores solution space)
        # Keep iterations low for speed — each iteration = full LLM call
        self.explorer = dspy.RLM(
            ExploreSignature,
            max_iterations=3,
            sub_lm=self.sub_lm,
        )
        self.hypothesizer = dspy.RLM(
            HypothesizeSignature,
            max_iterations=4,
            sub_lm=self.sub_lm,
        )
        self.iterator = dspy.RLM(
            IterateSignature,
            max_iterations=3,
            sub_lm=self.sub_lm,
        )
        # Fast CoT for execution phase (speed matters)
        self.executor = dspy.ChainOfThought(ExecuteSignature)

        self.call_count = 0

    def solve_step(self, history: GameHistory, visual_obs: dict, deep: bool = False) -> list[str]:
        """Route to the appropriate phase handler."""
        self.call_count += 1
        self.phase_step_count += 1
        current_frame = history.last_frame

        if current_frame is None:
            return list(self.available_actions)

        # Check if stuck - force exploration regardless of phase
        stuck, stuck_msg = history.is_stuck()
        if stuck:
            # Reset to explore if stuck during execute
            if self.phase == SolverPhase.EXECUTE:
                print(f"  [Phase] Stuck during EXECUTE, reverting to EXPLORE")
                self._transition_to(SolverPhase.EXPLORE)
            return self._forced_exploration(history)

        # Auto-advance phase based on step counts
        self._check_phase_transitions(history)

        # Record action effects from recent history
        self._update_action_effects(history)

        print(f"  [Phase] {self.phase.value.upper()} (step {self.phase_step_count})")

        with dspy.context(lm=self.lm):
            if self.phase == SolverPhase.EXPLORE:
                actions = self._explore(history)
            elif self.phase == SolverPhase.HYPOTHESIZE:
                actions = self._hypothesize(history)
            elif self.phase == SolverPhase.ITERATE:
                actions = self._iterate(history)
            else:  # EXECUTE
                actions = self._execute(history, visual_obs)

        return actions

    # ── Phase handlers ──

    def _explore(self, history: GameHistory) -> list[str]:
        """EXPLORE: Systematically try every action, maximize information gain."""
        # First batch: deterministic — try each action once to get baseline effects
        untested = [a for a in self.available_actions if not self.action_effects.get(a)]
        if untested:
            # Try untested actions first, then repeat each action to confirm
            batch = list(untested)
            # Add a second round to see if effects are consistent
            batch.extend(untested)
            return batch[:12]

        # Subsequent batches: use LLM to design experiments based on observations
        effects_text = self._format_action_effects()
        actions_desc = ", ".join(self.available_actions)
        prev_frame = history.steps[-2].frame if len(history.steps) >= 2 else None
        analysis = analyze_frame(history.last_frame, prev_frame)

        result = self.explorer(
            frame_analysis=analysis,
            game_state=history.get_state_summary(),
            action_effects_so_far=effects_text,
            available_actions=actions_desc,
        )

        actions = self._validate_actions(result.actions)

        # Ensure exploration diversity: if parsed actions lack variety, force it
        if len(set(actions)) < min(3, len(self.available_actions)):
            actions = self._diverse_exploration_batch()

        return actions

    def _hypothesize(self, history: GameHistory) -> list[str]:
        """HYPOTHESIZE: Synthesize observations into testable hypotheses."""
        exploration_log = history.get_recent_action_diffs(30)
        actions_desc = ", ".join(self.available_actions)
        prev_frame = history.steps[-2].frame if len(history.steps) >= 2 else None
        analysis = analyze_frame(history.last_frame, prev_frame)

        # Include action effects summary and any existing solver instructions
        enriched_log = self._format_action_effects() + "\n\n" + exploration_log
        if history.solver_instructions:
            enriched_log += f"\n\nOptimizer notes:\n{history.solver_instructions[:500]}"

        result = self.hypothesizer(
            frame_analysis=analysis,
            game_state=history.get_state_summary(),
            exploration_log=enriched_log,
            available_actions=actions_desc,
        )

        # DSPy parses Pydantic natively — result.hypotheses is list[GameHypothesis]
        if result.hypotheses and isinstance(result.hypotheses, list):
            self.hypotheses = self._coerce_hypotheses(result.hypotheses)
        else:
            # Fallback: build hypotheses from action effects programmatically
            self.hypotheses = self._build_fallback_hypotheses()
            print(f"  [Hypothesize] Built {len(self.hypotheses)} fallback hypotheses from action effects")

        if self.hypotheses:
            print(f"  [Hypothesize] {len(self.hypotheses)} hypotheses formed:")
            for h in self.hypotheses[:3]:
                print(f"    [{h.confidence}%] {h.mechanic}: {h.description[:70]}")

        # Transition to ITERATE after forming hypotheses
        self._transition_to(SolverPhase.ITERATE)

        return self._validate_actions(result.actions)

    def _iterate(self, history: GameHistory) -> list[str]:
        """ITERATE: Test hypotheses with targeted experiments."""
        self.iterate_rounds += 1
        actions_desc = ", ".join(self.available_actions)

        hypotheses_text = self._format_hypotheses_text()
        recent_results = history.get_recent_action_diffs(15)

        prev_frame = history.steps[-2].frame if len(history.steps) >= 2 else None
        analysis = analyze_frame(history.last_frame, prev_frame)

        result = self.iterator(
            frame_analysis=analysis,
            game_state=history.get_state_summary(),
            hypotheses=hypotheses_text,
            recent_test_results=recent_results,
            available_actions=actions_desc,
        )

        # DSPy parses Pydantic natively — result.updated_hypotheses is list[GameHypothesis]
        if result.updated_hypotheses and isinstance(result.updated_hypotheses, list):
            self.hypotheses = self._coerce_hypotheses(result.updated_hypotheses)
            # Promote high-confidence hypotheses to confirmed mechanics
            for h in self.hypotheses:
                if h.confidence >= self.HIGH_CONFIDENCE:
                    if not any(cm.mechanic == h.mechanic for cm in self.confirmed_mechanics):
                        h.confirmed = True
                        self.confirmed_mechanics.append(h)
                        print(f"  [Iterate] CONFIRMED: {h.mechanic}: {h.description[:60]}")

        # Check if ready to execute
        phase_rec = result.phase_recommendation if hasattr(result, 'phase_recommendation') else ""
        if "ready_to_execute" in phase_rec.lower() or len(self.confirmed_mechanics) >= self.MECHANICS_TO_EXECUTE:
            print(f"  [Iterate] {len(self.confirmed_mechanics)} mechanics confirmed, transitioning to EXECUTE")
            self._transition_to(SolverPhase.EXECUTE)

        return self._validate_actions(result.actions)

    def _execute(self, history: GameHistory, visual_obs: dict) -> list[str]:
        """EXECUTE: Fast goal-directed play using confirmed mechanics."""
        # Determine which actions are currently effective
        effective_actions = self._get_effective_actions()
        actions_desc = ", ".join(effective_actions) if effective_actions else ", ".join(self.available_actions)

        visual_text = visual_obs.get("visual_analysis", "No visual analysis.")
        if "recommended_strategy" in visual_obs:
            visual_text += f"\nStrategy: {visual_obs['recommended_strategy']}"

        solver_directives = history.solver_instructions or "Execute efficiently using confirmed mechanics."

        # Combine confirmed mechanics with optimizer instructions (which may be more detailed)
        mechanics_text = self._format_confirmed_mechanics()
        if history.solver_instructions:
            mechanics_text += f"\n\n--- Optimizer Strategy ---\n{history.solver_instructions[:800]}"

        prev_frame = history.steps[-2].frame if len(history.steps) >= 2 else None
        analysis = analyze_frame(history.last_frame, prev_frame)

        result = self.executor(
            frame_analysis=analysis,
            game_state=history.get_state_summary(),
            confirmed_mechanics=mechanics_text,
            visual_analysis=visual_text,
            solver_directives=solver_directives,
            available_actions=actions_desc,
        )

        actions = self._validate_actions(result.actions)

        # Post-filter: remove blocked actions from the batch
        if effective_actions and len(effective_actions) < len(self.available_actions):
            blocked = set(self.available_actions) - set(effective_actions)
            filtered = [a for a in actions if a not in blocked]
            if filtered:
                print(f"  [Execute] Filtered out blocked actions {blocked}, using: {filtered}")
                actions = filtered
            else:
                # All actions were blocked ones — fall back to effective actions
                actions = list(effective_actions) * 2
                print(f"  [Execute] All proposed actions blocked, using effective: {actions}")

        return actions

    # ── Phase transition logic ──

    def _check_phase_transitions(self, history: GameHistory):
        """Auto-advance phases based on batch counts and knowledge.

        phase_step_count tracks BATCHES (solve_step calls), not individual actions.
        Each batch typically produces 4-8 actions.
        """
        if self.phase == SolverPhase.EXPLORE:
            # Check if we have at least 1 observed effect for each action
            actions_with_data = sum(1 for effects in self.action_effects.values() if len(effects) >= 1)
            all_tested = actions_with_data >= len(self.available_actions)

            if self.phase_step_count >= self.EXPLORE_MAX_BATCHES:
                print(f"  [Phase] EXPLORE max batches reached, forcing HYPOTHESIZE")
                self._transition_to(SolverPhase.HYPOTHESIZE)
            elif self.phase_step_count >= self.EXPLORE_MIN_BATCHES and all_tested:
                print(f"  [Phase] All {actions_with_data} actions tested after {self.phase_step_count} batches, moving to HYPOTHESIZE")
                self._transition_to(SolverPhase.HYPOTHESIZE)

        elif self.phase == SolverPhase.ITERATE:
            if self.iterate_rounds >= self.ITERATE_MAX_ROUNDS:
                print(f"  [Phase] Max iterate rounds reached, forcing EXECUTE")
                self._transition_to(SolverPhase.EXECUTE)

    def _transition_to(self, new_phase: SolverPhase):
        """Transition to a new phase."""
        old_phase = self.phase
        self.phase = new_phase
        self.phase_step_count = 0
        if new_phase == SolverPhase.ITERATE:
            self.iterate_rounds = 0
        print(f"  [Phase] {old_phase.value} -> {new_phase.value}")

    def on_level_transition(self):
        """Called by agent.py when a level transition is detected.
        Reset to EXPLORE for the new level."""
        print(f"  [Phase] Level transition detected, resetting to EXPLORE")
        self._transition_to(SolverPhase.EXPLORE)
        self.action_effects = {a: [] for a in self.available_actions}
        self.action_effectiveness = {}
        self.action_directions = {}
        self.hypotheses = []
        # Keep confirmed_mechanics — some may carry over between levels

    # ── Knowledge tracking ──

    def _update_action_effects(self, history: GameHistory):
        """Extract action-effect pairs from recent history with rich diff info.
        Also track movement directions per action."""
        import re
        for i, step in enumerate(history.steps[-20:]):
            if step.action == "RESET" or not step.grid_diff:
                continue
            action = step.action.split("(")[0]  # Strip coords from ACTION6(x,y)
            if action not in self.action_effects:
                continue

            # Compute direction from frame diff if we have two consecutive frames
            direction_info = ""
            step_idx = len(history.steps) - 20 + i
            if step_idx > 0 and step_idx < len(history.steps):
                prev_step = history.steps[step_idx - 1]
                if prev_step.frame is not None and step.frame is not None:
                    prev_f = prev_step.frame
                    curr_f = step.frame
                    diff_mask = prev_f != curr_f
                    if diff_mask.any():
                        from config import COLOR_MAP
                        bg = np.bincount(prev_f.flatten()).argmax()
                        new_pos = np.argwhere((prev_f == bg) & (curr_f != bg))
                        old_pos = np.argwhere((prev_f != bg) & (curr_f == bg))
                        if len(new_pos) > 0 and len(old_pos) > 0:
                            nc = new_pos.mean(axis=0)
                            oc = old_pos.mean(axis=0)
                            dr, dc = nc[0] - oc[0], nc[1] - oc[1]
                            dirs = []
                            if abs(dr) > 1: dirs.append("DOWN" if dr > 0 else "UP")
                            if abs(dc) > 1: dirs.append("RIGHT" if dc > 0 else "LEFT")
                            if dirs:
                                direction_info = f" [DIRECTION: {'+'.join(dirs)}]"
                                # Track direction per action
                                dir_key = "+".join(dirs)
                                if action not in self.action_directions:
                                    self.action_directions[action] = []
                                if dir_key not in self.action_directions[action]:
                                    self.action_directions[action].append(dir_key)

            # Build a richer effect description
            effect = f"step{step.step_number}: {step.grid_diff[:120]}{direction_info}"

            # Keep only last 3 effects per action to stay current
            effects = self.action_effects[action]
            if len(effects) >= 3:
                effects.pop(0)
            if not any(step.grid_diff[:80] in e for e in effects):
                effects.append(effect)

        # Update action effectiveness tracking — detect when actions become ineffective
        self._update_action_effectiveness(history)

    def _update_action_effectiveness(self, history: GameHistory):
        """Track which actions are producing useful changes vs. being blocked.
        Recalculates from scratch each time using the last 15 steps."""
        import re
        self.action_effectiveness = {}
        recent = history.steps[-15:]
        for step in recent:
            if step.action == "RESET" or not step.grid_diff:
                continue
            action = step.action.split("(")[0]
            px_match = re.search(r"(\d+) pixels changed", step.grid_diff)
            px = int(px_match.group(1)) if px_match else 0

            if action not in self.action_effectiveness:
                self.action_effectiveness[action] = {"productive": 0, "tiny": 0, "total": 0}

            stats = self.action_effectiveness[action]
            stats["total"] += 1
            if px <= 4:
                stats["tiny"] += 1
            elif px < 1000:  # Normal movement, not a reset
                stats["productive"] += 1

    def _format_action_effects(self) -> str:
        """Format known action effects for prompts, including direction and effectiveness."""
        lines = []

        # Direction summary at top (most useful info)
        if self.action_directions:
            lines.append("=== ACTION DIRECTION MAP ===")
            for action in sorted(self.action_directions.keys()):
                dirs = self.action_directions[action]
                lines.append(f"  {action} -> {', '.join(dirs)}")
            lines.append("")

        for action in sorted(self.action_effects.keys()):
            effects = self.action_effects[action]
            dir_str = f" [DIR: {','.join(self.action_directions.get(action, ['?']))}]" if action in self.action_directions else ""
            if effects:
                lines.append(f"{action}{dir_str}: {'; '.join(effects)}")
            else:
                lines.append(f"{action}: NOT YET TESTED")

        # Add effectiveness warnings
        warnings = []
        for action, stats in self.action_effectiveness.items():
            if stats["total"] >= 3 and stats["tiny"] > stats["productive"]:
                warnings.append(f"WARNING: {action} is mostly producing tiny/no changes — it may be BLOCKED")
        if warnings:
            lines.append("\n" + "\n".join(warnings))

        return "\n".join(lines)

    def _format_confirmed_mechanics(self) -> str:
        """Format confirmed mechanics for the execute phase, including effectiveness data."""
        lines = []

        if self.confirmed_mechanics:
            lines.append("Confirmed mechanics:")
            for m in self.confirmed_mechanics:
                lines.append(f"  [{m.confidence}%] {m.mechanic}: {m.description}")
        elif self.hypotheses:
            lines.append("Unconfirmed hypotheses:")
            for h in self.hypotheses:
                lines.append(f"  [{h.confidence}%] {h.mechanic}: {h.description}")
        else:
            lines.append("No mechanics confirmed yet. Explore and observe.")

        # Add action effectiveness data
        if self.action_effectiveness:
            lines.append("\nAction effectiveness (recent):")
            for action in sorted(self.action_effectiveness.keys()):
                stats = self.action_effectiveness[action]
                status = "WORKING" if stats["productive"] > stats["tiny"] else "BLOCKED/WEAK"
                lines.append(f"  {action}: {stats['productive']} productive, {stats['tiny']} tiny/blocked out of {stats['total']} total -> {status}")

        return "\n".join(lines)

    def _format_hypotheses_text(self) -> str:
        """Format hypotheses as text for the iterate signature input."""
        if not self.hypotheses:
            return "No hypotheses yet."
        items = []
        for h in self.hypotheses:
            items.append(h.model_dump())
        return json.dumps(items, indent=2)

    def _get_effective_actions(self) -> list[str]:
        """Return actions that are currently producing useful changes.
        If we don't have enough data, return all actions."""
        if not self.action_effectiveness:
            return list(self.available_actions)

        effective = []
        for action in self.available_actions:
            stats = self.action_effectiveness.get(action)
            if not stats or stats["total"] < 2:
                # Not enough data — assume it works
                effective.append(action)
            elif stats["productive"] > 0:
                # At least some productive results
                effective.append(action)
            # else: action has been tested and is consistently tiny/blocked

        return effective if effective else list(self.available_actions)

    # ── Utility methods ──

    def _coerce_hypotheses(self, raw: list) -> list[GameHypothesis]:
        """Coerce a list of mixed dicts/GameHypothesis into list[GameHypothesis]."""
        result = []
        for item in raw:
            if isinstance(item, GameHypothesis):
                result.append(item)
            elif isinstance(item, dict):
                try:
                    result.append(GameHypothesis.model_validate(item))
                except Exception:
                    # Best-effort: extract what we can
                    result.append(GameHypothesis(
                        mechanic=str(item.get("mechanic", "unknown")),
                        description=str(item.get("description", "")),
                        confidence=int(item.get("confidence", 50)),
                        test=str(item.get("test", "")),
                    ))
        return result

    def _build_fallback_hypotheses(self) -> list[GameHypothesis]:
        """Build hypotheses by analyzing action effects programmatically."""
        import re
        hypotheses = []

        # Group actions by their effect regions
        row_groups = {}  # row_range -> list of actions
        for action, effects in self.action_effects.items():
            if not effects:
                continue
            # Extract row ranges from effects
            for e in effects:
                m = re.search(r"rows (\d+)-(\d+)", e)
                if m:
                    row_range = f"rows {m.group(1)}-{m.group(2)}"
                    row_groups.setdefault(row_range, []).append(action)
                    break

            hypotheses.append(GameHypothesis(
                mechanic=f"{action}_movement",
                description=f"{action} shifts objects in {effects[-1][:120]}",
                confidence=70,
                test=f"Repeat {action} and check column progression",
            ))

        # Add grouping hypothesis if actions affect different row ranges
        if len(row_groups) > 1:
            groups_desc = "; ".join(f"{k}: {', '.join(set(v))}" for k, v in row_groups.items())
            hypotheses.insert(0, GameHypothesis(
                mechanic="action_row_groups",
                description=f"Actions affect DIFFERENT row ranges: {groups_desc}",
                confidence=85,
                test="Compare row ranges when different actions are at same column position",
            ))

        return hypotheses

    def _diverse_exploration_batch(self) -> list[str]:
        """Generate a batch that tests each action at least twice."""
        batch = []
        for action in self.available_actions:
            batch.extend([action] * 2)
        # Shuffle to avoid positional bias
        import random
        random.shuffle(batch)
        return batch[:12]

    def _forced_exploration(self, history: GameHistory) -> list[str]:
        """When stuck, systematically try all directions to find which ones work."""
        recent = history.get_recent_actions(10)
        from collections import Counter
        stuck_action = Counter(recent).most_common(1)[0][0]

        other_actions = [a for a in self.available_actions if a != stuck_action]
        explore = []
        for action in other_actions:
            explore.extend([action] * 3)
        return explore

    def _inject_exploration(self, actions: list[str]) -> list[str]:
        """If solver outputs all-same actions, inject diversity."""
        base = actions[0]
        others = [a for a in self.available_actions if a != base]
        result = list(actions)
        for i, other in enumerate(others):
            if i + 1 < len(result):
                result[i + 1] = other
        return result

    def _validate_actions(self, actions_output) -> list[str]:
        """Validate and filter actions from DSPy typed output.

        With typed signatures, actions_output should already be list[str].
        Falls back to parsing if DSPy returns a string instead.
        """
        valid_actions = set(self.available_actions)

        # If DSPy parsed it correctly as a list
        if isinstance(actions_output, list):
            result = []
            for a in actions_output:
                clean = str(a).strip().upper()
                if clean in valid_actions:
                    result.append(clean)
            if result:
                return result

        # Fallback: parse from string (handles JSON strings, comma-separated, etc.)
        actions_str = str(actions_output)
        try:
            parsed = json.loads(actions_str)
            if isinstance(parsed, list):
                result = [str(a).strip().upper() for a in parsed if str(a).strip().upper() in valid_actions]
                if result:
                    return result
        except (json.JSONDecodeError, TypeError):
            pass

        # Last resort: extract action names from text
        actions = []
        for token in actions_str.upper().replace(",", " ").replace('"', " ").replace("[", " ").replace("]", " ").split():
            clean = token.strip()
            if clean in valid_actions:
                actions.append(clean)
        return actions or list(self.available_actions)

    @property
    def phase_name(self) -> str:
        return self.phase.value
