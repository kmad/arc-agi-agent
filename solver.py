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


class SolverPhase(Enum):
    EXPLORE = "explore"
    HYPOTHESIZE = "hypothesize"
    ITERATE = "iterate"
    EXECUTE = "execute"


# ── Phase-specific RLM signatures ──────────────────────────────────

class ExploreSignature(dspy.Signature):
    """You are systematically exploring an unknown puzzle game on a 64x64 grid.

    Your ONLY goal right now is to discover what each action does.
    For each available action, you need to determine:
    - Does it cause movement? In which direction? How many pixels?
    - Does it interact with something? What changes?
    - Does it do nothing? (blocked by wall, invalid state)
    - Does it cost a resource? (timer bar, move counter)

    Design an exploration sequence that tests EVERY available action
    multiple times in different contexts (different positions on the grid).
    Vary your sequences: don't just repeat the same action.

    Output a batch of actions as a JSON list. MAXIMIZE DIVERSITY."""

    frame_analysis: str = dspy.InputField(desc="Programmatic analysis: objects by color, positions, sizes, shapes, diff from previous frame")
    game_state: str = dspy.InputField(desc="Game state: step count, levels, recent actions")
    action_effects_so_far: str = dspy.InputField(desc="What we know about each action's effect so far")
    available_actions: str = dspy.InputField(desc="All available actions in this game")

    exploration_plan: str = dspy.OutputField(desc="Which actions to test and why. What do we still NOT know?")
    actions: str = dspy.OutputField(desc='JSON list of 6-12 actions designed to maximize information gain. e.g. ["ACTION1","ACTION2","ACTION1","ACTION3"]')


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

    hypotheses: str = dspy.OutputField(desc='JSON list of hypotheses: [{"mechanic": "...", "description": "...", "confidence": 0-100, "test": "how to verify"}]')
    actions: str = dspy.OutputField(desc='JSON list of 4-8 actions to begin testing the top hypothesis')


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

    updated_hypotheses: str = dspy.OutputField(desc='JSON: updated hypotheses with new confidence levels after testing')
    phase_recommendation: str = dspy.OutputField(desc='Either "keep_iterating" or "ready_to_execute" with reason')
    actions: str = dspy.OutputField(desc='JSON list of 4-8 targeted test actions')


class ExecuteSignature(dspy.Signature):
    """You are playing a puzzle game with CONFIRMED understanding of mechanics.

    You know what the actions do, what the objects are, and what the goal is.
    Now plan and execute efficient action sequences to complete the level.

    Use your knowledge to:
    1. Identify current position relative to the goal
    2. Plan the shortest/safest path
    3. Account for any resource limits (timer, move counter)
    4. Handle obstacles using known mechanics

    Be EFFICIENT. Every wasted action may cost a resource.
    If you get stuck or something unexpected happens, say so in your reasoning."""

    frame_analysis: str = dspy.InputField(desc="Programmatic analysis: objects, positions, confirmed mechanics context")
    game_state: str = dspy.InputField(desc="Game state with level info and knowledge")
    confirmed_mechanics: str = dspy.InputField(desc="Confirmed game mechanics and rules")
    visual_analysis: str = dspy.InputField(desc="Visual observer's spatial analysis")
    solver_directives: str = dspy.InputField(desc="Dynamic solver directives from optimizer")
    available_actions: str = dspy.InputField(desc="Available actions")

    plan: str = dspy.OutputField(desc="Step-by-step plan: where am I, where's the goal, what's the path")
    actions: str = dspy.OutputField(desc='JSON list of 3-8 precise goal-directed actions')


# ── Phase-aware Solver ─────────────────────────────────────────────

class Solver:
    """Phase-aware solver: explore → hypothesize → iterate → execute.

    Uses RLM (with sub-LLM) for explore/hypothesize/iterate phases where
    the sub-LLM can explore multiple candidate solutions in parallel.
    Uses fast CoT for the execute phase where we need speed.
    """

    # Phase transition thresholds
    EXPLORE_MIN_STEPS = 8        # Minimum steps before leaving EXPLORE
    EXPLORE_MAX_STEPS = 40       # Force transition to HYPOTHESIZE
    ITERATE_MAX_ROUNDS = 5       # Max hypothesis test rounds before EXECUTE
    HIGH_CONFIDENCE = 85         # Hypothesis confidence to count as "confirmed"
    MECHANICS_TO_EXECUTE = 2     # Confirmed mechanics needed to enter EXECUTE

    def __init__(self, lm: dspy.LM, sub_lm: Optional[dspy.LM] = None,
                 available_actions: Optional[list[str]] = None):
        self.lm = lm
        self.sub_lm = sub_lm or lm
        self.available_actions = available_actions or ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

        # Phase state
        self.phase = SolverPhase.EXPLORE
        self.phase_step_count = 0  # Steps within current phase
        self.iterate_rounds = 0

        # Knowledge accumulated across phases
        self.action_effects: dict[str, list[str]] = {a: [] for a in self.available_actions}
        self.hypotheses: list[dict] = []
        self.confirmed_mechanics: list[dict] = []

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

        actions = self._parse_batch_actions(result.actions)

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

        result = self.hypothesizer(
            frame_analysis=analysis,
            game_state=history.get_state_summary(),
            exploration_log=exploration_log,
            available_actions=actions_desc,
        )

        # Parse hypotheses
        try:
            parsed = json.loads(result.hypotheses)
            if isinstance(parsed, list):
                self.hypotheses = parsed
                print(f"  [Hypothesize] {len(self.hypotheses)} hypotheses formed:")
                for h in self.hypotheses[:3]:
                    print(f"    [{h.get('confidence', '?')}%] {h.get('mechanic', '?')}: {h.get('description', '?')[:70]}")
        except (json.JSONDecodeError, TypeError):
            print(f"  [Hypothesize] Could not parse hypotheses, keeping existing")

        # Transition to ITERATE after forming hypotheses
        self._transition_to(SolverPhase.ITERATE)

        return self._parse_batch_actions(result.actions)

    def _iterate(self, history: GameHistory) -> list[str]:
        """ITERATE: Test hypotheses with targeted experiments."""
        self.iterate_rounds += 1
        actions_desc = ", ".join(self.available_actions)

        hypotheses_text = json.dumps(self.hypotheses, indent=2) if self.hypotheses else "No hypotheses yet."
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

        # Update hypotheses from iteration results
        try:
            updated = json.loads(result.updated_hypotheses)
            if isinstance(updated, list):
                self.hypotheses = updated
                # Promote high-confidence hypotheses to confirmed mechanics
                newly_confirmed = [h for h in self.hypotheses if h.get("confidence", 0) >= self.HIGH_CONFIDENCE]
                for h in newly_confirmed:
                    if h not in self.confirmed_mechanics:
                        self.confirmed_mechanics.append(h)
                        print(f"  [Iterate] CONFIRMED: {h.get('mechanic', '?')}: {h.get('description', '?')[:60]}")
        except (json.JSONDecodeError, TypeError):
            pass

        # Check if ready to execute
        phase_rec = result.phase_recommendation if hasattr(result, 'phase_recommendation') else ""
        if "ready_to_execute" in phase_rec.lower() or len(self.confirmed_mechanics) >= self.MECHANICS_TO_EXECUTE:
            print(f"  [Iterate] {len(self.confirmed_mechanics)} mechanics confirmed, transitioning to EXECUTE")
            self._transition_to(SolverPhase.EXECUTE)

        return self._parse_batch_actions(result.actions)

    def _execute(self, history: GameHistory, visual_obs: dict) -> list[str]:
        """EXECUTE: Fast goal-directed play using confirmed mechanics."""
        actions_desc = ", ".join(self.available_actions)

        visual_text = visual_obs.get("visual_analysis", "No visual analysis.")
        if "recommended_strategy" in visual_obs:
            visual_text += f"\nStrategy: {visual_obs['recommended_strategy']}"

        solver_directives = history.solver_instructions or "Execute efficiently using confirmed mechanics."

        mechanics_text = self._format_confirmed_mechanics()
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

        actions = self._parse_batch_actions(result.actions)

        # Inject diversity if all-same
        if len(actions) >= 5 and len(set(actions)) == 1:
            actions = self._inject_exploration(actions)

        return actions

    # ── Phase transition logic ──

    def _check_phase_transitions(self, history: GameHistory):
        """Auto-advance phases based on step counts and knowledge."""
        if self.phase == SolverPhase.EXPLORE:
            # Enough exploration? Check if we have effects for most actions
            actions_with_data = sum(1 for effects in self.action_effects.values() if len(effects) >= 2)
            has_enough_data = actions_with_data >= len(self.available_actions) * 0.7

            if self.phase_step_count >= self.EXPLORE_MAX_STEPS:
                print(f"  [Phase] EXPLORE max steps reached, forcing HYPOTHESIZE")
                self._transition_to(SolverPhase.HYPOTHESIZE)
            elif self.phase_step_count >= self.EXPLORE_MIN_STEPS and has_enough_data:
                print(f"  [Phase] Enough exploration data ({actions_with_data}/{len(self.available_actions)} actions tested), moving to HYPOTHESIZE")
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
        self.hypotheses = []
        # Keep confirmed_mechanics — some may carry over between levels

    # ── Knowledge tracking ──

    def _update_action_effects(self, history: GameHistory):
        """Extract action-effect pairs from recent history."""
        for step in history.steps[-10:]:
            if step.action == "RESET" or not step.grid_diff:
                continue
            action = step.action.split("(")[0]  # Strip coords from ACTION6(x,y)
            if action in self.action_effects:
                effect = step.grid_diff[:100]
                # Keep only last 5 effects per action to stay current
                effects = self.action_effects[action]
                if len(effects) >= 5:
                    effects.pop(0)
                if effect not in effects:
                    effects.append(effect)

    def _format_action_effects(self) -> str:
        """Format known action effects for prompts."""
        lines = []
        for action in sorted(self.action_effects.keys()):
            effects = self.action_effects[action]
            if effects:
                lines.append(f"{action}: {'; '.join(effects)}")
            else:
                lines.append(f"{action}: NOT YET TESTED")
        return "\n".join(lines)

    def _format_confirmed_mechanics(self) -> str:
        """Format confirmed mechanics for the execute phase."""
        if not self.confirmed_mechanics:
            # Fall back to hypotheses if nothing confirmed yet
            if self.hypotheses:
                return "Unconfirmed hypotheses:\n" + "\n".join(
                    f"  [{h.get('confidence', '?')}%] {h.get('mechanic', '?')}: {h.get('description', '?')}"
                    for h in self.hypotheses
                )
            return "No mechanics confirmed yet. Explore and observe."

        lines = ["Confirmed mechanics:"]
        for m in self.confirmed_mechanics:
            lines.append(f"  [{m.get('confidence', '?')}%] {m.get('mechanic', '?')}: {m.get('description', '?')}")
        return "\n".join(lines)

    # ── Utility methods ──

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

    def _parse_batch_actions(self, actions_str: str) -> list[str]:
        """Parse a batch of actions from solver output."""
        valid_actions = set(self.available_actions)
        try:
            actions = json.loads(actions_str)
            if isinstance(actions, list):
                parsed = [a.strip().upper() for a in actions if isinstance(a, str)]
                result = [a for a in parsed if a in valid_actions]
                if result:
                    return result
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: extract action names from text
        actions = []
        for token in actions_str.upper().replace(",", " ").replace('"', " ").replace("[", " ").replace("]", " ").split():
            clean = token.strip()
            if clean in valid_actions:
                actions.append(clean)
        return actions or list(self.available_actions)

    @property
    def phase_name(self) -> str:
        return self.phase.value
