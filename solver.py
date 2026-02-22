"""Main solver agent that plays the ARC AGI 3 game.

Uses a fast CoT solver for action batches, with periodic RLM deep analysis."""

import dspy
import json
import numpy as np
from typing import Optional
from game_state import GameHistory, frame_to_text, compute_grid_diff
from actions import format_action_space, get_valid_action_names


class FastSolverSignature(dspy.Signature):
    """You are playing an unknown interactive puzzle game on a 64x64 grid.
    Your goal is to discover game mechanics through play and complete all levels.

    APPROACH:
    1. Early game: Explore systematically. Try every available action and observe what changes.
       - Actions causing large pixel changes (40+) are meaningful moves.
       - Actions causing tiny changes (2-4 pixels) may just be a timer/counter ticking.
       - Actions causing no change may be blocked by walls or invalid in context.
    2. Mid game: Once you understand which actions move you and what the goal is,
       execute efficient sequences to progress.
    3. Use the visual_analysis to understand spatial layout: find moving objects,
       stationary structures, goals, and obstacles.

    Output a BATCH of 3-8 actions as a JSON list. Prefer SHORT precise sequences."""

    current_frame: str = dspy.InputField(desc="Current 64x64 game grid (active region)")
    game_state: str = dspy.InputField(desc="Current game state summary with stuck warnings")
    recent_action_history: str = dspy.InputField(desc="Recent actions with their pixel change impacts - TINY means unproductive!")
    knowledge: str = dspy.InputField(desc="Accumulated game knowledge and REPL tips")
    visual_analysis: str = dspy.InputField(desc="Visual observer analysis of game objects and layout")
    solver_directives: str = dspy.InputField(desc="Dynamic solver directives")
    available_actions: str = dspy.InputField(desc="List of available actions in this game")

    reasoning: str = dspy.OutputField(desc="1. What objects do I see? 2. What changed from my last actions? 3. What should I try next?")
    actions: str = dspy.OutputField(desc='JSON list of 3-8 actions from the available set. e.g. ["ACTION1","ACTION2","ACTION3"]')


class DeepAnalysisSignature(dspy.Signature):
    """You are deeply analyzing an unknown interactive puzzle game on a 64x64 grid.
    Use careful grid analysis to understand the game state.

    Analyze the grid to find:
    - Color clusters: groups of same-colored pixels that may be objects
    - Moving objects: compare with recent history to identify what moves
    - Stationary structures: walls, borders, platforms, goals
    - Patterns: repeating structures, symmetry, or arrangements that suggest a puzzle

    Based on your analysis, plan actions to make progress (complete levels, reach goals,
    solve the puzzle). Be efficient - each action may cost a resource."""

    current_frame: str = dspy.InputField(desc="Current 64x64 game grid")
    game_state: str = dspy.InputField(desc="Current game state summary with stuck warnings")
    recent_action_history: str = dspy.InputField(desc="Recent actions with pixel change impacts")
    knowledge: str = dspy.InputField(desc="All accumulated knowledge")
    solver_directives: str = dspy.InputField(desc="Solver directives")
    available_actions: str = dspy.InputField(desc="List of available actions in this game")

    analysis: str = dspy.OutputField(desc="Object positions, movement patterns, identified goals, recommended strategy")
    actions: str = dspy.OutputField(desc='JSON list of planned actions from the available set')


class Solver:
    """Hybrid solver: fast CoT for most batches, RLM for periodic deep analysis."""

    def __init__(self, lm: dspy.LM, sub_lm: Optional[dspy.LM] = None,
                 available_actions: Optional[list[str]] = None):
        self.lm = lm
        self.sub_lm = sub_lm or lm
        self.available_actions = available_actions or ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        self.fast_solver = dspy.ChainOfThought(FastSolverSignature)
        self.deep_solver = dspy.RLM(
            DeepAnalysisSignature,
            max_iterations=10,
            sub_lm=self.sub_lm,
        )
        self.call_count = 0

    def solve_step(self, history: GameHistory, visual_obs: dict, deep: bool = False) -> list[str]:
        """Decide the next batch of actions.

        Args:
            deep: If True, use RLM for deep analysis. Otherwise use fast CoT.
        """
        self.call_count += 1
        current_frame = history.last_frame
        if current_frame is None:
            return self.available_actions[:5] if len(self.available_actions) >= 5 else list(self.available_actions)

        # Check if stuck - force exploration if so
        stuck, stuck_msg = history.is_stuck()
        if stuck:
            return self._forced_exploration(history)

        frame_text = frame_to_text(current_frame)
        visual_text = visual_obs.get("visual_analysis", "No visual analysis.")
        if "recommended_strategy" in visual_obs:
            visual_text += f"\nStrategy: {visual_obs['recommended_strategy']}"

        solver_directives = history.solver_instructions or (
            "Discover a new game. Explore systematically. "
            "Try all available actions and observe pixel changes. "
            "Large changes (40+ pixels) indicate meaningful moves. "
            "Tiny changes (2-4 pixels) may just be a counter/timer."
        )

        # Combine knowledge bases
        knowledge = history.get_game_kb_text() + "\n\n" + history.get_repl_kb_text()

        # Recent action history with diffs
        recent_diffs = history.get_recent_action_diffs(15)

        actions_desc = ", ".join(self.available_actions)

        with dspy.context(lm=self.lm):
            if deep:
                result = self.deep_solver(
                    current_frame=frame_text,
                    game_state=history.get_state_summary(),
                    recent_action_history=recent_diffs,
                    knowledge=knowledge,
                    solver_directives=solver_directives,
                    available_actions=actions_desc,
                )
            else:
                result = self.fast_solver(
                    current_frame=frame_text,
                    game_state=history.get_state_summary(),
                    recent_action_history=recent_diffs,
                    knowledge=knowledge,
                    visual_analysis=visual_text,
                    solver_directives=solver_directives,
                    available_actions=actions_desc,
                )

        actions = self._parse_batch_actions(result.actions)

        # Post-processing: ensure action diversity if actions are all the same
        if len(actions) >= 5 and len(set(actions)) == 1:
            actions = self._inject_exploration(actions)

        return actions

    def _forced_exploration(self, history: GameHistory) -> list[str]:
        """When stuck, systematically try all directions to find which ones work."""
        # Find which action the agent is stuck on
        recent = history.get_recent_actions(10)
        from collections import Counter
        stuck_action = Counter(recent).most_common(1)[0][0]

        # Try all OTHER available actions first, then the stuck one
        other_actions = [a for a in self.available_actions if a != stuck_action]

        # Systematic exploration: try each other action 3 times
        explore = []
        for action in other_actions:
            explore.extend([action] * 3)
        return explore

    def _inject_exploration(self, actions: list[str]) -> list[str]:
        """If solver outputs all-same actions, inject diversity."""
        base = actions[0]
        others = [a for a in self.available_actions if a != base]

        # Replace some actions with exploration
        result = list(actions)
        for i, other in enumerate(others):
            if i + 1 < len(result):
                result[i + 1] = other
        return result

    def _parse_batch_actions(self, actions_str: str) -> list[str]:
        """Parse a batch of actions from the solver output."""
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
