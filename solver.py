"""Main RLM solver agent that plays the ARC AGI 3 game."""

import dspy
import json
import numpy as np
from typing import Optional
from game_state import GameHistory, frame_to_text, compute_grid_diff


class SolverSignature(dspy.Signature):
    """You are an expert AI agent playing an interactive puzzle game (ARC AGI 3).
    You control a player on a 64x64 grid. Your goal is to complete all levels.

    Available actions:
    - ACTION1: Move up (shifts player/block 5 pixels up)
    - ACTION2: Move down (shifts player/block 5 pixels down)
    - ACTION3: Move left (shifts player/block 5 pixels left)
    - ACTION4: Move right (shifts player/block 5 pixels right)

    You have access to numpy, pandas, and a Python REPL to analyze the grid.
    The grid uses integer color values (0=black, 1=blue, 2=red, 3=green,
    4=yellow, 5=grey, 6=magenta, 7=orange, 8=cyan, 9=brown).

    IMPORTANT: You must output exactly one action per call: ACTION1, ACTION2, ACTION3, or ACTION4.
    Think carefully about the game state and your accumulated knowledge before acting.

    Strategy: Use the REPL to analyze the grid, find the player position, identify the goal,
    and plan a path. Then output the next action to take."""

    current_frame: str = dspy.InputField(desc="Current 64x64 game grid as text")
    game_state: str = dspy.InputField(desc="Current game state summary")
    game_knowledge: str = dspy.InputField(desc="Accumulated game mechanics knowledge base")
    repl_knowledge: str = dspy.InputField(desc="REPL usage tips and patterns for this game")
    visual_analysis: str = dspy.InputField(desc="Latest visual observer analysis")
    solver_instructions: str = dspy.InputField(desc="Dynamic instructions from the optimizer")

    reasoning: str = dspy.OutputField(desc="Your reasoning about what to do next")
    action: str = dspy.OutputField(desc="Exactly one of: ACTION1, ACTION2, ACTION3, ACTION4")


class BatchSolverSignature(dspy.Signature):
    """You are an expert AI agent playing an interactive puzzle game (ARC AGI 3).
    You control a player on a 64x64 grid. Your goal is to complete all levels.

    Available actions:
    - ACTION1: Move up (shifts player/block 5 pixels up)
    - ACTION2: Move down (shifts player/block 5 pixels down)
    - ACTION3: Move left (shifts player/block 5 pixels left)
    - ACTION4: Move right (shifts player/block 5 pixels right)

    You have access to numpy, pandas, and a Python REPL to analyze the grid.
    The grid uses integer color values (0=black, 1=blue, 2=red, 3=green,
    4=yellow, 5=grey, 6=magenta, 7=orange, 8=cyan, 9=brown).

    Output a BATCH of actions as a JSON list. Each action should be one of:
    ACTION1, ACTION2, ACTION3, ACTION4.
    Output between 1 and 10 actions that form a logical sequence toward the goal.

    Use the REPL to analyze the grid, find the player, identify goals,
    and plan an efficient path. Then output the batch of actions."""

    current_frame: str = dspy.InputField(desc="Current 64x64 game grid as text")
    game_state: str = dspy.InputField(desc="Current game state summary")
    game_knowledge: str = dspy.InputField(desc="Accumulated game mechanics knowledge base")
    repl_knowledge: str = dspy.InputField(desc="REPL usage tips and patterns for this game")
    visual_analysis: str = dspy.InputField(desc="Latest visual observer analysis")
    solver_instructions: str = dspy.InputField(desc="Dynamic instructions from the optimizer")

    reasoning: str = dspy.OutputField(desc="Your analysis and reasoning for the planned actions")
    actions: str = dspy.OutputField(desc='JSON list of actions, e.g. ["ACTION1", "ACTION3", "ACTION3", "ACTION2"]')


class Solver:
    """Main RLM-based solver agent."""

    def __init__(self, lm: dspy.LM, sub_lm: Optional[dspy.LM] = None, batch_mode: bool = True):
        self.lm = lm
        self.sub_lm = sub_lm or lm
        self.batch_mode = batch_mode

        if batch_mode:
            self.rlm = dspy.RLM(
                BatchSolverSignature,
                max_iterations=15,
                sub_lm=self.sub_lm,
                tools=[],  # numpy/pandas available in sandbox
            )
        else:
            self.rlm = dspy.RLM(
                SolverSignature,
                max_iterations=10,
                sub_lm=self.sub_lm,
            )

    def solve_step(self, history: GameHistory, visual_obs: dict) -> list[str]:
        """Decide the next action(s) to take.

        Returns a list of action strings.
        """
        current_frame = history.last_frame
        if current_frame is None:
            return ["ACTION1"]  # Default first move

        frame_text = frame_to_text(current_frame)
        visual_text = visual_obs.get("visual_analysis", "No visual analysis available.")
        if "recommended_strategy" in visual_obs:
            visual_text += f"\nRecommended strategy: {visual_obs['recommended_strategy']}"

        solver_instructions = history.solver_instructions or (
            "Explore the game board systematically. "
            "Use the REPL to analyze the grid and find patterns. "
            "Identify the player (movable colored block), the goal, and obstacles. "
            "Plan efficient paths to minimize wasted moves."
        )

        with dspy.context(lm=self.lm):
            if self.batch_mode:
                result = self.rlm(
                    current_frame=frame_text,
                    game_state=history.get_state_summary(),
                    game_knowledge=history.get_game_kb_text(),
                    repl_knowledge=history.get_repl_kb_text(),
                    visual_analysis=visual_text,
                    solver_instructions=solver_instructions,
                )
                # Parse batch actions
                actions = self._parse_batch_actions(result.actions)
                return actions
            else:
                result = self.rlm(
                    current_frame=frame_text,
                    game_state=history.get_state_summary(),
                    game_knowledge=history.get_game_kb_text(),
                    repl_knowledge=history.get_repl_kb_text(),
                    visual_analysis=visual_text,
                    solver_instructions=solver_instructions,
                )
                action = self._parse_single_action(result.action)
                return [action]

    def _parse_batch_actions(self, actions_str: str) -> list[str]:
        """Parse a batch of actions from the solver output."""
        valid_actions = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
        try:
            actions = json.loads(actions_str)
            if isinstance(actions, list):
                parsed = [a.strip().upper() for a in actions if isinstance(a, str)]
                return [a for a in parsed if a in valid_actions] or ["ACTION1"]
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: extract action names from text
        actions = []
        for token in actions_str.upper().split():
            clean = token.strip('[]",')
            if clean in valid_actions:
                actions.append(clean)
        return actions or ["ACTION1"]

    def _parse_single_action(self, action_str: str) -> str:
        """Parse a single action from the solver output."""
        valid_actions = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
        cleaned = action_str.strip().upper()
        for valid in valid_actions:
            if valid in cleaned:
                return valid
        return "ACTION1"  # Default
