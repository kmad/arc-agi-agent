"""Reflection agents that observe game state and build knowledge bases."""

import dspy
import json
from typing import Optional
from game_state import GameHistory, frame_to_text, frame_to_image_bytes, analyze_frame


# ============================================================
# 1. Visual Observer - inspects game frames visually
# ============================================================

class VisualObserverSignature(dspy.Signature):
    """You are a visual observer for an interactive puzzle game (ARC AGI 3).
    You receive a screenshot of the game and a structured analysis of the grid.

    The frame_analysis already tells you WHAT objects exist and WHERE they are.
    Your job is to interpret the MEANING: what is the player? What is the goal?
    What are obstacles? What strategy should we use?

    Be specific about spatial relationships and movement patterns."""

    game_state_summary: str = dspy.InputField(desc="Current game state summary")
    frame_analysis: str = dspy.InputField(desc="Programmatic analysis: objects, colors, regions, sizes, diff from previous frame")
    screenshot: dspy.Image = dspy.InputField(desc="Screenshot of the current game frame")
    existing_knowledge: str = dspy.InputField(desc="Existing game knowledge base")

    visual_analysis: str = dspy.OutputField(desc="What do the objects represent? Player, goal, obstacles, UI elements?")
    player_position: str = dspy.OutputField(desc="Estimated player object position (row, col) and which color/shape it is")
    goal_description: str = dspy.OutputField(desc="What appears to be the goal or objective")
    recommended_strategy: str = dspy.OutputField(desc="Recommended next strategy based on visual analysis")


class VisualObserverTextSignature(dspy.Signature):
    """You are a visual observer for an interactive puzzle game (ARC AGI 3).
    You receive a structured analysis of the grid computed by numpy.

    The frame_analysis already tells you WHAT objects exist and WHERE they are.
    Your job is to interpret the MEANING: what is the player? What is the goal?
    What are obstacles? What strategy should we use?

    Be specific about spatial relationships and movement patterns."""

    game_state_summary: str = dspy.InputField(desc="Current game state summary")
    frame_analysis: str = dspy.InputField(desc="Programmatic analysis: objects, colors, regions, sizes, diff from previous frame")
    existing_knowledge: str = dspy.InputField(desc="Existing game knowledge base")

    visual_analysis: str = dspy.OutputField(desc="What do the objects represent? Player, goal, obstacles, UI elements?")
    player_position: str = dspy.OutputField(desc="Estimated player object position (row, col) and which color/shape it is")
    goal_description: str = dspy.OutputField(desc="What appears to be the goal or objective")
    recommended_strategy: str = dspy.OutputField(desc="Recommended next strategy based on visual analysis")


class VisualObserver:
    """Observes game frames and provides visual analysis.

    Supports two modes:
    - use_screenshots=True: sends a dspy.Image screenshot (requires multimodal LLM)
    - use_screenshots=False: sends structured text analysis only
    """

    def __init__(self, lm: dspy.LM, use_screenshots: bool = True):
        self.lm = lm
        self.use_screenshots = use_screenshots
        self.predict_with_image = dspy.Predict(VisualObserverSignature)
        self.predict_text_only = dspy.Predict(VisualObserverTextSignature)

    def observe(self, history: GameHistory) -> dict:
        current_frame = history.last_frame
        if current_frame is None:
            return {"visual_analysis": "No frame available yet."}

        # Precompute analysis programmatically (numpy, not LLM)
        prev_frame = history.steps[-2].frame if len(history.steps) >= 2 else None
        analysis = analyze_frame(current_frame, prev_frame)

        with dspy.context(lm=self.lm):
            if self.use_screenshots:
                screenshot = dspy.Image(url=frame_to_image_bytes(current_frame))
                result = self.predict_with_image(
                    game_state_summary=history.get_state_summary(),
                    frame_analysis=analysis,
                    screenshot=screenshot,
                    existing_knowledge=history.get_game_kb_text(),
                )
            else:
                result = self.predict_text_only(
                    game_state_summary=history.get_state_summary(),
                    frame_analysis=analysis,
                    existing_knowledge=history.get_game_kb_text(),
                )

        return {
            "visual_analysis": result.visual_analysis,
            "player_position": result.player_position,
            "goal_description": result.goal_description,
            "recommended_strategy": result.recommended_strategy,
        }


# ============================================================
# 2. Game Mechanics Observer (RLM) - builds game knowledge base
# ============================================================

class GameMechanicsSignature(dspy.Signature):
    """You are a game mechanics analyst for an interactive puzzle game (ARC AGI 3).
    You have access to the full game state history and metadata.
    Your job is to identify and document game mechanics, rules, and patterns.

    Output a JSON list of knowledge entries, each with:
    - category: one of "player_mechanics", "game_rules", "level_structure", "win_condition", "obstacle", "item"
    - text: clear description of the mechanic/rule
    - confidence: 0-100 how confident you are
    - source: what evidence supports this

    Focus on NEW discoveries not already in the existing knowledge base.
    Update confidence of existing entries if you have new evidence."""

    game_state_history: str = dspy.InputField(desc="Full game state history with actions and diffs")
    action_space: str = dspy.InputField(desc="Available actions in this game")
    existing_game_kb: str = dspy.InputField(desc="Existing game knowledge base")
    visual_observations: str = dspy.InputField(desc="Recent visual observer analyses")

    new_knowledge_entries: str = dspy.OutputField(desc="JSON list of new/updated knowledge entries")
    mechanics_summary: str = dspy.OutputField(desc="Summary of understood game mechanics so far")


class GameMechanicsObserver:
    """RLM-based observer that builds a knowledge base of game mechanics."""

    def __init__(self, lm: dspy.LM, sub_lm: Optional[dspy.LM] = None,
                 action_desc: str = ""):
        self.lm = lm
        self.sub_lm = sub_lm or lm
        self.action_desc = action_desc
        # Use RLM for deep analysis when history is large
        self.rlm = dspy.RLM(
            GameMechanicsSignature,
            max_iterations=4,
            sub_lm=self.sub_lm,
        )
        # Use simple predict for early game
        self.predict = dspy.Predict(GameMechanicsSignature)

    def observe(self, history: GameHistory, visual_observations: list[dict]) -> dict:
        # Build history summary
        history_lines = []
        for step in history.steps[-50:]:  # Last 50 steps
            history_lines.append(
                f"Step {step.step_number}: {step.action} -> state={step.state}, "
                f"levels={step.levels_completed}, diff={step.grid_diff or 'N/A'}"
            )
        history_text = "\n".join(history_lines) if history_lines else "No history yet."

        # Format visual observations
        vis_text = "\n---\n".join(
            f"Step {i}: {obs.get('visual_analysis', 'N/A')}"
            for i, obs in enumerate(visual_observations[-5:])
        ) if visual_observations else "No visual observations yet."

        # Use dynamic action description
        action_space = self.action_desc or "Unknown action space"

        with dspy.context(lm=self.lm):
            # Use RLM for larger histories, predict for small ones
            if len(history.steps) > 20:
                result = self.rlm(
                    game_state_history=history_text,
                    action_space=action_space,
                    existing_game_kb=history.get_game_kb_text(),
                    visual_observations=vis_text,
                )
            else:
                result = self.predict(
                    game_state_history=history_text,
                    action_space=action_space,
                    existing_game_kb=history.get_game_kb_text(),
                    visual_observations=vis_text,
                )

        # Parse new knowledge entries
        new_entries = []
        try:
            parsed = json.loads(result.new_knowledge_entries)
            if isinstance(parsed, list):
                for entry in parsed:
                    entry["step"] = history.current_step
                    new_entries.append(entry)
        except (json.JSONDecodeError, TypeError):
            # If JSON parsing fails, create a single entry from the text
            new_entries.append({
                "category": "observation",
                "text": str(result.new_knowledge_entries),
                "confidence": 50,
                "source": "mechanics_observer",
                "step": history.current_step,
            })

        return {
            "new_entries": new_entries,
            "mechanics_summary": result.mechanics_summary,
        }


# ============================================================
# 3. REPL Strategy Observer (RLM) - builds REPL interaction KB
# ============================================================

class REPLStrategySignature(dspy.Signature):
    """You are a REPL interaction strategist for an AI agent playing an interactive puzzle game.
    The solver agent uses a Python REPL to analyze game state and decide actions.

    Your job is to analyze the REPL history (what code the solver has been running)
    and build a knowledge base of effective REPL patterns, suggestions for improvement,
    and tips for this specific game.

    Think of this as building a skill.md for how to effectively use the REPL for this game.

    Output a JSON list of REPL tips/patterns, each with:
    - category: one of "analysis_pattern", "efficiency_tip", "code_template", "anti_pattern", "game_specific_tool"
    - text: clear description of the pattern/tip
    - code_example: optional example code snippet

    Focus on NEW discoveries. Suggest numpy/pandas patterns for grid analysis."""

    repl_history: str = dspy.InputField(desc="Recent REPL execution history (code and outputs)")
    game_state_summary: str = dspy.InputField(desc="Current game state summary")
    existing_repl_kb: str = dspy.InputField(desc="Existing REPL knowledge base")
    game_mechanics: str = dspy.InputField(desc="Known game mechanics")

    new_repl_entries: str = dspy.OutputField(desc="JSON list of new REPL tips/patterns")
    repl_strategy_summary: str = dspy.OutputField(desc="Summary of recommended REPL usage strategy")


class REPLStrategyObserver:
    """RLM-based observer that builds a knowledge base for REPL interaction patterns."""

    def __init__(self, lm: dspy.LM, sub_lm: Optional[dspy.LM] = None):
        self.lm = lm
        self.sub_lm = sub_lm or lm
        self.rlm = dspy.RLM(
            REPLStrategySignature,
            max_iterations=4,
            sub_lm=self.sub_lm,
        )
        self.predict = dspy.Predict(REPLStrategySignature)

    def observe(self, history: GameHistory, repl_traces: list[dict]) -> dict:
        # Format REPL traces
        repl_lines = []
        for trace in repl_traces[-20:]:
            code = trace.get("code", "")
            output = trace.get("output", "")
            repl_lines.append(f">>> {code}\n{output}")
        repl_text = "\n---\n".join(repl_lines) if repl_lines else "No REPL history yet."

        with dspy.context(lm=self.lm):
            if len(repl_traces) > 10:
                result = self.rlm(
                    repl_history=repl_text,
                    game_state_summary=history.get_state_summary(),
                    existing_repl_kb=history.get_repl_kb_text(),
                    game_mechanics=history.get_game_kb_text(),
                )
            else:
                result = self.predict(
                    repl_history=repl_text,
                    game_state_summary=history.get_state_summary(),
                    existing_repl_kb=history.get_repl_kb_text(),
                    game_mechanics=history.get_game_kb_text(),
                )

        # Parse new entries
        new_entries = []
        try:
            parsed = json.loads(result.new_repl_entries)
            if isinstance(parsed, list):
                new_entries = parsed
        except (json.JSONDecodeError, TypeError):
            new_entries.append({
                "category": "tip",
                "text": str(result.new_repl_entries),
            })

        return {
            "new_entries": new_entries,
            "repl_strategy_summary": result.repl_strategy_summary,
        }
