"""GEPA-powered optimizer that evolves the solver's strategy artifact.

Instead of just tweaking instructions, GEPA evolves the entire solver strategy
(instructions + analysis patterns + action sequences) as a text artifact.
Uses Actionable Side Information (ASI) from game execution to guide mutations.

This implements the "Agent Architecture Discovery" pattern from GEPA's
optimize_anything, adapted for ARC AGI 3 game-playing.
"""

import json
import numpy as np
import dspy
from dataclasses import dataclass
from typing import Optional
from arcengine.enums import GameAction, GameState
from arc_agi import Arcade

from config import GEMINI_MODEL, GEMINI_MODEL_MINI
from game_state import (
    GameHistory, StepRecord, compute_grid_diff, frame_to_text,
)


# === Seed strategy artifact ===
SEED_STRATEGY = """# ARC AGI 3 Solver Strategy

## Grid Analysis Protocol
1. On each turn, identify the active region of the grid (non-background pixels)
2. Find the player block: look for a small colored cluster (typically 5x5) that changes position between frames
3. Find the goal: look for matching color patterns or outlined regions
4. Identify obstacles: walls, barriers, or other collidable sprites

## Movement Rules
- ACTION1 = UP: Moves player block 5 pixels up
- ACTION2 = DOWN: Moves player block 5 pixels down
- ACTION3 = LEFT: Moves player block 5 pixels left
- ACTION4 = RIGHT: Moves player block 5 pixels right

## Strategy
1. First, explore to understand the grid layout
2. Identify the goal location relative to the player
3. Plan the shortest path avoiding obstacles
4. Execute the path as a batch of actions
5. After each batch, re-analyze to correct course

## REPL Analysis Tips
- Use numpy to find unique colors: np.unique(grid)
- Find player position: np.argwhere(grid == player_color)
- Find goal position: np.argwhere(grid == goal_color)
- Compute distance: abs(player_pos - goal_pos)
- Check for walls between player and goal

## Known Game Patterns
- Move limit bar (yellow/dark_yellow) depletes with each action
- Some levels require matching shapes or colors
- The player block and goal may share a color/pattern
- Interaction (ACTION5 if available) may be needed at the goal
"""


@dataclass
class GameEpisode:
    """A single game episode for evaluation."""
    game_id: str
    seed: int = 0


def create_evaluator(game_id: str = "ls20", max_episode_steps: int = 100):
    """Create a GEPA evaluator that scores strategy artifacts by playing the game.

    The evaluator:
    1. Parses the strategy artifact
    2. Uses it to configure a simple solver
    3. Plays the game for max_episode_steps
    4. Returns score (levels completed / total) + ASI diagnostics
    """

    def evaluate(candidate: str, data_instance=None) -> tuple[float, dict]:
        """Evaluate a strategy artifact by playing the game."""
        lm = dspy.LM(GEMINI_MODEL, max_tokens=4096)

        # Set up game
        arcade = Arcade()
        env = arcade.make(game_id)
        obs = env.reset()
        initial_frame = np.array(obs.frame)[0]

        win_levels = obs.win_levels
        levels_completed = 0
        steps_taken = 0
        action_history = []
        frame_diffs = []
        errors = []

        # Simple solver using the candidate strategy
        solver_sig = dspy.Signature(
            {
                "strategy": dspy.InputField(desc="Solver strategy document"),
                "current_frame": dspy.InputField(desc="Current game grid"),
                "action_history": dspy.InputField(desc="Recent actions taken"),
                "frame_changes": dspy.InputField(desc="Recent frame changes"),
            },
            "You are playing an ARC AGI 3 game. Follow the strategy to decide the next action."
        ).append("reasoning", dspy.OutputField(), type_=str
        ).append("action", dspy.OutputField(desc="One of: ACTION1, ACTION2, ACTION3, ACTION4"), type_=str)

        predict = dspy.Predict(solver_sig)
        prev_frame = initial_frame

        with dspy.context(lm=lm):
            for step in range(max_episode_steps):
                current_frame = prev_frame if step == 0 else np.array(result.frame)[0]
                diff = compute_grid_diff(prev_frame, current_frame) if step > 0 else "Initial"
                frame_diffs.append(diff)

                try:
                    pred = predict(
                        strategy=candidate,
                        current_frame=frame_to_text(current_frame),
                        action_history=", ".join(action_history[-10:]) or "None",
                        frame_changes="\n".join(frame_diffs[-5:]),
                    )
                    action_str = pred.action.strip().upper()
                    # Extract valid action
                    valid = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
                    action_name = next((a for a in valid if a in action_str), "ACTION1")
                except Exception as e:
                    errors.append(f"Step {step}: {str(e)}")
                    action_name = "ACTION1"

                game_action = {
                    "ACTION1": GameAction.ACTION1,
                    "ACTION2": GameAction.ACTION2,
                    "ACTION3": GameAction.ACTION3,
                    "ACTION4": GameAction.ACTION4,
                }[action_name]

                try:
                    result = env.step(game_action)
                except Exception as e:
                    errors.append(f"Step {step} env error: {str(e)}")
                    break

                action_history.append(action_name)
                steps_taken += 1
                prev_frame = np.array(result.frame)[0]
                levels_completed = result.levels_completed

                if result.state == GameState.WIN:
                    break
                if result.state == GameState.GAME_OVER:
                    break

        # Compute score
        score = levels_completed / max(win_levels, 1)

        # Build ASI (Actionable Side Information)
        from collections import Counter
        action_dist = Counter(action_history)

        # Detect stuck patterns
        stuck = False
        if len(action_history) >= 20:
            last_20 = action_history[-20:]
            if len(set(last_20)) <= 2:
                stuck = True

        asi = {
            "levels_completed": levels_completed,
            "total_levels": win_levels,
            "steps_taken": steps_taken,
            "action_distribution": dict(action_dist),
            "stuck_detected": stuck,
            "errors": errors[:10],
            "last_frame_changes": frame_diffs[-5:],
            "final_state": str(result.state) if 'result' in dir() else "UNKNOWN",
        }

        return score, asi

    return evaluate


def run_gepa_optimization(
    game_id: str = "ls20",
    max_metric_calls: int = 20,
    reflection_lm: str = GEMINI_MODEL,
    episode_steps: int = 100,
):
    """Run GEPA optimization to evolve the solver strategy.

    This uses GEPA's optimize loop to:
    1. Start with a seed strategy
    2. Evaluate it by playing the game
    3. Use ASI (game performance diagnostics) to guide mutations
    4. Evolve toward better strategies

    Returns the best strategy artifact found.
    """
    from gepa import optimize

    evaluator = create_evaluator(game_id=game_id, max_episode_steps=episode_steps)

    # Create a simple dataset of game instances (same game, different seeds for diversity)
    dataset = [GameEpisode(game_id=game_id, seed=i) for i in range(3)]

    result = optimize(
        seed_candidate={"strategy": SEED_STRATEGY},
        trainset=dataset,
        evaluator=evaluator,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        display_progress_bar=True,
        raise_on_exception=False,
    )

    print(f"\nGEPA Optimization Complete!")
    print(f"Best score: {result.best_score}")
    print(f"Best candidate preview: {result.best_candidate['strategy'][:500]}...")

    return result


if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    calls = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print(f"Running GEPA optimization for {game} with {calls} metric calls")
    result = run_gepa_optimization(game_id=game, max_metric_calls=calls)
