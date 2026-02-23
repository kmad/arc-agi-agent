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
from arc_agi import Arcade, OperationMode

from config import GEMINI_MODEL, GEMINI_MODEL_MINI
from actions import build_action_map, format_action_space, get_valid_action_names
from game_state import (
    GameHistory, StepRecord, compute_grid_diff, frame_to_text,
)


# === Seed strategy artifact (game-agnostic) ===
SEED_STRATEGY = """# ARC AGI 3 Solver Strategy

## Discovery Protocol
1. This is an unknown game on a 64x64 grid. Do NOT assume any specific game type.
2. Begin by trying every available action and carefully observing pixel changes.
3. Large pixel changes (40+) indicate a meaningful game action (movement, interaction).
4. Tiny pixel changes (2-4) may indicate a timer, counter, or resource bar ticking.
5. No change means the action is blocked or invalid in the current context.

## Grid Analysis
1. Identify the active region: find all non-zero pixels and their bounding box.
2. Look for color clusters: groups of same-colored pixels that may be game objects.
3. Track changes between frames to identify moving objects vs. static structures.
4. Note any repeating patterns, borders, or symmetric structures.

## Action Strategy
1. First batch: Try each available action 2-3 times and record effects.
2. Classify actions: which ones move something? Which ones interact? Which are blocked?
3. Once you understand movement, identify the goal (what changes when you progress).
4. Plan efficient paths: minimize actions to conserve any resource/timer.

## Adaptive Play
- If stuck (same tiny changes repeatedly): try completely different actions.
- If a large screen change occurs: you may have completed a level or triggered a reset.
- Watch for resource depletion: if a bar fills up or counts down, you're on a timer.
- Prioritize exploration early, execution once you understand the mechanics.

## Level Progression
- Each game has multiple levels. Completing one may change the grid layout entirely.
- Knowledge from earlier levels may or may not apply to later ones.
- After each level transition, re-explore before committing to a strategy.
"""


@dataclass
class GameEpisode:
    """A single game episode for evaluation."""
    game_id: str
    seed: int = 0


def create_evaluator(game_id: str = "ls20", max_episode_steps: int = 100, local: bool = False):
    """Create a GEPA evaluator that scores strategy artifacts by playing the game.

    The evaluator:
    1. Parses the strategy artifact
    2. Uses it to configure a simple solver
    3. Plays the game for max_episode_steps
    4. Returns score (levels completed / total) + ASI diagnostics

    Args:
        local: If True, use OperationMode.OFFLINE (~2000 FPS, no API key needed).
    """

    def evaluate(candidate: str, data_instance=None) -> tuple[float, dict]:
        """Evaluate a strategy artifact by playing the game."""
        lm = dspy.LM(GEMINI_MODEL, max_tokens=4096)

        # Set up game
        if local:
            arcade = Arcade(operation_mode=OperationMode.OFFLINE)
        else:
            arcade = Arcade()
        env = arcade.make(game_id)
        obs = env.reset()
        initial_frame = np.array(obs.frame)[0]

        # Build dynamic action map
        action_map = build_action_map(obs.available_actions)
        valid_action_names = get_valid_action_names(action_map)
        actions_desc = format_action_space(action_map)
        action_names_list = sorted(valid_action_names, key=lambda n: int(n.replace("ACTION", "")))

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
                "available_actions": dspy.InputField(desc="Available actions in this game"),
            },
            "You are playing an unknown ARC AGI 3 puzzle game. Follow the strategy to decide the next action."
        ).append("reasoning", dspy.OutputField(), type_=str
        ).append("action", dspy.OutputField(desc="One action from the available set"), type_=str)

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
                        available_actions=actions_desc,
                    )
                    action_str = pred.action.strip().upper()
                    # Extract valid action
                    action_name = next((a for a in valid_action_names if a in action_str), action_names_list[0])
                except Exception as e:
                    errors.append(f"Step {step}: {str(e)}")
                    action_name = action_names_list[0]

                game_action = action_map[action_name]

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

        # Action diversity metric (0-1, higher = more diverse)
        action_diversity = len(set(action_history)) / max(len(valid_action_names), 1) if action_history else 0

        # Exploration score: ratio of unique actions used in first 20 steps
        early_actions = action_history[:20]
        exploration_score = len(set(early_actions)) / max(len(valid_action_names), 1) if early_actions else 0

        asi = {
            "levels_completed": levels_completed,
            "total_levels": win_levels,
            "steps_taken": steps_taken,
            "action_distribution": dict(action_dist),
            "stuck_detected": stuck,
            "action_diversity": round(action_diversity, 3),
            "exploration_score": round(exploration_score, 3),
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
