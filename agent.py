"""Main orchestrator: RLM solver + 3 reflection agents + custom optimizer.

Architecture:
  - Solver (RLM): Plays the game, outputs batches of actions
  - Visual Observer (Predict): Analyzes game frames visually
  - Game Mechanics Observer (RLM): Builds game mechanics knowledge base
  - REPL Strategy Observer (RLM): Builds REPL usage knowledge base
  - Instruction Optimizer (CoT): Rewrites solver instructions after each reflection cycle
"""

import sys
import time
import json
import numpy as np
import dspy
from arcengine.enums import GameAction, GameState
from arc_agi import Arcade

from config import (
    GEMINI_MODEL, GEMINI_MODEL_MINI,
    RLM_MAX_ITERATIONS, RLM_BATCH_SIZE,
    DEFAULT_GAME, MAX_STEPS,
)
from game_state import GameHistory, StepRecord, compute_grid_diff, frame_to_text
from observers import VisualObserver, GameMechanicsObserver, REPLStrategyObserver
from solver import Solver
from optimizer import InstructionOptimizer


ACTION_MAP = {
    "ACTION1": GameAction.ACTION1,
    "ACTION2": GameAction.ACTION2,
    "ACTION3": GameAction.ACTION3,
    "ACTION4": GameAction.ACTION4,
}

ACTION_NAMES = {
    GameAction.ACTION1: "ACTION1 (up)",
    GameAction.ACTION2: "ACTION2 (down)",
    GameAction.ACTION3: "ACTION3 (left)",
    GameAction.ACTION4: "ACTION4 (right)",
}


def run_agent(game_id: str = DEFAULT_GAME, max_steps: int = MAX_STEPS, batch_size: int = RLM_BATCH_SIZE):
    """Run the full multi-agent system on a game."""

    # === Configure LMs ===
    print(f"Configuring language models...")
    main_lm = dspy.LM(GEMINI_MODEL, max_tokens=8192)
    mini_lm = dspy.LM(GEMINI_MODEL_MINI, max_tokens=4096)
    dspy.configure(lm=main_lm)

    # === Initialize game ===
    print(f"Initializing game: {game_id}")
    arcade = Arcade()
    env = arcade.make(game_id, render_mode="terminal")

    obs = env.reset()
    initial_frame = np.array(obs.frame)[0]  # (1, 64, 64) -> (64, 64)

    history = GameHistory(
        game_id=game_id,
        total_levels=obs.win_levels,
    )

    # Record initial state
    history.add_step(StepRecord(
        step_number=0,
        action="RESET",
        frame=initial_frame,
        state=str(obs.state),
        levels_completed=obs.levels_completed,
        grid_diff="Initial frame.",
    ))

    print(f"Game initialized: {game_id}")
    print(f"  Win levels: {obs.win_levels}")
    print(f"  Available actions: {obs.available_actions}")
    print(f"  Frame shape: {initial_frame.shape}")
    print(f"  Grid values: {np.unique(initial_frame)}")
    print()

    # === Initialize agents ===
    print("Initializing agents...")
    solver = Solver(lm=main_lm, sub_lm=mini_lm, batch_mode=True)
    visual_observer = VisualObserver(lm=main_lm)
    mechanics_observer = GameMechanicsObserver(lm=main_lm, sub_lm=mini_lm)
    repl_observer = REPLStrategyObserver(lm=main_lm, sub_lm=mini_lm)
    optimizer = InstructionOptimizer(lm=main_lm)

    # Track REPL traces from the solver's RLM
    repl_traces = []
    visual_observations = []
    move_count = 0
    batch_number = 0

    print("All agents initialized. Starting game loop.\n")
    print("=" * 70)

    # === Main game loop ===
    while move_count < max_steps:
        batch_number += 1
        print(f"\n{'='*70}")
        print(f"BATCH {batch_number} | Step {move_count} | Levels: {history.levels_completed}/{history.total_levels}")
        print(f"{'='*70}")

        # --- Phase 1: Visual observation ---
        print("\n[Visual Observer] Analyzing current frame...")
        try:
            vis_obs = visual_observer.observe(history)
            visual_observations.append(vis_obs)
            print(f"  Analysis: {vis_obs.get('visual_analysis', 'N/A')[:200]}...")
            print(f"  Player: {vis_obs.get('player_position', 'unknown')}")
            print(f"  Strategy: {vis_obs.get('recommended_strategy', 'N/A')[:150]}")
        except Exception as e:
            print(f"  [ERROR] Visual observer failed: {e}")
            vis_obs = {"visual_analysis": "Observer failed.", "recommended_strategy": "Explore."}
            visual_observations.append(vis_obs)

        # --- Phase 2: Solver decides actions ---
        print(f"\n[Solver RLM] Planning actions...")
        try:
            actions = solver.solve_step(history, vis_obs)
            # Collect REPL traces if available
            if hasattr(solver.rlm, '_last_trajectory'):
                for step in solver.rlm._last_trajectory:
                    repl_traces.append(step)
        except Exception as e:
            print(f"  [ERROR] Solver failed: {e}")
            actions = ["ACTION1"]  # Default fallback

        print(f"  Planned actions: {actions}")

        # --- Phase 3: Execute actions ---
        for action_str in actions:
            if move_count >= max_steps:
                break

            game_action = ACTION_MAP.get(action_str, GameAction.ACTION1)
            prev_frame = history.last_frame

            try:
                result = env.step(game_action)
            except Exception as e:
                print(f"  [ERROR] step() failed: {e}")
                continue

            new_frame = np.array(result.frame)[0]
            diff = compute_grid_diff(prev_frame, new_frame)

            move_count += 1
            history.add_step(StepRecord(
                step_number=move_count,
                action=action_str,
                frame=new_frame,
                state=str(result.state),
                levels_completed=result.levels_completed,
                grid_diff=diff,
            ))

            print(f"  Step {move_count}: {action_str} -> {diff[:80]}")

            # Check for win/game over
            if result.state == GameState.WIN:
                print(f"\n{'*'*70}")
                print(f"GAME WON in {move_count} steps!")
                print(f"{'*'*70}")
                _print_final_stats(history, arcade)
                return history

            if result.state == GameState.GAME_OVER:
                print(f"\n{'!'*70}")
                print(f"GAME OVER at step {move_count}.")
                print(f"{'!'*70}")
                _print_final_stats(history, arcade)
                return history

        # --- Phase 4: Reflection cycle (every batch_size moves) ---
        if batch_number % 1 == 0:  # Reflect after every batch
            print(f"\n--- Reflection Cycle ---")

            # 4a. Game Mechanics Observer
            print("[Mechanics Observer] Analyzing game mechanics...")
            try:
                mech_result = mechanics_observer.observe(history, visual_observations)
                new_entries = mech_result.get("new_entries", [])
                if new_entries:
                    history.game_knowledge_base.extend(new_entries)
                    print(f"  Added {len(new_entries)} new knowledge entries")
                    for entry in new_entries:
                        print(f"    [{entry.get('confidence', '?')}%] {entry.get('text', '')[:100]}")
                print(f"  Summary: {mech_result.get('mechanics_summary', 'N/A')[:200]}")
            except Exception as e:
                print(f"  [ERROR] Mechanics observer failed: {e}")

            # 4b. REPL Strategy Observer
            print("[REPL Observer] Analyzing REPL patterns...")
            try:
                repl_result = repl_observer.observe(history, repl_traces)
                new_repl = repl_result.get("new_entries", [])
                if new_repl:
                    history.repl_knowledge_base.extend(new_repl)
                    print(f"  Added {len(new_repl)} new REPL tips")
                print(f"  Summary: {repl_result.get('repl_strategy_summary', 'N/A')[:200]}")
            except Exception as e:
                print(f"  [ERROR] REPL observer failed: {e}")

            # 4c. Instruction Optimizer
            print("[Optimizer] Rewriting solver instructions...")
            try:
                new_instructions = optimizer.optimize(history, vis_obs)
                history.solver_instructions = new_instructions
                print(f"  Updated instructions ({len(new_instructions)} chars)")
                print(f"  Preview: {new_instructions[:200]}...")
            except Exception as e:
                print(f"  [ERROR] Optimizer failed: {e}")

            print(f"--- End Reflection ---\n")

    print(f"\nMax steps ({max_steps}) reached.")
    _print_final_stats(history, arcade)
    return history


def _print_final_stats(history: GameHistory, arcade: Arcade):
    """Print final game statistics."""
    print(f"\n{'='*70}")
    print("FINAL STATISTICS")
    print(f"{'='*70}")
    print(f"Total steps: {history.current_step}")
    print(f"Levels completed: {history.levels_completed}/{history.total_levels}")
    print(f"Game KB entries: {len(history.game_knowledge_base)}")
    print(f"REPL KB entries: {len(history.repl_knowledge_base)}")

    print(f"\n--- Game Knowledge Base ---")
    print(history.get_game_kb_text()[:2000])
    print(f"\n--- REPL Knowledge Base ---")
    print(history.get_repl_kb_text()[:2000])
    print(f"\n--- Final Solver Instructions ---")
    print(history.solver_instructions[:2000] if history.solver_instructions else "N/A")

    try:
        scorecard = arcade.get_scorecard()
        print(f"\n--- Scorecard ---")
        print(scorecard)
    except Exception:
        pass


if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GAME
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_STEPS
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else RLM_BATCH_SIZE

    print(f"ARC AGI 3 Multi-Agent System")
    print(f"  Game: {game}")
    print(f"  Max steps: {steps}")
    print(f"  Batch size: {batch}")
    print()

    history = run_agent(game_id=game, max_steps=steps, batch_size=batch)
