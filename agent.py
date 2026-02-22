"""Main orchestrator: Fast CoT solver + periodic RLM deep analysis + 3 reflection agents.

Architecture:
  - Solver (CoT fast / RLM deep): Plays the game, outputs batches of 5-10 actions
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
from actions import build_action_map, format_action_space, parse_action_string, get_valid_action_names
from game_state import GameHistory, StepRecord, compute_grid_diff, frame_to_text
from observers import VisualObserver, GameMechanicsObserver, REPLStrategyObserver
from solver import Solver
from optimizer import InstructionOptimizer


# How often to run reflection agents (every N batches)
REFLECT_EVERY = 3


def run_agent(game_id: str = DEFAULT_GAME, max_steps: int = MAX_STEPS,
              batch_size: int = RLM_BATCH_SIZE, reflect_every: int = REFLECT_EVERY):
    """Run the full multi-agent system on a game."""

    # === Configure LMs ===
    print(f"Configuring LMs: {GEMINI_MODEL}")
    main_lm = dspy.LM(GEMINI_MODEL, max_tokens=4096)
    mini_lm = dspy.LM(GEMINI_MODEL_MINI, max_tokens=2048)
    dspy.configure(lm=main_lm)

    # === Initialize game ===
    print(f"Initializing game: {game_id}")
    arcade = Arcade()
    env = arcade.make(game_id)

    obs = env.reset()
    initial_frame = np.array(obs.frame)[0]

    # Build dynamic action map from game's available actions
    action_map = build_action_map(obs.available_actions)
    action_names = sorted(action_map.keys(), key=lambda n: int(n.replace("ACTION", "")))
    action_desc = format_action_space(action_map)

    history = GameHistory(
        game_id=game_id,
        total_levels=obs.win_levels,
    )

    # Load persisted knowledge from previous runs
    if history.load_knowledge():
        print(f"  Loaded knowledge: {len(history.game_knowledge_base)} game entries, "
              f"{len(history.repl_knowledge_base)} REPL tips")
        if history.solver_instructions:
            print(f"  Loaded solver instructions ({len(history.solver_instructions)} chars)")

    history.add_step(StepRecord(
        step_number=0,
        action="RESET",
        frame=initial_frame,
        state=str(obs.state),
        levels_completed=obs.levels_completed,
        grid_diff="Initial frame.",
    ))

    print(f"  Win levels: {obs.win_levels}")
    print(f"  Actions: {obs.available_actions}")
    print(f"  Action map: {action_desc}")
    print(f"  Grid values: {np.unique(initial_frame)}")

    # === Initialize agents ===
    solver = Solver(lm=main_lm, sub_lm=mini_lm, available_actions=action_names)
    visual_observer = VisualObserver(lm=main_lm, use_screenshots=False)
    mechanics_observer = GameMechanicsObserver(lm=main_lm, sub_lm=mini_lm, action_desc=action_desc)
    repl_observer = REPLStrategyObserver(lm=main_lm, sub_lm=mini_lm)
    optimizer = InstructionOptimizer(lm=main_lm)

    repl_traces = []
    visual_observations = []
    move_count = 0
    batch_number = 0
    # Track actions since last level completion for level knowledge
    level_actions_buffer = []

    print(f"Starting game loop (max {max_steps} steps, reflect every {reflect_every} batches)\n")

    # === Main game loop ===
    while move_count < max_steps:
        batch_number += 1

        print(f"\n{'='*70}")
        print(f"BATCH {batch_number} [{solver.phase_name.upper()}] | Step {move_count} | Levels: {history.levels_completed}/{history.total_levels}")
        print(f"{'='*70}")

        # --- Visual observation (quick, every batch) ---
        try:
            vis_obs = visual_observer.observe(history)
            visual_observations.append(vis_obs)
            print(f"[Visual] Player: {vis_obs.get('player_position', '?')} | {vis_obs.get('recommended_strategy', 'N/A')[:100]}")
        except Exception as e:
            print(f"[Visual] ERROR: {e}")
            vis_obs = {"visual_analysis": "Failed.", "recommended_strategy": "Explore."}
            visual_observations.append(vis_obs)

        # --- Solver decides actions ---
        t0 = time.time()
        try:
            actions = solver.solve_step(history, vis_obs)
        except Exception as e:
            print(f"[Solver] ERROR: {e}")
            actions = action_names[:4] if len(action_names) >= 4 else list(action_names)  # Explore fallback
        elapsed = time.time() - t0
        print(f"[Solver] {len(actions)} actions in {elapsed:.1f}s: {actions}")

        # --- Execute actions ---
        for action_str in actions:
            if move_count >= max_steps:
                break

            # Parse action string (handles ACTION6(x,y) format)
            action_name, action_params = parse_action_string(action_str)

            game_action = action_map.get(action_name)
            if game_action is None:
                print(f"  [WARN] Unknown action {action_name}, skipping")
                continue

            prev_frame = history.last_frame
            prev_levels = history.levels_completed

            try:
                if action_params and "x" in action_params and "y" in action_params:
                    # Click action with coordinates
                    result = env.step(game_action, x=action_params["x"], y=action_params["y"])
                else:
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
            level_actions_buffer.append(action_str)

            # Compact logging
            if diff == "No change.":
                print(f"  #{move_count} {action_str} -> no change")
            else:
                print(f"  #{move_count} {action_str} -> {diff[:60]}")

            # --- Level transition detection ---
            transition = history.detect_level_transition(
                prev_levels, result.levels_completed, prev_frame, new_frame
            )

            if transition == "level_complete":
                completed_level = prev_levels + 1
                print(f"\n  *** LEVEL {completed_level} COMPLETE at step {move_count}! ***")
                history.on_level_complete(completed_level, level_actions_buffer)
                level_actions_buffer = []
                solver.on_level_transition()
                # Force a reflection cycle on level completion
                print(f"  --- Forced Reflection (level complete) ---")
                _run_reflection(history, visual_observations, repl_traces,
                               mechanics_observer, repl_observer, optimizer, vis_obs)
                print(f"  --- End Forced Reflection ---")

            elif transition == "full_reset":
                current_level = result.levels_completed + 1
                history.level_attempts[current_level] = history.level_attempts.get(current_level, 0) + 1
                level_actions_buffer = []
                solver.on_level_transition()
                print(f"\n  !!! FULL RESET detected. Level {current_level} attempt #{history.level_attempts[current_level]} !!!")

            elif transition == "level_reset":
                current_level = result.levels_completed + 1
                history.level_attempts[current_level] = history.level_attempts.get(current_level, 0) + 1
                level_actions_buffer = []
                solver.on_level_transition()
                print(f"\n  !!! LEVEL RESET detected. Level {current_level} attempt #{history.level_attempts[current_level]} !!!")

            if result.state == GameState.WIN:
                print(f"\n*** GAME WON in {move_count} steps! ***")
                history.save_knowledge()
                _print_final_stats(history, arcade)
                return history

            if result.state == GameState.GAME_OVER:
                print(f"\n!!! GAME OVER at step {move_count} !!!")
                history.save_knowledge()
                _print_final_stats(history, arcade)
                return history

        # --- Reflection cycle (periodic) ---
        if batch_number % reflect_every == 0:
            print(f"\n--- Reflection Cycle (batch {batch_number}) ---")
            _run_reflection(history, visual_observations, repl_traces,
                           mechanics_observer, repl_observer, optimizer, vis_obs)
            print(f"--- End Reflection ---")

    print(f"\nMax steps ({max_steps}) reached.")
    history.save_knowledge()
    _print_final_stats(history, arcade)
    return history


def _run_reflection(history, visual_observations, repl_traces,
                    mechanics_observer, repl_observer, optimizer, vis_obs):
    """Run all three reflection agents and the optimizer."""
    # Game Mechanics Observer
    try:
        mech_result = mechanics_observer.observe(history, visual_observations)
        new_entries = mech_result.get("new_entries", [])
        if new_entries:
            history.game_knowledge_base.extend(new_entries)
            print(f"[Mechanics] +{len(new_entries)} entries")
            for e in new_entries[:3]:
                print(f"  [{e.get('confidence','?')}%] {e.get('text','')[:80]}")
    except Exception as e:
        print(f"[Mechanics] ERROR: {e}")

    # REPL Strategy Observer
    try:
        repl_result = repl_observer.observe(history, repl_traces)
        new_repl = repl_result.get("new_entries", [])
        if new_repl:
            history.repl_knowledge_base.extend(new_repl)
            print(f"[REPL] +{len(new_repl)} tips")
    except Exception as e:
        print(f"[REPL] ERROR: {e}")

    # Instruction Optimizer
    try:
        new_instructions = optimizer.optimize(history, vis_obs)
        history.solver_instructions = new_instructions
        print(f"[Optimizer] Updated instructions ({len(new_instructions)} chars)")
    except Exception as e:
        print(f"[Optimizer] ERROR: {e}")

    # Persist knowledge after each reflection
    history.save_knowledge()


def _print_final_stats(history: GameHistory, arcade: Arcade):
    """Print final game statistics."""
    print(f"\n{'='*70}")
    print(f"FINAL: {history.current_step} steps | {history.levels_completed}/{history.total_levels} levels")
    print(f"KB: {len(history.game_knowledge_base)} game entries, {len(history.repl_knowledge_base)} REPL tips")
    if history.level_attempts:
        print(f"Level attempts: {dict(history.level_attempts)}")
    if history.level_best_actions:
        for lvl, acts in history.level_best_actions.items():
            print(f"Level {lvl} best: {len(acts)} actions")
    print(f"{'='*70}")

    print(f"\n--- Game Knowledge ---")
    print(history.get_game_kb_text()[:1500])
    print(f"\n--- Solver Instructions ---")
    print(history.solver_instructions[:1500] if history.solver_instructions else "N/A")

    try:
        scorecard = arcade.get_scorecard()
        print(f"\n--- Scorecard ---")
        print(scorecard)
    except Exception:
        pass


def run_gepa_then_play(game_id: str = DEFAULT_GAME, max_steps: int = MAX_STEPS,
                       gepa_calls: int = 10, episode_steps: int = 80):
    """GEPA pre-optimization then full multi-agent play."""
    from gepa_optimizer import run_gepa_optimization

    print("=" * 70)
    print("PHASE 1: GEPA Strategy Evolution")
    print("=" * 70)
    result = run_gepa_optimization(
        game_id=game_id,
        max_metric_calls=gepa_calls,
        episode_steps=episode_steps,
    )

    evolved_strategy = result.best_candidate.get("strategy", "")
    print(f"\nEvolved strategy ({len(evolved_strategy)} chars)")

    print("\n" + "=" * 70)
    print("PHASE 2: Full Multi-Agent System")
    print("=" * 70)
    history = run_agent(game_id=game_id, max_steps=max_steps)
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ARC AGI 3 Multi-Agent System")
    parser.add_argument("--game", default=DEFAULT_GAME, help="Game ID")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="Max steps")
    parser.add_argument("--batch", type=int, default=RLM_BATCH_SIZE, help="Batch size")
    parser.add_argument("--reflect", type=int, default=REFLECT_EVERY, help="Reflect every N batches")
    parser.add_argument("--gepa", action="store_true", help="Run GEPA pre-optimization")
    parser.add_argument("--gepa-calls", type=int, default=10, help="GEPA metric calls")
    args = parser.parse_args()

    print(f"ARC AGI 3 Multi-Agent System")
    print(f"  Game: {args.game} | Steps: {args.steps} | Reflect: every {args.reflect} batches")
    print(f"  Model: {GEMINI_MODEL} | GEPA: {args.gepa}")
    print()

    if args.gepa:
        run_gepa_then_play(game_id=args.game, max_steps=args.steps, gepa_calls=args.gepa_calls)
    else:
        run_agent(game_id=args.game, max_steps=args.steps, batch_size=args.batch, reflect_every=args.reflect)
