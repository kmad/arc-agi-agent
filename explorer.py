"""Programmatic Pre-Exploration: Systematic action mapping without LLM.

Runs before the LLM solver to build a complete action-effect map.
For each action, executes it N times in isolation, records frame diffs,
then resets. Produces a pandas DataFrame that the RLM can query.

Strategy (game-agnostic):
    For each action in action_space:
        1. Reset the game to a clean state
        2. Record the initial frame
        3. Execute the action N times, recording each frame
        4. Compute per-step diffs (pixels changed, direction, region)
        5. Store results in a structured DataFrame

The resulting DataFrame has one row per (action, repetition) with columns:
    action, rep, pixels_changed, direction, min_row, max_row, min_col, max_col,
    color_transitions, blocked, cumulative_displacement_r, cumulative_displacement_c
"""

import numpy as np
import pandas as pd
from arcengine.enums import GameAction, GameState


def run_exploration(env, reps_per_action: int = 10) -> pd.DataFrame:
    """Run systematic action exploration and return a DataFrame of results.

    Args:
        env: An initialized arc_agi environment (already reset once).
        reps_per_action: How many times to repeat each action.

    Returns:
        DataFrame with one row per (action, rep).
    """
    actions = env.action_space
    rows = []

    for action in actions:
        action_name = action.name
        is_complex = action.is_complex()

        if is_complex:
            # For click-type actions, test a few canonical positions
            _explore_complex_action(env, action, action_name, rows, reps_per_action)
        else:
            _explore_simple_action(env, action, action_name, rows, reps_per_action)

    df = pd.DataFrame(rows)

    # Add summary columns
    if not df.empty:
        df["effective"] = df["pixels_changed"] > 4  # More than just progress bar
        df["no_change"] = df["pixels_changed"] == 0

    return df


def _explore_simple_action(env, action, action_name, rows, reps):
    """Explore a simple (non-click) action by repeating it N times from reset."""
    # Reset to clean state
    obs = env.reset()
    if obs is None:
        return
    initial_frame = np.array(obs.frame)[0]
    prev_frame = initial_frame.copy()

    # Find background color (most common)
    vals, counts = np.unique(initial_frame, return_counts=True)
    bg_color = int(vals[counts.argmax()])

    cumulative_dr = 0.0
    cumulative_dc = 0.0

    for rep in range(reps):
        obs = env.step(action)
        if obs is None:
            break

        frame = np.array(obs.frame)[0]
        row_data = _compute_diff(
            prev_frame, frame, action_name, rep, bg_color,
        )
        row_data["is_complex"] = False
        row_data["click_x"] = None
        row_data["click_y"] = None
        row_data["levels_completed"] = obs.levels_completed
        row_data["game_state"] = str(obs.state)

        # Track cumulative displacement
        cumulative_dr += row_data.get("displacement_r", 0)
        cumulative_dc += row_data.get("displacement_c", 0)
        row_data["cumulative_dr"] = cumulative_dr
        row_data["cumulative_dc"] = cumulative_dc

        rows.append(row_data)
        prev_frame = frame.copy()

        if obs.state == GameState.WIN or obs.state == GameState.GAME_OVER:
            break


def _explore_complex_action(env, action, action_name, rows, reps):
    """Explore a complex (click) action by testing canonical grid positions."""
    # Test clicks at center, corners, and edges
    test_positions = [
        (32, 32),  # center
        (16, 16),  # upper-left quadrant
        (16, 48),  # upper-right quadrant
        (48, 16),  # lower-left quadrant
        (48, 48),  # lower-right quadrant
    ]

    for x, y in test_positions:
        obs = env.reset()
        if obs is None:
            continue
        prev_frame = np.array(obs.frame)[0]
        vals, counts = np.unique(prev_frame, return_counts=True)
        bg_color = int(vals[counts.argmax()])

        for rep in range(min(reps, 3)):  # Fewer reps for click positions
            obs = env.step(action, data={"x": x, "y": y})
            if obs is None:
                break

            frame = np.array(obs.frame)[0]
            row_data = _compute_diff(prev_frame, frame, action_name, rep, bg_color)
            row_data["is_complex"] = True
            row_data["click_x"] = x
            row_data["click_y"] = y
            row_data["levels_completed"] = obs.levels_completed
            row_data["game_state"] = str(obs.state)
            row_data["cumulative_dr"] = 0
            row_data["cumulative_dc"] = 0

            rows.append(row_data)
            prev_frame = frame.copy()

            if obs.state == GameState.WIN or obs.state == GameState.GAME_OVER:
                break


def _compute_diff(prev_frame, curr_frame, action_name, rep, bg_color) -> dict:
    """Compute a structured diff between two frames."""
    diff_mask = prev_frame != curr_frame
    n_changed = int(diff_mask.sum())

    result = {
        "action": action_name,
        "rep": rep,
        "pixels_changed": n_changed,
        "direction": None,
        "displacement_r": 0.0,
        "displacement_c": 0.0,
        "min_row": None,
        "max_row": None,
        "min_col": None,
        "max_col": None,
        "color_transitions": "",
        "blocked": n_changed <= 4,  # Only progress bar changed
    }

    if n_changed == 0:
        result["blocked"] = True
        return result

    # Bounding box of changes
    changed_pos = np.argwhere(diff_mask)
    result["min_row"] = int(changed_pos[:, 0].min())
    result["max_row"] = int(changed_pos[:, 0].max())
    result["min_col"] = int(changed_pos[:, 1].min())
    result["max_col"] = int(changed_pos[:, 1].max())

    # Color transitions
    old_vals = prev_frame[diff_mask]
    new_vals = curr_frame[diff_mask]
    transitions = {}
    for ov, nv in zip(old_vals, new_vals):
        key = f"{int(ov)}->{int(nv)}"
        transitions[key] = transitions.get(key, 0) + 1
    # Top 3 transitions
    top = sorted(transitions.items(), key=lambda x: -x[1])[:3]
    result["color_transitions"] = "; ".join(f"{k}:{v}px" for k, v in top)

    # Movement direction detection
    new_positions = np.argwhere((prev_frame == bg_color) & (curr_frame != bg_color))
    old_positions = np.argwhere((prev_frame != bg_color) & (curr_frame == bg_color))

    if len(new_positions) > 0 and len(old_positions) > 0:
        new_center = new_positions.mean(axis=0)
        old_center = old_positions.mean(axis=0)
        dr = new_center[0] - old_center[0]
        dc = new_center[1] - old_center[1]
        result["displacement_r"] = float(dr)
        result["displacement_c"] = float(dc)

        if abs(dr) > abs(dc) and abs(dr) > 1:
            result["direction"] = "DOWN" if dr > 0 else "UP"
        elif abs(dc) > 1:
            result["direction"] = "RIGHT" if dc > 0 else "LEFT"

    return result


def summarize_exploration(df: pd.DataFrame) -> str:
    """Generate a text summary of exploration results for LLM consumption."""
    if df.empty:
        return "No exploration data."

    lines = ["=== Programmatic Exploration Results ===\n"]

    # Per-action summary
    for action_name in df["action"].unique():
        adf = df[df["action"] == action_name]
        lines.append(f"--- {action_name} ---")

        avg_px = adf["pixels_changed"].mean()
        blocked_pct = adf["blocked"].mean() * 100
        effective_pct = adf["effective"].mean() * 100 if "effective" in adf else 0

        lines.append(f"  Avg pixels changed: {avg_px:.1f}")
        lines.append(f"  Blocked: {blocked_pct:.0f}% | Effective: {effective_pct:.0f}%")

        # Direction consensus
        directions = adf["direction"].dropna()
        if not directions.empty:
            dir_counts = directions.value_counts()
            primary_dir = dir_counts.index[0]
            confidence = dir_counts.iloc[0] / len(adf) * 100
            lines.append(f"  Direction: {primary_dir} ({confidence:.0f}% of steps)")

            # Cumulative displacement
            last = adf.iloc[-1]
            lines.append(f"  Cumulative displacement: dr={last['cumulative_dr']:.1f}, dc={last['cumulative_dc']:.1f}")
        else:
            lines.append(f"  Direction: UNKNOWN (no clear movement detected)")

        # Color transitions
        all_transitions = adf["color_transitions"].dropna()
        if not all_transitions.empty:
            lines.append(f"  Typical color changes: {all_transitions.iloc[0]}")

        # Level completions
        if adf["levels_completed"].max() > 0:
            lines.append(f"  ** Completed level(s) during exploration! **")

        lines.append("")

    # Cross-action comparison
    lines.append("--- Cross-Action Summary ---")
    action_stats = df.groupby("action").agg(
        avg_pixels=("pixels_changed", "mean"),
        blocked_rate=("blocked", "mean"),
        primary_direction=("direction", lambda x: x.mode().iloc[0] if not x.mode().empty else "UNKNOWN"),
    ).round(1)
    lines.append(action_stats.to_string())

    return "\n".join(lines)


def exploration_to_knowledge(df: pd.DataFrame) -> list[dict]:
    """Convert exploration DataFrame to KnowledgeEntry-compatible dicts."""
    from models import KnowledgeCategory

    entries = []

    for action_name in df["action"].unique():
        adf = df[df["action"] == action_name]

        avg_px = adf["pixels_changed"].mean()
        blocked_pct = adf["blocked"].mean() * 100
        directions = adf["direction"].dropna()

        # Direction mapping
        if not directions.empty:
            dir_counts = directions.value_counts()
            primary_dir = dir_counts.index[0]
            confidence = min(95, int(dir_counts.iloc[0] / len(adf) * 100) + 20)
            entries.append({
                "category": KnowledgeCategory.PLAYER_MECHANICS.value,
                "text": f"{action_name} moves {primary_dir} (~{avg_px:.0f}px per step)",
                "confidence": confidence,
                "source": "programmatic_exploration",
            })
        elif blocked_pct > 80:
            entries.append({
                "category": KnowledgeCategory.GAME_RULES.value,
                "text": f"{action_name} appears blocked or ineffective ({blocked_pct:.0f}% blocked)",
                "confidence": 70,
                "source": "programmatic_exploration",
            })
        else:
            entries.append({
                "category": KnowledgeCategory.OBSERVATION.value,
                "text": f"{action_name} causes {avg_px:.0f}px change on average, no clear direction",
                "confidence": 40,
                "source": "programmatic_exploration",
            })

    return entries
