"""Shared action utilities for the ARC AGI 3 multi-agent system.

Provides dynamic action map construction, formatting, and parsing
used by agent.py, solver.py, and gepa_optimizer.py.
"""

import re
from arcengine.enums import GameAction


def build_action_map(available_actions: list[int]) -> dict[str, GameAction]:
    """Build a name->GameAction map from the game's available_actions list.

    Args:
        available_actions: List of action IDs from obs.available_actions.

    Returns:
        Dict mapping "ACTION1" etc. to GameAction enum values.
    """
    action_map = {}
    for action_id in available_actions:
        action = GameAction.from_id(action_id)
        action_map[action.name] = action
    return action_map


def format_action_space(action_map: dict[str, GameAction]) -> str:
    """Format available actions as a human-readable string for LLM prompts.

    Returns something like:
        "Available actions: ACTION1, ACTION2, ACTION3, ACTION4"
    """
    names = sorted(action_map.keys(), key=lambda n: int(n.replace("ACTION", "")))
    return f"Available actions: {', '.join(names)}"


def parse_action_string(action_str: str) -> tuple[str, dict | None]:
    """Parse an action string, handling ACTION6(x,y) click coordinate format.

    Args:
        action_str: e.g. "ACTION3" or "ACTION6(32,16)"

    Returns:
        Tuple of (action_name, params_or_None).
        For "ACTION3" -> ("ACTION3", None)
        For "ACTION6(32,16)" -> ("ACTION6", {"x": 32, "y": 16})
    """
    action_str = action_str.strip().upper()
    m = re.match(r"(ACTION\d+)\((\d+)\s*,\s*(\d+)\)", action_str)
    if m:
        name = m.group(1)
        x, y = int(m.group(2)), int(m.group(3))
        return name, {"x": x, "y": y}
    # Strip any trailing parens/garbage
    clean = re.match(r"(ACTION\d+)", action_str)
    if clean:
        return clean.group(1), None
    return action_str, None


def get_valid_action_names(action_map: dict[str, GameAction]) -> set[str]:
    """Get the set of valid action name strings for validation."""
    return set(action_map.keys())
