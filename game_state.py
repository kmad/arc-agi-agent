"""Game state tracking and history management."""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json
import io

# Directory for persisting knowledge between runs
KB_DIR = "knowledge_base"


@dataclass
class StepRecord:
    """Record of a single game step."""
    step_number: int
    action: str
    frame: np.ndarray
    state: str
    levels_completed: int
    grid_diff: Optional[str] = None  # Diff from previous frame


@dataclass
class GameHistory:
    """Full game history and knowledge bases."""
    game_id: str
    steps: list[StepRecord] = field(default_factory=list)
    game_knowledge_base: list[dict] = field(default_factory=list)
    repl_knowledge_base: list[dict] = field(default_factory=list)
    solver_instructions: str = ""
    total_levels: int = 0
    levels_completed: int = 0

    # Per-level tracking
    level_knowledge: dict[int, list[dict]] = field(default_factory=dict)
    level_attempts: dict[int, int] = field(default_factory=lambda: {})
    level_best_actions: dict[int, list[str]] = field(default_factory=dict)

    def add_step(self, step: StepRecord):
        self.steps.append(step)
        self.levels_completed = step.levels_completed

    @property
    def current_step(self) -> int:
        return len(self.steps)

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        if self.steps:
            return self.steps[-1].frame
        return None

    @property
    def last_n_frames(self, n: int = 5) -> list[np.ndarray]:
        return [s.frame for s in self.steps[-n:]]

    def get_recent_actions(self, n: int = 20) -> list[str]:
        return [s.action for s in self.steps[-n:]]

    def get_recent_action_diffs(self, n: int = 15) -> str:
        """Get recent action->diff pairs so the solver can see what worked."""
        lines = []
        for step in self.steps[-n:]:
            if step.action == "RESET":
                continue
            diff = step.grid_diff or "?"
            # Include the full diff info (now includes directional shift data)
            lines.append(f"  Step {step.step_number}: {step.action} -> {diff[:150]}")
        return "\n".join(lines) if lines else "No history yet."

    def is_stuck(self, window: int = 10) -> tuple[bool, str]:
        """Detect if the agent is stuck repeating unproductive actions."""
        recent = self.steps[-window:]
        if len(recent) < window:
            return False, ""

        actions = [s.action for s in recent if s.action != "RESET"]
        if not actions:
            return False, ""

        # Check if all actions are the same
        from collections import Counter
        counts = Counter(actions)
        dominant_action, dominant_count = counts.most_common(1)[0]

        if dominant_count < window * 0.8:
            return False, ""

        # Check if diffs are all tiny (<=4 pixels = just progress bar)
        import re
        tiny_count = 0
        for s in recent:
            if s.grid_diff and "pixels changed" in s.grid_diff:
                m = re.search(r"(\d+) pixels changed", s.grid_diff)
                if m and int(m.group(1)) <= 4:
                    tiny_count += 1

        if tiny_count >= window * 0.7:
            return True, (
                f"STUCK: Last {window} actions were mostly {dominant_action} "
                f"with only tiny pixel changes (progress bar). "
                f"The player is NOT moving. Try different actions!"
            )

        return False, ""

    def detect_level_transition(
        self,
        prev_levels: int,
        new_levels: int,
        prev_frame: np.ndarray,
        new_frame: np.ndarray,
    ) -> str:
        """Classify what kind of transition happened between two steps.

        Returns one of:
            "level_complete" - a new level was completed
            "full_reset" - game reset to beginning (levels went backwards)
            "level_reset" - same level count but big frame change (level restarted)
            "none" - no transition detected
        """
        if new_levels > prev_levels:
            return "level_complete"

        if new_levels < prev_levels:
            return "full_reset"

        # Check for large frame change that could indicate level reset
        if prev_frame is not None and new_frame is not None:
            diff = prev_frame != new_frame
            changed = np.sum(diff)
            total = prev_frame.size
            # If >50% of pixels changed, likely a reset
            if changed > total * 0.5:
                return "level_reset"

        return "none"

    def on_level_complete(self, level_num: int, actions: list[str]) -> None:
        """Record that a level was completed with the given action sequence."""
        if level_num not in self.level_knowledge:
            self.level_knowledge[level_num] = []

        self.level_knowledge[level_num].append({
            "completed_at_step": self.current_step,
            "action_count": len(actions),
        })

        # Track best (shortest) action sequence for this level
        if level_num not in self.level_best_actions or len(actions) < len(self.level_best_actions[level_num]):
            self.level_best_actions[level_num] = actions

    def get_level_knowledge(self, level_num: int) -> list[dict]:
        """Get accumulated knowledge for a specific level."""
        return self.level_knowledge.get(level_num, [])

    def get_state_summary(self) -> str:
        """Get a compact summary of current game state."""
        current_level = self.levels_completed + 1
        attempts = self.level_attempts.get(current_level, 0)

        lines = [
            f"Game: {self.game_id}",
            f"Step: {self.current_step}",
            f"Levels completed: {self.levels_completed}/{self.total_levels}",
            f"Current level: {current_level} (attempt #{attempts + 1})",
            f"State: {self.steps[-1].state if self.steps else 'NOT_STARTED'}",
        ]

        # Add level-specific knowledge if available
        level_kb = self.get_level_knowledge(current_level)
        if level_kb:
            lines.append(f"Level {current_level} knowledge: {len(level_kb)} entries")

        if self.steps:
            recent = self.get_recent_actions(10)
            lines.append(f"Recent actions: {', '.join(recent)}")

            # Add stuck warning if applicable
            stuck, msg = self.is_stuck()
            if stuck:
                lines.append(f"\n*** WARNING: {msg} ***")

        return "\n".join(lines)

    def get_game_kb_text(self) -> str:
        """Format game knowledge base as text."""
        if not self.game_knowledge_base:
            return "No game knowledge accumulated yet."
        entries = []
        for entry in self.game_knowledge_base:
            confidence = entry.get("confidence", "?")
            category = entry.get("category", "observation")
            text = entry.get("text", "")
            source = entry.get("source", "unknown")
            entries.append(f"[{confidence}%] [{category}] {text} (source: {source}, step {entry.get('step', '?')})")
        return "\n".join(entries)

    def get_repl_kb_text(self) -> str:
        """Format REPL knowledge base as text."""
        if not self.repl_knowledge_base:
            return "No REPL knowledge accumulated yet."
        entries = []
        for entry in self.repl_knowledge_base:
            category = entry.get("category", "tip")
            text = entry.get("text", "")
            entries.append(f"[{category}] {text}")
        return "\n".join(entries)

    def save_knowledge(self):
        """Persist knowledge bases to disk for bootstrapping future runs."""
        os.makedirs(KB_DIR, exist_ok=True)
        kb_path = os.path.join(KB_DIR, f"{self.game_id}.json")
        data = {
            "game_id": self.game_id,
            "game_knowledge_base": self.game_knowledge_base,
            "repl_knowledge_base": self.repl_knowledge_base,
            "solver_instructions": self.solver_instructions,
            "total_steps": self.current_step,
            "levels_completed": self.levels_completed,
            "level_knowledge": {str(k): v for k, v in self.level_knowledge.items()},
            "level_attempts": {str(k): v for k, v in self.level_attempts.items()},
            "level_best_actions": {str(k): v for k, v in self.level_best_actions.items()},
        }
        with open(kb_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_knowledge(self) -> bool:
        """Load persisted knowledge from a previous run. Returns True if loaded."""
        kb_path = os.path.join(KB_DIR, f"{self.game_id}.json")
        if not os.path.exists(kb_path):
            return False
        try:
            with open(kb_path) as f:
                data = json.load(f)
            self.game_knowledge_base = data.get("game_knowledge_base", [])
            self.repl_knowledge_base = data.get("repl_knowledge_base", [])
            self.solver_instructions = data.get("solver_instructions", "")
            # Load level-specific data
            self.level_knowledge = {int(k): v for k, v in data.get("level_knowledge", {}).items()}
            self.level_attempts = {int(k): v for k, v in data.get("level_attempts", {}).items()}
            self.level_best_actions = {int(k): v for k, v in data.get("level_best_actions", {}).items()}
            return True
        except (json.JSONDecodeError, KeyError):
            return False


def analyze_frame(frame: np.ndarray, prev_frame: np.ndarray = None) -> str:
    """Programmatic frame analysis: extracts objects, colors, regions, diffs.

    Returns a structured text summary that replaces sending the raw grid to the LLM.
    The LLM interprets meaning; numpy does the pixel math.
    """
    from config import COLOR_MAP
    lines = []

    # 1. Grid overview
    unique_vals = np.unique(frame)
    color_names = [f"{v}={COLOR_MAP.get(v, f'color{v}')}" for v in unique_vals]
    lines.append(f"Grid: 64x64, colors present: {', '.join(color_names)}")

    # 2. Color cluster analysis — find distinct objects by connected regions
    bg_color = 0
    # Find the most common color as potential background
    vals, counts = np.unique(frame, return_counts=True)
    bg_color = vals[counts.argmax()]
    lines.append(f"Background: {bg_color}={COLOR_MAP.get(int(bg_color), '?')} ({counts.max()} pixels, {counts.max()*100//frame.size}%)")

    # 3. Per-color region summary
    for val in unique_vals:
        if val == bg_color:
            continue
        mask = frame == val
        pixel_count = mask.sum()
        positions = np.argwhere(mask)
        min_r, min_c = positions.min(axis=0)
        max_r, max_c = positions.max(axis=0)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        color_name = COLOR_MAP.get(int(val), f"color{val}")

        # Classify shape
        bbox_area = height * width
        fill_ratio = pixel_count / max(bbox_area, 1)

        if pixel_count <= 6:
            shape = "dot/small sprite"
        elif fill_ratio > 0.8 and height <= 3:
            shape = "horizontal bar"
        elif fill_ratio > 0.8 and width <= 3:
            shape = "vertical bar"
        elif fill_ratio > 0.7:
            shape = "filled rectangle"
        elif fill_ratio < 0.3:
            shape = "sparse/scattered"
        else:
            shape = "irregular shape"

        lines.append(
            f"  {val}={color_name}: {pixel_count}px, rows {min_r}-{max_r} cols {min_c}-{max_c} "
            f"({height}x{width}), {shape}, fill={fill_ratio:.0%}"
        )

    # 4. Frame diff (if previous frame provided)
    if prev_frame is not None:
        diff_mask = frame != prev_frame
        n_changed = diff_mask.sum()
        if n_changed == 0:
            lines.append("Diff: No change from previous frame.")
        else:
            changed_pos = np.argwhere(diff_mask)
            min_r, min_c = changed_pos.min(axis=0)
            max_r, max_c = changed_pos.max(axis=0)
            lines.append(f"Diff: {n_changed} pixels changed, region rows {min_r}-{max_r} cols {min_c}-{max_c}")

            # What colors changed to what
            old_vals = prev_frame[diff_mask]
            new_vals = frame[diff_mask]
            transitions = {}
            for ov, nv in zip(old_vals, new_vals):
                key = (int(ov), int(nv))
                transitions[key] = transitions.get(key, 0) + 1
            for (ov, nv), count in sorted(transitions.items(), key=lambda x: -x[1])[:5]:
                old_name = COLOR_MAP.get(ov, f"c{ov}")
                new_name = COLOR_MAP.get(nv, f"c{nv}")
                lines.append(f"    {old_name}->{new_name}: {count}px")

            # Detect movement direction from diff pattern
            # New pixels appearing (were background, now object) = destination
            # Old pixels disappearing (were object, now background) = source
            new_positions = np.argwhere((prev_frame == bg_color) & (frame != bg_color))
            old_positions = np.argwhere((prev_frame != bg_color) & (frame == bg_color))
            if len(new_positions) > 0 and len(old_positions) > 0:
                new_center = new_positions.mean(axis=0)
                old_center = old_positions.mean(axis=0)
                dr = new_center[0] - old_center[0]
                dc = new_center[1] - old_center[1]
                direction = []
                if abs(dr) > 1:
                    direction.append("DOWN" if dr > 0 else "UP")
                if abs(dc) > 1:
                    direction.append("RIGHT" if dc > 0 else "LEFT")
                if direction:
                    lines.append(f"  Movement: {'+'.join(direction)} (dr={dr:.1f}, dc={dc:.1f})")

    return "\n".join(lines)


def compute_grid_diff(prev: np.ndarray, curr: np.ndarray) -> str:
    """Compute a human-readable diff between two 64x64 grids."""
    if prev is None:
        return "Initial frame."
    diff = prev != curr
    if not diff.any():
        return "No change."
    changed_positions = np.argwhere(diff)
    n_changed = len(changed_positions)

    if n_changed <= 30:
        details = []
        for pos in changed_positions:
            r, c = pos
            details.append(f"({r},{c}): {prev[r, c]}->{curr[r, c]}")
        return f"{n_changed} pixels changed. " + "; ".join(details)

    # Show bounding box of changes
    min_r, min_c = changed_positions.min(axis=0)
    max_r, max_c = changed_positions.max(axis=0)
    summary = f"{n_changed} pixels changed. Region: rows {min_r}-{max_r}, cols {min_c}-{max_c}"

    # Add directional shift info: detect if objects moved left/right/up/down
    # by checking which rows have changes and the column shift pattern
    changed_rows = np.unique(changed_positions[:, 0])
    row_span = f"rows {changed_rows[0]}-{changed_rows[-1]}" if len(changed_rows) > 1 else f"row {changed_rows[0]}"

    # Check for consistent column shift (objects sliding)
    old_vals = prev[diff]
    new_vals = curr[diff]
    # Find columns that gained/lost pixels
    col_changes = changed_positions[:, 1]
    left_edge = col_changes.min()
    right_edge = col_changes.max()
    summary += f". Affected {row_span}, cols {left_edge}-{right_edge}"

    return summary


def frame_to_text(frame: np.ndarray, compact: bool = True) -> str:
    """Convert a 64x64 frame to a compact text representation."""
    if compact:
        # Show only non-zero regions
        nonzero = np.argwhere(frame != 0)
        if len(nonzero) == 0:
            return "Empty grid (all zeros)."
        min_r, min_c = nonzero.min(axis=0)
        max_r, max_c = nonzero.max(axis=0)
        # Add 1 pixel border
        min_r = max(0, min_r - 1)
        min_c = max(0, min_c - 1)
        max_r = min(63, max_r + 1)
        max_c = min(63, max_c + 1)
        sub = frame[min_r:max_r+1, min_c:max_c+1]
        header = f"Active region: rows {min_r}-{max_r}, cols {min_c}-{max_c} ({sub.shape[0]}x{sub.shape[1]})\n"
        buf = io.StringIO()
        for row in sub:
            buf.write(" ".join(f"{v:2d}" for v in row) + "\n")
        return header + buf.getvalue()
    else:
        buf = io.StringIO()
        for row in frame:
            buf.write(" ".join(f"{v:2d}" for v in row) + "\n")
        return buf.getvalue()


def frame_to_image_bytes(frame: np.ndarray) -> bytes:
    """Convert a 64x64 grid to a PNG image for visual analysis."""
    from PIL import Image

    # ARC color palette (approximate)
    palette = {
        0: (0, 0, 0),        # black
        1: (0, 116, 217),    # blue
        2: (255, 65, 54),    # red
        3: (46, 204, 64),    # green
        4: (255, 220, 0),    # yellow
        5: (128, 128, 128),  # grey
        6: (177, 13, 201),   # magenta
        7: (255, 133, 27),   # orange
        8: (0, 255, 255),    # cyan
        9: (139, 69, 19),    # brown
        10: (255, 255, 255), # white
        11: (128, 0, 0),     # maroon
        12: (204, 204, 0),   # dark yellow
    }

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for val, color in palette.items():
        mask = frame == val
        img[mask] = color

    # Scale up 8x for visibility
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((512, 512), Image.NEAREST)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()
