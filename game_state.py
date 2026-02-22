"""Game state tracking and history management."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json
import io


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

    def get_state_summary(self) -> str:
        """Get a compact summary of current game state."""
        lines = [
            f"Game: {self.game_id}",
            f"Step: {self.current_step}",
            f"Levels completed: {self.levels_completed}/{self.total_levels}",
            f"State: {self.steps[-1].state if self.steps else 'NOT_STARTED'}",
        ]
        if self.steps:
            recent = self.get_recent_actions(10)
            lines.append(f"Recent actions: {', '.join(recent)}")
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


def compute_grid_diff(prev: np.ndarray, curr: np.ndarray) -> str:
    """Compute a human-readable diff between two 64x64 grids."""
    if prev is None:
        return "Initial frame."
    diff = prev != curr
    if not diff.any():
        return "No change."
    changed_positions = np.argwhere(diff)
    n_changed = len(changed_positions)
    summary = f"{n_changed} pixels changed."
    if n_changed <= 30:
        details = []
        for pos in changed_positions:
            r, c = pos
            details.append(f"({r},{c}): {prev[r, c]}->{curr[r, c]}")
        summary += " " + "; ".join(details)
    else:
        # Show bounding box of changes
        min_r, min_c = changed_positions.min(axis=0)
        max_r, max_c = changed_positions.max(axis=0)
        summary += f" Region: rows {min_r}-{max_r}, cols {min_c}-{max_c}"
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
