"""Pydantic models for the ARC AGI 3 multi-agent system.

Provides typed data structures used across solver, observers, orchestrator,
and agent communication. Replaces ad-hoc dicts with validated models.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import hashlib


# ── Enums ─────────────────────────────────────────────────────────

class MovementDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class KnowledgeCategory(str, Enum):
    PLAYER_MECHANICS = "player_mechanics"
    GAME_RULES = "game_rules"
    LEVEL_STRUCTURE = "level_structure"
    WIN_CONDITION = "win_condition"
    OBSTACLE = "obstacle"
    ITEM = "item"
    OBSERVATION = "observation"


class REPLCategory(str, Enum):
    ANALYSIS_PATTERN = "analysis_pattern"
    EFFICIENCY_TIP = "efficiency_tip"
    CODE_TEMPLATE = "code_template"
    ANTI_PATTERN = "anti_pattern"
    GAME_SPECIFIC_TOOL = "game_specific_tool"
    TIP = "tip"


class AgentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    IDLE = "idle"


# ── Solver Models ─────────────────────────────────────────────────

class ActionEffect(BaseModel):
    """Effect observed from executing a single action."""
    action: str
    direction: Optional[MovementDirection] = None
    magnitude: float = 0.0
    affected_rows: Optional[tuple[int, int]] = None
    affected_cols: Optional[tuple[int, int]] = None
    blocked: bool = False
    observations: list[str] = Field(default_factory=list)


class ColorRegion(BaseModel):
    """A contiguous region of a single color on the grid."""
    color_value: int
    color_name: str
    pixel_count: int
    bounding_box: tuple[int, int, int, int]  # min_r, min_c, max_r, max_c
    shape: str
    fill_ratio: float


class FrameAnalysis(BaseModel):
    """Programmatic analysis of a single game frame."""
    grid_size: tuple[int, int] = (64, 64)
    background: tuple[int, str] = (0, "black")  # (color_value, color_name)
    regions: list[ColorRegion] = Field(default_factory=list)
    movement: Optional[MovementDirection] = None
    pixels_changed: int = 0
    change_region: Optional[tuple[int, int, int, int]] = None  # min_r, min_c, max_r, max_c
    raw_text: str = ""  # The full text analysis for backward compat

    def to_prompt_text(self) -> str:
        """Convert to text for LLM prompts. Falls back to raw_text if available."""
        if self.raw_text:
            return self.raw_text
        lines = [f"Grid: {self.grid_size[0]}x{self.grid_size[1]}"]
        bg_val, bg_name = self.background
        lines.append(f"Background: {bg_val}={bg_name}")
        for r in self.regions:
            min_r, min_c, max_r, max_c = r.bounding_box
            h, w = max_r - min_r + 1, max_c - min_c + 1
            lines.append(
                f"  {r.color_value}={r.color_name}: {r.pixel_count}px, "
                f"rows {min_r}-{max_r} cols {min_c}-{max_c} ({h}x{w}), "
                f"{r.shape}, fill={r.fill_ratio:.0%}"
            )
        if self.pixels_changed > 0 and self.change_region:
            cr = self.change_region
            lines.append(f"Diff: {self.pixels_changed} pixels changed, region rows {cr[0]}-{cr[2]} cols {cr[1]}-{cr[3]}")
        if self.movement:
            lines.append(f"  Movement: {self.movement.value}")
        return "\n".join(lines)


class GameHypothesis(BaseModel):
    """A hypothesis about a game mechanic."""
    mechanic: str
    description: str
    confidence: int = Field(ge=0, le=100)
    test: str = ""
    evidence: list[str] = Field(default_factory=list)
    confirmed: bool = False


# ── Knowledge Models ──────────────────────────────────────────────

class KnowledgeEntry(BaseModel):
    """A single entry in the game knowledge base."""
    category: KnowledgeCategory = KnowledgeCategory.OBSERVATION
    text: str
    confidence: int = Field(default=50, ge=0, le=100)
    source: str = "unknown"
    step: int = 0
    game_id: str = ""
    agent_id: str = ""
    fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        """Generate a dedup fingerprint from category + text."""
        raw = f"{self.category.value}:{self.text.lower().strip()}"
        self.fingerprint = hashlib.md5(raw.encode()).hexdigest()[:12]
        return self.fingerprint


class REPLTip(BaseModel):
    """A single entry in the REPL knowledge base."""
    category: REPLCategory = REPLCategory.TIP
    text: str
    code_example: str = ""


# ── Observer Output Models ────────────────────────────────────────

class VisualAnalysis(BaseModel):
    """Output from the visual observer."""
    visual_analysis: str = ""
    player_position: str = ""
    goal_description: str = ""
    recommended_strategy: str = ""


class MechanicsAnalysis(BaseModel):
    """Output from the game mechanics observer."""
    new_entries: list[KnowledgeEntry] = Field(default_factory=list)
    mechanics_summary: str = ""


class REPLAnalysis(BaseModel):
    """Output from the REPL strategy observer."""
    new_entries: list[REPLTip] = Field(default_factory=list)
    repl_strategy_summary: str = ""


# ── Agent Communication Models ────────────────────────────────────

class GameAssignment(BaseModel):
    """A game assigned to an agent by the orchestrator."""
    game_id: str
    max_steps: int = 500
    batch_size: int = 5
    reflect_every: int = 3


class AgentReport(BaseModel):
    """Report from an agent after a sync cycle or completion."""
    agent_id: str
    game_id: str
    status: AgentStatus = AgentStatus.RUNNING
    steps_taken: int = 0
    levels_completed: int = 0
    total_levels: int = 0
    knowledge: list[KnowledgeEntry] = Field(default_factory=list)
    repl_tips: list[REPLTip] = Field(default_factory=list)
    instructions: str = ""


class OrchestratorState(BaseModel):
    """State of the multi-agent orchestrator."""
    mode: str = "local"  # "local" or "sandbox"
    num_agents: int = 4
    games: list[str] = Field(default_factory=list)
    sync_interval: int = 60
    agents: dict[str, AgentStatus] = Field(default_factory=dict)
    merged_knowledge: list[KnowledgeEntry] = Field(default_factory=list)
    merged_repl_tips: list[REPLTip] = Field(default_factory=list)
    total_steps: int = 0
    total_levels: int = 0


# ── Strategic Reasoner Models ────────────────────────────────────

class KnowledgeConflict(BaseModel):
    """Two knowledge entries that contradict each other."""
    entry_a: KnowledgeEntry
    entry_b: KnowledgeEntry
    resolution: str = ""


class KnowledgeGap(BaseModel):
    """An identified gap in the knowledge base."""
    description: str
    suggested_exploration: str
    priority: int = Field(default=50, ge=0, le=100)


class AgentDirective(BaseModel):
    """Targeted instructions for a specific agent."""
    agent_id: str
    focus_area: str  # What to prioritize
    avoid: list[str] = Field(default_factory=list)  # What not to waste time on
    try_actions: list[str] = Field(default_factory=list)  # Specific actions to attempt
    knowledge_to_inject: list[KnowledgeEntry] = Field(default_factory=list)
    updated_instructions: str = ""  # Agent-specific instructions override


class ReasoningContext(BaseModel):
    """Growing context maintained across sync cycles."""
    cycle_number: int = 0
    strategy_evolution: list[str] = Field(default_factory=list)  # One entry per cycle
    confirmed_knowledge: list[KnowledgeEntry] = Field(default_factory=list)
    unresolved_conflicts: list[KnowledgeConflict] = Field(default_factory=list)
    open_gaps: list[KnowledgeGap] = Field(default_factory=list)
    agent_performance: dict[str, dict] = Field(default_factory=dict)  # agent_id -> {levels, steps, trend}


class StrategicReasoningOutput(BaseModel):
    """Output from the StrategicReasoner."""
    merged_knowledge: list[KnowledgeEntry] = Field(default_factory=list)
    merged_repl_tips: list[REPLTip] = Field(default_factory=list)
    directives: list[AgentDirective] = Field(default_factory=list)
    strategy_notes: str = ""
    conflicts_found: list[KnowledgeConflict] = Field(default_factory=list)
    gaps_found: list[KnowledgeGap] = Field(default_factory=list)
