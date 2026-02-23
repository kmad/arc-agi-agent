"""Configuration for the ARC AGI 3 multi-agent system."""

import os
from dotenv import load_dotenv

load_dotenv()

# ARC API
ARC_API_KEY = os.getenv("ARC_API_KEY")

# LLM configuration (via litellm/dspy)
MAIN_MODEL = "openrouter/google/gemini-3-flash-preview"
SUB_MODEL = "openrouter/google/gemini-3-flash-preview"
# Legacy aliases
GEMINI_MODEL = MAIN_MODEL
GEMINI_MODEL_MINI = SUB_MODEL

# RLM settings
RLM_MAX_ITERATIONS = 25
RLM_BATCH_SIZE = 5  # Number of moves between reflection cycles

# Game settings
DEFAULT_GAME = "ls20"
MAX_STEPS = 500

# Multi-agent orchestrator settings
NUM_AGENTS = 4
SYNC_INTERVAL_SECONDS = 60
ORCHESTRATOR_MODE = "local"  # "local" or "sandbox"
DAYTONA_API_KEY = os.getenv("DAYTONA_API_KEY", "")

# Strategic Reasoner settings
STRATEGIC_REASONER_ENABLED = True
STRATEGIC_REASONER_MAX_ITERATIONS = 8

# Color map for ARC grids (index -> name)
# Default action labels for display (game-agnostic)
DEFAULT_ACTION_LABELS = {
    "ACTION1": "action 1",
    "ACTION2": "action 2",
    "ACTION3": "action 3",
    "ACTION4": "action 4",
    "ACTION5": "action 5",
    "ACTION6": "action 6 (click)",
}

# Color map for ARC grids (index -> name)
COLOR_MAP = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "brown",
    10: "white",
    11: "maroon",
    12: "dark_yellow",
}
