"""Configuration for the ARC AGI 3 multi-agent system."""

import os
from dotenv import load_dotenv

load_dotenv()

# ARC API
ARC_API_KEY = os.getenv("ARC_API_KEY")

# Gemini configuration (via litellm/dspy)
GEMINI_MODEL = "gemini/gemini-2.5-flash"
GEMINI_MODEL_MINI = "gemini/gemini-2.5-flash"

# RLM settings
RLM_MAX_ITERATIONS = 25
RLM_BATCH_SIZE = 5  # Number of moves between reflection cycles

# Game settings
DEFAULT_GAME = "ls20"
MAX_STEPS = 500

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
