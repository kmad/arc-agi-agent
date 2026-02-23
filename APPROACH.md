# StrategicReasoner: RLM-Powered Knowledge Synthesis

## Problem

The multi-agent orchestrator's `KnowledgeMerger` is purely mechanical:
- Fingerprint-based dedup, keep highest confidence, pick longest instructions
- Zero reasoning at merge time
- All agents receive the same merged knowledge blob
- Conflicting knowledge resolved by confidence score alone
- No targeted instructions per agent
- Can't identify gaps in exploration or generate hypotheses

## Solution

`StrategicReasoner` replaces `KnowledgeMerger` as the knowledge synthesis layer. It uses
DSPy's RLM (Recursive Language Model) to reason over agent reports iteratively, producing
per-agent directives instead of a uniform knowledge blob.

## Architecture

```
  Before (KnowledgeMerger):
    Reports --> merge_knowledge() --> same blob --> all agents

  After (StrategicReasoner):
    Reports --> reason() --> per-agent directives --> each agent
               |                                        |
               +-- ReasoningContext (persists across cycles) --+
```

## How It Works

### 1. Report Collection

Each sync cycle, the orchestrator collects `AgentReport` from all running agents.
Reports contain: knowledge entries, REPL tips, instructions, step count, level progress.

### 2. Strategic Reasoning (RLM)

The `StrategicReasoner.reason()` method:

1. **Builds a report summary**: Per-agent progress, top knowledge by confidence,
   cross-agent analysis (shared vs unique findings).

2. **Serializes reasoning context**: The growing `ReasoningContext` from previous cycles
   includes strategy evolution notes, confirmed knowledge, unresolved conflicts, open gaps,
   and per-agent performance trends.

3. **Calls the RLM**: For cycle > 0 (when context exists), the DSPy RLM iterates over
   the data using `llm_query()` sub-calls. For the first cycle, a single-pass `Predict`
   is used (less overhead when context is minimal).

4. **Parses directives**: The RLM outputs `strategy_notes` and `directives_json` (a JSON
   array of `AgentDirective` objects).

5. **Updates context**: Increments cycle number, appends strategy notes, tracks agent
   performance trends (improving/stable/regressing), updates confirmed knowledge.

### 3. Per-Agent Directives

Each `AgentDirective` contains:
- `focus_area`: What this agent should prioritize (e.g., "explore level 2 mechanics")
- `avoid`: Actions/strategies proven ineffective (e.g., ["repeated ACTION1 spam"])
- `try_actions`: Specific actions to attempt (e.g., ["ACTION3", "ACTION4 then ACTION2"])
- `updated_instructions`: Full instructions override if the agent needs a new strategy
- `knowledge_to_inject`: Curated knowledge entries relevant to this agent's focus

### 4. Directive Application (Agent Side)

When an agent loads injected knowledge (`_load_injected_knowledge` in `agent.py`), it
checks for a `directive` field. If present:
- `GameHistory.apply_directive()` stores the directive
- Directive knowledge is injected into the agent's knowledge base
- If `updated_instructions` is set, it overrides `solver_instructions`
- The `focus_area` and `avoid` fields appear in `get_state_summary()`, influencing
  the solver's next decisions

### 5. Growing Context

The `ReasoningContext` persists across sync cycles within a single orchestrator run:
- **Strategy evolution**: One note per cycle tracking how the approach is changing
- **Confirmed knowledge**: High-confidence (>=80%) entries agreed upon across agents
- **Unresolved conflicts**: Knowledge entries that contradict each other
- **Open gaps**: Areas that need more exploration
- **Agent performance**: Per-agent trend tracking (levels, steps, direction)

## Fallback

If the RLM fails (timeout, parse error, import error), the system falls back to
the original `KnowledgeMerger` behavior: mechanical fingerprint dedup, same blob to
all agents. The system never breaks.

The fallback is triggered in two places:
1. `StrategicReasoner.reason()` catches exceptions and calls `_fallback_merge()`
2. `MultiAgentOrchestrator.__init__()` falls back to `KnowledgeMerger` if `StrategicReasoner`
   fails to initialize

## Configuration

In `config.py`:
- `STRATEGIC_REASONER_ENABLED = True` — toggle the reasoner on/off
- `STRATEGIC_REASONER_MAX_ITERATIONS = 8` — RLM iteration budget per sync cycle

Set `STRATEGIC_REASONER_ENABLED = False` to revert to pure mechanical merge.

## Key Files

| File | Role |
|------|------|
| `strategic_reasoner.py` | Core RLM reasoning module |
| `models.py` | Data models (AgentDirective, ReasoningContext, etc.) |
| `orchestrator.py` | Sync cycle dispatches to reasoner or merger |
| `agent.py` | Loads and applies directives from injected data |
| `game_state.py` | Stores active directive, surfaces it in state summary |
| `config.py` | Feature flag and iteration config |
