"""Multi-Agent Orchestrator for ARC AGI 3.

Spawns N agents (local processes or Daytona sandboxes) playing the same game,
periodically merges their knowledge bases, and redistributes merged knowledge
to all agents.

Architecture:
    orchestrator.py
    ┌──────────────────────────────┐
    │  MultiAgentOrchestrator      │
    │  - KnowledgeMerger           │
    │  - LocalAgentRunner OR       │
    │    SandboxAgentRunner        │
    └────────┬─────────────────────┘
             │ spawn / sync / collect
    ┌────────┼────────┐
    Agent 0  Agent 1  Agent N
    └────────┼────────┘
             │
       Merged Knowledge Base

Sync cycle (configurable interval):
  1. Collect reports from all agents
  2. Merge knowledge (dedup by fingerprint, keep highest confidence)
  3. Redistribute merged knowledge to all agents

Usage:
    python orchestrator.py --games ls20 --agents 4 --steps 500 --mode local
    python orchestrator.py --games ls20 --agents 2 --steps 500 --mode sandbox
"""

import os
import sys
import json
import time
import tempfile
import multiprocessing
import threading
from typing import Optional

from models import (
    KnowledgeEntry, REPLTip, AgentReport, AgentStatus,
    OrchestratorState, AgentDirective,
)
from config import (
    NUM_AGENTS, SYNC_INTERVAL_SECONDS, ORCHESTRATOR_MODE,
    STRATEGIC_REASONER_ENABLED, STRATEGIC_REASONER_MAX_ITERATIONS,
    GEMINI_MODEL, GEMINI_MODEL_MINI,
)


# ── Knowledge Merger ──────────────────────────────────────────────

class KnowledgeMerger:
    """Merges knowledge from multiple agents with fingerprint-based dedup."""

    def merge_knowledge(self, reports: list[AgentReport]) -> list[KnowledgeEntry]:
        """Merge knowledge entries from all agent reports.

        Dedup by fingerprint, keeping the highest confidence version.
        Sort by confidence descending.
        """
        best_by_fp: dict[str, KnowledgeEntry] = {}

        for report in reports:
            for entry in report.knowledge:
                if not entry.fingerprint:
                    entry.compute_fingerprint()
                fp = entry.fingerprint
                existing = best_by_fp.get(fp)
                if existing is None or entry.confidence > existing.confidence:
                    best_by_fp[fp] = entry

        merged = sorted(best_by_fp.values(), key=lambda e: e.confidence, reverse=True)
        return merged

    def merge_repl_tips(self, reports: list[AgentReport]) -> list[REPLTip]:
        """Merge REPL tips from all agent reports. Dedup by text."""
        seen_texts: set[str] = set()
        merged = []
        for report in reports:
            for tip in report.repl_tips:
                key = tip.text.strip().lower()
                if key not in seen_texts:
                    seen_texts.add(key)
                    merged.append(tip)
        return merged

    def merge_instructions(self, reports: list[AgentReport]) -> str:
        """Pick the longest (most detailed) instructions from all agents."""
        best = ""
        for report in reports:
            if len(report.instructions) > len(best):
                best = report.instructions
        return best


# ── Local Agent Runner ────────────────────────────────────────────

class LocalAgentRunner:
    """Runs agents as local processes with shared filesystem sync."""

    def __init__(self, sync_dir: str):
        self.sync_dir = sync_dir
        self.processes: dict[str, multiprocessing.Process] = {}
        os.makedirs(sync_dir, exist_ok=True)

    def spawn_agent(self, agent_id: str, game_id: str, max_steps: int,
                    batch_size: int = 5, reflect_every: int = 3) -> None:
        """Spawn a single agent as a local process."""
        def _run():
            from agent import run_agent
            run_agent(
                game_id=game_id,
                max_steps=max_steps,
                batch_size=batch_size,
                reflect_every=reflect_every,
                local_mode=True,
                agent_id=agent_id,
                sync_dir=self.sync_dir,
            )

        p = multiprocessing.Process(target=_run, name=f"agent-{agent_id}")
        p.start()
        self.processes[agent_id] = p
        print(f"[Orchestrator] Spawned local agent {agent_id} (PID {p.pid})")

    def spawn_many(self, agent_ids: list[str], game_id: str, max_steps: int,
                   batch_size: int = 5, reflect_every: int = 3) -> None:
        """Spawn multiple agents."""
        for agent_id in agent_ids:
            self.spawn_agent(agent_id, game_id, max_steps, batch_size, reflect_every)

    def collect_report(self, agent_id: str) -> AgentReport | None:
        """Read an agent's report from the sync directory."""
        report_path = os.path.join(self.sync_dir, f"{agent_id}_report.json")
        if not os.path.exists(report_path):
            return None
        try:
            with open(report_path) as f:
                data = json.load(f)
            return AgentReport.model_validate(data)
        except Exception as e:
            print(f"  [Orchestrator] Failed to read report for {agent_id}: {e}")
            return None

    def collect_all_reports(self, agent_ids: list[str]) -> list[AgentReport]:
        """Collect reports from all agents."""
        reports = []
        for agent_id in agent_ids:
            report = self.collect_report(agent_id)
            if report:
                reports.append(report)
        return reports

    def inject_knowledge(self, agent_id: str, knowledge: list[KnowledgeEntry],
                         repl_tips: list[REPLTip]) -> None:
        """Write merged knowledge for an agent to consume."""
        inject_path = os.path.join(self.sync_dir, f"{agent_id}_injected.json")
        data = {
            "knowledge": [e.model_dump() for e in knowledge],
            "repl_tips": [t.model_dump() for t in repl_tips],
        }
        with open(inject_path, "w") as f:
            json.dump(data, f, indent=2)

    def is_alive(self, agent_id: str) -> bool:
        """Check if an agent process is still running."""
        p = self.processes.get(agent_id)
        return p is not None and p.is_alive()

    def any_alive(self) -> bool:
        """Check if any agent is still running."""
        return any(self.is_alive(aid) for aid in self.processes)

    def join_all(self, timeout: float | None = None) -> None:
        """Wait for all agents to finish."""
        for agent_id, p in self.processes.items():
            p.join(timeout=timeout)


# ── Sandbox Agent Runner ──────────────────────────────────────────

class SandboxAgentRunner:
    """Runs agents in Daytona sandboxes with thread-per-sandbox management."""

    def __init__(self):
        from sandbox import SandboxManager
        self.manager = SandboxManager()
        self.threads: dict[str, threading.Thread] = {}
        self.results: dict[str, str] = {}  # agent_id -> stdout

    def spawn_agent(self, agent_id: str, game_id: str, max_steps: int) -> None:
        """Create a sandbox and run the agent in a background thread."""
        self.manager.create_sandbox(agent_id)

        def _run():
            try:
                output = self.manager.run_agent(agent_id, game_id, max_steps, local=False)
                self.results[agent_id] = output
            except Exception as e:
                self.results[agent_id] = f"ERROR: {e}"

        t = threading.Thread(target=_run, name=f"sandbox-{agent_id}")
        t.start()
        self.threads[agent_id] = t
        print(f"[Orchestrator] Spawned sandbox agent {agent_id}")

    def spawn_many(self, agent_ids: list[str], game_id: str, max_steps: int) -> None:
        """Create sandboxes and spawn agents."""
        for agent_id in agent_ids:
            self.spawn_agent(agent_id, game_id, max_steps)

    def collect_report(self, agent_id: str) -> AgentReport | None:
        """Download an agent's report from its sandbox."""
        data = self.manager.download_report(agent_id)
        if data:
            try:
                return AgentReport.model_validate(data)
            except Exception:
                pass
        return None

    def collect_all_reports(self, agent_ids: list[str]) -> list[AgentReport]:
        """Collect reports from all agents."""
        reports = []
        for agent_id in agent_ids:
            report = self.collect_report(agent_id)
            if report:
                reports.append(report)
        return reports

    def inject_knowledge(self, agent_id: str, knowledge: list[KnowledgeEntry],
                         repl_tips: list[REPLTip]) -> None:
        """Upload merged knowledge to a sandbox."""
        data = {
            "knowledge": [e.model_dump() for e in knowledge],
            "repl_tips": [t.model_dump() for t in repl_tips],
        }
        self.manager.upload_knowledge(agent_id, data)

    def upload_knowledge_data(self, agent_id: str, data: dict) -> None:
        """Upload pre-built knowledge data dict (with optional directive) to a sandbox."""
        self.manager.upload_knowledge(agent_id, data)

    def is_alive(self, agent_id: str) -> bool:
        """Check if an agent thread is still running."""
        t = self.threads.get(agent_id)
        return t is not None and t.is_alive()

    def any_alive(self) -> bool:
        """Check if any agent is still running."""
        return any(self.is_alive(aid) for aid in self.threads)

    def join_all(self, timeout: float | None = None) -> None:
        """Wait for all agents to finish."""
        for agent_id, t in self.threads.items():
            t.join(timeout=timeout)

    def cleanup(self) -> None:
        """Delete all sandboxes."""
        self.manager.cleanup_all()


# ── Multi-Agent Orchestrator ──────────────────────────────────────

class MultiAgentOrchestrator:
    """Orchestrates N agents with periodic knowledge sync.

    Sync cycle:
    1. Collect reports from all agents
    2. Merge knowledge (dedup by fingerprint, keep highest confidence)
    3. Redistribute merged knowledge to all agents
    """

    def __init__(self, mode: str = "local", num_agents: int = 4,
                 sync_interval: int = 60, sync_dir: str | None = None):
        self.mode = mode
        self.num_agents = num_agents
        self.sync_interval = sync_interval
        self.sync_dir = sync_dir or tempfile.mkdtemp(prefix="arc_sync_")
        self.merger = KnowledgeMerger()
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]

        # Initialize StrategicReasoner if enabled
        self.reasoner = None
        if STRATEGIC_REASONER_ENABLED:
            try:
                import dspy
                from strategic_reasoner import StrategicReasoner
                lm = dspy.LM(GEMINI_MODEL, max_tokens=4096)
                sub_lm = dspy.LM(GEMINI_MODEL_MINI, max_tokens=2048)
                self.reasoner = StrategicReasoner(
                    lm=lm, sub_lm=sub_lm,
                    max_iterations=STRATEGIC_REASONER_MAX_ITERATIONS,
                )
                print(f"[Orchestrator] StrategicReasoner enabled (max_iter={STRATEGIC_REASONER_MAX_ITERATIONS})")
            except Exception as e:
                print(f"[Orchestrator] Failed to initialize StrategicReasoner, using KnowledgeMerger: {e}")

        self.state = OrchestratorState(
            mode=mode,
            num_agents=num_agents,
            sync_interval=sync_interval,
        )

        # Initialize the appropriate runner
        if mode == "local":
            self.runner = LocalAgentRunner(self.sync_dir)
        elif mode == "sandbox":
            self.runner = SandboxAgentRunner()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def run(self, game_id: str, max_steps: int = 500,
            batch_size: int = 5, reflect_every: int = 3) -> OrchestratorState:
        """Run the full multi-agent orchestration loop."""
        self.state.games = [game_id]

        print(f"\n{'='*70}")
        print(f"MULTI-AGENT ORCHESTRATOR")
        print(f"  Mode: {self.mode} | Agents: {self.num_agents}")
        print(f"  Game: {game_id} | Steps: {max_steps}")
        print(f"  Sync interval: {self.sync_interval}s")
        print(f"  Sync dir: {self.sync_dir}")
        print(f"{'='*70}\n")

        # Spawn all agents
        if self.mode == "local":
            self.runner.spawn_many(
                self.agent_ids, game_id, max_steps,
                batch_size=batch_size, reflect_every=reflect_every,
            )
        else:
            self.runner.spawn_many(self.agent_ids, game_id, max_steps)

        # Mark all agents as running
        for aid in self.agent_ids:
            self.state.agents[aid] = AgentStatus.RUNNING

        # Sync loop
        sync_count = 0
        while self.runner.any_alive():
            time.sleep(self.sync_interval)
            sync_count += 1

            print(f"\n--- Sync Cycle {sync_count} ---")
            self._sync_cycle()
            print(f"--- End Sync Cycle {sync_count} ---")

        # Print sandbox agent output/errors for diagnosis
        if self.mode == "sandbox" and hasattr(self.runner, 'results'):
            for aid in self.agent_ids:
                output = self.runner.results.get(aid, "")
                if output:
                    is_error = output.startswith("ERROR:") or "Traceback" in output
                    label = "ERROR" if is_error else "OUTPUT"
                    # Show last 2000 chars (tail of output) for errors, 500 for normal
                    tail = 2000 if is_error else 500
                    print(f"\n  [{label}] {aid}:\n{output[-tail:]}")

        # Final collection
        print(f"\n--- Final Collection ---")
        self._sync_cycle()

        # Update agent statuses
        for aid in self.agent_ids:
            report = self._get_latest_report(aid)
            if report:
                self.state.agents[aid] = report.status
                self.state.total_levels = max(self.state.total_levels, report.levels_completed)
                self.state.total_steps += report.steps_taken

        # Print final summary
        self._print_summary()

        # Cleanup sandboxes if needed
        if self.mode == "sandbox":
            self.runner.cleanup()

        return self.state

    def _sync_cycle(self) -> None:
        """Run one sync cycle: collect → reason/merge → redistribute."""
        # 1. Collect reports
        reports = self.runner.collect_all_reports(self.agent_ids)
        if not reports:
            print("  No reports collected yet.")
            return

        print(f"  Collected {len(reports)} reports:")
        for r in reports:
            print(f"    {r.agent_id}: steps={r.steps_taken}, levels={r.levels_completed}/{r.total_levels}, "
                  f"kb={len(r.knowledge)}, status={r.status.value}")

        # 2. Reason (StrategicReasoner) or merge (KnowledgeMerger)
        if self.reasoner is not None:
            output = self.reasoner.reason(reports, self.agent_ids)
            merged_knowledge = output.merged_knowledge
            merged_tips = output.merged_repl_tips
            directives_by_agent = {d.agent_id: d for d in output.directives}
        else:
            merged_knowledge = self.merger.merge_knowledge(reports)
            merged_tips = self.merger.merge_repl_tips(reports)
            directives_by_agent = {}

        self.state.merged_knowledge = merged_knowledge
        self.state.merged_repl_tips = merged_tips

        print(f"  Merged: {len(merged_knowledge)} knowledge entries, {len(merged_tips)} REPL tips")

        # 3. Redistribute to all running agents (with per-agent directives when available)
        for agent_id in self.agent_ids:
            if self.runner.is_alive(agent_id):
                directive = directives_by_agent.get(agent_id)
                self._inject_with_directive(agent_id, merged_knowledge, merged_tips, directive)
                if directive:
                    print(f"  -> Injected knowledge + directive to {agent_id} (focus: {directive.focus_area})")
                else:
                    print(f"  -> Injected merged knowledge to {agent_id}")

    def _inject_with_directive(self, agent_id: str, knowledge: list[KnowledgeEntry],
                               repl_tips: list[REPLTip],
                               directive: AgentDirective | None = None) -> None:
        """Write merged knowledge + optional directive for an agent to consume."""
        data = {
            "knowledge": [e.model_dump() for e in knowledge],
            "repl_tips": [t.model_dump() for t in repl_tips],
        }
        if directive:
            # Exclude knowledge_to_inject from serialized directive (already in top-level knowledge)
            directive_data = directive.model_dump()
            directive_data.pop("knowledge_to_inject", None)
            data["directive"] = directive_data

        if self.mode == "sandbox":
            self.runner.upload_knowledge_data(agent_id, data)
        else:
            inject_path = os.path.join(self.sync_dir, f"{agent_id}_injected.json")
            with open(inject_path, "w") as f:
                json.dump(data, f, indent=2)

    def _get_latest_report(self, agent_id: str) -> AgentReport | None:
        """Get the latest report for an agent."""
        return self.runner.collect_report(agent_id)

    def _print_summary(self) -> None:
        """Print orchestration summary."""
        print(f"\n{'='*70}")
        print(f"ORCHESTRATION COMPLETE")
        print(f"  Mode: {self.mode} | Agents: {self.num_agents}")
        print(f"  Total merged knowledge: {len(self.state.merged_knowledge)} entries")
        print(f"  Total merged REPL tips: {len(self.state.merged_repl_tips)} tips")
        print(f"{'='*70}")

        # Print per-agent results
        for aid in self.agent_ids:
            report = self._get_latest_report(aid)
            if report:
                print(f"\n  {aid}: {report.status.value}")
                print(f"    Steps: {report.steps_taken} | Levels: {report.levels_completed}/{report.total_levels}")
                print(f"    Knowledge: {len(report.knowledge)} entries")
            else:
                print(f"\n  {aid}: no report available")

        # Print top knowledge entries
        if self.state.merged_knowledge:
            print(f"\n--- Top Merged Knowledge ---")
            for entry in self.state.merged_knowledge[:10]:
                print(f"  [{entry.confidence}%] [{entry.category.value}] {entry.text[:100]}")

        # Save merged knowledge to disk
        self._save_merged_knowledge()

    def _save_merged_knowledge(self) -> None:
        """Persist the merged knowledge to disk."""
        output_path = os.path.join(self.sync_dir, "merged_knowledge.json")
        data = {
            "knowledge": [e.model_dump() for e in self.state.merged_knowledge],
            "repl_tips": [t.model_dump() for t in self.state.merged_repl_tips],
            "agents": {aid: status.value for aid, status in self.state.agents.items()},
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Merged knowledge saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ARC AGI 3 Multi-Agent Orchestrator")
    parser.add_argument("--games", default="ls20", help="Game ID(s), comma-separated")
    parser.add_argument("--agents", type=int, default=NUM_AGENTS, help=f"Number of agents (default: {NUM_AGENTS})")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per agent (default: 500)")
    parser.add_argument("--batch", type=int, default=5, help="Batch size per agent (default: 5)")
    parser.add_argument("--reflect", type=int, default=3, help="Reflect every N batches (default: 3)")
    parser.add_argument("--mode", default=ORCHESTRATOR_MODE,
                        choices=["local", "sandbox"],
                        help=f"Runner mode (default: {ORCHESTRATOR_MODE})")
    parser.add_argument("--sync-interval", type=int, default=SYNC_INTERVAL_SECONDS,
                        help=f"Sync interval in seconds (default: {SYNC_INTERVAL_SECONDS})")
    parser.add_argument("--sync-dir", default=None, help="Sync directory (auto-created if not specified)")
    args = parser.parse_args()

    game_id = args.games.split(",")[0]  # For now, single game

    orchestrator = MultiAgentOrchestrator(
        mode=args.mode,
        num_agents=args.agents,
        sync_interval=args.sync_interval,
        sync_dir=args.sync_dir,
    )

    state = orchestrator.run(
        game_id=game_id,
        max_steps=args.steps,
        batch_size=args.batch,
        reflect_every=args.reflect,
    )

    # Exit code based on whether any agent completed
    completed = sum(1 for s in state.agents.values() if s == AgentStatus.COMPLETED)
    sys.exit(0 if completed > 0 else 1)


if __name__ == "__main__":
    main()
