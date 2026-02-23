"""Daytona sandbox manager for ARC AGI 3 multi-agent system.

Manages multiple Daytona sandboxes for parallel agent execution.
Uses git clone + uv sync for reproducible environment setup.

Usage (single sandbox):
    python sandbox.py --game ls20 --steps 500 [--gepa] [--keep]

Usage (via orchestrator):
    Used programmatically by orchestrator.py's SandboxAgentRunner.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

from daytona import Daytona, DaytonaConfig, CreateSandboxBaseParams


REPO_URL = "https://github.com/kmad/arc-agi-agent.git"
REPO_BRANCH = "main"
REPO_DIR = "/home/daytona/arc"

# Files to upload to each sandbox (only used for local overrides)
SOURCE_FILES = [
    "models.py", "agent.py", "solver.py", "game_state.py", "config.py",
    "observers.py", "optimizer.py", "gepa_optimizer.py",
    "actions.py", "orchestrator.py", "strategic_reasoner.py",
]


class SandboxManager:
    """Manages multiple Daytona sandboxes for parallel agent execution."""

    def __init__(self, api_key: str | None = None):
        config = DaytonaConfig()
        if api_key:
            config.api_key = api_key
        self.daytona = Daytona(config)
        self.sandboxes: dict[str, object] = {}  # agent_id -> sandbox
        self._uv_path = "uv"  # updated by _setup_environment

    def create_sandbox(self, agent_id: str = "agent_0") -> object:
        """Create a single sandbox and set up the environment."""
        print(f"[Sandbox] Creating sandbox for {agent_id}...")
        sandbox = self.daytona.create(CreateSandboxBaseParams(language="python"))
        self.sandboxes[agent_id] = sandbox
        print(f"  Sandbox created: {sandbox.id}")

        self._setup_environment(sandbox)
        return sandbox

    def create_many(self, agent_ids: list[str]) -> dict[str, object]:
        """Create multiple sandboxes in parallel."""
        for agent_id in agent_ids:
            self.create_sandbox(agent_id)
        return dict(self.sandboxes)

    def _setup_environment(self, sandbox) -> None:
        """Clone repo, install uv, sync dependencies, upload .env."""
        # 1. Clone the repo
        print(f"  Cloning {REPO_URL} ({REPO_BRANCH})...")
        sandbox.git.clone(
            url=REPO_URL,
            path=REPO_DIR,
            branch=REPO_BRANCH,
        )
        print("  Repo cloned.")

        # 2. Find or install uv
        print("  Setting up uv...")
        result = sandbox.process.exec("which uv 2>/dev/null || echo NOT_FOUND")
        uv_location = getattr(result, "result", "").strip()

        if uv_location and uv_location != "NOT_FOUND":
            self._uv_path = uv_location
            print(f"    uv already available: {uv_location}")
        else:
            result = sandbox.process.exec(
                "curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh && sh /tmp/uv-install.sh 2>&1"
            )
            print(f"    uv install exit code: {getattr(result, 'exit_code', '?')}")
            # Find it
            result = sandbox.process.exec(
                "chmod +x /root/.local/bin/uv 2>/dev/null; "
                "chmod +x /home/daytona/.local/bin/uv 2>/dev/null; "
                "which uv || find / -name uv -type f 2>/dev/null | head -1"
            )
            uv_path = getattr(result, "result", "").strip().split("\n")[0]
            if uv_path:
                sandbox.process.exec(f"chmod +x {uv_path}")
                self._uv_path = uv_path
            else:
                sandbox.process.exec("pip install uv")
                self._uv_path = "uv"
            print(f"    uv path: {self._uv_path}")

        # 3. Upload files not in repo (.env, environment_files/)
        base_dir = Path(__file__).parent

        env_path = base_dir / ".env"
        if env_path.exists():
            content = env_path.read_bytes()
            sandbox.fs.upload_file(content, f"{REPO_DIR}/.env")
            print("  Uploaded: .env")

        # Upload game environment files (gitignored but needed for offline mode)
        env_files_dir = base_dir / "environment_files"
        if env_files_dir.exists():
            game_files = list(env_files_dir.rglob("*"))
            file_count = 0
            for file_path in game_files:
                if file_path.is_file():
                    rel_path = file_path.relative_to(base_dir)
                    content = file_path.read_bytes()
                    sandbox.fs.upload_file(content, f"{REPO_DIR}/{rel_path}")
                    file_count += 1
            print(f"  Uploaded: {file_count} environment files")

        # 4. Sync dependencies
        uv = self._uv_path
        print(f"  Running {uv} sync...")
        result = sandbox.process.exec(f"cd {REPO_DIR} && {uv} sync 2>&1")
        output = getattr(result, "result", str(result))
        exit_code = getattr(result, "exit_code", None)
        print(f"    -> exit code: {exit_code}")
        if exit_code != 0:
            print(f"    -> output: {output[-1000:]}")

        print("  Environment setup complete.")

    def run_agent(self, agent_id: str, game_id: str, steps: int,
                  local: bool = True, sync_dir: str | None = None,
                  use_gepa: bool = False) -> str:
        """Run the agent inside a sandbox.

        Returns stdout from the agent run.
        """
        sandbox = self.sandboxes.get(agent_id)
        if not sandbox:
            raise ValueError(f"No sandbox found for {agent_id}")

        if sync_dir is None:
            sync_dir = f"{REPO_DIR}/sync"

        # Build env vars
        env_vars = {}
        for key in ("ARC_API_KEY", "OPENROUTER_API_KEY"):
            val = os.environ.get(key)
            if val:
                env_vars[key] = val

        env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())

        cmd_parts = [
            f"cd {REPO_DIR} &&",
            env_str,
            f"{self._uv_path} run python agent.py",
            f"--game {game_id}",
            f"--steps {steps}",
            f"--agent-id {agent_id}",
            f"--sync-dir {sync_dir}",
        ]
        if local:
            cmd_parts.append("--local")
        if use_gepa:
            cmd_parts.append("--gepa")

        cmd = " ".join(cmd_parts)
        print(f"[Sandbox] Running {agent_id}: {cmd}")
        result = sandbox.process.exec(cmd)
        output = getattr(result, "result", str(result))
        print(f"[Sandbox] {agent_id} finished, exit code: {getattr(result, 'exit_code', '?')}")
        return output

    def upload_knowledge(self, agent_id: str, knowledge_data: dict,
                         sync_dir: str | None = None) -> None:
        """Upload merged knowledge to a sandbox for injection."""
        if sync_dir is None:
            sync_dir = f"{REPO_DIR}/sync"
        sandbox = self.sandboxes.get(agent_id)
        if not sandbox:
            raise ValueError(f"No sandbox found for {agent_id}")

        # Ensure sync dir exists
        sandbox.process.exec(f"mkdir -p {sync_dir}")

        inject_path = f"{sync_dir}/{agent_id}_injected.json"
        content = json.dumps(knowledge_data).encode("utf-8")
        sandbox.fs.upload_file(content, inject_path)

    def download_report(self, agent_id: str,
                        sync_dir: str | None = None) -> dict | None:
        """Download an agent's report from its sandbox."""
        if sync_dir is None:
            sync_dir = f"{REPO_DIR}/sync"
        sandbox = self.sandboxes.get(agent_id)
        if not sandbox:
            return None

        report_path = f"{sync_dir}/{agent_id}_report.json"
        try:
            content = sandbox.fs.download_file(report_path)
            if content:
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                return json.loads(content)
        except Exception as e:
            print(f"  [Sandbox] Could not download report for {agent_id}: {e}")

        return None

    def cleanup(self, agent_id: str) -> None:
        """Delete a single sandbox."""
        sandbox = self.sandboxes.pop(agent_id, None)
        if sandbox:
            print(f"[Sandbox] Cleaning up {agent_id} ({sandbox.id})...")
            self.daytona.delete(sandbox)
            print(f"  Sandbox deleted.")

    def cleanup_all(self) -> None:
        """Delete all managed sandboxes."""
        for agent_id in list(self.sandboxes.keys()):
            self.cleanup(agent_id)


def main():
    """CLI for running a single sandbox agent."""
    import argparse
    parser = argparse.ArgumentParser(description="ARC AGI 3 Daytona Sandbox Runner")
    parser.add_argument("--game", default="ls20", help="Game ID (default: ls20)")
    parser.add_argument("--steps", type=int, default=500, help="Max steps (default: 500)")
    parser.add_argument("--gepa", action="store_true", help="Run GEPA pre-optimization")
    parser.add_argument("--keep", action="store_true", help="Keep sandbox after run")
    parser.add_argument("--local", action="store_true", help="Use offline mode in sandbox")
    args = parser.parse_args()

    manager = SandboxManager()
    agent_id = "sandbox_agent_0"

    try:
        manager.create_sandbox(agent_id)
        output = manager.run_agent(agent_id, args.game, args.steps,
                                   local=args.local, use_gepa=args.gepa)
        print(f"\n--- Agent Output ---")
        print(output[:5000] if len(output) > 5000 else output)

        report = manager.download_report(agent_id)
        if report:
            print(f"\n--- Results Summary ---")
            print(f"  Levels completed: {report.get('levels_completed', '?')}")
            print(f"  Steps taken: {report.get('steps_taken', '?')}")
            print(f"  Knowledge entries: {len(report.get('knowledge', []))}")
    except Exception as e:
        print(f"ERROR: {e}")
        raise
    finally:
        if not args.keep:
            manager.cleanup_all()
        else:
            for aid, sb in manager.sandboxes.items():
                print(f"Sandbox kept alive: {aid} -> {sb.id}")


if __name__ == "__main__":
    main()
