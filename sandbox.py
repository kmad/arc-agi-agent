"""Daytona sandbox runner for ARC AGI 3 multi-agent system.

Creates a Daytona sandbox, sets up the environment, runs the agent,
collects results, and cleans up.

Usage:
    python sandbox.py --game ls20 --steps 500 [--gepa] [--keep]
"""

import os
import sys
import argparse
import json
from pathlib import Path

from daytona import Daytona, DaytonaConfig, CreateSandboxParams


AGENT_REPO = "https://github.com/kmad/arc.git"  # Adjust to actual repo
ARCENGINE_PACKAGE = "arc-agi"


def create_sandbox(daytona: Daytona) -> object:
    """Create a new Daytona sandbox."""
    print("Creating Daytona sandbox...")
    sandbox = daytona.create(CreateSandboxParams(language="python"))
    print(f"  Sandbox created: {sandbox.id}")
    return sandbox


def setup_environment(daytona: Daytona, sandbox) -> None:
    """Set up the sandbox environment: clone repo, install deps."""
    print("Setting up sandbox environment...")

    # Install system deps and clone
    commands = [
        "pip install --upgrade pip",
        "pip install arc-agi dspy numpy pandas pillow python-dotenv",
    ]

    # If gepa is needed
    commands.append("pip install gepa || true")
    commands.append("pip install daytona || true")

    for cmd in commands:
        print(f"  Running: {cmd}")
        response = sandbox.process.start(cmd)
        print(f"    -> exit code: {response.exit_code}")

    # Upload local source files
    source_files = [
        "agent.py", "solver.py", "game_state.py", "config.py",
        "observers.py", "optimizer.py", "gepa_optimizer.py",
        "actions.py", "sandbox.py",
    ]

    for filename in source_files:
        local_path = Path(__file__).parent / filename
        if local_path.exists():
            content = local_path.read_text()
            sandbox.fs.upload_file(f"/home/daytona/{filename}", content)
            print(f"  Uploaded: {filename}")

    # Upload .env if it exists
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        content = env_path.read_text()
        sandbox.fs.upload_file("/home/daytona/.env", content)
        print("  Uploaded: .env")

    print("  Environment setup complete.")


def run_agent(daytona: Daytona, sandbox, game_id: str, steps: int,
              use_gepa: bool = False) -> str:
    """Run the agent inside the sandbox.

    Returns:
        stdout from the agent run.
    """
    print(f"Running agent: game={game_id}, steps={steps}, gepa={use_gepa}")

    # Build env vars
    env_vars = {}
    for key in ("ARC_API_KEY", "GEMINI_API_KEY"):
        val = os.environ.get(key)
        if val:
            env_vars[key] = val

    env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())

    if use_gepa:
        cmd = f"{env_str} python /home/daytona/agent.py --game {game_id} --steps {steps} --gepa"
    else:
        cmd = f"{env_str} python /home/daytona/agent.py --game {game_id} --steps {steps}"

    print(f"  Command: {cmd}")
    response = sandbox.process.start(cmd)
    print(f"  Exit code: {response.exit_code}")

    output = response.result if hasattr(response, "result") else str(response)
    return output


def collect_results(daytona: Daytona, sandbox, game_id: str) -> dict | None:
    """Download knowledge_base from the sandbox."""
    print("Collecting results...")

    kb_path = f"/home/daytona/knowledge_base/{game_id}.json"
    try:
        content = sandbox.fs.download_file(kb_path)
        if content:
            results = json.loads(content)
            # Save locally
            local_kb_dir = Path(__file__).parent / "knowledge_base"
            local_kb_dir.mkdir(exist_ok=True)
            local_path = local_kb_dir / f"{game_id}_sandbox.json"
            with open(local_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved results to {local_path}")
            return results
    except Exception as e:
        print(f"  Could not collect results: {e}")

    return None


def cleanup(daytona: Daytona, sandbox) -> None:
    """Delete the sandbox."""
    print(f"Cleaning up sandbox {sandbox.id}...")
    daytona.delete(sandbox)
    print("  Sandbox deleted.")


def main():
    parser = argparse.ArgumentParser(description="ARC AGI 3 Daytona Sandbox Runner")
    parser.add_argument("--game", default="ls20", help="Game ID (default: ls20)")
    parser.add_argument("--steps", type=int, default=500, help="Max steps (default: 500)")
    parser.add_argument("--gepa", action="store_true", help="Run GEPA pre-optimization")
    parser.add_argument("--keep", action="store_true", help="Keep sandbox after run (don't cleanup)")
    args = parser.parse_args()

    # Initialize Daytona client
    config = DaytonaConfig()
    daytona = Daytona(config)

    sandbox = None
    try:
        sandbox = create_sandbox(daytona)
        setup_environment(daytona, sandbox)
        output = run_agent(daytona, sandbox, args.game, args.steps, args.gepa)
        print(f"\n--- Agent Output ---")
        print(output[:5000] if len(output) > 5000 else output)

        results = collect_results(daytona, sandbox, args.game)
        if results:
            print(f"\n--- Results Summary ---")
            print(f"  Levels completed: {results.get('levels_completed', '?')}")
            print(f"  Total steps: {results.get('total_steps', '?')}")
            print(f"  Game KB entries: {len(results.get('game_knowledge_base', []))}")
    except Exception as e:
        print(f"ERROR: {e}")
        raise
    finally:
        if sandbox and not args.keep:
            cleanup(daytona, sandbox)
        elif sandbox:
            print(f"Sandbox kept alive: {sandbox.id}")


if __name__ == "__main__":
    main()
