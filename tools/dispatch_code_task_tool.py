#!/usr/bin/env python3
"""
Dispatch Code Task Tool

Enqueues a coding task to the GitHub-backed shiri-queue repo so that the
PC-side poller (shiri-deliver) can pick it up, run it via Claude Code, and
write the result back to outbox/.

Queue repo layout:
    inbox/<id>.json   — task written here (this tool)
    outbox/<id>.json  — result written by the PC poller
    delivered/<id>.marker — delivery receipt written by shiri-deliver

Task JSON schema:
    {
      "id": "<UTC-timestamp>-<short-random>",
      "repo": "<logical repo name>",
      "prompt": "<task description>",
      "branch": "<optional branch name>",
      "origin": "<optional opaque routing string>",
      "created_at": "<ISO8601 UTC>"
    }
"""

import json
import logging
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading (mirrors claude_code_tool._load_claude_code_config pattern)
# ---------------------------------------------------------------------------

def _load_dispatch_config() -> dict:
    """Load dispatch_code_task section from CLI_CONFIG or persistent config.yaml."""
    try:
        from cli import CLI_CONFIG  # type: ignore
        cfg = CLI_CONFIG.get("dispatch_code_task") or {}
        if cfg:
            return cfg
    except Exception:
        pass
    try:
        from hermes_cli.config import load_config
        full = load_config()
        return (full or {}).get("dispatch_code_task") or {}
    except Exception:
        return {}


class _DispatchConfig:
    """Parsed configuration for the dispatch_code_task tool."""

    DEFAULT_QUEUE_DIR = "~/.hermes/shiri-queue"

    def __init__(self):
        self.queue_dir: str = self.DEFAULT_QUEUE_DIR
        self.enabled: bool = True

    @classmethod
    def load(cls) -> "_DispatchConfig":
        raw = _load_dispatch_config()
        obj = cls()
        if not isinstance(raw, dict):
            return obj
        if "queue_dir" in raw and raw["queue_dir"]:
            obj.queue_dir = str(raw["queue_dir"])
        if "enabled" in raw:
            obj.enabled = bool(raw["enabled"])
        return obj

    @property
    def resolved_queue_dir(self) -> str:
        """Return queue_dir with ~ expanded."""
        return os.path.expanduser(self.queue_dir)


# ---------------------------------------------------------------------------
# check_fn
# ---------------------------------------------------------------------------

def _check_queue_available() -> bool:
    """Return True only if queue_dir exists and contains a .git directory.

    The tool hides itself when the queue clone isn't set up locally — this
    is intentional: on the PC the clone exists; on other hosts it does not.
    """
    cfg = _DispatchConfig.load()
    if not cfg.enabled:
        return False
    q = cfg.resolved_queue_dir
    return os.path.isdir(q) and os.path.isdir(os.path.join(q, ".git"))


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _run_git(args: list, cwd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a git command and return the CompletedProcess."""
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _git_commit_and_push(queue_dir: str, inbox_rel: str, task_id: str) -> Optional[str]:
    """Pull, stage, commit, and push the new task file.

    Returns None on success, or an error message string on failure.
    Retries the push once after a rebase if the remote has moved.
    """
    # 1. Pull --rebase --autostash so we're up-to-date before adding
    r = _run_git(["pull", "--rebase", "--autostash"], cwd=queue_dir, timeout=60)
    if r.returncode != 0:
        logger.warning("git pull --rebase returned %d: %s", r.returncode, r.stderr.strip())
        # Non-fatal: the repo may have no upstream yet or be offline; proceed.

    # 2. Stage the inbox file
    r = _run_git(["add", inbox_rel], cwd=queue_dir)
    if r.returncode != 0:
        return f"git add failed (exit {r.returncode}): {r.stderr.strip()}"

    # 3. Commit with explicit author identity (required in CI / headless envs)
    r = _run_git(
        [
            "-c", "user.email=shiri@cloud",
            "-c", "user.name=Shiri",
            "commit",
            "-m", f"task: {task_id}",
        ],
        cwd=queue_dir,
    )
    if r.returncode != 0:
        return f"git commit failed (exit {r.returncode}): {r.stderr.strip()}"

    # 4. Push — retry once after rebase on rejection (exit 1 + "rejected")
    r = _run_git(["push"], cwd=queue_dir, timeout=60)
    if r.returncode != 0:
        stderr_lower = (r.stderr or "").lower()
        if "rejected" in stderr_lower or "non-fast-forward" in stderr_lower:
            logger.info("Push rejected; rebasing and retrying once")
            _run_git(["pull", "--rebase", "--autostash"], cwd=queue_dir, timeout=60)
            r = _run_git(["push"], cwd=queue_dir, timeout=60)

    if r.returncode != 0:
        return f"git push failed (exit {r.returncode}): {r.stderr.strip()}"

    return None  # success


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def dispatch_code_task(
    repo: str,
    prompt: str,
    branch: Optional[str] = None,
) -> str:
    """Enqueue a coding task to the shiri-queue GitHub repo for PC-side execution.

    Args:
        repo:    Logical name of the repository where the task will run
                 (e.g. "hermes-agent"). The PC poller maps this to a real path.
        prompt:  Full description of the coding task to perform.
        branch:  Optional desired branch name for the resulting change.

    Returns:
        JSON string with dispatched, id, repo, message  — or an error.
    """
    from tools.registry import tool_error, tool_result  # noqa: F401

    cfg = _DispatchConfig.load()

    if not cfg.enabled:
        return tool_error("dispatch_code_task is disabled in config.")

    queue_dir = cfg.resolved_queue_dir

    # Validate queue directory
    if not os.path.isdir(queue_dir):
        return tool_error(
            f"Queue directory does not exist: {queue_dir!r}. "
            "Clone gonenha/shiri-queue to that path and try again."
        )
    if not os.path.isdir(os.path.join(queue_dir, ".git")):
        return tool_error(
            f"Queue directory {queue_dir!r} is not a git repository (no .git dir)."
        )

    # Validate required args
    repo = (repo or "").strip()
    prompt = (prompt or "").strip()
    if not repo:
        return tool_error("'repo' is required.")
    if not prompt:
        return tool_error("'prompt' is required.")

    # Generate task ID: UTC-timestamp + 8-char hex random suffix
    now_utc = datetime.now(timezone.utc)
    ts_str = now_utc.strftime("%Y%m%dT%H%M%S")
    rand_suffix = secrets.token_hex(4)  # 8 hex chars
    task_id = f"{ts_str}-{rand_suffix}"
    created_at = now_utc.isoformat()

    # Build task dict
    task: dict = {
        "id": task_id,
        "repo": repo,
        "prompt": prompt,
        "created_at": created_at,
    }
    if branch:
        task["branch"] = branch
    # origin intentionally left out (None) unless trivially available
    task["origin"] = None

    # Write inbox/<id>.json
    inbox_dir = Path(queue_dir) / "inbox"
    try:
        inbox_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return tool_error(f"Failed to create inbox directory: {exc}")

    inbox_file = inbox_dir / f"{task_id}.json"
    inbox_rel = f"inbox/{task_id}.json"
    try:
        inbox_file.write_text(
            json.dumps(task, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        return tool_error(f"Failed to write task file: {exc}")

    logger.info("Wrote task %s to %s", task_id, inbox_file)

    # Git commit + push
    git_error = _git_commit_and_push(queue_dir, inbox_rel, task_id)
    if git_error:
        # Clean up the file we wrote so we don't leave orphaned files on retry
        try:
            inbox_file.unlink(missing_ok=True)
        except Exception:
            pass
        return tool_error(f"Failed to push task to queue: {git_error}")

    return tool_result({
        "dispatched": True,
        "id": task_id,
        "repo": repo,
        "message": (
            "Task queued; it will run on your PC and the result will be "
            "delivered when ready."
        ),
    })


# ---------------------------------------------------------------------------
# OpenAI function-calling schema
# ---------------------------------------------------------------------------

DISPATCH_CODE_TASK_SCHEMA = {
    "name": "dispatch_code_task",
    "description": (
        "Enqueue a coding task to the shiri-queue GitHub repo so the PC-side "
        "worker can execute it via Claude Code CLI and deliver the result back.\n\n"
        "Use this when a coding task should run on the user's PC (where repos "
        "live and Claude Code is authenticated) rather than being executed "
        "directly. The task is written to inbox/<id>.json, committed, and pushed "
        "to gonenha/shiri-queue. The PC poller picks it up, runs it, and writes "
        "the result to outbox/. You will be notified when it completes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": (
                    "Logical name of the repository where the coding task "
                    "should run (e.g. 'hermes-agent', 'my-project'). The PC "
                    "poller maps this to the real local path."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Full description of the coding task to perform. Be specific: "
                    "include filenames, desired behaviour, acceptance criteria, "
                    "and any constraints."
                ),
            },
            "branch": {
                "type": "string",
                "description": (
                    "Optional desired feature branch name for the resulting change. "
                    "Auto-generated by the PC worker if omitted."
                ),
            },
        },
        "required": ["repo", "prompt"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

from tools.registry import registry, tool_error, tool_result  # noqa: E402


def _dispatch_code_task_handler(args: dict, **kwargs) -> str:
    return dispatch_code_task(
        repo=args.get("repo", ""),
        prompt=args.get("prompt", ""),
        branch=args.get("branch"),
    )


registry.register(
    name="dispatch_code_task",
    toolset="code-dispatch",
    schema=DISPATCH_CODE_TASK_SCHEMA,
    handler=_dispatch_code_task_handler,
    check_fn=_check_queue_available,
    description="Enqueue a coding task to the shiri-queue GitHub repo for PC-side execution",
    emoji="📬",
)
