#!/usr/bin/env python3
"""
Claude Code Bridge Tool

Dispatches coding tasks to `Claude Code` CLI (headless `claude -p`) inside a
mandatory git worktree, then returns a structured result.

Security constraints (non-negotiable):
- repo_dir validated via os.path.realpath under each allowed_repo_roots entry
  (blocks ../symlink/UNC traversal).
- ANTHROPIC_API_KEY and ANTHROPIC_TOKEN stripped from subprocess env so the
  worker authenticates via CLAUDE_CODE_OAUTH_TOKEN (Max subscription) and
  never charges per-token.
- Binary resolved via HERMES_CLAUDE_CODE_PATH or shutil.which — never a bare
  "claude" string in subprocess (Windows .cmd shim safety).
- Mandatory git worktree + feature branch per dispatch (prevents index
  corruption on parallel runs).
- Secret redaction applied to summary/stderr before returning and before
  writing to status_log.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading (mirrors delegate_tool.py:2436 _load_config pattern)
# ---------------------------------------------------------------------------

def _load_claude_code_config() -> dict:
    """Load claude_code section from CLI_CONFIG or persistent config.yaml."""
    try:
        from cli import CLI_CONFIG  # type: ignore
        cfg = CLI_CONFIG.get("claude_code") or {}
        if cfg:
            return cfg
    except Exception:
        pass
    try:
        from hermes_cli.config import load_config
        full = load_config()
        return (full or {}).get("claude_code") or {}
    except Exception:
        return {}


@dataclass
class _ClaudeCodeConfig:
    enabled: bool = True
    command: str = "claude"
    default_allowed_tools: List[str] = field(default_factory=lambda: [
        "Read", "Edit", "Bash(git:*)", "Bash(gh:*)",
    ])
    permission_mode: str = "acceptEdits"
    model: str = ""
    max_turns: int = 30
    timeout_seconds: int = 1200
    allowed_repo_roots: List[str] = field(default_factory=lambda: ["C:/Dev/Dev"])
    status_log: str = "~/.hermes/shiri/tasks.jsonl"

    @classmethod
    def load(cls) -> "_ClaudeCodeConfig":
        raw = _load_claude_code_config()
        obj = cls()
        if not isinstance(raw, dict):
            return obj
        if "enabled" in raw:
            obj.enabled = bool(raw["enabled"])
        if "command" in raw:
            obj.command = str(raw["command"] or "claude")
        if "default_allowed_tools" in raw and isinstance(raw["default_allowed_tools"], list):
            obj.default_allowed_tools = raw["default_allowed_tools"]
        if "permission_mode" in raw:
            obj.permission_mode = str(raw["permission_mode"] or "acceptEdits")
        if "model" in raw:
            obj.model = str(raw.get("model") or "")
        if "max_turns" in raw:
            try:
                obj.max_turns = int(raw["max_turns"])
            except (TypeError, ValueError):
                pass
        if "timeout_seconds" in raw:
            try:
                obj.timeout_seconds = int(raw["timeout_seconds"])
            except (TypeError, ValueError):
                pass
        if "allowed_repo_roots" in raw and isinstance(raw["allowed_repo_roots"], list):
            obj.allowed_repo_roots = [str(r) for r in raw["allowed_repo_roots"]]
        if "status_log" in raw:
            obj.status_log = str(raw["status_log"])
        return obj


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------

def _resolve_claude_binary(cfg: _ClaudeCodeConfig) -> Optional[str]:
    """Resolve the claude CLI binary path.

    Priority:
    1. HERMES_CLAUDE_CODE_PATH env var (explicit override for Windows service PATH)
    2. shutil.which(cfg.command or "claude") — returns full .cmd path on Windows
    """
    override = os.environ.get("HERMES_CLAUDE_CODE_PATH", "").strip()
    if override:
        if os.path.isfile(override):
            return override
        logger.warning("HERMES_CLAUDE_CODE_PATH=%r does not point to a file", override)

    cmd = (cfg.command or "claude").strip()
    found = shutil.which(cmd)
    if found:
        return found

    # On Windows, .cmd shims may not be found without the extension
    if os.name == "nt":
        for ext in (".cmd", ".bat", ".exe"):
            found = shutil.which(cmd + ext)
            if found:
                return found

    return None


def _check_claude_binary() -> bool:
    """check_fn: returns True if the claude binary can be resolved."""
    cfg = _ClaudeCodeConfig.load()
    return _resolve_claude_binary(cfg) is not None


# ---------------------------------------------------------------------------
# Repo dir validation
# ---------------------------------------------------------------------------

def _validate_repo_dir(repo_dir: str, allowed_repo_roots: List[str]) -> Optional[str]:
    """Validate repo_dir against allowed_repo_roots.

    Returns None if valid (and repo_dir has a .git entry), otherwise an error
    message string describing why it was rejected.

    Uses os.path.realpath for both sides to block:
    - ../traversal attacks
    - symlink escapes
    - UNC path shenanigans on Windows
    """
    if not repo_dir:
        return "repo_dir is required"

    try:
        real_repo = os.path.realpath(os.path.abspath(repo_dir))
    except Exception as exc:
        return f"Cannot resolve repo_dir path: {exc}"

    if not os.path.isdir(real_repo):
        return f"repo_dir does not exist or is not a directory: {repo_dir!r}"

    # Must contain .git (file or directory — for worktrees, .git is a file)
    git_marker = os.path.join(real_repo, ".git")
    if not os.path.exists(git_marker):
        return f"repo_dir does not appear to be a git repository (no .git): {repo_dir!r}"

    # Must be under at least one allowed root
    for root in allowed_repo_roots:
        try:
            real_root = os.path.realpath(os.path.abspath(root))
        except Exception:
            continue
        # os.path.commonpath requires both to share a prefix
        try:
            common = os.path.commonpath([real_repo, real_root])
            if common == real_root:
                return None  # valid
        except ValueError:
            # Different drives on Windows — commonpath raises ValueError
            continue

    return (
        f"repo_dir {repo_dir!r} is not under any allowed_repo_roots "
        f"({allowed_repo_roots!r}). Dispatch rejected for security."
    )


# ---------------------------------------------------------------------------
# Scoped env for subprocess
# ---------------------------------------------------------------------------

def _build_scoped_env() -> Dict[str, str]:
    """Build subprocess env: inherit os.environ MINUS API key vars, then
    supplement auth tokens from Hermes' .env reader.

    Per security requirements:
    - Strip ANTHROPIC_API_KEY and ANTHROPIC_TOKEN (would override OAuth, billing risk)
    - Pass through CLAUDE_CODE_OAUTH_TOKEN and GH_TOKEN/GITHUB_TOKEN

    Hermes does not propagate ~/.hermes/.env into os.environ, so worker auth
    tokens (CLAUDE_CODE_OAUTH_TOKEN, GH_TOKEN, GITHUB_TOKEN) would be absent
    from the subprocess environment if sourced only from os.environ.  We
    therefore resolve each missing token via get_env_value (hermes_cli/config.py)
    which reads ~/.hermes/.env directly.  ANTHROPIC keys are never injected.
    """
    blocked = {"ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN"}
    env = {k: v for k, v in os.environ.items() if k not in blocked}

    # Inject auth tokens that may live only in ~/.hermes/.env
    _AUTH_TOKENS = ("CLAUDE_CODE_OAUTH_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
    try:
        from hermes_cli.config import get_env_value  # type: ignore
        for key in _AUTH_TOKENS:
            if not env.get(key):
                value = get_env_value(key)
                if value:
                    env[key] = value
    except Exception:
        # Import may fail in isolated test contexts — proceed without injection
        pass

    return env


# ---------------------------------------------------------------------------
# Git worktree management
# ---------------------------------------------------------------------------

def _create_worktree(repo_dir: str, branch: Optional[str]) -> tuple:
    """Create a git worktree + feature branch for this dispatch.

    Returns (worktree_path: str, branch_name: str) or raises RuntimeError.
    """
    short_id = uuid.uuid4().hex[:8]
    branch_name = branch or f"shiri/task-{short_id}"

    # Place worktrees under the repo's parent directory to avoid nesting
    # worktrees inside the main repo (which confuses some tools).
    parent = os.path.dirname(repo_dir)
    repo_name = os.path.basename(repo_dir)
    worktree_dir = os.path.join(parent, f".worktree-{repo_name}-{short_id}")

    # Create the worktree + new branch in one command
    result = subprocess.run(
        ["git", "-C", repo_dir, "worktree", "add", "-b", branch_name, worktree_dir],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git worktree add failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    logger.info("Created worktree at %s on branch %s", worktree_dir, branch_name)
    return worktree_dir, branch_name


def _remove_worktree(repo_dir: str, worktree_dir: str) -> None:
    """Remove the worktree when done (best-effort)."""
    try:
        subprocess.run(
            ["git", "-C", repo_dir, "worktree", "remove", "--force", worktree_dir],
            capture_output=True,
            text=True,
            timeout=30,
        )
        logger.debug("Removed worktree %s", worktree_dir)
    except Exception as exc:
        logger.warning("Failed to remove worktree %s: %s", worktree_dir, exc)


# ---------------------------------------------------------------------------
# Changed-files detection
# ---------------------------------------------------------------------------

def _get_changed_files(worktree_dir: str) -> List[str]:
    """Return files changed in the worktree (porcelain + diff HEAD)."""
    files: set = set()
    try:
        r = subprocess.run(
            ["git", "-C", worktree_dir, "status", "--porcelain"],
            capture_output=True, text=True, timeout=15,
        )
        for line in r.stdout.splitlines():
            if line.strip():
                # porcelain format: XY filename (or XY old -> new for renames)
                parts = line[3:].split(" -> ")
                files.add(parts[-1].strip())
    except Exception as exc:
        logger.debug("git status failed: %s", exc)

    try:
        r = subprocess.run(
            ["git", "-C", worktree_dir, "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        for line in r.stdout.splitlines():
            line = line.strip()
            if line:
                files.add(line)
    except Exception as exc:
        logger.debug("git diff failed: %s", exc)

    return sorted(files)


# ---------------------------------------------------------------------------
# PR URL detection
# ---------------------------------------------------------------------------

def _get_pr_url(worktree_dir: str, branch: str) -> Optional[str]:
    """Try to get the PR URL for this branch using gh cli (best-effort)."""
    try:
        gh = shutil.which("gh")
        if not gh:
            return None
        r = subprocess.run(
            [gh, "pr", "view", branch, "--json", "url", "-q", ".url"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=worktree_dir,
        )
        url = r.stdout.strip()
        if url and url.startswith("http"):
            return url
    except Exception as exc:
        logger.debug("gh pr view failed (non-fatal): %s", exc)
    return None


# ---------------------------------------------------------------------------
# Status log
# ---------------------------------------------------------------------------

_LOG_ROTATE_BYTES = 5 * 1024 * 1024  # 5 MB


def _append_status_log(log_path_raw: str, entry: dict) -> None:
    """Append a JSON line to the status log; rotate at 5 MB."""
    try:
        log_path = Path(os.path.expanduser(log_path_raw))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotate if over size limit
        if log_path.exists() and log_path.stat().st_size > _LOG_ROTATE_BYTES:
            rotated = log_path.with_suffix(f".{int(time.time())}.jsonl")
            log_path.rename(rotated)
            logger.info("Rotated status_log: %s -> %s", log_path, rotated)

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to write status_log %s: %s", log_path_raw, exc)


# ---------------------------------------------------------------------------
# Secret redaction helper
# ---------------------------------------------------------------------------

def _redact(text: Optional[str]) -> Optional[str]:
    """Apply secret redaction. Uses agent.redact module if available."""
    if not text:
        return text
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True)
    except Exception:
        pass
    # Minimal fallback: scrub obvious token patterns in case import fails
    _FALLBACK_RE = re.compile(
        r"(?<![A-Za-z0-9_-])(sk-[A-Za-z0-9_-]{10,}|ghp_[A-Za-z0-9]{10,}"
        r"|github_pat_[A-Za-z0-9_]{10,}|AKIA[A-Z0-9]{16})(?![A-Za-z0-9_-])"
    )
    return _FALLBACK_RE.sub("[REDACTED]", text)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def claude_code(
    prompt: str,
    repo_dir: str,
    resume_session_id: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
    model: Optional[str] = None,
    max_turns: Optional[int] = None,
    branch: Optional[str] = None,
) -> str:
    """Dispatch a coding task to Claude Code CLI in an isolated git worktree.

    Args:
        prompt:             The coding task to perform.
        repo_dir:           Absolute path to the git repository root.
        resume_session_id:  Resume a previous claude session (--resume).
        allowed_tools:      Override the default allowed tool list.
        model:              Override the claude model (optional).
        max_turns:          Override max-turns limit (optional).
        branch:             Desired feature branch name (auto-generated if omitted).

    Returns:
        JSON string with: success, session_id, summary, changed_files, branch,
        pr_url, num_turns, duration_seconds, cost_usd, repo_dir, is_error.
    """
    from tools.registry import tool_error, tool_result  # noqa: F401 (used below)

    t_start = time.monotonic()
    cfg = _ClaudeCodeConfig.load()

    # ---- 1. Validate repo_dir ----
    err = _validate_repo_dir(repo_dir, cfg.allowed_repo_roots)
    if err:
        return tool_error(err)

    # ---- 2. Resolve binary ----
    claude_bin = _resolve_claude_binary(cfg)
    if not claude_bin:
        return tool_error(
            "Claude Code CLI not found. Set HERMES_CLAUDE_CODE_PATH or install "
            "claude via `npm install -g @anthropic-ai/claude-code` and ensure "
            "it is on PATH (or set HERMES_CLAUDE_CODE_PATH to the full path)."
        )

    # ---- 3. Create worktree ----
    worktree_dir: Optional[str] = None
    branch_name: Optional[str] = None
    try:
        worktree_dir, branch_name = _create_worktree(repo_dir, branch)
    except RuntimeError as exc:
        return tool_error(f"Failed to create git worktree: {exc}")

    # ---- 4. Build argv ----
    effective_max_turns = max_turns or cfg.max_turns
    effective_tools = allowed_tools or cfg.default_allowed_tools
    effective_model = model or cfg.model

    argv = [
        claude_bin,
        "-p", prompt,
        "--output-format", "json",
        "--max-turns", str(effective_max_turns),
        "--permission-mode", cfg.permission_mode,
    ]

    if effective_tools:
        # Claude Code expects --allowedTools as a comma-separated string or repeated flags
        argv += ["--allowedTools", ",".join(effective_tools)]

    if effective_model:
        argv += ["--model", effective_model]

    if resume_session_id:
        argv += ["--resume", resume_session_id]

    # Sanity check: never add bypassPermissions
    if "--permission-mode" in argv:
        idx = argv.index("--permission-mode")
        if idx + 1 < len(argv) and argv[idx + 1] == "bypassPermissions":
            return tool_error(
                "bypassPermissions is not allowed. Use acceptEdits or default."
            )

    # ---- 5. Scoped env ----
    scoped_env = _build_scoped_env()

    # ---- 6. Run subprocess ----
    raw_stdout = ""
    raw_stderr = ""
    exit_code: Optional[int] = None
    timed_out = False

    try:
        logger.info(
            "Dispatching claude_code: branch=%s worktree=%s",
            branch_name,
            worktree_dir,
        )
        proc = subprocess.run(
            argv,
            cwd=worktree_dir,
            env=scoped_env,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_seconds,
        )
        raw_stdout = proc.stdout or ""
        raw_stderr = proc.stderr or ""
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        logger.warning("claude_code subprocess timed out after %ss", cfg.timeout_seconds)
    except Exception as exc:
        logger.error("claude_code subprocess error: %s", exc)
        _remove_worktree(repo_dir, worktree_dir)
        return tool_error(f"Subprocess error: {exc}")

    # ---- 7. Parse stdout JSON ----
    session_id: Optional[str] = None
    result_text: Optional[str] = None
    total_cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    duration_ms: Optional[int] = None
    is_error: bool = timed_out or (exit_code is not None and exit_code != 0)

    if raw_stdout:
        try:
            envelope = json.loads(raw_stdout)
            session_id = envelope.get("session_id")
            result_text = envelope.get("result") or envelope.get("content")
            total_cost_usd = envelope.get("total_cost_usd")
            num_turns = envelope.get("num_turns")
            duration_ms = envelope.get("duration_ms")
            if envelope.get("is_error") is not None:
                is_error = bool(envelope["is_error"])
        except json.JSONDecodeError:
            # stdout is not JSON — use as plain summary
            result_text = raw_stdout[:4000]

    if timed_out:
        result_text = f"[TIMEOUT after {cfg.timeout_seconds}s] " + (result_text or "")

    # ---- 8. Collect changed files + PR URL ----
    changed_files = _get_changed_files(worktree_dir)
    pr_url = _get_pr_url(worktree_dir, branch_name) if branch_name else None

    # ---- 9. Redact before returning / logging ----
    stderr_tail = (raw_stderr or "")[-2000:]  # last 2KB of stderr
    summary = _redact(result_text or "")
    stderr_tail = _redact(stderr_tail)

    # ---- 10. Compute elapsed ----
    elapsed_seconds = round(time.monotonic() - t_start, 2)
    cost_usd: Optional[float] = total_cost_usd

    # ---- 11. Append to status_log (after redaction) ----
    log_entry = {
        "ts": int(time.time()),
        "repo_dir": repo_dir,
        "branch": branch_name,
        "session_id": session_id,
        "success": not is_error,
        "num_turns": num_turns,
        "duration_seconds": elapsed_seconds,
        "cost_usd": cost_usd,
        "changed_files": changed_files,
        "pr_url": pr_url,
        "summary_snippet": (summary or "")[:500],
        "stderr_snippet": (stderr_tail or "")[:500],
        "exit_code": exit_code,
        "timed_out": timed_out,
    }
    _append_status_log(cfg.status_log, log_entry)

    # ---- 12. Remove worktree (best-effort) ----
    _remove_worktree(repo_dir, worktree_dir)

    # ---- 13. Return structured result ----
    payload = {
        "success": not is_error,
        "session_id": session_id,
        "summary": summary,
        "changed_files": changed_files,
        "branch": branch_name,
        "pr_url": pr_url,
        "num_turns": num_turns,
        "duration_seconds": elapsed_seconds,
        "cost_usd": cost_usd,
        "repo_dir": repo_dir,
        "is_error": is_error,
    }
    if is_error and not summary:
        payload["stderr_tail"] = stderr_tail

    if is_error:
        return tool_error(
            summary or stderr_tail or "Claude Code returned an error.",
            **{k: v for k, v in payload.items() if k not in ("success",)},
        )

    return tool_result(payload)


# ---------------------------------------------------------------------------
# OpenAI function-calling schema
# ---------------------------------------------------------------------------

CLAUDE_CODE_SCHEMA = {
    "name": "claude_code",
    "description": (
        "Dispatch a coding task to Claude Code CLI (headless `claude -p`) inside "
        "an isolated git worktree on an approved repository. Returns a structured "
        "result: success, summary, changed_files, branch, pr_url, session_id.\n\n"
        "IMPORTANT: Only repositories under allowed_repo_roots (default C:/Dev/Dev) "
        "are accepted. The worker runs in acceptEdits permission mode — it may edit "
        "files but never bypasses permissions. Use `resume_session_id` to continue "
        "a multi-turn coding session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "The coding task to perform. Be specific: include filenames, "
                    "desired behaviour, acceptance criteria, and any constraints."
                ),
            },
            "repo_dir": {
                "type": "string",
                "description": (
                    "Absolute path to the git repository root where the work "
                    "should happen. Must be under allowed_repo_roots."
                ),
            },
            "resume_session_id": {
                "type": "string",
                "description": (
                    "Session ID returned by a previous claude_code call. "
                    "Pass to continue a multi-turn coding session (--resume)."
                ),
            },
            "allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Claude Code tool list to allow (e.g. [\"Read\",\"Edit\",\"Bash(git:*)\"]). "
                    "Defaults to config default_allowed_tools."
                ),
            },
            "model": {
                "type": "string",
                "description": "Model override for this dispatch (e.g. 'claude-opus-4-5'). Defaults to Claude Code's default.",
            },
            "max_turns": {
                "type": "integer",
                "description": "Max agentic turns for this dispatch. Defaults to config max_turns (30).",
                "minimum": 1,
                "maximum": 100,
            },
            "branch": {
                "type": "string",
                "description": (
                    "Desired feature branch name. Auto-generated as 'shiri/task-<uuid>' "
                    "if omitted. Must not be 'main' or 'master'."
                ),
            },
        },
        "required": ["prompt", "repo_dir"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

from tools.registry import registry, tool_error, tool_result  # noqa: E402


def _claude_code_handler(args: dict, **kwargs) -> str:
    return claude_code(
        prompt=args.get("prompt", ""),
        repo_dir=args.get("repo_dir", ""),
        resume_session_id=args.get("resume_session_id"),
        allowed_tools=args.get("allowed_tools"),
        model=args.get("model"),
        max_turns=args.get("max_turns"),
        branch=args.get("branch"),
    )


registry.register(
    name="claude_code",
    toolset="coding-worker",
    schema=CLAUDE_CODE_SCHEMA,
    handler=_claude_code_handler,
    check_fn=_check_claude_binary,
    description="Dispatch coding tasks to Claude Code CLI in an isolated git worktree",
    emoji="🔨",
)
