#!/usr/bin/env python3
"""ComfyRanch worker bootstrap.

Runs inside the rented GPU container (Vast.ai `onstart`). It consumes a
ProvisioningManifest produced by the ComfyRanch app and makes the worker ready,
then launches ComfyUI.

Everything here is cloud -> cloud: the worker pulls custom-node git repos and
model weights straight from their upstream URLs. The operator machine never
uploads or downloads anything.

Manifest sources, in order of precedence:
  1. COMFRANCH_MANIFEST_URL   - HTTP(S) URL returning the manifest JSON
  2. COMFRANCH_MANIFEST_B64   - base64-encoded manifest JSON (env var)
  3. COMFRANCH_MANIFEST       - raw manifest JSON (env var)
  4. COMFRANCH_MANIFEST_PATH  - path on disk (default: /opt/comfyranch/manifest.json)

A missing manifest is not fatal: the worker still launches ComfyUI so it can be
wired up manually. Useful env knobs (all optional):

  COMFY_DIR                       ComfyUI checkout (default: /opt/ComfyUI)
  COMFRANCH_MODELS_DIR            models/ root (default: $COMFY_DIR/models)
  COMFRANCH_MODEL_SOURCE_DIR      copy weights from here if present (volume mount)
  COMFRANCH_INSTALL_CUSTOM_NODES  1/0 (default: 1)
  COMFRANCH_INSTALL_REQUIREMENTS  1/0 (default: 1)
  COMFRANCH_SYNC_COMFY            1/0 (default: 1) fast-forward ComfyUI to min ver
  COMFRANCH_MAX_CONCURRENT_DOWNLOADS   (default: 3)
  COMFRANCH_VERIFY_SHA256         1/0 (default: 1, only when manifest has sha256)
  HF_TOKEN                        Hugging Face token for gated repos
  COMFRANCH_PORT                  ComfyUI listen port (default: 8188)
  COMFRANCH_LAUNCH                1/0 (default: 1) exec ComfyUI at the end
  COMFRANCH_EXTRA_ARGS            extra args appended to ComfyUI
  COMFRANCH_USER_WORKFLOWS_DIR    preload dir for workflow JS (default:
                                  $COMFY_DIR/user/default/workflows)

Headless automation (defaults on). The worker runs ComfyUI in the background,
starts a read-only status endpoint, and starts a runner that executes the task
manifest **baked into the deploy env** the moment ComfyUI is up:
  COMFRANCH_AUTORUN               1/0 (default: 1) headless mode; 0 = foreground
                                  ComfyUI only, no status plane/runner
  COMFRANCH_RUNNER                1/0 (default: 1) start the task runner
  COMFRANCH_TASKS_B64             base64 task manifest baked in at deploy (self-start)
  COMFRANCH_TASKS                 raw task manifest JSON (alternative to _B64)
  COMFRANCH_TASKS_PATH            task manifest path (alternative)
  COMFRANCH_CONTROL_PORT          read-only status HTTP port (default: 8189)
  COMFRANCH_AUTOMATION_DIR        tasks/status/log dir (default: /workspace/automation)
  COMFRANCH_RUNNER_DIR            runner scripts dir (default: /opt/comfyranch/runner)
  COMFRANCH_LOG_DIR               worker log dir (default: /workspace/logs)
  COMFRANCH_RUNNER_WAIT_TASKS     seconds to wait for a manifest, 0 = forever

Telemetry for the (planned) in-worker QoL panel is written to
COMFRANCH_TELEMETRY_PATH (default: $COMFY_DIR/comfyranch_runtime.json):
  COMFRANCH_STARTED_AT, COMFRANCH_DPH, COMFRANCH_CONTRACT, COMFRANCH_LABEL.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("comfyranch.bootstrap")

DEFAULT_COMFY_DIR = "/opt/ComfyUI"
DEFAULT_MANIFEST_PATH = "/opt/comfyranch/manifest.json"
COMFY_REPO = os.environ.get("COMFRANCH_COMFY_REPO", "https://github.com/Comfy-Org/ComfyUI.git")
USER_AGENT = "ComfyRanch-Worker/0.1 (+https://github.com/Hype-Mode/ComfyRanch)"


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    log.debug("$ %s", " ".join(cmd))
    try:
        return subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, check=check,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"required executable not found: {cmd[0]}") from exc


# --------------------------------------------------------------------------- #
# Manifest loading
# --------------------------------------------------------------------------- #

def load_manifest() -> Optional[Dict[str, Any]]:
    url = os.environ.get("COMFRANCH_MANIFEST_URL", "").strip()
    if url:
        log.info("Loading manifest from URL %s", url)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=30) as resp:
            return json.load(resp)

    b64 = os.environ.get("COMFRANCH_MANIFEST_B64", "").strip()
    if b64:
        log.info("Loading manifest from COMFRANCH_MANIFEST_B64 (%d b64 chars)", len(b64))
        return json.loads(base64.b64decode(b64).decode("utf-8"))

    raw = os.environ.get("COMFRANCH_MANIFEST", "").strip()
    if raw:
        log.info("Loading manifest from COMFRANCH_MANIFEST (%d chars)", len(raw))
        return json.loads(raw)

    path = Path(os.environ.get("COMFRANCH_MANIFEST_PATH", DEFAULT_MANIFEST_PATH))
    if path.is_file():
        log.info("Loading manifest from %s", path)
        return json.loads(path.read_text(encoding="utf-8"))

    log.warning("No provisioning manifest supplied; launching ComfyUI as-is.")
    return None


def load_tasks_manifest() -> Optional[Dict[str, Any]]:
    """Load the headless batch baked into the deploy env.

    Sources, in order of precedence:
      1. COMFRANCH_TASKS_B64   - base64-encoded task manifest JSON
      2. COMFRANCH_TASKS       - raw task manifest JSON
      3. COMFRANCH_TASKS_PATH  - path on disk (default: none)
    """
    b64 = os.environ.get("COMFRANCH_TASKS_B64", "").strip()
    if b64:
        log.info("Loading task manifest from COMFRANCH_TASKS_B64 (%d b64 chars)", len(b64))
        return json.loads(base64.b64decode(b64).decode("utf-8"))

    raw = os.environ.get("COMFRANCH_TASKS", "").strip()
    if raw:
        log.info("Loading task manifest from COMFRANCH_TASKS (%d chars)", len(raw))
        return json.loads(raw)

    path = os.environ.get("COMFRANCH_TASKS_PATH", "").strip()
    if path and Path(path).is_file():
        log.info("Loading task manifest from %s", path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    log.info("No headless task manifest baked in; runner will idle until one is supplied.")
    return None


def write_tasks_file(automation_dir: Path) -> None:
    """Materialize the baked task manifest to tasks.json for the runner.

    Always starts from a clean slate: a previous boot's tasks.json/status/DONE
    must never cause a stale batch to re-run when a container restarts.
    """
    tasks_path = automation_dir / "tasks.json"
    for stale in (automation_dir / "DONE", automation_dir / "status.json"):
        try:
            stale.unlink()
        except OSError:
            pass

    tasks = load_tasks_manifest()
    if not tasks:
        try:
            tasks_path.unlink()
        except OSError:
            pass
        log.info("No headless task manifest baked in; runner will idle.")
        return

    try:
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        count = len(tasks.get("tasks") or []) if isinstance(tasks, dict) else len(tasks)
        log.info("Wrote %d baked task(s) to %s", count, tasks_path)
    except OSError as exc:
        log.error("Could not write baked tasks to %s: %s", tasks_path, exc)


# --------------------------------------------------------------------------- #
# ComfyUI checkout / version
# --------------------------------------------------------------------------- #

def _version_tuple(value: Optional[str]) -> tuple:
    if not value:
        return ()
    parts: List[int] = []
    for chunk in str(value).strip().lstrip("vV").split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def read_comfy_version(comfy_dir: Path) -> Optional[str]:
    version_file = comfy_dir / "comfyui_version.py"
    if version_file.is_file():
        for line in version_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("__version__"):
                return line.split("=", 1)[1].strip().strip("'\"")
    git = shutil.which("git")
    if git and (comfy_dir / ".git").is_dir():
        proc = run([git, "-C", str(comfy_dir), "describe", "--tags", "--abbrev=0"], check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip().lstrip("vV")
    return None


def find_comfy_dir() -> Optional[Path]:
    candidates = [
        os.environ.get("COMFY_DIR", ""),
        os.environ.get("COMFRANCH_COMFY_DIR", ""),
        DEFAULT_COMFY_DIR,
        "/workspace/ComfyUI",
        "/ComfyUI",
        "/app",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "main.py").is_file():
            return Path(candidate)
    return None


def sync_comfyui(comfy_dir: Path, required: Optional[str]) -> None:
    """Ensure ComfyUI exists and meets the manifest's minimum version."""
    if not comfy_dir.is_dir():
        log.info("ComfyUI not found; cloning %s into %s", COMFY_REPO, comfy_dir)
        comfy_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", COMFY_REPO, str(comfy_dir)])
    elif not (comfy_dir / "main.py").is_file():
        raise RuntimeError(f"{comfy_dir} exists but has no main.py")

    if not required:
        return
    current = read_comfy_version(comfy_dir)
    if current and _version_tuple(current) >= _version_tuple(required):
        log.info("ComfyUI %s already satisfies required >= %s", current, required)
        return

    git = shutil.which("git")
    if not git or not (comfy_dir / ".git").is_dir():
        log.warning(
            "ComfyUI %s is older than required %s and is not a git checkout; cannot fast-forward. "
            "Rebuild the worker image with a newer COMFYUI_REF.", current or "?", required,
        )
        return

    ref = required if required.startswith("v") else f"v{required}"
    log.info("Fast-forwarding ComfyUI %s -> %s", current or "?", ref)
    run([git, "-C", str(comfy_dir), "fetch", "--tags", "--depth", "1", "origin"], check=False)
    if run([git, "-C", str(comfy_dir), "checkout", ref], check=False).returncode != 0:
        run([git, "-C", str(comfy_dir), "checkout", required], check=False)


def pip_install_requirements(req_file: Path) -> None:
    if not req_file.is_file():
        return
    log.info("Installing Python requirements from %s", req_file)
    proc = run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=False)
    if proc.returncode != 0:
        log.warning("pip install failed for %s:\n%s", req_file, proc.stdout[-2000:])


# --------------------------------------------------------------------------- #
# Custom nodes
# --------------------------------------------------------------------------- #

def _repo_slug(node: Dict[str, Any]) -> Optional[str]:
    repo = (node.get("repo") or "").strip()
    if not repo:
        return None
    if repo.startswith("http://") or repo.startswith("https://") or repo.startswith("git@"):
        return repo
    if "/" in repo and " " not in repo:
        return f"https://github.com/{repo}.git"
    return None


def install_custom_node(node: Dict[str, Any], comfy_dir: Path) -> str:
    nodes_dir = comfy_dir / "custom_nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    slug = _repo_slug(node)
    node_id = node.get("id") or (Path(slug).stem if slug else "unknown")
    if not slug:
        log.warning("Custom node '%s' has no git repo (cnr id only); skipping. Install it manually or add a repo URL.", node_id)
        return f"skipped:{node_id}"

    version = (node.get("version") or "").strip() or None
    target = nodes_dir / Path(slug).stem.replace(".git", "")
    git = shutil.which("git")
    if not git:
        raise RuntimeError("git is required to install custom nodes")

    if (target / ".git").is_dir():
        run([git, "-C", str(target), "fetch", "--tags", "--depth", "1", "origin"], check=False)
        if version:
            run([git, "-C", str(target), "checkout", version], check=False)
        log.info("Updated custom node %s", target.name)
    else:
        cmd = [git, "clone", "--depth", "1"]
        if version:
            cmd += ["--branch", version]
        cmd += [slug, str(target)]
        log.info("Cloning custom node %s", target.name)
        run(cmd)

    if env_bool("COMFRANCH_INSTALL_REQUIREMENTS", True):
        pip_install_requirements(target / "requirements.txt")
    return f"installed:{target.name}"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def resolve_target(model: Dict[str, Any], models_dir: Path) -> Path:
    directory = (model.get("directory") or "").strip().replace("models/", "").strip("/")
    filename = (model.get("filename") or "").strip()
    if not filename and model.get("target"):
        directory = directory or str(Path(model["target"]).parent).replace("models", "", 1).strip("/\\")
        filename = Path(model["target"]).name
    if not filename:
        raise ValueError(f"model entry has no filename: {model!r}")
    return models_dir / directory / filename if directory else models_dir / filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_size(model: Dict[str, Any]) -> Optional[int]:
    for key in ("sizeBytes", "size_bytes"):
        if model.get(key):
            try:
                return int(model[key])
            except (TypeError, ValueError):
                return None
    return None


def is_model_present(target: Path, model: Dict[str, Any]) -> bool:
    if not target.is_file():
        return False
    expected = _expected_size(model)
    actual = target.stat().st_size
    if expected and abs(actual - expected) > max(1024 * 1024, expected * 0.01):
        log.warning("Model %s size mismatch (%d vs expected %d); re-downloading.", target.name, actual, expected)
        return False
    if env_bool("COMFRANCH_VERIFY_SHA256", True) and model.get("sha256"):
        if _sha256(target).lower() != str(model["sha256"]).lower():
            log.warning("Model %s sha256 mismatch; re-downloading.", target.name)
            return False
    return True


def _copy_from_source(source: Optional[Path], target: Path) -> bool:
    if not source or not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("Linking %s from mounted model cache", target.name)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return True


def download_model(model: Dict[str, Any], models_dir: Path) -> str:
    target = resolve_target(model, models_dir)
    name = target.name
    source_dir = os.environ.get("COMFRANCH_MODEL_SOURCE_DIR", "").strip()
    if source_dir:
        rel = target.relative_to(models_dir)
        if _copy_from_source(Path(source_dir) / rel, target):
            return f"copied:{name}"

    if is_model_present(target, model):
        log.info("Model %s already present", name)
        return f"present:{name}"

    url = (model.get("url") or "").strip()
    if not url:
        log.warning("Model %s has no URL; cannot provision. Add a URL to the workflow export.", name)
        return f"unresolved:{name}"

    target.parent.mkdir(parents=True, exist_ok=True)
    header = None
    token = os.environ.get("HF_TOKEN", "").strip()
    if token and "huggingface.co" in url:
        header = f"Authorization: Bearer {token}"

    aria2c = shutil.which("aria2c")
    log.info("Downloading %s -> %s", name, target.parent)
    if aria2c:
        cmd = [
            aria2c, "--continue=true", "--auto-file-renaming=false", "--allow-overwrite=true",
            "--file-allocation=none", "--max-connection-per-server=8", "--split=8",
            "--min-split-size=1M", "--console-log-level=warn", "--summary-interval=0",
            "--user-agent=" + USER_AGENT, "-d", str(target.parent), "-o", name,
        ]
        if header:
            cmd.append(f"--header={header}")
        cmd.append(url)
        proc = run(cmd, check=False)
        if proc.returncode != 0 and not target.is_file():
            raise RuntimeError(f"aria2c failed for {name}: {proc.stdout[-500:]}")
    else:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **({"Authorization": header.split(": ", 1)[1]} if header else {})})
        partial = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(request, timeout=120) as resp, partial.open("wb") as out:
            shutil.copyfileobj(resp, out, 1024 * 1024)
        partial.replace(target)

    if model.get("sha256") and env_bool("COMFRANCH_VERIFY_SHA256", True):
        if _sha256(target).lower() != str(model["sha256"]).lower():
            raise RuntimeError(f"sha256 mismatch after downloading {name}")
    return f"downloaded:{name}"


# --------------------------------------------------------------------------- #
# Workflow files (preloaded into ComfyUI's UI sidebar)
# --------------------------------------------------------------------------- #

def resolve_user_workflows_dir(comfy_dir: Path) -> Path:
    """Where ComfyUI's workflow sidebar reads from (user/default/workflows)."""
    override = os.environ.get("COMFRANCH_USER_WORKFLOWS_DIR", "").strip()
    if override:
        return Path(override)
    user_dir = os.environ.get("COMFRANCH_USER_DIR", "").strip()
    base = Path(user_dir) if user_dir else comfy_dir / "user" / "default"
    return base / "workflows"


def write_workflow_files(comfy_dir: Path, manifest: Optional[Dict[str, Any]]) -> None:
    """Drop the manifest's workflow graph(s) into the ComfyUI UI's workflow list.

    API-format JSON is loadable in the modern ComfyUI frontend (without layout),
    so the headless dispatch path and the UI share a single artifact.
    """
    workflows = list((manifest or {}).get("workflows") or [])
    if not workflows:
        return
    target_dir = resolve_user_workflows_dir(comfy_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("Could not create workflows dir %s: %s", target_dir, exc)
        return

    for entry in workflows:
        content = entry.get("content")
        if not isinstance(content, dict) or not content:
            continue
        raw_name = str(entry.get("filename") or entry.get("name") or "workflow.json").strip()
        filename = Path(raw_name).name or "workflow.json"
        if not filename.lower().endswith(".json"):
            filename += ".json"
        path = target_dir / filename
        try:
            path.write_text(json.dumps(content, indent=2), encoding="utf-8")
            log.info("Wrote workflow '%s' -> %s", entry.get("name") or filename, path)
        except OSError as exc:
            log.warning("Could not write workflow %s: %s", path, exc)


# --------------------------------------------------------------------------- #
# Telemetry (for the planned in-worker QoL panel)
# --------------------------------------------------------------------------- #

def collect_gpu_info() -> Dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {}
    proc = run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    line = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    info = {"name": parts[0] if parts else None, "driver": parts[2] if len(parts) > 2 else None}
    if len(parts) > 1:
        try:
            info["vramTotalMb"] = int(parts[1])
        except ValueError:
            pass
    return info


def write_runtime_info(comfy_dir: Path, comfy_version: Optional[str]) -> None:
    path = Path(os.environ.get("COMFRANCH_TELEMETRY_PATH", str(comfy_dir / "comfyranch_runtime.json")))
    payload = {
        "schemaVersion": 1,
        "contract": os.environ.get("COMFRANCH_CONTRACT"),
        "label": os.environ.get("COMFRANCH_LABEL"),
        "dph": os.environ.get("COMFRANCH_DPH"),
        "startedAt": os.environ.get("COMFRANCH_STARTED_AT") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comfyVersion": comfy_version,
        "comfyDir": str(comfy_dir),
        "gpu": collect_gpu_info(),
        "writtenAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Wrote runtime info to %s", path)
    except OSError as exc:
        log.warning("Could not write runtime info to %s: %s", path, exc)


# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #

def comfyui_extra_args() -> str:
    """Extra ComfyUI args. `--preview-method auto` is added by default so the
    browser connects as a *visual counterpart* and sees live latent previews."""
    extra = os.environ.get("COMFRANCH_EXTRA_ARGS", "").strip()
    if "--preview-method" not in extra:
        extra = f"{extra} --preview-method auto".strip()
    return extra


def build_comfyui_cmd(comfy_dir: Path) -> List[str]:
    port = os.environ.get("COMFRANCH_PORT", "8188")
    cmd = [sys.executable, str(comfy_dir / "main.py"), "--listen", "0.0.0.0", "--port", port]
    extra = comfyui_extra_args()
    if "--enable-cors-header" not in extra:
        cmd += ["--enable-cors-header", "*"]
    if extra:
        cmd += extra.split()
    return cmd


def _log_dir() -> Path:
    path = Path(os.environ.get("COMFRANCH_LOG_DIR", "/workspace/logs"))
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return path


def launch_comfyui(comfy_dir: Path) -> int:
    """Foreground launch (used by `--no-runner`/legacy paths)."""
    cmd = build_comfyui_cmd(comfy_dir)
    log.info("Launching ComfyUI: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(comfy_dir))


def launch_comfyui_background(comfy_dir: Path) -> subprocess.Popen:
    """Start ComfyUI detached so the control server + runner can run alongside."""
    cmd = build_comfyui_cmd(comfy_dir)
    logfile = _log_dir() / "comfyui.log"
    log.info("Launching ComfyUI (background): %s -> %s", " ".join(cmd), logfile)
    handle = open(logfile, "ab", buffering=0)
    return subprocess.Popen(  # noqa: SIM115 - handle lives for the process lifetime
        cmd, cwd=str(comfy_dir), stdout=handle, stderr=subprocess.STDOUT
    )


# --------------------------------------------------------------------------- #
# Control server + on-instance runner
# --------------------------------------------------------------------------- #

def _runner_dir() -> Path:
    return Path(os.environ.get("COMFRANCH_RUNNER_DIR", "/opt/comfyranch/runner"))


def start_control_server(automation_dir: Path) -> Optional[Any]:
    """Start the read-only status endpoint the app polls for progress."""
    port = env_int("COMFRANCH_CONTROL_PORT", 8189)
    runner_dir = _runner_dir()
    if not (runner_dir / "control_server.py").is_file():
        log.warning("status server not found in %s; skipping", runner_dir)
        return None
    if str(runner_dir) not in sys.path:
        sys.path.insert(0, str(runner_dir))
    try:
        import control_server  # type: ignore  # noqa: PLC0415

        return control_server.start_control_server(port, automation_dir, logger=log)
    except Exception as exc:  # noqa: BLE001 - never block ComfyUI on the status plane
        log.error("status server failed to start: %s", exc)
        return None


def start_runner(automation_dir: Path) -> Optional[subprocess.Popen]:
    """Start the headless runner (waits for a manifest, then runs it)."""
    runner = _runner_dir() / "runner.py"
    if not runner.is_file():
        log.warning("runner not found at %s; skipping", runner)
        return None
    port = os.environ.get("COMFRANCH_PORT", "8188")
    cmd = [
        sys.executable,
        str(runner),
        "--tasks", str(automation_dir / "tasks.json"),
        "--status", str(automation_dir / "status.json"),
        "--outdir", str(automation_dir / "output"),
        "--comfy", f"http://127.0.0.1:{port}",
        "--wait-tasks", os.environ.get("COMFRANCH_RUNNER_WAIT_TASKS", "0"),
        "--task-timeout", os.environ.get("COMFRANCH_RUNNER_TASK_TIMEOUT", "1800"),
    ]
    logfile = _log_dir() / "runner.log"
    log.info("Launching runner: %s -> %s", " ".join(cmd), logfile)
    handle = open(logfile, "ab", buffering=0)
    return subprocess.Popen(  # noqa: SIM115 - handle lives for the process lifetime
        cmd, stdout=handle, stderr=subprocess.STDOUT
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def bootstrap(dry_run: bool, no_launch: bool) -> int:
    comfy_dir = find_comfy_dir() or Path(os.environ.get("COMFY_DIR", DEFAULT_COMFY_DIR))
    models_dir = Path(os.environ.get("COMFRANCH_MODELS_DIR", str(comfy_dir / "models")))

    manifest = load_manifest()
    required_version: Optional[str] = None
    models: List[Dict[str, Any]] = []
    nodes: List[Dict[str, Any]] = []
    if manifest:
        required_version = manifest.get("comfyMinVersion")
        models = list(manifest.get("models") or [])
        nodes = list(manifest.get("customNodes") or [])

    log.info("Worker plan: comfy_dir=%s models_dir=%s comfy_min=%s models=%d nodes=%d",
             comfy_dir, models_dir, required_version or "n/a", len(models), len(nodes))

    if dry_run:
        for node in nodes:
            log.info("[plan] custom node: %s (%s)", node.get("id"), _repo_slug(node) or "NO REPO")
        for model in models:
            try:
                target = resolve_target(model, models_dir)
            except ValueError as exc:
                log.warning("[plan] bad model entry: %s", exc)
                continue
            log.info("[plan] model: %s  <- %s", target, model.get("url") or "NO URL")
        for entry in (manifest or {}).get("workflows") or []:
            log.info("[plan] workflow: %s (format=%s)", entry.get("filename"), entry.get("format"))
        if required_version:
            log.info("[plan] ComfyUI must be >= %s", required_version)
        return 0

    sync_comfyui(comfy_dir, required_version)
    comfy_version = read_comfy_version(comfy_dir)

    if env_bool("COMFRANCH_INSTALL_CUSTOM_NODES", True) and nodes:
        with ThreadPoolExecutor(max_workers=1) as pool:  # serial: git operations share a work tree
            futures = {pool.submit(install_custom_node, node, comfy_dir): node for node in nodes}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - keep provisioning going
                    log.warning("Custom node %s failed: %s", futures[future].get("id"), exc)

    pending = []
    for model in models:
        try:
            target = resolve_target(model, models_dir)
        except ValueError as exc:
            log.warning("Skipping bad model entry: %s", exc)
            continue
        source_dir = os.environ.get("COMFRANCH_MODEL_SOURCE_DIR", "").strip()
        if not (source_dir and (Path(source_dir) / target.relative_to(models_dir)).is_file()) and is_model_present(target, model):
            log.info("Model %s already present", target.name)
            continue
        pending.append(model)

    if pending:
        workers = max(1, env_int("COMFRANCH_MAX_CONCURRENT_DOWNLOADS", 3))
        log.info("Materializing %d model file(s) with %d concurrent download(s)", len(pending), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_model, model, models_dir): model for model in pending}
            for future in as_completed(futures):
                try:
                    log.info("  %s", future.result())
                except Exception as exc:  # noqa: BLE001
                    log.error("Model %s failed: %s", futures[future].get("filename"), exc)

    write_workflow_files(comfy_dir, manifest)
    write_runtime_info(comfy_dir, comfy_version)

    if no_launch or not env_bool("COMFRANCH_LAUNCH", True):
        log.info("Provisioning complete (launch disabled).")
        return 0

    # Legacy foreground mode: just run ComfyUI (no control plane/runner).
    if not env_bool("COMFRANCH_AUTORUN", True):
        return launch_comfyui(comfy_dir)

    # Headless mode: control endpoint (for task injection) + on-instance runner,
    # with ComfyUI in the background so the browser can attach as an observer.
    automation_dir = Path(os.environ.get("COMFRANCH_AUTOMATION_DIR", "/workspace/automation"))
    automation_dir.mkdir(parents=True, exist_ok=True)

    # Materialize the batch baked into the deploy env, then expose read-only
    # progress. The runner queues it the instant ComfyUI answers /system_stats.
    write_tasks_file(automation_dir)
    start_control_server(automation_dir)
    proc = launch_comfyui_background(comfy_dir)
    runner = start_runner(automation_dir) if env_bool("COMFRANCH_RUNNER", True) else None

    log.info("ComfyUI PID %s; control plane live; awaiting task manifest.", proc.pid)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        log.info("Interrupt received; terminating worker processes")
        if runner and runner.poll() is None:
            runner.terminate()
        proc.terminate()
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(description="ComfyRanch worker bootstrap")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without changing anything")
    parser.add_argument("--no-launch", action="store_true", help="provision only; do not start ComfyUI")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    try:
        return bootstrap(args.dry_run, args.no_launch)
    except Exception as exc:  # noqa: BLE001
        log.exception("Bootstrap failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
