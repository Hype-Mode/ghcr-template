#!/usr/bin/env python3
"""ComfyRanch on-instance headless runner.

Waits for ComfyUI to become ready, then executes a task manifest **sequentially
on the instance**, independent of any browser. Each task carries a pre-built
API graph (the app does the parameter injection); the runner POSTs it to
``/prompt`` with the UI graph attached as ``extra_data.extra_pnginfo.workflow``
so every output embeds the full visual canvas.

The browser stays the *visual counterpart*: it connects to the same ComfyUI on
``0.0.0.0:8188``, so it receives the same execution/progress broadcasts and can
load the preloaded UI workflow to watch nodes light up in real time.

Stdlib only (``urllib``) — no extra dependencies, nothing to break the image.

Task manifest schema (written by the control server from the app):

    {
      "workflowName": "Z-Image Turbo",
      "uiGraph": { ... },                # optional, UI-format canvas
      "tasks": [
        { "id": "job-1", "label": "...", "graph": { ...API graph... } }
      ]
    }

A bare JSON array of task objects is also accepted (one task per element, each
with an ``id``/``graph``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_AUTOMATION_DIR = os.environ.get("COMFRANCH_AUTOMATION_DIR", "/workspace/automation")
DEFAULT_COMFY = os.environ.get("COMFRANCH_COMFY_URL", "http://127.0.0.1:8188")
USER_AGENT = "ComfyRanch-Runner/0.1"

STATE_WAITING = "waiting"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_ERROR = "error"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [runner] {msg}", flush=True)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:  # never let status writes kill the run
        log(f"could not write {path}: {exc}")


class StatusWriter:
    def __init__(
        self, path: Path, total: int, workflow_name: str, task_ids: List[str]
    ) -> None:
        self.path = path
        self.total = total
        self.workflow_name = workflow_name
        self.task_ids = task_ids
        self.completed = 0
        self.failed = 0
        self.outputs: List[Dict[str, Any]] = []
        self.state = STATE_RUNNING
        self.current_task_id: Optional[str] = None
        self.current_task_label: Optional[str] = None
        self.error: Optional[str] = None

    def write(self) -> None:
        _atomic_write_json(
            self.path,
            {
                "schemaVersion": 1,
                "state": self.state,
                "workflowName": self.workflow_name,
                "total": self.total,
                "taskIds": self.task_ids,
                "completed": self.completed,
                "failed": self.failed,
                "currentTaskId": self.current_task_id,
                "currentTaskLabel": self.current_task_label,
                "outputs": self.outputs,
                "error": self.error,
                "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )


class ComfyUIClient:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def _get(self, path: str, timeout: float = 5.0) -> Optional[bytes]:
        req = urllib.request.Request(f"{self.base}{path}", headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError, TimeoutError):
            return None

    def wait_for_ready(self, timeout_seconds: int = 1800) -> bool:
        log(f"probing ComfyUI at {self.base} ...")
        start = time.time()
        while time.time() - start < timeout_seconds:
            body = self._get("/system_stats", timeout=3.0)
            if body is not None:
                log("ComfyUI is live and responsive.")
                return True
            time.sleep(2)
        return False

    def queue_prompt(self, graph: Dict[str, Any], ui_graph: Optional[Dict[str, Any]]) -> str:
        payload: Dict[str, Any] = {"client_id": self.client_id, "prompt": graph}
        if ui_graph:
            payload["extra_data"] = {"extra_pnginfo": {"workflow": ui_graph}}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/prompt",
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:600]
            raise RuntimeError(f"ComfyUI rejected prompt (HTTP {exc.code}): {detail}")
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Could not reach ComfyUI to queue prompt: {exc}")
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI returned no prompt_id: {result}")
        return str(prompt_id)

    def wait_for_outputs(
        self, prompt_id: str, timeout_seconds: int = 1800, poll_seconds: float = 2.0
    ) -> List[Dict[str, Any]]:
        start = time.time()
        while time.time() - start < timeout_seconds:
            body = self._get(f"/history/{urllib.parse.quote(prompt_id)}", timeout=5.0)
            if body is not None:
                history = json.loads(body.decode("utf-8"))
                entry = history.get(prompt_id)
                if entry is not None:
                    status = entry.get("status") or {}
                    status_str = status.get("status_str") or ""
                    if status_str == "error":
                        raise RuntimeError(
                            f"ComfyUI execution error: {json.dumps(status.get('messages', ''))[:400]}"
                        )
                    if status.get("completed") or status_str == "success":
                        return list((entry.get("outputs") or {}).values())
            time.sleep(poll_seconds)
        raise TimeoutError(f"prompt {prompt_id} did not finish within {timeout_seconds}s")

    def download_artifact(self, image: Dict[str, Any], dest_dir: Path) -> Optional[str]:
        query = urllib.parse.urlencode(
            {
                "filename": image.get("filename", ""),
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            }
        )
        req = urllib.request.Request(f"{self.base}/view?{query}", headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                blob = resp.read()
        except (urllib.error.URLError, OSError, TimeoutError):
            return None
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / str(image.get("filename") or "output.bin")
        try:
            dest.write_bytes(blob)
            return str(dest)
        except OSError:
            return None


def extract_images(outputs) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    for node_output in outputs or []:
        for image in (node_output or {}).get("images") or []:
            if image and image.get("filename"):
                images.append(image)
    return images


def normalize_manifest(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, list):
        return {"workflowName": "workflow", "tasks": raw}
    if isinstance(raw, dict):
        return raw
    return {"workflowName": "workflow", "tasks": []}


def run_pipeline(
    tasks_path: Path,
    status_path: Path,
    comfy_base: str,
    output_dir: Path,
    wait_ready_seconds: int,
    wait_tasks_seconds: int,
    per_task_timeout: int,
) -> int:
    # 1. Wait for the app to push a task manifest (control server writes it).
    log(f"waiting for task manifest at {tasks_path} ...")
    start = time.time()
    raw: Optional[Any] = None
    while True:
        if tasks_path.is_file():
            try:
                raw = json.loads(tasks_path.read_text(encoding="utf-8"))
                break
            except (OSError, json.JSONDecodeError) as exc:
                log(f"manifest present but unreadable ({exc}); retrying")
        if wait_tasks_seconds and time.time() - start > wait_tasks_seconds:
            log("no manifest arrived before the wait window expired; idling")
            return 0
        time.sleep(2)

    manifest = normalize_manifest(raw)
    ui_graph = manifest.get("uiGraph")
    workflow_name = str(manifest.get("workflowName") or "workflow")
    tasks = [t for t in (manifest.get("tasks") or []) if isinstance(t, dict) and t.get("graph")]
    log(f"manifest received: {len(tasks)} task(s) for '{workflow_name}'")

    task_ids = [str(t.get("id") or f"task_{i:03d}") for i, t in enumerate(tasks, start=1)]
    writer = StatusWriter(
        status_path,
        total=len(tasks),
        workflow_name=workflow_name,
        task_ids=task_ids,
    )
    writer.write()

    client = ComfyUIClient(comfy_base)
    if not client.wait_for_ready(timeout_seconds=wait_ready_seconds):
        writer.state = STATE_ERROR
        writer.error = "ComfyUI did not become ready"
        writer.write()
        log("ComfyUI never became ready; aborting")
        return 1

    for index, task in enumerate(tasks, start=1):
        task_id = str(task.get("id") or f"task_{index:03d}")
        label = str(task.get("label") or task_id)
        writer.current_task_id = task_id
        writer.current_task_label = label
        writer.write()
        log(f"=== [{index}/{len(tasks)}] {label} ({task_id}) ===")
        try:
            task_ui_graph = task.get("uiGraph") or ui_graph
            prompt_id = client.queue_prompt(task["graph"], task_ui_graph)
            log(f"    queued as {prompt_id}")
            outputs = client.wait_for_outputs(prompt_id, timeout_seconds=per_task_timeout)
            images = extract_images(outputs)
            for image in images:
                local = client.download_artifact(image, output_dir)
                writer.outputs.append(
                    {
                        "taskId": task_id,
                        "promptId": prompt_id,
                        "filename": image.get("filename"),
                        "subfolder": image.get("subfolder", ""),
                        "type": image.get("type", "output"),
                        "localPath": local,
                    }
                )
            writer.completed += 1
            log(f"    {len(images)} output(s) saved")
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            writer.failed += 1
            writer.error = str(exc)
            log(f"    FAILED: {exc}")
        writer.write()

    writer.current_task_id = None
    writer.current_task_label = None
    writer.state = STATE_ERROR if writer.failed and not writer.completed else STATE_DONE
    writer.write()
    try:
        (status_path.parent / "DONE").write_text(
            json.dumps({"completed": writer.completed, "failed": writer.failed}), encoding="utf-8"
        )
    except OSError:
        pass
    log(f"batch complete: {writer.completed} ok, {writer.failed} failed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ComfyRanch on-instance runner")
    parser.add_argument("--tasks", default=str(Path(DEFAULT_AUTOMATION_DIR) / "tasks.json"))
    parser.add_argument("--status", default=str(Path(DEFAULT_AUTOMATION_DIR) / "status.json"))
    parser.add_argument("--comfy", default=DEFAULT_COMFY)
    parser.add_argument("--outdir", default=str(Path(DEFAULT_AUTOMATION_DIR) / "output"))
    parser.add_argument("--wait-ready", type=int, default=1800)
    parser.add_argument("--wait-tasks", type=int, default=0, help="0 = wait forever")
    parser.add_argument("--task-timeout", type=int, default=1800)
    args = parser.parse_args()

    return run_pipeline(
        tasks_path=Path(args.tasks),
        status_path=Path(args.status),
        comfy_base=args.comfy,
        output_dir=Path(args.outdir),
        wait_ready_seconds=args.wait_ready,
        wait_tasks_seconds=args.wait_tasks,
        per_task_timeout=args.task_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
