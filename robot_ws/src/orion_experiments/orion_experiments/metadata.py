"""Provenance for one trial: what code, firmware and model ran, and under which conditions."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
from typing import Optional

SCHEMA_VERSION = 1
OUTCOME_NAMES = {0: 'SUCCESS', 1: 'TIMEOUT', 2: 'ENTRAPPED', 3: 'ABORT', 4: 'SAFETY'}


def find_repo_root(start: str) -> Optional[str]:
    """Nearest ancestor of start (or start) that holds .git (a dir, or a file in worktrees)."""
    path = os.path.realpath(start)
    while True:
        if os.path.exists(os.path.join(path, '.git')):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def git_state(repo: Optional[str]) -> tuple[Optional[str], Optional[bool]]:
    """(HEAD sha, working tree dirty) of repo, or (None, None) if it is not a git checkout."""
    if not repo:
        return None, None
    try:
        sha = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True,
                             text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(
            ['git', '-C', repo, 'status', '--porcelain', '--untracked-files=no'],
            capture_output=True, text=True, check=True,
            timeout=10).stdout.strip() != ''
    except (OSError, subprocess.SubprocessError):
        return None, None
    return sha, dirty


def onnx_provenance(model_dir: str) -> Optional[dict]:
    """
    sha256 of every file named by model_dir/manifest.json, recomputed from disk.

    None for a non-learnt controller (empty model_dir). A file whose recomputed hash differs
    from the manifest is recorded with manifest_match false rather than hidden.
    """
    if not model_dir:
        return None
    with open(os.path.join(model_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    files = {}
    for role, name in manifest.get('files', {}).items():
        if not name:
            continue
        h = hashlib.sha256()
        with open(os.path.join(model_dir, name), 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        files[name] = {'role': role, 'sha256': h.hexdigest(),
                       'manifest_match': h.hexdigest() == manifest.get('sha256', {}).get(name)}
    return {'model_dir': os.path.abspath(model_dir), 'controller': manifest.get('controller'),
            'obs_spec_hash': manifest.get('obs_spec_hash'), 'files': files}


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='milliseconds')


def write_json_atomic(path: str, data: dict) -> None:
    """Write JSON so a crash never leaves a half-written file behind."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)
