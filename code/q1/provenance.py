"""Q1 local artifact verification. No network access or cache side effects."""

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_MANIFESTS = (
    ("data/clean/q1/m1_evidence/snapshot_manifest.json", "b34faa9acfaadc4efc70788c98017a7fae1f06ac48d6c695817ca0f52c7ea438"),
    ("data/clean/q1/m15_recovery/audit_manifest.json", "be7b08c321a1ddacba28f7b81f3d205726dba9f576b31456f7536e8435adb7de"),
)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def local_path(root, relative):
    path = Path(relative)
    root = Path(root).resolve()
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Artifact paths must be relative and cannot traverse parents")
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Artifact path escapes repository root")
    return result


def verify_manifest(root, relative, expected_sha256):
    path = local_path(root, relative)
    payload = path.read_bytes()
    if sha256(payload) != expected_sha256:
        raise ValueError(f"Manifest SHA-256 mismatch: {relative}")
    manifest = json.loads(payload)
    entries = manifest["files"]
    seen = set()
    for item in entries:
        if item["path"] in seen:
            raise ValueError("Duplicate manifest artifact path")
        seen.add(item["path"])
        artifact = local_path(root, item["path"])
        if artifact.stat().st_size >= 50_000_000:
            raise ValueError("Audit artifact exceeds the 50 MB snapshot policy")
        content = artifact.read_bytes()
        if len(content) != item["bytes"] or sha256(content) != item["sha256"]:
            raise ValueError(f"Artifact checksum/size mismatch: {item['path']}")
    return {"path": relative, "sha256": expected_sha256, "files": entries}


def verify_existing_inputs(root=ROOT):
    return [verify_manifest(root, path, checksum) for path, checksum in INPUT_MANIFESTS]


class OfflineArchive:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.manifests = verify_existing_inputs(self.root)
        self.files = {item["path"]: item for manifest in self.manifests for item in manifest["files"]}
        self.consumed = set()

    def read_bytes(self, relative):
        if relative not in self.files:
            raise ValueError(f"File is not covered by a pinned manifest: {relative}")
        payload = local_path(self.root, relative).read_bytes()
        entry = self.files[relative]
        if len(payload) != entry["bytes"] or sha256(payload) != entry["sha256"]:
            raise ValueError(f"Artifact changed after verification: {relative}")
        self.consumed.add(relative)
        return payload

    def json(self, relative):
        return json.loads(self.read_bytes(relative))

    def csv(self, relative):
        import io
        return pd.read_csv(io.BytesIO(self.read_bytes(relative)), dtype=str, keep_default_na=False)
