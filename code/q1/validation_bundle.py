"""Lossless, labelled SYNTHETIC verification snapshots; never empirical input data.

Capture reuses existing metrics/checkpoints after their original manifests pass.
Restore requires tracked gzip snapshots, not ignored results or model training.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/"tests/q1/fixtures/verified_runs"
RUNS = {
    "m4":"results/verification/approach1/q1_synthetic_svi_verification",
    "m5":"results/m5_final/verification/approach2/q1_synthetic_svi_optimizer_comparison",
    "m6":"results/verification/approach2/q1_synthetic_neural_verification",
    "m7":"results/verification/approach2/q1_synthetic_missing_quotes",
    "m4_gate":"results/approach1/q1_510050",
    "m5_gate":"results/approach2/q1_510050",
    "m6_gate":"results/approach2/q1_510050_neural",
    "m7_gate":"results/approach2/q1_510050_missing_quotes",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def confined(path, root=ROOT):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Path escapes repository")
    return path


def capture(directory=DEFAULT):
    directory = confined(directory)
    directory.mkdir(parents=True,exist_ok=False)
    entries,inventory = [],[]
    for stage,relative in RUNS.items():
        source = ROOT/relative
        manifest_bytes = (source/"artifact_manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        files = manifest["files"]
        for item in files:
            path = confined(source/item["name"],source)
            data = path.read_bytes()
            if len(data) != item["bytes"] or sha(data) != item["sha256"]:
                raise ValueError("Original artifact integrity failure: "+str(path))
            inventory.append({"stage":stage,"original":path.relative_to(ROOT).as_posix(),"bytes":len(data),"sha256":sha(data)})
        selected = [source/"metrics.json",source/"artifact_manifest.json"]
        if (source/"diagnostics.json").exists():
            selected.append(source/"diagnostics.json")
        selected.extend(sorted(source.rglob("*.pt")))
        for path in selected:
            data = path.read_bytes()
            name = stage+"_"+sha(path.relative_to(source).as_posix().encode())[:12]+".gz"
            compressed = gzip.compress(data,mtime=0)
            (directory/name).write_bytes(compressed)
            entries.append({"stage":stage,"original":path.relative_to(ROOT).as_posix(),"archive":name,"bytes":len(data),
                            "sha256":sha(data),"compressed_sha256":sha(compressed),"compressed_bytes":len(compressed)})
    result = {"kind":"SYNTHETIC_VERIFICATION_AND_BLOCKED_GATE_EVIDENCE_NOT_MARKET_DATA","format":1,
              "purpose":"Lossless replay/inference audit without training; original HTML is indexed, not duplicated",
              "runs":RUNS,"files":entries,"original_artifact_inventory":inventory}
    (directory/"manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8",newline="\n")
    return result


def restore(directory, destination):
    directory,destination = confined(directory),confined(destination)
    if destination.exists():
        raise FileExistsError("Restore requires a new output directory")
    manifest = json.loads((directory/"manifest.json").read_bytes())
    if manifest["kind"] != "SYNTHETIC_VERIFICATION_AND_BLOCKED_GATE_EVIDENCE_NOT_MARKET_DATA":
        raise ValueError("Unknown bundle origin")
    destination.mkdir(parents=True)
    for item in manifest["files"]:
        payload = confined(directory/item["archive"],directory).read_bytes()
        if sha(payload) != item["compressed_sha256"]:
            raise ValueError("Compressed snapshot checksum mismatch")
        data = gzip.decompress(payload)
        if len(data) != item["bytes"] or sha(data) != item["sha256"]:
            raise ValueError("Restored snapshot checksum mismatch")
        path = confined(destination/item["original"],destination)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["capture","restore"])
    parser.add_argument("--bundle",type=Path,default=DEFAULT)
    parser.add_argument("--destination",type=Path)
    args = parser.parse_args()
    if args.action == "capture":
        result = capture(args.bundle)
    else:
        if args.destination is None:
            parser.error("restore needs --destination")
        result = restore(args.bundle,args.destination)
    print(json.dumps({"snapshot_files":len(result["files"]),"source_artifacts_indexed":len(result["original_artifact_inventory"]),"training_performed":False}))


if __name__ == "__main__":
    main()
