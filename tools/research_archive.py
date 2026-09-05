"""Archive retired research files without changing retained experiment code.

The plan is conservative: retain transitive file/module references, including
historical helpers still imported by current runners. Archives contain source
only, never outputs or checkpoints. No Git operation is performed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / ".research_archive" / "retired-validation-20260905.zip"
MANIFEST = ROOT / "docs" / "research_archive_manifest.json"
DIRECTORIES = ("experiments", "config", "ibex_bash", "tests")
SUFFIXES = {".py", ".yaml", ".yml", ".sh"}
SNAPSHOTS = ("README.md", "docs/Verify_experiments.md")
SIDECAR_RE = re.compile(r"(?:outputs?|config)/[\w./-]+\.(?:json|yaml|yml)\b")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory() -> dict[str, str]:
    files = {}
    for directory in DIRECTORIES:
        for path in sorted((ROOT / directory).iterdir()):
            if path.is_file() and path.suffix in SUFFIXES:
                if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                    raise ValueError(f"unsafe source: {path}")
                files[path.relative_to(ROOT).as_posix()] = path.read_text(encoding="utf-8-sig")
    return files


def historical_candidate(name: str) -> bool:
    path = Path(name)
    stem = path.stem
    if path.parent.name == "experiments":
        if any(token in stem for token in ("Task123", "Task4", "Task5", "Uniform")):
            return False
        return stem.startswith(("Audit_", "Select_", "Diagnose_", "Verify_", "Screen_"))
    if path.parent.name == "config":
        if any(token in stem for token in ("Task123", "Task4", "Task5", "Uniform")):
            return False
        return stem.startswith(("Verify_", "Confirm_", "Other_", "Calibrate_"))
    if path.parent.name == "ibex_bash":
        if any(token in stem for token in ("task123", "task4", "task5", "uniform")):
            return False
        return stem.startswith(("verify_", "confirm_"))
    return False


@lru_cache(maxsize=1)
def reference_index(names: tuple[str, ...]) -> tuple:
    lookup = {}
    for name in names:
        path = Path(name)
        tokens = [path.name]
        if path.parent.name == "experiments":
            tokens.append(path.stem)
        for token in tokens:
            lookup.setdefault(token, set()).add(name)
    pattern = re.compile(r"(?<![\w])(?:" + "|".join(
        re.escape(token) for token in sorted(lookup, key=len, reverse=True)
    ) + r")(?![\w.])")
    return pattern, lookup


@lru_cache(maxsize=None)
def read_sidecar(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def references(text: str, files: dict[str, str], visited=None) -> set[str]:
    # Match both explicit paths and old flat imports. Boundary checks prevent
    # e.g. a reference to v1.1 retaining v1.10 by substring coincidence.
    pattern, lookup = reference_index(tuple(files))
    found = set()
    for match in pattern.finditer(text):
        found.update(lookup[match.group()])
    # Frozen selection/source manifests can name runtime dependencies that
    # appear in neither an import statement nor the top-level configuration.
    visited = set() if visited is None else visited
    for match in SIDECAR_RE.finditer(text):
        name = match.group()
        if name in files or name in visited:
            continue
        visited.add(name)
        path = ROOT / name
        if path.is_file() and path.resolve().is_relative_to(ROOT) and not path.is_symlink():
            found.update(references(read_sidecar(path), files, visited))
    return found


def make_plan() -> dict:
    files = inventory()
    candidates = {name for name in files if historical_candidate(name)}
    refs = {name: references(body, files) - {name} for name, body in files.items()}
    tests = {name for name in files if name.startswith("tests/")}
    # Keep substantive library tests; retire version-specific workflow tests
    # only when they require files that will actually be archived.
    roots = set(files) - candidates - tests
    for directory in ("FMT_Utils", "FLowUtils", "DeepUtils", "pnn", "tools"):
        for path in (ROOT / directory).rglob("*.py"):
            if path.resolve() != Path(__file__).resolve():
                roots.update(references(path.read_text(encoding="utf-8-sig"), files))
    keep = set(roots)
    while True:
        expanded = keep | set().union(*(refs[name] for name in keep))
        if expanded == keep:
            break
        keep = expanded
    archive = set(files) - keep - tests
    # Iterative because tests may import other tests.
    while True:
        newly = {name for name in tests - archive if refs[name] & archive}
        if not newly:
            break
        archive.update(newly)
    keep = set(files) - archive
    dangling = {name: sorted(refs[name] & archive) for name in keep if refs[name] & archive}
    if dangling:
        raise ValueError(f"retained references to archived files: {dangling}")
    entries = []
    for name in sorted(archive):
        data = (ROOT / name).read_bytes()
        entries.append({"path": name, "bytes": len(data), "sha256": digest(data)})
    return {"schema": 1, "archive": ARCHIVE.relative_to(ROOT).as_posix(),
            "scope": "Retired validation source/configuration/Slurm/tests; no results or models",
            "files": entries,
            "snapshots": [{"path": name, "bytes": (ROOT / name).stat().st_size,
                           "sha256": digest((ROOT / name).read_bytes())}
                          for name in SNAPSHOTS if (ROOT / name).is_file()],
            "retained_counts": dict(Counter(Path(n).parts[0] for n in keep)),
            "archived_counts": dict(Counter(Path(n).parts[0] for n in archive)),
            "retained_historical_dependencies": sorted(candidates & keep)}


def checked_path(name: str) -> Path:
    path = ROOT / name
    if (Path(name).is_absolute() or ".." in Path(name).parts or
            Path(name).parts[0] not in DIRECTORIES or
            path.suffix not in SUFFIXES or path.is_symlink() or
            not path.resolve().is_relative_to(ROOT)):
        raise ValueError(f"unsafe archive member: {name}")
    return path


def snapshot_path(name: str) -> Path:
    if name not in SNAPSHOTS:
        raise ValueError(f"unrecognized document snapshot: {name}")
    path = ROOT / name
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"unsafe snapshot: {name}")
    return path


def apply_plan(plan: dict) -> None:
    if ARCHIVE.exists() or MANIFEST.exists():
        raise FileExistsError("archive/manifest already exists; never overwrite research history")
    for entry in plan["files"]:
        if digest(checked_path(entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError(f"source changed: {entry['path']}")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for entry in plan["files"]:
            handle.write(checked_path(entry["path"]), entry["path"])
        for entry in plan["snapshots"]:
            handle.write(snapshot_path(entry["path"]), entry["path"])
    # Read and hash EVERY archived member before removing ANY working file.
    with zipfile.ZipFile(ARCHIVE) as handle:
        for entry in plan["files"] + plan["snapshots"]:
            if digest(handle.read(entry["path"])) != entry["sha256"]:
                raise ValueError(f"archive verification failed: {entry['path']}")
    plan["archive_sha256"] = digest(ARCHIVE.read_bytes())
    # Generated inventory, not a handwritten research document.
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    for entry in plan["files"]:
        path = checked_path(entry["path"])
        if digest(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"source changed during archival; backup preserved: {path}")
        path.unlink()


def verify_or_restore(restore: bool) -> None:
    plan = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if digest(ARCHIVE.read_bytes()) != plan["archive_sha256"]:
        raise ValueError("archive checksum mismatch")
    with zipfile.ZipFile(ARCHIVE) as handle:
        for entry in plan["snapshots"]:
            snapshot_path(entry["path"])
            if digest(handle.read(entry["path"])) != entry["sha256"]:
                raise ValueError(f"snapshot checksum mismatch: {entry['path']}")
        for entry in plan["files"]:
            path = checked_path(entry["path"])
            if digest(handle.read(entry["path"])) != entry["sha256"]:
                raise ValueError(f"member checksum mismatch: {path}")
            if restore and path.exists() and digest(path.read_bytes()) != entry["sha256"]:
                raise FileExistsError(f"refusing to overwrite changed file: {path}")
        if restore:
            for entry in plan["files"]:
                path = checked_path(entry["path"])
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as target:
                        target.write(handle.read(entry["path"]))
    print(f"{'Restored' if restore else 'Verified'} {len(plan['files'])} files")


def check_retained_references() -> None:
    plan = json.loads(MANIFEST.read_text(encoding="utf-8"))
    archived = {entry["path"] for entry in plan["files"]}
    files = inventory()
    universe = dict.fromkeys(archived, "") | files
    sources = dict(files)
    for directory in ("FMT_Utils", "FLowUtils", "DeepUtils", "pnn", "tools"):
        for path in (ROOT / directory).rglob("*.py"):
            if path.resolve() != Path(__file__).resolve():
                sources[path.relative_to(ROOT).as_posix()] = path.read_text(encoding="utf-8-sig")
    issues = {}
    for name, body in sources.items():
        missing = (references(body, universe) & archived) - set(files)
        if missing:
            issues[name] = sorted(missing)
        if name.endswith(".py"):
            compile(body, name, "exec")
    if issues:
        raise ValueError(f"references require archived files: {issues}")
    print(f"Checked {len(sources)} retained source/configuration files: no references to removed files; Python syntax valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "apply", "verify", "restore", "check"))
    parser.add_argument("--list", action="store_true", help="print the full per-file inventory")
    args = parser.parse_args()
    if args.action == "check":
        check_retained_references()
        return
    if args.action in {"verify", "restore"}:
        verify_or_restore(args.action == "restore")
        return
    plan = make_plan()
    print(json.dumps(plan if args.list else {
        key: value for key, value in plan.items() if key != "files"
    }, indent=2))
    if args.action == "apply":
        apply_plan(plan)


if __name__ == "__main__":
    main()
