"""Validate the repository layout and canonical Markdown documents.

This check is deliberately read-only.  It catches the structural regressions that caused
the 2026-09-02 cleanup: split Task3/Verify/Audit documents, machine results under docs,
root-level experiment scripts, broken local Markdown links, and malformed Markdown tables.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REQUIRED_DOCS = (
    "experiment_log.md",
    "ibex_run_registry.md",
    "mainExp_Task3_3D.md",
    "Verify_experiments.md",
    "Audit_experiments.md",
    "paper_tables_tasks_3d.md",
    "research_tasks_and_protocol.md",
)
ALLOWED_ROOT_MARKDOWN = {"AGENTS.md", "README.md"}
EXPERIMENT_PREFIXES = (
    "Audit_",
    "Build_",
    "Confirm_",
    "Diagnose_",
    "Evaluate_",
    "Freeze_",
    "Merge_",
    "Prepare_",
    "Preflight_",
    "Repair_",
    "Replay_",
    "Run_",
    "Screen_",
    "Search_",
    "Select_",
    "Summarize_",
    "Sweep_",
    "Train_",
    "Verify_",
    "Visualize_",
)
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#]+)(?:#[^)]+)?\)")
SEPARATOR_RE = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+\s*$")
DIRECT_SCRIPT_RE = re.compile(
    r"\bpython(?:3)?\s+(?:-u\s+)?(?:\./)?(?:experiments[/\\])?[A-Za-z0-9_]+\.py\b"
)


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8"
    )
    return [ROOT / line for line in output.splitlines() if line]


def unescaped_pipe_count(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line))


def check_markdown_table(path: Path) -> list[str]:
    issues: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_table = False
    expected_pipes = 0
    for number, line in enumerate(lines, start=1):
        if line.startswith("|"):
            pipes = unescaped_pipe_count(line)
            if SEPARATOR_RE.fullmatch(line):
                expected_pipes = pipes
                in_table = True
            elif in_table and expected_pipes and pipes != expected_pipes:
                issues.append(
                    f"{path.relative_to(ROOT)}:{number}: table has {pipes - 1} cells; "
                    f"expected {expected_pipes - 1}"
                )
        elif in_table and line.strip():
            in_table = False
            expected_pipes = 0
    return issues


def check_local_links(path: Path) -> list[str]:
    issues: list[str] = []
    text = path.read_text(encoding="utf-8")
    for match in LINK_RE.finditer(text):
        target = match.group(1).strip("<>")
        if re.match(r"^(?:https?:|mailto:|#|/)", target):
            continue
        if not (path.parent / target).exists():
            issues.append(f"{path.relative_to(ROOT)}: missing link target {target}")
    return issues


def main() -> int:
    issues: list[str] = []
    tracked = tracked_files()

    for name in REQUIRED_DOCS:
        if not (DOCS / name).is_file():
            issues.append(f"missing canonical document: docs/{name}")

    if (DOCS / "results").exists():
        issues.append("docs/results must not exist; machine results belong in outputs/<id>/")
    if (DOCS / "paper_tables_task123_3d.md").exists():
        issues.append("legacy paper_tables_task123_3d.md still exists")

    split_patterns = (
        "mainExp_Task3_3D_[0-9]*.md",
        "Verify_*.md",
        "Audit_*.md",
    )
    allowed = {DOCS / "Verify_experiments.md", DOCS / "Audit_experiments.md"}
    for pattern in split_patterns:
        for path in DOCS.glob(pattern):
            if path not in allowed:
                issues.append(f"split document remains: {path.relative_to(ROOT)}")

    for path in ROOT.glob("*.py"):
        if path.name.startswith(EXPERIMENT_PREFIXES) or path.name in {
            "FMT_Clustering.py",
            "FMT_Clustering_3D.py",
            "FittingVatistasParam.py",
            "VatistasFlowDatasetGenerator.py",
        }:
            issues.append(f"experiment script remains at repository root: {path.name}")

    for path in tracked:
        if path.suffix.lower() == ".md" and DOCS not in path.parents:
            if path.parent != ROOT or path.name not in ALLOWED_ROOT_MARKDOWN:
                issues.append(f"tracked research Markdown outside docs/: {path.relative_to(ROOT)}")

    for path in DOCS.rglob("*.md"):
        issues.extend(check_markdown_table(path))
        issues.extend(check_local_links(path))

    for path in (ROOT / "ibex_bash").glob("*.sh"):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if DIRECT_SCRIPT_RE.search(line):
                issues.append(
                    f"{path.relative_to(ROOT)}:{number}: use python -m experiments.<module>"
                )

    if issues:
        print("Repository layout validation: FAIL")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Repository layout validation: PASS")
    print(f"- canonical documents: {len(REQUIRED_DOCS)}")
    print(f"- Markdown files checked: {len(list(DOCS.rglob('*.md')))}")
    print(f"- tracked files checked: {len(tracked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
