"""Guard the retired Task1--Task3 paper-table builder.

This historical entry point combined Task2 2.3/2.4 and Task3 3.1 results.
Those are no longer the current paper evidence. Keeping the old generator
active could silently overwrite the audited table with superseded numbers.

The canonical paper table is ``docs/paper_tables_tasks_3d.md``. Its evidence
boundary and current Task3 confirmation results are documented in
``docs/Audit_experiments.md`` and ``docs/mainExp_Task3_3D.md``.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DOC = ROOT / "docs" / "paper_tables_tasks_3d.md"


def build() -> None:
    """Refuse to regenerate the canonical table from superseded evidence."""

    raise RuntimeError(
        "Build_Paper_Tables_3D is retired because its frozen inputs are no "
        "longer the current paper evidence. Do not overwrite "
        f"{CANONICAL_DOC}. Update the audited evidence pipeline under a new "
        "experiment version instead."
    )


if __name__ == "__main__":
    build()
