#!/usr/bin/env python3
"""Regenerate the published JSON Schemas under docs/schemas/.

The Pydantic models in ``src/`` are the source of truth for audio-refinery's
output shapes; this script materializes them as JSON Schema artifacts so
downstream consumers (who don't have access to the Python package) can
validate transcript and batch-summary documents.

Run via ``make generate-schemas``. CI invokes ``make check-schemas`` which
runs this script then ``git diff --exit-code docs/schemas/`` — so any model
change that affects the published shape must be accompanied by a regen.

Note: Pydantic's JSON Schema output is deterministic within a Pydantic
version. A Pydantic upgrade may legitimately reshuffle the schema (key
order, ``$defs`` naming); the drift check will fail and force a deliberate
regen, which is the intended behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.service.schemas import BatchSummary, CombinedTranscript  # noqa: E402

SCHEMAS_DIR = REPO_ROOT / "docs" / "schemas"

SCHEMA_TARGETS: list[tuple[str, Any]] = [
    ("combined-transcript-v1.json", CombinedTranscript),
    ("batch-summary-v1.json", BatchSummary),
]


def _dump(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def generate() -> list[Path]:
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, model in SCHEMA_TARGETS:
        path = SCHEMAS_DIR / filename
        path.write_text(_dump(model.model_json_schema(mode="serialization")))
        written.append(path)
    return written


if __name__ == "__main__":
    for path in generate():
        print(f"wrote {path.relative_to(REPO_ROOT)}")
