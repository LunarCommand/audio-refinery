"""Drift check: the committed docs/schemas/*.json files must match what
``scripts/generate_schemas.py`` would produce against the current Pydantic
models. CI runs this; if it fails, run ``make generate-schemas`` to bring
the committed artifacts back in sync.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_schemas import SCHEMA_TARGETS, SCHEMAS_DIR, _dump  # noqa: E402


@pytest.mark.parametrize(("filename", "model"), SCHEMA_TARGETS)
def test_committed_schema_matches_model(filename: str, model: type[BaseModel]) -> None:
    expected = _dump(model.model_json_schema(mode="serialization"))
    committed = (SCHEMAS_DIR / filename).read_text()
    assert committed == expected, (
        f"docs/schemas/{filename} is out of sync with the Pydantic model. Run `make generate-schemas` to regenerate."
    )
