"""Input normalization for the pipeline entry point.

The standard ``reviewed_full_*.json`` input is a mapping of sentence IDs to
objects containing ``previous_sentence``, ``evaluative_sentence``, and
``next_sentence``.  The pipeline receives one context-enriched string per ID.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _sentence_id_key(sentence_id: object) -> tuple[int, int, str]:
    """Sort numeric IDs naturally while retaining a deterministic fallback."""
    value = str(sentence_id)
    if value.isdigit():
        return (0, int(value), value)
    match = re.search(r"(\d+)$", value)
    if match:
        return (1, int(match.group(1)), value)
    return (2, 0, value)


def load_pipeline_sentences(
    input_path: str | Path,
    *,
    entity_source: str = "",
    additional_source: str = "",
) -> list[tuple[str, str]]:
    """Load standard reviewed input into ``(sentence_id, text)`` tuples.

    ``entity_source`` and ``additional_source`` are accepted by the CLI's
    current interface.  They are intentionally not merged here: the pipeline
    has no consumption contract for those external schemas yet, so ignoring
    them is safer than silently mixing unrelated sentence records.
    """
    del entity_source, additional_source
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object keyed by sentence ID: {path}")

    sentences: list[tuple[str, str]] = []
    for sentence_id in sorted(raw, key=_sentence_id_key):
        entry = raw[sentence_id]
        if not isinstance(entry, dict):
            continue
        parts = (
            entry.get("previous_sentence", ""),
            entry.get("evaluative_sentence", ""),
            entry.get("next_sentence", ""),
        )
        text = " ".join(str(part).strip() for part in parts if part).strip()
        if text:
            sentences.append((str(sentence_id), text))
    return sentences
