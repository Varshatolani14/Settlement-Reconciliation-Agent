"""
Append-only audit log. Every reconciliation decision is recorded here —
matched-by-rule, matched-by-LLM (with rationale), or unresolved (with reason) —
so the whole run is auditable after the fact. Persisted to ``audit_log.json``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_PATH = Path("audit_log.json")


class AuditLog:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, event: str, ref: str, actor: str, method: str, detail: str, status: str) -> None:
        self.records.append(
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "event": event,
                "ref": ref,
                "actor": actor,
                "method": method,
                "detail": detail,
                "status": status,
            }
        )

    def extend(self, rows: list[dict]) -> None:
        self.records.extend(rows)

    def to_json(self) -> str:
        return json.dumps(self.records, indent=2)

    def save(self, path: Path | str = AUDIT_PATH) -> Path:
        p = Path(path)
        p.write_text(self.to_json(), encoding="utf-8")
        return p
