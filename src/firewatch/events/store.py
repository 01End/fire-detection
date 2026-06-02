"""Shared event store (JSONL) read/written by sinks and the dashboard.

JSONL was chosen over SQLite for the first build: events are append-only, low-volume,
and trivially tailed by the Streamlit dashboard without a DB dependency.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List


class EventStore:
    def __init__(self, root: str):
        self.root = root
        self.snapshots_dir = os.path.join(root, "snapshots")
        self.events_path = os.path.join(root, "events.jsonl")
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def append(self, event_dict: Dict[str, Any]) -> None:
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event_dict) + "\n")

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not os.path.exists(self.events_path):
            return []
        with open(self.events_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        out = []
        for line in lines[-limit:]:
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out
