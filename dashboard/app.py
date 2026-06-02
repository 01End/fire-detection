"""FireWatch monitoring dashboard (Streamlit).

Reads the same event store the pipeline's EvidenceSink writes (``<output>/events.jsonl``
+ ``<output>/snapshots/``) and shows active floor alerts, recent detections, and the
latest annotated snapshot per camera.

Run with::

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from firewatch.events.store import EventStore  # noqa: E402

st.set_page_config(page_title="FireWatch", page_icon="🔥", layout="wide")
st.title("🔥 FireWatch — Fire Detection Monitor")

output_dir = st.sidebar.text_input("Output directory", "output")
active_window = st.sidebar.slider("Active-alert window (seconds)", 5, 300, 60)
refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 1, 30, 3)
limit = st.sidebar.number_input("Events to load", 10, 2000, 200, step=10)

store = EventStore(output_dir)
events = store.recent(int(limit))
now = time.time()

# --- Active alerts ----------------------------------------------------------
active = [e for e in events if now - e.get("timestamp", 0) <= active_window]
if active:
    floors = sorted(
        {str(e["floor"]) if e["floor"] is not None else "UNMAPPED" for e in active}
    )
    cams = sorted({e["camera_id"] for e in active})
    st.error(
        f"🚨 ACTIVE FIRE ALERT — floor(s) {', '.join(floors)} "
        f"(camera(s): {', '.join(cams)})"
    )
else:
    st.success("✅ No active alerts.")

# --- Latest snapshot per camera ---------------------------------------------
latest_by_cam = {}
for e in events:
    latest_by_cam[e["camera_id"]] = e  # events are append-order; last wins

if latest_by_cam:
    st.subheader("Latest detections by camera")
    cols = st.columns(min(len(latest_by_cam), 3))
    for i, (cam, e) in enumerate(latest_by_cam.items()):
        with cols[i % len(cols)]:
            floor = e["floor"] if e["floor"] is not None else "UNMAPPED"
            st.caption(f"{cam} — {e['label']} on floor {floor} ({e['confidence']:.2f})")
            snap = e.get("snapshot_path")
            if snap and os.path.exists(snap):
                st.image(snap, channels="BGR", use_container_width=True)
            else:
                st.info("no snapshot saved")

# --- Recent events table ----------------------------------------------------
st.subheader("Recent events")
if events:
    rows = [
        {
            "time": datetime.fromtimestamp(e["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "camera": e["camera_id"],
            "floor": e["floor"] if e["floor"] is not None else "UNMAPPED",
            "label": e["label"],
            "confidence": round(e["confidence"], 2),
        }
        for e in reversed(events)
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.write("No events recorded yet. Start the pipeline with `firewatch run`.")

# Auto-refresh by re-running after a short pause.
time.sleep(refresh_seconds)
st.rerun()
