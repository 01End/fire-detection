"""Temporal debouncing of detections to suppress false alarms.

A detection is only *confirmed* once it persists across at least ``confirm_n`` of the
last ``window_m`` processed frames for a given key (a ``(floor, label)`` pair). After a
confirmation, a per-key cooldown prevents the same ongoing fire from emitting a flood of
events.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Hashable, List, Set


class _KeyState:
    __slots__ = ("window", "cooldown_until")

    def __init__(self, window_m: int):
        self.window: Deque[int] = deque(maxlen=window_m)
        self.cooldown_until: float = float("-inf")


class FloorDebouncer:
    def __init__(
        self,
        confirm_n: int = 3,
        window_m: int = 5,
        cooldown_seconds: float = 30.0,
    ):
        if confirm_n > window_m:
            raise ValueError("confirm_n cannot exceed window_m")
        self.confirm_n = confirm_n
        self.window_m = window_m
        self.cooldown_seconds = cooldown_seconds
        self._states: Dict[Hashable, _KeyState] = {}

    def update(self, active_keys: Set[Hashable], timestamp: float) -> List[Hashable]:
        """Advance one frame.

        ``active_keys`` is the set of keys with a qualifying detection in this frame.
        Returns the keys that became newly confirmed (i.e. should emit an event now).
        """
        # Ensure every active key has state, then record presence/absence for all
        # currently-tracked keys so the sliding window reflects this frame.
        for key in active_keys:
            self._states.setdefault(key, _KeyState(self.window_m))

        confirmed: List[Hashable] = []
        for key, state in list(self._states.items()):
            state.window.append(1 if key in active_keys else 0)

            if sum(state.window) >= self.confirm_n and timestamp >= state.cooldown_until:
                confirmed.append(key)
                state.cooldown_until = timestamp + self.cooldown_seconds

            # Prune keys that have gone fully quiet and aren't cooling down.
            if sum(state.window) == 0 and timestamp >= state.cooldown_until:
                del self._states[key]

        return confirmed
