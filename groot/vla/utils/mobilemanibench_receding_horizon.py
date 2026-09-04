"""Real-observation history utilities for 1.6 s MobileManiBench WAM control."""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimedMobileObservation:
    timestamp: float
    head_rgb: np.ndarray
    wrist_rgb: np.ndarray
    eef_position: np.ndarray
    eef_rotation_rpy: np.ndarray


class RealObservationBuffer:
    """Bounded monotonic buffer with nearest-timestamp 5 Hz sampling."""

    def __init__(
        self,
        *,
        video_fps: float = 5.0,
        block_duration_seconds: float = 1.6,
        max_history_blocks: int = 3,
    ) -> None:
        if video_fps <= 0 or block_duration_seconds <= 0:
            raise ValueError("Video frequency and block duration must be positive")
        self.video_fps = float(video_fps)
        self.block_duration_seconds = float(block_duration_seconds)
        self.max_history_blocks = int(max_history_blocks)
        self.max_time_error = 0.5 / self.video_fps
        capacity = int(
            np.ceil(
                (self.max_history_blocks * self.block_duration_seconds + 1.0)
                * 30.0
            )
        )
        self._items: deque[TimedMobileObservation] = deque(maxlen=capacity)

    def clear(self) -> None:
        self._items.clear()

    @property
    def latest_timestamp(self) -> float | None:
        return self._items[-1].timestamp if self._items else None

    def append(self, observation: TimedMobileObservation) -> None:
        if self._items and observation.timestamp <= self._items[-1].timestamp:
            raise ValueError("Observation timestamps must be strictly increasing")
        self._items.append(observation)

    def _nearest(self, timestamp: float) -> TimedMobileObservation:
        if not self._items:
            raise ValueError("Real observation buffer is empty")
        times = [item.timestamp for item in self._items]
        index = bisect_left(times, timestamp)
        candidates = []
        if index < len(times):
            candidates.append(self._items[index])
        if index:
            candidates.append(self._items[index - 1])
        result = min(candidates, key=lambda item: abs(item.timestamp - timestamp))
        if abs(result.timestamp - timestamp) > self.max_time_error + 1e-9:
            raise ValueError(
                f"No real observation within {self.max_time_error:.3f}s of "
                f"timestamp {timestamp:.6f}"
            )
        return result

    def rebuild_sequence(self) -> list[list[TimedMobileObservation]]:
        """Return one initial frame followed by up to three complete 9-frame blocks."""
        if not self._items:
            raise ValueError("Real observation buffer is empty")
        now = self._items[-1].timestamp
        available = now - self._items[0].timestamp
        history_blocks = min(
            self.max_history_blocks,
            int(np.floor((available + self.max_time_error) / self.block_duration_seconds)),
        )
        oldest = now - history_blocks * self.block_duration_seconds
        sequence: list[list[TimedMobileObservation]] = [[self._nearest(oldest)]]
        samples_per_block = int(round(self.block_duration_seconds * self.video_fps))
        for block_index in range(history_blocks):
            start = oldest + block_index * self.block_duration_seconds
            timestamps = start + np.arange(samples_per_block + 1) / self.video_fps
            sequence.append([self._nearest(float(value)) for value in timestamps])
        return sequence


def reset_wam_cache(action_head: Any) -> None:
    """Clear all persistent inference cache state before a real-history rebuild."""
    action_head.current_start_frame = 0
    for name in (
        "kv_cache1",
        "kv_cache_neg",
        "crossattn_cache",
        "crossattn_cache_neg",
    ):
        if hasattr(action_head, name):
            setattr(action_head, name, None)


def executable_waypoint_indices(
    offsets: list[int] | tuple[int, ...], execution_horizon_ticks: int
) -> tuple[int, ...]:
    return tuple(
        index
        for index, offset in enumerate(offsets)
        if int(offset) <= int(execution_horizon_ticks)
    )
