"""Deterministic task-balanced sampling for MobileManiBench evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _balanced_quotas(
    capacities: Mapping[str, int],
    sample_count: int,
) -> dict[str, int]:
    """Allocate samples round-robin, redistributing quota from small tasks."""
    quotas = {task: 0 for task in capacities}
    remaining = min(sample_count, sum(capacities.values()))
    active_tasks = [
        task for task, capacity in capacities.items() if capacity > 0
    ]

    while remaining > 0:
        next_active_tasks: list[str] = []
        for task in active_tasks:
            if remaining == 0:
                break
            quotas[task] += 1
            remaining -= 1
            if quotas[task] < capacities[task]:
                next_active_tasks.append(task)
        active_tasks = next_active_tasks
        if not active_tasks and remaining:
            raise RuntimeError("Unable to allocate the requested sample quota")

    return quotas


def _evenly_spaced(values: Sequence[int], count: int) -> list[int]:
    """Select ``count`` ordered values while spanning the full sequence."""
    if count <= 0:
        return []
    if count >= len(values):
        return list(values)
    if count == 1:
        return [values[(len(values) - 1) // 2]]

    final_index = len(values) - 1
    return [
        values[(sample_index * final_index) // (count - 1)]
        for sample_index in range(count)
    ]


def select_task_balanced_indices(
    all_steps: Sequence[tuple[int, int]],
    episode_ids: set[int],
    episode_tasks: Mapping[int, str],
    *,
    stride: int,
    max_samples: int,
) -> list[int]:
    """Select split anchors with equal task quotas when a cap is requested.

    ``stride`` is applied before balancing. Within each task, selected anchors
    are evenly spaced across the task's ordered candidate list, which spreads
    samples across its episodes and temporal extent.
    """
    if stride < 1:
        raise ValueError("--sample-stride must be >= 1")

    candidate_indices = [
        index
        for index, (episode_id, _) in enumerate(all_steps)
        if int(episode_id) in episode_ids
    ]
    candidate_indices = candidate_indices[::stride]
    if not candidate_indices:
        raise ValueError(
            "No dataset anchors remain after split/stride/max-samples filtering"
        )
    if max_samples <= 0 or max_samples >= len(candidate_indices):
        return candidate_indices

    indices_by_task: dict[str, list[int]] = {}
    for index in candidate_indices:
        episode_id = int(all_steps[index][0])
        try:
            task = episode_tasks[episode_id]
        except KeyError as exc:
            raise ValueError(
                f"Episode {episode_id} has no task in meta/episodes.jsonl"
            ) from exc
        indices_by_task.setdefault(task, []).append(index)

    quotas = _balanced_quotas(
        {task: len(indices) for task, indices in indices_by_task.items()},
        max_samples,
    )
    selected = [
        index
        for task, task_indices in indices_by_task.items()
        for index in _evenly_spaced(task_indices, quotas[task])
    ]
    return sorted(selected)


def count_tasks_for_indices(
    all_steps: Sequence[tuple[int, int]],
    indices: Sequence[int],
    episode_tasks: Mapping[int, str],
) -> dict[str, int]:
    """Count selected anchors by task in deterministic task order."""
    counts: dict[str, int] = {}
    for index in indices:
        episode_id = int(all_steps[index][0])
        task = episode_tasks[episode_id]
        counts[task] = counts.get(task, 0) + 1
    return counts
