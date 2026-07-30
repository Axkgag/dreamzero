from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts/eval/mobilemanibench_sampling.py"
SPEC = importlib.util.spec_from_file_location(
    "mobilemanibench_sampling",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
SAMPLING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAMPLING)


class MobileManiBenchSamplingTest(unittest.TestCase):
    def test_max_samples_are_balanced_across_five_tasks(self) -> None:
        all_steps: list[tuple[int, int]] = []
        episode_tasks: dict[int, str] = {}
        for task_index in range(5):
            episode_id = 100 + task_index
            episode_tasks[episode_id] = f"task-{task_index}"
            all_steps.extend(
                (episode_id, frame_index) for frame_index in range(300)
            )

        indices = SAMPLING.select_task_balanced_indices(
            all_steps,
            set(episode_tasks),
            episode_tasks,
            stride=1,
            max_samples=1024,
        )
        counts = SAMPLING.count_tasks_for_indices(
            all_steps,
            indices,
            episode_tasks,
        )

        self.assertEqual(len(indices), 1024)
        self.assertEqual(
            list(counts.values()),
            [205, 205, 205, 205, 204],
        )
        for task_index, count in enumerate(counts.values()):
            task_indices = indices[
                sum(list(counts.values())[:task_index]) :
                sum(list(counts.values())[:task_index]) + count
            ]
            self.assertEqual(
                [all_steps[index][1] for index in task_indices][0],
                0,
            )
            self.assertEqual(
                [all_steps[index][1] for index in task_indices][-1],
                299,
            )

    def test_small_task_quota_is_redistributed(self) -> None:
        all_steps = (
            [(1, 0)]
            + [(2, frame_index) for frame_index in range(5)]
            + [(3, frame_index) for frame_index in range(5)]
        )
        episode_tasks = {1: "small", 2: "medium-a", 3: "medium-b"}

        indices = SAMPLING.select_task_balanced_indices(
            all_steps,
            set(episode_tasks),
            episode_tasks,
            stride=1,
            max_samples=8,
        )
        counts = SAMPLING.count_tasks_for_indices(
            all_steps,
            indices,
            episode_tasks,
        )

        self.assertEqual(
            counts,
            {"small": 1, "medium-a": 4, "medium-b": 3},
        )

    def test_unlimited_sampling_preserves_stride_behavior(self) -> None:
        all_steps = [(1, frame_index) for frame_index in range(10)]

        indices = SAMPLING.select_task_balanced_indices(
            all_steps,
            {1},
            {1: "task"},
            stride=3,
            max_samples=0,
        )

        self.assertEqual(indices, [0, 3, 6, 9])


if __name__ == "__main__":
    unittest.main()
