from __future__ import annotations

import unittest

import numpy as np

from groot.vla.utils.mobilemanibench_receding_horizon import (
    RealObservationBuffer,
    TimedMobileObservation,
    executable_waypoint_indices,
)


class RecedingHorizonBufferTest(unittest.TestCase):
    def test_rebuild_uses_only_real_history_at_five_hz(self) -> None:
        buffer = RealObservationBuffer()
        for tick in range(145):
            value = np.asarray([tick], dtype=np.float32)
            buffer.append(
                TimedMobileObservation(
                    timestamp=tick / 30.0,
                    head_rgb=value,
                    wrist_rgb=value,
                    eef_position=value,
                    eef_rotation_rpy=value,
                )
            )
        sequence = buffer.rebuild_sequence()
        self.assertEqual([len(block) for block in sequence], [1, 9, 9, 9])
        self.assertAlmostEqual(sequence[-1][-1].timestamp, 144 / 30.0)
        self.assertEqual(executable_waypoint_indices([1, 2, 4, 8, 16, 32, 48], 16), (0, 1, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
