from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "train"
    / "calibrate_mobile_plan_loss_weights.py"
)
SPEC = importlib.util.spec_from_file_location("mobile_plan_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MobilePlanLossCalibrationTest(unittest.TestCase):
    def test_collect_and_robust_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainer_state.json"
            path.write_text(
                json.dumps(
                    {
                        "log_history": [
                            {
                                MODULE.PLAN_COMPONENT_KEY: value,
                                MODULE.BASE_EEF_CONSISTENCY_KEY: value * 0.5,
                                MODULE.PRIOR_KEY: value * 0.25,
                                MODULE.EEF_PRIOR_KEY: value * 0.125,
                                MODULE.JOINT_PRIOR_KEY: value * 0.0625,
                            }
                            for value in (0.1, 0.2, 0.3)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            values = MODULE.collect([path])
        self.assertEqual(
            values[MODULE.PLAN_COMPONENT_KEY], [0.1, 0.2, 0.3]
        )
        self.assertAlmostEqual(
            MODULE.robust_weight(values[MODULE.PLAN_COMPONENT_KEY], 0, 1),
            0.2,
        )
        self.assertAlmostEqual(
            MODULE.robust_weight(
                values[MODULE.BASE_EEF_CONSISTENCY_KEY], 0, 1
            ),
            0.1,
        )
        self.assertAlmostEqual(
            MODULE.robust_weight(values[MODULE.PRIOR_KEY], 0, 1),
            0.05,
        )
        self.assertAlmostEqual(
            MODULE.robust_weight(values[MODULE.EEF_PRIOR_KEY], 0, 1),
            0.025,
        )
        self.assertAlmostEqual(
            MODULE.robust_weight(values[MODULE.JOINT_PRIOR_KEY], 0, 1),
            0.0125,
        )


if __name__ == "__main__":
    unittest.main()
