import unittest
import numpy as np
from flash.analysis import analyze
from flash.detection import Detector


class Algorithms(unittest.TestCase):
    def test_reverse_rank_and_stage1(self):
        specs = [
            dict(tag="expert_dp", ranks=[0, 8]),
            dict(tag="expert_dp", ranks=[1, 9]),
        ]
        stats = [
            dict(hash=h, rank=r, size=2, mean_us=v, count=100)
            for h, rs, vs in [("a", [0, 8], [100, 10]), ("b", [1, 9], [20, 20])]
            for r, v in zip(rs, vs)
        ]
        got = analyze(stats, specs)
        self.assertEqual(got["suspects"][0]["rank"], 8)

    def test_low_cv_and_missing_rank_reject(self):
        specs = [dict(tag="dp", ranks=[0, 1])]
        stats = [
            dict(hash="a", rank=r, size=2, mean_us=v, count=10)
            for r, v in [(0, 100), (1, 99)]
        ]
        self.assertFalse(analyze(stats, specs)["suspects"])
        self.assertFalse(analyze(stats[:1], specs)["suspects"])

    def test_detector_sustained_vs_recovery_and_low_severity(self):
        rng = np.random.default_rng(2)

        def detect(v):
            return Detector().update([(i + 4, x) for i, x in enumerate(v)])

        base = rng.normal(100, 0.3, 20)
        self.assertTrue(detect(np.r_[base, rng.normal(150, 0.3, 10)]))
        self.assertFalse(detect(np.r_[base, [200], rng.normal(100, 0.3, 9)]))
        self.assertFalse(detect(np.r_[base, rng.normal(105, 0.3, 10)]))


if __name__ == "__main__":
    unittest.main()
