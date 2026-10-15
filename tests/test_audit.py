import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from flash.analysis import analyze
from flash.detection import Detector
from flash.profile import complete_nodes, process_state
from flash.local_controller import aggregate


class AuditRegressions(unittest.TestCase):
    def test_adjacent_points_deduplicated_within_one_update(self):
        rb = SimpleNamespace(
            beast=lambda *a, **kw: SimpleNamespace(
                trend=SimpleNamespace(cp=[11, 12], cpPr=[1, 1])
            )
        )
        values = [100.0] * 10 + [150.0] * 20
        with patch.dict("sys.modules", {"Rbeast": rb}):
            d = Detector()
            got = d.update([(i + 4, x) for i, x in enumerate(values)])
        self.assertEqual([r["iteration"] for r in got], [14])

    def test_all_nodes_must_finish_same_epoch(self):
        a = dict(node="0", epoch=3, complete=True)
        b = dict(node="1", epoch=3, complete=False)
        self.assertFalse(complete_nodes([a, b], 3)[1])
        self.assertFalse(complete_nodes([a, dict(b, epoch=2, complete=True)], 3)[1])
        self.assertTrue(complete_nodes([a, dict(b, complete=True)], 3)[1])

    def test_multiple_handles_per_rank_are_aggregated(self):
        # The last handle would make rank 1 look fast. All records make rank 0 fast.
        stats = [
            dict(hash="x", rank=0, size=2, count=9, mean_us=1),
            dict(hash="x", rank=0, size=2, count=1, mean_us=100),
            dict(hash="x", rank=1, size=2, count=10, mean_us=40),
        ]
        got = analyze(stats, [dict(tag="dp", ranks=[0, 1])])
        self.assertEqual(got["suspects"][0]["rank"], 0)

    def test_unmatched_group_has_special_tag_but_ep_is_excluded(self):
        stats = [
            dict(hash="x", rank=r, size=2, count=10, mean_us=v)
            for r, v in [(0, 100), (1, 1)]
        ]
        self.assertEqual(analyze(stats, [])["suspects"][0]["rank"], 1)
        self.assertFalse(analyze(stats, [dict(tag="ep", ranks=[0, 1])])["suspects"])

    def test_liveness_is_evidence_only(self):
        self.assertTrue(process_state(os.getpid())["process_alive"])
        self.assertFalse(process_state(2147483647)["process_exists"])

    def test_handle_reuse_keeps_hash_generations_distinct(self):
        acc = {(0, 123, 456): [1, 3], (0, 123, 789): [100, 200]}
        metadata = {
            (0, 123, 456): dict(comm=123, hash="456", rank=0, local=0, size=2),
            (0, 123, 789): dict(comm=123, hash="789", rank=0, local=0, size=2),
        }
        got = {r["hash"]: r["mean_us"] for r in aggregate(acc, metadata)}
        self.assertEqual(got, {"456": 2, "789": 150})


if __name__ == "__main__":
    unittest.main()
