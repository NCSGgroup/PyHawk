import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Solver.AdjustSST import RangeRate


class TestKBRRArcSegmentation(unittest.TestCase):

    @staticmethod
    def assert_all_epochs_covered_once(time, nodes):
        coverage = np.zeros(time.size, dtype=np.int64)
        for start, end in zip(nodes[:-1], nodes[1:]):
            coverage += (time >= start) & (time < end)
        np.testing.assert_array_equal(coverage, np.ones(time.size, dtype=np.int64))

    def test_arc_shorter_than_fitting_length_still_has_one_interval(self):
        time = np.arange(0.0, 5.322 * 3600.0 + 1.0, 5.0)

        nodes = RangeRate._build_arc_nodes(time, arc_length=6 * 3600)

        self.assertEqual(len(nodes), 2)
        self.assert_all_epochs_covered_once(time, nodes)

    def test_short_tail_is_merged_without_dropping_epochs(self):
        time = np.arange(0.0, 12.2 * 3600.0 + 1.0, 5.0)

        nodes = RangeRate._build_arc_nodes(time, arc_length=6 * 3600)

        self.assertEqual(len(nodes), 3)
        self.assertAlmostEqual(nodes[1], 6 * 3600)
        self.assert_all_epochs_covered_once(time, nodes)

    def test_long_tail_forms_a_separate_interval(self):
        time = np.arange(0.0, 12.75 * 3600.0 + 1.0, 5.0)

        nodes = RangeRate._build_arc_nodes(time, arc_length=6 * 3600)

        self.assertEqual(len(nodes), 4)
        self.assert_all_epochs_covered_once(time, nodes)


if __name__ == '__main__':
    unittest.main()
