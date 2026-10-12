"""Read-only regressions against local KBRR residual fixtures.

Run from the repository root:
python -m unittest discover -s test -p test_kbrr_cluster_regression.py
Missing monthly fixtures are skipped. Neither residuals nor QC files change.
"""
import json
import pathlib
import unittest
import tempfile

import h5py
import numpy as np

from src.Preference.Pre_AdjustOrbit import AdjustOrbitConfig
from src.Solver.CalibrateOrbit import CalibrateOrbit


class KBRRClusterRegression(unittest.TestCase):
    def replay(self, month):
        root = pathlib.Path(__file__).resolve().parents[1]
        folder = root / 'temp' / 'RangeRate' / month
        paths = sorted(folder.glob('RangeRate_*.hdf5'))
        if not paths:
            self.skipTest('Local residual fixtures unavailable: ' + month)
        watched = paths + list(folder.glob('KBRR_ArcQuality.json'))
        before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in watched}
        config = AdjustOrbitConfig()
        with (root / 'setting' / 'Calibrate' / 'AdjustOrbitConfig.json').open() as f:
            config.__dict__.update(json.load(f))
        config.PathOfFilesConfig['RangeRateTemp'] = str(folder.parent)
        calibration = CalibrateOrbit()
        calibration._CalibrateOrbit__adjust_config = config
        calibration._CalibrateOrbit__date_span = month.split('_')
        calibration._CalibrateOrbit__writeKBRRArcQuality = lambda **kw: None
        arcs = sorted(int(p.stem.split('_')[-1]) for p in paths)
        report = folder / 'KBRR_ArcQuality.json'
        if report.exists():
            with report.open() as f:
                arcs = sorted(r['arc'] for r in json.load(f)['records'])
        _, rejected = calibration._CalibrateOrbit__screenKBRRArcs(arcs)
        after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in watched}
        self.assertEqual(before, after)
        return rejected, calibration._CalibrateOrbit__kbrr_qc_records

    def test_alternating_noise_does_not_reject_ordinary_pairs(self):
        rejected, records = self.replay('2013-05-01_2013-05-31')
        self.assertEqual(rejected, [1])
        for record in records:
            if record['arc'] in (16, 17, 18, 54, 55):
                self.assertTrue(record['accepted'])
                self.assertFalse(record['cluster_upper_candidate'])

    def test_updated_noise_floor_does_not_reject_early_arcs(self):
        rejected, records = self.replay('2016-08-08_2016-09-03')
        self.assertEqual(rejected, [73])
        self.assertFalse(set(r['arc'] for r in records) & {72, 90, 102})
        for record in records:
            if record['arc'] in (7, 8, 9, 11, 15, 16):
                self.assertTrue(record['accepted'])
                self.assertGreater(record['epoch_outlier_count'], 0)
                self.assertEqual(record['qc_classification'],
                                 'retained_with_epoch_qc')

    def test_low_residual_short_arc_screen_is_preserved(self):
        rejected, records = self.replay('2020-03-01_2020-03-31')
        self.assertEqual(rejected, [112, 120, 122])
        record = next(r for r in records if r['arc'] == 122)
        self.assertEqual(record['reason'], 'short_arc_lower_esd')

    def test_sparse_spikes_are_removed_but_dense_failures_are_not_clipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = pathlib.Path(temporary) / 'test_start_test_end'
            folder.mkdir()
            rng = np.random.default_rng(31)
            for arc in range(40):
                residual = rng.normal(
                    0.0, 0.2e-6 * np.exp(0.2 * np.sin(2 * np.pi * arc / 7)),
                    200
                )
                if arc == 10:
                    residual[[60, 140]] = [20e-6, -20e-6]
                if arc == 30:
                    residual[20:80:4] = 8e-6
                with h5py.File(folder / ('RangeRate_%d.hdf5' % arc), 'w') as h:
                    h['post_residual'] = residual
                    h['post_t'] = np.arange(200) * 5.0
                    h['pre_t'] = np.arange(200) * 5.0
            config = AdjustOrbitConfig()
            config.PathOfFilesConfig['RangeRateTemp'] = temporary
            calibration = CalibrateOrbit()
            calibration._CalibrateOrbit__adjust_config = config
            calibration._CalibrateOrbit__date_span = ['test_start', 'test_end']
            captured = {}
            calibration._CalibrateOrbit__writeKBRRArcQuality = (
                lambda **kw: captured.update(kw)
            )
            _, rejected = calibration._CalibrateOrbit__screenKBRRArcs(range(40))
            self.assertEqual(rejected, [30])
            records = calibration._CalibrateOrbit__kbrr_qc_records
            sparse, dense = records[10], records[30]
            self.assertTrue(sparse['pre_epoch_upper_candidate'])
            self.assertTrue(sparse['accepted'])
            self.assertEqual(sparse['epoch_outlier_count'], 2)
            self.assertFalse(dense['accepted'])
            self.assertEqual(dense['epoch_outlier_count'], 0)
            self.assertEqual(dense['qc_classification'], 'persistent_arc_anomaly')
            # Exercise real HDF5/JSON persistence only on temporary fixtures.
            CalibrateOrbit._CalibrateOrbit__writeKBRRArcQuality(
                calibration, **captured
            )
            with h5py.File(folder / 'RangeRate_10.hdf5', 'r') as h:
                self.assertEqual(np.count_nonzero(~h['arc_qc_epoch_keep'][:]), 2)
                self.assertTrue(h.attrs['arc_qc_accepted'])


if __name__ == '__main__':
    unittest.main()
