# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.Auxilary.StandardACC import StandardACC
from src.Auxilary.StandardKBRR import StandardKBRR
from src.Auxilary.StandardLGD import StandardLGD
from src.Auxilary.StandardNEQ import StandardNEQ
from src.Solver.AdjustSST import RangeRate
from src.Solver.CalibrateOrbit import CalibrateOrbit


class TestStandardProducts(unittest.TestCase):
    _TIME = np.array([825595200.0, 825595205.0])
    _DATE_SPAN = ['2026-03-01', '2026-03-01']
    _GLOBAL_ATTRIBUTES = (
        'creator', 'creator_institution', 'date_created', 'instrument',
        'source', 'summary',
        'time_coverage_start', 'time_coverage_stop', 'title'
    )

    def assert_standard_header(self, content, record_number=2):
        self.assertIn('  dimensions:\n', content)
        self.assertIn('    num_records: %d\n' % record_number, content)
        self.assertIn('  global_attributes:\n', content)
        for attribute in self._GLOBAL_ATTRIBUTES:
            self.assertIn('    %s:' % attribute, content)
        self.assertIn('  variables:\n', content)

    def test_acc_header_contains_dimensions_and_global_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = StandardACC().configure(
                directory, self._DATE_SPAN, mission='GRACE_FO_RL04'
            ).prepare(satellites=['A'])
            writer.save_arc(
                gps_time=self._TIME,
                acceleration=np.zeros((2, 3)),
                bias=np.ones((2, 3)),
                satellite='A',
                arc_no=0
            )
            output = Path(writer.finalize()[0])
            self.assertEqual(
                output.parent, Path(directory) / '2026' / '2026-03'
            )
            self.assert_standard_header(output.read_text(encoding='ascii'))

    def test_kbrr_combines_raw_pre_post_and_geofit_in_one_daily_file(self):
        with tempfile.TemporaryDirectory() as directory:
            post_writer = StandardKBRR().configure(
                directory, self._DATE_SPAN, stage='Post',
                mission='GRACE_FO_RL04'
            ).prepare()
            coordinates = (
                np.array([10.0, 11.0]),
                np.array([20.0, 21.0]),
                np.array([500000.0, 500010.0])
            )
            for stage, rate in (
                    ('Raw', np.array([3.0e-6, 4.0e-6])),
                    ('Pre', np.array([1.0e-6, 2.0e-6])),
                    ('Post', np.array([1.0e-8, 2.0e-8])),
                    ('Geofit', np.array([7.0e-7, 8.0e-7]))):
                StandardKBRR().configure(
                    directory, self._DATE_SPAN, stage=stage,
                    mission='GRACE_FO_RL04'
                ).save_arc(
                    self._TIME, coordinates[0], coordinates[1],
                    coordinates[2], rate, arc_no=0
                )

            outputs = post_writer.finalize()
            self.assertEqual(len(outputs), 1)
            output = Path(outputs[0])
            self.assertEqual(
                output.parent, Path(directory) / '2026' / '2026-03'
            )
            self.assertEqual(output.name, 'Hawk_KBRR1C_2026-03-01_01.txt')
            content = output.read_text(encoding='ascii')
            self.assert_standard_header(content)
            self.assertIn('    - pre_range_rate:\n', content)
            self.assertIn('    - post_range_rate:\n', content)
            self.assertIn('    - raw_oc_range_rate:\n', content)
            self.assertIn('    - geofit_range_rate:\n', content)
            rows = content.split('# End of YAML header\n', 1)[1].strip().splitlines()
            self.assertTrue(all(len(row.split()) == 8 for row in rows))
            self.assertAlmostEqual(float(rows[0].split()[4]), 1.0e-6)
            self.assertAlmostEqual(float(rows[0].split()[5]), 1.0e-8)
            self.assertAlmostEqual(float(rows[0].split()[6]), 3.0e-6)
            self.assertAlmostEqual(float(rows[0].split()[7]), 7.0e-7)

    def test_kbrr_discard_arcs_removes_rejected_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = StandardKBRR().configure(
                directory, self._DATE_SPAN, stage='Post',
                mission='GRACE_FO_RL04'
            ).prepare()
            coordinates = (
                np.array([10.0, 11.0]),
                np.array([20.0, 21.0]),
                np.array([500000.0, 500010.0])
            )
            for arc_no, offset in ((0, 0.0), (1, 10.0)):
                for stage in ('Raw', 'Pre', 'Post', 'Geofit'):
                    StandardKBRR().configure(
                        directory, self._DATE_SPAN, stage=stage,
                        mission='GRACE_FO_RL04'
                    ).save_arc(
                        self._TIME + offset,
                        coordinates[0], coordinates[1], coordinates[2],
                        np.array([1.0e-8, 2.0e-8]),
                        arc_no=arc_no
                    )

            removed = writer.discard_arcs([1])
            output = Path(writer.finalize()[0])
            content = output.read_text(encoding='ascii')

            self.assertEqual(len(removed), 4)
            self.assert_standard_header(content, record_number=2)

    def test_adjust_sst_exposes_aligned_raw_and_empirical_prefit(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'RangeRate_0.hdf5'
            with h5py.File(filename, 'w') as h5:
                h5.create_dataset('empirical_prefit_t', data=self._TIME)
                h5.create_dataset(
                    'empirical_prefit_residual',
                    data=np.array([3.0e-6, 4.0e-6])
                )
                h5.create_dataset('empirical_postfit_t', data=self._TIME)
                h5.create_dataset(
                    'empirical_postfit_residual',
                    data=np.array([1.0e-6, 2.0e-6])
                )

            raw_time, raw_oc = RangeRate.load_empirical_residual(
                filename, stage='Pre'
            )
            pre_time, pre_residual = RangeRate.load_empirical_residual(
                filename, stage='Post'
            )

            np.testing.assert_array_equal(raw_time, pre_time)
            np.testing.assert_array_equal(
                raw_oc, np.array([3.0e-6, 4.0e-6])
            )
            np.testing.assert_array_equal(
                pre_residual, np.array([1.0e-6, 2.0e-6])
            )

    def test_geofit_excludes_accelerometer_scale_parameters(self):
        design = np.array([
            [2.0, 3.0, 5.0],
            [7.0, 11.0, 13.0]
        ])
        parameters = np.array([17.0, 19.0, 23.0])
        geofit = CalibrateOrbit._gravity_field_fit(
            design, parameters, scale_parameter_count=1
        )
        np.testing.assert_array_equal(
            geofit,
            np.array([3.0 * 19.0 + 5.0 * 23.0,
                      11.0 * 19.0 + 13.0 * 23.0])
        )

    def test_lgd_header_contains_dimensions_and_global_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = StandardLGD().configure(
                directory, self._DATE_SPAN, mission='GRACE_FO_RL04'
            ).prepare()
            writer.save_arc(
                self._TIME,
                np.array([10.0, 11.0]),
                np.array([20.0, 21.0]),
                np.array([500000.0, 500010.0]),
                np.array([1.0e-10, 2.0e-10]),
                np.array([1.0e-11, 2.0e-11]),
                arc_no=0
            )
            output = Path(writer.finalize()[0])
            self.assertEqual(
                output.parent, Path(directory) / '2026' / '2026-03'
            )
            self.assert_standard_header(output.read_text(encoding='ascii'))

    def test_neq_uses_level_1c_monthly_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            output = StandardNEQ().configure(
                filedir=directory,
                date_span=['2026-03-01', '2026-03-31'],
                degree_max=2,
                degree_min=2
            ).save(NMatrix=np.eye(5), LMatrix=np.ones((5, 1)))
            self.assertEqual(
                Path(output).name,
                'Hawk-NEQ1C_n2_2026-03.snx'
            )
            self.assertEqual(Path(output).parent, Path(directory) / '2026')


if __name__ == '__main__':
    unittest.main()
