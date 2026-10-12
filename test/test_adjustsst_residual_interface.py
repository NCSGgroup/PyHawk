import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.Solver.AdjustSST import RangeRate


class TestAdjustSSTResidualInterface(unittest.TestCase):
    def test_loads_aligned_empirical_pre_and_post(self):
        time = np.array([10.0, 15.0, 20.0])
        prefit = np.array([3.0e-6, 2.0e-6, 1.0e-6])
        postfit = np.array([3.0e-8, 2.0e-8, 1.0e-8])

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'RangeRate_0.hdf5'
            with h5py.File(filename, 'w') as h5:
                h5.create_dataset('empirical_prefit_t', data=time)
                h5.create_dataset(
                    'empirical_prefit_residual', data=prefit
                )
                h5.create_dataset('empirical_postfit_t', data=time)
                h5.create_dataset(
                    'empirical_postfit_residual', data=postfit
                )

            loaded_time, loaded_prefit = (
                RangeRate.load_empirical_residual(filename, stage='Pre')
            )
            _, loaded_postfit = RangeRate.load_empirical_residual(
                filename, stage='Post'
            )

        np.testing.assert_array_equal(loaded_time, time)
        np.testing.assert_array_equal(loaded_prefit, prefit)
        np.testing.assert_array_equal(loaded_postfit, postfit)

    def test_rejects_gravity_postfit_stage_name(self):
        with self.assertRaises(ValueError):
            RangeRate.load_empirical_residual(
                'unused.hdf5', stage='GravityPost'
            )

    def test_applies_non_destructive_epoch_qc_mask_to_residual_and_dm(self):
        time = np.array([10.0, 15.0, 20.0, 25.0])
        residual = np.array([1.0e-8, 9.0e-6, 2.0e-8, 3.0e-8])
        keep = np.array([True, False, True, True])
        matrix = np.arange(12, dtype=float).reshape(4, 3)

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'RangeRate_0.hdf5'
            with h5py.File(filename, 'w') as h5:
                for stage in ('prefit', 'postfit'):
                    h5.create_dataset('empirical_%s_t' % stage, data=time)
                    h5.create_dataset(
                        'empirical_%s_residual' % stage, data=residual
                    )
                h5.create_dataset('design_matrix', data=matrix)
                h5.create_dataset('arc_qc_epoch_keep', data=keep)

            loaded_time, loaded_residual = (
                RangeRate.load_empirical_residual(filename, stage='Post')
            )
            loaded_matrix = RangeRate.load_empirical_design_matrix(filename)

        np.testing.assert_array_equal(loaded_time, time[keep])
        np.testing.assert_array_equal(loaded_residual, residual[keep])
        np.testing.assert_array_equal(loaded_matrix, matrix[keep, :])

    def test_detection_only_onecpr_passes_retained_oc_without_dm(self):
        time = np.array([10.0, 15.0, 20.0, 25.0])
        retained_oc = np.array([3.0e-6, 2.0e-6, 1.0e-6, 4.0e-6])
        detection_residual = np.array([3.0e-8, 2.0e-8, 1.0e-8, 4.0e-8])
        keep = np.array([True, False, True, True])
        matrix = np.arange(12, dtype=float).reshape(4, 3)

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'RangeRate_0.hdf5'
            with h5py.File(filename, 'w') as h5:
                h5.create_dataset('empirical_prefit_t', data=time)
                h5.create_dataset(
                    'empirical_prefit_residual', data=retained_oc
                )
                h5.create_dataset('empirical_postfit_t', data=time)
                h5.create_dataset(
                    'empirical_postfit_residual', data=detection_residual
                )
                h5.create_dataset('retained_oc_t', data=time)
                h5.create_dataset(
                    'retained_oc_residual', data=retained_oc
                )
                h5.create_dataset('design_matrix', data=matrix)
                h5.create_dataset('arc_qc_epoch_keep', data=keep)
                h5.attrs['onecpr_role'] = 'outlier_detection_only'

            loaded_time, loaded_residual = (
                RangeRate.load_empirical_residual(filename, stage='Post')
            )
            loaded_matrix = RangeRate.load_empirical_design_matrix(filename)

        np.testing.assert_array_equal(loaded_time, time[keep])
        np.testing.assert_array_equal(loaded_residual, retained_oc[keep])
        self.assertEqual(loaded_matrix.shape, (np.count_nonzero(keep), 0))


if __name__ == '__main__':
    unittest.main()
