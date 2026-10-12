import unittest

import numpy as np

from src.ODE.Common.MultiStepODE import MatrixGaussJackson
from src.Preference.Pre_ODE import ODEConfig


class TestCorrectorStatistics(unittest.TestCase):
    def test_both_propagation_interfaces_count_correctors(self):
        config = ODEConfig()
        for one_step in (False, True):
            with self.subTest(one_step=one_step):
                integrator = MatrixGaussJackson().configure(config).setOrder(8)
                # Constant velocity is an exact zero-acceleration solution.
                times = np.arange(-4, 5, dtype=float) * 5.0
                position = np.zeros((9, 2, 3))
                velocity = np.ones((9, 2, 3))
                position[:] = times[:, None, None] * velocity
                integrator.setInitial(times, position, velocity)
                integrator.setForce(lambda t, r, v: np.zeros_like(r))
                integrator.setIterArgs(iterNum=10, rltot=1e-13)
                integrator.setRecord()
                if one_step:
                    state = integrator.propagate_one_step(5., True, None)
                    for _ in range(8):
                        state = integrator.propagate_one_step(5., False, state)
                else:
                    integrator.propagate(8, 5.)
                stats = integrator.get_corrector_stats()
                self.assertEqual(stats['steps'], 8)
                self.assertGreaterEqual(stats['iterations'], 8)
                self.assertLessEqual(stats['max_iterations'], 10)
                self.assertEqual(stats['nonconverged_steps'], 0)
                self.assertEqual(stats['mean_iterations'],
                                 stats['iterations'] / stats['steps'])


if __name__ == '__main__':
    unittest.main()
