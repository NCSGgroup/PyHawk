import os
import tempfile
import unittest

import h5py
import numpy as np

from src.Solver.KalmanFilter import (
    AutoregressiveModel,
    InformationKalmanFilter,
    ITSGGFCWriter,
    NormalEquation,
    PyHawkKalmanInterface,
    eliminate_local_parameters,
    run_pyhawk_kalman,
)
from src.Solver.AOHISTrainer import _pseudo_observation
from src.Auxilary.SortCS import SortCS
from src.Interface.LoadSH import LoadGif48


class TestAutoregressiveModel(unittest.TestCase):
    def test_ar1_round_trip(self):
        model = AutoregressiveModel.from_coefficients(
            [np.array([[0.5]])], np.array([[2.0]])
        )
        transition, covariance = model.order_one_representation()
        np.testing.assert_allclose(transition, [[0.5]])
        np.testing.assert_allclose(covariance, [[2.0]])

    def test_ar2_companion_representation(self):
        model = AutoregressiveModel.from_coefficients(
            [np.array([[0.6]]), np.array([[-0.1]])], np.array([[0.25]])
        )
        transition, covariance = model.order_one_representation()
        np.testing.assert_allclose(transition, [[0.6, -0.1], [1.0, 0.0]])
        np.testing.assert_allclose(covariance, [[0.25, 0.0], [0.0, 0.0]])


class TestInformationKalmanFilter(unittest.TestCase):
    def test_scalar_information_update_and_missing_epoch(self):
        kalman_filter = InformationKalmanFilter(
            transition=np.array([[1.0]]),
            process_covariance=np.array([[1.0]]),
            initial_covariance=np.array([[1.0]]),
        )
        observed = kalman_filter.step(NormalEquation([[4.0]], [8.0]))
        np.testing.assert_allclose(observed.updated_covariance, [[2.0 / 9.0]])
        np.testing.assert_allclose(observed.updated_state, [16.0 / 9.0])
        self.assertTrue(observed.measurement_used)

        missing = kalman_filter.step(None)
        np.testing.assert_allclose(missing.updated_state, [16.0 / 9.0])
        np.testing.assert_allclose(missing.updated_covariance, [[11.0 / 9.0]])
        self.assertFalse(missing.measurement_used)

    def test_ar2_updates_only_current_epoch_block(self):
        model = AutoregressiveModel.from_coefficients(
            [np.array([[0.5]]), np.array([[0.2]])], np.array([[1.0]])
        )
        kalman_filter = InformationKalmanFilter.from_autoregressive_model(
            model, initial_covariance=np.eye(2)
        )
        step = kalman_filter.step(NormalEquation([[2.0]], [3.0]))
        self.assertEqual(step.updated_state.shape, (2,))
        self.assertEqual(step.updated_covariance.shape, (2, 2))
        self.assertTrue(np.all(np.linalg.eigvalsh(step.updated_covariance) > 0.0))

    def test_update_matches_normal_equation_formula(self):
        transition = np.array([[0.8, 0.1], [0.0, 0.9]])
        process_covariance = np.array([[0.3, 0.05], [0.05, 0.2]])
        initial_covariance = np.array([[1.2, 0.1], [0.1, 0.8]])
        initial_state = np.array([0.4, -0.2])
        normal = np.array([[3.0, 0.2], [0.2, 2.0]])
        right_hand_side = np.array([1.0, -0.5])

        predicted_state = transition @ initial_state
        predicted_covariance = (
            process_covariance + transition @ initial_covariance @ transition.T
        )
        expected_covariance = np.linalg.inv(
            normal + np.linalg.inv(predicted_covariance)
        )
        expected_state = expected_covariance @ (
            right_hand_side + np.linalg.solve(predicted_covariance, predicted_state)
        )

        kalman_filter = InformationKalmanFilter(
            transition,
            process_covariance,
            initial_covariance,
            initial_state,
        )
        step = kalman_filter.step(NormalEquation(normal, right_hand_side))
        np.testing.assert_allclose(step.updated_covariance, expected_covariance)
        np.testing.assert_allclose(step.updated_state, expected_state)

    def test_rts_smoother_uses_later_epochs(self):
        kalman_filter = InformationKalmanFilter(
            transition=np.array([[1.0]]),
            process_covariance=np.array([[0.01]]),
            initial_covariance=np.array([[1.0]]),
        )
        steps = kalman_filter.filter(
            [
                NormalEquation([[100.0]], [0.0]),
                NormalEquation([[100.0]], [100.0]),
            ]
        )
        smoothed = kalman_filter.smooth_steps(steps)
        self.assertGreater(smoothed[0].updated_state[0], steps[0].updated_state[0])
        self.assertLess(smoothed[0].updated_state[0], smoothed[1].updated_state[0])

    def test_var_warmup_uses_order_zero_to_three(self):
        covariance = np.array([[1.0]])
        models = {
            0: _pseudo_observation(tuple(), covariance),
            1: _pseudo_observation((np.array([[0.5]]),), covariance),
            2: _pseudo_observation(
                (np.array([[0.5]]), np.array([[0.1]])), covariance
            ),
            3: _pseudo_observation(
                (np.array([[0.5]]), np.array([[0.1]]), np.array([[0.05]])),
                covariance,
            ),
        }
        kalman_filter = InformationKalmanFilter.from_autoregressive_model(
            AutoregressiveModel(models[3]),
            initial_covariance=np.eye(3),
            model_sequence=models,
        )
        steps = kalman_filter.filter([None, None, None, None, None])
        np.testing.assert_allclose(steps[0].transition[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(steps[1].transition[0], [0.5, 0.0, 0.0])
        np.testing.assert_allclose(steps[2].transition[0], [0.5, 0.1, 0.0])
        np.testing.assert_allclose(steps[3].transition[0], [0.5, 0.1, 0.05])
        np.testing.assert_allclose(steps[4].transition, steps[3].transition)


class TestPyHawkAdapter(unittest.TestCase):
    def test_schur_elimination(self):
        reduced = eliminate_local_parameters(
            NormalEquation([[2.0, 1.0], [1.0, 3.0]], [1.0, 4.0]), 1
        )
        np.testing.assert_allclose(reduced.normal, [[2.5]])
        np.testing.assert_allclose(reduced.right_hand_side, [3.5])

    def test_hdf5_streaming_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            neq_file = os.path.join(directory, "daily_neq.hdf5")
            output_file = os.path.join(directory, "filtered.hdf5")
            with h5py.File(neq_file, "w") as h5:
                h5.create_dataset("NMatrix", data=[[2.0]])
                h5.create_dataset("LMatrix", data=[[3.0]])

            model = AutoregressiveModel.from_coefficients(
                [np.array([[1.0]])], np.array([[1.0]])
            )
            kalman_filter = InformationKalmanFilter.from_autoregressive_model(
                model, initial_covariance=np.array([[1.0]])
            )
            interface = PyHawkKalmanInterface(kalman_filter)
            result = interface.run([neq_file, None], output_file)

            self.assertEqual(result, output_file)
            with h5py.File(result, "r") as h5:
                self.assertEqual(h5["updated_state"].shape, (2, 1))
                np.testing.assert_array_equal(h5["measurement_used"][:], [True, False])

    def test_config_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            ar_file = os.path.join(directory, "ar.hdf5")
            covariance_file = os.path.join(directory, "initial.hdf5")
            neq_file = os.path.join(directory, "daily.hdf5")
            output_file = os.path.join(directory, "filtered.hdf5")

            model = AutoregressiveModel.from_coefficients(
                [np.array([[1.0]])], np.array([[1.0]])
            )
            with h5py.File(ar_file, "w") as h5:
                h5.create_dataset("model", data=model.pseudo_observation)
            with h5py.File(covariance_file, "w") as h5:
                h5.create_dataset("covariance", data=[[1.0]])
            with h5py.File(neq_file, "w") as h5:
                h5.create_dataset("NMatrix", data=[[2.0]])
                h5.create_dataset("LMatrix", data=[[3.0]])

            result = run_pyhawk_kalman(
                {
                    "Enabled": True,
                    "AutoregressiveModelFile": ar_file,
                    "AutoregressiveModelDataset": "model",
                    "InitialCovarianceFile": covariance_file,
                    "InitialCovarianceDataset": "covariance",
                    "NormalEquationFiles": [neq_file],
                    "OutputFile": output_file,
                }
            )
            self.assertEqual(result, output_file)
            self.assertTrue(os.path.exists(output_file))

    def test_itsg_gfc_writer_restores_reference_and_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = os.path.join(directory, "filtered.hdf5")
            reference_file = os.path.join(directory, "reference.gfc")
            output_directory = os.path.join(directory, "gfc")
            degree_max = 2
            sort = SortCS(degree_max, 2, method="SortByOrder")
            residual_c = np.zeros(6)
            residual_s = np.zeros(6)
            residual_c[3] = 1.0e-6  # C20
            residual_s[5] = -2.0e-6  # S22
            state = sort.sort(residual_c, residual_s)

            with h5py.File(state_file, "w") as h5:
                h5.create_dataset("smoothed_state", data=[state])
                h5.create_dataset("epoch", data=np.array([b"2020-01-02"]))

            with open(reference_file, "w") as file:
                file.write("product_type gravity_field\n")
                file.write("modelname test_reference\n")
                file.write("earth_gravity_constant 3.986004415e14\n")
                file.write("radius 6.3781363e6\n")
                file.write("max_degree 2\n")
                file.write("norm fully_normalized\n")
                file.write("tide_system zero_tide\n")
                file.write("errors no\n")
                file.write("end_of_head\n")
                for degree in range(3):
                    for order in range(degree + 1):
                        c20 = 0.1 if (degree, order) == (2, 0) else 0.0
                        file.write(
                            "gfc %d %d %.16e 0.0 0.0 0.0\n"
                            % (degree, order, c20)
                        )

            def write_correction(path, c10=0.0, c11=0.0, s11=0.0, c20=0.0):
                with open(path, "w") as file:
                    file.write("max_degree 2\n")
                    file.write("earth_gravity_constant 3.986004415e14\n")
                    file.write("radius 6.3781363e6\n")
                    file.write("norm fully_normalized\n")
                    file.write("tide_system zero_tide\n")
                    file.write("end_of_head\n")
                    for degree in range(3):
                        for order in range(degree + 1):
                            cosine = (
                                c10 if (degree, order) == (1, 0)
                                else c11 if (degree, order) == (1, 1)
                                else c20 if (degree, order) == (2, 0)
                                else 0.0
                            )
                            sine = s11 if (degree, order) == (1, 1) else 0.0
                            file.write(
                                "gfc %d %d %.16e %.16e 0.0 0.0\n"
                                % (degree, order, cosine, sine)
                            )

            aod_file = os.path.join(directory, "aod1b.gfc")
            gia_file = os.path.join(directory, "gia.gfc")
            degree1_file = os.path.join(directory, "degree1.gfc")
            write_correction(aod_file, c20=0.01)
            write_correction(gia_file, c20=0.003)
            write_correction(degree1_file, c10=0.02, c11=0.03, s11=0.04)
            writer = ITSGGFCWriter(
                output_directory,
                reference_file,
                degree_max=degree_max,
                degree_min=2,
                aod1b_reference=aod_file,
                gia_reference=gia_file,
                degree1_reference=degree1_file,
                restore_aod1b=True,
                remove_gia=True,
                restore_degree1=True,
            )
            output_files = writer.write_from_hdf5(state_file)
            self.assertEqual(len(output_files), 1)
            loaded = LoadGif48().load(output_files[0])
            c, s = loaded.getCS(degree_max)
            self.assertEqual(loaded.maxDegree, degree_max)
            np.testing.assert_allclose(c[1], 0.02)
            np.testing.assert_allclose(c[2], 0.03)
            np.testing.assert_allclose(s[2], 0.04)
            np.testing.assert_allclose(c[3], 0.107001)
            np.testing.assert_allclose(s[5], -0.000002)


if __name__ == "__main__":
    unittest.main()
