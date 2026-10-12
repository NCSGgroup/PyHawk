import os
import tarfile
import tempfile
import unittest
from datetime import datetime, timedelta

import h5py
import numpy as np

from src.Solver.AOHISTrainer import AOHISTrainer, EWHGrid, train_aohis_from_config
from src.Solver.KalmanFilter import AutoregressiveModel, run_pyhawk_kalman


class TestAOHISTrainer(unittest.TestCase):
    @staticmethod
    def _six_hourly_samples(days=120, dimension=5):
        start = datetime(2000, 1, 1)
        epochs = [
            start + timedelta(hours=6 * index)
            for index in range(days * 4)
        ]
        rng = np.random.default_rng(1234)
        state = np.zeros((len(epochs), dimension))
        for index in range(1, len(epochs)):
            state[index] = (
                0.82 * state[index - 1] + rng.normal(0.0, 0.01, dimension)
            )
        ao_error = 0.7 * state
        hydrology = 0.3 * state
        return epochs, ao_error, hydrology

    def test_daily_aohis_training_builds_groops_model(self):
        epochs, ao_error, hydrology = self._six_hourly_samples()
        trainer = AOHISTrainer(
            degree_max=2,
            degree_min=2,
            order=2,
            remove_trend=False,
            remove_annual=False,
        )
        result = trainer.fit(epochs, ao_error, hydrology)

        self.assertEqual(len(result.epochs), 120)
        self.assertEqual(result.residuals.shape, (120, 5))
        self.assertEqual(result.lag_covariances.shape, (3, 5, 5))
        self.assertEqual(result.pseudo_observation.shape, (5, 15))
        self.assertEqual(set(result.models), {0, 1, 2})
        model = AutoregressiveModel(result.pseudo_observation)
        self.assertEqual(model.dimension, 5)
        self.assertEqual(model.order, 2)
        for expected, actual in zip(result.coefficients, model.coefficients()):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_hdf5_input_and_model_persistence(self):
        epochs, ao_error, hydrology = self._six_hourly_samples(days=75)
        with tempfile.TemporaryDirectory() as directory:
            input_file = os.path.join(directory, "esm_training.hdf5")
            model_file = os.path.join(directory, "aohis_ar.hdf5")
            labels = np.asarray(
                [epoch.strftime("%Y-%m-%dT%H:%M:%S") for epoch in epochs],
                dtype="S19",
            )
            with h5py.File(input_file, "w") as h5:
                h5.create_dataset("epochs", data=labels)
                h5.create_dataset("aoErr", data=ao_error)
                h5.create_dataset("H", data=hydrology)

            trainer = AOHISTrainer(
                degree_max=2,
                degree_min=2,
                order=1,
                remove_trend=False,
                remove_annual=False,
            )
            result = trainer.fit_from_file(input_file)
            self.assertEqual(result.residuals.shape, (75, 5))
            result.save(model_file, include_residuals=True)
            with h5py.File(model_file, "r") as h5:
                self.assertEqual(h5["model"].shape, (5, 10))
                self.assertEqual(h5["coefficients"].shape, (1, 5, 5))
                self.assertEqual(h5["residuals"].shape, (75, 5))
                self.assertEqual(h5["model_order_0"].shape, (5, 5))

    def test_gfc_directories_are_joined_by_epoch(self):
        start = datetime(2001, 1, 29)
        epochs = [start + timedelta(days=index) for index in range(75)]
        rng = np.random.default_rng(42)
        ao_values = rng.normal(0.0, 1.0e-3, (len(epochs), 5))
        hydro_values = rng.normal(0.0, 1.0e-3, (len(epochs), 5))
        with tempfile.TemporaryDirectory() as directory:
            ao_directory = os.path.join(directory, "aoerr")
            hydro_directory = os.path.join(directory, "hydrology")
            os.makedirs(ao_directory)
            os.makedirs(hydro_directory)
            for index, epoch in enumerate(epochs):
                date_label = epoch.strftime("%Y%m%d")
                for root, values in (
                    (ao_directory, ao_values[index]),
                    (hydro_directory, hydro_values[index]),
                ):
                    filename = os.path.join(root, "field_%s.gfc" % date_label)
                    with open(filename, "w") as file:
                        file.write("max_degree 2\n")
                        file.write(
                            "time_period_of_data %s %s\n"
                            % (date_label, date_label)
                        )
                        for degree in range(3):
                            for order in range(degree + 1):
                                full_index = degree * (degree + 1) // 2 + order
                                cosine = (
                                    values[full_index - 3]
                                    if degree >= 2
                                    else 0.0
                                )
                                sine = (
                                    values[2 + order]
                                    if degree >= 2 and order > 0
                                    else 0.0
                                )
                                file.write(
                                    "gfc %d %d %.16e %.16e 0.0 0.0\n"
                                    % (degree, order, cosine, sine)
                                )

            # ESA ESM distributes annual tar.gz archives inside these folders.
            for root in (ao_directory, hydro_directory):
                files = [
                    os.path.join(root, name)
                    for name in os.listdir(root)
                    if name.endswith(".gfc")
                ]
                archive_file = os.path.join(root, "annual.tar.gz")
                with tarfile.open(archive_file, "w:gz") as archive:
                    for filename in files:
                        archive.add(filename, arcname=os.path.basename(filename))
                for filename in files:
                    os.remove(filename)

            trainer = AOHISTrainer(
                degree_max=2,
                degree_min=2,
                order=1,
                remove_trend=False,
                remove_annual=False,
            )
            result = trainer.fit_from_esm_directories(
                ao_directory, hydro_directory
            )
            self.assertEqual(len(result.epochs), 75)
            self.assertEqual(result.pseudo_observation.shape, (5, 10))

    def test_config_entry_point_trains_before_filtering(self):
        epochs, ao_error, hydrology = self._six_hourly_samples(days=75)
        with tempfile.TemporaryDirectory() as directory:
            input_file = os.path.join(directory, "esm_training.hdf5")
            model_file = os.path.join(directory, "aohis_ar.hdf5")
            covariance_file = os.path.join(directory, "initial.hdf5")
            neq_file = os.path.join(directory, "daily.hdf5")
            output_file = os.path.join(directory, "filtered.hdf5")
            labels = np.asarray(
                [epoch.strftime("%Y-%m-%dT%H:%M:%S") for epoch in epochs],
                dtype="S19",
            )
            with h5py.File(input_file, "w") as h5:
                h5.create_dataset("epochs", data=labels)
                h5.create_dataset("aoErr", data=ao_error)
                h5.create_dataset("H", data=hydrology)
            with h5py.File(covariance_file, "w") as h5:
                h5.create_dataset("covariance", data=np.eye(5))
            with h5py.File(neq_file, "w") as h5:
                h5.create_dataset("NMatrix", data=np.eye(5))
                h5.create_dataset("LMatrix", data=np.zeros(5))

            result = run_pyhawk_kalman(
                {
                    "Enabled": True,
                    "TrainAOHIS": True,
                    "AOHISInputFile": input_file,
                    "AOHISModelOutputFile": model_file,
                    "AOHISDegreeMax": 2,
                    "AOHISDegreeMin": 2,
                    "AOHISOrder": 1,
                    "AOHISRemoveTrend": False,
                    "AOHISRemoveAnnual": False,
                    "InitialCovarianceFile": covariance_file,
                    "InitialCovarianceDataset": "covariance",
                    "NormalEquationFiles": [neq_file],
                    "OutputFile": output_file,
                }
            )
            self.assertEqual(result, output_file)
            self.assertTrue(os.path.exists(model_file))
            with h5py.File(model_file, "r") as h5:
                self.assertEqual(h5["model"].shape, (5, 10))

    def test_warmup_models_can_be_disabled(self):
        epochs, ao_error, hydrology = self._six_hourly_samples(days=75)
        trainer = AOHISTrainer(
            degree_max=2,
            degree_min=2,
            order=1,
            remove_trend=False,
            remove_annual=False,
            warmup_models=False,
        )
        result = trainer.fit(epochs, ao_error, hydrology)
        self.assertEqual(set(result.models), {1})

    def test_cryosphere_and_ewh_partition_are_used(self):
        epochs, ao_error, hydrology = self._six_hourly_samples(days=120)
        rng = np.random.default_rng(9)
        cryosphere = rng.normal(0.0, 1.0e-3, ao_error.shape)
        latitude = np.repeat([-60.0, 0.0, 60.0], 4)
        longitude = np.tile([0.0, 90.0, 180.0, 270.0], 3)
        grid = EWHGrid(
            latitude=latitude,
            longitude=longitude,
            land_mask=np.array(
                [True, True, True, True, False, False, False, False,
                 False, False, False, False]
            ),
        )
        trainer = AOHISTrainer(
            degree_max=2,
            degree_min=2,
            order=1,
            remove_trend=False,
            remove_annual=False,
            include_cryosphere=True,
            covariance_domain="ewh",
            ewh_grid=grid,
            distance_scale_km=1000.0,
            distance_cutoff_km=3000.0,
            ewh_block_size=4,
        )
        result = trainer.fit(epochs, ao_error, hydrology, cryosphere)
        self.assertEqual(result.metadata["component_definition"], "aoErr+H+I")
        self.assertEqual(result.metadata["covariance_domain"], "ewh")
        self.assertEqual(result.metadata["land_ocean_cross_covariance"], "zero")
        self.assertTrue(np.all(np.isfinite(result.pseudo_observation)))

    def test_cryosphere_is_required_in_exact_mode(self):
        epochs, ao_error, hydrology = self._six_hourly_samples(days=75)
        trainer = AOHISTrainer(
            degree_max=2,
            degree_min=2,
            order=1,
            remove_trend=False,
            remove_annual=False,
            include_cryosphere=True,
        )
        with self.assertRaises(ValueError):
            trainer.fit(epochs, ao_error, hydrology)

    def test_order_zero_result_can_be_saved_as_diagnostic(self):
        epochs, ao_error, hydrology = self._six_hourly_samples(days=75)
        trainer = AOHISTrainer(
            degree_max=2,
            degree_min=2,
            order=0,
            remove_trend=False,
            remove_annual=False,
        )
        result = trainer.fit(epochs, ao_error, hydrology)
        with tempfile.TemporaryDirectory() as directory:
            output_file = os.path.join(directory, "order0.hdf5")
            result.save(output_file)
            with h5py.File(output_file, "r") as h5:
                self.assertEqual(h5["coefficients"].shape, (0, 5, 5))


if __name__ == "__main__":
    unittest.main()
