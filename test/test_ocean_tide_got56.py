import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.ForceModel.OceanTide import AbsOceanTide, EOT11a, FES2014, GOT56
from src.Interface.LoadSH import LoadGOT56
from src.Preference.EnumType import InstanceOceanTide, ForceType
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.SecDerivative.Common.ConfigForce_GravHar import ConfigGravHar


def ocean_config(path, degree=120):
    config = ForceModelConfig()
    config.instance_GravHar['OceanTide'] = 'GOT56'
    config.PathOfFilesConfig['GOT56'] = str(path)
    config.OceanTideConfig.update(maxDegree=degree, gainFactor=0.)
    config.include_GravHarType = {key: key == 'OceanTide' for key in config.include_GravHarType}
    return config


class TestGOT56Adapter(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.config = ocean_config(self.path, degree=2)
        self.c = np.arange(6, dtype=float) * 1e-12
        self.s = -self.c
        header = ('modelname GOT5.6_OCN_test\nproduct_type gravity_field\n'
                  'earth_gravity_constant 3.986004415D+14\nradius 6.3781363d+06\n'
                  'max_degree 2\nnorm fully_normalized\nerrors no\nend_of_head\n')
        for name in ('cos.gfc', 'sin.gfc'):
            rows = []
            for n in range(3):
                for m in range(n + 1):
                    index = n * (n + 1) // 2 + m
                    c, s = (self.c[index], self.s[index]) if name == 'cos.gfc' else (0., 0.)
                    rows.append('gfc %d %d %.16e %.16e' % (n, m, c, s))
            (self.path / name).write_text(header + '\n'.join(rows) + '\n')
        (self.path / 'GOT5.6_OCN_001fileList.txt').write_text('cos.gfc sin.gfc\n')
        np.savetxt(self.path / 'GOT5.6_OCN_002doodson.txt', np.zeros((1, 6)), fmt='%d')
        np.savetxt(self.path / 'GOT5.6_OCN_003admittance.txt', [[1.]])
        self.read_patch = patch.object(GOT56, '_GOT56__read_gfc', wraps=GOT56._GOT56__read_gfc)
        self.read_gfc = self.read_patch.start()
        self.addCleanup(self.read_patch.stop)
        self.time = SimpleNamespace(getTDB_mjd=56500., getTheta=1.)

    def model(self):
        model = GOT56().configure(self.config).speedUp().load()
        model._ng = Mock(wraps=model._ng)
        return model

    def test_enum_and_both_existing_models_preserved(self):
        self.assertEqual(InstanceOceanTide.EOT11a.value, 0)
        self.assertEqual(InstanceOceanTide.FES2014.value, 1)
        self.assertIsInstance(AbsOceanTide.new_OceanTide(InstanceOceanTide.EOT11a), EOT11a)
        self.assertIsInstance(AbsOceanTide.new_OceanTide(InstanceOceanTide.FES2014), FES2014)
        self.assertIsInstance(AbsOceanTide.new_OceanTide(InstanceOceanTide.GOT56), GOT56)
        self.assertEqual(ForceModelConfig().instance_GravHar['OceanTide'], 'EOT11a')
        self.assertEqual(ForceModelConfig().PathOfFilesConfig['GOT56'], '../data/GOT5.6/GOT5.6_OCN')

    def test_native_synthesis_and_ce_preserved(self):
        model = self.model().setTime(self.time)
        self.assertEqual(self.read_gfc.call_count, 2)
        c, s = model.getCS()
        model._ng.getNutArg.assert_called_once_with(56500., 1.)
        np.testing.assert_array_equal(c, self.c)
        np.testing.assert_array_equal(s, self.s)
        c[1] = 99.
        s[1] = 99.
        self.assertNotEqual(model.getCS()[0][1], 99.)
        self.assertNotEqual(model.getCS()[1][1], 99.)

    def test_cache_uses_seconds_and_copies_results(self):
        self.config.OceanTideConfig['gainFactor'] = 60.
        model = self.model().setTime(self.time)
        c, _ = model.getCS()
        c[:] = 99.
        model.setTime(SimpleNamespace(getTDB_mjd=56500. + 10. / 86400., getTheta=1.1))
        np.testing.assert_array_equal(model.getCS()[0], self.c)
        self.assertEqual(model._ng.getNutArg.call_count, 1)
        model.setTime(SimpleNamespace(getTDB_mjd=56500. + 61. / 86400., getTheta=1.2)).getCS()
        self.assertEqual(model._ng.getNutArg.call_count, 2)

    def test_reload_clears_epoch_cache(self):
        self.config.OceanTideConfig['gainFactor'] = 60.
        model = self.model().setTime(self.time)
        model.getCS()
        model.load().getCS()
        self.assertEqual(model._ng.getNutArg.call_count, 2)
        self.assertEqual(self.read_gfc.call_count, 4)

    def test_uninitialized_and_unsupported_degree_fail_early(self):
        with self.assertRaises(RuntimeError):
            GOT56().getCS()
        with self.assertRaises(RuntimeError):
            self.model().getCS()
        self.config.OceanTideConfig['maxDegree'] = 181
        with self.assertRaises(ValueError):
            self.model()
        self.config.OceanTideConfig['maxDegree'] = 2
        self.config.PathOfFilesConfig['GOT56'] = str(self.path / 'cos.gfc')
        with self.assertRaises(ValueError):
            self.model()

    def test_config_gravhar_factory_and_total_coefficients(self):
        with patch('src.SecDerivative.Common.ConfigForce_GravHar.ClibGravHar'):
            factory = ConfigGravHar(None).configure(self.config)
        self.assertIsInstance(factory.GravHarForce['OceanTide'], GOT56)
        c, s = factory.setTime(self.time).getTotalCS()
        np.testing.assert_array_equal(c, self.c)
        np.testing.assert_array_equal(s, self.s)

    def test_orbit_reconfigure_keeps_loaded_tides_but_explicit_configure_reloads(self):
        from src.SecDerivative.Common.ConfigForce import ConfigForce

        self.config.include_ForceType = {
            key: key == 'GravHar' for key in self.config.include_ForceType
        }
        with patch('src.SecDerivative.Common.ConfigForce_GravHar.ClibGravHar'):
            force = ConfigForce(arc=0, ep=None).configure(self.config)
            gravity = force.getForceType()['GravHar']
            gravity.setTime(self.time).getTotalCS()
            force.reConfigure()
            self.assertIs(force.getForceType()['GravHar'], gravity)
            force.getForceType()['GravHar'].setTime(self.time).getTotalCS()
            self.assertEqual(self.read_gfc.call_count, 2)
            force.configure(self.config)
            self.assertIsNot(force.getForceType()['GravHar'], gravity)
            self.assertEqual(self.read_gfc.call_count, 4)

    def test_top_level_force_factory_selects_got56(self):
        from src.SecDerivative.Common.ConfigForce import ConfigForce

        self.config.include_ForceType = {key: key == 'GravHar' for key in self.config.include_ForceType}
        with patch('src.SecDerivative.Common.ConfigForce_GravHar.ClibGravHar'):
            force = ConfigForce(arc=0, ep=None).configure(self.config).getForceType()
        self.assertIsInstance(force['GravHar'].GravHarForce['OceanTide'], GOT56)
        c, s = force['GravHar'].setTime(self.time).getTotalCS()
        np.testing.assert_array_equal(c, self.c)
        np.testing.assert_array_equal(s, self.s)

    def test_old_path_dictionary_uses_new_default(self):
        del self.config.PathOfFilesConfig['GOT56']
        model = GOT56().configure(self.config)
        self.assertEqual(model._PathOfFilesConfig.GOT56, '../data/GOT5.6/GOT5.6_OCN')

    def test_native_reader_rejects_invalid_records(self):
        component = self.path / 'cos.gfc'
        original = component.read_text()
        invalid = [original + 'gfc 2 2 0 0\n',
                   '\n'.join(original.splitlines()[:-1]) + '\n',
                   original.replace('gfc 2 2', 'gfc 2 3'),
                   original.replace('fully_normalized', 'unnormalized'),
                   original.replace('gfc 0 0 0.0000000000000000e+00', 'gfc 0 0 nan')]
        for content in invalid:
            component.write_text(content)
            with self.assertRaises(ValueError):
                self.model()

    def test_force_model_does_not_construct_loadsh_reader(self):
        with patch('src.Interface.LoadSH.LoadGOT56', side_effect=AssertionError('unexpected reader')):
            model = self.model().setTime(self.time)
            model.getCS()
            model.setTime(SimpleNamespace(getTDB_mjd=56501., getTheta=2.)).getCS()
            self.assertEqual(model._ng.getNutArg.call_count, 2)
            model.load().getCS()
            self.assertEqual(model._ng.getNutArg.call_count, 3)
        import inspect
        from src.ForceModel import OceanTide
        self.assertNotIn('LoadSH', inspect.getsource(OceanTide))
        self.assertNotIn('LoadGOT56', inspect.getsource(OceanTide))


PRODUCT = Path(__file__).resolve().parents[1] / 'data' / 'GOT5.6' / 'GOT5.6_OCN'


@unittest.skipUnless(PRODUCT.is_dir(), 'Official GOT5.6 product unavailable')
class TestOfficialGOT56Factory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ocean_config(PRODUCT, degree=180)
        cls.factory = ConfigGravHar(None).configure(cls.config)
        cls.direct = LoadGOT56().load(PRODUCT)

    def test_factory_matches_loadsh_at_multiple_epochs(self):
        for mjd, theta in ((51544.5, 0.), (54650.25, 2.), (60310.125, 5.)):
            with self.subTest(mjd=mjd):
                time = SimpleNamespace(getTDB_mjd=mjd, getTheta=theta)
                actual = self.factory.setTime(time).getTotalCS()
                expected = self.direct.setTime(mjd, theta, frame='CE').getCS(180)
                for a, e in zip(actual, expected):
                    self.assertEqual(a.shape, (16471,))
                    np.testing.assert_allclose(a, e, rtol=5e-14, atol=1e-23)

    def test_degree_120_truncation(self):
        factory = ConfigGravHar(None).configure(ocean_config(PRODUCT, degree=120))
        time = SimpleNamespace(getTDB_mjd=56500.75, getTheta=1.5)
        actual = factory.setTime(time).getTotalCS()
        expected = self.direct.setTime(time.getTDB_mjd, time.getTheta, frame='CE').getCS(120)
        for a, e in zip(actual, expected):
            self.assertEqual(a.shape, (7381,))
            np.testing.assert_allclose(a, e, rtol=5e-14, atol=1e-23)

    def test_assembly_applies_cm_and_matches_direct_acceleration(self):
        from src.ForceModel.GravHar import ClibGravHar
        from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative

        # Isolate the tidal force, without loading unrelated ephemerides/AOD data.
        self.factory.InputPosCoordinate = 'ITRS'
        self.factory.OutputCoordinate = 'ITRS'
        time = SimpleNamespace(getTDB_mjd=56500.75, getTheta=1.5, getRotationMatrix=np.eye(3))
        assembly = Assemble2ndDerivative()
        assembly.ForceType = {ForceType.GravHar.name: self.factory}
        assembly._Assemble2ndDerivative__time = time
        assembly._TransitionMatrixConfig = SimpleNamespace(isRequired=False)
        assembly._StokesCoefficientsConfig = SimpleNamespace(isRequired=False)
        position = np.array([6800000., 100000., 500000.])
        actual = assembly.setPosAndVel(position, np.zeros(3)).getCS().calculation().getAcceleration()
        c, s = self.direct.setTime(time.getTDB_mjd, time.getTheta, frame='CM').getCS(180)
        c[0] = s[0] = 0.
        expected = ClibGravHar().configure(self.config).setPar(c, s, position).getAcceleration()
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(np.linalg.norm(actual), 0.)
        np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=1e-20)


if __name__ == '__main__':
    unittest.main()
