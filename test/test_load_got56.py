import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.Auxilary.FunNutArg import FunNutArg
from src.Interface.LoadSH import LoadGOT56, LoadSH


HEADER = '''Ocean tide potential including loading, in CE.
modelname GOT5.6_OCN_test
product_type gravity_field
earth_gravity_constant 3.9860044150D+14
radius 6.3781363000d+06
max_degree 2
norm fully_normalized
errors no
end_of_head
'''


class TestLoadGOT56(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.c = np.array([0., 1., 2., 3., 4., 5.]) * 1e-10
        self.s = -self.c
        self.component = self.path / 'cos.gfc'
        self._write_component(self.component, self.c, self.s)

    def _write_component(self, path, c, s, header=HEADER):
        records = []
        # Reverse order checks that the loader packs by n,m, not file order.
        for n in reversed(range(3)):
            for m in reversed(range(n + 1)):
                i = n * (n + 1) // 2 + m
                records.append('gfc %d %d %.15e %.15e' % (n, m, c[i], s[i]))
        path.write_text(header + '\n'.join(records) + '\n', encoding='utf-8')

    def _bundle(self):
        self._write_component(self.path / 'sin.gfc', self.c * 2, self.s * 3)
        (self.path / 'GOT5.6_OCN_001fileList.txt').write_text('cos.gfc sin.gfc\n', encoding='ascii')
        np.savetxt(self.path / 'GOT5.6_OCN_002doodson.txt', [[1, 0, 0, 0, 0, 0]], fmt='%d')
        np.savetxt(self.path / 'GOT5.6_OCN_003admittance.txt', [[1.]])
        return LoadGOT56().load(self.path)

    def test_component_metadata_and_packed_coefficients(self):
        loader = LoadGOT56().load(self.component)
        self.assertIsInstance(loader, LoadSH)
        c, s = loader.getCS(2)
        np.testing.assert_array_equal(c, self.c)
        np.testing.assert_array_equal(s, self.s)
        self.assertEqual(loader.GM, 3.986004415e14)
        self.assertEqual(loader.Radius, 6378136.3)
        self.assertEqual(loader.norm, 'fully_normalized')
        self.assertEqual(loader.output_frame, 'CE')
        self.assertTrue(loader.includes_loading)
        self.assertIsNone(loader.zero_tide)
        c[3] = 99
        self.assertEqual(loader.getCS()[0][3], self.c[3])

    def test_truncation_and_invalid_degrees(self):
        loader = LoadGOT56().load(self.component, maxDegree=1)
        self.assertEqual(loader.sourceMaxDegree, 2)
        np.testing.assert_array_equal(loader.getCS()[0], self.c[:3])
        for degree in (-1, 1.5, True, 3):
            with self.subTest(degree=degree), self.assertRaises(ValueError):
                LoadGOT56().load(self.component, maxDegree=degree)
        with self.assertRaises(ValueError):
            loader.getCS(2)
        with self.assertRaises(NotImplementedError):
            loader.getSigmaCS(1)

    def test_bundle_cos_sin_synthesis_and_frames(self):
        loader = self._bundle()
        with self.assertRaises(RuntimeError):
            loader.getCS()
        for phase in (0., np.pi / 2, .7):
            with self.subTest(phase=phase):
                loader.setDoodsonArguments([phase, 0, 0, 0, 0, 0], frame='CE')
                c, s = loader.getCS()
                np.testing.assert_allclose(c, self.c * (np.cos(phase) + 2 * np.sin(phase)), atol=1e-24)
                np.testing.assert_allclose(s, self.s * (np.cos(phase) + 3 * np.sin(phase)), atol=1e-24)
                loader.setDoodsonArguments([phase, 0, 0, 0, 0, 0])
                cmc, cms = loader.getCS()
                np.testing.assert_array_equal(cmc[1:3], 0.)
                np.testing.assert_array_equal(cms[1:3], 0.)
                np.testing.assert_array_equal(cmc[3:], c[3:])
                np.testing.assert_array_equal(cms[3:], s[3:])

    def test_time_uses_project_arguments(self):
        loader = self._bundle()
        loader.setTime(54650.00059240741, 4.9101027041591285)
        actual = loader.getCS()
        beta = FunNutArg().getNutArg(54650.00059240741, 4.9101027041591285)
        expected = loader.setDoodsonArguments(beta).getCS()
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)

    def test_invalid_synthesis_inputs(self):
        loader = self._bundle()
        for beta in ([0] * 5, [np.nan] * 6, [np.inf] * 6):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                loader.setDoodsonArguments(beta)
        with self.assertRaises(ValueError):
            loader.setDoodsonArguments([0] * 6, frame='CF')
        with self.assertRaises(ValueError):
            loader.setTime(np.nan, 0.)
        with self.assertRaises(RuntimeError):
            LoadGOT56().load(self.component).setTime(51544.5, 0.)

    def test_bad_or_incomplete_coefficients_rejected(self):
        original = self.component.read_text(encoding='utf-8')
        records = original.splitlines()
        bad_inputs = [original.replace('end_of_head', 'missing_end'),
                      original.replace('fully_normalized', 'unnormalized'),
                      original.replace('errors no', 'errors formal'),
                      original + records[-1] + '\n',
                      '\n'.join(records[:-1]) + '\n',
                      original.replace('gfc 2 2', 'gfc 2 3'),
                      original.replace('gfc 2 2 5.000000000000000e-10', 'gfc 2 2 nan')]
        for text in bad_inputs:
            with self.subTest(text=text[:80]):
                self.component.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    LoadGOT56().load(self.component)

    def test_matrix_mismatch_and_noninteger_doodson_rejected(self):
        self._bundle()
        np.savetxt(self.path / 'GOT5.6_OCN_003admittance.txt', [[1., 2.]])
        with self.assertRaises(ValueError):
            LoadGOT56().load(self.path)
        np.savetxt(self.path / 'GOT5.6_OCN_003admittance.txt', [[1.]])
        np.savetxt(self.path / 'GOT5.6_OCN_002doodson.txt', [[.5, 0, 0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            LoadGOT56().load(self.path)

    def test_component_metadata_mismatch_and_missing_file_rejected(self):
        self._bundle()
        self._write_component(self.path / 'sin.gfc', self.c, self.s,
                              HEADER.replace('6.3781363000d+06', '6.0d+06'))
        with self.assertRaises(ValueError):
            LoadGOT56().load(self.path)
        (self.path / 'GOT5.6_OCN_001fileList.txt').write_text('cos.gfc missing.gfc\n', encoding='ascii')
        with self.assertRaises(FileNotFoundError):
            LoadGOT56().load(self.path)

    def test_reload_clears_atlas_state(self):
        loader = self._bundle().setTime(51544.5, 0.)
        loader.load(self.component)
        self.assertIsNone(loader.admittance)
        self.assertEqual(loader.output_frame, 'CE')
        np.testing.assert_array_equal(loader.getCS()[0], self.c)


ROOT = Path(__file__).resolve().parents[1]
PRODUCT = ROOT / 'data' / 'GOT5.6' / 'GOT5.6_OCN'
REFERENCE = PRODUCT.parent / 'tide_prediction_official.py'


@unittest.skipUnless(PRODUCT.is_dir() and REFERENCE.is_file(), 'Official local GOT5.6 product unavailable')
class TestOfficialGOT56(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = LoadGOT56().load(PRODUCT)
        spec = importlib.util.spec_from_file_location('official_tide_prediction', REFERENCE)
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)
        cls.cos = [cls.reference.read_gfc(PRODUCT / pair[0]) for pair in cls.loader.fileList]
        cls.sin = [cls.reference.read_gfc(PRODUCT / pair[1]) for pair in cls.loader.fileList]

    def test_official_bundle_dimensions(self):
        self.assertEqual(self.loader.maxDegree, 180)
        self.assertEqual(self.loader.fileList.shape, (26, 2))
        self.assertEqual(self.loader.doodsonMatrix.shape, (414, 6))
        self.assertEqual(self.loader.admittance.shape, (26, 414))
        self.loader.setDoodsonArguments(np.zeros(6))
        self.assertEqual(len(self.loader.getCS(180)[0]), 16471)

    def test_load_truncation_matches_full_product(self):
        beta = self.reference.doodson_arguments(56500.75)
        full_c, full_s = self.loader.setDoodsonArguments(beta).getCS(120)
        truncated = LoadGOT56().load(PRODUCT, maxDegree=120)
        c, s = truncated.setDoodsonArguments(beta).getCS()
        self.assertEqual(c.shape, (7381,))
        np.testing.assert_allclose(c, full_c, rtol=5e-14, atol=1e-23)
        np.testing.assert_allclose(s, full_s, rtol=5e-14, atol=1e-23)

    def test_full_synthesis_matches_official_at_multiple_epochs(self):
        max_difference = 0.
        for mjd in (51544.5, 54650.25, 56500.75, 60310.125):
            beta = self.reference.doodson_arguments(mjd)
            phase = self.loader.doodsonMatrix @ beta
            fc = self.loader.admittance @ np.cos(phase)
            fs = self.loader.admittance @ np.sin(phase)
            expected = np.zeros((181, 181))
            for k in range(26):
                expected += fc[k] * self.cos[k] + fs[k] * self.sin[k]
            expected[:2, :2] = 0.
            c = np.array([expected[n, m] for n in range(181) for m in range(n + 1)])
            s = np.array([expected[m - 1, n] if m else 0. for n in range(181) for m in range(n + 1)])
            actual_c, actual_s = self.loader.setDoodsonArguments(beta).getCS(180)
            np.testing.assert_allclose(actual_c, c, rtol=5e-14, atol=1e-23)
            np.testing.assert_allclose(actual_s, s, rtol=5e-14, atol=1e-23)
            max_difference = max(max_difference, np.max(np.abs(actual_c - c)), np.max(np.abs(actual_s - s)))
        print('GOT5.6 vs official synthesis max coefficient difference: %.3e' % max_difference)


if __name__ == '__main__':
    unittest.main()
