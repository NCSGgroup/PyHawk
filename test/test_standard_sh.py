# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.Auxilary.StandardSH import FormatWrite_FO, StandardSH
from src.Interface.LoadSH import LoadGif48


class _ReferenceModel:
    GM = 3.986004415e14
    Radius = 6378136.3
    norm = 'fully_normalized'
    zero_tide = True


class TestStandardSH(unittest.TestCase):
    def setUp(self):
        self.degree = 3
        self.size = (self.degree + 1) * (self.degree + 2) // 2
        self.c = np.arange(self.size, dtype=np.float64) * 1e-12
        self.s = -self.c
        self.sigma_c = np.arange(1, self.size + 1, dtype=np.float64) * 1e-16
        self.sigma_s = self.sigma_c * 2

    def test_grace_fo_product_round_trip_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = StandardSH().configure(
                filedir=directory,
                data=['2026-03-01', '2026-03-02'],
                degree=self.degree,
                c=self.c,
                s=self.s,
                sigmaC=self.sigma_c,
                sigmaS=self.sigma_s,
                mission='GRACE_FO_RL04',
                reference_model=_ReferenceModel(),
                unused_days=['2026-03-02', '2026-04-01']
            )
            output = writer.unfiltered()

            self.assertIsInstance(output, Path)
            self.assertEqual(output.parent.name, output.stem)
            self.assertEqual(
                output.name,
                'Hawk_3x3_unfiltered_GSM-2_2026-03-01-2026-03-02_'
                'GRFO_0100.gfc'
            )
            content = output.read_text(encoding='ascii')
            self.assertIn(
                'modelname                      '
                'GSM-2_2026-03-01-2026-03-02_GRFO_Hawk_BA01_0100',
                content
            )
            self.assertNotIn('UTCSR', content)
            self.assertIn(
                'dictionary            : '
                'https://github.com/NCSGgroup/PyHawk/blob/main/README.md',
                content
            )
            self.assertIn(
                'time_coverage_start   : 2026-03-01T00:00:00.00', content
            )
            self.assertIn(
                'time_coverage_end     : 2026-03-03T00:00:00.00', content
            )
            self.assertIn('unused_days           : [2026-03-02]', content)
            self.assertIn(
                'time_period_of_data    20260301 - 20260302   '
                '(mid: 20260302)', content
            )
            self.assertIn(
                'generating_institute    '
                'https://github.com/NCSGgroup', content
            )

            records = [line.split() for line in content.splitlines()
                       if line.startswith('gfc')]
            self.assertEqual(len(records), self.size)
            self.assertEqual(
                [(int(row[1]), int(row[2])) for row in records[:5]],
                [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1)]
            )

            loaded = LoadGif48().load(str(output))
            loaded_c, loaded_s = loaded.getCS(self.degree)
            loaded_sigma_c, loaded_sigma_s = loaded.getSigmaCS(self.degree)
            np.testing.assert_array_equal(loaded_c, self.c)
            np.testing.assert_array_equal(loaded_s, self.s)
            np.testing.assert_array_equal(loaded_sigma_c, self.sigma_c)
            np.testing.assert_array_equal(loaded_sigma_s, self.sigma_s)

    def test_grace_product_uses_grac(self):
        with tempfile.TemporaryDirectory() as directory:
            output = StandardSH().configure(
                filedir=directory,
                data=['2017-06-01', '2017-06-30'],
                degree=self.degree,
                c=self.c,
                s=self.s,
                sigmaC=self.sigma_c,
                sigmaS=self.sigma_s,
                mission='GRACE_RL03'
            ).unfiltered()
            self.assertEqual(
                output.name,
                'Hawk_3x3_unfiltered_GSM-2_2017-06-01-2017-06-30_'
                'GRAC_0100.gfc'
            )

    def test_product_filename_is_available_before_solution_write(self):
        self.assertEqual(
            StandardSH.product_filename(
                data=['2026-03-01', '2026-03-02'],
                degree=60,
                mission='GRACE_FO_RL04'
            ),
            'Hawk_60x60_unfiltered_GSM-2_2026-03-01-2026-03-02_'
            'GRFO_0100.gfc'
        )

    def test_rejects_incomplete_triangular_array(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'degree requires 10'):
                StandardSH().configure(
                    filedir=directory,
                    data=['2026-03-01', '2026-03-02'],
                    degree=self.degree,
                    c=self.c[:-1],
                    s=self.s,
                    sigmaC=self.sigma_c,
                    sigmaS=self.sigma_s
                )

    def test_fo_compatibility_writer_uses_grfo(self):
        with tempfile.TemporaryDirectory() as directory:
            output = FormatWrite_FO().configure(
                filedir=directory,
                data=['2026-03-01', '2026-03-02'],
                degree=self.degree,
                c=self.c,
                s=self.s,
                sigmaC=self.sigma_c,
                sigmaS=self.sigma_s
            ).unfiltered()
            self.assertIn('_GRFO_0100.gfc', output.name)

    def test_rejects_unsupported_mission(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'unsupported SH mission'):
                StandardSH().configure(
                    filedir=directory,
                    data=['2026-03-01', '2026-03-02'],
                    degree=self.degree,
                    c=self.c,
                    s=self.s,
                    sigmaC=self.sigma_c,
                    sigmaS=self.sigma_s,
                    mission='XX4_RL01'
                )


if __name__ == '__main__':
    unittest.main()
