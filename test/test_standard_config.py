# -*- coding: utf-8 -*-
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.Auxilary.StandardConfig import StandardConfig
from src.Auxilary.StandardSH import StandardSH
from src.Solver.CalibrateOrbit import CalibrateOrbit


class TestStandardConfig(unittest.TestCase):
    def test_load_json_immediately_archives_selected_setting_directory(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'setting' / 'Calibrate'
            shutil.copytree(repository / 'setting' / 'Calibrate', source)

            neq_path = source / 'NEQConfig.json'
            neq = json.loads(neq_path.read_text(encoding='utf-8'))
            neq['PathOfFilesConfig']['ResultCS'] = str(root / 'result')
            neq_path.write_text(
                json.dumps(neq, indent=4) + '\n', encoding='utf-8'
            )

            interface = json.loads(
                (source / 'InterfaceConfig.json').read_text(encoding='utf-8')
            )
            gravity = json.loads(
                (source / 'GravityDesignParameterConfig.json').read_text(
                    encoding='utf-8'
                )
            )
            filename = StandardSH.product_filename(
                data=interface['date_span'],
                degree=gravity['StokesCoefficientsConfig']['MaxDegree'],
                mission=interface['mission']
            )

            CalibrateOrbit().loadJson(setting_directory=source)

            output = (
                root / 'result' / Path(filename).stem
                / (Path(filename).stem + '_setting')
            )
            for setting in source.glob('*.json'):
                self.assertEqual(
                    (output / setting.name).read_bytes(),
                    setting.read_bytes()
                )

    def test_json_settings_are_preserved_in_reusable_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'setting' / 'Calibrate'
            source.mkdir(parents=True)
            first = b'{\n    "value": 1\n}\n'
            second = b'{"mode":"EstimatedFull"}\n'
            (source / 'FirstConfig.json').write_bytes(first)
            (source / 'SecondConfig.json').write_bytes(second)
            (source / 'ignored.txt').write_text(
                'not a JSON setting', encoding='ascii'
            )

            product = root / 'Hawk_60x60_unfiltered_test.gfc'
            output = StandardConfig(source).save(product)

            self.assertEqual(
                output,
                root / 'Hawk_60x60_unfiltered_test_setting'
            )
            self.assertEqual(
                (output / 'FirstConfig.json').read_bytes(), first
            )
            self.assertEqual(
                (output / 'SecondConfig.json').read_bytes(), second
            )
            self.assertFalse((output / 'ignored.txt').exists())

    def test_stale_json_is_removed_when_snapshot_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'setting' / 'Calibrate'
            source.mkdir(parents=True)
            (source / 'CurrentConfig.json').write_text(
                '{}\n', encoding='ascii'
            )
            product = root / 'product.gfc'
            output = (
                root / 'product_setting'
            )
            output.mkdir(parents=True)
            (output / 'StaleConfig.json').write_text(
                '{}\n', encoding='ascii'
            )

            StandardConfig(source).save(product)

            self.assertFalse((output / 'StaleConfig.json').exists())
            self.assertTrue((output / 'CurrentConfig.json').exists())

    def test_arc_files_are_archived_beside_product(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'setting' / 'Calibrate'
            source.mkdir(parents=True)
            (source / 'Config.json').write_text('{}\n', encoding='ascii')
            arcft = root / 'source_arcft.txt'
            arcft.write_text(
                'Arc NO: 0 True ----- ;\n', encoding='ascii'
            )
            arc = root / 'source_arc.txt'
            arc.write_text(
                'Arc NO: 0; 0 ==> 1; Length: 2\n', encoding='ascii'
            )
            product = root / 'product.gfc'

            outputs = StandardConfig(source).archive_quality_control(
                spherical_harmonic_path=product,
                arc_filter_path=arcft,
                arc_definition_path=arc
            )

            self.assertEqual(
                outputs['arc_filter'], root / 'product_arcft.txt'
            )
            self.assertEqual(
                outputs['arc_definition'], root / 'product_arc.txt'
            )
            self.assertEqual(
                outputs['arc_filter'].read_bytes(), arcft.read_bytes()
            )
            self.assertEqual(
                outputs['arc_definition'].read_bytes(), arc.read_bytes()
            )


if __name__ == '__main__':
    unittest.main()
