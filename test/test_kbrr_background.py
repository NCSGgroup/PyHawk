import unittest
from unittest.mock import patch
import numpy as np
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Solver.KBRRBackground import restore_kbrr_background


class TestKBRRBackground(unittest.TestCase):
    def config(self, epoch=True):
        f = ForceModelConfig()
        f.RefGravModelConfig.update(StaticModel='GOCO06S', TimeVariableDate=epoch, Nmax=2)
        f.PathOfFilesConfig['GOCO06s'] = 'mock.gfc'
        return f

    def restore(self, f, design=None, geofit=None):
        return restore_kbrr_background(
            np.eye(5) if design is None else design,
            np.arange(5.)+10 if geofit is None else geofit, f,
            dict(MaxDegree=2, MinDegree=2, SortMethod='Normal'),
            ['2020-05-01', '2020-05-31'])

    @patch('src.Solver.KBRRBackground.LoadGOCO06S')
    def test_dated_response_and_cumulative_added_once(self, loader):
        class Field:
            maxDegree = 2
            def __init__(self, date): self.date = date
            def getCS(self, degree):
                if self.date is None: return np.zeros(6), np.zeros(6)
                return np.array([0,0,0,1,2,3.]), np.array([0,0,0,0,4,5.])
        loader.return_value.load.side_effect = lambda path, date: Field(date)
        force = self.config()
        data, attrs = self.restore(force)
        np.testing.assert_array_equal(data['kbrr_background_timevar'], [1,2,3,4,5])
        np.testing.assert_array_equal(data['geofit_kbrr_static'], [11,13,15,17,19])
        self.assertEqual(attrs['kbrr_background_reference_epoch'], '2020-05')
        self.assertIs(force.RefGravModelConfig['TimeVariableDate'], True)

    @patch('src.Solver.KBRRBackground.LoadGOCO06S')
    def test_static_reference_is_noop(self, loader):
        data, _ = self.restore(self.config(False))
        loader.assert_not_called()
        np.testing.assert_array_equal(data['kbrr_background_timevar'], np.zeros(5))
        np.testing.assert_array_equal(data['geofit_kbrr_static'], np.arange(5.)+10)

    def test_rejects_unsliced_scale_columns(self):
        with self.assertRaises(ValueError):
            self.restore(self.config(False), design=np.zeros((5,23)))

    def test_rejects_nonfinite_input(self):
        with self.assertRaises(ValueError):
            self.restore(self.config(False), geofit=np.full(5,np.nan))

    def test_empty_arc(self):
        data, _ = self.restore(self.config(False), np.zeros((0,5)), np.zeros(0))
        self.assertEqual(data['geofit_kbrr_static'].shape, (0,))

if __name__ == '__main__':
    unittest.main()
