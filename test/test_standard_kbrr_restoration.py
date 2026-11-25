import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.Auxilary.StandardKBRR import StandardKBRR

class TestRestoredStandardKBRR(unittest.TestCase):
    def write(self, directory, omit=None, mismatch=False):
        span=['2020-05-01','2020-05-01']
        t=np.array([641563200.,641563205.])
        writer=StandardKBRR().configure(directory,span,'Post').prepare()
        for stage,value in [('Raw',1),('Pre',2),('Post',3),('Geofit',4),('Background',5),('Geofitstatic',9)]:
            if stage==omit: continue
            epochs=t+5 if mismatch and stage=='Background' else t
            StandardKBRR().configure(directory,span,stage).save_arc(
                epochs,np.zeros(2),np.zeros(2),np.ones(2),np.full(2,value*1e-8),0)
        return writer

    def test_eight_columns_with_restored_geofit(self):
        with tempfile.TemporaryDirectory() as d:
            output=self.write(d).finalize()[0]
            content=Path(output).read_text()
            self.assertNotIn('    - kbrr_background_timevar:',content)
            self.assertIn('comment: 8th column',content)
            self.assertIn('Column 7 plus dated-minus-static',content)
            self.assertIn('    - geofit_kbrr_static:',content)
            rows=np.array([[float(x) for x in line.split()] for line in content.split('# End of YAML header\n')[1].strip().splitlines()])
            self.assertEqual(rows.shape,(2,8))
            np.testing.assert_allclose(rows[:,4:],np.tile(np.array([2,3,4,9])*1e-8,(2,1)))
            np.testing.assert_allclose(rows[:,7],rows[:,6]+5e-8)

    def test_missing_restored_component_fails(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'missing'):
                self.write(d,omit='Geofitstatic').finalize()

    def test_mismatched_background_epochs_fail(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'epoch mismatch'):
                self.write(d,mismatch=True).finalize()

if __name__=='__main__': unittest.main()
