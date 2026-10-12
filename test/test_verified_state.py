import tempfile,unittest
from pathlib import Path
import numpy as np
import h5py
from src.Solver.VerifiedState import save_verified_state
class Tests(unittest.TestCase):
 def test_roundtrip_and_reject_bad_state(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'state.h5';r=np.ones((70,6,3));r[:,3:,0]=2
   save_verified_state(p,(np.arange(70.),r,r))
   with h5py.File(p) as h:np.testing.assert_array_equal(h['r'][()],r)
   r[2,:,0]=0
   with self.assertRaisesRegex(ValueError,'in-memory'):save_verified_state(p,(np.arange(70.),r,r))
