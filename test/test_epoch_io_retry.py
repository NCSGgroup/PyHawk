import unittest
from unittest.mock import patch,MagicMock
from src.Solver.JointChannel.initial_weights import read_epoch_times
class RetryTests(unittest.TestCase):
 def test_retry_close_error(self):
  bad=MagicMock();bad.__exit__.side_effect=RuntimeError('errno = 5 Input/output error')
  good=MagicMock();good.__enter__.return_value.__getitem__.return_value.__getitem__.return_value=[1.,2.]
  with patch('src.Solver.JointChannel.initial_weights.h5py.File',side_effect=[bad,good]) as f,patch('time.sleep'):
   self.assertEqual(read_epoch_times('test.h5',8),[1.,2.]);self.assertEqual(f.call_count,2)
 def test_no_retry_for_non_io(self):
  with patch('src.Solver.JointChannel.initial_weights.h5py.File',side_effect=RuntimeError('bad format')) as f:
   with self.assertRaisesRegex(RuntimeError,'arc=8'):read_epoch_times('test.h5',8)
   self.assertEqual(f.call_count,1)
