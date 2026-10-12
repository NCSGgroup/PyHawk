"""Validate physical integration states and verify disk round trips before publication."""
import os
from pathlib import Path
import h5py
import numpy as np


def save_verified_state(filename, state, attempts=2):
    arrays = dict(zip(('t','r','v'), state))
    r = np.asarray(arrays['r'])
    physical = r[:,:,0] if r.ndim == 3 else r
    if not all(np.isfinite(x).all() for x in arrays.values()):
        raise ValueError('non-finite in-memory integration state: '+str(filename))
    if physical.shape[1] != 6 or np.any(np.linalg.norm(physical.reshape(-1,2,3),axis=2)<=0) or np.any(np.linalg.norm(physical[:,:3]-physical[:,3:],axis=1)<=0):
        raise ValueError('invalid in-memory satellite separation: '+str(filename))
    target=Path(filename);tmp=target.with_name(target.name+'.writing-'+str(os.getpid()))
    for attempt in range(attempts):
        with h5py.File(tmp,'w') as h:
            for key,value in arrays.items():h.create_dataset(key,data=value)
            h.flush()
        mismatch=None
        with h5py.File(tmp,'r') as h:
            for key,value in arrays.items():
                for start in range(0,len(value),32):
                    if not np.array_equal(h[key][start:start+32],value[start:start+32]):
                        mismatch=(key,start);break
                if mismatch:break
        if mismatch is None:
            os.replace(str(tmp),str(target));return
        print('Integration disk round-trip mismatch: %s %s; attempt %d/%d' % (filename,mismatch,attempt+1,attempts),flush=True)
    raise IOError('integration file differs from in-memory state: %s %s' % (filename,mismatch))
