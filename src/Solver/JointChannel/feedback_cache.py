"""Reuse valid feedback integrations without reusing incomplete state groups."""
from pathlib import Path
import os
import h5py
import numpy as np

def reuse_integration(settings, arc, cache, root):
    folder=Path(cache)/'DesignMatrix'/settings.tag
    names=['%d_AB.hdf5'%arc,'Orbit_%d_AB.hdf5'%arc,'RangeRate_%d.hdf5'%arc]
    if not all((folder/n).is_file() for n in names):return None
    try:
        with h5py.File(folder/names[1],'r') as h:
            t,r,v=(h[k][()] for k in ('t','r','v'))
            if not (np.isfinite(r).all() and np.isfinite(v).all() and np.isfinite(t).all()):return None
            if np.any(np.diff(t)<=0) or np.any(np.linalg.norm(r.reshape(-1,2,3),axis=2)<=0):return None
            if np.any(np.linalg.norm(r[:,:3]-r[:,3:],axis=1)<=0):return None
        # Verify all cached derivative values, including the raw integration state.
        for name in names:
            with h5py.File(folder/name,'r') as h:
                if not np.array_equal(h['t'][()],t):return None
                for key in h:
                    for start in range(0,len(h[key]),32):
                        if not np.isfinite(h[key][start:start+32]).all():return None
    except (OSError,ValueError,KeyError):return None
    dest=Path(root)/'DesignMatrix'/settings.tag;dest.mkdir(parents=True,exist_ok=True)
    for name in names:os.link(str(folder/name),str(dest/name))
    return t,r,v
