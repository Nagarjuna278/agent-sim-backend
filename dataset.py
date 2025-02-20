import h5py
import random
def fetch_grid(index=random.randint(0, 1000000)):
    with h5py.File('valid_grids.h5', 'r') as f:
        dset = f['grids']
        return dset[index//100]