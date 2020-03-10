import numpy as np
import h5py
f = h5py.File('peppers_0503-1332.mat', 'r')
data = f.get('data/variable1')
data = np.array(data)
print(type(data), data)