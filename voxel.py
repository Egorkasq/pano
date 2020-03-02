import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData, PlyElement

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

x = []
y = []
z = []
plydata = PlyData.read('odm_mesh.ply')

for i in plydata.elements[0].data:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])


new_x = []
new_y = []
new_z = []


for i in range(len(plydata.elements[0].data)):
    plydata.elements[0].data[i][0] = int(plydata.elements[0].data[i][0] + abs(min(x)))
    plydata.elements[0].data[i][1] = int(plydata.elements[0].data[i][1] + abs(min(y)))
    plydata.elements[0].data[i][2] = int(plydata.elements[0].data[i][2] - min(z))

print(plydata.elements[0].data)

new = np.zeros((420, 420, 100), dtype=bool)

for i in plydata.elements[0]:
    new[int(i[0]), int(i[1]), int(i[2])] = True

# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(new)
ax.set(xlabel='r', ylabel='g', zlabel='b')

plt.show()
