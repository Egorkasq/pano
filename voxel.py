from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import time

import voxelization



def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x



voxelization.voxelization('odm_mesh.ply')
'''
v = ''
with open('odm_textured_model.obj', 'r') as f:
    for i in f:
        if i[0] == 'v' and i[1] == ' ':
            v = v + str(i)
v = v.split('\n')
k = []
for i in v:
    temp = i.split(' ')
    k.append([int(float(i) * 10) for i in temp[1:]])


x = []
y = []
z = []

for i in k[:-1]:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])
print(min(z), max(z))
print(min(x), max(x))
print(min(y), max(y))
sizeZ = abs(max(z)) - abs(min(z))
sizeX = abs(min(x)) + abs(max(x))
sizeY = abs(min(y)) + abs(max(y))
print(sizeX, sizeY, sizeZ)
# prepare some coordinates, and attach rgb values to each
'''
r, g, b = np.indices((200, 200, 50)) / 10.0  # ((размеры х, у, з)) / точность построения
#r, g, b = np.indices((sizeX, sizeY, sizeZ)) / 5.0  # ((размеры х, у, з)) / точность построения

rc = midpoints(r)
gc = midpoints(g)
bc = midpoints(b)

# sphere = (rc - 0.5) ** 2 + (gc - 0.5) ** 2 + (bc - 0.5) ** 2 < 0.5 ** 2
sphere = (rc - 0.5) ** 2 + (gc - 0.5) ** 2 + (bc - 0.5) ** 2 < 0.5 ** 2

print(sphere.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(r, g, b, sphere, linewidth=0.5)
ax.set(xlabel='r', ylabel='g', zlabel='b')
plt.show()

