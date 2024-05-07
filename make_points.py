import os
import numpy as np
from numpy import pi,cos,sin

###-------------------------------------------------------------------###

def make_measurement_points(m,n,r):
    angles = []
    for i in range(m):
        T = 2*pi/m
        angles.append(0+i*T)
        angles.append(pi/m+i*T)
        angles.append(pi/(2*m)+i*T)
        angles.append(3*pi/(2*m)+i*T)
    angles = np.asarray(angles)-np.ones(len(angles))*pi
    count = 0
    z = np.linspace(-1,1,n)
    pos = np.zeros((n*len(angles)*len(r),3))
    for i in range(n):
        for k in range(len(r)):
            for j in range(len(angles)):
                pos[count][0] = r[k]*cos(angles[j])
                pos[count][1] = r[k]*sin(angles[j])
                pos[count][2] = z[i]
                count += 1
    return pos

m = 5
n = 8
r = [1,0.75,0.5,0.25]

points = make_measurement_points(m,n,r)

filename = "testpos.csv"

with open(filename,'w+') as f:
    f.write('x,y,z\n')
    for i in range(len(points)):
        f.write('{},{},{}\n'.format(points[i][0],points[i][1],points[i][2]))
f.close()
