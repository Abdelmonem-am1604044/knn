import matplotlib.pyplot as plt
import numpy as np
from random import randint
import math


n_data_points = 200

rad = 0.1
rad2 = 0.2
rad3 = 0.3
thk = 0.1
sep = 0.1



# We use random radius in the interval [rad, rad+thk]
#  and random angles from 0 to pi radians.
r1 = np.random.rand(50)*thk
a1 = np.random.rand(50)*np.pi
r2 = np.random.rand(50)*thk
a2 = np.random.rand(50)*np.pi+np.pi

# In order to plot it we convert it to cartesian:
p1 = np.array((r1*np.cos(a1), r1*np.sin(a1)))
p2 = np.array((r2*np.cos(a2), r2*np.sin(a2)))

x1, y1 = (p1[0], p1[1])
x2, y2 = (p2[0], p2[1])

x1 = np.concatenate((x1, x2))
y1 = np.concatenate((y1, y2))
c1 = np.zeros(50*2)
# We use random radius in the interval [rad, rad+thk]
#  and random angles from 0 to pi radians.
r3 = np.random.rand(n_data_points)*thk+rad
a3 = np.random.rand(n_data_points)*np.pi

r4 = np.random.rand(n_data_points)*thk+rad
a4 = np.random.rand(n_data_points)*np.pi+np.pi

# In order to plot it we convert it to cartesian:
p3 = np.array((r3*np.cos(a3), r3*np.sin(a3)))
p4 = np.array((r4*np.cos(a4), r4*np.sin(a4)))

x3, y3 = (p3[0], p3[1])
x4, y4 = (p4[0], p4[1])

x3 = np.concatenate((x3, x4))
y3 = np.concatenate((y3, y4))
c3 = np.ones(n_data_points*2)

# We use random radius in the interval [rad, rad+thk]
#  and random angles from 0 to pi radians.
r5 = np.random.rand(n_data_points)*thk+rad2
a5 = np.random.rand(n_data_points)*np.pi

r6 = np.random.rand(n_data_points)*thk+rad2
a6 = np.random.rand(n_data_points)*np.pi+np.pi

# In order to plot it we convert it to cartesian:
p5 = np.array((r5*np.cos(a5), r5*np.sin(a5)))
p6 = np.array((r6*np.cos(a6), r6*np.sin(a6)))

x5, y5 = (p5[0], p5[1])
x6, y6 = (p6[0], p6[1])

x5 = np.concatenate((x5, x6))
y5 = np.concatenate((y5, y6))
c5 = np.zeros(n_data_points*2)

r7 = np.random.rand(50)*thk+rad3
a7 = np.random.rand(50)*np.pi

r8 = np.random.rand(50)*thk+rad3
a8 = np.random.rand(50)*np.pi+np.pi

# In order to plot it we convert it to cartesian:
p7 = np.array((r7*np.cos(a7), r7*np.sin(a7)))
p8 = np.array((r8*np.cos(a8), r8*np.sin(a8)))

x7, y7 = (p7[0], p7[1])
x8, y8 = (p8[0], p8[1])

x7 = np.concatenate((x7, x8))
y7 = np.concatenate((y7, y8))
c7 = np.ones(50*2)

x = []
y = []
c = []

x = np.concatenate((x,x1))
x = np.concatenate((x,x3))
x = np.concatenate((x,x5))
x = np.concatenate((x,x7))

y = np.concatenate((y,y1))
y = np.concatenate((y,y3))
y = np.concatenate((y,y5))
y = np.concatenate((y,y7))

c = np.concatenate((c,c1))
c = np.concatenate((c,c3))
c = np.concatenate((c,c5))
c = np.concatenate((c,c7))

points = []
for i in range(len(x)):
    points.append([x[i],y[i],c[i]])

plt.scatter(x, y,c=c)

np.savetxt("data.csv", points, delimiter=",")

plt.show()