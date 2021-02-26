import matplotlib.pyplot as plt
import numpy as np
from random import randint


inner_circles_points = 200
outer_circles_points = 50

circle_no2_rad = 0.1
circle_no3_rad = 0.2
circle_no4_rad = 0.3

circle_thickness = 0.1


# draw the first circle
r1 = np.random.rand(outer_circles_points)*circle_thickness
a1 = np.random.rand(outer_circles_points)*np.pi
r2 = np.random.rand(outer_circles_points)*circle_thickness
a2 = np.random.rand(outer_circles_points)*np.pi+np.pi

p1 = np.array((r1*np.cos(a1), r1*np.sin(a1)))
p2 = np.array((r2*np.cos(a2), r2*np.sin(a2)))

x1, y1 = (p1[0], p1[1])
x2, y2 = (p2[0], p2[1])

x1 = np.concatenate((x1, x2))
y1 = np.concatenate((y1, y2))
c1 = np.zeros(outer_circles_points*2)


# draw the second circle
r3 = np.random.rand(inner_circles_points)*circle_thickness+circle_no2_rad
a3 = np.random.rand(inner_circles_points)*np.pi

r4 = np.random.rand(inner_circles_points)*circle_thickness+circle_no2_rad
a4 = np.random.rand(inner_circles_points)*np.pi+np.pi

p3 = np.array((r3*np.cos(a3), r3*np.sin(a3)))
p4 = np.array((r4*np.cos(a4), r4*np.sin(a4)))

x3, y3 = (p3[0], p3[1])
x4, y4 = (p4[0], p4[1])

x3 = np.concatenate((x3, x4))
y3 = np.concatenate((y3, y4))
c3 = np.ones(inner_circles_points*2)


# draw the third circle
r5 = np.random.rand(inner_circles_points)*circle_thickness+circle_no3_rad
a5 = np.random.rand(inner_circles_points)*np.pi

r6 = np.random.rand(inner_circles_points)*circle_thickness+circle_no3_rad
a6 = np.random.rand(inner_circles_points)*np.pi+np.pi

p5 = np.array((r5*np.cos(a5), r5*np.sin(a5)))
p6 = np.array((r6*np.cos(a6), r6*np.sin(a6)))

x5, y5 = (p5[0], p5[1])
x6, y6 = (p6[0], p6[1])

x5 = np.concatenate((x5, x6))
y5 = np.concatenate((y5, y6))
c5 = np.zeros(inner_circles_points*2)

# draw the fourth circle
r7 = np.random.rand(outer_circles_points)*circle_thickness+circle_no4_rad
a7 = np.random.rand(outer_circles_points)*np.pi

r8 = np.random.rand(outer_circles_points)*circle_thickness+circle_no4_rad
a8 = np.random.rand(outer_circles_points)*np.pi+np.pi

p7 = np.array((r7*np.cos(a7), r7*np.sin(a7)))
p8 = np.array((r8*np.cos(a8), r8*np.sin(a8)))

x7, y7 = (p7[0], p7[1])
x8, y8 = (p8[0], p8[1])

x7 = np.concatenate((x7, x8))
y7 = np.concatenate((y7, y8))
c7 = np.ones(outer_circles_points*2)


# join all points
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

# plot the data
plt.scatter(x, y,c=c)
plt.show()


# save the data to the data.csv file
np.savetxt("data.csv", points, delimiter=",")
