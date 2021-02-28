import matplotlib.pyplot as plt
import numpy as np


inner_circles_points = 400
outer_circles_points = 100

circle_no2_rad = 0.1
circle_no3_rad = 0.2
circle_no4_rad = 0.3

circle_thickness = 0.1

x = np.array([])
y = np.array([])
c = np.array([])


# draw the first circle
r1 = np.random.rand(outer_circles_points)*circle_thickness
a1 = np.random.rand(outer_circles_points)*np.pi*2

x = np.concatenate((x, r1*np.cos(a1)))
y = np.concatenate((y, r1*np.sin(a1)))
c = np.concatenate((c, np.zeros(outer_circles_points)))

# draw the second circle
r2 = np.random.rand(inner_circles_points)*circle_thickness+circle_no2_rad
a2 = np.random.rand(inner_circles_points)*np.pi*2

x = np.concatenate((x, r2*np.cos(a2)))
y = np.concatenate((y, r2*np.sin(a2)))
c = np.concatenate((c, np.ones(inner_circles_points)))

# draw the third circle
r3 = np.random.rand(inner_circles_points)*circle_thickness+circle_no3_rad
a3 = np.random.rand(inner_circles_points)*np.pi*2

x = np.concatenate((x, r3*np.cos(a3)))
y = np.concatenate((y, r3*np.sin(a3)))
c = np.concatenate((c, np.zeros(inner_circles_points)))

# draw the fourth circle
r4 = np.random.rand(outer_circles_points)*circle_thickness+circle_no4_rad
a4 = np.random.rand(outer_circles_points)*np.pi*2

x = np.concatenate((x, r4*np.cos(a4)))
y = np.concatenate((y, r4*np.sin(a4)))
c = np.concatenate((c, np.ones(outer_circles_points)))

plt.scatter(x, y,c=c)
plt.show()


points = []
for i in range(len(x)):
    points.append([x[i],y[i],c[i]])


# save the data to the data.csv file
np.savetxt("data.csv", points, delimiter=",")
