import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

steps = 1000
t = np.linspace(0, 10, steps)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set(xlim=[-8, 8], ylim=[-8, 8], zlim=[0, 8])
psi = np.linspace(0, 20 * np.pi, 1000)

y = 3.5 * np.cos(psi)
z = 3.5 * np.sin(psi)

Z_PointCentral = 8.5
Y_PointCentral = 0
X_PointCentral = 0

PointCentral = ax.plot(X_PointCentral, Y_PointCentral, Z_PointCentral, color='blue', marker='o', markeredgewidth=1)[0]

Z_PointM = 5
X_PointM = 0
Y_PointM = 0

circle = ax.plot(y * np.cos(np.pi), y * np.sin(np.pi), z + 5, linewidth=5, color='green', alpha=.2)[0]
PointM = ax.plot(X_PointM, Y_PointM, Z_PointM, color='orange', marker='o', markeredgewidth=4)[0]


def get_spring_xyz(coils, diameter, start, end):
    x = np.linspace(start[0], end[0], coils * 2)
    y = np.linspace(start[1], end[1], coils * 2)
    z = np.linspace(start[2], end[2], coils * 2)

    for i in range(1, len(z) - 1):
        z[i] = z[i] + diameter * pow(-1, i)

    return np.array([x, y, z])


# shaft
ax.plot([0, 0], [0, 0], [0, 10.80], linewidth=2, color='black', alpha=.8)

# spring coordinates
spring_xyz = get_spring_xyz(30, 0.1, [0, 0, Z_PointCentral], [1, 3, 2])
spring = ax.plot(spring_xyz[0], spring_xyz[1], spring_xyz[2], linewidth=2, color='black')[0]


def animation(i):
    circle.set_data_3d(y * np.cos(psi[i]), y * np.sin(psi[i]), z + 5)
    current_z = (Z_PointM + 3.5 * np.cos(psi[i]))

    if current_z >= 5:
        current_z = current_z - 2 * (current_z - 5)

    new_x = (X_PointM + 3.5 * np.sin(psi[i])) * np.cos(psi[i])
    new_y = (Y_PointM + 3.5 * np.sin(psi[i])) * np.sin(psi[i])
    new_z = current_z

    PointM.set_data_3d(new_x, new_y, new_z)

    spring_coordinates = get_spring_xyz(30, 0.2, [0, 0, Z_PointCentral], [new_x, new_y, new_z])
    spring.set_data_3d(spring_coordinates[0], spring_coordinates[1], spring_coordinates[2])

    return [circle, PointM, spring]


anima = FuncAnimation(fig, animation, frames=500, interval=20)
plt.show()
