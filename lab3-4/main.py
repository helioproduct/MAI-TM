import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import sympy as sp
import math


def formY(y, t, fV, fOm):
    y1, y2, y3, y4 = y
    dYdT = [y3, y4, fV(y1, y2, y3, y4), fOm(y1, y2, y3, y4)]
    return dYdT


# defining parameters
alpha = math.pi / 15
M = 1
m = 2
R = 0.3
c = 20
l0 = 1
g = 9.81

# start
y0 = [0, np.pi / 15, -0.5, 0]

# defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')

# phi, ksi, v_phi = d(phi) / dt and v_psi = d(psi) / dt as functions of t
phi = sp.Function('phi')(t)
psi = sp.Function('psi')(t)
v_phi = sp.Function('v_phi')(t)
v_psi = sp.Function('v_psi')(t)
length = 2 * R * sp.cos(phi)

# 1 defining the kinetic energy
TT1 = M * R ** 2 * v_phi ** 2 / 2
V1 = 2 * v_psi * R
V2 = v_phi * R * sp.sin(2 * psi)
Vr2 = V1 ** 2 + V2 ** 2
TT2 = m * Vr2 / 2
TT = TT1 + TT2

# 2 defining the potential energy
Pi1 = 2 * R * m * g * sp.sin(psi) ** 2
Pi2 = (c * (length - l0) ** 2) / 2
Pi = Pi1 + Pi2

# 3 Not potential force
M = alpha * phi ** 2

# Lagrange function
L = TT - Pi

# equations
ur1 = sp.diff(sp.diff(L, v_phi), t) - sp.diff(L, phi) - M
ur2 = sp.diff(sp.diff(L, v_psi), t) - sp.diff(L, psi)

a11 = ur1.coeff(sp.diff(v_phi, t), 1)
a12 = ur1.coeff(sp.diff(v_psi, t), 1)
a21 = ur2.coeff(sp.diff(v_phi, t), 1)
a22 = ur2.coeff(sp.diff(v_psi, t), 1)
b1 = -(ur1.coeff(sp.diff(v_phi, t), 0)).coeff(sp.diff(v_psi, t), 0).subs(
    [(sp.diff(phi, t), v_phi), (sp.diff(psi, t), v_psi)])
b2 = -(ur2.coeff(sp.diff(v_phi, t), 0)).coeff(sp.diff(v_psi, t), 0).subs(
    [(sp.diff(phi, t), v_phi), (sp.diff(psi, t), v_psi)])

detA = a11 * a22 - a12 * a21
detA1 = b1 * a22 - b2 * a21
detA2 = a11 * b2 - b1 * a21

# Constructing the system of differential equations
T = np.linspace(0, 25, 2000)
fv_phi = sp.lambdify([phi, psi, v_phi, v_psi], detA1 / detA, "numpy")
fv_psi = sp.lambdify([phi, psi, v_phi, v_psi], detA2 / detA, "numpy")
sol = odeint(formY, y0, T, args=(fv_phi, fv_psi))

t = sp.Symbol('t')
phi = sol[:, 0]
psi = sol[:, 1]
v_phi = sol[:, 2]
v_psi = sol[:, 3]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set(xlim=[-8, 8], ylim=[-8, 8], zlim=[0, 8])
y = 3.5 * np.cos(psi)
z = 3.5 * np.sin(psi)
Z_PointCentral = 8.5
Y_PointCentral = 0
X_PointCentral = 0
PointCentral = ax.plot(X_PointCentral, Y_PointCentral, Z_PointCentral, color='blue', marker='o', markeredgewidth=1)[0]

Z_PointM = 5
X_PointM = 0
Y_PointM = 0
circle = ax.plot(y * np.cos(phi), y * np.sin(phi), z + 5, linewidth=5, color='green', alpha=.2)[0]
PointM = ax.plot(X_PointM, Y_PointM, Z_PointM, color='orange', marker='o', markeredgewidth=4)[0]


def get_spring_line(coils, diameter, start, end):
    x = np.linspace(start[0], end[0], coils * 2)
    y = np.linspace(start[1], end[1], coils * 2)
    z = np.linspace(start[2], end[2], coils * 2)

    for i in range(1, len(z) - 1):
        z[i] = z[i] + diameter * 1 * (-1) ** i

    return np.array([x, y, z], dtype=object)


ax.plot([0, 0], [0, 0], [0, 10.80], linewidth=2, color='black', alpha=.8)  # stick

# spring
spring_xyz = get_spring_line(30, 0.1, [0, 0, 8.5], [1, 3, 2])
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

    sprint_coordinates = get_spring_line(30, 0.2, [0, 0, 8.5], [new_x, new_y, new_z])
    spring.set_data_3d(sprint_coordinates[0], sprint_coordinates[1], sprint_coordinates[2])

    return [circle, PointM, spring]


anima = FuncAnimation(fig, animation, frames=500, interval=20)
plt.show()
