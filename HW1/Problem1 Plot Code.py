import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------
#  CONSTANTS
# ---------------------------------------------------------
mu = 398600.0    # km^3 / s^2
RE = 6378.0      # km

# ---------------------------------------------------------
#  GIVEN INITIAL CONDITIONS
# ---------------------------------------------------------
r0 = np.array([3500.0, 6805.0, 2200.0])  # km
v0 = np.array([-7.511, 0.357, 4.447])    # km/s

# Initial magnitudes
r0_norm = np.linalg.norm(r0)
v0_norm = np.linalg.norm(v0)

# ---------------------------------------------------------
#  1) QUICK ORBITAL ELEMENTS CALC (ELLIPTICAL CASE)
# ---------------------------------------------------------
# Angular momentum
h_vec = np.cross(r0, v0)
h = np.linalg.norm(h_vec)

# Eccentricity vector
rdotv = np.dot(r0, v0)
e_vec = (1.0/mu) * ((v0_norm**2 - mu/r0_norm)*r0 - rdotv*v0)
e = np.linalg.norm(e_vec)

# Semi-major axis
energy = 0.5*(v0_norm**2) - mu/r0_norm
a = - mu / (2.0*energy)

# Apogee altitude
r_ap = a*(1 + e)
h_ap = r_ap - RE

print(">>> Semi-major axis a   = {:.2f} km".format(a))
print(">>> Eccentricity e      = {:.4f}".format(e))
print(">>> Apogee radius       = {:.2f} km".format(r_ap))
print(">>> Max altitude (km)   = {:.2f}".format(h_ap))

# Orbital period
T = 2.0*np.pi * np.sqrt(a**3 / mu)
T_hours = T / 3600.0
print(">>> Orbital period      = {:.2f} s = {:.2f} hr".format(T, T_hours))

# ---------------------------------------------------------
#  2) TWO-BODY ODE FUNCTION
# ---------------------------------------------------------
def two_body(t, state):
    rx, ry, rz, vx, vy, vz = state
    rr = np.sqrt(rx*rx + ry*ry + rz*rz)
    ax = -mu * rx / rr**3
    ay = -mu * ry / rr**3
    az = -mu * rz / rr**3
    return [vx, vy, vz, ax, ay, az]

# ---------------------------------------------------------
#  3) INTEGRATE FOR ~ 1 ORBIT (maybe 1.2*T)
# ---------------------------------------------------------
state0 = np.hstack((r0, v0))
t_span = (0, 1.2*T)  # from t=0 to about 1.2 orbits
t_eval = np.linspace(0, 1.2*T, 2000)  # times at which to store solution

sol = solve_ivp(two_body, t_span, state0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

# Unpack solution
t_arr = sol.t
rx_arr = sol.y[0,:]
ry_arr = sol.y[1,:]
rz_arr = sol.y[2,:]

# Altitude vs time
r_arr = np.sqrt(rx_arr**2 + ry_arr**2 + rz_arr**2)
h_arr = r_arr - RE

# Find maximum altitude and time
idx_max = np.argmax(h_arr)
h_max_num = h_arr[idx_max]
t_to_ap = t_arr[idx_max]

print(">>> Numerical max alt   = {:.2f} km".format(h_max_num))
print(">>> Time of max alt     = {:.2f} s = {:.2f} hr".format(t_to_ap, t_to_ap/3600.0))

# ---------------------------------------------------------
#  4) PLOTS
# ---------------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(t_arr/3600.0, h_arr, label='Altitude')
plt.plot((t_to_ap/3600.0,), (h_max_num,), 'ro', label='Max altitude')
plt.xlabel('Time since t0 (hours)')
plt.ylabel('Altitude above Earth (km)')
plt.title('Satellite Altitude vs. Time')
plt.grid(True)
plt.legend()
plt.show()

# Optional 3D plot of trajectory
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(rx_arr, ry_arr, rz_arr, 'b-')
ax.scatter(r0[0], r0[1], r0[2], color='red', label='Initial position')
# draw Earth (very rough)
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 30)
xe = RE * np.outer(np.cos(u), np.sin(v))
ye = RE * np.outer(np.sin(u), np.sin(v))
ze = RE * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(xe, ye, ze, color='c', alpha=0.3)

ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_zlabel('z (km)')
ax.set_title('Orbit in ECI Frame')
plt.legend()
plt.show()