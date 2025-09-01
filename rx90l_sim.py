import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def dh(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    T = np.array([[ct, -st*ca,  st*sa, a*ct],
                  [st,  ct*ca, -ct*sa, a*st],
                  [0.0,    sa,     ca,    d],
                  [0.0,   0.0,    0.0,  1.0]], dtype=float)
    return T

def fk_chain(dh_rows, q):
    T = np.eye(4)
    pts = [T[:3, 3].copy()]
    for (a, alpha, d, theta0, is_rev), qi in zip(dh_rows, q):
        theta = theta0 + (qi if is_rev else 0.0)
        T = T @ dh(a, alpha, d, theta)
        pts.append(T[:3, 3].copy())
    return np.array(pts)

# --- Replace with your RX-90L DH (meters/radians) ---
DH = [
    (0.000,        -np.pi/2, 0.350, 0.0,          True),
    (0.450,         0.0,     0.000, -np.pi/2,     True),
    (0.050,        -np.pi/2, 0.000,  np.pi/2,     True),
    (0.425,         np.pi/2, 0.000,  0.0,         True),
    (0.000,        -np.pi/2, 0.000,  0.0,         True),
    (0.000,         0.0,     0.100,  0.0,         True),
]

q_lim = np.deg2rad(np.array([[-160, 160],
                             [-137.5, 137.5],
                             [-142.5, 142.5],
                             [-270, 270],
                             [-105, 120],
                             [-270, 270]], dtype=float))

T_final, fps = 10.0, 30
N = int(T_final * fps)
t = np.linspace(0, T_final, N)

amp = 0.4 * (q_lim[:,1] - q_lim[:,0]) / 2.0
center = (q_lim[:,1] + q_lim[:,0]) / 2.0
freqs = np.array([0.2, 0.31, 0.17, 0.27, 0.23, 0.29])
phases = np.linspace(0, np.pi, 6)
q_traj = np.zeros((N, 6))
for j in range(6):
    q_traj[:, j] = center[j] + amp[j] * np.sin(2*np.pi*freqs[j]*t + phases[j])

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
reach = 1.3
ax.set_xlim([-reach, reach]); ax.set_ylim([-reach, reach]); ax.set_zlim([0.0, reach*1.2])
ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
ax.set_title('Stäubli RX‑90L (approx) – forward kinematics demo')
link_lines = []; joint_scatter = ax.scatter([], [], [], s=20); tcp_trail, = ax.plot([], [], [], lw=1, alpha=0.5)
for _ in range(6):
    line, = ax.plot([], [], [], lw=3); link_lines.append(line)
trail_pts = []

def init():
    for line in link_lines:
        line.set_data([], []); line.set_3d_properties([])
    tcp_trail.set_data([], []); tcp_trail.set_3d_properties([])
    return link_lines + [tcp_trail, joint_scatter]

def update(frame):
    q = q_traj[frame]
    pts = fk_chain(DH, q)
    xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
    for i, line in enumerate(link_lines):
        line.set_data(xs[i:i+2], ys[i:i+2]); line.set_3d_properties(zs[i:i+2])
    joint_scatter._offsets3d = (xs[:-1], ys[:-1], zs[:-1])
    trail_pts.append(pts[-1]); tp = np.array(trail_pts[-300:])
    tcp_trail.set_data(tp[:,0], tp[:,1]); tcp_trail.set_3d_properties(tp[:,2])
    ax.view_init(elev=25, azim=35 + 0.2*frame)
    return link_lines + [tcp_trail, joint_scatter]

anim = FuncAnimation(fig, update, frames=N, init_func=init, interval=1000/fps, blit=False)
plt.show()
