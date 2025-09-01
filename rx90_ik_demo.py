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

def fk_Ts(dh_rows, q):
    T = np.eye(4)
    Ts = [T.copy()]
    for (a, alpha, d, theta0, is_rev), qi in zip(dh_rows, q):
        theta = theta0 + (qi if is_rev else 0.0)
        T = T @ dh(a, alpha, d, theta)
        Ts.append(T.copy())
    pts = np.array([Ti[:3, 3] for Ti in Ts])
    return pts, Ts

def approach_dir_from_T(T):
    return T[:3, :3] @ np.array([0.0, 0.0, 1.0])

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

def numerical_jacobian(f, q, eps=1e-6):
    q = q.copy()
    y0 = f(q)
    m = y0.size
    n = q.size
    J = np.zeros((m, n))
    for i in range(n):
        qi = q.copy(); qi[i] += eps
        yi = f(qi)
        J[:, i] = (yi - y0) / eps
    return y0, J

def ik_xy_dir(DH, q0, target_xy, target_dir, d6=0.0, iters=100, lam=1e-2):
    td = np.asarray(target_dir, float)
    nrm = np.linalg.norm(td)
    td = td if nrm < 1e-12 else td / nrm

    def error_fn(q):
        _, Ts = fk_Ts(DH, q)
        Te = Ts[-1].copy()
        Te[:3, 3] += Te[:3, :3] @ np.array([0.0, 0.0, d6])
        p = Te[:3, 3]
        a = approach_dir_from_T(Te)
        a_n = a / (np.linalg.norm(a) + 1e-12)
        ex = p[0] - target_xy[0]
        ey = p[1] - target_xy[1]
        c = np.cross(a_n, td)
        return np.array([ex, ey, c[0], c[1], c[2]])

    q = q0.copy()
    for _ in range(iters):
        e, J = numerical_jacobian(error_fn, q)
        if np.linalg.norm(e[:2]) < 1e-4 and np.linalg.norm(e[2:]) < 1e-3:
            break
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (lam**2)*np.eye(JJt.shape[0]), -e)
        q = q + dq
        q = np.minimum(np.maximum(q, q_lim[:,0]), q_lim[:,1])
    return q

def build_xy_circle(center_xy=(0.6, 0.0), radius=0.15, samples=240):
    t = np.linspace(0, 2*np.pi, samples, endpoint=False)
    xs = center_xy[0] + radius*np.cos(t)
    ys = center_xy[1] + radius*np.sin(t)
    return np.stack([xs, ys], axis=1)

def solve_path_ik(DH, q_seed, xy_path, target_dir=(0,0,1), d6=0.1, iters=100, lam=1e-2):
    qs = []
    q = q_seed.copy()
    for xy in xy_path:
        q = ik_xy_dir(DH, q, xy, target_dir, d6=d6, iters=iters, lam=lam)
        qs.append(q.copy())
    return np.array(qs)

fps = 30
xy_path = build_xy_circle(center_xy=(0.6, 0.1), radius=0.12, samples=240)
q0 = np.zeros(6)
q_traj = solve_path_ik(DH, q0, xy_path, target_dir=(0,0,1), d6=0.10, iters=120, lam=2e-2)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
reach = 1.3
ax.set_xlim([-reach, reach]); ax.set_ylim([-reach, reach]); ax.set_zlim([0.0, reach*1.2])
ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
ax.set_title('RX‑90L IK demo: follow XY circle + keep tool vertical')

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
    q = q_traj[frame % len(q_traj)]
    pts, Ts = fk_Ts(DH, q)
    Te = Ts[-1].copy()
    Te[:3, 3] += Te[:3, :3] @ np.array([0.0, 0.0, 0.10])
    tcp = Te[:3, 3]
    xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
    for i, line in enumerate(link_lines):
        line.set_data(xs[i:i+2], ys[i:i+2]); line.set_3d_properties(zs[i:i+2])
    joint_scatter._offsets3d = (xs[:-1], ys[:-1], zs[:-1])
    trail_pts.append(tcp); tp = np.array(trail_pts[-300:])
    tcp_trail.set_data(tp[:,0], tp[:,1]); tcp_trail.set_3d_properties(tp[:,2])
    ax.view_init(elev=25, azim=35 + 0.4*frame)
    return link_lines + [tcp_trail, joint_scatter]

anim = FuncAnimation(fig, update, frames=len(q_traj), init_func=init, interval=1000/fps, blit=False)
plt.show()
