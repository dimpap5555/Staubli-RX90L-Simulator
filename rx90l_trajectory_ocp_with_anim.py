# rx90l_trajectory_ocp_with_anim.py
# ============================================================
# Optimal trajectory optimization for a 6R arm (RX-90L-like)
# + the SAME 3D animation you used before (TCP trail, rotating camera)
#
# Solver: CasADi + IPOPT (multiple shooting, RK4)
# FK/animation: numpy-based DH for plotting q_opt
#
import os
import sys
import math
import numpy as np

try:
    import casadi as ca
except ImportError as e:
    print("This script requires CasADi. Install with: pip install casadi")
    sys.exit(1)

# ---------------- KINEMATICS (CasADi graph) ----------------
def dh_T(a, alpha, d, theta):
    ca_, sa_ = ca.cos(alpha), ca.sin(alpha)
    ct, st = ca.cos(theta), ca.sin(theta)
    return ca.vertcat(
        ca.horzcat(ct, -st*ca_,  st*sa_, a*ct),
        ca.horzcat(st,  ct*ca_, -ct*sa_, a*st),
        ca.horzcat(0,       sa_,     ca_,    d),
        ca.horzcat(0,          0,       0,    1),
    )

# RX-90L-ish DH (replace with measured values)
DH = [
    (0.000,       -math.pi/2, 0.350, 0.0,       True),  # J1
    (0.450,        0.0,       0.000, -math.pi/2,True),  # J2
    (0.050,       -math.pi/2, 0.000,  math.pi/2,True),  # J3
    (0.425,        math.pi/2, 0.000,  0.0,      True),  # J4
    (0.000,       -math.pi/2, 0.000,  0.0,      True),  # J5
    (0.000,        0.0,       0.100,  0.0,      True),  # J6
]

def fk_pose(q):
    T = ca.SX.eye(4)
    for i in range(len(DH)):
        a, alpha, d, theta0, is_rev = DH[i]
        qi = q[i] if is_rev else 0.0
        theta = theta0 + qi
        T = ca.mtimes(T, dh_T(a, alpha, d, theta))
    return T

def ee_pos(q):
    T = fk_pose(q)
    return T[0:3, 3]

# === Pose/orientation helpers (task-space constraints) ===
def skew(v):
    return ca.vertcat(
        ca.horzcat(    0, -v[2],  v[1]),
        ca.horzcat( v[2],     0, -v[0]),
        ca.horzcat(-v[1],  v[0],     0)
    )

def vee(S):
    # S is 3x3 skew-symmetric; return vector
    return ca.vertcat(S[2,1], S[0,2], S[1,0])

def rpy_to_R(roll, pitch, yaw):
    # XYZ convention: R = Rx(roll) * Ry(pitch) * Rz(yaw)
    cr, sr = ca.cos(roll),  ca.sin(roll)
    cp, sp = ca.cos(pitch), ca.sin(pitch)
    cy, sy = ca.cos(yaw),   ca.sin(yaw)
    Rx = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, cr, -sr),
        ca.horzcat(0, sr,  cr)
    )
    Ry = ca.vertcat(
        ca.horzcat(cp, 0, sp),
        ca.horzcat( 0, 1,  0),
        ca.horzcat(-sp,0, cp)
    )
    Rz = ca.vertcat(
        ca.horzcat(cy, -sy, 0),
        ca.horzcat(sy,  cy, 0),
        ca.horzcat( 0,   0, 1)
    )
    return ca.mtimes(Rx, ca.mtimes(Ry, Rz))

def ori_err(R, Rd):
    # Angle-axis oriented error: e_R = 0.5 * vee(Rd^T R - R^T Rd)
    return 0.5 * vee(ca.mtimes(Rd.T, R) - ca.mtimes(R.T, Rd))

# ---------------- DYNAMICS ----------------
def available_pinocchio():
    try:
        import pinocchio as pin  # noqa: F401
        return True
    except Exception:
        return False

class ForwardDynamics:
    def __init__(self):
        self.nq = 6
        self.q_min = np.deg2rad(np.array([-160, -137.5, -142.5, -270, -105, -270], float))
        self.q_max = np.deg2rad(np.array([ 160,  137.5,  142.5,  270,  120,  270], float))
        self.dq_max = np.deg2rad(np.array([356, 356, 296, 409, 480, 1125], float))
        self.tau_max = np.array([70, 70, 50, 30, 20, 12], float)

        self.M_diag = np.array([7.0, 6.0, 3.5, 1.2, 0.8, 0.4], float)
        self.B_visc = np.array([2.0, 2.0, 1.5, 0.6, 0.4, 0.2], float)
        self.C_coul = np.array([2.5, 2.0, 1.5, 0.8, 0.6, 0.3], float)

        self.has_pin = False
        self.using_placeholder = True
        urdf_path = os.environ.get("RX90L_URDF", "").strip()
        if urdf_path and available_pinocchio():
            try:
                import pinocchio as pin
                model = pin.buildModelFromUrdf(urdf_path)
                data = model.createData()
                self.pin_model = model
                self.pin_data = data
                self.has_pin = True
                self.using_placeholder = False
                print("[Dynamics] Using Pinocchio model from URDF:", urdf_path)
            except Exception as e:
                print("[Dynamics] Pinocchio failed, fallback to placeholder. Reason:", repr(e))
        else:
            print("[Dynamics] Placeholder model in use (set RX90L_URDF and install pinocchio for accurate dynamics).")

    def casadi_forward_dyn(self):
        q  = ca.SX.sym("q",  self.nq)
        dq = ca.SX.sym("dq", self.nq)
        tau= ca.SX.sym("tau",self.nq)

        if self.has_pin:
            import pinocchio as pin
            class PinFD(ca.Callback):
                def __init__(self, name, fd_obj):
                    ca.Callback.__init__(self); self.fd = fd_obj; self.construct(name)
                def get_n_in(self):  return 3
                def get_n_out(self): return 1
                def get_sparsity_in(self, i):
                    if i == 0: return ca.Sparsity.dense(self.fd.nq, 1)
                    if i == 1: return ca.Sparsity.dense(self.fd.nq, 1)
                    if i == 2: return ca.Sparsity.dense(self.fd.nq, 1)
                def get_sparsity_out(self, i):
                    return ca.Sparsity.dense(self.fd.nq, 1)
                def eval(self, args):
                    qv  = np.array(args[0]).reshape((-1,))
                    dqv = np.array(args[1]).reshape((-1,))
                    tauv= np.array(args[2]).reshape((-1,))
                    pin.crba(self.fd.pin_model, self.fd.pin_data, qv)
                    M = self.fd.pin_data.M.copy()
                    pin.computeCoriolisMatrix(self.fd.pin_model, self.fd.pin_data, qv, dqv)
                    C = self.fd.pin_data.C.copy()
                    g = pin.computeGeneralizedGravity(self.fd.pin_model, self.fd.pin_data, qv)
                    visc = np.array([2.0,2.0,1.5,0.6,0.4,0.2]) * dqv
                    coul = np.array([2.5,2.0,1.5,0.8,0.6,0.3]) * np.tanh(50.0 * dqv)
                    rhs = tauv - (C @ dqv) - g - visc - coul
                    ddq = np.linalg.solve(M, rhs)
                    return [ddq.reshape((-1,1))]
            fd_cb = PinFD("pin_fd", self)
            fd = ca.Function("fd", [q, dq, tau], [fd_cb(q, dq, tau)])
            return fd

        # Placeholder forward dynamics
        p = ee_pos(q)
        z = p[2]
        w = ca.vertcat(1.0, 0.9, 0.7, 0.4, 0.2, 0.1)
        g_tau = 9.81 * z * w
        visc = ca.diag(ca.DM([2.0,2.0,1.5,0.6,0.4,0.2])) @ dq
        coul = ca.diag(ca.DM([2.5,2.0,1.5,0.8,0.6,0.3])) @ ca.tanh(50*dq)
        rhs = tau - visc - coul - g_tau
        Minv = ca.diag(1.0/ca.DM([7.0,6.0,3.5,1.2,0.8,0.4]))
        ddq = Minv @ rhs
        return ca.Function("fd", [q, dq, tau], [ddq])

FD = ForwardDynamics()
fd = FD.casadi_forward_dyn()

# ---------------- OCP ----------------
nq = 6
nx = 2*nq
nu = nq
N = 60
T_min, T_max = 2.0, 8.0
T_sym = ca.SX.sym("T")
dt = T_sym / N

X = ca.SX.sym("X", nx, N+1)
U = ca.SX.sym("U", nu, N)

# === USER: set start & goal EE pose (meters, degrees) ===
x_start, y_start, z_start = 0.55, 0.0, 0.50
roll_start, pitch_start, yaw_start = np.deg2rad([0.0, 0.0, 0.0])   # deg → rad

x_goal,  y_goal,  z_goal  = 0.7, 0.0, 0.50
roll_goal,  pitch_goal,  yaw_goal  = np.deg2rad([0.0, 0.0, 0.0])  # deg → rad

p_start_vec = ca.DM([x_start, y_start, z_start])
p_goal_vec  = ca.DM([x_goal,  y_goal,  z_goal])

R_start = rpy_to_R(roll_start, pitch_start, yaw_start)
R_goal  = rpy_to_R(roll_goal,  pitch_goal,  yaw_goal)

# Initial-guess joint states (seed) – tweak or fill via your IK if you have it
q_start_guess = np.deg2rad(np.array([0, -30, 60, 0, 45, 0], float))
q_goal_guess  = np.deg2rad(np.array([30, -10, 40, 20, 30, -20], float))
dq_start = np.zeros(nq)
dq_goal  = np.zeros(nq)

def rk4_step(xk, uk, dt):
    qk  = xk[0:nq]
    dqk = xk[nq:2*nq]
    def f(x, u):
        q, dq = x[0:nq], x[nq:2*nq]
        ddq = fd(q, dq, u)
        return ca.vertcat(dq, ddq)
    k1 = f(xk,        uk)
    k2 = f(xk + 0.5*dt*k1, uk)
    k3 = f(xk + 0.5*dt*k2, uk)
    k4 = f(xk +      dt*k3, uk)
    xk1 = xk + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return xk1

g_constr = []; g_l = []; g_u = []
J = 0.0
W_tau, W_dq, W_track, W_T = 1e-3, 1e-4, 10.0, 1e-1

def build_task_path(N):
    T0 = fk_pose(ca.DM(q_start))
    p0 = np.array([float(T0[0,3]), float(T0[1,3]), max(0.2, float(T0[2,3]))])
    radius = 0.12
    center = p0.copy()
    ts = np.linspace(0, 2*math.pi, N+1)
    Ps = []
    for t in ts:
        x = center[0] + radius*math.cos(t)
        y = center[1] + radius*math.sin(t)
        z = center[2]
        Ps.append(np.array([x,y,z]))
    return np.array(Ps)

# Initial state constraints
q0 = X[0:nq, 0]
T0 = fk_pose(q0); p0 = T0[0:3, 3]; R0 = T0[0:3, 0:3]
g_constr += [ p0 - p_start_vec ];             g_l += [ np.zeros(3) ]; g_u += [ np.zeros(3) ]
g_constr += [ ori_err(R0, R_start) ];     g_l += [ np.zeros(3) ]; g_u += [ np.zeros(3) ]
g_constr += [ X[nq:2*nq,0] - ca.DM(dq_start) ]; g_l += [ np.zeros(nq) ]; g_u += [ np.zeros(nq) ]

for k in range(N):
    xk = X[:,k]; uk = U[:,k]
    xk1 = rk4_step(xk, uk, dt)
    g_constr += [ X[:,k+1] - xk1 ]; g_l += [ np.zeros(nx) ]; g_u += [ np.zeros(nx) ]
    J += W_tau * ca.sumsqr(uk) + W_dq * ca.sumsqr(xk[nq:2*nq])
    qk = xk[0:nq]; pk = ee_pos(qk)

# Terminal constraints
qT = X[0:nq, -1]
TT = fk_pose(qT); pT = TT[0:3, 3]; RT = TT[0:3, 0:3]
g_constr += [ pT - p_goal_vec ];              g_l += [ np.zeros(3) ]; g_u += [ np.zeros(3) ]
g_constr += [ ori_err(RT, R_goal) ];      g_l += [ np.zeros(3) ]; g_u += [ np.zeros(3) ]
g_constr += [ X[nq:2*nq,-1] - ca.DM(dq_goal) ]; g_l += [ np.zeros(nq) ]; g_u += [ np.zeros(nq) ]

J += W_T * T_sym

# Decision vector
w = ca.vertcat(X.reshape((-1,1)), U.reshape((-1,1)), ca.vertcat(T_sym))

# Bounds
w_l = []; w_u = []
for k in range(N+1):
    w_l += list(FD.q_min); w_u += list(FD.q_max)         # q
    w_l += list(-FD.dq_max); w_u += list(FD.dq_max)      # dq
for k in range(N):
    w_l += list(-FD.tau_max); w_u += list(FD.tau_max)    # tau
w_l += [2.0]; w_u += [8.0]                               # T

w_l = ca.DM(w_l); w_u = ca.DM(w_u)
g = ca.vertcat(*[gc.reshape((-1,1)) for gc in g_constr])
g_l = ca.vertcat(*[ca.DM(gl).reshape((-1,1)) for gl in g_l])
g_u = ca.vertcat(*[ca.DM(gu).reshape((-1,1)) for gu in g_u])

# Initial guess
w0 = []
for k in range(N+1):
    alpha = k/float(N)
    qg = (1-alpha)*np.array(q_start_guess) + alpha*np.array(q_goal_guess)
    w0 += list(qg); w0 += list(np.zeros(nq))
for k in range(N):
    w0 += list(np.zeros(nu))
w0 += [(T_min + T_max)/2]
w0 = ca.DM(w0)

# Solve
nlp = {"x": w, "f": J, "g": g}
opts = {"ipopt": {"print_level": 5, "max_iter": 1000, "tol": 1e-4, "linear_solver": "mumps"}, "print_time": True}
solver = ca.nlpsol("solver", "ipopt", nlp, opts)
sol = solver(x0=w0, lbx=w_l, ubx=w_u, lbg=g_l, ubg=g_u)
w_opt = sol["x"].full().flatten()

def unpack(wvec, N, nx, nu):
    idx = 0
    X_opt = np.zeros((nx, N+1))
    U_opt = np.zeros((nu, N))
    for k in range(N+1):
        X_opt[:,k] = wvec[idx:idx+nx]; idx += nx
    for k in range(N):
        U_opt[:,k] = wvec[idx:idx+nu]; idx += nu
    T_opt = wvec[idx]
    return X_opt, U_opt, T_opt

X_opt, U_opt, T_opt = unpack(w_opt, N, nx, nu)
q_opt  = X_opt[0:6,:]              # (6, N+1)
dq_opt = X_opt[6:12,:]
tau_opt= U_opt
print("Solved. T* = {:.3f} s".format(float(T_opt)))

# ---------------- ANIMATION (NumPy FK for plot) ----------------
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def dh_np(a, alpha, d, theta):
    ca_, sa_ = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    T = np.array([[ct, -st*ca_,  st*sa_, a*ct],
                  [st,  ct*ca_, -ct*sa_, a*st],
                  [0.0,    sa_,     ca_,    d],
                  [0.0,   0.0,    0.0,  1.0]], dtype=float)
    return T

def fk_Ts_np(DH, q):
    T = np.eye(4)
    Ts = [T.copy()]
    pts = [T[:3,3].copy()]
    for i in range(len(DH)):
        a, alpha, d, theta0, is_rev = DH[i]
        theta = theta0 + (q[i] if is_rev else 0.0)
        T = T @ dh_np(a, alpha, d, theta)
        Ts.append(T.copy())
        pts.append(T[:3,3].copy())
    return np.array(pts), Ts

# Build a higher-FPS joint trajectory by linear interp of q_opt over time grid
fps = 30
t_nodes = np.linspace(0, float(T_opt), N+1)
t_anim  = np.linspace(0, float(T_opt), int(fps*float(T_opt))+1)
q_traj  = np.empty((t_anim.size, 6))
for j in range(6):
    q_traj[:, j] = np.interp(t_anim, t_nodes, q_opt[j,:])

# Plot setup
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
reach = 1.3
ax.set_xlim([-reach, reach]); ax.set_ylim([-reach, reach]); ax.set_zlim([0.0, reach*1.2])
ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
ax.set_title('RX‑90L – OCP result animation (q_opt)')

link_lines = []
joint_scatter = ax.scatter([], [], [], s=20)
tcp_trail, = ax.plot([], [], [], lw=1, alpha=0.5)
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
    pts, Ts = fk_Ts_np(DH, q)
    Te = Ts[-1].copy()
    Te[:3, 3] += Te[:3, :3] @ np.array([0.0, 0.0, 0.10])  # tool z offset for display
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
