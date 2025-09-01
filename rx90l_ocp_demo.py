# ============================================================
# Optimal trajectory optimization for a 6R arm (RX-90L-like)
# Method: Direct multiple shooting + RK4 integration
# Solver: IPOPT via CasADi
#
# Notes:
#  - Requires: `pip install casadi` (and optionally `pinocchio` for real dynamics).
#  - Dynamics: tries Pinocchio if available and RX90L URDF is provided; otherwise uses a
#    diagonal-mass placeholder (OK to get the pipeline working; not physically accurate).
#  - Replace DH with your exact RX-90L parameters for accurate FK (used for tracking cost).
#  - You can change the objective and constraints as needed.
#
# How to run:
#    python rx90l_trajectory_ocp.py
#
# Author: ChatGPT for Dim

import os
import sys
import math
import numpy as np

try:
    import casadi as ca
except ImportError as e:
    print("This script requires CasADi. Install with: pip install casadi")
    sys.exit(1)

# ------------------------------------------------------------
# 1) KINEMATICS (Standard DH) for FK used in tracking cost
# ------------------------------------------------------------
def dh_T(a, alpha, d, theta):
    ca_, sa_ = ca.cos(alpha), ca.sin(alpha)
    ct, st = ca.cos(theta), ca.sin(theta)
    return ca.vertcat(
        ca.horzcat(ct, -st*ca_,  st*sa_, a*ct),
        ca.horzcat(st,  ct*ca_, -ct*sa_, a*st),
        ca.horzcat(0,       sa_,     ca_,    d),
        ca.horzcat(0,          0,       0,    1),
    )

# Placeholder RX-90L-ish DH. Replace with true values (m, rad).
DH = [
    (0.000,       -math.pi/2, 0.350, 0.0,       True),  # J1
    (0.450,        0.0,       0.000, -math.pi/2,True),  # J2
    (0.050,       -math.pi/2, 0.000,  math.pi/2,True),  # J3
    (0.425,        math.pi/2, 0.000,  0.0,      True),  # J4
    (0.000,       -math.pi/2, 0.000,  0.0,      True),  # J5
    (0.000,        0.0,       0.100,  0.0,      True),  # J6 (flange offset)
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
    """End-effector position (x,y,z) from DH FK."""
    T = fk_pose(q)
    return T[0:3, 3]

# ------------------------------------------------------------
# 2) DYNAMICS: Forward dynamics ddq = f(q,dq,tau)
#    (A) Pinocchio path if available and URDF provided
#    (B) Placeholder diagonal M, viscous + Coulomb friction, simple gravity
# ------------------------------------------------------------
def available_pinocchio():
    try:
        import pinocchio as pin  # noqa: F401
        return True
    except Exception:
        return False

class ForwardDynamics:
    def __init__(self):
        self.has_pin = False
        self.using_placeholder = True
        self.nq = 6

        # Joint limits (rad, rad/s, N*m) — rough values; adjust to your robot
        self.q_min = np.deg2rad(np.array([-160, -137.5, -142.5, -270, -105, -270], float))
        self.q_max = np.deg2rad(np.array([ 160,  137.5,  142.5,  270,  120,  270], float))
        self.dq_max = np.deg2rad(np.array([356, 356, 296, 409, 480, 1125], float))
        self.tau_max = np.array([70, 70, 50, 30, 20, 12], float)  # EDIT with spec if available

        # Try Pinocchio if user provides RX-90L URDF path
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

        # Placeholder parameters
        self.M_diag = np.array([7.0, 6.0, 3.5, 1.2, 0.8, 0.4], float)  # kg·m²-ish
        self.B_visc = np.array([2.0, 2.0, 1.5, 0.6, 0.4, 0.2], float)  # N·m·s/rad
        self.C_coul = np.array([2.5, 2.0, 1.5, 0.8, 0.6, 0.3], float)  # N·m
        self.g_vec = np.array([0, 0, -9.81], float)

    def casadi_forward_dyn(self):
        """Return CasADi function fd(q,dq,tau)->ddq (SX graph)."""
        q  = ca.SX.sym("q",  self.nq)
        dq = ca.SX.sym("dq", self.nq)
        tau= ca.SX.sym("tau",self.nq)

        if self.has_pin:
            # Pinocchio-based forward dynamics with external (zero) forces
            import pinocchio as pin
            # Wrap via external function evaluation (note: this path requires casadi-external or callbacks).
            # For portability in a single file, approximate with one-step explicit solve of M(q)ddq = tau - h.
            # We use Pinocchio to compute M and h (bias: C*dq + g).
            # Implement as a CasADi callback using python-embedded evaluation (OK for small problems).
            # ----------------------------------------------------------------------------------------
            class PinFD(ca.Callback):
                def __init__(self, name, fd_obj):
                    ca.Callback.__init__(self)
                    self.fd = fd_obj
                    self.construct(name)

                def get_n_in(self):  return 3
                def get_n_out(self): return 1
                def get_sparsity_in(self, i):
                    if i == 0: return ca.Sparsity.dense(self.fd.nq, 1)   # q
                    if i == 1: return ca.Sparsity.dense(self.fd.nq, 1)   # dq
                    if i == 2: return ca.Sparsity.dense(self.fd.nq, 1)   # tau
                def get_sparsity_out(self, i):
                    return ca.Sparsity.dense(self.fd.nq, 1)              # ddq

                def eval(self, args):
                    qv  = np.array(args[0]).reshape((-1,))
                    dqv = np.array(args[1]).reshape((-1,))
                    tauv= np.array(args[2]).reshape((-1,))
                    pin = __import__("pinocchio")
                    pin.crba(self.fd.pin_model, self.fd.pin_data, qv)
                    M = self.fd.pin_data.M.copy()
                    pin.computeCoriolisMatrix(self.fd.pin_model, self.fd.pin_data, qv, dqv)
                    C = self.fd.pin_data.C.copy()
                    g = pin.computeGeneralizedGravity(self.fd.pin_model, self.fd.pin_data, qv)
                    # Forward dynamics: M ddq = tau - C*dq - g - friction
                    visc = self.B_visc * dqv
                    coul = self.C_coul * np.tanh(50.0 * dqv)  # smooth sign
                    rhs = tauv - (C @ dqv) - g - visc - coul
                    ddq = np.linalg.solve(M, rhs)
                    return [ddq.reshape((-1,1))]

            fd_cb = PinFD("pin_fd", self)
            fd = ca.Function("fd", [q, dq, tau], [fd_cb(q, dq, tau)])
            return fd

        # Placeholder forward dynamics (diagonal M + simple gravity & friction)
        # M(q) ~ diag(M_diag), g(q) ~ project tool mass via a crude lever arm using FK z-height (only qualitative).
        # This is solely for getting an OCP pipeline running; replace with Pinocchio for real results.
        p = ee_pos(q)
        z = p[2]
        # crude gravity torque trend: map z to joint torques via a decaying weighting from proximal to distal
        w = ca.vertcat(1.0, 0.9, 0.7, 0.4, 0.2, 0.1)
        g_tau = 9.81 * z * w  # not physical; placeholder

        visc = ca.diag(self.B_visc) @ dq
        coul = ca.diag(self.C_coul) @ ca.tanh(50*dq)

        rhs = tau - visc - coul - g_tau
        Minv = ca.diag(1.0/ca.DM(self.M_diag))
        ddq = Minv @ rhs
        fd = ca.Function("fd", [q, dq, tau], [ddq])
        return fd

FD = ForwardDynamics()
fd = FD.casadi_forward_dyn()

# ------------------------------------------------------------
# 3) OCP SETUP (multiple shooting, RK4)
# ------------------------------------------------------------
nq = 6
nx = 2*nq   # state: [q; dq]
nu = nq     # control: tau

# Horizon and grid
N = 60                   # number of shooting intervals
T_min, T_max = 2.0, 8.0  # bounds on final time [s]
T_sym = ca.SX.sym("T")   # free final time
dt = T_sym / N

# Decision variables
X = ca.SX.sym("X", nx, N+1)   # [q; dq] at each node
U = ca.SX.sym("U", nu, N)     # tau at each interval

# Initial and final boundary conditions (edit)
q_start = np.deg2rad(np.array([0, -30, 60, 0, 45, 0], float))
dq_start = np.zeros(nq)
q_goal  = np.deg2rad(np.array([30, -10, 40, 20, 30, -20], float))
dq_goal = np.zeros(nq)

# Build dynamics integrator (RK4)
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

# Constraints & objective
g_constr = []
g_l = []
g_u = []

J = 0

# Weighting
W_tau   = 1e-3   # effort regularization
W_dq    = 1e-4   # velocity regularization
W_track = 10.0   # Cartesian tracking weight
W_T     = 1e-1   # penalize long time

# Build a Cartesian reference path at knot points (circle in XY, fixed Z from initial FK)
def build_task_path(N):
    T0 = fk_pose(q_start)
    p0 = np.array([float(T0[0,3]), float(T0[1,3]), max(0.2, float(T0[2,3]))])
    radius = 0.12
    center = p0 + np.array([0.0, 0.0, 0.0])
    ts = np.linspace(0, 2*math.pi, N+1)
    Ps = []
    for t in ts:
        x = center[0] + radius*math.cos(t)
        y = center[1] + radius*math.sin(t)
        z = center[2]
        Ps.append(np.array([x,y,z]))
    return np.array(Ps)

P_ref = build_task_path(N)

# Node 0 boundary
g_constr += [ X[0:nq,0]   - ca.DM(q_start) ]
g_l      += [ np.zeros(nq) ]
g_u      += [ np.zeros(nq) ]

g_constr += [ X[nq:2*nq,0] - ca.DM(dq_start) ]
g_l      += [ np.zeros(nq) ]
g_u      += [ np.zeros(nq) ]

# Path and dynamics constraints
for k in range(N):
    xk = X[:,k]
    uk = U[:,k]

    # Dynamics
    xk1 = rk4_step(xk, uk, dt)
    g_constr += [ X[:,k+1] - xk1 ]
    g_l += [ np.zeros(nx) ]
    g_u += [ np.zeros(nx) ]

    # Joint/velocity/torque bounds as soft (cost) + hard (box constraints added later)
    J += W_tau * ca.sumsqr(uk)
    J += W_dq  * ca.sumsqr(xk[nq:2*nq])

    # Cartesian tracking (EE position) at nodes
    qk = xk[0:nq]
    pk = ee_pos(qk)
    pref = ca.DM(P_ref[k])
    J += W_track * ca.sumsqr(pk - pref)

# Terminal constraints
g_constr += [ X[0:nq, -1]   - ca.DM(q_goal) ]
g_l      += [ np.zeros(nq) ]
g_u      += [ np.zeros(nq) ]

g_constr += [ X[nq:2*nq, -1] - ca.DM(dq_goal) ]
g_l      += [ np.zeros(nq) ]
g_u      += [ np.zeros(nq) ]

# Final time cost
J += W_T * T_sym

# Decision vector & bounds
w   = [X.reshape((-1,1)), U.reshape((-1,1)), ca.vertcat(T_sym)]
w   = ca.vertcat(*w)

# Box bounds
w_l = []
w_u = []

# X bounds
for k in range(N+1):
    # q bounds
    w_l += list(FD.q_min)
    w_u += list(FD.q_max)
    # dq bounds
    w_l += list(-FD.dq_max)
    w_u += list( FD.dq_max)

# U bounds (tau)
for k in range(N):
    w_l += list(-FD.tau_max)
    w_u += list( FD.tau_max)

# T bounds
w_l += [T_min]
w_u += [T_max]

w_l = ca.DM(w_l)
w_u = ca.DM(w_u)

# Concatenate constraints
g = ca.vertcat(*[gc.reshape((-1,1)) for gc in g_constr])
g_l = ca.vertcat(*[ca.DM(gl).reshape((-1,1)) for gl in g_l])
g_u = ca.vertcat(*[ca.DM(gu).reshape((-1,1)) for gu in g_u])

# Initial guess
w0 = []
# X guess: linear interpolation q, zeros dq
for k in range(N+1):
    alpha = k/float(N)
    qg = (1-alpha)*q_start + alpha*q_goal
    w0 += list(qg)
    w0 += list((1-alpha)*dq_start + alpha*dq_goal)
# U guess: zeros
for k in range(N):
    w0 += list(np.zeros(nu))
# T guess
w0 += [(T_min+T_max)/2]

w0 = ca.DM(w0)

# Build NLP
nlp = {"x": w, "f": J, "g": g}
opts = {
    "ipopt": {
        "print_level": 5,
        "max_iter": 1000,
        "tol": 1e-4,
        "linear_solver": "mumps"
    },
    "print_time": True
}
solver = ca.nlpsol("solver", "ipopt", nlp, opts)

# Solve
sol = solver(x0=w0, lbx=w_l, ubx=w_u, lbg=g_l, ubg=g_u)
w_opt = sol["x"].full().flatten()

# Unpack solution
def unpack(wvec):
    idx = 0
    X_opt = np.zeros((nx, N+1))
    U_opt = np.zeros((nu, N))
    for k in range(N+1):
        X_opt[:,k] = wvec[idx:idx+nx]
        idx += nx
    for k in range(N):
        U_opt[:,k] = wvec[idx:idx+nu]
        idx += nu
    T_opt = wvec[idx]; idx += 1
    return X_opt, U_opt, T_opt

X_opt, U_opt, T_opt = unpack(w_opt)
q_opt  = X_opt[0:nq,:]
dq_opt = X_opt[nq:2*nq,:]
tau_opt= U_opt

print("Solved. T* = {:.3f} s".format(T_opt))
print("q[0]* (deg):", np.rad2deg(q_opt[:,0]))
print("q[end]* (deg):", np.rad2deg(q_opt[:,-1]))

# Optional: quick plot (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    tgrid = np.linspace(0, float(T_opt), N+1)
    fig, axs = plt.subplots(3, 1, figsize=(10,8), sharex=True)
    axs[0].plot(tgrid, np.rad2deg(q_opt).T); axs[0].set_ylabel("q [deg]")
    axs[1].plot(tgrid, np.rad2deg(dq_opt).T); axs[1].set_ylabel("dq [deg/s]")
    axs[2].plot(np.linspace(0, float(T_opt), N), tau_opt.T); axs[2].set_ylabel("tau [Nm]"); axs[2].set_xlabel("t [s]")
    axs[0].grid(True); axs[1].grid(True); axs[2].grid(True)
    plt.suptitle("RX-90L OCP solution (placeholder dynamics)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    pass
