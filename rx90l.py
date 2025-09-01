import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Rx90L:
    """Simple RX-90L utility class bundling FK, IK and OCP demos."""

    def __init__(self):
        # Denavit–Hartenberg parameters (a, alpha, d, theta0, revolute?)
        self.DH = [
            (0.000,        -np.pi/2, 0.350, 0.0,          True),
            (0.450,         0.0,     0.000, -np.pi/2,     True),
            (0.050,        -np.pi/2, 0.000,  np.pi/2,     True),
            (0.425,         np.pi/2, 0.000,  0.0,         True),
            (0.000,        -np.pi/2, 0.000,  0.0,         True),
            (0.000,         0.0,     0.100,  0.0,         True),
        ]
        # Joint limits [rad]
        self.q_lim = np.deg2rad(np.array([
            [-160, 160],
            [-137.5, 137.5],
            [-142.5, 142.5],
            [-270, 270],
            [-105, 120],
            [-270, 270],
        ], dtype=float))

    # ------------------------------------------------------------------
    # Basic kinematics helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dh(a, alpha, d, theta):
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([[ct, -st*ca,  st*sa, a*ct],
                         [st,  ct*ca, -ct*sa, a*st],
                         [0.0,    sa,     ca,    d],
                         [0.0,   0.0,    0.0,  1.0]], dtype=float)

    def fk_chain(self, q):
        """Return XYZ positions of each joint (including base and TCP)."""
        T = np.eye(4)
        pts = [T[:3, 3].copy()]
        for (a, alpha, d, theta0, is_rev), qi in zip(self.DH, q):
            theta = theta0 + (qi if is_rev else 0.0)
            T = T @ self._dh(a, alpha, d, theta)
            pts.append(T[:3, 3].copy())
        return np.array(pts)

    # ------------------------------------------------------------------
    # Forward kinematics demo trajectory
    # ------------------------------------------------------------------
    def forward_demo(self, T_final=10.0, fps=30):
        t = np.linspace(0, T_final, int(T_final*fps))
        amp = 0.4 * (self.q_lim[:,1] - self.q_lim[:,0]) / 2.0
        center = (self.q_lim[:,1] + self.q_lim[:,0]) / 2.0
        freqs = np.array([0.2, 0.31, 0.17, 0.27, 0.23, 0.29])
        phases = np.linspace(0, np.pi, 6)
        q_traj = np.zeros((len(t), 6))
        for j in range(6):
            q_traj[:, j] = center[j] + amp[j]*np.sin(2*np.pi*freqs[j]*t + phases[j])
        return q_traj

    # ------------------------------------------------------------------
    # Inverse kinematics demo (follow XY circle while tool stays vertical)
    # ------------------------------------------------------------------
    @staticmethod
    def _numerical_jacobian(f, q, eps=1e-6):
        q = q.copy()
        y0 = f(q)
        m = y0.size
        J = np.zeros((m, q.size))
        for i in range(q.size):
            qi = q.copy(); qi[i] += eps
            yi = f(qi)
            J[:, i] = (yi - y0) / eps
        return y0, J

    def _fk_Ts(self, q):
        T = np.eye(4)
        Ts = [T.copy()]
        for (a, alpha, d, theta0, is_rev), qi in zip(self.DH, q):
            theta = theta0 + (qi if is_rev else 0.0)
            T = T @ self._dh(a, alpha, d, theta)
            Ts.append(T.copy())
        pts = np.array([Ti[:3, 3] for Ti in Ts])
        return pts, Ts

    def ik_xy_dir(self, q0, target_xy, target_dir, d6=0.0, iters=100, lam=1e-2):
        td = np.asarray(target_dir, float)
        td = td / (np.linalg.norm(td) + 1e-12)

        def err(q):
            _, Ts = self._fk_Ts(q)
            Te = Ts[-1].copy()
            Te[:3, 3] += Te[:3, :3] @ np.array([0.0, 0.0, d6])
            p = Te[:3, 3]
            a = Te[:3, :3] @ np.array([0.0, 0.0, 1.0])
            a_n = a / (np.linalg.norm(a) + 1e-12)
            ex = p[0] - target_xy[0]
            ey = p[1] - target_xy[1]
            c = np.cross(a_n, td)
            return np.array([ex, ey, c[0], c[1], c[2]])

        q = q0.copy()
        for _ in range(iters):
            e, J = self._numerical_jacobian(err, q)
            if np.linalg.norm(e[:2]) < 1e-4 and np.linalg.norm(e[2:]) < 1e-3:
                break
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + (lam**2)*np.eye(JJt.shape[0]), -e)
            q = q + dq
            q = np.minimum(np.maximum(q, self.q_lim[:,0]), self.q_lim[:,1])
        return q

    def build_xy_circle(self, center_xy=(0.6,0.0), radius=0.15, samples=240):
        t = np.linspace(0, 2*np.pi, samples, endpoint=False)
        xs = center_xy[0] + radius*np.cos(t)
        ys = center_xy[1] + radius*np.sin(t)
        return np.stack([xs, ys], axis=1)

    def solve_path_ik(self, q_seed, xy_path, target_dir=(0,0,1), d6=0.1, iters=100, lam=1e-2):
        qs = []
        q = q_seed.copy()
        for xy in xy_path:
            q = self.ik_xy_dir(q, xy, target_dir, d6=d6, iters=iters, lam=lam)
            qs.append(q.copy())
        return np.array(qs)

    def ik_demo(self, samples=240):
        path = self.build_xy_circle(samples=samples)
        q0 = np.zeros(6)
        return self.solve_path_ik(q0, path, target_dir=(0,0,1), d6=0.10, iters=120, lam=2e-2)

    # ------------------------------------------------------------------
    # Trajectory optimization demo (CasADi multiple shooting)
    # ------------------------------------------------------------------
    def ocp_demo(self):
        try:
            import casadi as ca
        except ImportError:
            raise RuntimeError("CasADi is required for the OCP demo")

        # Helper for DH in CasADi graph
        def dh_T(a, alpha, d, theta):
            ca_, sa_ = ca.cos(alpha), ca.sin(alpha)
            ct, st = ca.cos(theta), ca.sin(theta)
            return ca.vertcat(
                ca.horzcat(ct, -st*ca_,  st*sa_, a*ct),
                ca.horzcat(st,  ct*ca_, -ct*sa_, a*st),
                ca.horzcat(0,       sa_,     ca_,    d),
                ca.horzcat(0,          0,       0,    1),
            )

        def fk_pose(q):
            T = ca.SX.eye(4)
            for i, (a, alpha, d, theta0, is_rev) in enumerate(self.DH):
                qi = q[i]
                theta = theta0 + (qi if is_rev else 0.0)
                T = ca.mtimes(T, dh_T(a, alpha, d, theta))
            return T

        def ee_pos(q):
            return fk_pose(q)[0:3,3]

        # Simplified placeholder dynamics
        def casadi_forward_dyn():
            q  = ca.SX.sym("q",6)
            dq = ca.SX.sym("dq",6)
            tau= ca.SX.sym("tau",6)
            p = ee_pos(q)
            z = p[2]
            w = ca.vertcat(1,0.9,0.7,0.4,0.2,0.1)
            g_tau = 9.81*z*w
            visc = ca.diag(ca.DM([2.0,2.0,1.5,0.6,0.4,0.2]))@dq
            coul = ca.diag(ca.DM([2.5,2.0,1.5,0.8,0.6,0.3]))@ca.tanh(50*dq)
            rhs = tau - visc - coul - g_tau
            Minv = ca.diag(1.0/ca.DM([7.0,6.0,3.5,1.2,0.8,0.4]))
            ddq = Minv @ rhs
            return ca.Function("fd", [q,dq,tau], [ddq])

        fd = casadi_forward_dyn()

        nq = 6
        nx = 2*nq
        nu = nq
        N  = 60
        T_min, T_max = 2.0, 8.0
        T_sym = ca.SX.sym("T")
        dt = T_sym / N
        X = ca.SX.sym("X", nx, N+1)
        U = ca.SX.sym("U", nu, N)

        q_start = np.deg2rad(np.array([0, -30, 60, 0, 45, 0], float))
        dq_start = np.zeros(nq)
        q_goal  = np.deg2rad(np.array([30, -10, 40, 20, 30, -20], float))
        dq_goal = np.zeros(nq)

        def rk4_step(xk, uk, dt):
            qk  = xk[0:nq]
            dqk = xk[nq:2*nq]
            def f(x,u):
                q, dq = x[0:nq], x[nq:2*nq]
                ddq = fd(q,dq,u)
                return ca.vertcat(dq,ddq)
            k1 = f(xk,        uk)
            k2 = f(xk+0.5*dt*k1, uk)
            k3 = f(xk+0.5*dt*k2, uk)
            k4 = f(xk+    dt*k3, uk)
            return xk + (dt/6)*(k1+2*k2+2*k3+k4)

        g_constr = []
        g_l = []
        g_u = []
        J = 0
        W_tau, W_dq, W_track, W_T = 1e-3, 1e-4, 10.0, 1e-1

        def build_task_path(N):
            T0 = fk_pose(q_start)
            p0 = np.array([float(T0[0,3]), float(T0[1,3]), max(0.2, float(T0[2,3]))])
            radius = 0.12
            ts = np.linspace(0, 2*np.pi, N+1)
            Ps = []
            for t in ts:
                x = p0[0] + radius*np.cos(t)
                y = p0[1] + radius*np.sin(t)
                z = p0[2]
                Ps.append(np.array([x,y,z]))
            return np.array(Ps)

        P_ref = build_task_path(N)

        g_constr += [ X[0:nq,0]   - ca.DM(q_start) ]
        g_l      += [ np.zeros(nq) ]
        g_u      += [ np.zeros(nq) ]
        g_constr += [ X[nq:2*nq,0] - ca.DM(dq_start) ]
        g_l      += [ np.zeros(nq) ]
        g_u      += [ np.zeros(nq) ]

        for k in range(N):
            xk = X[:,k]
            uk = U[:,k]
            xk1 = rk4_step(xk, uk, dt)
            g_constr += [ X[:,k+1] - xk1 ]
            g_l += [ np.zeros(nx) ]
            g_u += [ np.zeros(nx) ]
            J += W_tau*ca.sumsqr(uk) + W_dq*ca.sumsqr(xk[nq:2*nq])
            qk = xk[0:nq]
            pk = ee_pos(qk)
            pref = ca.DM(P_ref[k])
            J += W_track*ca.sumsqr(pk - pref)

        g_constr += [ X[0:nq,-1]   - ca.DM(q_goal) ]
        g_l      += [ np.zeros(nq) ]
        g_u      += [ np.zeros(nq) ]
        g_constr += [ X[nq:2*nq,-1] - ca.DM(dq_goal) ]
        g_l      += [ np.zeros(nq) ]
        g_u      += [ np.zeros(nq) ]

        J += W_T*T_sym

        w = ca.vertcat(X.reshape((-1,1)), U.reshape((-1,1)), T_sym)
        w_l = []
        w_u = []
        for k in range(N+1):
            w_l += list(self.q_lim[:,0]); w_u += list(self.q_lim[:,1])
            w_l += list(-np.deg2rad([356,356,296,409,480,1125]))
            w_u += list( np.deg2rad([356,356,296,409,480,1125]))
        for k in range(N):
            w_l += list(-np.array([70,70,50,30,20,12]))
            w_u += list( np.array([70,70,50,30,20,12]))
        w_l += [T_min]; w_u += [T_max]
        w_l = ca.DM(w_l); w_u = ca.DM(w_u)

        g = ca.vertcat(*[gc.reshape((-1,1)) for gc in g_constr])
        g_l = ca.vertcat(*[ca.DM(gl).reshape((-1,1)) for gl in g_l])
        g_u = ca.vertcat(*[ca.DM(gu).reshape((-1,1)) for gu in g_u])

        w0 = []
        for k in range(N+1):
            alpha = k/float(N)
            qg = (1-alpha)*q_start + alpha*q_goal
            w0 += list(qg); w0 += list((1-alpha)*dq_start + alpha*dq_goal)
        for k in range(N):
            w0 += list(np.zeros(nu))
        w0 += [(T_min+T_max)/2]
        w0 = ca.DM(w0)

        nlp = {"x": w, "f": J, "g": g}
        opts = {"ipopt": {"print_level": 0, "max_iter": 500}}
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(x0=w0, lbx=w_l, ubx=w_u, lbg=g_l, ubg=g_u)
        w_opt = sol["x"].full().flatten()

        idx = 0
        X_opt = np.zeros((nx, N+1))
        U_opt = np.zeros((nu, N))
        for k in range(N+1):
            X_opt[:,k] = w_opt[idx:idx+nx]; idx += nx
        for k in range(N):
            U_opt[:,k] = w_opt[idx:idx+nu]; idx += nu
        T_opt = w_opt[idx]

        q_opt  = X_opt[0:nq,:]
        dq_opt = X_opt[nq:2*nq,:]
        tau_opt= U_opt
        tgrid = np.linspace(0, float(T_opt), N+1)
        return {"q": q_opt, "dq": dq_opt, "tau": tau_opt, "tgrid": tgrid, "T": T_opt}

    # ------------------------------------------------------------------
    # Plotting and animation utilities
    # ------------------------------------------------------------------
    def animate(self, q_traj, fps=30, title="RX-90L animation"):
        pts = self.fk_chain(q_traj[0])
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        reach = 1.3
        ax.set_xlim([-reach, reach]); ax.set_ylim([-reach, reach]); ax.set_zlim([0.0, reach*1.2])
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
        ax.set_title(title)
        link_lines = []
        joint_scatter = ax.scatter([], [], [], s=20)
        tcp_trail, = ax.plot([], [], [], lw=1, alpha=0.5)
        for _ in range(6):
            line, = ax.plot([], [], [], lw=3)
            link_lines.append(line)
        trail_pts = []

        def init():
            for line in link_lines:
                line.set_data([], []); line.set_3d_properties([])
            tcp_trail.set_data([], []); tcp_trail.set_3d_properties([])
            return link_lines + [tcp_trail, joint_scatter]

        def update(frame):
            q = q_traj[frame % len(q_traj)]
            pts = self.fk_chain(q)
            xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
            for i, line in enumerate(link_lines):
                line.set_data(xs[i:i+2], ys[i:i+2]); line.set_3d_properties(zs[i:i+2])
            joint_scatter._offsets3d = (xs[:-1], ys[:-1], zs[:-1])
            trail_pts.append(pts[-1]); tp = np.array(trail_pts[-300:])
            tcp_trail.set_data(tp[:,0], tp[:,1]); tcp_trail.set_3d_properties(tp[:,2])
            ax.view_init(elev=25, azim=35 + 0.4*frame)
            return link_lines + [tcp_trail, joint_scatter]

        anim = FuncAnimation(fig, update, frames=len(q_traj), init_func=init, interval=1000/fps, blit=False)
        plt.show()
        return anim

    def plot_trajectory(self, tgrid, q, dq=None, tau=None):
        fig, axs = plt.subplots(3,1,figsize=(10,8), sharex=True)
        axs[0].plot(tgrid, np.rad2deg(q).T); axs[0].set_ylabel('q [deg]')
        if dq is not None:
            axs[1].plot(tgrid, np.rad2deg(dq).T); axs[1].set_ylabel('dq [deg/s]')
        if tau is not None:
            axs[2].plot(tgrid[:-1], tau.T); axs[2].set_ylabel('tau [Nm]')
        axs[2].set_xlabel('t [s]')
        for ax in axs:
            ax.grid(True)
        plt.tight_layout(); plt.show()
        return fig