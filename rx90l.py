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

    def ik_xyz_dir(self, q0, target_xyz, target_dir, d6=0.0, iters=100, lam=1e-2):
        td = np.asarray(target_dir, float)
        td = td / (np.linalg.norm(td) + 1e-12)

        def err(q):
            _, Ts = self._fk_Ts(q)
            Te = Ts[-1].copy()
            Te[:3, 3] += Te[:3, :3] @ np.array([0.0, 0.0, d6])
            p = Te[:3, 3]
            a = Te[:3, :3] @ np.array([0.0, 0.0, 1.0])
            a_n = a / (np.linalg.norm(a) + 1e-12)
            ex, ey, ez = p - target_xyz
            c = np.cross(a_n, td)
            return np.array([ex, ey, ez, c[0], c[1], c[2]])

        q = q0.copy()
        for _ in range(iters):
            e, J = self._numerical_jacobian(err, q)
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
                break
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(JJt.shape[0]), -e)
            q = q + dq
            q = np.minimum(np.maximum(q, self.q_lim[:, 0]), self.q_lim[:, 1])
        return q

    @staticmethod
    def rpy_to_dir(rpy):
        """Convert roll-pitch-yaw angles to a unit direction vector."""
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        # Rotation matrix for ZYX convention
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
        return R[:, 2]

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
        import numpy as np
        try:
            import casadi as ca
        except ImportError:
            raise RuntimeError("CasADi is required for the OCP demo")

        # -----------------------------
        # helpers (all defs included)
        # -----------------------------
        def dh_T(a, alpha, d, theta):
            ca_, sa_ = ca.cos(alpha), ca.sin(alpha)
            ct, st = ca.cos(theta), ca.sin(theta)
            return ca.vertcat(
                ca.horzcat(ct, -st * ca_, st * sa_, a * ct),
                ca.horzcat(st, ct * ca_, -ct * sa_, a * st),
                ca.horzcat(0, sa_, ca_, d),
                ca.horzcat(0, 0, 0, 1),
            )

        def fk_pose(q):
            T = ca.SX.eye(4)
            # robust to (hypothetical) prismatic entries
            for i, (a, alpha, d, theta0, is_rev) in enumerate(self.DH):
                qi = q[i]
                if is_rev:
                    theta_i = theta0 + qi
                    d_i = d
                else:
                    theta_i = theta0
                    d_i = d + qi
                T = ca.mtimes(T, dh_T(a, alpha, d_i, theta_i))
            return T

        def ee_pos(q):
            return fk_pose(q)[0:3, 3]

        def rpy_to_dir_np(rpy_rad):
            # tool z-axis in world for RPY (Z-Y-X)
            roll, pitch, yaw = rpy_rad
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            R = Rz @ Ry @ Rx
            return R[:, 2]  # z-axis

        def ik_xyz_dir_local(q0, target_p, target_dir, d6=0.10, iters=120, lam=2e-2):
            td = np.asarray(target_dir, float)
            td = td / (np.linalg.norm(td) + 1e-12)

            def err(q):
                _, Ts = self._fk_Ts(q)
                Te = Ts[-1].copy()
                p = (Te[:3, 3] + Te[:3, :3] @ np.array([0.0, 0.0, d6])).astype(float)
                a = (Te[:3, :3] @ np.array([0.0, 0.0, 1.0])).astype(float)
                a_n = a / (np.linalg.norm(a) + 1e-12)
                epos = p - target_p
                eori = np.cross(a_n, td)
                return np.concatenate([epos, eori])

            q = q0.copy()
            for _ in range(iters):
                y0, J = self._numerical_jacobian(err, q)  # uses class helper
                if np.linalg.norm(y0[:3]) < 1e-4 and np.linalg.norm(y0[3:]) < 1e-3:
                    break
                JJt = J @ J.T
                dq = J.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(JJt.shape[0]), -y0)
                q = q + dq
                q = np.minimum(np.maximum(q, self.q_lim[:, 0]), self.q_lim[:, 1])
            return q

        def casadi_forward_dyn():
            q = ca.SX.sym("q", 6)
            dq = ca.SX.sym("dq", 6)
            tau = ca.SX.sym("tau", 6)

            # Gravity via J^T Fg (cheap & more physical)
            p = ee_pos(q)
            Jp = ca.jacobian(p, q)  # 3x6
            m_eff = 3.0
            Fg = ca.vertcat(0, 0, -9.81 * m_eff)
            g_tau = ca.mtimes(Jp.T, Fg)  # 6x1

            B = ca.DM([2.0, 2.0, 1.5, 0.6, 0.4, 0.2])  # visc
            Cc = ca.DM([2.5, 2.0, 1.5, 0.8, 0.6, 0.3])  # Coulomb
            visc = ca.diag(B) @ dq
            coul = ca.diag(Cc) @ ca.tanh(50 * dq)

            Minv = ca.diag(1.0 / ca.DM([7.0, 6.0, 3.5, 1.2, 0.8, 0.4]))
            ddq = Minv @ (tau - visc - coul - g_tau)
            return ca.Function("fd", [q, dq, tau], [ddq])

        fd = casadi_forward_dyn()

        def rk4_step(xk, uk, dt):
            def f(x, u):
                q, dq = x[0:nq], x[nq:2 * nq]
                ddq = fd(q, dq, u)
                return ca.vertcat(dq, ddq)

            k1 = f(xk, uk)
            k2 = f(xk + 0.5 * dt * k1, uk)
            k3 = f(xk + 0.5 * dt * k2, uk)
            k4 = f(xk + dt * k3, uk)
            return xk + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # -----------------------------
        # problem setup
        # -----------------------------
        nq = 6;
        nx = 2 * nq;
        nu = nq
        N = 60
        T_min, T_max = 2.0, 8.0
        T_sym = ca.SX.sym("T")
        dt = T_sym / N
        X = ca.SX.sym("X", nx, N + 1)
        U = ca.SX.sym("U", nu, N)

        # Start/goal via IK (poses in meters & degrees for readability)
        pose_start = np.array([0.6, 0.1, 0.50, 0.0, 0.0, 0.0], float)
        pose_goal = np.array([0.8, 0.3, 0.50, 0.0, 0.0, 90.0], float)
        dir_start = rpy_to_dir_np(np.deg2rad(pose_start[3:]))
        dir_goal = rpy_to_dir_np(np.deg2rad(pose_goal[3:]))

        # use class IK if you have it; else local fallback
        if hasattr(self, "ik_xyz_dir"):
            q_start = self.ik_xyz_dir(np.zeros(nq), pose_start[:3], dir_start, d6=0.10, iters=120, lam=2e-2)
            q_goal = self.ik_xyz_dir(q_start, pose_goal[:3], dir_goal, d6=0.10, iters=120, lam=2e-2)
        else:
            q_start = ik_xyz_dir_local(np.zeros(nq), pose_start[:3], dir_start, d6=0.10, iters=120, lam=2e-2)
            q_goal = ik_xyz_dir_local(q_start, pose_goal[:3], dir_goal, d6=0.10, iters=120, lam=2e-2)
        dq_start = np.zeros(nq)
        dq_goal = np.zeros(nq)

        # -----------------------------
        # constraints (defects enforced)
        # -----------------------------
        g_constr, g_l, g_u = [], [], []

        # initial
        g_constr += [X[0:nq, 0] - ca.DM(q_start)];
        g_l += [ca.DM.zeros(nq)];
        g_u += [ca.DM.zeros(nq)]
        g_constr += [X[nq:2 * nq, 0] - ca.DM(dq_start)];
        g_l += [ca.DM.zeros(nq)];
        g_u += [ca.DM.zeros(nq)]

        # dynamics: multiple shooting defects
        for k in range(N):
            x_next = rk4_step(X[:, k], U[:, k], dt)
            g_constr += [X[:, k + 1] - x_next]
            g_l += [ca.DM.zeros(nx)]
            g_u += [ca.DM.zeros(nx)]

        # terminal
        g_constr += [X[0:nq, -1] - ca.DM(q_goal)];
        g_l += [ca.DM.zeros(nq)];
        g_u += [ca.DM.zeros(nq)]
        g_constr += [X[nq:2 * nq, -1] - ca.DM(dq_goal)];
        g_l += [ca.DM.zeros(nq)];
        g_u += [ca.DM.zeros(nq)]

        # -----------------------------
        # objective: energy + smoothness
        # -----------------------------
        W_mech, W_visc, W_i2r = 0 * 0.1, 0, 0 * 1e-3  # energy terms
        W_du = 0 * 1e-1  # torque-rate smoothing
        eps = 1e-6
        Bvec = ca.DM([2.0, 2.0, 1.5, 0.6, 0.4, 0.2])  # reuse outside loop

        J = 0
        for k in range(N):
            dqk = X[nq:2 * nq, k]
            uk = U[:, k]

            pow_elem = uk * dqk
            P_mech = ca.sum1(ca.sqrt(pow_elem * pow_elem + eps))  # Σ |τ_i ω_i|
            P_visc = ca.dot(Bvec, dqk * dqk)  # Σ b_i ω_i^2
            P_i2r = ca.dot(uk, uk)  # Σ τ_i^2

            stage = W_mech * P_mech + W_visc * P_visc + W_i2r * P_i2r
            J += stage * dt

        # ∫ ||du/dt||^2 dt  ≈  Σ ||Δu||^2 / dt
        J_du = 0
        for k in range(1, N):
            duk = U[:, k] - U[:, k - 1]
            J_du += ca.dot(duk, duk) / dt
        J += W_du * J_du

        # -----------------------------
        # decision vector, bounds, init
        # -----------------------------
        w = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)), T_sym)

        w_l, w_u = [], []
        for _ in range(N + 1):
            w_l += list(self.q_lim[:, 0]);
            w_u += list(self.q_lim[:, 1])  # q
            w_l += list(-np.deg2rad([356, 356, 296, 409, 480, 1125]))  # dq min
            w_u += list(np.deg2rad([356, 356, 296, 409, 480, 1125]))  # dq max
        for _ in range(N):
            w_l += list(-np.array([70, 70, 50, 30, 20, 12]))  # tau min
            w_u += list(np.array([70, 70, 50, 30, 20, 12]))  # tau max
        w_l += [T_min];
        w_u += [T_max]  # time window
        w_l = ca.DM(w_l);
        w_u = ca.DM(w_u)

        g = ca.vertcat(*[gc.reshape((-1, 1)) for gc in g_constr])

        # all constraints are equalities → g == 0
        g_l = ca.DM.zeros(g.size1(), 1)
        g_u = ca.DM.zeros(g.size1(), 1)

        # smooth quintic seed in joint space
        T0 = 0.5 * (T_min + T_max)
        w0 = []
        for k in range(N + 1):
            s = k / float(N)
            sigma = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
            dsigma = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T0
            qk = q_start + sigma * (q_goal - q_start)
            dqk = dsigma * (q_goal - q_start)
            w0 += list(qk);
            w0 += list(dqk)
        for _ in range(N):
            w0 += [0.0] * nu
        w0 += [T0]
        w0 = ca.DM(w0)

        # -----------------------------
        # solve
        # -----------------------------
        nlp = {"x": w, "f": J, "g": g}
        opts = {"ipopt": {
            "print_level": 0, "max_iter": 800,
            "mu_strategy": "adaptive",
            "linear_solver": "mumps",
            "tol": 1e-5, "acceptable_tol": 1e-4
        }}
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(x0=w0, lbx=w_l, ubx=w_u, lbg=g_l, ubg=g_u)
        w_opt = sol["x"].full().flatten()

        # debug
        stats = solver.stats()
        print("status:", stats.get("return_status"))
        print("iter:", stats.get("iter_count"))
        print("obj:", float(sol["f"]))
        print("T_opt:", float(w_opt[-1]))

        # unpack
        idx = 0
        X_opt = np.zeros((nx, N + 1))
        U_opt = np.zeros((nu, N))
        for k in range(N + 1):
            X_opt[:, k] = w_opt[idx:idx + nx];
            idx += nx
        for k in range(N):
            U_opt[:, k] = w_opt[idx:idx + nu];
            idx += nu
        T_opt = w_opt[idx]

        q_opt = X_opt[0:nq, :].T
        dq_opt = X_opt[nq:2 * nq, :].T
        tau_opt = U_opt.T
        tgrid = np.linspace(0, float(T_opt), N + 1)
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
        axs[0].plot(tgrid, np.rad2deg(q)); axs[0].set_ylabel('q [deg]')
        if dq is not None:
            axs[1].plot(tgrid, np.rad2deg(dq)); axs[1].set_ylabel('dq [deg/s]')
        if tau is not None:
            axs[2].plot(tgrid[:-1], tau); axs[2].set_ylabel('tau [Nm]')
        axs[2].set_xlabel('t [s]')
        for ax in axs:
            ax.grid(True)
        plt.tight_layout(); plt.show()
        return fig