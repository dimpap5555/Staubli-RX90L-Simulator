import numpy as np
from rx90l import Rx90L


def main():
    robot = Rx90L()

    # # Forward kinematics example
    # T_fk = 2.0
    # fps = 30
    # q_fk = robot.forward_demo(T_final=T_fk, fps=fps)
    # t_fk = np.linspace(0, T_fk, q_fk.shape[0])
    # robot.plot_trajectory(t_fk, q_fk)
    # robot.animate(q_fk, title="Forward kinematics demo")
    #
    # # Inverse kinematics example
    # q_ik = robot.ik_demo(samples=120)
    # t_ik = np.linspace(0, q_ik.shape[0] / fps, q_ik.shape[0])
    # robot.plot_trajectory(t_ik, q_ik)
    # robot.animate(q_ik, title="Inverse kinematics demo")

    # Trajectory optimisation example
    try:
        res = robot.ocp_demo()
        robot.plot_trajectory(res["tgrid"], res["q"], res["dq"], res["tau"])
        robot.animate(res["q"], title="OCP demo")
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()