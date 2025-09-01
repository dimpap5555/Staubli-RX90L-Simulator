from rx90l import Rx90L


def main():
    robot = Rx90L()

    # Forward kinematics example
    q_fk = robot.forward_demo(T_final=2.0, fps=30)
    robot.animate(q_fk, title="Forward kinematics demo")

    # Inverse kinematics example
    q_ik = robot.ik_demo(samples=120)
    robot.animate(q_ik, title="Inverse kinematics demo")

    # Trajectory optimisation example
    try:
        res = robot.ocp_demo()
        robot.plot_trajectory(res["tgrid"], res["q"], res["dq"], res["tau"])
        robot.animate(res["q"].T, title="OCP demo")
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()