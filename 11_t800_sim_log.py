"""仿真 10 秒，每 100 ms 打印一次机器人基座的 pos 和 rpy。"""

from pathlib import Path

import mujoco
import numpy as np

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")

PRINT_INTERVAL = 0.1  # 100 ms
SIM_DURATION = 10.0   # 10 s
BASE_BODY_NAME = "LINK_BASE"


def xmat_to_rpy_deg(xmat):
    """把 body 的 3x3 旋转矩阵转成 roll/pitch/yaw，单位为度。"""
    rotation = xmat.reshape(3, 3)

    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    pitch = np.arcsin(-rotation[2, 0])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])

    return np.degrees([roll, pitch, yaw])


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 先更新运动学，确保 t=0 时刻的 xpos/xmat 已正确初始化
    mujoco.mj_forward(model, data)

    timestep = model.opt.timestep
    steps_per_print = int(PRINT_INTERVAL / timestep)
    total_steps = int(SIM_DURATION / timestep)

    print(f"仿真时长: {SIM_DURATION} s")
    print(f"打印间隔: {PRINT_INTERVAL} s")
    print(f"base body: {BASE_BODY_NAME}\n")

    header = (
        f"{'time(s)':>8} "
        f"{'x(m)':>10} {'y(m)':>10} {'z(m)':>10} "
        f"{'roll(deg)':>11} {'pitch(deg)':>12} {'yaw(deg)':>10}"
    )
    print(header)
    print("-" * len(header))

    for step in range(total_steps + 1):
        if step % steps_per_print == 0:
            time = step * timestep
            pos = data.body(BASE_BODY_NAME).xpos
            roll, pitch, yaw = xmat_to_rpy_deg(data.body(BASE_BODY_NAME).xmat)

            print(
                f"{time:8.2f} "
                f"{pos[0]:10.4f} {pos[1]:10.4f} {pos[2]:10.4f} "
                f"{roll:11.4f} {pitch:12.4f} {yaw:10.4f}"
            )

        if step < total_steps:
            mujoco.mj_step(model, data)


if __name__ == "__main__":
    main()
