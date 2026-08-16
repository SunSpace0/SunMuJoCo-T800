"""遍历 t800 的所有关节电机，并打印 ctrlrange。"""

from pathlib import Path

import mujoco

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    header = f"{'ID':<3} {'Motor':<26} {'Joint':<22} {'ctrl_min':>10} {'ctrl_max':>10}"
    print(header)
    print("-" * len(header))

    for actuator_id in range(model.nu):
        motor_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )

        joint_id = model.actuator_trnid[actuator_id, 0]
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )

        ctrl_min = model.actuator_ctrlrange[actuator_id, 0]
        ctrl_max = model.actuator_ctrlrange[actuator_id, 1]

        print(
            f"{actuator_id:<3} {motor_name:<26} {joint_name:<22} "
            f"{ctrl_min:>10.1f} {ctrl_max:>10.1f}"
        )

    print("-" * len(header))
    print(f"电机总数: {model.nu}")


if __name__ == "__main__":
    main()
