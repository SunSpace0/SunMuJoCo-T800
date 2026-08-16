"""输入超大 ctrl，观察 MuJoCo 的 ctrlrange 限制是否生效。"""

from pathlib import Path

import mujoco

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")

INPUT_CTRL = 10000.0


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 给所有电机输入同一个超大控制值
    data.ctrl[:] = INPUT_CTRL

    # 执行一步，让 MuJoCo 根据 ctrlrange 生成真实 actuator force
    mujoco.mj_step(model, data)

    print(f"输入 ctrl 值: {INPUT_CTRL}")
    print()

    header = (
        f"{'ID':<3} {'Motor':<26} {'ctrl_min':>10} {'ctrl_max':>10} "
        f"{'raw_ctrl':>12} {'expected_clip':>15} {'actuator_force':>15}"
    )
    print(header)
    print("-" * len(header))

    for actuator_id in range(model.nu):
        motor_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )

        ctrl_min = model.actuator_ctrlrange[actuator_id, 0]
        ctrl_max = model.actuator_ctrlrange[actuator_id, 1]

        raw_ctrl = data.ctrl[actuator_id]
        expected_clip = mujoco.mju_clip(INPUT_CTRL, ctrl_min, ctrl_max)
        actual_force = data.actuator_force[actuator_id]

        print(
            f"{actuator_id:<3} {motor_name:<26} {ctrl_min:>10.1f} {ctrl_max:>10.1f} "
            f"{raw_ctrl:>12.1f} {expected_clip:>15.1f} {actual_force:>15.1f}"
        )

    print("-" * len(header))
    print(f"电机总数: {model.nu}")


if __name__ == "__main__":
    main()
