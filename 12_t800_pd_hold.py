"""使用 PD 控制器，让 t800 的所有电机关节保持在初始位置，并在 viewer 中显示。"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")

KP = 10.0   # 位置误差比例增益
KD = 20.0    # 速度阻尼增益
PRINT_INTERVAL = 0.5


def apply_pd_control(model, data):
    """对每个电机关节计算 PD 控制量，并写入 data.ctrl。"""
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]

        qpos_adr = model.jnt_qposadr[joint_id]
        qvel_adr = model.jnt_dofadr[joint_id]

        current_pos = data.qpos[qpos_adr]
        current_vel = data.qvel[qvel_adr]
        target_pos = model.qpos0[qpos_adr]

        ctrl = KP * (target_pos - current_pos) - KD * current_vel

        ctrl_min = model.actuator_ctrlrange[actuator_id, 0]
        ctrl_max = model.actuator_ctrlrange[actuator_id, 1]
        data.ctrl[actuator_id] = np.clip(ctrl, ctrl_min, ctrl_max)


def max_joint_position_error(model, data):
    """计算所有被控关节中最大的位置误差，用于验证保持效果。"""
    errors = []

    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        qpos_adr = model.jnt_qposadr[joint_id]
        errors.append(abs(model.qpos0[qpos_adr] - data.qpos[qpos_adr]))

    return max(errors)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    timestep = model.opt.timestep
    steps_per_print = int(PRINT_INTERVAL / timestep)

    header = f"{'time(s)':>8} {'base_z(m)':>10} {'max_joint_err(rad)':>20}"
    print(header)
    print("-" * len(header))

    step = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if step % steps_per_print == 0:
                time = step * timestep
                base_z = data.body("LINK_BASE").xpos[2]
                max_error = max_joint_position_error(model, data)
                print(f"{time:8.2f} {base_z:10.4f} {max_error:20.6f}")

            apply_pd_control(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()

            step += 1


if __name__ == "__main__":
    main()
