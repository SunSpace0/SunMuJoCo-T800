"""手动控制 t800 的 25 个电机关节（教学版）。

操作：
  左/右方向键：选择上一个/下一个关节
  上/下方向键：增大/减小当前关节目标角度
  R 键：恢复所有关节到初始角度
  viewer 内鼠标拖拽：施加物理扰动
"""

from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")

KP = 300.0
KD = 20.0
ANGLE_STEP = np.deg2rad(5.0)


def get_joint_controls(model):
    """把每个电机映射到它的关节 qpos/qvel 地址。"""
    controls = []

    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        controls.append({
            "actuator_id": actuator_id,
            "joint_id": joint_id,
            "qpos_adr": model.jnt_qposadr[joint_id],
            "qvel_adr": model.jnt_dofadr[joint_id],
        })

    return controls


def apply_pd_control(model, data, controls, targets):
    """驱动所有关节朝 targets 运动。"""
    for index, control in enumerate(controls):
        qpos_adr = control["qpos_adr"]
        qvel_adr = control["qvel_adr"]
        actuator_id = control["actuator_id"]

        ctrl = KP * (targets[index] - data.qpos[qpos_adr]) - KD * data.qvel[qvel_adr]
        ctrl_min = model.actuator_ctrlrange[actuator_id, 0]
        ctrl_max = model.actuator_ctrlrange[actuator_id, 1]
        data.ctrl[actuator_id] = np.clip(ctrl, ctrl_min, ctrl_max)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 教学模式：关闭重力，避免机器人在手动调节时直接摔倒
    model.opt.gravity[:] = 0.0

    controls = get_joint_controls(model)
    initial_targets = [model.qpos0[item["qpos_adr"]] for item in controls]
    targets = list(initial_targets)
    selected = 0

    def key_callback(key):
        nonlocal selected, targets

        if key == glfw.KEY_RIGHT:
            selected = (selected + 1) % len(controls)
        elif key == glfw.KEY_LEFT:
            selected = (selected - 1) % len(controls)
        elif key == glfw.KEY_UP:
            targets[selected] += ANGLE_STEP
        elif key == glfw.KEY_DOWN:
            targets[selected] -= ANGLE_STEP
        elif key == glfw.KEY_R:
            targets = list(initial_targets)

        joint_id = controls[selected]["joint_id"]
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        print(
            f"当前关节 [{selected:02d}] {joint_name}: "
            f"target = {targets[selected]:.4f} rad"
        )

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            apply_pd_control(model, data, controls, targets)
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
