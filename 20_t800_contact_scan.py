"""动态扫描 T800 初始高度与地面接触点的数量。

扫描范围：0.50m ~ 1.50m，步长 2cm。
每个高度只做一次 mj_forward()，不推进动力学，统计机器人身体与大地 floor 的接触点。
"""

from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "t800.xml"

Z_MIN = 0.50
Z_MAX = 1.50
Z_STEP = 0.02

# 沿用 18/19 行走示例里的初始关节姿态。
# 如果想使用 t800.xml 的默认关节姿态，可以把下面改成 False。
USE_HOMING_POSE = True

HOMING_QPOS = np.array([
    0, 0, 0.87, 1, 0, 0, 0,
    0, 0, -0.12, 0.24, -0.12, 0,
    0, 0, -0.12, 0.24, -0.12, 0,
    0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0,
])


def make_initial_qpos(model, data):
    """返回本次扫描使用的初始 qpos，后面只需覆盖 free joint 的高度。"""
    if USE_HOMING_POSE:
        return HOMING_QPOS.copy()

    mujoco.mj_resetData(model, data)
    return data.qpos.copy()


def scan_robot_ground_contacts(model, data, initial_qpos, height):
    """把机器人放在指定高度，返回 (接触点列表, 涉及身体列表)。"""
    data.qpos[:] = initial_qpos

    # free joint: [x, y, z, qw, qx, qy, qz]
    data.qpos[0:3] = [0.0, 0.0, height]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)

    floor_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    if floor_geom_id < 0:
        floor_geom_id = 0

    contact_points = []
    involved_bodies = set()

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2

        if geom1 == floor_geom_id and geom2 != floor_geom_id:
            robot_geom_id = geom2
        elif geom2 == floor_geom_id and geom1 != floor_geom_id:
            robot_geom_id = geom1
        else:
            continue

        body_id = model.geom_bodyid[robot_geom_id]
        if body_id <= 0:
            continue

        body_name = model.body(body_id).name or f"body_{body_id}"
        contact_points.append((body_name, contact.pos.copy()))
        involved_bodies.add(body_name)

    return contact_points, involved_bodies


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    initial_qpos = make_initial_qpos(model, data)

    num_steps = int(round((Z_MAX - Z_MIN) / Z_STEP)) + 1

    print(f"{'height(m)':>10} {'contacts':>9} {'bodies':>7}   contact bodies")
    print("-" * 80)

    first_zero_height = None
    first_nonzero_height = None

    for i in range(num_steps):
        height = Z_MIN + i * Z_STEP
        contact_points, involved_bodies = scan_robot_ground_contacts(
            model, data, initial_qpos, height
        )

        contact_count = len(contact_points)
        body_count = len(involved_bodies)
        body_names = ", ".join(sorted(involved_bodies)) if involved_bodies else "-"

        print(f"{height:10.2f} {contact_count:9d} {body_count:7d}   {body_names}")

        if contact_count == 0 and first_zero_height is None:
            first_zero_height = height
        if contact_count > 0 and first_nonzero_height is None:
            first_nonzero_height = height

    print("-" * 80)
    print(f"第一次出现地面接触的高度: {first_nonzero_height:.2f} m")
    print(f"第一次完全没有地面接触的高度: {first_zero_height:.2f} m")


if __name__ == "__main__":
    main()
