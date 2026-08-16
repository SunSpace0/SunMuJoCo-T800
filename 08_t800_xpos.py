"""加载 t800.xml，并打印每个 body 的世界坐标 xpos。"""

from pathlib import Path

import mujoco

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")


def body_name(model, body_id):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return name if name is not None else "world"


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 先计算运动学，data.xpos 才有当前 qpos 下的世界坐标
    mujoco.mj_forward(model, data)

    header = f"{'ID':<3} {'Body':<22} {'xpos (x, y, z)':>30}"
    print(header)
    print("-" * len(header))

    for body_id in range(model.nbody):
        name = body_name(model, body_id)
        pos = data.xpos[body_id]
        print(
            f"{body_id:<3} {name:<22} "
            f"({pos[0]:>10.4f}, {pos[1]:>10.4f}, {pos[2]:>10.4f})"
        )

    print("-" * len(header))
    print(f"body 总数: {model.nbody}")


if __name__ == "__main__":
    main()
