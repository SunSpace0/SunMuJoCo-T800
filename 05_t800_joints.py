"""加载 t800.xml，并打印所有关节信息。"""

from pathlib import Path

import mujoco

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")

JOINT_TYPE_NAMES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    print(f"关节总数: {model.njnt}\n")

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "(unnamed)"
        body_id = model.jnt_bodyid[joint_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "(unnamed)"
        joint_type = JOINT_TYPE_NAMES.get(
            int(model.jnt_type[joint_id]), str(model.jnt_type[joint_id])
        )

        print(
            f"[{joint_id:02d}] {joint_name:<20s} "
            f"type={joint_type:<6s} body={body_name}"
        )


if __name__ == "__main__":
    main()
