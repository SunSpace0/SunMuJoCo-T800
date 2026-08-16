"""通过 MuJoCo viewer 展示 t800 机器人模型。"""

from pathlib import Path

import mujoco
import mujoco.viewer

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
