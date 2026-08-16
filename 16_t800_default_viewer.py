"""使用 MuJoCo 默认 viewer 和默认电机控制滑块仿真 t800。"""

from pathlib import Path

import mujoco
import mujoco.viewer

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    # 打开 MuJoCo 自带 Simulate 界面，右侧面板会显示电机控制滑块
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
