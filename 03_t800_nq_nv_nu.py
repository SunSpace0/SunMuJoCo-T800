"""加载 t800.xml，并打印 MuJoCo 模型的 nq / nv / nu 参数。"""

from pathlib import Path

import mujoco

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    print(f"XML 文件: {XML_PATH}")
    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")


if __name__ == "__main__":
    main()
