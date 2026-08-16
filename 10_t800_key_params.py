"""加载 t800.xml，并打印关键状态量与控制量的数量和初始值。"""

from pathlib import Path

import mujoco

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")


def print_vector(name, values):
    text = ", ".join(f"{value: .6f}" for value in values)
    print(f"{name} 数量: {len(values)}")
    print(f"{name} 初始值:")
    print(f"  {text}")
    print()


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    print("模型维度参数")
    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")
    print()

    print("状态量与控制量")
    print_vector("qpos", data.qpos)
    print_vector("qvel", data.qvel)
    print_vector("ctrl", data.ctrl)


if __name__ == "__main__":
    main()
