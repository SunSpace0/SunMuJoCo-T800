"""加载 t800.xml，并用 matplotlib 绘制带关键点标注的机器人 3D 骨架。"""

from pathlib import Path

import matplotlib.pyplot as plt
import mujoco

# macOS 中文字体配置，避免中文乱码
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "Heiti SC",
    "SimHei",
]
plt.rcParams["axes.unicode_minus"] = False

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")

# 需要重点标注的关键部位 / 末端部位
KEY_BODY_LABELS = {
    "LINK_BASE": "基座",
    "LINK_WAIST_YAW": "腰部",
    "LINK_HEAD_YAW": "头部",
    "LINK_FOOT_L": "左脚末端",
    "LINK_FOOT_R": "右脚末端",
    "LINK_WRIST_END_L": "左手末端",
    "LINK_WRIST_END_R": "右手末端",
}


def body_id_by_name(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 计算运动学，得到每个 body 的世界坐标
    mujoco.mj_forward(model, data)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 普通 body 节点：跳过 0 号 world
    ax.scatter(
        data.xpos[1:, 0],
        data.xpos[1:, 1],
        data.xpos[1:, 2],
        s=30,
        c="#4C72B0",
        depthshade=True,
        label="普通 body",
    )

    # 画父子 body 之间的骨架连线
    for body_id in range(1, model.nbody):
        parent_id = model.body_parentid[body_id]
        x1, y1, z1 = data.xpos[parent_id]
        x2, y2, z2 = data.xpos[body_id]

        ax.plot(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color="#7F7F7F",
            linewidth=1.5,
            alpha=0.8,
        )

    # 查找并高亮关键 / 末端节点
    key_ids = []
    for name, label in KEY_BODY_LABELS.items():
        body_id = body_id_by_name(model, name)
        if body_id < 0:
            print(f"警告: 找不到 body: {name}")
            continue

        key_ids.append(body_id)
        x, y, z = data.xpos[body_id]
        ax.scatter(
            [x],
            [y],
            [z],
            s=90,
            c="#D62728",
            marker="*",
            depthshade=False,
            edgecolors="black",
            linewidths=0.6,
            label=label if len(key_ids) == 1 else None,
        )

        # 标注名称，根据左右位置做轻微偏移，避免镜像文字重叠
        y_offset = 0.03 if y >= 0 else -0.03
        ax.text(
            x + 0.02,
            y + y_offset,
            z + 0.04,
            label,
            color="#333333",
            fontsize=9,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("T800 机器人骨架结构（含关键/末端标注）")

    # 使用等比例坐标，保持真实形状；若只想拉伸显示，可删除这一行
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-65)

    # 关键节点单独做图例，避免 7 个关键点都重复出现
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
