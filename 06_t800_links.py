"""加载 t800.xml，并打印所有连杆的名称、尺寸和重量信息。"""

from pathlib import Path

import mujoco

# 脚本和 t800.xml 在同一个目录下
XML_PATH = Path(__file__).with_name("t800.xml")

GEOM_TYPE_NAMES = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}


def body_name(model, body_id):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return name if name is not None else "world"


def geom_size_str(model, geom_id):
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    type_name = GEOM_TYPE_NAMES.get(geom_type, f"type{geom_type}")

    x, y, z = (f"{value:.4f}" for value in size[:3])

    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return f"sphere(r={x})"
    if geom_type in {
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    }:
        return f"{type_name}(r={x}, h={y})"
    if geom_type in {
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_MESH),
    }:
        return f"{type_name}({x} x {y} x {z})"

    return f"{type_name}({x}, {y}, {z})"


def link_size_str(model, body_id):
    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] == body_id
    ]

    if not geom_ids:
        return "-"

    return "; ".join(geom_size_str(model, geom_id) for geom_id in geom_ids)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    header = f"{'ID':<3} {'Link':<22} {'Mass(kg)':>10}  Size"
    print(header)
    print("-" * len(header))

    # 0 号 body 是 world，不打印；从 1 号开始才是机器人连杆
    for link_id in range(1, model.nbody):
        link_name = body_name(model, link_id)
        mass = model.body_mass[link_id]
        size = link_size_str(model, link_id)

        print(f"{link_id:<3} {link_name:<22} {mass:>10.4f}  {size}")

    print("-" * len(header))
    print(f"连杆总数: {model.nbody - 1}")


if __name__ == "__main__":
    main()
