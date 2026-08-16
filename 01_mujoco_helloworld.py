"""MuJoCo 入门示例：木块自由落体。

每 200 ms 打印一次木块的高度。
"""

import mujoco


XML = """
<mujoco>
    <option timestep="0.002" gravity="0 0 -9.81"/>
    <worldbody>
        <light directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"
               pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="2 2 0.05" pos="0 0 0"
              rgba="0.8 0.8 0.8 1"/>
        <body name="box" pos="0 0 1">
            <freejoint/>
            <geom name="box_geom" type="box" size="0.05 0.05 0.05" mass="1"
                  rgba="0.8 0.3 0.2 1"/>
        </body>
    </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[2] = 100.0  # 初始高度 1 米
    mujoco.mj_forward(model, data)

    duration = 4.0
    print_interval = 0.3
    timestep = model.opt.timestep
    steps_per_print = int(print_interval / timestep)
    total_steps = int(duration / timestep)

    print(f"{'time (s)':>10} {'height (m)':>12}")
    for step in range(total_steps + 1):
        if step % steps_per_print == 0:
            height = data.body("box").xpos[2]
            print(f"{step * timestep:10.2f} {height:12.6f}")

        if step < total_steps:
            mujoco.mj_step(model, data)


if __name__ == "__main__":
    main()
