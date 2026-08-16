"""WASD 控制 t800 行走，并动态添加一个 8cm 红色正方体木块。

木块通过读取 t800.xml 字符串后注入 <worldbody>，不会修改 t800.xml 文件。

按键：
  W / S：前进 / 后退
  A / D：左转 / 右转
  Q / E：左移 / 右移
  R    ：停止，并把速度命令归零

macOS 下请用 mjpython 运行：
    mjpython 22_t800_mnn_wasd_control.py
"""

from pathlib import Path

import glfw
import numpy as np
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "t800.xml"
POLICY_PATH = ROOT / "policy" / "t800_260318_150533_60000.mnn"

# ---------------- 模型参数 ----------------
LEFT_LEG = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
]
RIGHT_LEG = [
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
]
WAIST = ["J12_TORSO_YAW"]
LEFT_ARM = [
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
]
RIGHT_ARM = [
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
]
HEAD = ["J23_HEAD_PITCH", "J24_HEAD_YAW"]

ACTIVE_JOINTS = LEFT_LEG + RIGHT_LEG + LEFT_ARM + RIGHT_ARM
ALL_JOINTS = LEFT_LEG + RIGHT_LEG + WAIST + LEFT_ARM + RIGHT_ARM + HEAD

DEFAULT_Q = np.array(
    [-0.06, 0.0, 0.0, 0.12, -0.06, 0.0] * 2
    + [0.0]
    + [0.0, 0.15, 0.0, -0.25, 0.0]
    + [0.0, -0.15, 0.0, -0.25, 0.0]
    + [0.0, 0.0],
)

KP = np.array(
    [180, 100, 100, 180, 40, 40] * 2
    + [100]
    + [60, 50, 50, 60, 50]
    + [60, 50, 50, 60, 50]
    + [100, 100],
)

KD = np.array(
    [5.0, 3.0, 3.0, 5.0, 0.3, 0.3] * 2
    + [5.0]
    + [1.8, 1.5, 1.5, 1.8, 1.2]
    + [1.8, 1.5, 1.5, 1.8, 1.2]
    + [1.0, 1.0],
)

ACTIVE_DEFAULT_Q = np.array(
    [-0.06, 0.0, 0.0, 0.12, -0.06, 0.0] * 2
    + [0.0, 0.15, 0.0, -0.25, 0.0]
    + [0.0, -0.15, 0.0, -0.25, 0.0],
)

ACTION_SCALE = np.array(
    [0.5, 0.2, 0.2, 0.5, 0.5, 0.2] * 2
    + [0.2, 0.2, 0.05, 0.2, 0.05]
    + [0.2, 0.2, 0.05, 0.2, 0.05],
)

OBS_SCALE = np.array(
    [1.0] * 22 + [0.05] * 22 + [1.0] * 22 + [1.0] * 3 + [1.0] * 3
)

COMMAND_SCALE = np.array([2.0, 2.0, 1.0])
HISTORY_STEPS = 15
CONTROL_DT = 0.01
OBS_CLIP = 100.0
ACTION_CLIP = 100.0

# 从 20 的扫描结果看，homing 姿态在 1.04m 开始无初始地面穿透。
INITIAL_BASE_HEIGHT = 1.04

# 8cm 正方体木块：MuJoCo box 的 size 是半边长，所以填 0.04m。
CUBE_POS = [0.6, 0.0, 0.04]
CUBE_HALF_SIZE = 0.04
CUBE_MASS = 0.2

VX_STEP = 0.2
VY_STEP = 0.2
YAW_RATE_STEP = 0.4
VX_LIMIT = 0.8
VY_LIMIT = 0.4
YAW_RATE_LIMIT = 1.0

HOMING_QPOS = np.array([
    0, 0, 0.87, 1, 0, 0, 0,
    0, 0, -0.12, 0.24, -0.12, 0,
    0, 0, -0.12, 0.24, -0.12, 0,
    0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0,
])



def load_model_with_cube():
    """读取 t800.xml 文本，在内存中注入红色木块，再编译成模型。"""
    xml_text = XML_PATH.read_text()

    cube_xml = f"""
    <worldbody>
        <body name="red_cube" pos="{CUBE_POS[0]} {CUBE_POS[1]} {CUBE_POS[2]}">
            <geom name="red_cube_geom"
                  type="box"
                  size="{CUBE_HALF_SIZE} {CUBE_HALF_SIZE} {CUBE_HALF_SIZE}"
                  mass="{CUBE_MASS}"
                  rgba="1 0 0 1"/>
            <freejoint name="red_cube_joint"/>
        </body>
    </worldbody>
    """

    xml_text = xml_text.replace("</mujoco>", cube_xml + "</mujoco>")
    return mujoco.MjModel.from_xml_string(xml_text)


class MnnPolicy:
    """最小 MNN 推理封装。"""

    def __init__(self, path):
        import MNN

        self.MNN = MNN
        self.net = MNN.Interpreter(str(path))
        self.session = self.net.createSession()
        self.input_tensor = self.net.getSessionInput(self.session)
        self.output_tensor = self.net.getSessionOutput(self.session)

    def infer(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)

        host_input = self.MNN.Tensor(
            self.input_tensor.getShape(),
            self.MNN.Halide_Type_Float,
            obs,
            self.MNN.Tensor_DimensionType_Caffe,
        )
        self.input_tensor.copyFrom(host_input)
        self.net.runSession(self.session)

        output_shape = self.output_tensor.getShape()
        host_output = self.MNN.Tensor(
            output_shape,
            self.MNN.Halide_Type_Float,
            np.zeros(output_shape, dtype=np.float32),
            self.MNN.Tensor_DimensionType_Caffe,
        )
        self.output_tensor.copyToHostTensor(host_output)

        return np.asarray(host_output.getData(), dtype=np.float32).reshape(-1)


def build_controls(model):
    controls = []
    for name in ALL_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_" + name
        )
        controls.append({
            "name": name,
            "actuator_id": actuator_id,
            "qpos_adr": model.jnt_qposadr[joint_id],
            "qvel_adr": model.jnt_dofadr[joint_id],
        })
    return controls


def build_single_obs(model, data, controls, prev_action):
    active_index = {name: i for i, name in enumerate(ACTIVE_JOINTS)}

    q = np.array([
        data.qpos[c["qpos_adr"]]
        for c in controls
        if c["name"] in active_index
    ])
    qd = np.array([
        data.qvel[c["qvel_adr"]]
        for c in controls
        if c["name"] in active_index
    ])

    imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    gyro_sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_angular_velocity"
    )
    gyro_adr = model.sensor_adr[gyro_sensor_id]

    rotation = data.site_xmat[imu_site_id].reshape(3, 3)
    angular_velocity = data.sensordata[gyro_adr:gyro_adr + 3]
    projected_gravity = -rotation.T @ np.array([0.0, 0.0, 1.0])

    single_obs = np.concatenate([
        q - ACTIVE_DEFAULT_Q,
        qd,
        prev_action,
        angular_velocity,
        projected_gravity,
    ])

    return np.clip(single_obs * OBS_SCALE, -OBS_CLIP, OBS_CLIP)


def apply_pd_control(model, data, controls, targets):
    for control, target in zip(controls, targets):
        q = data.qpos[control["qpos_adr"]]
        qd = data.qvel[control["qvel_adr"]]

        joint_index = ALL_JOINTS.index(control["name"])
        ctrl = KP[joint_index] * (target - q) - KD[joint_index] * qd

        actuator_id = control["actuator_id"]
        ctrl_min = model.actuator_ctrlrange[actuator_id, 0]
        ctrl_max = model.actuator_ctrlrange[actuator_id, 1]
        data.ctrl[actuator_id] = np.clip(ctrl, ctrl_min, ctrl_max)


def main():
    model = load_model_with_cube()
    data = mujoco.MjData(model)
    policy = MnnPolicy(POLICY_PATH)

    # 初始化机器人本体姿态，同时初始化木块的 free joint。
    data.qpos[:] = 0.0
    data.qpos[:len(HOMING_QPOS)] = HOMING_QPOS
    data.qpos[2] = INITIAL_BASE_HEIGHT

    cube_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint"
    )
    cube_qpos_adr = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_qpos_adr:cube_qpos_adr + 7] = [
        CUBE_POS[0], CUBE_POS[1], CUBE_POS[2], 1.0, 0.0, 0.0, 0.0,
    ]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    cube_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "red_cube_geom"
    )
    print(
        "已动态加入木块: "
        f"body=red_cube, geom_id={cube_geom_id}, "
        f"size={CUBE_HALF_SIZE * 2:.2f}m x {CUBE_HALF_SIZE * 2:.2f}m x {CUBE_HALF_SIZE * 2:.2f}m"
    )

    controls = build_controls(model)
    active_index = {name: i for i, name in enumerate(ACTIVE_JOINTS)}
    targets = DEFAULT_Q.copy()
    prev_action = np.zeros(len(ACTIVE_JOINTS), dtype=np.float32)
    history = None

    timestep = model.opt.timestep
    steps_per_policy = int(round(CONTROL_DT / timestep))
    step = 0

    command = np.zeros(3, dtype=np.float64)

    def key_callback(key):
        nonlocal command

        if key == glfw.KEY_W:
            command[0] = min(VX_LIMIT, command[0] + VX_STEP)
        elif key == glfw.KEY_S:
            command[0] = max(-VX_LIMIT, command[0] - VX_STEP)
        elif key == glfw.KEY_A:
            command[2] = min(YAW_RATE_LIMIT, command[2] + YAW_RATE_STEP)
        elif key == glfw.KEY_D:
            command[2] = max(-YAW_RATE_LIMIT, command[2] - YAW_RATE_STEP)
        elif key == glfw.KEY_Q:
            command[1] = min(VY_LIMIT, command[1] + VY_STEP)
        elif key == glfw.KEY_E:
            command[1] = max(-VY_LIMIT, command[1] - VY_STEP)
        elif key == glfw.KEY_R:
            command[:] = 0.0

        print(
            "速度命令 [vx, vy, yaw_rate] = "
            f"[{command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}]"
        )

    def one_step():
        nonlocal history, prev_action, step

        if step % steps_per_policy == 0:
            single_obs = build_single_obs(model, data, controls, prev_action)

            if history is None:
                history = np.tile(single_obs, (HISTORY_STEPS, 1))
            else:
                history[:-1] = history[1:]
                history[-1] = single_obs

            obs = np.concatenate([
                history.flatten(),
                command * COMMAND_SCALE,
            ])
            action = np.clip(policy.infer(obs), -ACTION_CLIP, ACTION_CLIP)
            prev_action = action

            for i, control in enumerate(controls):
                if control["name"] in active_index:
                    action_index = active_index[control["name"]]
                    joint_index = ALL_JOINTS.index(control["name"])
                    targets[i] = (
                        DEFAULT_Q[joint_index]
                        + action[action_index] * ACTION_SCALE[action_index]
                    )
                else:
                    targets[i] = DEFAULT_Q[joint_index]

        apply_pd_control(model, data, controls, targets)
        mujoco.mj_step(model, data)
        step += 1

    print("WASD 控制已启动：W/S 前进后退，A/D 左右转，Q/E 左右平移，R 停止")
    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        # 关闭默认的 body 跟踪，避免机器人移动时地面/场景跟着镜头联动。
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.trackbodyid = -1
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

        while viewer.is_running():
            one_step()
            viewer.sync()


if __name__ == "__main__":
    main()
