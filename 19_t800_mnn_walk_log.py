"""带步态日志的 MNN 行走控制（基于 18）。

每 100ms 打印：
  - base 的 pos / rpy
  - MNN 推理耗时和动作统计
  - PD 控制器的最大/均方根跟踪误差、控制力矩、饱和电机数

同时写入 19_t800_walk_log.csv，方便后续绘图分析。
"""

from pathlib import Path
import csv
import time

import numpy as np
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "t800.xml"
POLICY_PATH = ROOT / "policy" / "t800_260318_150533_60000.mnn"
LOG_CSV_PATH = ROOT / "19_t800_walk_log.csv"

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

COMMAND = np.array([0.4, 0.0, 0.0])
COMMAND_SCALE = np.array([2.0, 2.0, 1.0])

HISTORY_STEPS = 15
CONTROL_DT = 0.01
LOG_INTERVAL = 0.1
OBS_CLIP = 100.0
ACTION_CLIP = 100.0

# 教学默认 headless；需要 viewer 时改为 True，并用 mjpython 运行
USE_VIEWER = False

HOMING_QPOS = np.array([
    0, 0, 0.87, 1, 0, 0, 0,
    0, 0, -0.12, 0.24, -0.12, 0,
    0, 0, -0.12, 0.24, -0.12, 0,
    0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0,
])


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


def xmat_to_rpy_deg(xmat):
    """把 3x3 旋转矩阵转成 roll/pitch/yaw，单位为度。"""
    rotation = xmat.reshape(3, 3)

    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    pitch = np.arcsin(-rotation[2, 0])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])

    return np.degrees([roll, pitch, yaw])


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
    """执行 PD，并返回当前步的误差/力矩统计。"""
    pos_errors_rad = np.zeros(len(controls))
    ctrl_cmds = np.zeros(len(controls))
    ctrl_actuals = np.zeros(len(controls))
    saturated_count = 0

    for i, (control, target) in enumerate(zip(controls, targets)):
        joint_index = ALL_JOINTS.index(control["name"])

        q = data.qpos[control["qpos_adr"]]
        qd = data.qvel[control["qvel_adr"]]

        ctrl_cmd = KP[joint_index] * (target - q) - KD[joint_index] * qd

        actuator_id = control["actuator_id"]
        ctrl_min = model.actuator_ctrlrange[actuator_id, 0]
        ctrl_max = model.actuator_ctrlrange[actuator_id, 1]
        ctrl_actual = np.clip(ctrl_cmd, ctrl_min, ctrl_max)
        data.ctrl[actuator_id] = ctrl_actual

        pos_errors_rad[i] = abs(target - q)
        ctrl_cmds[i] = ctrl_cmd
        ctrl_actuals[i] = ctrl_actual

        if not np.isclose(ctrl_cmd, ctrl_actual, atol=1e-6):
            saturated_count += 1

    return pos_errors_rad, ctrl_cmds, ctrl_actuals, saturated_count


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    policy = MnnPolicy(POLICY_PATH)

    data.qpos[:] = HOMING_QPOS
    mujoco.mj_forward(model, data)

    controls = build_controls(model)
    active_index = {name: i for i, name in enumerate(ACTIVE_JOINTS)}
    targets = DEFAULT_Q.copy()
    prev_action = np.zeros(len(ACTIVE_JOINTS), dtype=np.float32)
    history = None

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_BASE")

    timestep = model.opt.timestep
    steps_per_policy = int(round(CONTROL_DT / timestep))
    steps_per_log = int(round(LOG_INTERVAL / timestep))

    csv_file = LOG_CSV_PATH.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_header = [
        "time", "x", "y", "z", "roll", "pitch", "yaw",
        "infer_ms", "action_abs_mean", "action_abs_max",
        "max_pos_err_deg", "rms_pos_err_deg",
        "max_ctrl_cmd", "max_ctrl_actual", "saturated_count",
        "max_pos_err_joint", "max_ctrl_joint", "leg_actions",
    ]
    csv_writer.writerow(csv_header)

    console_header = (
        f"{'time':>6} {'x':>7} {'y':>7} {'z':>7} "
        f"{'roll':>8} {'pitch':>8} {'yaw':>8} "
        f"{'infer_ms':>9} {'act_mean':>9} {'act_max':>9} "
        f"{'max_qerr':>9} {'rms_qerr':>9} "
        f"{'ctrl_cmd':>9} {'ctrl_act':>9} {'sat':>4}"
    )
    print(console_header)

    step = 0
    last_infer_ms = 0.0
    last_action = np.zeros(len(ACTIVE_JOINTS), dtype=np.float32)

    max_pos_err_deg = 0.0
    max_pos_err_joint = "-"
    sum_sq_pos_err_deg = 0.0
    pos_err_count = 0
    max_ctrl_cmd = 0.0
    max_ctrl_cmd_joint = "-"
    max_ctrl_actual = 0.0
    saturated_count = 0

    def one_step():
        nonlocal history, prev_action, step
        nonlocal last_infer_ms, last_action
        nonlocal max_pos_err_deg, max_pos_err_joint, sum_sq_pos_err_deg, pos_err_count
        nonlocal max_ctrl_cmd, max_ctrl_cmd_joint, max_ctrl_actual, saturated_count

        if step % steps_per_policy == 0:
            single_obs = build_single_obs(model, data, controls, prev_action)

            if history is None:
                history = np.tile(single_obs, (HISTORY_STEPS, 1))
            else:
                history[:-1] = history[1:]
                history[-1] = single_obs

            obs = np.concatenate([
                history.flatten(),
                COMMAND * COMMAND_SCALE,
            ])

            infer_start = time.perf_counter()
            action = np.clip(policy.infer(obs), -ACTION_CLIP, ACTION_CLIP)
            last_infer_ms = (time.perf_counter() - infer_start) * 1000.0
            last_action = action
            prev_action = action

            for i, control in enumerate(controls):
                joint_index = ALL_JOINTS.index(control["name"])
                if control["name"] in active_index:
                    action_index = active_index[control["name"]]
                    targets[i] = (
                        DEFAULT_Q[joint_index]
                        + action[action_index] * ACTION_SCALE[action_index]
                    )
                else:
                    targets[i] = DEFAULT_Q[joint_index]

        pos_errors_rad, ctrl_cmds, ctrl_actuals, current_saturated = (
            apply_pd_control(model, data, controls, targets)
        )

        pos_errors_deg = np.degrees(pos_errors_rad)
        for i, control in enumerate(controls):
            err = pos_errors_deg[i]
            if err > max_pos_err_deg:
                max_pos_err_deg = err
                max_pos_err_joint = control["name"]

            sum_sq_pos_err_deg += err * err
            pos_err_count += 1

            abs_cmd = abs(ctrl_cmds[i])
            if abs_cmd > max_ctrl_cmd:
                max_ctrl_cmd = abs_cmd
                max_ctrl_cmd_joint = control["name"]

            max_ctrl_actual = max(max_ctrl_actual, abs(ctrl_actuals[i]))

        saturated_count += current_saturated

        if step % steps_per_log == 0:
            pos = data.xpos[base_id]
            roll, pitch, yaw = xmat_to_rpy_deg(data.xmat[base_id])

            action_abs_mean = float(np.mean(np.abs(last_action)))
            action_abs_max = float(np.max(np.abs(last_action)))
            rms_pos_err_deg = (
                float(np.sqrt(sum_sq_pos_err_deg / pos_err_count))
                if pos_err_count > 0 else 0.0
            )

            leg_actions = last_action[:12]
            leg_actions_str = np.array2string(
                leg_actions, precision=3, suppress_small=True
            )

            row = [
                step * timestep, pos[0], pos[1], pos[2],
                roll, pitch, yaw,
                last_infer_ms, action_abs_mean, action_abs_max,
                max_pos_err_deg, rms_pos_err_deg,
                max_ctrl_cmd, max_ctrl_actual, saturated_count,
                max_pos_err_joint, max_ctrl_cmd_joint, leg_actions_str,
            ]
            csv_writer.writerow(row)

            print(
                f"{row[0]:6.2f} {row[1]:7.3f} {row[2]:7.3f} {row[3]:7.3f} "
                f"{row[4]:8.2f} {row[5]:8.2f} {row[6]:8.2f} "
                f"{row[7]:9.3f} {row[8]:9.4f} {row[9]:9.4f} "
                f"{row[10]:9.3f} {row[11]:9.3f} "
                f"{row[12]:9.2f} {row[13]:9.2f} {row[14]:4d}"
            )
            print(f"      leg_actions: {leg_actions_str}")
            print(f"      max_qerr_joint={row[15]}, max_ctrl_joint={row[16]}")

            max_pos_err_deg = 0.0
            max_pos_err_joint = "-"
            sum_sq_pos_err_deg = 0.0
            pos_err_count = 0
            max_ctrl_cmd = 0.0
            max_ctrl_cmd_joint = "-"
            max_ctrl_actual = 0.0
            saturated_count = 0

        mujoco.mj_step(model, data)
        step += 1

    if USE_VIEWER:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                one_step()
                viewer.sync()
    else:
        total_steps = int(10.0 / timestep)
        for _ in range(total_steps + 1):
            one_step()

    csv_file.close()
    print(f"日志已保存: {LOG_CSV_PATH}")


if __name__ == "__main__":
    main()
