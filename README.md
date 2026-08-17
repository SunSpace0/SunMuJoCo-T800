# SunMuJoCo-T800

T800 人形机器人的 MuJoCo 示例项目，包含模型加载、参数查看、关节/连杆信息、MNN 策略行走、WASD 控制、动态物体交互等教学示例。

## 环境

本项目在 macOS 上使用以下环境开发：

- Python 3.11
- conda 环境：`t800`
- MuJoCo
- NumPy
- MNN
- glfw
- matplotlib

激活环境：

```bash
conda activate t800
```

安装依赖：

```bash
pip install mujoco numpy MNN glfw matplotlib
```

## 运行说明

普通无 GUI 示例直接使用 Python：

```bash
cd "/Users/sunyu/Desktop/T800&MuJoCo"
python 02_t800_info.py
```

涉及 MuJoCo viewer 的示例，在 macOS 下需要使用 `mjpython`：

```bash
cd "/Users/sunyu/Desktop/T800&MuJoCo"
mjpython 21_t800_wasd_walk.py
```

## 文件说明

| 文件 | 功能 |
| --- | --- |
| `02_t800_info.py` | 打印身体部件、关节和电机数量 |
| `03_t800_nq_nv_nu.py` | 打印 `nq`、`nv`、`nu` |
| `04_t800_viewer.py` | 在 viewer 中显示 T800 |
| `05_t800_joints.py` | 打印所有关节 |
| `06_t800_links.py` | 打印所有连杆 |
| `07_t800_body_pos.py` | 打印每个 body 的局部位置 |
| `08_t800_xpos.py` | 打印每个 body 的世界坐标 |
| `09_t800_skeleton.py` | 用 matplotlib 绘制机器人骨架 |
| `10_t800_key_params.py` | 打印 `nq/nv/nu`、`qpos/qvel/ctrl` |
| `11_t800_sim_log.py` | 仿真过程中定时打印 `pos/rpy` |
| `12_t800_pd_hold.py` | PD 控制器保持关节位置 |
| `13_t800_ctrlrange.py` | 遍历打印电机 `ctrlrange` |
| `14_t800_ctrl_clamp.py` | 测试控制量 clamp |
| `15_t800_manual_control.py` | 手动控制单个关节 |
| `16_t800_default_viewer.py` | 使用默认 viewer 和控制滑块 |
| `17_t800_action_test.py` | MNN 动作输出测试 |
| `18_t800_mnn_walk.py` | MNN 策略前向行走 |
| `19_t800_mnn_walk_log.py` | MNN 行走并输出步态日志 |
| `20_t800_contact_scan.py` | 扫描初始高度与地面接触点数量 |
| `21_t800_wasd_walk.py` | WASD 控制机器人移动 |
| `22_t800_mnn_wasd_control.py` | WASD 控制，并动态添加 8cm 木块 |
| `23_t800_mnn_kick_cube.py` | AI 自动踢动 15cm 木块前进 |
| `24_t800_wasd_wrist_cube.py` | WASD 控制，木块绑定到右手腕 |

## 资源目录

- `t800.xml`：主模型入口
- `robot/`：机器人 XML、URDF、网格和纹理资源
- `environment/`：地面等环境资源
- `policy/`：MNN 行走策略模型

## 常见问题

### 1. viewer 启动失败

macOS 上如果直接运行：

```bash
python 21_t800_wasd_walk.py
```

可能报错：

```text
launch_passive requires that the Python script be run under mjpython on macOS
```

请改用：

```bash
mjpython 21_t800_wasd_walk.py
```

### 2. 机器人初始高度

示例默认使用 `INITIAL_BASE_HEIGHT = 1.04`，可通过 `20_t800_contact_scan.py` 扫描接触点，分析不同初始高度的效果。
