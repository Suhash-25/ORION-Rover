# 🚀 ORION-Rover

> **ORION** (Optimized Robotic Intelligent Off-road Navigator) is a four-wheel autonomous rover being developed as part of the **ARES (Adaptive Rover Exploration System)** project.

ORION is built using **ROS 2 Jazzy**, **Gazebo Harmonic**, and **RViz2** with a modular Xacro-based architecture. The long-term objective is to develop a research-grade autonomous rover capable of perception, localization, navigation, and adaptive decision-making using **Meta Reinforcement Learning**.

---

# 📸 Demo

> **Coming Soon**

- 📷 Gazebo Simulation Screenshot
- 📷 RViz Visualization
- 🎥 Keyboard Teleoperation GIF

---

# 🌟 Features

- ✅ Modular URDF/Xacro Robot Description
- ✅ Four-Wheel Differential Drive Rover
- ✅ Gazebo Harmonic Simulation
- ✅ RViz2 Visualization
- ✅ ros2_control Integration
- ✅ Differential Drive Controller
- ✅ Keyboard Teleoperation
- ✅ Version Controlled Development

Upcoming Features

- 🚧 2D LiDAR
- 🚧 RGB Camera
- 🚧 IMU
- 🚧 GPS
- 🚧 SLAM
- 🚧 Navigation2
- 🚧 Autonomous Exploration
- 🚧 Meta Reinforcement Learning

---

# 🏗️ Project Architecture

```
ARES
│
├── ORION-Rover
│   ├── robot_ws
│   ├── orion_description
│   ├── orion_bringup
│   └── orion_control
│
└── Future
    ├── SLAM
    ├── Navigation2
    ├── Perception
    └── Meta Reinforcement Learning
```

---

# 🛠️ Technology Stack

| Category | Technologies |
|-----------|--------------|
| Robot Middleware | ROS 2 Jazzy |
| Simulator | Gazebo Harmonic |
| Visualization | RViz2 |
| Language | Python, XML, YAML |
| Robot Description | URDF, Xacro |
| Control | ros2_control |
| Controller | Diff Drive Controller |
| Build Tool | colcon |
| Version Control | Git & GitHub |

---

# 📂 Repository Structure

```
robot_ws
│
├── src
│   ├── orion_description
│   ├── orion_bringup
│   └── orion_control
│
├── build
├── install
└── log
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Suhash-25/ORION-Rover.git

cd ORION-Rover
```

---

## Build Workspace

```bash
cd robot_ws

colcon build

source install/setup.bash
```

---

# 🚀 Launch ORION

```bash
ros2 launch orion_bringup gazebo.launch.py
```

---

# 🎮 Teleoperation

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Keyboard Controls

```
u    i    o
j    k    l
m    ,    .
```

| Key | Action |
|------|--------|
| i | Forward |
| , | Backward |
| j | Rotate Left |
| l | Rotate Right |
| k | Stop |

---

# 📊 Development Progress

| Sprint | Status |
|---------|--------|
| Sprint 1 - Robot Description | ✅ Completed |
| Sprint 2 - Four Wheel Rover | ✅ Completed |
| Sprint 3 - Gazebo Simulation | ✅ Completed |
| Sprint 4 - Differential Drive | ✅ Completed |
| Sprint 5 - Sensors | 🚧 In Progress |
| Sprint 6 - SLAM | ⏳ Planned |
| Sprint 7 - Navigation2 | ⏳ Planned |
| Sprint 8 - Autonomous Exploration | ⏳ Planned |
| Sprint 9 - Meta Reinforcement Learning | ⏳ Planned |

---

# 🎯 Project Roadmap

```
Robot Description
        │
        ▼
Gazebo Simulation
        │
        ▼
Differential Drive
        │
        ▼
Sensor Integration
        │
        ▼
SLAM
        │
        ▼
Localization
        │
        ▼
Navigation2
        │
        ▼
Terrain Understanding
        │
        ▼
Meta Reinforcement Learning
```

---

# 📈 Current Status

✅ Robot Model Completed

✅ Gazebo Simulation Completed

✅ Differential Drive Completed

🚧 Sensor Integration Under Development

---

# 👨‍💻 Developer

**Suhas G**

Computer Science Engineering (AI & ML)

Maharaja Institute of Technology Mysore

GitHub

https://github.com/Suhash-25

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

---

# 📜 License

This project is licensed under the MIT License.


## ARES Engineering Handoff and Checkpoint Runbook
**Handoff revision:** 4 November 2026  
**Repository:** https://github.com/Suhash-25/ORION-Rover  
**Purpose:** Evidence-based handoff for a teammate or coding agent continuing ORION engineering and ARES Meta-RL research.

> This document records the known recovered state and remaining work. It does not claim all tests or hardware work are complete. Mark a checkpoint PASS only when its test and evidence are recorded.

## 1. Project objective and research contribution

ARES is a Self-Adaptive Meta-Reinforcement Learning framework for autonomous locomotion in planetary-style environments. The research goal is to train a locomotion policy across parameterized terrain tasks and adapt it to held-out terrain using a small number of gradient updates.

The project synopsis identifies MAML-PPO as the core algorithm, FOMAML and Reptile as lower-compute alternatives, PEARL as a comparison, standard PPO and multi-task PPO baselines, a parameterized Gazebo terrain generator, a ROS 2/Gymnasium proprioceptive interface, held-out evaluation, domain randomization, sim-to-real analysis, and reproducible source/checkpoints/logs.

### Research targets from the synopsis

| Item | Target |
|---|---|
| Terrain parameters | Friction, slope, roughness, gravity, surface compliance |
| Training tasks | 35 configurations |
| Held-out tasks | 8 configurations |
| Inner updates | 3 by default |
| Rapid adaptation | At least 80% of scratch-policy performance within 5 or fewer updates |
| Comparisons | MAML-PPO, PPO from scratch, multi-task PPO, PEARL |

These are targets, not achieved results.

### Scope distinction

The project synopsis explicitly excludes visual navigation and SLAM from the core research scope. The current ORION workspace also contains LiDAR, SLAM Toolbox and Nav2 engineering work. Track these separately:

1. Robotics engineering: robot model, Gazebo, control, sensors, SLAM and navigation bringup.
2. Core research: terrain tasks, proprioceptive observations/actions/rewards, PPO, MAML-PPO, baselines, adaptation and sim-to-real evaluation.

Do not allow SLAM debugging to replace the Meta-RL contribution. Keep a simple validated locomotion task available while algorithm development proceeds.

## 2. Repository and Git state

Public repository: https://github.com/Suhash-25/ORION-Rover.git  
Default branch: `main`  
Recovered development branch: `sprint-8-debug`

GitHub branch heads observed on 4 November 2026:

- `main`: `6dc85246fa917b9514836a6779f7e7a7133718fb`
- `sprint-8-debug`: `f2362c5c0f2fdbe7bdfabb918830a4ae3ceade72`

The recovered machine previously reported local HEAD `6dc8524` on `sprint-8-debug`. The remote branch has since advanced to a different commit. Do not assume remote and recovered local trees match.

### Clone and inspect

```bash
git clone https://github.com/Suhash-25/ORION-Rover.git
cd ORION-Rover
git fetch --all --tags
git branch -avv
git status --short
git log --oneline --decorate -10
```

### Critical working-tree warning

The recovered local tree had modified tracked files, deleted older map files, untracked map/config files, diagnostic outputs and recovery build directories. A normal clone does not include uncommitted or untracked content.

Last recorded modified paths:

- `robot_ws/src/orion_bringup/config/nav2_params.yaml`
- `robot_ws/src/orion_bringup/launch/localization.launch.py`
- `robot_ws/src/orion_bringup/launch/slam.launch.py`
- `robot_ws/src/orion_bringup/worlds/empty.sdf`
- `robot_ws/src/orion_bringup/worlds/mapping_world.sdf`

Last recorded deleted paths:

- `robot_ws/src/orion_bringup/maps/my_first_map.pgm`
- `robot_ws/src/orion_bringup/maps/my_first_map.yaml`

Last recorded untracked files included `nav2_params.yaml.pre_duplicate_fix`, `orion_map.pgm`, `orion_map.yaml`, recovery build/install/log outputs and diagnostic files.

Recovery archives recorded on the original machine:

```text
~/ARES-recovery-backups/ARES-pre-build-20261012-231141.tar.gz
~/ARES-recovery-backups/ARES-build-artifacts-prebuild-20261021-185702.tar.gz
~/ARES-recovery-backups/ARES-world-lidar-prepatch-20261021-193523.tar.gz
```

These archives are not automatically included in GitHub. Transfer them separately if required. Never run `git reset --hard`, `git clean -fdx`, discard changes, or overwrite maps as a shortcut.

## 3. Recovered development environment

| Component | Recorded value |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| ROS 2 | Jazzy |
| Python | 3.12.3 |
| Gazebo | Gazebo Sim 8.15.0 (Harmonic) |
| Git | 2.43.0 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB |
| NVIDIA driver | 535.288.01 |

The project synopsis originally listed Ubuntu 22.04, ROS 2 Humble and Gazebo Classic as tentative requirements. The recovered ORION workspace was tested at least at static-validation/build level on Jazzy/Harmonic. Do not mix Humble/Classic and Jazzy/Harmonic instructions without a deliberate migration plan.

Install ROS 2 Jazzy from the official instructions first: https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

Then install the package set recorded on the recovered system:

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop ros-dev-tools python3-colcon-common-extensions \
  python3-rosdep ros-jazzy-xacro ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-ros-gz \
  ros-jazzy-gz-ros2-control ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
  ros-jazzy-slam-toolbox build-essential cmake git
```

Initialize rosdep only if not already initialized:

```bash
sudo rosdep init
rosdep update
cd ~/ORION-Rover/robot_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

The recovered Gazebo executable path was:

```bash
export PATH="/opt/ros/jazzy/opt/gz_tools_vendor/bin:$PATH"
gz sim --versions
```

Do not install a separate Gazebo distribution merely because `gz` is missing from PATH.

## 4. Workspace and package structure

```text
ORION-Rover/
└── robot_ws/
    └── src/
        ├── orion_bringup/      # launch, worlds, maps, config, RViz
        ├── orion_control/      # ros2_control config and teleop
        └── orion_description/  # URDF/Xacro, links, wheels, sensors
```

## 5. Safe isolated build

The old CMake cache referred to `/home/suhash/ARES/robot_ws`, while the recovered workspace was under `/home/suhash/Desktop/ARES/robot_ws`. This caused a stale path mismatch.

For a fresh clone, build into isolated directories:

```bash
cd ~/ORION-Rover/robot_ws
source /opt/ros/jazzy/setup.bash
colcon --log-base log_handoff build \
  --symlink-install \
  --build-base build_handoff \
  --install-base install_handoff \
  --event-handlers console_direct+
source install_handoff/setup.bash
ros2 pkg prefix orion_bringup
ros2 pkg prefix orion_control
ros2 pkg prefix orion_description
```

Expected: all three package prefixes resolve below `robot_ws/install_handoff/`. The original recovered machine already used `build_recovery`, `install_recovery`, and `log_recovery`; do not overwrite those directories while inspecting that machine.

## 6. Recorded robot/sensor baseline

| Parameter | Recorded value |
|---|---:|
| Wheel radius | 0.12 m |
| Controller wheel separation | 0.52 m |
| Wheel lateral positions | +0.26 m / -0.26 m |
| Front/rear wheel X positions | +0.25 m / -0.25 m |
| LiDAR frame | `lidar_link` |
| `base_link → lidar_link` Z | 0.55 m |
| LiDAR rate | 10 Hz |
| Scan samples | 720 |
| Range | 0.12–12.0 m |
| Field of view | approximately 360° |

The recorded wheel separation matches wheel joint lateral positions. Do not change geometry without repeatable new measurements.

## 7. Runtime launch and diagnostics

In each terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ORION-Rover/robot_ws/install_handoff/setup.bash
export PATH="/opt/ros/jazzy/opt/gz_tools_vendor/bin:$PATH"
```

The previous mapping session was started using:

```bash
ros2 launch orion_bringup slam.launch.py
```

Verify the actual checked-out launch file before relying on this behavior. The recovered session launched the mapping world, spawned the rover, started controllers, bridged `/scan` and started SLAM Toolbox. Do not launch a second Gazebo instance while one is already running. The recorded `localization.launch.py` also launches Gazebo, so inspect its launch graph before running it alongside SLAM.

Runtime checks:

```bash
ros2 node list
ros2 control list_controllers
ros2 topic list
ros2 topic info /scan --verbose
ros2 topic info /tf --verbose
ros2 topic info /tf_static --verbose
ros2 topic hz /scan
ros2 topic echo /diff_drive_controller/odom --once
ros2 topic echo /map --once --field info
```

SLAM parameter checks:

```bash
ros2 param get /slam_toolbox map_frame
ros2 param get /slam_toolbox odom_frame
ros2 param get /slam_toolbox base_frame
ros2 param get /slam_toolbox scan_topic
ros2 param get /slam_toolbox minimum_time_interval
ros2 run tf2_ros tf2_echo map odom
```

Previously observed values were `map`, `odom`, `base_link`, `/scan`, and `0.5`. Allow TF discovery to settle before treating an initial unavailable-frame message as a persistent fault.

## 8. Twelve engineering checkpoints

Statuses reflect the last recorded diagnostics, not a fresh clone's test results.

| # | Checkpoint | Status | Evidence required |
|---:|---|---|---|
| 1 | Preserve source, Git state and backups | PASS on recovered machine | Verify transferred snapshot; preserve diffs/maps/archives. |
| 2 | ROS 2 environment and workspace build | PASS on recovered machine | Re-run isolated build on receiving machine. |
| 3 | Validate SDF, YAML, launch syntax, package XML and Xacro | PASS for recorded snapshot | Re-run after any edits. |
| 4 | Runtime nodes and controllers | PASS in earlier session | Verify expected nodes and active controllers after clean launch. |
| 5 | LiDAR and sensor TF | PASS in earlier session | `/scan` at about 10 Hz, 720 samples, 0.12–12 m, expected frame and static transform. |
| 6 | Straight-line odometry | PASS for short test | 0.1 m/s for 6 s yielded ~0.63 m for 0.60 m commanded, near-zero yaw. |
| 7 | TF and scan-matching stability | PARTIAL | `map→odom` looked stable during straight motion, but doubled wall outlines remained. |
| 8 | Controlled in-place rotation | NEXT / not yet evidenced | Compare commanded rotation and odometry yaw; sample `map→odom`; inspect RViz. |
| 9 | Isolate doubled-map root cause | PENDING | Diagnose TF ownership, rotation odometry, extrinsics, timing, bridge/QoS, matching and world geometry. |
| 10 | Evidence-supported correction | PENDING | Back up affected files, make minimal fix, isolated rebuild, focused runtime test. |
| 11 | Mapping regression | PENDING | Straight, rotation, return-path and stop tests; inspect geometry. |
| 12 | Save/document validated map | PENDING | Save to new filenames only after quality acceptance; never overwrite old maps. |

### Checkpoint 8: rotation test procedure

1. Keep Gazebo, RViz and SLAM running.
2. Record odometry while stationary.
3. Command a slow in-place turn of approximately 30–45 degrees.
4. Explicitly stop and wait several seconds.
5. Record odometry again and sample `map→odom` over several readings.
6. Compare commanded angular displacement to odometry yaw, accounting for wrap at ±π.
7. Inspect whether previously mapped walls remain aligned or spread.
8. Save command values, timestamps, odometry before/after, TF samples and RViz screenshot.

Do not perform a rapid or full spin.

### Checkpoint 9: root-cause investigation order

- Verify unique TF ownership for `odom→base_link`, `map→odom` and static sensor transforms.
- Compare commanded rotation with encoder-derived yaw.
- Inspect full `base_link→lidar_link` translation and orientation from expanded URDF and TF, not only Z.
- Compare `/scan` timestamps to `/clock`; verify `use_sim_time` consistently.
- Check scan rate, message drops, frame ID, angular ordering, ranges and invalid readings.
- Inspect `ros_gz_bridge` direction and QoS. Do not change without endpoint evidence.
- Inspect SLAM live parameters and logs. A previous Ceres warning that a request for 50 threads was bounded to 8 was nonfatal and does not independently explain distortion.
- Confirm world geometry and object placement.

For every suspected cause, record both the observation supporting it and a test that could falsify it.

### Known map issue

The map remained visibly distorted with doubled/misaligned wall edges after the straight movement test. Last recorded metadata:

```text
resolution: 0.05 m/cell
width: 509
height: 572
origin: x=-15.5207 m, y=-20.2146 m
```

This map is not approved as final. The root cause is unresolved. Do not save or label it as validated.

## 9. Separate Meta-RL research deliverables

The 12 engineering checkpoints do not complete the major project's research. Track these separately:

1. Pin reproducible Python/PyTorch/Gymnasium environment and record versions/seeds.
2. Build terrain generator for friction, slope, roughness, gravity and compliance.
3. Create 35 training tasks and 8 held-out tasks; store configs and split manifest; test no leakage.
4. Build ROS 2 ↔ Gymnasium locomotion interface with explicit proprioceptive observations, bounded actions, reward, reset and termination.
5. Implement and validate PPO baseline.
6. Implement MAML-PPO with separate support/query rollouts, configurable inner updates and outer meta-update.
7. Implement FOMAML and Reptile.
8. Implement multi-task PPO and PEARL comparisons with controlled budgets.
9. Run domain-randomization breadth experiments and record sampled parameters.
10. Evaluate held-out success, speed, completion time, slip, safety stops, adaptation curves, update count and compute cost across multiple seeds.
11. Perform safe physical testbed validation or systematic sim-to-real gap analysis.
12. Publish code, environment lock files, configs, checkpoints, logs, plots, report and reproduction instructions.

Report actual measured results and conditions; do not fabricate the 80% / 5-step target.

## 10. Hardware approval gates

The earlier component spreadsheet was a planning estimate, not a certified bill of materials. Before purchasing, verify motor voltage/stall current, encoder counts/voltage, wheel dimensions, driver current limits, MCU firmware interface, battery/BMS current, charger profile, regulator, fuses, wiring and emergency stop.

Earlier candidates were Raspberry Pi 5 8GB, RPLIDAR A1M8, four 12V encoder gear motors, motor drivers, BNO085 IMU, 12.8V LiFePO4 battery with BMS, regulated 5V compute supply, fused distribution, emergency stop, chassis, wheels, wires and sensor brackets. These remain candidates only.

Do not order Cytron MDD20A until the selected firmware's PWM/direction interface, four-motor wiring, voltage and current requirements are checked against the driver's datasheet. Initially run Meta-training on suitable laptop/workstation or remote compute; do not assume it will train efficiently on a Pi.

Before hardware policy tests: test motors with wheels lifted; verify motor directions and encoder signs; calibrate wheel geometry; calibrate IMU; verify LiDAR; enforce conservative speed/acceleration/watchdog; test physical E-stop; use a controlled low-speed area.

## 11. Static validation

Git safety:

```bash
git status --short
git diff --check
git rev-parse HEAD
git diff --stat
```

Launch syntax:

```bash
python3 - <<'PY'
import ast
from pathlib import Path
files = sorted(Path('robot_ws/src').rglob('*.launch.py'))
for file in files:
    ast.parse(file.read_text(), filename=str(file))
print(f'Parsed {len(files)} launch files')
PY
```

Package XML and Xacro XML:

```bash
python3 - <<'PY'
import xml.etree.ElementTree as ET
from pathlib import Path
for f in Path('robot_ws/src').rglob('package.xml'):
    ET.parse(f)
    print('XML OK:', f)
for f in Path('robot_ws/src/orion_description').rglob('*.xacro'):
    ET.parse(f)
    print('Xacro XML OK:', f)
PY
```

World validation:

```bash
GZ=/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz
"$GZ" sdf -k robot_ws/src/orion_bringup/worlds/empty.sdf
"$GZ" sdf -k robot_ws/src/orion_bringup/worlds/mapping_world.sdf
git diff --check
```

Use a YAML validator that explicitly rejects duplicate keys. Ordinary `yaml.safe_load()` alone may silently accept duplicates.

## 12. Rules for the receiving developer / coding agent

1. Inspect Git status, branches, source, package manifests, launch files, worlds, URDF/Xacro and evidence before editing.
2. Never reset, clean, delete backups or overwrite existing maps.
3. Do not start duplicate Gazebo/robot sessions.
4. Change one suspected cause at a time.
5. Back up affected files before edits.
6. Build in fresh isolated output directories.
7. Distinguish static validation, build success, simulation success and physical-hardware success.
8. Record each checkpoint as PASS, FAIL, BLOCKED or NOT RUN, with commands, outputs, files and evidence.
9. Human-only hardware actions remain BLOCKED until performed and evidenced.
10. Keep the Meta-RL research objective central.

## 13. Coding-agent execution prompt

> You are taking over ARES / ORION-Rover. Read this README fully. Repository: `https://github.com/Suhash-25/ORION-Rover.git`. Inspect branches, commits, Git status, source tree, launch files, worlds, model, configuration, manifests and tests. The recovered local source contained uncommitted changes that may not exist on GitHub; a fresh clone may differ.

> Preserve source and maps. Never run `git reset --hard`, `git clean -fdx`, or delete existing build/install/log directories as a shortcut. Verify transferred recovery snapshot first.

> Continue engineering checkpoints 8–12: rotation test, doubled-map diagnosis, minimal evidence-based correction with rollback, mapping regression and new map save only after validation. Do not claim hardware readiness until motors, encoders, driver, MCU, battery, wiring, firmware, calibration and E-stop checks pass.

> Separately implement terrain generator, 35 train/8 held-out tasks, proprioceptive Gymnasium/ROS 2 interface, PPO, MAML-PPO, FOMAML/Reptile, PEARL and multi-task comparisons, domain randomization, few-shot evaluation and sim-to-real analysis.

> For each checkpoint record status, evidence, commands, outputs, modified files, risks and rollback. Human-only tests must be given as exact safe procedures and remain BLOCKED until evidence is supplied. Never fabricate results or declare success merely because code exists.

## 14. References

- ROS 2 Jazzy: https://docs.ros.org/en/jazzy/
- Gazebo Harmonic: https://gazebosim.org/docs/harmonic/
- Nav2: https://docs.nav2.org/
- SLAM Toolbox: https://github.com/SteveMacenski/slam_toolbox
- ros2_control: https://control.ros.org/
- Linorobot2 Jazzy, optional reference: https://github.com/linorobot/linorobot2/tree/jazzy
- Linorobot2 hardware, optional reference: https://github.com/linorobot/linorobot2_hardware

**Open handoff gaps:** recovered-source snapshot transfer, rotation-test evidence, doubled-map root cause, final hardware compatibility sign-off, and Meta-RL implementation/evaluation evidence.