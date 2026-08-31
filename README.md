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
