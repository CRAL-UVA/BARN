# BARN

This repository provides the **Jackal FreeTube Planner**, a lightweight ROS 2-based motion planner designed for dynamic and unstructured environments.

## Goal-Oriented Motion Tube Planner (ROS 2)

A local motion planner for mobile robots (tested on Clearpath Jackal) that generates, evaluates, and selects short-term motion “tubes” — dynamically feasible trajectories based on laser scan data — to navigate toward a goal while avoiding obstacles.

---

## 1. Overview

The **Goal-Oriented Motion Tube Planner** computes multiple short-term forward trajectories (“tubes”) at each cycle, evaluates their feasibility based on LiDAR range data, and selects the safest, most goal-progressing tube to execute.

It publishes:

- Smoothed velocity commands: `/cmd_vel`
- Visualization markers for RViz: `/motion_tubes`

It accepts goals from:

- `/move_base_simple/goal` (RViz 2D Nav Goal)
- `/freetube/goal` (custom topic)
- `/navigate_to_pose` (if `nav2_msgs` is available)

---

## 2. Core Algorithm

At each planning cycle, the node:

1. **Generates** multiple candidate motion tubes from combinations of linear velocity `v`, angular velocity `w`, and different time horizons.
2. **Checks collisions** using graded clearance evaluation against LiDAR ranges.
3. **Computes a composite cost** combining:
   - Goal progress
   - Heading alignment
   - Obstacle penalties
   - Curvature smoothness
   - Trajectory length and speed
4. **Selects** the tube with the lowest cost and publishes smoothed velocity commands.
5. **Visualizes** all candidate tubes in RViz:
   - Feasible → green / yellow
   - Selected → cyan
   - Infeasible → red

---

## 3. Installation

```bash
cd ~/ros2_ws/src
git clone git@github.com:CRAL-UVA/BARN.git
cd ~/ros2_ws
colcon build 
source install/setup.bash
```

## 4.Running the Planner
Launch manually:
```bash
ros2 run jackal_freetube_planner fixed_granular_ros2 \
  --ros-args \
  -p scan_topic:=/j100_0896/scan \
  -p odom_topic:=/j100_0896/platform/odom \
  -p cmd_vel_topic:=/j100_0896/platform/cmd_vel_unstamped \
  -p marker_topic:=/j100_0896/motion_tubes \
  -p scan_qos_best_effort:=true \
  -p base_to_laser_yaw:=1.57 \
  -p critical_pad:=0.01 \
  -p warning_pad:=0.05 \
  -p max_v:=0.1 \
  -p velocity_layers:="[0.03, 0.06, 0.1]" \
  -p max_w:=0.6
  ```

In RViz, add topic `/motion_tubes` of type MarkerArray to visualize tubes.

Send a goal via `/move_base_simple/goal` or `/freetube/goal`.

## 5. Diagnostics Output

The planner periodically prints diagnostics (every 3 seconds), for example:

```text
[INFO] [planner]: tubes=150 feas=97 fwd=2.14m
```

Where:

tubes — total number of generated candidate tubes

feas — number of feasible (collision-free) tubes

fwd — current forward clearance in meters

## 6. Simulation

During simulation, the robot (Clearpath Jackal) navigates through cluttered corridors and unstructured layouts using dynamically generated motion tubes.
<img width="1440" height="436" alt="image" src="https://github.com/user-attachments/assets/de816ecc-a16d-4f62-8c4c-e841898a4b6d" />

<img width="1453" height="439" alt="image" src="https://github.com/user-attachments/assets/de18bfda-68cc-4427-9e6b-d9d99d07c862" />

Below shows the **Motion Tube Visualization** in RViz.  
Each colored arc represents a simulated feasible path segment (“tube”) evaluated by the planner.  
- 🟩 Green: feasible tubes  
- 🟥 Red: infeasible or collision-risk tubes  
- 🟦 Cyan: selected trajectory for execution  

<img width="803" height="416" alt="image" src="https://github.com/user-attachments/assets/1f847587-8755-44c1-97e1-aa247799e65d" />



