# Motion Tube Planner (ROS2)

## 1. Introduction

The **Motion Tube Planner** is a local navigation planner for the Clearpath Jackal that selects a short-horizon motion command by generating many candidate trajectories (“motion tubes”), checking their collision feasibility against the latest laser scan, scoring them with a composite cost function, and executing the best one.

Instead of optimizing over a continuous control space online, the planner evaluates a structured set of preconfigured motion tubes with different angular velocities `w` and time horizons `T`. This makes the planner easy to tune, fast enough for real-time execution, and interpretable during debugging.

In the current ROS2 implementation, the planner:
- subscribes to **LaserScan** and **Odometry**,
- exposes a **NavigateToPose** action server,
- continuously generates and evaluates candidate motion tubes,
- publishes the selected command as **Twist**,
- visualizes all tubes in **RViz** using `MarkerArray`.

---

## 2. Planner Workflow and Algorithm Logic

### 2.1 High-level workflow

At every planning cycle, the node performs the following steps:

1. **Read the latest state**  
   The planner receives:
   - current laser scan,
   - current odometry / pose,
   - active goal from `NavigateToPose`.

2. **Validate sensor freshness**  
   If scan or odometry data are too old, the planner stops the robot by publishing zero velocity.

3. **Estimate forward clearance**  
   It computes a forward-sector clearance value from the scan. This is later used to slow down linear speed when obstacles are close in front.

4. **Generate candidate motion tubes**  
   Based on grouped `(w, T)` configurations, the planner creates multiple candidate trajectories.

5. **Perform collision checking**  
   Each tube is checked against the laser scan using a half-width collision model. Infeasible tubes are discarded.

6. **Measure goal progress**  
   For feasible tubes, the planner estimates how much each tube would reduce the distance to the goal.

7. **Compute composite cost**  
   Each feasible tube is scored using a weighted cost function that balances:
   - goal progress,
   - heading alignment,
   - curvature penalty,
   - tube length reward,
   - speed reward,
   - obstacle clearance,
   - left/right side clearance,
   - center-balance,
   - revisit penalty,
   - opposite-turn suppression during turn commitment.
     
8. **Select the best tube**  
   The planner applies group priority, angular-velocity holding, and turn-commit logic before choosing the final tube.

9. **Publish command and visualization**  
   The selected tube is converted into `cmd_vel`, and all tubes are published to RViz as markers.

---

### 2.2 Motion tube generation

A motion tube is a short trajectory template parameterized by:
- `v`: linear speed,
- `w`: angular speed,
- `T`: execution horizon,
- `samples`: sampled points along the trajectory,
- `beam_indices`: the associated laser beams used later for collision checking,
- `arc_len`: trajectory arc length,
- `group_name`: the tube group it belongs to.

#### Tube groups
The implementation organizes candidate tubes into four groups:

- **G1_low_w_longT**: low angular velocity, long horizon
- **G2_mid_w_turn**: medium turning tubes
- **G3_low_w_midT**: low angular velocity, medium horizon
- **G4_high_w_shortT**: aggressive short turning tubes

Each group is defined by:
- a `w` range (`w_min`, `w_max`),
- a list of horizons `T_list`.

Then for each sampled absolute angular velocity:
- if `w = 0`, one straight tube is created,
- otherwise both `+w` and `-w` tubes are created.

#### Tube shape
For each tube:
- if `|w|` is near zero, the tube is straight:
  - `x = v * t`, `y = 0`
- otherwise the tube is an arc:
  - `R = v / w`
  - `theta = w * t`
  - `x = R * sin(theta)`
  - `y = R * (1 - cos(theta))`

The trajectory is sampled into multiple points. The number of samples increases with arc length, so longer tubes are represented with more points.

#### Forward-clearance pruning
Before creating a tube, the planner filters out overly long, nearly straight tubes when forward clearance is small. This prevents the robot from proposing long forward motion into a nearby obstacle.

---

### 2.3 Collision checking

Collision checking is done in `collision_check_halfwidth_hard()`.

#### Core idea
For several sampled points along the tube, the planner:
1. computes the distance from the robot origin to the sample point,
2. maps that direction to the corresponding laser beam index,
3. expands to a beam fan using the robot half-width,
4. checks whether laser ranges minus travel distance minus robot half-width remain positive.

The effective safety radius used here is:
- `eff_r = robot.half_width`

The planner records:
- `min_clearance`
- `left_clearance`
- `right_clearance`
- `center_balance`

#### Feasibility rule
If any local clearance becomes negative, the tube is marked infeasible immediately.

This method is efficient because it uses pre-associated beam indices per sample and checks only a local fan around each trajectory point.

---

### 2.4 Goal progress calculation

For each feasible tube, the planner estimates how useful it is for reaching the goal.

It does this by:
1. computing the robot’s current distance to the goal,
2. selecting up to 5 sample points along the tube,
3. transforming those points from the robot frame into the world frame,
4. computing how much closer each point gets to the goal,
5. combining the results as:

- `0.6 * average improvement + 0.4 * best improvement`

Negative progress is clipped to zero.

This design encourages tubes that make consistent forward progress while still rewarding a tube that has a particularly good future point.

---

### 2.5 Composite cost function

The planner assigns each feasible tube a scalar cost. Lower cost is better.

The current implementation includes the following terms:

#### Reward terms
- **Goal progress reward**: prefer tubes that reduce goal distance
- **Tube length reward**: prefer longer effective motion
- **Speed reward**: prefer moving rather than stopping

#### Penalty terms
- **Heading error penalty**: penalize mismatch between tube end heading and goal bearing
- **Curvature penalty**: penalize larger turning magnitude
- **Front clearance penalty**: penalize tubes that get too close to obstacles
- **Near-collision extra penalty**: extra penalty if `min_clearance < 0.08`
- **Side clearance penalty**: penalize insufficient lateral room
- **Center balance penalty**: penalize asymmetric left/right clearance, encouraging centered motion
- **Revisit penalty**: penalize tubes whose endpoint is too close to recently visited positions
- **Opposite-turn penalty**: suppress switching to the opposite turning direction during a turn-commit window
This composite scoring makes the planner goal-directed but still strongly constrained by local safety and stability.

---

### 2.6 Tube selection logic

After all costs are computed, the planner does not simply choose the global minimum cost tube. It applies additional decision logic.

#### Group priority
The function `_best_by_group_priority()` checks groups in the order they are defined and returns the best feasible tube from the first non-empty group. This lets you bias planner behavior structurally through group ordering.

#### Angular velocity hold (`w_hold_time`)
Once a new angular velocity is selected, the planner can temporarily prefer keeping the same `w` value. This reduces oscillation between neighboring tubes.

#### Turn commitment (`turn_commit_time`)
If the planner has committed to turning left or right, tubes with the opposite sign of `w` are penalized during the commitment window. This helps avoid the common trap where the robot starts turning one way, then immediately switches back because the goal is on the opposite side.

#### Fixed-speed filtering
The final selection currently filters feasible tubes so only tubes with `v == fixed_speed` are considered.

---

### 2.7 Command publishing

Once the best tube is selected, the planner converts it into a velocity command:
- `linear.x = selected_tube.v`
- `angular.z = selected_tube.w`

Then additional adjustments are applied:

- **Deadband on angular speed**: small `w` values are forced to zero
- **Forward slowdown**: reduce `v` based on forward clearance
- **Extra slowdown near obstacles**: reduce `v` if `min_clearance < 0.12`
- **Extra slowdown for sharp turns**: reduce `v` when `|w| > 0.8`

If no tube is feasible, the planner publishes zero velocity.

---

### 2.8 RViz visualization

The planner publishes every tube as a line strip marker.

Color convention:
- **Cyan**: currently selected tube
- **Red / transparent**: infeasible tube
- **Yellow to green**: feasible tubes, colored according to relative cost

This makes it easy to debug:
- what the planner generated,
- which tubes were rejected,
- which tube won,
- how the cost landscape changes around obstacles.

---

## 3. Overview: Inputs and Outputs

### Inputs

The planner consumes the following inputs:

#### 1. Laser scan
Type: `sensor_msgs/msg/LaserScan`  
Purpose:
- obstacle detection,
- clearance estimation,
- collision checking.

Configured by parameter:
- `scan_topic`

Current run command uses:
- `/scan` or ‘/j100_0896/scan’

#### 2. Odometry
Type: `nav_msgs/msg/Odometry`  
Purpose:
- current robot position,
- current heading,
- goal distance / bearing computation,
- world-frame transformation of tube samples.

Configured by parameter:
- `odom_topic`

Current run command uses:
- `/j100_0896/platform/odom`

#### 3. Navigation goal
Type: `nav2_msgs/action/NavigateToPose`  
Purpose:
- target position for local navigation.

Current goal command sends a goal to:
- `/navigate_to_pose`

#### 4. Parameters
The planner behavior is heavily shaped by ROS2 parameters, including:
- topic names,
- geometry / timing,
- clearance thresholds,
- tube grouping,
- turn commitment,
- revisit suppression.

---

### Outputs

#### 1. Velocity command
Type: `geometry_msgs/msg/Twist`  
Purpose:
- drive the Jackal.

Configured by parameter:
- `cmd_topic`

Current run command uses:
- `/j100_0896/platform/cmd_vel_unstamped`

#### 2. Tube visualization
Type: `visualization_msgs/msg/MarkerArray`  
Purpose:
- visualize all motion tubes and the selected one in RViz.

Configured by parameter:
- `marker_topic`

Current run command uses:
- `/j100_0896/motion_tubes`

#### 3. Action feedback/result
Type: `NavigateToPose` feedback/result  
Purpose:
- report current pose during navigation,
- signal success or cancellation.

---


## 4. Installation

```bash
mkdir -p ~/freetube_ws/src
cd ~/freetube_ws/src
git clone git@github.com:CRAL-UVA/BARN.git
cd ~/freetube_ws
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source ~/freetube_ws/install/setup.bash
```

## 5. Running the Planner

### 5.1 Start the robot-side stack first

Before running the planner, make sure the Jackal platform and required ROS2 topics are available, especially:
- scan topic,
- odometry topic,
- command topic,
- the frame used for the goal.

You should confirm the topics with:

```bash
ros2 topic list
ros2 topic echo /scan
ros2 topic echo /j100_0896/platform/odom
```

---

### 5.2 Launch the Motion Tube Planner

Use the command you provided:

```bash
ros2 run jackal_freetube_planner fixed_granular_ros2 --ros-args \
  -p scan_topic:=/scan \
  -p odom_topic:=/j100_0896/platform/odom \
  -p cmd_topic:=/j100_0896/platform/cmd_vel_unstamped \
  -p marker_topic:=/j100_0896/motion_tubes \
  -p base_to_laser_yaw:=0.0 \
  -p w_hold_time:=3.0 \
  -p w_sample_step:=0.05
```

### 5.3 What this command means

- `scan_topic:=/scan`  
  Uses the robot laser scan for collision checking.

- `odom_topic:=/j100_0896/platform/odom`  
  Uses Jackal odometry for pose estimation.

- `cmd_topic:=/j100_0896/platform/cmd_vel_unstamped`  
  Publishes velocity commands directly to the robot platform command topic.

- `marker_topic:=/j100_0896/motion_tubes`  
  Publishes RViz markers for all candidate tubes.

- `base_to_laser_yaw:=0.0`  
  Assumes no yaw offset between base frame and laser frame.

- `w_hold_time:=3.0`  
  Keeps the selected angular velocity for a short time to reduce oscillation.

- `w_sample_step:=0.05`  
  Samples angular velocity more densely, generating finer tube resolution.

---

### 5.4 Send a goal

Use your action goal command:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
"{pose: {header: {frame_id: '96/odom'}, pose: {position: {x: 20.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}"
```

### 5.5 Goal frame note

The frame id in the action goal must match the odometry / navigation frame used by the robot. In current setup, the goal is expressed in:
- `96/odom`

If this frame does not match the planner’s pose frame, the robot may not move correctly.

---

### 5.6 Visualize in RViz

Add a `MarkerArray` display in RViz and subscribe to:

```bash
/j100_0896/motion_tubes
```

see:
- many candidate tubes around the robot,
- red infeasible ones,
- feasible yellow/green ones,
- the selected cyan tube.

This is one of the most useful tools for debugging parameter tuning.

---

## 6. Important Parameters

### Topics and interfaces
- `scan_topic`
- `odom_topic`
- `cmd_topic`
- `marker_topic`
- `action_name`

### Motion generation
- `fixed_speed`
- `max_v`
- `max_w`
- `w_sample_step`
- `group1_w_min`, `group1_w_max`, `group1_T`
- `group2_w_min`, `group2_w_max`, `group2_T`
- `group3_w_min`, `group3_w_max`, `group3_T`
- `group4_w_min`, `group4_w_max`, `group4_T`

### Goal and timing
- `goal_tolerance`
- `loop_dt`
- `scan_timeout`
- `odom_timeout`

### Speed regulation
- `use_fwd_slowdown`
- `fwd_slow_half_angle_deg`
- `fwd_slow_gain`
- `min_forward_scale`

### Clearance / safety
- `w_clearance`
- `clearance_safe_dist`
- `w_side_clearance`
- `side_clearance_safe_dist`
- `w_center_balance`

### Anti-oscillation / anti-trap behavior
- `w_hold_time`
- `turn_commit_time`
- `opposite_turn_penalty`
- `recent_pos_memory_sec`
- `revisit_radius`
- `revisit_penalty_weight`

---

## 🔧 Sensor Setup (2D LiDAR vs 3D LiDAR)
LiDAR Topic Configuration

Due to hardware differences in the lab setup, the height and mounting configuration of the 2D and 3D LiDAR sensors are different. Therefore, the topic used by the planner must be adjusted accordingly:

2D LiDAR (Hokuyo)
```bash
/scan
```

3D LiDAR (e.g., Ouster)
```bash
/j100_0896/scan
```
⚠️ Important:
Make sure to update the planner parameter:
```bash
-p scan_topic:=<your_scan_topic>
```

### 2D LiDAR (Hokuyo) Setup

The 2D LiDAR requires manual network configuration and TF setup before use.

Step 1: Configure Network and Start Driver
```bash
sudo ip addr add 192.168.0.100/24 dev br0
ip route get 192.168.0.10
			———192.168.0.10 dev br0 src 192.168.0.100
ros2 run urg_node urg_node_driver \
  --ros-args -p ip_address:=192.168.0.10
```

Step 2: Publish Static Transform (TF)
```bash
ros2 run tf2_ros static_transform_publisher \
  0.15 0.0 0.0 0.0 0.0 0.0 \
  96/base_link laser
```

### 3D LiDAR Setup
For the 3D LiDAR, the system already publishes scan data:
```bash
/j100_0896/scan
```
No additional network or TF setup is required (assuming the driver is already running).



## 7. Diagnostics Output

The planner periodically prints diagnostics (every 3 seconds), for example:

```text
[INFO] [1775421483.600019833] [goal_oriented_motion_tube_planner_ros2_grouped_no_prune]: === DIAG === tubes=0 feas=0 scan_age=0.015 locked_w=None w_hold_left=0.00 turn_sign=0 turn_hold_left=0.00 recent=0

[INFO] [1775422285.459940042] [goal_oriented_motion_tube_planner_ros2_grouped_no_prune]: === DIAG === tubes=135 feas=83 scan_age=0.045 locked_w=0.9 w_hold_left=0.96 turn_sign=1 turn_hold_left=1.81 recent=34
```

## 8. Rviz
<img width="470" height="342" alt="image" src="https://github.com/user-attachments/assets/83c9699f-6e35-4959-9dad-b88438f5f15e" />

MarkerArray topic : /j100_0896/motion_tubes 

## 9. Simulation

During simulation, the robot (Clearpath Jackal) navigates through cluttered corridors and unstructured layouts using dynamically generated motion tubes.
<img width="1440" height="436" alt="image" src="https://github.com/user-attachments/assets/de816ecc-a16d-4f62-8c4c-e841898a4b6d" />

<img width="1453" height="439" alt="image" src="https://github.com/user-attachments/assets/de18bfda-68cc-4427-9e6b-d9d99d07c862" />

Below shows the **Motion Tube Visualization** in RViz.  
Each colored arc represents a simulated feasible path segment (“tube”) evaluated by the planner.  
- 🟩 Green: feasible tubes  
- 🟥 Red: infeasible or collision-risk tubes  
- 🟦 Cyan: selected trajectory for execution  

<img width="803" height="416" alt="image" src="https://github.com/user-attachments/assets/1f847587-8755-44c1-97e1-aa247799e65d" />



