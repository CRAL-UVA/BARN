#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from builtin_interfaces.msg import Duration


class TemplateType(object):
    MOVE_STRAIGHT = 0
    STEER_LEFT = 1
    STEER_RIGHT = 2


class RobotGeometry(object):
    def __init__(self):
        self.length = 0.45
        self.width = 0.37
        self.half_length = self.length / 2.0
        self.half_width = self.width / 2.0
        self.radius = math.hypot(self.half_length, self.half_width)


class MotionTube(object):
    def __init__(self, template_type, v, w, T, samples, beam_indices, arc_len, group_name="UNGROUPED"):
        self.template_type = template_type
        self.v = v
        self.w = w
        self.T = T
        self.samples = samples
        self.beam_indices = beam_indices
        self.arc_len = arc_len
        self.group_name = group_name

        self.cost = float("inf")
        self.is_feasible = False
        self.goal_progress = 0.0
        self.min_clearance = float("inf")
        self.heading_after = 0.0
        self.obstacle_penalty = 0.0

        self.left_clearance = float("inf")
        self.right_clearance = float("inf")
        self.center_balance = 0.0


class GoalOrientedMotionTubePlannerROS2(Node):
    def __init__(self):
        super().__init__("goal_oriented_motion_tube_planner_ros2_grouped_no_prune")

        self.cb_group = ReentrantCallbackGroup()
        self.goal_lock = threading.Lock()
        self._throttle_state = {}

        self.latest_scan = None
        self.current_pose = None
        self.current_goal = None
        self.current_goal_handle = None

        self.latest_scan_topic = None
        self.latest_odom_topic = None
        self.scan_sub = None
        self.odom_sub = None

        self.sensor_config = {
            "min_angle": -np.pi,
            "max_angle": np.pi,
            "angle_increment": 0.01,
            "range_min": 0.1,
            "range_max": 30.0,
            "num_beams": 0,
        }

        self.last_scan_time = 0.0
        self.last_odom_time = 0.0
        self.planning_enabled = False

        self._declare_parameters()
        self._load_parameters()

        cmd_qos = QoSProfile(depth=10)
        marker_qos = QoSProfile(depth=10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, cmd_qos)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, marker_qos)

        self.robot = RobotGeometry()

        self.progress_history = []
        self.recent_positions = []
        self.stuck_counter = 0

        self.motion_tubes = []
        self.selected_tube = None
        self.min_cost = 0.0
        self.max_cost = 1.0

        self.goal_start_time = None
        self._ranges_clipped = None
        self._fwd_cache = float("inf")
        self.no_feasible = False

        self.w_lock_until = 0.0
        self.locked_w_value = None

        self.turn_commit_until = 0.0
        self.committed_turn_sign = 0

        self._tube_configs = self._build_grouped_tube_configs()
        self._setup_subscriptions_or_discovery()

        self.planning_timer = self.create_timer(self.loop_dt, self.planning_cycle, callback_group=self.cb_group)
        self.diag_timer = self.create_timer(3.0, self.print_diagnostics, callback_group=self.cb_group)

        self.action_server = ActionServer(
            self,
            NavigateToPose,
            self.action_name,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group,
        )

    def _declare_parameters(self):
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("marker_topic", "/motion_tubes")
        self.declare_parameter("action_name", "navigate_to_pose")

        self.declare_parameter("scan_topic", "")
        self.declare_parameter("odom_topic", "")
        self.declare_parameter("scan_topic_candidates", ["/front/scan", "/laser/scan", "/base_scan", "/scan"])
        self.declare_parameter("odom_topic_candidates", ["/odometry/filtered", "/odom", "/jackal/odom", "/robot/odom"])

        self.declare_parameter("fixed_speed", 0.3)
        self.declare_parameter("max_v", 1.2)
        self.declare_parameter("max_w", 1.2)
        self.declare_parameter("w_sample_step", 0.1)

        self.declare_parameter("base_to_laser_yaw", 0.0)
        self.declare_parameter("deadband_w", 0.05)
        self.declare_parameter("goal_tolerance", 0.45)

        self.declare_parameter("loop_dt", 0.15)
        self.declare_parameter("scan_timeout", 0.225)
        self.declare_parameter("odom_timeout", 0.50)

        self.declare_parameter("use_fwd_slowdown", True)
        self.declare_parameter("fwd_slow_half_angle_deg", 25.0)
        self.declare_parameter("fwd_slow_gain", 0.40)
        self.declare_parameter("min_forward_scale", 0.20)

        self.declare_parameter("w_clearance", 10.0)
        self.declare_parameter("clearance_safe_dist", 0.10)

        self.declare_parameter("w_center_balance", 6.0)
        self.declare_parameter("w_side_clearance", 8.0)
        self.declare_parameter("side_clearance_safe_dist", 0.14)

        self.declare_parameter("stuck_window_sec", 2.0)
        self.declare_parameter("stuck_min_progress", 0.10)
        self.declare_parameter("stuck_min_motion", 0.18)
        self.declare_parameter("stuck_confirm_count", 3)

        self.declare_parameter("w_hold_time", 1.0)
        self.declare_parameter("turn_commit_time", 4.0)
        self.declare_parameter("opposite_turn_penalty", 196.0)
        self.declare_parameter("recent_pos_memory_sec", 5.0)
        self.declare_parameter("revisit_radius", 0.35)
        self.declare_parameter("revisit_penalty_weight", 5.0)

        self.declare_parameter("group1_w_min", 0.0)
        self.declare_parameter("group1_w_max", 0.1) ########0.5
        self.declare_parameter("group1_T", [10.0, 15.0])

        self.declare_parameter("group2_w_min", 0.2)
        self.declare_parameter("group2_w_max", 0.9)
        self.declare_parameter("group2_T", [3.0, 4.0,5.0])

        self.declare_parameter("group3_w_min", 0.0)
        self.declare_parameter("group3_w_max", 0.1)
        self.declare_parameter("group3_T", [2.0, 3.0, 4.0])

        self.declare_parameter("group4_w_min", 1.0)
        self.declare_parameter("group4_w_max", 1.2)
        self.declare_parameter("group4_T", [1.0, 2.0])

    def _load_parameters(self):
        gp = self.get_parameter
        self.cmd_topic = gp("cmd_topic").value
        self.marker_topic = gp("marker_topic").value
        self.action_name = gp("action_name").value

        self.scan_topic = gp("scan_topic").value
        self.odom_topic = gp("odom_topic").value
        self.scan_topic_candidates = list(gp("scan_topic_candidates").value)
        self.odom_topic_candidates = list(gp("odom_topic_candidates").value)

        self.fixed_speed = float(gp("fixed_speed").value)
        self.max_v = float(gp("max_v").value)
        self.max_w = float(gp("max_w").value)
        self.w_sample_step = float(gp("w_sample_step").value)

        self.base_to_laser_yaw = float(gp("base_to_laser_yaw").value)
        self.deadband_w = float(gp("deadband_w").value)
        self.goal_tolerance = float(gp("goal_tolerance").value)

        self.loop_dt = float(gp("loop_dt").value)
        self.scan_timeout = float(gp("scan_timeout").value)
        self.odom_timeout = float(gp("odom_timeout").value)

        self.use_fwd_slowdown = bool(gp("use_fwd_slowdown").value)
        self.fwd_slow_half_angle_deg = float(gp("fwd_slow_half_angle_deg").value)
        self.fwd_slow_gain = float(gp("fwd_slow_gain").value)
        self.min_forward_scale = float(gp("min_forward_scale").value)

        self.w_clearance = float(gp("w_clearance").value)
        self.clearance_safe_dist = float(gp("clearance_safe_dist").value)

        self.w_center_balance = float(gp("w_center_balance").value)
        self.w_side_clearance = float(gp("w_side_clearance").value)
        self.side_clearance_safe_dist = float(gp("side_clearance_safe_dist").value)

        self.stuck_window_sec = float(gp("stuck_window_sec").value)
        self.stuck_min_progress = float(gp("stuck_min_progress").value)
        self.stuck_min_motion = float(gp("stuck_min_motion").value)
        self.stuck_confirm_count = int(gp("stuck_confirm_count").value)

        self.w_hold_time = float(gp("w_hold_time").value)
        self.turn_commit_time = float(gp("turn_commit_time").value)
        self.opposite_turn_penalty = float(gp("opposite_turn_penalty").value)
        self.recent_pos_memory_sec = float(gp("recent_pos_memory_sec").value)
        self.revisit_radius = float(gp("revisit_radius").value)
        self.revisit_penalty_weight = float(gp("revisit_penalty_weight").value)

        self.group_defs = [
            {"name": "G1_low_w_longT", "w_min": float(gp("group1_w_min").value), "w_max": float(gp("group1_w_max").value), "T_list": [float(x) for x in gp("group1_T").value]},
            {"name": "G2_mid_w_turn", "w_min": float(gp("group2_w_min").value), "w_max": float(gp("group2_w_max").value), "T_list": [float(x) for x in gp("group2_T").value]},
            {"name": "G3_low_w_midT", "w_min": float(gp("group3_w_min").value), "w_max": float(gp("group3_w_max").value), "T_list": [float(x) for x in gp("group3_T").value]},
            {"name": "G4_high_w_shortT", "w_min": float(gp("group4_w_min").value), "w_max": float(gp("group4_w_max").value), "T_list": [float(x) for x in gp("group4_T").value]},
        ]

        count = int(round(self.max_w / self.w_sample_step))
        self.abs_w_samples = [round(i * self.w_sample_step, 4) for i in range(count + 1)]

    def _setup_subscriptions_or_discovery(self):
        if self.scan_topic:
            self._create_scan_subscription(self.scan_topic)
        if self.odom_topic:
            self._create_odom_subscription(self.odom_topic)

        if self.scan_sub is None or self.odom_sub is None:
            self.discovery_timer = self.create_timer(1.0, self._topic_discovery_step, callback_group=self.cb_group)
            self._topic_discovery_step()

    def _topic_discovery_step(self):
        topic_map = {name: types for name, types in self.get_topic_names_and_types()}

        if self.scan_sub is None:
            for topic in self.scan_topic_candidates:
                if topic in topic_map and "sensor_msgs/msg/LaserScan" in topic_map[topic]:
                    self._create_scan_subscription(topic)
                    break

        if self.odom_sub is None:
            for topic in self.odom_topic_candidates:
                if topic in topic_map and "nav_msgs/msg/Odometry" in topic_map[topic]:
                    self._create_odom_subscription(topic)
                    break

        if self.scan_sub is not None and self.odom_sub is not None and hasattr(self, "discovery_timer"):
            self.discovery_timer.cancel()

    def _create_scan_subscription(self, topic_name):
        if self.scan_sub is not None:
            return
        self.scan_sub = self.create_subscription(LaserScan, topic_name, self.scan_callback, qos_profile_sensor_data, callback_group=self.cb_group)
        self.latest_scan_topic = topic_name

    def _create_odom_subscription(self, topic_name):
        if self.odom_sub is not None:
            return
        odom_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE, depth=10)
        self.odom_sub = self.create_subscription(Odometry, topic_name, self.odom_callback, odom_qos, callback_group=self.cb_group)
        self.latest_odom_topic = topic_name

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _log_throttle(self, level, key, period, msg):
        now = self._now_sec()
        last = self._throttle_state.get(key, None)
        if last is None or (now - last) >= period:
            self._throttle_state[key] = now
            if level == "warn":
                self.get_logger().warning(msg)
            elif level == "error":
                self.get_logger().error(msg)
            else:
                self.get_logger().info(msg)

    def _publish_zero(self):
        self.cmd_pub.publish(Twist())

    def _make_duration_msg(self, seconds):
        seconds = max(0.0, float(seconds))
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1e9)
        return Duration(sec=sec, nanosec=nanosec)

    def _make_action_result(self, success=True):
        result = NavigateToPose.Result()
        if hasattr(result, "error_code"):
            result.error_code = NavigateToPose.Result.NONE if success and hasattr(NavigateToPose.Result, "NONE") else (0 if success else 1)
        return result

    def _same_w(self, a, b, tol=1e-3):
        return abs(float(a) - float(b)) <= tol

    def _turn_sign(self, w):
        if w > 0.05:
            return 1
        if w < -0.05:
            return -1
        return 0

    def _build_grouped_tube_configs(self):
        configs = []

        def in_range(val, lo, hi, tol=1e-9):
            return (val >= lo - tol) and (val <= hi + tol)

        for group in self.group_defs:
            for abs_w in self.abs_w_samples:
                if not in_range(abs_w, group["w_min"], group["w_max"]):
                    continue
                for T in group["T_list"]:
                    if abs(abs_w) < 1e-9:
                        configs.append({"v": self.fixed_speed, "w": 0.0, "T": float(T), "group_name": group["name"]})
                    else:
                        configs.append({"v": self.fixed_speed, "w": float(abs_w), "T": float(T), "group_name": group["name"]})
                        configs.append({"v": self.fixed_speed, "w": float(-abs_w), "T": float(T), "group_name": group["name"]})
        return configs

    def scan_callback(self, msg):
        self.latest_scan = msg
        self.last_scan_time = self._now_sec()

        if self.sensor_config["num_beams"] == 0:
            inc = msg.angle_increment if abs(msg.angle_increment) > 1e-9 else 1e-3
            self.sensor_config.update({
                "min_angle": msg.angle_min,
                "max_angle": msg.angle_max,
                "angle_increment": inc,
                "range_min": msg.range_min,
                "range_max": msg.range_max,
                "num_beams": len(msg.ranges),
            })
            self.planning_enabled = True

        arr = np.asarray(msg.ranges, dtype=np.float32)
        if arr.size > 0:
            arr[~np.isfinite(arr)] = float(self.sensor_config["range_max"])
            arr = np.clip(arr, float(self.sensor_config["range_min"]), float(self.sensor_config["range_max"]))
            self._ranges_clipped = arr
        else:
            self._ranges_clipped = None

    def odom_callback(self, msg):
        self.current_pose = PoseStamped()
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.pose.pose
        self.last_odom_time = self._now_sec()

    def goal_callback(self, goal_request):
        with self.goal_lock:
            if self.current_goal_handle is not None and self.current_goal_handle.is_active:
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        with self.goal_lock:
            self.current_goal_handle = goal_handle
            self.current_goal = goal_handle.request.pose
            self.no_feasible = False
            self.goal_start_time = self._now_sec()
            self.progress_history = []
            self.recent_positions = []
            self.stuck_counter = 0
            self.locked_w_value = None
            self.w_lock_until = 0.0
            self.committed_turn_sign = 0
            self.turn_commit_until = 0.0

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self._publish_zero()
                goal_handle.canceled()
                with self.goal_lock:
                    self.current_goal = None
                    self.current_goal_handle = None
                return self._make_action_result(success=False)

            if self.current_pose is not None and self.is_goal_reached():
                self._publish_zero()
                goal_handle.succeed()
                with self.goal_lock:
                    self.current_goal = None
                    self.current_goal_handle = None
                return self._make_action_result(success=True)

            if self.goal_start_time is not None:
                fb = NavigateToPose.Feedback()
                if hasattr(fb, "current_pose") and self.current_pose is not None:
                    fb.current_pose = self.current_pose
                if hasattr(fb, "navigation_time"):
                    fb.navigation_time = self._make_duration_msg(self._now_sec() - self.goal_start_time)
                goal_handle.publish_feedback(fb)

            time.sleep(0.10)

        self._publish_zero()
        with self.goal_lock:
            self.current_goal = None
            self.current_goal_handle = None
        return self._make_action_result(success=False)

    def is_goal_reached(self):
        if not self.current_goal or not self.current_pose:
            return False
        dx = self.current_goal.pose.position.x - self.current_pose.pose.position.x
        dy = self.current_goal.pose.position.y - self.current_pose.pose.position.y
        return math.hypot(dx, dy) < self.goal_tolerance

    def planning_cycle(self):
        now = self._now_sec()
        if not self.planning_enabled or self.current_goal is None or self.current_pose is None or self.latest_scan is None:
            self._publish_zero()
            return

        scan_age = now - self.last_scan_time
        odom_age = now - self.last_odom_time
        if scan_age > self.scan_timeout or odom_age > self.odom_timeout:
            self._publish_zero()
            return

        self._fwd_cache = self._compute_forward_clearance_cached()
        self.update_progress_history(now)

        self.generate_motion_tubes()
        self.evaluate_tubes()

        feas = [t for t in self.motion_tubes if t.is_feasible]
        self.no_feasible = len(feas) == 0

        if self.no_feasible:
            self.selected_tube = None
            self._publish_zero()
            self._publish_tube_markers()
            return

        self.select_best_tube()
        self.publish_commands()
        self._publish_tube_markers()

    def _pose_xy_yaw(self):
        if not self.current_pose:
            return 0.0, 0.0, 0.0
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        q = self.current_pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return x, y, yaw

    def _wrap(self, ang):
        while ang > math.pi:
            ang -= 2.0 * math.pi
        while ang < -math.pi:
            ang += 2.0 * math.pi
        return ang

    def angle_to_beam_idx(self, angle_in_base):
        if not self.latest_scan or self.sensor_config["num_beams"] == 0:
            return 0
        angle_in_laser = self._wrap(angle_in_base + self.base_to_laser_yaw)
        ang_min = self.sensor_config["min_angle"]
        ang_max = self.sensor_config["max_angle"]
        inc = self.sensor_config["angle_increment"]
        n = self.sensor_config["num_beams"]

        if inc > 0:
            if angle_in_laser <= ang_min:
                return 0
            if angle_in_laser >= ang_max:
                return n - 1
        else:
            if angle_in_laser >= ang_min:
                return 0
            if angle_in_laser <= ang_max:
                return n - 1

        idx = int((angle_in_laser - ang_min) / inc)
        return max(0, min(n - 1, idx))

    def _compute_forward_clearance_cached(self):
        if self._ranges_clipped is None or self.sensor_config["num_beams"] == 0:
            return float("inf")

        ang_min = self.sensor_config["min_angle"]
        inc = self.sensor_config["angle_increment"]
        n = self.sensor_config["num_beams"]
        forward_laser = self._wrap(-self.base_to_laser_yaw)
        half = math.radians(self.fwd_slow_half_angle_deg)

        vals = []
        for i in range(n):
            a = ang_min + i * inc
            if abs(self._wrap(a - forward_laser)) <= half:
                vals.append(float(self._ranges_clipped[i]))
        if len(vals) == 0:
            return float("inf")
        return float(np.percentile(vals, 10))

    def update_progress_history(self, now):
        if self.current_pose is None or self.current_goal is None:
            return

        x, y, _ = self._pose_xy_yaw()
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        goal_dist = math.hypot(gx - x, gy - y)

        self.progress_history.append((now, x, y, goal_dist))
        self.progress_history = [p for p in self.progress_history if p[0] >= now - self.stuck_window_sec]

        self.recent_positions.append((now, x, y))
        self.recent_positions = [p for p in self.recent_positions if p[0] >= now - self.recent_pos_memory_sec]

    def generate_motion_tubes(self):
        self.motion_tubes = []
        fwd = self._fwd_cache

        for cfg in self._tube_configs:
            v = cfg["v"]
            w = cfg["w"]
            T = cfg["T"]
            group_name = cfg["group_name"]

            approx_len = abs(v) * T
            if fwd < 0.65 and abs(w) < 0.18 and approx_len > (fwd + 0.20):
                continue

            tube = self.create_motion_tube(v, w, T, group_name)
            if tube:
                self.motion_tubes.append(tube)

    def create_motion_tube(self, v, w, T, group_name):
        if abs(w) < 0.05:
            ttype = TemplateType.MOVE_STRAIGHT
        elif w > 0:
            ttype = TemplateType.STEER_LEFT
        else:
            ttype = TemplateType.STEER_RIGHT

        samples, beam_indices = [], []
        arc_len = abs(v) * T
        n_samples = int(np.clip(8 + arc_len * 12.0, 12, 48))

        for i in range(1, n_samples + 1):
            t = T * i / float(n_samples)
            if abs(w) < 1e-3:
                x, y = v * t, 0.0
                heading = 0.0
            else:
                R = v / w
                theta = w * t
                x = R * math.sin(theta)
                y = R * (1.0 - math.cos(theta))
                heading = theta

            samples.append(np.array([x, y], dtype=np.float32))
            ang_base = math.atan2(y, x)
            idx = self.angle_to_beam_idx(ang_base + 0.04 * heading)
            beam_indices.append(idx)

        return MotionTube(ttype, v, w, T, samples, beam_indices, arc_len, group_name)

    def collision_check_halfwidth_hard(self, tube):
        if self._ranges_clipped is None:
            tube.left_clearance = -1.0
            tube.right_clearance = -1.0
            tube.center_balance = 1.0
            return False, -1.0, 0.0

        inc = float(self.sensor_config["angle_increment"])
        n = int(self.sensor_config["num_beams"])
        eff_r = float(self.robot.half_width)

        min_clr = float("inf")
        left_min_clr = float("inf")
        right_min_clr = float("inf")

        idxs = np.linspace(0, len(tube.samples) - 1, num=min(10, len(tube.samples)), dtype=int)

        for k in idxs:
            s = tube.samples[k]
            dist = float(np.linalg.norm(s))
            if dist < 1e-3:
                continue

            half = math.atan2(eff_r, max(0.05, dist))
            center_idx = tube.beam_indices[k]
            beam_span = max(1, int(half / max(abs(inc), 1e-6)))

            i0 = max(0, center_idx - beam_span)
            i1 = min(n - 1, center_idx + beam_span)

            local_min = float("inf")
            local_left_min = float("inf")
            local_right_min = float("inf")

            for j in range(i0, i1 + 1):
                r = float(self._ranges_clipped[j])
                clr = r - dist - eff_r

                if clr < local_min:
                    local_min = clr
                if clr < min_clr:
                    min_clr = clr

                if j < center_idx:
                    if clr < local_right_min:
                        local_right_min = clr
                    if clr < right_min_clr:
                        right_min_clr = clr
                elif j > center_idx:
                    if clr < local_left_min:
                        local_left_min = clr
                    if clr < left_min_clr:
                        left_min_clr = clr
                else:
                    if clr < local_left_min:
                        local_left_min = clr
                    if clr < local_right_min:
                        local_right_min = clr
                    if clr < left_min_clr:
                        left_min_clr = clr
                    if clr < right_min_clr:
                        right_min_clr = clr

            if local_min < 0.0:
                tube.left_clearance = left_min_clr if np.isfinite(left_min_clr) else -1.0
                tube.right_clearance = right_min_clr if np.isfinite(right_min_clr) else -1.0
                tube.center_balance = abs(max(-0.2, tube.left_clearance) - max(-0.2, tube.right_clearance))
                return False, min_clr, 0.0

        if not np.isfinite(min_clr):
            min_clr = -1.0
        if not np.isfinite(left_min_clr):
            left_min_clr = min_clr
        if not np.isfinite(right_min_clr):
            right_min_clr = min_clr

        tube.left_clearance = left_min_clr
        tube.right_clearance = right_min_clr
        tube.center_balance = abs(left_min_clr - right_min_clr)

        return True, min_clr, 0.0

    def goal_progress_along_tube(self, tube):
        cx, cy, yaw = self._pose_xy_yaw()
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        now_dist = math.hypot(gx - cx, gy - cy)

        if len(tube.samples) == 0:
            return 0.0

        idxs = np.linspace(0, len(tube.samples) - 1, num=min(5, len(tube.samples)), dtype=int)
        improvements = []

        for k in idxs:
            s = tube.samples[k]
            sx, sy = float(s[0]), float(s[1])

            fx = cx + sx * math.cos(yaw) - sy * math.sin(yaw)
            fy = cy + sx * math.sin(yaw) + sy * math.cos(yaw)

            after_dist = math.hypot(gx - fx, gy - fy)
            improvements.append(now_dist - after_dist)

        if len(improvements) == 0:
            return 0.0

        avg_prog = float(np.mean(improvements))
        best_prog = float(np.max(improvements))
        prog = 0.6 * avg_prog + 0.4 * best_prog
        return max(0.0, prog)

    def revisit_penalty(self, tube):
        if self.current_pose is None or len(self.recent_positions) == 0 or len(tube.samples) == 0:
            return 0.0

        cx, cy, yaw = self._pose_xy_yaw()
        end_local = tube.samples[-1]
        sx, sy = float(end_local[0]), float(end_local[1])

        ex = cx + sx * math.cos(yaw) - sy * math.sin(yaw)
        ey = cy + sx * math.sin(yaw) + sy * math.cos(yaw)

        min_d = float("inf")
        for _, px, py in self.recent_positions:
            d = math.hypot(ex - px, ey - py)
            if d < min_d:
                min_d = d

        if min_d < self.revisit_radius:
            return self.revisit_penalty_weight * (1.0 - min_d / max(1e-6, self.revisit_radius))
        return 0.0

    def composite_cost(self, t, goal_bearing, goal_factor=1.0):
        w_progress = 14.0
        w_heading = 5.0
        w_curvature = 0.8
        w_length = 2.0
        w_speed = 0.8
        w_clearance = self.w_clearance

        c = 0.0
        c -= goal_factor * w_progress * t.goal_progress

        heading_err = abs(self._wrap(t.heading_after - goal_bearing))
        c += goal_factor * w_heading * heading_err

        c += w_curvature * abs(t.w)

        c -= w_length * t.arc_len
        c -= w_speed * t.v

        if t.min_clearance < self.clearance_safe_dist:
            clearance_ratio = max(0.0, 1.0 - t.min_clearance / max(1e-6, self.clearance_safe_dist))
            c += w_clearance * clearance_ratio

        if t.min_clearance < 0.08:
            c += 3.0

        side_min = min(t.left_clearance, t.right_clearance)
        if side_min < self.side_clearance_safe_dist:
            side_ratio = max(0.0, 1.0 - side_min / max(1e-6, self.side_clearance_safe_dist))
            c += self.w_side_clearance * side_ratio

        c += self.w_center_balance * min(0.25, t.center_balance)

        return c

    def evaluate_tubes(self):
        if not self.motion_tubes:
            return

        now = self._now_sec()
        yaw = self._pose_xy_yaw()[2]
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        goal_bearing = math.atan2(gy - cy, gx - cx)
        goal_factor = 1.0

        for t in self.motion_tubes:
            t.is_feasible, t.min_clearance, t.obstacle_penalty = self.collision_check_halfwidth_hard(t)
            if not t.is_feasible:
                continue

            t.heading_after = yaw + t.w * t.T
            t.goal_progress = self.goal_progress_along_tube(t)
            t.cost = self.composite_cost(t, goal_bearing, goal_factor)

            tube_sign = self._turn_sign(t.w)
            if now < self.turn_commit_until and self.committed_turn_sign != 0:
                if tube_sign != 0 and tube_sign != self.committed_turn_sign:
                    t.cost += self.opposite_turn_penalty

            t.cost += self.revisit_penalty(t)

        feas = [x for x in self.motion_tubes if x.is_feasible]
        if feas:
            self.min_cost = min(x.cost for x in feas)
            self.max_cost = max(x.cost for x in feas)
        else:
            self.min_cost = self.max_cost = 0.0

    def _best_by_group_priority(self, feas):
        for group in self.group_defs:
            group_name = group["name"]
            group_feas = [t for t in feas if t.group_name == group_name]
            if len(group_feas) == 0:
                continue
            return min(group_feas, key=lambda z: z.cost)
        return None

    def select_best_tube(self):
        now = self._now_sec()
        feas = [t for t in self.motion_tubes if t.is_feasible]
        self.no_feasible = len(feas) == 0

        if not feas:
            self.selected_tube = None
            return

        target_v = self.fixed_speed
        v_tol = 1e-3
        feas = [t for t in feas if abs(float(t.v) - target_v) <= v_tol]

        if not feas:
            self.selected_tube = None
            return

        if self.locked_w_value is not None and now < self.w_lock_until:
            same_w_feas = [t for t in feas if self._same_w(t.w, self.locked_w_value)]
            if same_w_feas:
                candidate = self._best_by_group_priority(same_w_feas)
                if candidate is None:
                    candidate = min(same_w_feas, key=lambda z: z.cost)
                self.selected_tube = candidate
                return
            else:
                self.locked_w_value = None
                self.w_lock_until = 0.0

        candidate = self._best_by_group_priority(feas)
        if candidate is None:
            candidate = min(feas, key=lambda z: z.cost)

        prev_w = None if self.selected_tube is None else self.selected_tube.w
        prev_sign = 0 if self.selected_tube is None else self._turn_sign(self.selected_tube.w)
        new_sign = self._turn_sign(candidate.w)

        if prev_w is None or not self._same_w(prev_w, candidate.w):
            self.locked_w_value = candidate.w
            self.w_lock_until = now + self.w_hold_time

        if new_sign != 0 and new_sign != prev_sign:
            self.committed_turn_sign = new_sign
            self.turn_commit_until = now + self.turn_commit_time

        self.selected_tube = candidate

    def publish_commands(self):
        cmd = Twist()

        if self.selected_tube:
            v = max(0.0, min(self.max_v, self.selected_tube.v))
            w = max(-self.max_w, min(self.max_w, self.selected_tube.w))

            if abs(w) < self.deadband_w:
                w = 0.0

            if self.use_fwd_slowdown and self.fwd_slow_gain > 1e-6:
                fwd = self._fwd_cache
                scale = max(self.min_forward_scale, min(1.0, fwd / (1.0 + 1e-6)))
                v *= scale ** (1.0 + self.fwd_slow_gain)

            if self.selected_tube.min_clearance < 0.12:
                v *= 0.75
            if abs(w) > 0.8:
                v *= 0.85

            cmd.linear.x = v
            cmd.angular.z = w
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    def _init_marker_pose(self, m):
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

    def _tube_color(self, tube):
        if tube is self.selected_tube:
            return ColorRGBA(r=0.00, g=0.95, b=1.00, a=1.0)
        if not tube.is_feasible:
            return ColorRGBA(r=0.95, g=0.15, b=0.15, a=0.35)
        try:
            c = max(0.0, min(1.0, (tube.cost - self.min_cost) / (self.max_cost - self.min_cost + 1e-6)))
        except Exception:
            c = 0.5
        return ColorRGBA(r=1.0 * c, g=1.0, b=0.0, a=0.9)

    def _publish_tube_markers(self):
        if not self.motion_tubes or not self.current_pose:
            return

        rx, ry, ryaw = self._pose_xy_yaw()
        cy, sy = math.cos(ryaw), math.sin(ryaw)
        ma = MarkerArray()
        frame = self.current_pose.header.frame_id if self.current_pose.header.frame_id else "odom"
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=frame)

        wipe = Marker()
        wipe.header = header
        self._init_marker_pose(wipe)
        wipe.ns = "motion_tubes"
        wipe.id = 0
        wipe.action = Marker.DELETEALL
        ma.markers.append(wipe)

        mid = 1
        Z = 0.03
        BASE_W = 0.05
        SEL_W = 0.09

        for t in self.motion_tubes:
            m = Marker()
            m.header = header
            self._init_marker_pose(m)
            m.ns = "motion_tubes"
            m.id = mid
            mid += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = BASE_W
            m.color = self._tube_color(t)

            pts = []
            for s in t.samples:
                sx, sy_local = float(s[0]), float(s[1])
                ox = rx + (sx * cy - sy_local * sy)
                oy = ry + (sx * sy + sy_local * cy)
                pts.append(Point(x=ox, y=oy, z=Z))
            m.points = pts
            ma.markers.append(m)

        if self.selected_tube is not None:
            t = self.selected_tube
            m = Marker()
            m.header = header
            self._init_marker_pose(m)
            m.ns = "motion_tubes"
            m.id = mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = SEL_W
            m.color = ColorRGBA(r=0.00, g=0.95, b=1.00, a=1.0)
            pts = []
            for s in t.samples:
                sx, sy_local = float(s[0]), float(s[1])
                ox = rx + (sx * cy - sy_local * sy)
                oy = ry + (sx * sy + sy_local * cy)
                pts.append(Point(x=ox, y=oy, z=Z))
            m.points = pts
            ma.markers.append(m)

        self.marker_pub.publish(ma)

    def print_diagnostics(self):
        feas = [t for t in self.motion_tubes if t.is_feasible]
        scan_age = self._now_sec() - self.last_scan_time if self.last_scan_time > 0.0 else -1.0
        self.get_logger().info(
            "=== DIAG === tubes=%d feas=%d scan_age=%.3f locked_w=%s w_hold_left=%.2f turn_sign=%d turn_hold_left=%.2f recent=%d"
            % (
                len(self.motion_tubes),
                len(feas),
                scan_age,
                str(self.locked_w_value),
                max(0.0, self.w_lock_until - self._now_sec()),
                self.committed_turn_sign,
                max(0.0, self.turn_commit_until - self._now_sec()),
                len(self.recent_positions),
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = GoalOrientedMotionTubePlannerROS2()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
