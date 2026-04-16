#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
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
    def __init__(self, length=0.45, width=0.40):
        self.length = float(length)
        self.width = float(width)
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
        super().__init__("goal_oriented_motion_tube_planner_ros2_active_group_recovery_fixedscan")

        self.cb_group = ReentrantCallbackGroup()
        self.timer_group = MutuallyExclusiveCallbackGroup()

        self.goal_lock = threading.Lock()
        self.planning_cycle_lock = threading.Lock()
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
        self._is_shutting_down = False

        self._declare_parameters()
        self._load_parameters()

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, QoSProfile(depth=10))
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, QoSProfile(depth=10))

        self.robot = RobotGeometry(length=self.robot_length, width=self.robot_width)

        self.progress_history = []
        self.recent_positions = []
        self.stuck_counter = 0

        self.motion_tubes = []
        self.selected_tube = None
        self.min_cost = 0.0
        self.max_cost = 1.0

        self.goal_start_time = None
        self.goal_start_xy = None

        self._ranges_clipped = None
        self._fwd_cache = float("inf")
        self.no_feasible = False

        self.w_lock_until = 0.0
        self.locked_w_value = None

        self.turn_commit_until = 0.0
        self.committed_turn_sign = 0

        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        # active group
        self.active_group = None
        self.active_group_until = 0.0

        # trap memory
        self.trap_memory = []
        self.last_goal_dist = None
        self.g1_blocked_counter = 0
        self.g2_selected_counter = 0
        self.trap_progress_counter = 0

        # recovery state
        self.recovery_state = "NONE"   # NONE / PRE_RECOVERY_CHECK / BACKUP / SCAN
        self.recovery_enter_t = 0.0
        self.recovery_backup_start_xy = None
        self.recovery_backup_start_t = 0.0
        self.recovery_base_yaw = 0.0
        self.recovery_target_yaw = 0.0
        self.recovery_current_offset_deg = 0.0
        self.recovery_last_switch_t = 0.0
        self.recovery_points = []
        self.recovery_scan_sequence_deg = []
        self.recovery_scan_index = 0

        self.pre_recovery_start_t = 0.0
        self.pre_recovery_rechecks_done = 0

        self._setup_subscriptions_or_discovery()
        self._tube_configs = self._build_grouped_tube_configs()

        self.planning_timer = self.create_timer(
            self.loop_dt,
            self.planning_cycle,
            callback_group=self.timer_group
        )
        self.diag_timer = self.create_timer(
            self.diag_period_sec,
            self.print_diagnostics,
            callback_group=self.timer_group
        )

        self.action_server = ActionServer(
            self,
            NavigateToPose,
            self.action_name,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group,
        )

        self.get_logger().info(
            "Planner params: robot=(L=%.2f,W=%.2f) weights=(progress=%.2f, heading=%.2f, curvature=%.2f, length=%.2f, speed=%.2f) prefilter=(%.2f, %.2f, %.2f) cmd_post=(clr<%.2f=>x%.2f, |w|>%.2f=>x%.2f) diag=%.2fs sleep=%.2fs threads=%d"
            % (
                self.robot_length, self.robot_width,
                self.w_progress, self.w_heading, self.w_curvature, self.w_length, self.w_speed,
                self.tube_prefilter_forward_dist, self.tube_prefilter_low_w, self.tube_prefilter_margin,
                self.cmd_min_clearance_slowdown_dist, self.cmd_min_clearance_slowdown_scale,
                self.cmd_high_turn_thresh, self.cmd_high_turn_scale,
                self.diag_period_sec, self.execute_sleep_sec, self.executor_threads,
            )
        )
        self.get_logger().info("Planner initialized: fixed recovery scan sequence")

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

        self.declare_parameter("w_clearance", 15.0)
        self.declare_parameter("clearance_safe_dist", 0.15)

        self.declare_parameter("w_center_balance", 26.0)
        self.declare_parameter("w_side_clearance", 18.0)
        self.declare_parameter("side_clearance_safe_dist", 0.15)

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

        self.declare_parameter("goal_start_grace_sec", 1.5)

        # active group
        self.declare_parameter("group_hold_time", 1.2)
        self.declare_parameter("g1_min_open_forward", 1.20)
        self.declare_parameter("g1_min_arc_len", 2.50)
        self.declare_parameter("g1_vs_g2_progress_margin", -0.02)
        self.declare_parameter("g2_return_to_g1_open_forward", 1.80)

        # trap memory
        self.declare_parameter("trap_radius", 0.90)
        self.declare_parameter("trap_memory_sec", 18.0)
        self.declare_parameter("trap_trigger_g2_count", 8)
        self.declare_parameter("trap_trigger_low_progress_count", 6)
        self.declare_parameter("trap_progress_eps", 0.05)
        self.declare_parameter("trap_g1_suppress_sec", 12.0)

        # recovery
        self.declare_parameter("enable_recovery", True)
        self.declare_parameter("recovery_backup_dist", 0.20)
        self.declare_parameter("recovery_backup_v", -0.10)
        self.declare_parameter("recovery_backup_timeout", 3.0)
        self.declare_parameter("recovery_backup_min_time", 0.8)
        self.declare_parameter("recovery_scan_step_deg", 50.0)
        self.declare_parameter("recovery_scan_max_deg", 360.0)
        self.declare_parameter("recovery_yaw_tol_deg", 3.0)
        self.declare_parameter("recovery_w", 0.55)
        self.declare_parameter("recovery_min_pause", 0.08)

        self.declare_parameter("pre_recovery_pause_sec", 0.20)
        self.declare_parameter("pre_recovery_recheck_count", 1)

        # extra cost weights
        self.declare_parameter("w_recovery_avoid", 4.0)
        self.declare_parameter("recovery_avoid_radius", 1.5)
        self.declare_parameter("recovery_point_memory_sec", 120.0)
        self.declare_parameter("w_away_from_start", 2.0)

        # final command smoothing
        self.declare_parameter("cmd_v_alpha", 0.45)
        self.declare_parameter("cmd_w_alpha", 0.28)

        # parameterized cost weights / thresholds / geometry / runtime
        self.declare_parameter("w_progress", 34.0)
        self.declare_parameter("w_heading", 15.0)
        self.declare_parameter("w_curvature", 0.8)
        self.declare_parameter("w_length", 2.0)
        self.declare_parameter("w_speed", 0.8)
        self.declare_parameter("goal_factor", 1.0)
        self.declare_parameter("hard_clearance_penalty_dist", 0.08)
        self.declare_parameter("hard_clearance_penalty_value", 3.0)
        self.declare_parameter("center_balance_clip", 0.25)

        self.declare_parameter("tube_prefilter_forward_dist", 0.65)
        self.declare_parameter("tube_prefilter_low_w", 0.18)
        self.declare_parameter("tube_prefilter_margin", 0.20)

        self.declare_parameter("cmd_min_clearance_slowdown_dist", 0.12)
        self.declare_parameter("cmd_min_clearance_slowdown_scale", 0.75)
        self.declare_parameter("cmd_high_turn_thresh", 0.8)
        self.declare_parameter("cmd_high_turn_scale", 0.85)

        self.declare_parameter("robot_length", 0.45)
        self.declare_parameter("robot_width", 0.40)
        self.declare_parameter("diag_period_sec", 3.0)
        self.declare_parameter("execute_sleep_sec", 0.10)
        self.declare_parameter("executor_threads", 4)

        self.declare_parameter("group1_w_min", 0.0)
        self.declare_parameter("group1_w_max", 0.2)
        self.declare_parameter("group1_T", [7.0, 10.0])

        self.declare_parameter("group2_w_min", 0.5)
        self.declare_parameter("group2_w_max", 1.2)
        self.declare_parameter("group2_T", [2.5, 4.0, 5.0])

        self.declare_parameter("group3_w_min", 0.0)
        self.declare_parameter("group3_w_max", 0.2)
        self.declare_parameter("group3_T", [2.5, 3.0])

        self.declare_parameter("group4_w_min", 0.0)
        self.declare_parameter("group4_w_max", 1.2)
        self.declare_parameter("group4_T", [1.0, 1.5])

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

        self.goal_start_grace_sec = float(gp("goal_start_grace_sec").value)

        self.group_hold_time = float(gp("group_hold_time").value)
        self.g1_min_open_forward = float(gp("g1_min_open_forward").value)
        self.g1_min_arc_len = float(gp("g1_min_arc_len").value)
        self.g1_vs_g2_progress_margin = float(gp("g1_vs_g2_progress_margin").value)
        self.g2_return_to_g1_open_forward = float(gp("g2_return_to_g1_open_forward").value)

        self.trap_radius = float(gp("trap_radius").value)
        self.trap_memory_sec = float(gp("trap_memory_sec").value)
        self.trap_trigger_g2_count = int(gp("trap_trigger_g2_count").value)
        self.trap_trigger_low_progress_count = int(gp("trap_trigger_low_progress_count").value)
        self.trap_progress_eps = float(gp("trap_progress_eps").value)
        self.trap_g1_suppress_sec = float(gp("trap_g1_suppress_sec").value)

        self.enable_recovery = bool(gp("enable_recovery").value)
        self.recovery_backup_dist = float(gp("recovery_backup_dist").value)
        self.recovery_backup_v = float(gp("recovery_backup_v").value)
        self.recovery_backup_timeout = float(gp("recovery_backup_timeout").value)
        self.recovery_backup_min_time = float(gp("recovery_backup_min_time").value)
        self.recovery_scan_step_deg = float(gp("recovery_scan_step_deg").value)
        self.recovery_scan_max_deg = float(gp("recovery_scan_max_deg").value)
        self.recovery_yaw_tol_deg = float(gp("recovery_yaw_tol_deg").value)
        self.recovery_w = float(gp("recovery_w").value)
        self.recovery_min_pause = float(gp("recovery_min_pause").value)

        self.pre_recovery_pause_sec = float(gp("pre_recovery_pause_sec").value)
        self.pre_recovery_recheck_count = int(gp("pre_recovery_recheck_count").value)

        self.w_recovery_avoid = float(gp("w_recovery_avoid").value)
        self.recovery_avoid_radius = float(gp("recovery_avoid_radius").value)
        self.recovery_point_memory_sec = float(gp("recovery_point_memory_sec").value)
        self.w_away_from_start = float(gp("w_away_from_start").value)

        self.cmd_v_alpha = float(gp("cmd_v_alpha").value)
        self.cmd_w_alpha = float(gp("cmd_w_alpha").value)

        self.w_progress = float(gp("w_progress").value)
        self.w_heading = float(gp("w_heading").value)
        self.w_curvature = float(gp("w_curvature").value)
        self.w_length = float(gp("w_length").value)
        self.w_speed = float(gp("w_speed").value)
        self.goal_factor = float(gp("goal_factor").value)
        self.hard_clearance_penalty_dist = float(gp("hard_clearance_penalty_dist").value)
        self.hard_clearance_penalty_value = float(gp("hard_clearance_penalty_value").value)
        self.center_balance_clip = float(gp("center_balance_clip").value)

        self.tube_prefilter_forward_dist = float(gp("tube_prefilter_forward_dist").value)
        self.tube_prefilter_low_w = float(gp("tube_prefilter_low_w").value)
        self.tube_prefilter_margin = float(gp("tube_prefilter_margin").value)

        self.cmd_min_clearance_slowdown_dist = float(gp("cmd_min_clearance_slowdown_dist").value)
        self.cmd_min_clearance_slowdown_scale = float(gp("cmd_min_clearance_slowdown_scale").value)
        self.cmd_high_turn_thresh = float(gp("cmd_high_turn_thresh").value)
        self.cmd_high_turn_scale = float(gp("cmd_high_turn_scale").value)

        self.robot_length = float(gp("robot_length").value)
        self.robot_width = float(gp("robot_width").value)
        self.diag_period_sec = float(gp("diag_period_sec").value)
        self.execute_sleep_sec = float(gp("execute_sleep_sec").value)
        self.executor_threads = int(gp("executor_threads").value)

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
            self.discovery_timer = self.create_timer(
                1.0,
                self._topic_discovery_step,
                callback_group=self.cb_group
            )
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
        self.scan_sub = self.create_subscription(
            LaserScan, topic_name, self.scan_callback, qos_profile_sensor_data, callback_group=self.cb_group
        )
        self.latest_scan_topic = topic_name
        self.get_logger().info("Subscribed scan: %s" % topic_name)

    def _create_odom_subscription(self, topic_name):
        if self.odom_sub is not None:
            return
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.odom_sub = self.create_subscription(
            Odometry, topic_name, self.odom_callback, odom_qos, callback_group=self.cb_group
        )
        self.latest_odom_topic = topic_name
        self.get_logger().info("Subscribed odom: %s" % topic_name)

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

    def _safe_publish_cmd(self, cmd):
        if self._is_shutting_down:
            return
        try:
            self.cmd_pub.publish(cmd)
        except Exception:
            pass

    def _safe_publish_markers(self, marker_array):
        if self._is_shutting_down:
            return
        try:
            self.marker_pub.publish(marker_array)
        except Exception:
            pass

    def _publish_zero(self):
        cmd = Twist()
        self._safe_publish_cmd(cmd)
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

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

    def _group_rank(self, group_name):
        ranks = {
            "G1_low_w_longT": 0,
            "G2_mid_w_turn": 1,
            "G3_low_w_midT": 2,
            "G4_high_w_shortT": 3,
        }
        return ranks.get(group_name, 99)

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
            self.get_logger().info(
                "Laser initialized: beams=%d fov=%.1f deg"
                % (len(msg.ranges), math.degrees(msg.angle_max - msg.angle_min))
            )

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
            self.goal_start_xy = self._current_xy()

            self.progress_history = []
            self.recent_positions = []
            self.stuck_counter = 0

            self.locked_w_value = None
            self.w_lock_until = 0.0
            self.committed_turn_sign = 0
            self.turn_commit_until = 0.0

            self.last_cmd_v = 0.0
            self.last_cmd_w = 0.0

            self.active_group = None
            self.active_group_until = 0.0

            self.last_goal_dist = None
            self.g1_blocked_counter = 0
            self.g2_selected_counter = 0
            self.trap_progress_counter = 0

            self.recovery_state = "NONE"
            self.recovery_enter_t = 0.0
            self.recovery_backup_start_xy = None
            self.recovery_backup_start_t = 0.0
            self.recovery_current_offset_deg = 0.0
            self.recovery_last_switch_t = 0.0
            self.recovery_scan_sequence_deg = []
            self.recovery_scan_index = 0

            self.pre_recovery_start_t = 0.0
            self.pre_recovery_rechecks_done = 0

        self.get_logger().info(
            "New goal: (%.2f, %.2f) frame=%s"
            % (
                self.current_goal.pose.position.x,
                self.current_goal.pose.position.y,
                self.current_goal.header.frame_id,
            )
        )

        while rclpy.ok() and (not self._is_shutting_down):
            if goal_handle.is_cancel_requested:
                self._publish_zero()
                try:
                    goal_handle.canceled()
                except Exception:
                    pass
                with self.goal_lock:
                    self.current_goal = None
                    self.current_goal_handle = None
                return self._make_action_result(success=False)

            if self.current_pose is not None and self.is_goal_reached():
                self._publish_zero()
                try:
                    goal_handle.succeed()
                except Exception:
                    pass
                with self.goal_lock:
                    self.current_goal = None
                    self.current_goal_handle = None
                return self._make_action_result(success=True)

            if self.goal_start_time is not None:
                try:
                    fb = NavigateToPose.Feedback()
                    if hasattr(fb, "current_pose") and self.current_pose is not None:
                        fb.current_pose = self.current_pose
                    if hasattr(fb, "navigation_time"):
                        fb.navigation_time = self._make_duration_msg(self._now_sec() - self.goal_start_time)
                    goal_handle.publish_feedback(fb)
                except Exception:
                    break

            time.sleep(self.execute_sleep_sec)

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
        if not self.planning_cycle_lock.acquire(blocking=False):
            return

        try:
            now = self._now_sec()
            if not self.planning_enabled or self.current_goal is None or self.current_pose is None or self.latest_scan is None:
                self._publish_zero()
                return

            scan_age = now - self.last_scan_time
            odom_age = now - self.last_odom_time
            if scan_age > self.scan_timeout or odom_age > self.odom_timeout:
                self._publish_zero()
                return

            self._cleanup_trap_memory(now)
            self._cleanup_recovery_points(now)

            self._fwd_cache = self._compute_forward_clearance_cached()
            self.update_progress_history(now)

            if self.recovery_state != "NONE":
                self._recovery_step(now)
                return

            self.generate_motion_tubes()
            self.evaluate_tubes()

            feas = [t for t in self.motion_tubes if t.is_feasible]
            self.no_feasible = len(feas) == 0

            in_goal_start_grace = (
                self.goal_start_time is not None
                and (now - self.goal_start_time) < self.goal_start_grace_sec
            )

            if in_goal_start_grace and len(feas) == 0:
                self.selected_tube = None
                self._publish_zero()
                self._publish_tube_markers()
                self._log_throttle(
                    "info",
                    "goal_start_grace",
                    0.5,
                    "Goal-start grace active: t=%.2f/%.2fs feas=%d"
                    % (now - self.goal_start_time, self.goal_start_grace_sec, len(feas)),
                )
                return

            if self.no_feasible:
                self.selected_tube = None
                if self.enable_recovery:
                    self._enter_pre_recovery(now)
                    self._recovery_step(now)
                else:
                    self._publish_zero()
                    self._publish_tube_markers()
                return

            self.select_best_tube()
            self._update_trap_state(now)
            self.publish_commands()
            self._publish_tube_markers()

        finally:
            self.planning_cycle_lock.release()

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

    def _current_xy(self):
        if self.current_pose is None:
            return (0.0, 0.0)
        p = self.current_pose.pose.position
        return (float(p.x), float(p.y))

    def _dist_xy(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

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
            if fwd < self.tube_prefilter_forward_dist and abs(w) < self.tube_prefilter_low_w and approx_len > (fwd + self.tube_prefilter_margin):
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

        idxs = np.linspace(0, len(tube.samples) - 1, num=min(12, len(tube.samples)), dtype=int)

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

    def _tube_end_world(self, tube):
        cx, cy, yaw = self._pose_xy_yaw()
        if tube is None or len(tube.samples) == 0:
            return (cx, cy)
        end_local = tube.samples[-1]
        sx, sy = float(end_local[0]), float(end_local[1])
        ex = cx + sx * math.cos(yaw) - sy * math.sin(yaw)
        ey = cy + sx * math.sin(yaw) + sy * math.cos(yaw)
        return (ex, ey)

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

        ex, ey = self._tube_end_world(tube)

        min_d = float("inf")
        for _, px, py in self.recent_positions:
            d = math.hypot(ex - px, ey - py)
            if d < min_d:
                min_d = d

        if min_d < self.revisit_radius:
            return self.revisit_penalty_weight * (1.0 - min_d / max(1e-6, self.revisit_radius))
        return 0.0

    def recovery_avoid_penalty(self, tube, now):
        if len(self.recovery_points) == 0:
            return 0.0

        ex, ey = self._tube_end_world(tube)
        penalty = 0.0

        kept = []
        for item in self.recovery_points:
            if (now - item["t"]) <= self.recovery_point_memory_sec:
                kept.append(item)
                d = math.hypot(ex - item["xy"][0], ey - item["xy"][1])
                if d < self.recovery_avoid_radius:
                    penalty += self.w_recovery_avoid * (1.0 - d / max(1e-6, self.recovery_avoid_radius))
        self.recovery_points = kept
        return penalty

    def away_from_start_reward(self, tube):
        if self.goal_start_xy is None:
            return 0.0
        ex, ey = self._tube_end_world(tube)
        d = math.hypot(ex - self.goal_start_xy[0], ey - self.goal_start_xy[1])
        return self.w_away_from_start * min(d, 2.0)

    def composite_cost(self, t, goal_bearing, goal_factor=1.0):
        now = self._now_sec()

        w_progress = self.w_progress
        w_heading = self.w_heading
        w_curvature = self.w_curvature
        w_length = self.w_length
        w_speed = self.w_speed
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

        if t.min_clearance < self.hard_clearance_penalty_dist:
            c += self.hard_clearance_penalty_value

        side_min = min(t.left_clearance, t.right_clearance)
        if side_min < self.side_clearance_safe_dist:
            side_ratio = max(0.0, 1.0 - side_min / max(1e-6, self.side_clearance_safe_dist))
            c += self.w_side_clearance * side_ratio

        c += self.w_center_balance * min(self.center_balance_clip, t.center_balance)

        c += self.recovery_avoid_penalty(t, now)
        c -= self.away_from_start_reward(t)

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
        goal_factor = self.goal_factor

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

    def _best_in_group(self, group_name, feas):
        group_feas = [t for t in feas if t.group_name == group_name]
        if not group_feas:
            return None
        return min(group_feas, key=lambda z: z.cost)

    def _trap_suppresses_g1_here(self, now):
        xy = self._current_xy()
        for item in self.trap_memory:
            if (now - item["t"]) <= item["ttl"]:
                if self._dist_xy(xy, item["xy"]) <= self.trap_radius:
                    if item.get("suppress_g1", False):
                        return True
        return False

    def _cleanup_trap_memory(self, now):
        kept = []
        for item in self.trap_memory:
            if (now - item["t"]) <= item["ttl"]:
                kept.append(item)
        self.trap_memory = kept

    def _cleanup_recovery_points(self, now):
        kept = []
        for item in self.recovery_points:
            if (now - item["t"]) <= self.recovery_point_memory_sec:
                kept.append(item)
        self.recovery_points = kept

    def _record_trap_memory(self, now):
        xy = self._current_xy()
        self.trap_memory.append({
            "xy": xy,
            "t": now,
            "ttl": self.trap_g1_suppress_sec,
            "suppress_g1": True,
        })
        self._log_throttle(
            "warn",
            "trap_memory_add",
            1.0,
            "Trap memory added at (%.2f, %.2f), suppress G1 for %.1fs"
            % (xy[0], xy[1], self.trap_g1_suppress_sec),
        )

    def _choose_desired_group(self, feas, now):
        g1 = self._best_in_group("G1_low_w_longT", feas)
        g2 = self._best_in_group("G2_mid_w_turn", feas)
        g3 = self._best_in_group("G3_low_w_midT", feas)
        g4 = self._best_in_group("G4_high_w_shortT", feas)

        suppress_g1 = self._trap_suppresses_g1_here(now)

        g1_good = False
        if g1 is not None and not suppress_g1:
            g1_good = (
                self._fwd_cache >= self.g1_min_open_forward
                and g1.arc_len >= self.g1_min_arc_len
            )
            if g2 is not None:
                g1_good = g1_good and (g1.goal_progress >= g2.goal_progress + self.g1_vs_g2_progress_margin)

        if self.active_group is not None and now < self.active_group_until:
            active_best = self._best_in_group(self.active_group, feas)
            if active_best is not None:
                return self.active_group

        if self.active_group == "G2_mid_w_turn":
            if g2 is not None and self._fwd_cache < self.g2_return_to_g1_open_forward:
                return "G2_mid_w_turn"

        if g1_good:
            return "G1_low_w_longT"
        if g2 is not None:
            return "G2_mid_w_turn"
        if g3 is not None:
            return "G3_low_w_midT"
        if g4 is not None:
            return "G4_high_w_shortT"

        if g1 is not None and not suppress_g1:
            return "G1_low_w_longT"
        if g3 is not None:
            return "G3_low_w_midT"
        if g4 is not None:
            return "G4_high_w_shortT"
        return None

    def _candidate_key_in_group(self, tube):
        transition_penalty = 0.0
        if self.selected_tube is not None:
            transition_penalty += 1.2 * abs(tube.w - self.selected_tube.w)
            transition_penalty += 0.20 * abs(tube.T - self.selected_tube.T)
            if tube.group_name == self.selected_tube.group_name:
                transition_penalty -= 0.15
        return tube.cost + transition_penalty

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

        desired_group = self._choose_desired_group(feas, now)
        if desired_group is None:
            self.selected_tube = min(feas, key=lambda z: z.cost)
            return

        if self.active_group != desired_group:
            self.active_group = desired_group
            self.active_group_until = now + self.group_hold_time

        group_feas = [t for t in feas if t.group_name == self.active_group]
        if not group_feas:
            self.selected_tube = min(feas, key=lambda z: (self._group_rank(z.group_name), z.cost))
            return

        if self.locked_w_value is not None and now < self.w_lock_until:
            same_w = [t for t in group_feas if self._same_w(t.w, self.locked_w_value)]
            if same_w:
                candidate = min(same_w, key=self._candidate_key_in_group)
                self.selected_tube = candidate
                return
            else:
                self.locked_w_value = None
                self.w_lock_until = 0.0

        candidate = min(group_feas, key=self._candidate_key_in_group)

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

    def _update_trap_state(self, now):
        if self.current_goal is None or self.current_pose is None or self.selected_tube is None:
            return

        x, y, _ = self._pose_xy_yaw()
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        goal_dist = math.hypot(gx - x, gy - y)

        if self.last_goal_dist is None:
            delta_goal = 0.0
        else:
            delta_goal = self.last_goal_dist - goal_dist
        self.last_goal_dist = goal_dist

        g1_best = self._best_in_group("G1_low_w_longT", [t for t in self.motion_tubes if t.is_feasible])
        if g1_best is None:
            self.g1_blocked_counter += 1
        else:
            if self._fwd_cache < self.g1_min_open_forward or g1_best.arc_len < self.g1_min_arc_len:
                self.g1_blocked_counter += 1
            else:
                self.g1_blocked_counter = max(0, self.g1_blocked_counter - 1)

        if self.selected_tube.group_name == "G2_mid_w_turn":
            self.g2_selected_counter += 1
        else:
            self.g2_selected_counter = max(0, self.g2_selected_counter - 1)

        if delta_goal < self.trap_progress_eps:
            self.trap_progress_counter += 1
        else:
            self.trap_progress_counter = max(0, self.trap_progress_counter - 1)

        should_record_trap = (
            self.g2_selected_counter >= self.trap_trigger_g2_count
            and self.trap_progress_counter >= self.trap_trigger_low_progress_count
            and self.g1_blocked_counter >= self.trap_trigger_low_progress_count
        )

        if should_record_trap:
            if not self._trap_suppresses_g1_here(now):
                self._record_trap_memory(now)
            self.g2_selected_counter = 0
            self.trap_progress_counter = 0
            self.g1_blocked_counter = 0

    # =========================
    # Recovery
    # =========================
    def _build_fixed_recovery_scan_sequence(self):
        seq = [+30, +60, -60, +90, +120, +150]
        max_deg = float(self.recovery_scan_max_deg)
        return [x for x in seq if abs(x) <= max_deg]

    def _enter_recovery(self, now):
        if self.recovery_state != "NONE":
            return

        self.recovery_state = "BACKUP"
        self.recovery_enter_t = now
        self.recovery_backup_start_t = now
        self.recovery_backup_start_xy = self._current_xy()

        _, _, yaw = self._pose_xy_yaw()
        self.recovery_base_yaw = yaw
        self.recovery_target_yaw = yaw
        self.recovery_current_offset_deg = 0.0
        self.recovery_last_switch_t = now

        self.recovery_scan_sequence_deg = self._build_fixed_recovery_scan_sequence()
        self.recovery_scan_index = 0

        self.recovery_points.append({
            "xy": self._current_xy(),
            "t": now,
        })

        self._log_throttle(
            "warn",
            "enter_recovery",
            0.5,
            "RECOVERY enter at (%.2f, %.2f): BACKUP then fixed SCAN sequence"
            % (self._current_xy()[0], self._current_xy()[1]),
        )

    def _has_representative_recovery_tube(self, feasible_tubes):
        for t in feasible_tubes:
            if t.group_name in ("G1_low_w_longT", "G2_mid_w_turn", "G3_low_w_midT"):
                return True
        return False

    def _exit_recovery(self):
        self.recovery_state = "NONE"
        self.recovery_backup_start_xy = None
        self.recovery_backup_start_t = 0.0
        self.recovery_current_offset_deg = 0.0
        self.recovery_last_switch_t = 0.0
        self.recovery_scan_sequence_deg = []
        self.recovery_scan_index = 0

    def _enter_pre_recovery(self, now):
        if self.recovery_state != "NONE":
            return
        self.recovery_state = "PRE_RECOVERY_CHECK"
        self.pre_recovery_start_t = now
        self.pre_recovery_rechecks_done = 0
        self.selected_tube = None
        self._publish_zero()
        self._log_throttle(
            "warn",
            "enter_pre_recovery",
            0.5,
            "PRE_RECOVERY_CHECK enter: stop first, then recheck feasible tubes"
        )

    def _recovery_backup_cmd_or_done(self, now):
        if self.recovery_backup_start_xy is None:
            self.recovery_backup_start_xy = self._current_xy()
        if self.recovery_backup_start_t <= 1e-6:
            self.recovery_backup_start_t = now

        moved = self._dist_xy(self._current_xy(), self.recovery_backup_start_xy)
        elapsed = now - self.recovery_backup_start_t
        done_dist = moved >= self.recovery_backup_dist
        done_time = elapsed >= self.recovery_backup_min_time

        if (now - self.recovery_enter_t) > self.recovery_backup_timeout:
            return Twist(), True

        if done_dist and done_time:
            return Twist(), True

        cmd = Twist()
        cmd.linear.x = self.recovery_backup_v
        cmd.angular.z = 0.0
        return cmd, False

    def _recovery_rotate_cmd_or_done(self):
        _, _, yaw = self._pose_xy_yaw()
        err = self._wrap(self.recovery_target_yaw - yaw)
        tol = math.radians(self.recovery_yaw_tol_deg)

        if abs(err) <= tol:
            return Twist(), True

        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = self.recovery_w if err > 0 else -self.recovery_w
        return cmd, False

    def _recovery_step(self, now):
        if self.recovery_state == "PRE_RECOVERY_CHECK":
            self._publish_zero()
            self._publish_tube_markers()

            if (now - self.pre_recovery_start_t) < self.pre_recovery_pause_sec:
                return

            self.generate_motion_tubes()
            self.evaluate_tubes()
            feas = [t for t in self.motion_tubes if t.is_feasible]
            has_representative = self._has_representative_recovery_tube(feas)

            self.pre_recovery_rechecks_done += 1

            if has_representative:
                self._log_throttle(
                    "info",
                    "pre_recovery_cancelled",
                    0.5,
                    "PRE_RECOVERY_CHECK cancelled: representative tube found, resume normal mode"
                )
                self.recovery_state = "NONE"
                self.pre_recovery_start_t = 0.0
                self.pre_recovery_rechecks_done = 0

                self.select_best_tube()
                self.publish_commands()
                self._publish_tube_markers()
                return

            if self.pre_recovery_rechecks_done >= self.pre_recovery_recheck_count:
                self.recovery_state = "NONE"
                self.pre_recovery_start_t = 0.0
                self.pre_recovery_rechecks_done = 0
                self._enter_recovery(now)
                self._publish_zero()
                return

            self.pre_recovery_start_t = now
            return

        if self.recovery_state == "BACKUP":
            cmd, done = self._recovery_backup_cmd_or_done(now)
            self._safe_publish_cmd(cmd)
            self._publish_tube_markers()

            if not done:
                return

            self.recovery_state = "SCAN"
            _, _, yaw = self._pose_xy_yaw()
            self.recovery_base_yaw = yaw

            if len(self.recovery_scan_sequence_deg) == 0:
                self.recovery_scan_sequence_deg = self._build_fixed_recovery_scan_sequence()
            self.recovery_scan_index = 0

            if len(self.recovery_scan_sequence_deg) == 0:
                self.get_logger().warning("RECOVERY scan sequence empty, exit recovery")
                self._exit_recovery()
                self._publish_zero()
                return

            first_offset = self.recovery_scan_sequence_deg[self.recovery_scan_index]
            self.recovery_current_offset_deg = first_offset
            self.recovery_target_yaw = self._wrap(
                yaw + math.radians(first_offset)
            )
            self.recovery_last_switch_t = now
            self._safe_publish_cmd(Twist())
            return

        if self.recovery_state == "SCAN":
            cmd, done = self._recovery_rotate_cmd_or_done()
            self._safe_publish_cmd(cmd)
            self._publish_tube_markers()

            if not done:
                return

            if (now - self.recovery_last_switch_t) < self.recovery_min_pause:
                self._safe_publish_cmd(Twist())
                return

            self.generate_motion_tubes()
            self.evaluate_tubes()
            feas = [t for t in self.motion_tubes if t.is_feasible]

            has_representative = self._has_representative_recovery_tube(feas)

            if has_representative:
                self._log_throttle(
                    "warn",
                    "recovery_found_repr",
                    0.5,
                    "RECOVERY found representative feasible tubes at %.0f deg"
                    % self.recovery_current_offset_deg
                )
                self._exit_recovery()
                self.select_best_tube()
                self.publish_commands()
                self._publish_tube_markers()
                return

            if len(feas) > 0 and not has_representative:
                self._log_throttle(
                    "info",
                    "recovery_only_local",
                    0.5,
                    "RECOVERY sees only local tubes (e.g. G4) at %.0f deg, keep rotating"
                    % self.recovery_current_offset_deg
                )

            self.recovery_scan_index += 1

            seq_len = len(self.recovery_scan_sequence_deg)
            if seq_len == 0:
                self.get_logger().warning("RECOVERY scan sequence empty during SCAN, exit recovery")
                self._exit_recovery()
                self._publish_zero()
                return

            if self.recovery_scan_index >= seq_len:
                self._log_throttle(
                    "warn",
                    "recovery_full_scan",
                    0.5,
                    "RECOVERY exhausted fixed scan sequence, restarting"
                )
                _, _, yaw = self._pose_xy_yaw()
                self.recovery_base_yaw = yaw
                self.recovery_scan_index = 0

            if self.recovery_scan_index < 0 or self.recovery_scan_index >= len(self.recovery_scan_sequence_deg):
                self.get_logger().warning(
                    "RECOVERY index invalid after reset: idx=%d len=%d, force reset"
                    % (self.recovery_scan_index, len(self.recovery_scan_sequence_deg))
                )
                self.recovery_scan_index = 0
                if len(self.recovery_scan_sequence_deg) == 0:
                    self._exit_recovery()
                    self._publish_zero()
                    return

            next_offset = self.recovery_scan_sequence_deg[self.recovery_scan_index]
            self.recovery_current_offset_deg = next_offset
            self.recovery_target_yaw = self._wrap(
                self.recovery_base_yaw + math.radians(next_offset)
            )
            self.recovery_last_switch_t = now
            self._safe_publish_cmd(Twist())
            return

        self._exit_recovery()
        self._publish_zero()

    def publish_commands(self):
        raw_v = 0.0
        raw_w = 0.0

        if self.selected_tube:
            raw_v = max(0.0, min(self.max_v, self.selected_tube.v))
            raw_w = max(-self.max_w, min(self.max_w, self.selected_tube.w))

            if abs(raw_w) < self.deadband_w:
                raw_w = 0.0

            if self.use_fwd_slowdown and self.fwd_slow_gain > 1e-6:
                fwd = self._fwd_cache
                scale = max(self.min_forward_scale, min(1.0, fwd / (1.0 + 1e-6)))
                raw_v *= scale ** (1.0 + self.fwd_slow_gain)

            if self.selected_tube.min_clearance < self.cmd_min_clearance_slowdown_dist:
                raw_v *= self.cmd_min_clearance_slowdown_scale
            if abs(raw_w) > self.cmd_high_turn_thresh:
                raw_v *= self.cmd_high_turn_scale

        v_alpha = max(0.0, min(1.0, self.cmd_v_alpha))
        w_alpha = max(0.0, min(1.0, self.cmd_w_alpha))

        smooth_v = v_alpha * raw_v + (1.0 - v_alpha) * self.last_cmd_v
        smooth_w = w_alpha * raw_w + (1.0 - w_alpha) * self.last_cmd_w

        if abs(smooth_w) < self.deadband_w:
            smooth_w = 0.0

        cmd = Twist()
        cmd.linear.x = smooth_v
        cmd.angular.z = smooth_w
        self._safe_publish_cmd(cmd)

        self.last_cmd_v = smooth_v
        self.last_cmd_w = smooth_w

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
        if not self.motion_tubes or not self.current_pose or self._is_shutting_down:
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

        self._safe_publish_markers(ma)

    def print_diagnostics(self):
        feas = [t for t in self.motion_tubes if t.is_feasible]
        scan_age = self._now_sec() - self.last_scan_time if self.last_scan_time > 0.0 else -1.0
        sel = "None"
        if self.selected_tube is not None:
            sel = "g=%s v=%.2f w=%.2f T=%.1f cost=%.2f" % (
                self.selected_tube.group_name,
                self.selected_tube.v,
                self.selected_tube.w,
                self.selected_tube.T,
                self.selected_tube.cost,
            )

        self.get_logger().info(
            "=== DIAG === tubes=%d feas=%d scan_age=%.3f fwd=%.2f active_group=%s hold_left=%.2f "
            "recovery=%s rec_pts=%d rec_idx=%d locked_w=%s w_hold_left=%.2f turn_sign=%d turn_hold_left=%.2f "
            "trap_mem=%d cmd=(%.2f, %.2f) sel=[%s]"
            % (
                len(self.motion_tubes),
                len(feas),
                scan_age,
                self._fwd_cache,
                str(self.active_group),
                max(0.0, self.active_group_until - self._now_sec()),
                self.recovery_state,
                len(self.recovery_points),
                self.recovery_scan_index,
                str(self.locked_w_value),
                max(0.0, self.w_lock_until - self._now_sec()),
                self.committed_turn_sign,
                max(0.0, self.turn_commit_until - self._now_sec()),
                len(self.trap_memory),
                self.last_cmd_v,
                self.last_cmd_w,
                sel,
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = GoalOrientedMotionTubePlannerROS2()
    executor = MultiThreadedExecutor(num_threads=node.executor_threads)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._is_shutting_down = True
        try:
            node._publish_zero()
        except Exception:
            pass
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()