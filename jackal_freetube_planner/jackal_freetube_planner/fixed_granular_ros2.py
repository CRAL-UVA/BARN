#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
ros2 run jackal_freetube_planner fixed_granular_ros2 --ros-args   -p vfh_recovery_threshold:=3.0   -p vfh_recovery_min_valley_width:=2   -p vfh_recovery_wide_valley_min:=3   -p vfh_recovery_smooth_width:=1   -p vfh_recovery_sector_count:=120   -p scan_topic:=/scan   -p odom_topic:=/j100_0896/platform/odom   -p cmd_topic:=/j100_0896/platform/cmd_vel_unstamped   -p sweep_aug_dist:=0.01   -p sweep_extra_margin:=0.01   -p clearance_safe_dist:=0.02   -p side_clearance_safe_dist:=0.04   -p vfh_recovery_front_bias:=0.0   -p vfh_recovery_retry_turn_deg:=35.0   -p sweep_sample_dist:=0.03   -p w_side_clearance:=10.0   -p tube_obstacle_proximity_dist:=0.10   -p w_tube_obstacle_proximity:=0.0   -p enable_green_center_selection:=true   -p green_cost_ratio:=0.3   -p green_center_min_candidates:=2 

ros2 run jackal_freetube_planner fixed_granular_ros2 --ros-args   -p vfh_recovery_threshold:=3.0   -p vfh_recovery_min_valley_width:=2   -p vfh_recovery_wide_valley_min:=3   -p vfh_recovery_smooth_width:=1   -p vfh_recovery_sector_count:=120   -p scan_topic:=/scan   -p odom_topic:=/j100_0896/platform/odom   -p cmd_topic:=/j100_0896/platform/cmd_vel_unstamped   -p sweep_aug_dist:=0.01   -p sweep_extra_margin:=0.01   -p clearance_safe_dist:=0.02   -p side_clearance_safe_dist:=0.04   -p vfh_recovery_front_bias:=0.0   -p vfh_recovery_retry_turn_deg:=35.0   -p sweep_sample_dist:=0.03   -p w_side_clearance:=10.0   -p tube_obstacle_proximity_dist:=0.10   -p w_tube_obstacle_proximity:=0.0   -p enable_green_center_selection:=true   -p green_cost_ratio:=0.5   -p green_center_min_candidates:=2 -p vfh_recovery_front_bias:=0.15
'''


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
    def __init__(self, length=0.4, width=0.4):
        self.length = float(length)
        self.width = float(width)
        self.half_length = self.length / 2.0
        self.half_width = self.width / 2.0
        self.radius = math.hypot(self.half_length, self.half_width)


class MotionTube(object):
    def __init__(self, template_type, v, w, T, samples, beam_indices, arc_len, group_name="UNGROUPED", centerline_samples=None):
        self.template_type = template_type
        self.v = v
        self.w = w
        self.T = T
        self.samples = samples
        self.beam_indices = beam_indices
        self.arc_len = arc_len
        self.group_name = group_name
        self.centerline_samples = centerline_samples if centerline_samples is not None else list(samples)

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
        self.path_trace = []
        self.stuck_counter = 0

        self.motion_tubes = []
        self.precomputed_tubes = []
        self.tubes_precomputed = False
        self._tube_precompute_signature = None
        self.selected_tube = None
        self.min_cost = 0.0
        self.max_cost = 1.0

        self.goal_start_time = None
        self.goal_start_xy = None

        self._ranges_clipped = None
        self._fwd_cache = float("inf")
        self.no_feasible = False
        self.no_feasible_since = None

        self.w_lock_until = 0.0
        self.locked_w_value = None

        self.turn_commit_until = 0.0
        self.committed_turn_sign = 0

        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        # active group
        self.active_group = None
        self.active_group_until = 0.0

        # recovery state
        self.recovery_state = "NONE"   # NONE / BACKUP / SCAN
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
        self.vfh_recovery_last_heading = None
        self.vfh_recovery_last_turn_sign = 1.0
        self.vfh_recovery_force_explore = False
        self.recovery_checked_yaws = []
        self.recovery_path_deferred_allowed = False

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
            "Planner params: robot=(L=%.2f,W=%.2f) weights=(progress=%.2f, heading=%.2f, clearance=%.2f, side=%.2f, proximity=%.2f, center=%.2f) prefilter=(%.2f, %.2f, %.2f) cmd_post=(clr<%.2f=>x%.2f, |w|>%.2f=>x%.2f) diag=%.2fs sleep=%.2fs threads=%d"
            % (
                self.robot_length, self.robot_width,
                self.w_progress, self.w_heading, self.w_clearance, self.w_side_clearance,
                self.w_tube_obstacle_proximity, self.w_center_balance,
                self.tube_prefilter_forward_dist, self.tube_prefilter_low_w, self.tube_prefilter_margin,
                self.cmd_min_clearance_slowdown_dist, self.cmd_min_clearance_slowdown_scale,
                self.cmd_high_turn_thresh, self.cmd_high_turn_scale,
                self.diag_period_sec, self.execute_sleep_sec, self.executor_threads,
            )
        )
        self.get_logger().info("Planner initialized: VFH recovery scan with fixed-scan fallback")

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
        self.declare_parameter("max_w", 2.0)
        self.declare_parameter("w_sample_step", 0.05)

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

        self.declare_parameter("w_center_balance", 3.0)
        self.declare_parameter("w_side_clearance", 18.0)
        self.declare_parameter("side_clearance_safe_dist", 0.15)

        self.declare_parameter("stuck_window_sec", 2.0)
        self.declare_parameter("stuck_min_progress", 0.10)
        self.declare_parameter("stuck_min_motion", 0.18)
        self.declare_parameter("stuck_confirm_count", 3)
        self.declare_parameter("enable_path_trace_marker", True)
        self.declare_parameter("path_trace_memory_sec", 90.0)
        self.declare_parameter("path_trace_min_dist", 0.05)
        self.declare_parameter("path_trace_max_points", 1200)

        self.declare_parameter("w_hold_time", 0.15)
        self.declare_parameter("turn_commit_time", 4.0)
        self.declare_parameter("opposite_turn_penalty", 196.0)
        self.declare_parameter("recent_pos_memory_sec", 5.0)
        self.declare_parameter("revisit_radius", 0.35)
        self.declare_parameter("revisit_penalty_weight", 5.0)

        self.declare_parameter("goal_start_grace_sec", 1.5)
        self.declare_parameter("no_feasible_confirm_sec", 1.0)

        # active group
        self.declare_parameter("enable_active_group", True)
        self.declare_parameter("group_hold_time", 0.15)
        self.declare_parameter("g1_min_open_forward", 1.20)
        self.declare_parameter("g1_min_arc_len", 2.50)
        self.declare_parameter("g1_vs_g2_progress_margin", -0.02)
        self.declare_parameter("g2_return_to_g1_open_forward", 1.80)
        self.declare_parameter("enable_early_turn", True)
        self.declare_parameter("early_turn_forward_dist", 0.80)
        self.declare_parameter("early_turn_min_abs_w", 0.18)
        self.declare_parameter("early_turn_w_bias", 0.35)
        self.declare_parameter("enable_turn_group_w_bias", True)
        self.declare_parameter("turn_group_w_bias", 0.60)
        self.declare_parameter("enable_group_long_tube_bias", False)
        self.declare_parameter("group_long_tube_bias", 0.0)
        self.declare_parameter("enable_green_center_selection", True)
        self.declare_parameter("green_cost_ratio", 0.25)
        self.declare_parameter("green_center_min_candidates", 2)

        # trap memory
        self.declare_parameter("trap_radius", 0.90)
        self.declare_parameter("trap_memory_sec", 18.0)
        self.declare_parameter("trap_trigger_g2_count", 8)
        self.declare_parameter("trap_trigger_low_progress_count", 6)
        self.declare_parameter("trap_progress_eps", 0.05)
        self.declare_parameter("trap_g1_suppress_sec", 12.0)

        # recovery
        self.declare_parameter("enable_recovery", True)
        self.declare_parameter("recovery_use_backup", False)
        self.declare_parameter("recovery_backup_dist", 0.20)
        self.declare_parameter("recovery_backup_v", -0.10)
        self.declare_parameter("recovery_backup_timeout", 3.0)
        self.declare_parameter("recovery_backup_min_time", 0.8)
        self.declare_parameter("recovery_scan_step_deg", 50.0)
        self.declare_parameter("recovery_scan_max_deg", 360.0)
        self.declare_parameter("recovery_yaw_tol_deg", 3.0)
        self.declare_parameter("recovery_w", 0.55)
        self.declare_parameter("recovery_min_pause", 0.08)

        self.declare_parameter("enable_vfh_recovery", True)
        self.declare_parameter("vfh_recovery_sector_count", 90)
        self.declare_parameter("vfh_recovery_smooth_width", 3)
        self.declare_parameter("vfh_recovery_threshold", 3.4)
        self.declare_parameter("vfh_recovery_min_valley_width", 3)
        self.declare_parameter("vfh_recovery_wide_valley_min", 4)
        self.declare_parameter("vfh_recovery_max_range", 4.0)
        self.declare_parameter("vfh_recovery_min_open_range", 0.15)
        self.declare_parameter("vfh_recovery_max_candidates", 5)
        self.declare_parameter("vfh_recovery_min_turn_deg", 8.0)
        self.declare_parameter("vfh_recovery_retry_turn_deg", 30.0)
        self.declare_parameter("vfh_recovery_straight_deadband_deg", 12.0)
        self.declare_parameter("enable_recovery_path_avoidance", True)
        self.declare_parameter("recovery_path_avoidance_angle_deg", 90.0)
        self.declare_parameter("recovery_path_lookback_sec", 12.0)
        self.declare_parameter("recovery_path_min_dist", 0.25)
        self.declare_parameter("recovery_path_max_dist", 2.5)
        self.declare_parameter("recovery_heading_checked_window_deg", 22.0)
        self.declare_parameter("recovery_checked_max_count", 80)
        self.declare_parameter("vfh_recovery_goal_bias", 0.15)
        self.declare_parameter("vfh_recovery_front_bias", 0.05)
        self.declare_parameter("vfh_recovery_memory_weight", 0.20)

        # extra cost weights
        self.declare_parameter("w_recovery_avoid", 4.0)
        self.declare_parameter("recovery_avoid_radius", 1.5)
        self.declare_parameter("recovery_point_memory_sec", 120.0)
        self.declare_parameter("w_away_from_start", 2.0)

        # final command smoothing
        self.declare_parameter("cmd_v_alpha", 0.45)
        self.declare_parameter("cmd_w_alpha", 0.28)

        # parameterized cost weights / thresholds / geometry / runtime
        self.declare_parameter("w_progress", 14.0)
        self.declare_parameter("w_heading", 5.0)
        self.declare_parameter("w_curvature", 0.0)
        self.declare_parameter("w_length", 0.0)
        self.declare_parameter("w_speed", 0.0)
        self.declare_parameter("goal_factor", 1.0)
        self.declare_parameter("hard_clearance_penalty_dist", 0.08)
        self.declare_parameter("hard_clearance_penalty_value", 0.0)
        self.declare_parameter("center_balance_clip", 0.25)
        self.declare_parameter("tube_obstacle_proximity_dist", 0.45)
        self.declare_parameter("w_tube_obstacle_proximity", 8.0)

        self.declare_parameter("tube_prefilter_forward_dist", 0.65)
        self.declare_parameter("tube_prefilter_low_w", 0.18)
        self.declare_parameter("tube_prefilter_margin", 0.20)
        self.declare_parameter("sweep_sample_dist", 0.12)
        self.declare_parameter("sweep_aug_dist", 0.06)
        self.declare_parameter("sweep_extra_margin", 0.03)

        self.declare_parameter("cmd_min_clearance_slowdown_dist", 0.12)
        self.declare_parameter("cmd_min_clearance_slowdown_scale", 0.75)
        self.declare_parameter("cmd_high_turn_thresh", 0.8)
        self.declare_parameter("cmd_high_turn_scale", 0.85)

        self.declare_parameter("robot_length", 0.45)
        self.declare_parameter("robot_width", 0.40)
        self.declare_parameter("diag_period_sec", 3.0)
        self.declare_parameter("execute_sleep_sec", 0.10)
        self.declare_parameter("executor_threads", 4)

        # self.declare_parameter("group1_w_min", 0.0)
        # self.declare_parameter("group1_w_max", 0.2)
        # self.declare_parameter("group1_T", [1.0])


        # self.declare_parameter("group2_w_min", 0.3)
        # self.declare_parameter("group2_w_max", 1.2)
        # self.declare_parameter("group2_T", [1.0])

        # self.declare_parameter("group3_w_min", 1.2)
        # self.declare_parameter("group3_w_max", 1.6)
        # self.declare_parameter("group3_T", [1.0])

        # self.declare_parameter("group4_w_min", 1.6)
        # self.declare_parameter("group4_w_max", 2.0)
        # self.declare_parameter("group4_T", [1.0])

################################################################
        self.declare_parameter("group1_w_min", 0.0)
        self.declare_parameter("group1_w_max", 0.2)
        self.declare_parameter("group1_T", [7.0])

        self.declare_parameter("group2_w_min", 0.0)
        self.declare_parameter("group2_w_max", 1.0)
        self.declare_parameter("group2_T", [3.0])

        self.declare_parameter("group3_w_min", 0.2)
        self.declare_parameter("group3_w_max", 1.0)
        self.declare_parameter("group3_T", [2.0])

        self.declare_parameter("group4_w_min", 0.2)
        self.declare_parameter("group4_w_max", 1.0)
        self.declare_parameter("group4_T", [0.5, 1.0])

        # self.declare_parameter("group1_w_min", 0.0)
        # self.declare_parameter("group1_w_max", 0.1)
        # self.declare_parameter("group1_T", [7.0, 10.0])

        # self.declare_parameter("group2_w_min", 0.3)
        # self.declare_parameter("group2_w_max", 1.2)
        # self.declare_parameter("group2_T", [3.0])

        # self.declare_parameter("group3_w_min", 0.0)
        # self.declare_parameter("group3_w_max", 0.2)
        # self.declare_parameter("group3_T", [3.0])

        # self.declare_parameter("group4_w_min", 1.3)
        # self.declare_parameter("group4_w_max", 2.0)
        # self.declare_parameter("group4_T", [1.5])

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
        self.enable_path_trace_marker = bool(gp("enable_path_trace_marker").value)
        self.path_trace_memory_sec = float(gp("path_trace_memory_sec").value)
        self.path_trace_min_dist = float(gp("path_trace_min_dist").value)
        self.path_trace_max_points = int(gp("path_trace_max_points").value)

        self.w_hold_time = float(gp("w_hold_time").value)
        self.turn_commit_time = float(gp("turn_commit_time").value)
        self.opposite_turn_penalty = float(gp("opposite_turn_penalty").value)
        self.recent_pos_memory_sec = float(gp("recent_pos_memory_sec").value)
        self.revisit_radius = float(gp("revisit_radius").value)
        self.revisit_penalty_weight = float(gp("revisit_penalty_weight").value)

        self.goal_start_grace_sec = float(gp("goal_start_grace_sec").value)
        self.no_feasible_confirm_sec = float(gp("no_feasible_confirm_sec").value)

        self.enable_active_group = bool(gp("enable_active_group").value)
        self.group_hold_time = float(gp("group_hold_time").value)
        self.g1_min_open_forward = float(gp("g1_min_open_forward").value)
        self.g1_min_arc_len = float(gp("g1_min_arc_len").value)
        self.g1_vs_g2_progress_margin = float(gp("g1_vs_g2_progress_margin").value)
        self.g2_return_to_g1_open_forward = float(gp("g2_return_to_g1_open_forward").value)
        self.enable_early_turn = bool(gp("enable_early_turn").value)
        self.early_turn_forward_dist = float(gp("early_turn_forward_dist").value)
        self.early_turn_min_abs_w = float(gp("early_turn_min_abs_w").value)
        self.early_turn_w_bias = float(gp("early_turn_w_bias").value)
        self.enable_turn_group_w_bias = bool(gp("enable_turn_group_w_bias").value)
        self.turn_group_w_bias = float(gp("turn_group_w_bias").value)
        self.enable_group_long_tube_bias = bool(gp("enable_group_long_tube_bias").value)
        self.group_long_tube_bias = float(gp("group_long_tube_bias").value)
        self.enable_green_center_selection = bool(gp("enable_green_center_selection").value)
        self.green_cost_ratio = float(gp("green_cost_ratio").value)
        self.green_center_min_candidates = int(gp("green_center_min_candidates").value)

        self.trap_radius = float(gp("trap_radius").value)
        self.trap_memory_sec = float(gp("trap_memory_sec").value)
        self.trap_trigger_g2_count = int(gp("trap_trigger_g2_count").value)
        self.trap_trigger_low_progress_count = int(gp("trap_trigger_low_progress_count").value)
        self.trap_progress_eps = float(gp("trap_progress_eps").value)
        self.trap_g1_suppress_sec = float(gp("trap_g1_suppress_sec").value)

        self.enable_recovery = bool(gp("enable_recovery").value)
        self.recovery_use_backup = bool(gp("recovery_use_backup").value)
        self.recovery_backup_dist = float(gp("recovery_backup_dist").value)
        self.recovery_backup_v = float(gp("recovery_backup_v").value)
        self.recovery_backup_timeout = float(gp("recovery_backup_timeout").value)
        self.recovery_backup_min_time = float(gp("recovery_backup_min_time").value)
        self.recovery_scan_step_deg = float(gp("recovery_scan_step_deg").value)
        self.recovery_scan_max_deg = float(gp("recovery_scan_max_deg").value)
        self.recovery_yaw_tol_deg = float(gp("recovery_yaw_tol_deg").value)
        self.recovery_w = float(gp("recovery_w").value)
        self.recovery_min_pause = float(gp("recovery_min_pause").value)

        self.enable_vfh_recovery = bool(gp("enable_vfh_recovery").value)
        self.vfh_recovery_sector_count = int(gp("vfh_recovery_sector_count").value)
        self.vfh_recovery_smooth_width = int(gp("vfh_recovery_smooth_width").value)
        self.vfh_recovery_threshold = float(gp("vfh_recovery_threshold").value)
        self.vfh_recovery_min_valley_width = int(gp("vfh_recovery_min_valley_width").value)
        self.vfh_recovery_wide_valley_min = int(gp("vfh_recovery_wide_valley_min").value)
        self.vfh_recovery_max_range = float(gp("vfh_recovery_max_range").value)
        self.vfh_recovery_min_open_range = float(gp("vfh_recovery_min_open_range").value)
        self.vfh_recovery_max_candidates = int(gp("vfh_recovery_max_candidates").value)
        self.vfh_recovery_min_turn_deg = float(gp("vfh_recovery_min_turn_deg").value)
        self.vfh_recovery_retry_turn_deg = float(gp("vfh_recovery_retry_turn_deg").value)
        self.vfh_recovery_straight_deadband_deg = float(gp("vfh_recovery_straight_deadband_deg").value)
        self.enable_recovery_path_avoidance = bool(gp("enable_recovery_path_avoidance").value)
        self.recovery_path_avoidance_angle_deg = float(gp("recovery_path_avoidance_angle_deg").value)
        self.recovery_path_lookback_sec = float(gp("recovery_path_lookback_sec").value)
        self.recovery_path_min_dist = float(gp("recovery_path_min_dist").value)
        self.recovery_path_max_dist = float(gp("recovery_path_max_dist").value)
        self.recovery_heading_checked_window_deg = float(gp("recovery_heading_checked_window_deg").value)
        self.recovery_checked_max_count = int(gp("recovery_checked_max_count").value)
        self.vfh_recovery_goal_bias = float(gp("vfh_recovery_goal_bias").value)
        self.vfh_recovery_front_bias = float(gp("vfh_recovery_front_bias").value)
        self.vfh_recovery_memory_weight = float(gp("vfh_recovery_memory_weight").value)

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
        self.tube_obstacle_proximity_dist = float(gp("tube_obstacle_proximity_dist").value)
        self.w_tube_obstacle_proximity = float(gp("w_tube_obstacle_proximity").value)

        self.tube_prefilter_forward_dist = float(gp("tube_prefilter_forward_dist").value)
        self.tube_prefilter_low_w = float(gp("tube_prefilter_low_w").value)
        self.tube_prefilter_margin = float(gp("tube_prefilter_margin").value)
        self.sweep_sample_dist = float(gp("sweep_sample_dist").value)
        self.sweep_aug_dist = float(gp("sweep_aug_dist").value)
        self.sweep_extra_margin = float(gp("sweep_extra_margin").value)

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

        inc = msg.angle_increment if abs(msg.angle_increment) > 1e-9 else 1e-3
        new_sensor_config = {
            "min_angle": msg.angle_min,
            "max_angle": msg.angle_max,
            "angle_increment": inc,
            "range_min": msg.range_min,
            "range_max": msg.range_max,
            "num_beams": len(msg.ranges),
        }
        sensor_changed = any(
            self.sensor_config.get(k) != new_sensor_config[k]
            for k in new_sensor_config.keys()
        )

        if self.sensor_config["num_beams"] == 0 or sensor_changed:
            self.sensor_config.update(new_sensor_config)
            self._invalidate_precomputed_tubes()
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

        self._ensure_precomputed_tubes()

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
            self.no_feasible_since = None
            self.goal_start_time = self._now_sec()
            self.goal_start_xy = self._current_xy()

            self.progress_history = []
            self.path_trace = []
            self.stuck_counter = 0

            self.locked_w_value = None
            self.w_lock_until = 0.0
            self.committed_turn_sign = 0
            self.turn_commit_until = 0.0

            self.last_cmd_v = 0.0
            self.last_cmd_w = 0.0

            self.active_group = None
            self.active_group_until = 0.0

            self.recovery_state = "NONE"
            self.recovery_enter_t = 0.0
            self.recovery_backup_start_xy = None
            self.recovery_backup_start_t = 0.0
            self.recovery_current_offset_deg = 0.0
            self.recovery_last_switch_t = 0.0
            self.recovery_scan_sequence_deg = []
            self.recovery_scan_index = 0
            self.vfh_recovery_last_heading = None
            self.vfh_recovery_last_turn_sign = 1.0
            self.vfh_recovery_force_explore = False
            self.recovery_checked_yaws = []
            self.recovery_path_deferred_allowed = False

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
                self.no_feasible_since = None
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
                if self.no_feasible_since is None:
                    self.no_feasible_since = now
                no_feasible_elapsed = now - self.no_feasible_since
                self.selected_tube = None
                self._publish_zero()
                self._publish_tube_markers()

                if not self.enable_recovery:
                    return

                if no_feasible_elapsed < self.no_feasible_confirm_sec:
                    self._log_throttle(
                        "info",
                        "no_feasible_debounce",
                        0.3,
                        "No feasible tubes: waiting %.2f/%.2fs before recovery"
                        % (no_feasible_elapsed, self.no_feasible_confirm_sec),
                    )
                    return

                self._log_throttle(
                    "warn",
                    "no_feasible_confirmed",
                    0.5,
                    "No feasible tubes confirmed for %.2fs, entering recovery"
                    % no_feasible_elapsed,
                )
                self._enter_recovery(now)
                self._recovery_step(now)
                return

            self.no_feasible_since = None
            self.select_best_tube()
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

    def sample_to_beam_index(self, sample):
        if self.sensor_config["num_beams"] == 0:
            return False, -1, -1.0, 0.0

        sx = float(sample[0])
        sy = float(sample[1])
        cy = math.cos(self.base_to_laser_yaw)
        syaw = math.sin(self.base_to_laser_yaw)
        x_laser = sx * cy - sy * syaw
        y_laser = sx * syaw + sy * cy

        distance = math.hypot(x_laser, y_laser)
        angle = math.atan2(y_laser, x_laser)

        range_min = float(self.sensor_config["range_min"])
        range_max = float(self.sensor_config["range_max"])
        if distance < range_min:
            return False, -1, distance, angle
        if distance > range_max:
            return False, -1, distance, angle

        ang_min = float(self.sensor_config["min_angle"])
        ang_max = float(self.sensor_config["max_angle"])
        lo = min(ang_min, ang_max)
        hi = max(ang_min, ang_max)
        if angle < lo or angle > hi:
            return False, -1, distance, angle

        inc = float(self.sensor_config["angle_increment"])
        n = int(self.sensor_config["num_beams"])
        idx = int((angle - ang_min) / inc)
        if idx < 0 or idx >= n:
            return False, -1, distance, angle

        return True, idx, distance, angle

    def _invalidate_precomputed_tubes(self):
        self.precomputed_tubes = []
        self.tubes_precomputed = False
        self._tube_precompute_signature = None

    def _current_tube_precompute_signature(self):
        tube_cfg_sig = tuple(
            (float(cfg["v"]), float(cfg["w"]), float(cfg["T"]), str(cfg["group_name"]))
            for cfg in self._tube_configs
        )
        sensor_sig = (
            float(self.sensor_config["min_angle"]),
            float(self.sensor_config["max_angle"]),
            float(self.sensor_config["angle_increment"]),
            float(self.sensor_config["range_min"]),
            float(self.sensor_config["range_max"]),
            int(self.sensor_config["num_beams"]),
        )
        geom_sig = (
            float(self.robot.half_width),
            float(self.robot.half_length),
        )
        sweep_sig = (
            float(self.sweep_sample_dist),
            float(self.sweep_aug_dist),
            float(self.sweep_extra_margin),
            float(self.base_to_laser_yaw),
        )
        return (tube_cfg_sig, sensor_sig, geom_sig, sweep_sig)

    def _ensure_precomputed_tubes(self):
        if self.sensor_config["num_beams"] == 0:
            return

        sig = self._current_tube_precompute_signature()
        if self.tubes_precomputed and self._tube_precompute_signature == sig:
            return

        self.precomputed_tubes = []
        for cfg in self._tube_configs:
            tube = self.create_motion_tube(cfg["v"], cfg["w"], cfg["T"], cfg["group_name"])
            if tube is not None:
                self.precomputed_tubes.append(tube)

        total = len(self.precomputed_tubes)
        evaluable = sum(1 for t in self.precomputed_tubes if getattr(t, "evaluable", True))
        invalid = total - evaluable
        self.get_logger().info(
            "Precomputed tubes: total=%d evaluable=%d invalid=%d"
            % (total, evaluable, invalid)
        )

        self._tube_precompute_signature = sig
        self.tubes_precomputed = True

    def _make_runtime_tube_copy(self, template_tube):
        tube = MotionTube(
            template_tube.template_type,
            template_tube.v,
            template_tube.w,
            template_tube.T,
            template_tube.samples,
            template_tube.beam_indices,
            template_tube.arc_len,
            template_tube.group_name,
            centerline_samples=template_tube.centerline_samples,
        )
        tube.sample_distances = template_tube.sample_distances
        tube.evaluable = template_tube.evaluable
        tube.cost = float("inf")
        tube.is_feasible = False
        tube.goal_progress = 0.0
        tube.min_clearance = float("inf")
        tube.heading_after = 0.0
        tube.obstacle_penalty = 0.0
        tube.left_clearance = float("inf")
        tube.right_clearance = float("inf")
        tube.center_balance = 0.0
        return tube

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
        self._update_path_trace(now, x, y)

    def _update_path_trace(self, now, x, y):
        min_t = now - max(0.1, self.path_trace_memory_sec)
        self.path_trace = [p for p in self.path_trace if p[0] >= min_t]

        if self.path_trace:
            _, px, py = self.path_trace[-1]
            if math.hypot(x - px, y - py) < self.path_trace_min_dist:
                return

        self.path_trace.append((now, float(x), float(y)))

        max_points = max(2, int(self.path_trace_max_points))
        if len(self.path_trace) > max_points:
            self.path_trace = self.path_trace[-max_points:]

    def generate_motion_tubes(self):
        self.motion_tubes = []
        fwd = self._fwd_cache
        self._ensure_precomputed_tubes()

        for template_tube in self.precomputed_tubes:
            v = template_tube.v
            w = template_tube.w
            T = template_tube.T

            approx_len = abs(v) * T
            if fwd < self.tube_prefilter_forward_dist and abs(w) < 1e-9 and approx_len > (fwd + self.tube_prefilter_margin):
                continue

            self.motion_tubes.append(self._make_runtime_tube_copy(template_tube))

    def create_motion_tube(self, v, w, T, group_name):
        if abs(w) < 0.05:
            ttype = TemplateType.MOVE_STRAIGHT
        elif w > 0:
            ttype = TemplateType.STEER_LEFT
        else:
            ttype = TemplateType.STEER_RIGHT

        samples, beam_indices = [], []
        sample_distances = []
        centerline_samples = []
        arc_len = abs(v) * T
        sample_dist = max(1e-3, self.sweep_sample_dist)
        n_samples = max(1, int(math.ceil(max(arc_len, sample_dist) / sample_dist)))
        offset = self.sweep_aug_dist + self.sweep_extra_margin
        sweep_half_width = self.robot.half_width + offset

        def center_pose(t):
            if abs(w) < 1e-3:
                return v * t, 0.0
            theta = w * t
            R = v / w
            return R * math.sin(theta), R * (1.0 - math.cos(theta))

        def rigid_body_point(x0, y0, t):
            if abs(w) < 1e-3:
                return x0 + v * t, y0
            theta = w * t
            R = v / w
            x = x0 * math.cos(theta) - y0 * math.sin(theta) + R * math.sin(theta)
            y = x0 * math.sin(theta) + y0 * math.cos(theta) + R * (1.0 - math.cos(theta))
            return x, y

        def sample_point_trajectory(x0, y0):
            if abs(w) < 1e-3:
                p0 = np.array([x0, y0], dtype=np.float32)
                p1 = np.array([x0 + v * T, y0], dtype=np.float32)
                return sample_segment(p0, p1)

            rp = math.sqrt(x0 * x0 + (v / w - y0) * (v / w - y0))
            denom = max(1e-6, abs(w) * max(rp, 1e-6))
            dt = sample_dist / denom
            n = max(2, int(math.ceil(T / max(dt, 1e-6))) + 1)
            dt = T / float(n - 1)

            pts = []
            for i in range(n):
                t = dt * i
                x, y = rigid_body_point(x0, y0, t)
                pts.append(np.array([x, y], dtype=np.float32))
            return pts

        def sample_segment(p0, p1):
            p0 = np.array(p0, dtype=np.float32)
            p1 = np.array(p1, dtype=np.float32)
            seg_len = float(np.linalg.norm(p1 - p0))
            n = max(2, int(math.ceil(seg_len / sample_dist)) + 1)

            pts = []
            for a in np.linspace(0.0, 1.0, num=n):
                pts.append((1.0 - float(a)) * p0 + float(a) * p1)
            return pts

        for i in range(1, n_samples + 1):
            t = T * i / float(n_samples)
            x, y = center_pose(t)
            centerline_samples.append(np.array([x, y], dtype=np.float32))

        sweep_ts = np.linspace(0.0, T, num=n_samples + 1)

        if ttype == TemplateType.MOVE_STRAIGHT:
            final_x = v * T
            left_y = sweep_half_width
            right_y = -sweep_half_width

            for t in sweep_ts:
                x = v * t
                samples.append(np.array([x, left_y], dtype=np.float32))
                samples.append(np.array([x, right_y], dtype=np.float32))

            cap_count = max(2, int(math.ceil((2.0 * sweep_half_width) / sample_dist)) + 1)
            for y in np.linspace(right_y, left_y, num=cap_count):
                samples.append(np.array([final_x, float(y)], dtype=np.float32))
        elif ttype == TemplateType.STEER_LEFT:
            outer_x0 = self.robot.half_length + offset
            outer_y0 = -self.robot.half_width - offset
            inner_x0 = 0.0
            inner_y0 = self.robot.half_width + offset

            front_left_x0 = self.robot.half_length + offset
            front_left_y0 = self.robot.half_width + offset

            samples.extend(sample_point_trajectory(outer_x0, outer_y0))
            samples.extend(sample_point_trajectory(inner_x0, inner_y0))

            front_right_final = np.array(rigid_body_point(outer_x0, outer_y0, T), dtype=np.float32)
            front_left_final = np.array(rigid_body_point(front_left_x0, front_left_y0, T), dtype=np.float32)
            axle_left_final = np.array(rigid_body_point(inner_x0, inner_y0, T), dtype=np.float32)

            samples.extend(sample_segment(front_right_final, front_left_final))
            samples.extend(sample_segment(front_left_final, axle_left_final))
        else:
            outer_x0 = self.robot.half_length + offset
            outer_y0 = self.robot.half_width + offset
            inner_x0 = 0.0
            inner_y0 = -self.robot.half_width - offset

            front_right_x0 = self.robot.half_length + offset
            front_right_y0 = -self.robot.half_width - offset

            samples.extend(sample_point_trajectory(outer_x0, outer_y0))
            samples.extend(sample_point_trajectory(inner_x0, inner_y0))

            front_left_final = np.array(rigid_body_point(outer_x0, outer_y0, T), dtype=np.float32)
            front_right_final = np.array(rigid_body_point(front_right_x0, front_right_y0, T), dtype=np.float32)
            axle_right_final = np.array(rigid_body_point(inner_x0, inner_y0, T), dtype=np.float32)

            samples.extend(sample_segment(front_left_final, front_right_final))
            samples.extend(sample_segment(front_right_final, axle_right_final))

        for s in samples:
            valid, idx, dist, _ = self.sample_to_beam_index(s)
            if dist < 0.0:
                dist = float(np.linalg.norm(s))
            if not valid:
                beam_indices.append(-1)
            else:
                beam_indices.append(idx)
            sample_distances.append(dist)

        tube = MotionTube(ttype, v, w, T, samples, beam_indices, arc_len, group_name, centerline_samples=centerline_samples)
        tube.sample_distances = sample_distances
        tube.evaluable = True
        return tube

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

        idxs = np.linspace(0, len(tube.samples) - 1, num=min(24, len(tube.samples)), dtype=int)

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
                clr = r - dist 

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

    def collision_check_sweeping(self, tube):
        if self._ranges_clipped is None or not getattr(tube, "evaluable", True):
            tube.left_clearance = -1.0
            tube.right_clearance = -1.0
            tube.center_balance = 1.0
            return False, -1.0, 0.0

        min_clr = float("inf")
        left_min = float("inf")
        right_min = float("inf")
        sample_distances = getattr(tube, "sample_distances", None)

        for k, (s, idx) in enumerate(zip(tube.samples, tube.beam_indices)):
            if idx < 0 or idx >= len(self._ranges_clipped):
                continue

            if sample_distances is not None and k < len(sample_distances):
                dist = float(sample_distances[k])
            else:
                dist = float(np.linalg.norm(s))

            if dist < 1e-3:
                continue

            r = float(self._ranges_clipped[idx])
            clr = r - dist
            min_clr = min(min_clr, clr)

            if s[1] >= 0.0:
                left_min = min(left_min, clr)
            else:
                right_min = min(right_min, clr)

            if clr < 0.0:
                tube.left_clearance = left_min if np.isfinite(left_min) else -1.0
                tube.right_clearance = right_min if np.isfinite(right_min) else -1.0
                if np.isfinite(left_min) and np.isfinite(right_min):
                    tube.center_balance = abs(left_min - right_min)
                else:
                    tube.center_balance = 1.0
                return False, clr, 1.0

        if not np.isfinite(min_clr):
            min_clr = -1.0
        if not np.isfinite(left_min):
            left_min = min_clr
        if not np.isfinite(right_min):
            right_min = min_clr

        tube.left_clearance = left_min
        tube.right_clearance = right_min
        tube.center_balance = abs(left_min - right_min) if np.isfinite(left_min) and np.isfinite(right_min) else 0.0

        return True, min_clr, 0.0

    def _tube_end_world(self, tube):
        cx, cy, yaw = self._pose_xy_yaw()
        centerline = [] if tube is None else tube.centerline_samples
        if tube is None or len(centerline) == 0:
            return (cx, cy)
        end_local = centerline[-1]
        sx, sy = float(end_local[0]), float(end_local[1])
        ex = cx + sx * math.cos(yaw) - sy * math.sin(yaw)
        ey = cy + sx * math.sin(yaw) + sy * math.cos(yaw)
        return (ex, ey)

    def goal_progress_along_tube(self, tube):
        cx, cy, yaw = self._pose_xy_yaw()
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        now_dist = math.hypot(gx - cx, gy - cy)

        if len(tube.centerline_samples) == 0:
            return 0.0

        idxs = np.linspace(0, len(tube.centerline_samples) - 1, num=min(5, len(tube.centerline_samples)), dtype=int)
        improvements = []

        for k in idxs:
            s = tube.centerline_samples[k]
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

    def composite_cost(self, t, goal_bearing, goal_factor=1.0):
        now = self._now_sec()

        c = 0.0
        c -= goal_factor * self.w_progress * t.goal_progress

        heading_err = abs(self._wrap(t.heading_after - goal_bearing))
        c += goal_factor * self.w_heading * heading_err

        if t.min_clearance < self.clearance_safe_dist:
            clearance_ratio = max(0.0, 1.0 - t.min_clearance / max(1e-6, self.clearance_safe_dist))
            c += self.w_clearance * clearance_ratio

        side_min = min(t.left_clearance, t.right_clearance)
        if side_min < self.side_clearance_safe_dist:
            side_ratio = max(0.0, 1.0 - side_min / max(1e-6, self.side_clearance_safe_dist))
            c += self.w_side_clearance * side_ratio

        tube_min_clr = min(t.min_clearance, t.left_clearance, t.right_clearance)
        if tube_min_clr < self.tube_obstacle_proximity_dist:
            prox_ratio = max(
                0.0,
                1.0 - tube_min_clr / max(1e-6, self.tube_obstacle_proximity_dist)
            )
            c += self.w_tube_obstacle_proximity * (prox_ratio ** 2)

        c += self.w_center_balance * min(self.center_balance_clip, t.center_balance)

        c += self.recovery_avoid_penalty(t, now)

        if self.enable_turn_group_w_bias and t.group_name in ("G2_mid_w_turn", "G4_high_w_shortT"):
            turn_strength = min(1.0, abs(t.w) / max(1e-6, self.max_w))
            c -= self.turn_group_w_bias * turn_strength

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
            t.is_feasible, t.min_clearance, t.obstacle_penalty = self.collision_check_sweeping(t)
            if not t.is_feasible:
                continue

            t.heading_after = yaw + t.w * t.T
            t.goal_progress = self.goal_progress_along_tube(t)
            t.cost = self.composite_cost(t, goal_bearing, goal_factor)

            tube_sign = self._turn_sign(t.w)
            if now < self.turn_commit_until and self.committed_turn_sign != 0:
                if tube_sign != 0 and tube_sign != self.committed_turn_sign:
                    t.cost += self.opposite_turn_penalty

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

    def _cleanup_recovery_points(self, now):
        kept = []
        for item in self.recovery_points:
            if (now - item["t"]) <= self.recovery_point_memory_sec:
                kept.append(item)
        self.recovery_points = kept

    def _choose_desired_group(self, feas, now):
        g1 = self._best_in_group("G1_low_w_longT", feas)
        g2 = self._best_in_group("G2_mid_w_turn", feas)
        g3 = self._best_in_group("G3_low_w_midT", feas)
        g4 = self._best_in_group("G4_high_w_shortT", feas)

        suppress_g1 = False

        g1_good = False
        if g1 is not None and not suppress_g1:
            g1_good = (
                self._fwd_cache >= self.g1_min_open_forward
                and g1.arc_len >= self.g1_min_arc_len
            )
            if g2 is not None:
                g1_good = g1_good and (g1.goal_progress >= g2.goal_progress + self.g1_vs_g2_progress_margin)

        if self.enable_active_group and self.active_group is not None and now < self.active_group_until:
            active_best = self._best_in_group(self.active_group, feas)
            if active_best is not None:
                return self.active_group

        if self.enable_active_group and self.active_group == "G2_mid_w_turn":
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

    def _tube_lateral_value(self, tube):
        samples = getattr(tube, "centerline_samples", [])
        if samples:
            return float(samples[-1][1])
        return float(tube.w * tube.T)

    def _tube_side_margin(self, tube):
        vals = []
        for v in (tube.left_clearance, tube.right_clearance, tube.min_clearance):
            if np.isfinite(v):
                vals.append(float(v))
        if not vals:
            return -1.0
        return min(vals)

    def _green_ratio(self, tube):
        span = float(self.max_cost - self.min_cost)
        if span <= 1e-6:
            return 0.0
        return max(0.0, min(1.0, (float(tube.cost) - float(self.min_cost)) / span))

    def _green_reference_turn_sign(self, best_by_rank):
        sign = self._turn_sign(self.last_cmd_w)
        if sign == 0 and self.committed_turn_sign != 0 and self._now_sec() < self.turn_commit_until:
            sign = self.committed_turn_sign
        if sign == 0 and self.selected_tube is not None:
            sign = self._turn_sign(self.selected_tube.w)
        if sign == 0 and best_by_rank is not None:
            sign = self._turn_sign(best_by_rank.w)
        return sign

    def _select_green_opposite_tube(self, candidates, rank_fn=None, reason="normal"):
        raw_candidates = [t for t in candidates if t is not None and t.is_feasible]
        candidates = [t for t in raw_candidates if np.isfinite(t.cost)]
        if not candidates:
            if raw_candidates:
                return min(raw_candidates, key=lambda z: z.cost)
            return None

        if rank_fn is None:
            rank_fn = self._candidate_key_in_group

        best_by_rank = min(candidates, key=rank_fn)
        if not self.enable_green_center_selection:
            return best_by_rank

        ratio_limit = max(0.0, min(1.0, float(self.green_cost_ratio)))
        min_candidates = max(1, int(self.green_center_min_candidates))
        green = [t for t in candidates if self._green_ratio(t) <= ratio_limit]

        if len(green) < min_candidates:
            local_min = min(float(t.cost) for t in candidates)
            local_max = max(float(t.cost) for t in candidates)
            local_span = max(1e-6, local_max - local_min)
            green = [
                t for t in candidates
                if ((float(t.cost) - local_min) / local_span) <= ratio_limit
            ]

        if len(green) < min_candidates:
            return best_by_rank

        reference_sign = self._green_reference_turn_sign(best_by_rank)
        if reference_sign == 0:
            return best_by_rank

        def opposite_key(tube):
            lateral = self._tube_lateral_value(tube)
            edge_key = lateral if reference_sign > 0 else -lateral
            return (
                edge_key,
                -self._tube_side_margin(tube),
                min(self.center_balance_clip, tube.center_balance),
                rank_fn(tube),
            )

        candidate = min(green, key=opposite_key)
        target_side = "right" if reference_sign > 0 else "left"
        self._log_throttle(
            "info",
            "green_opposite_selection",
            0.5,
            "Green-opposite selection(%s): turn_sign=%d target=%s green=%d/%d ratio<=%.2f selected g=%s w=%.2f T=%.1f y=%.2f cost=%.2f side=%.2f"
            % (
                reason,
                reference_sign,
                target_side,
                len(green),
                len(candidates),
                ratio_limit,
                candidate.group_name,
                candidate.w,
                candidate.T,
                self._tube_lateral_value(candidate),
                candidate.cost,
                self._tube_side_margin(candidate),
            )
        )
        return candidate

    def _should_force_early_turn(self, feas):
        if not self.enable_early_turn:
            return False

        if self._fwd_cache >= self.early_turn_forward_dist:
            return False

        turning_feas = [t for t in feas if abs(t.w) >= self.early_turn_min_abs_w]
        if not turning_feas:
            return False

        g1_all = [t for t in self.motion_tubes if t.group_name == "G1_low_w_longT"]
        if not g1_all:
            return True

        longest_detect = max(g1_all, key=lambda z: z.arc_len)
        if not longest_detect.is_feasible:
            return True

        if abs(longest_detect.w) < self.early_turn_min_abs_w and longest_detect.arc_len > (self._fwd_cache + self.tube_prefilter_margin):
            return True

        return False

    def _pick_early_turn_candidate(self, feas):
        turning_feas = [t for t in feas if abs(t.w) >= self.early_turn_min_abs_w]
        if not turning_feas:
            return None

        def early_turn_key(tube):
            if tube.group_name == "G2_mid_w_turn":
                group_pref = 0
            elif tube.group_name == "G4_high_w_shortT":
                group_pref = 1
            else:
                group_pref = 2
            return (
                group_pref,
                self._candidate_key_in_group(tube) - self.early_turn_w_bias * abs(tube.w),
                -abs(tube.w),
            )

        return self._select_green_opposite_tube(
            turning_feas,
            rank_fn=early_turn_key,
            reason="early_turn",
        )

    def _commit_selected_tube(self, candidate, now):
        if candidate is None:
            self.selected_tube = None
            return

        prev_w = None if self.selected_tube is None else self.selected_tube.w
        prev_sign = 0 if self.selected_tube is None else self._turn_sign(self.selected_tube.w)
        new_sign = self._turn_sign(candidate.w)

        if self.enable_active_group and self.active_group != candidate.group_name:
            self.active_group = candidate.group_name
            self.active_group_until = now + self.group_hold_time

        if prev_w is None or not self._same_w(prev_w, candidate.w):
            self.locked_w_value = candidate.w
            self.w_lock_until = now + self.w_hold_time

        if new_sign != 0 and new_sign != prev_sign:
            self.committed_turn_sign = new_sign
            self.turn_commit_until = now + self.turn_commit_time

        self.selected_tube = candidate

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

        if self._should_force_early_turn(feas):
            candidate = self._pick_early_turn_candidate(feas)
            if candidate is not None:
                self._log_throttle(
                    "warn",
                    "early_turn_override",
                    0.5,
                    "EARLY TURN override: fwd=%.2f select group=%s w=%.2f T=%.2f"
                    % (self._fwd_cache, candidate.group_name, candidate.w, candidate.T),
                )
                self._commit_selected_tube(candidate, now)
                return

        desired_group = self._choose_desired_group(feas, now)
        if desired_group is None:
            candidate = self._select_green_opposite_tube(
                feas,
                rank_fn=lambda z: z.cost,
                reason="all_feasible",
            )
            self._commit_selected_tube(candidate, now)
            return

        if not self.enable_active_group:
            group_feas = [t for t in feas if t.group_name == desired_group]
            if group_feas:
                candidate = self._select_green_opposite_tube(
                    group_feas,
                    rank_fn=self._candidate_key_in_group,
                    reason=desired_group,
                )
            else:
                candidate = self._select_green_opposite_tube(
                    feas,
                    rank_fn=lambda z: (self._group_rank(z.group_name), z.cost),
                    reason="fallback_no_active_group",
                )
            self._commit_selected_tube(candidate, now)
            return

        if self.active_group != desired_group:
            self.active_group = desired_group
            self.active_group_until = now + self.group_hold_time

        group_feas = [t for t in feas if t.group_name == self.active_group]
        if not group_feas:
            candidate = self._select_green_opposite_tube(
                feas,
                rank_fn=lambda z: (self._group_rank(z.group_name), z.cost),
                reason="fallback_active_group",
            )
            self._commit_selected_tube(candidate, now)
            return

        if self.locked_w_value is not None and now < self.w_lock_until:
            same_w = [t for t in group_feas if self._same_w(t.w, self.locked_w_value)]
            if same_w:
                candidate = self._select_green_opposite_tube(
                    same_w,
                    rank_fn=self._candidate_key_in_group,
                    reason="locked_w",
                )
                self.selected_tube = candidate
                return
            else:
                self.locked_w_value = None
                self.w_lock_until = 0.0

        candidate = self._select_green_opposite_tube(
            group_feas,
            rank_fn=self._candidate_key_in_group,
            reason=self.active_group,
        )
        self._commit_selected_tube(candidate, now)

    # =========================
    # Recovery
    # =========================
    def _build_fixed_recovery_scan_sequence(self):
        seq = [+30, +60, -60, +90, +120, +150]
        max_deg = float(self.recovery_scan_max_deg)
        return [x for x in seq if abs(x) <= max_deg]

    def _goal_relative_angle(self):
        if self.current_goal is None or self.current_pose is None:
            return 0.0
        x, y, yaw = self._pose_xy_yaw()
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        goal_bearing = math.atan2(gy - y, gx - x)
        return self._wrap(goal_bearing - yaw)

    def _vfh_sector_to_base_angle(self, sector_idx, sector_count):
        n = max(1, int(self.sensor_config["num_beams"]))
        sector_count = max(1, int(sector_count))
        beam_pos = (float(sector_idx) + 0.5) * float(n) / float(sector_count)
        beam_idx = max(0, min(n - 1, int(beam_pos)))
        laser_angle = float(self.sensor_config["min_angle"]) + beam_idx * float(self.sensor_config["angle_increment"])
        return self._wrap(laser_angle + self.base_to_laser_yaw)

    def _angle_to_vfh_sector(self, angle, sector_count):
        if self.sensor_config["num_beams"] <= 0:
            return 0
        laser_angle = self._wrap(angle - self.base_to_laser_yaw)
        rel = laser_angle - float(self.sensor_config["min_angle"])
        span = abs(float(self.sensor_config["angle_increment"])) * max(1, int(self.sensor_config["num_beams"]) - 1)
        if span <= 1e-6:
            return 0
        rel = max(0.0, min(span, rel))
        idx = int((rel / span) * max(1, int(sector_count)))
        return max(0, min(max(1, int(sector_count)) - 1, idx))

    def _is_angle_in_recovery_path_zone(self, target_angle):
        if not self.enable_recovery_path_avoidance:
            return False, None

        half = 0.5 * math.radians(max(0.0, float(self.recovery_path_avoidance_angle_deg)))
        if half <= 1e-6:
            return False, None

        # Always defer the direct rear 90-degree zone first.
        rear_err = abs(self._wrap(target_angle - math.pi))
        if rear_err <= half or abs(self._wrap(target_angle + math.pi)) <= half:
            return True, math.pi

        if self.current_pose is None or not self.path_trace:
            return False, None

        now = self._now_sec()
        x, y, yaw = self._pose_xy_yaw()
        min_dist = max(0.0, float(self.recovery_path_min_dist))
        max_dist = max(min_dist, float(self.recovery_path_max_dist))
        min_t = now - max(0.1, float(self.recovery_path_lookback_sec))

        for t, px, py in reversed(self.path_trace):
            if t < min_t:
                break
            dx = float(px) - x
            dy = float(py) - y
            d = math.hypot(dx, dy)
            if d < min_dist or d > max_dist:
                continue

            rel = self._wrap(math.atan2(dy, dx) - yaw)
            if abs(self._wrap(target_angle - rel)) <= half:
                return True, rel

        return False, None

    def _recovery_abs_yaw_for_rel(self, rel_angle):
        _, _, yaw = self._pose_xy_yaw()
        return self._wrap(yaw + float(rel_angle))

    def _is_recovery_heading_checked(self, rel_angle):
        if not self.recovery_checked_yaws:
            return False
        target_yaw = self._recovery_abs_yaw_for_rel(rel_angle)
        window = math.radians(max(1.0, float(self.recovery_heading_checked_window_deg)))
        for checked_yaw in self.recovery_checked_yaws:
            if abs(self._wrap(target_yaw - checked_yaw)) <= window:
                return True
        return False

    def _mark_recovery_heading_checked(self):
        _, _, yaw = self._pose_xy_yaw()
        window = math.radians(max(1.0, float(self.recovery_heading_checked_window_deg)))
        for checked_yaw in self.recovery_checked_yaws:
            if abs(self._wrap(yaw - checked_yaw)) <= window:
                return

        self.recovery_checked_yaws.append(yaw)
        max_count = max(4, int(self.recovery_checked_max_count))
        if len(self.recovery_checked_yaws) > max_count:
            self.recovery_checked_yaws = self.recovery_checked_yaws[-max_count:]

    def _build_preferred_recovery_sweep_sequence(self):
        max_deg = min(135.0, abs(float(self.recovery_scan_max_deg)))
        seq = [0, +30, -30, +60, -60, +90, -90, +120, -120]
        return [float(x) for x in seq if abs(float(x)) <= max_deg]

    def _build_deferred_recovery_sweep_sequence(self):
        max_deg = abs(float(self.recovery_scan_max_deg))
        seq = [+150, -150, +180]
        return [float(x) for x in seq if abs(float(x)) <= max_deg]

    def _select_unchecked_fixed_recovery_angle(self, allow_deferred=False):
        seq = self._build_deferred_recovery_sweep_sequence() if allow_deferred else self._build_preferred_recovery_sweep_sequence()
        for deg in seq:
            rel = math.radians(float(deg))
            in_path_zone, _ = self._is_angle_in_recovery_path_zone(rel)
            if in_path_zone and not allow_deferred:
                continue
            if (not in_path_zone) and allow_deferred:
                continue
            if self._is_recovery_heading_checked(rel):
                continue
            return rel
        return None

    def _compute_vfh_recovery_histogram(self):
        if self._ranges_clipped is None or self.sensor_config["num_beams"] == 0:
            return None, None

        sector_count = max(8, int(self.vfh_recovery_sector_count))
        n = len(self._ranges_clipped)
        max_range = min(float(self.vfh_recovery_max_range), float(self.sensor_config["range_max"]))
        danger_sum = np.zeros(sector_count, dtype=np.float32)
        open_sum = np.zeros(sector_count, dtype=np.float32)
        counts = np.zeros(sector_count, dtype=np.int32)

        for i, raw_r in enumerate(self._ranges_clipped):
            sector = min(sector_count - 1, int(i * sector_count / max(1, n)))
            r = min(max_range, float(raw_r))
            danger_sum[sector] += max(0.0, max_range - r)
            open_sum[sector] += r
            counts[sector] += 1

        counts_safe = np.maximum(counts, 1)
        h = danger_sum / counts_safe
        sector_open = open_sum / counts_safe

        radius = max(0, int(self.vfh_recovery_smooth_width))
        if radius <= 0:
            hp = h.copy()
        else:
            weights = np.array(
                [radius + 1 - abs(k) for k in range(-radius, radius + 1)],
                dtype=np.float32,
            )
            hp = np.zeros_like(h)
            for i in range(sector_count):
                acc = 0.0
                wsum = 0.0
                for off, weight in zip(range(-radius, radius + 1), weights):
                    j = i + off
                    if j < 0 or j >= sector_count:
                        continue
                    acc += float(h[j]) * float(weight)
                    wsum += float(weight)
                hp[i] = acc / max(1e-6, wsum)

        occupied = hp > float(self.vfh_recovery_threshold)
        return occupied, sector_open

    def _find_vfh_recovery_valleys(self, occupied):
        if occupied is None:
            return []
        occupied = np.asarray(occupied, dtype=bool)
        sector_count = len(occupied)
        if sector_count == 0:
            return []

        free = np.logical_not(occupied)
        if not np.any(free):
            return []

        valleys = []
        start = None
        for i, is_free in enumerate(free):
            if is_free and start is None:
                start = i
            elif (not is_free) and start is not None:
                valleys.append((start, i - 1))
                start = None
        if start is not None:
            valleys.append((start, sector_count - 1))

        min_width = max(1, int(self.vfh_recovery_min_valley_width))
        return [(s, e) for s, e in valleys if (e - s + 1) >= min_width]

    def _compute_vfh_recovery_candidates(self):
        if not self.enable_vfh_recovery:
            return []

        occupied, sector_open = self._compute_vfh_recovery_histogram()
        valleys = self._find_vfh_recovery_valleys(occupied)
        if occupied is None or sector_open is None or not valleys:
            return []

        sector_count = len(occupied)
        goal_rel = self._goal_relative_angle()
        goal_sector = self._angle_to_vfh_sector(goal_rel, sector_count)
        candidates = []

        for start_idx, end_idx in valleys:
            valley_indices = list(range(start_idx, end_idx + 1))
            valley_width = len(valley_indices)
            center_cont = 0.5 * (start_idx + end_idx)

            if valley_width <= self.vfh_recovery_wide_valley_min:
                target_cont = round(center_cont)
            else:
                buffer = int(math.ceil(self.vfh_recovery_wide_valley_min / 2.0)) + 1
                adj_start = min(end_idx, start_idx + buffer)
                adj_end = max(start_idx, end_idx - buffer)
                if goal_sector < adj_start:
                    target_cont = adj_start
                elif goal_sector > adj_end:
                    target_cont = adj_end
                else:
                    target_cont = goal_sector

            target_sector = int(round(target_cont))
            target_angle = self._vfh_sector_to_base_angle(target_sector, sector_count)
            open_range = float(np.percentile(sector_open[valley_indices], 75))
            if open_range < self.vfh_recovery_min_open_range:
                continue

            goal_alignment = -abs(self._wrap(target_angle - goal_rel))
            front_bias = -abs(target_angle)
            memory_penalty = 0.0
            if self.vfh_recovery_last_heading is not None:
                memory_penalty = abs(self._wrap(target_angle - self.vfh_recovery_last_heading))

            score = (
                1.6 * open_range
                + self.vfh_recovery_goal_bias * goal_alignment
                + self.vfh_recovery_front_bias * front_bias
                - self.vfh_recovery_memory_weight * memory_penalty
                + 0.03 * valley_width
            )
            candidates.append({
                "angle": target_angle,
                "open_range": open_range,
                "width": valley_width,
                "score": float(score),
            })

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    def _select_vfh_recovery_target_angle(self):
        candidates = self._compute_vfh_recovery_candidates()
        min_turn = abs(float(self.vfh_recovery_min_turn_deg))
        max_deg = abs(float(self.recovery_scan_max_deg))

        preferred = []
        deferred_path = []
        for item in candidates:
            in_path_zone, path_angle = self._is_angle_in_recovery_path_zone(float(item["angle"]))
            item["path_deferred"] = in_path_zone
            item["path_angle"] = path_angle
            item["checked"] = self._is_recovery_heading_checked(float(item["angle"]))
            if in_path_zone:
                deferred_path.append(item)
            else:
                preferred.append(item)

        unchecked_preferred = [item for item in preferred if not item.get("checked", False)]
        unchecked_deferred = [item for item in deferred_path if not item.get("checked", False)]

        if candidates:
            self._log_throttle(
                "info",
                "vfh_recovery_path_avoid",
                0.5,
                "VFH recovery coverage: preferred=%d unchecked=%d deferred=%d checked_yaws=%d deferred_allowed=%s"
                % (
                    len(preferred),
                    len(unchecked_preferred),
                    len(deferred_path),
                    len(self.recovery_checked_yaws),
                    str(self.recovery_path_deferred_allowed),
                )
            )

        if unchecked_preferred:
            cand = unchecked_preferred[0]

            raw_deg = math.degrees(self._wrap(float(cand["angle"])))
            deg = raw_deg
            straight_deadband = max(min_turn, abs(float(self.vfh_recovery_straight_deadband_deg)))
            if self.vfh_recovery_force_explore and abs(deg) < straight_deadband:
                sign = self.vfh_recovery_last_turn_sign
                retry_turn = max(min_turn, abs(float(self.vfh_recovery_retry_turn_deg)))
                deg = sign * retry_turn
                self._log_throttle(
                    "warn",
                    "vfh_recovery_continue_explore",
                    0.5,
                    "VFH recovery raw target %.0f deg is near front after failed recheck; continue %.0f deg exploration"
                    % (raw_deg, deg)
                )
            elif abs(deg) < min_turn:
                if abs(deg) > 1e-3:
                    sign = 1.0 if deg > 0.0 else -1.0
                else:
                    sign = self.vfh_recovery_last_turn_sign
                deg = sign * min_turn
            if abs(deg) > max_deg:
                deg = (1.0 if deg > 0.0 else -1.0) * max_deg

            self._log_throttle(
                "info",
                "vfh_recovery_target",
                0.5,
                "VFH recovery target: raw=%.0f deg cmd=%.0f deg open=%.2f width=%d score=%.2f path_deferred=%s"
                % (
                    raw_deg,
                    deg,
                    float(cand.get("open_range", 0.0)),
                    int(cand.get("width", 0)),
                    float(cand.get("score", 0.0)),
                    str(bool(cand.get("path_deferred", False))),
                )
            )
            return math.radians(deg)

        fixed_preferred = self._select_unchecked_fixed_recovery_angle(allow_deferred=False)
        if fixed_preferred is not None:
            deg = math.degrees(fixed_preferred)
            self._log_throttle(
                "info",
                "vfh_recovery_fixed_preferred",
                0.5,
                "VFH recovery preferred 270 sweep: no unchecked VFH preferred candidate, checking fixed %.0f deg"
                % deg
            )
            return fixed_preferred

        self.recovery_path_deferred_allowed = True

        if unchecked_deferred:
            cand = unchecked_deferred[0]
            deg = math.degrees(self._wrap(float(cand["angle"])))
            if abs(deg) < min_turn:
                deg = (1.0 if deg >= 0.0 else -1.0) * min_turn
            if abs(deg) > max_deg:
                deg = (1.0 if deg > 0.0 else -1.0) * max_deg
            self._log_throttle(
                "warn",
                "vfh_recovery_path_fallback",
                0.5,
                "VFH recovery preferred 270 exhausted; allowing deferred rear/path target %.0f deg"
                % deg
            )
            return math.radians(deg)

        fixed_deferred = self._select_unchecked_fixed_recovery_angle(allow_deferred=True)
        if fixed_deferred is not None:
            deg = math.degrees(fixed_deferred)
            self._log_throttle(
                "warn",
                "vfh_recovery_fixed_deferred",
                0.5,
                "VFH recovery preferred 270 exhausted; checking deferred rear/path fixed %.0f deg"
                % deg
            )
            return fixed_deferred

        if deferred_path:
            cand = deferred_path[0]
            deg = math.degrees(self._wrap(float(cand["angle"])))
            self._log_throttle(
                "warn",
                "vfh_recovery_deferred_repeat",
                0.5,
                "VFH recovery all headings checked; repeating best deferred target %.0f deg"
                % deg
            )
            return math.radians(deg)

        self._log_throttle(
            "warn",
            "vfh_recovery_no_target",
            0.5,
            "VFH recovery found no usable valley, fallback to fixed step"
        )

        return None

    def _select_fixed_recovery_target_angle(self):
        target = self._select_unchecked_fixed_recovery_angle(allow_deferred=False)
        if target is not None:
            return target
        return self._select_unchecked_fixed_recovery_angle(allow_deferred=True)

    def _start_recovery_scan(self, now):
        self.recovery_state = "SCAN"
        _, _, yaw = self._pose_xy_yaw()
        self.recovery_base_yaw = yaw

        target_rel = self._select_vfh_recovery_target_angle()
        if target_rel is None:
            target_rel = self._select_fixed_recovery_target_angle()

        if target_rel is None:
            self.get_logger().warning("RECOVERY has no VFH/fixed target, exit recovery")
            self._exit_recovery()
            self._publish_zero()
            return False

        self.recovery_current_offset_deg = math.degrees(target_rel)
        self.recovery_target_yaw = self._wrap(yaw + target_rel)
        self.recovery_last_switch_t = now
        self.vfh_recovery_last_heading = target_rel
        if abs(target_rel) > 1e-3:
            self.vfh_recovery_last_turn_sign = 1.0 if target_rel > 0.0 else -1.0
        self.vfh_recovery_force_explore = False
        self._publish_zero()
        return True

    def _enter_recovery(self, now):
        if self.recovery_state != "NONE":
            return

        self.recovery_enter_t = now
        self.recovery_backup_start_t = 0.0
        self.recovery_backup_start_xy = None

        _, _, yaw = self._pose_xy_yaw()
        self.recovery_base_yaw = yaw
        self.recovery_target_yaw = yaw
        self.recovery_current_offset_deg = 0.0
        self.recovery_last_switch_t = now

        self.recovery_scan_sequence_deg = []
        self.recovery_scan_index = 0
        self.vfh_recovery_force_explore = False
        self.recovery_checked_yaws = []
        self.recovery_path_deferred_allowed = False

        self.recovery_points.append({
            "xy": self._current_xy(),
            "t": now,
        })

        if not self.recovery_use_backup:
            self._log_throttle(
                "warn",
                "enter_recovery_vfh",
                0.5,
                "RECOVERY enter at (%.2f, %.2f): stop, VFH scan, then motion-tube recheck"
                % (self._current_xy()[0], self._current_xy()[1]),
            )
            self._start_recovery_scan(now)
            return

        self.recovery_state = "BACKUP"
        self.recovery_backup_start_t = now
        self.recovery_backup_start_xy = self._current_xy()
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
        self.no_feasible = False
        self.no_feasible_since = None
        self.vfh_recovery_force_explore = False
        self.recovery_backup_start_xy = None
        self.recovery_backup_start_t = 0.0
        self.recovery_current_offset_deg = 0.0
        self.recovery_last_switch_t = 0.0
        self.recovery_scan_sequence_deg = []
        self.recovery_scan_index = 0
        self.recovery_checked_yaws = []
        self.recovery_path_deferred_allowed = False

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
        if self.recovery_state == "BACKUP":
            cmd, done = self._recovery_backup_cmd_or_done(now)
            self._safe_publish_cmd(cmd)
            self._publish_tube_markers()

            if not done:
                return

            self._start_recovery_scan(now)
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
            self._mark_recovery_heading_checked()

            has_representative = self._has_representative_recovery_tube(feas)
            self._log_throttle(
                "info",
                "recovery_recheck_counts",
                0.5,
                "RECOVERY recheck after %.0f deg: tubes=%d feasible=%d representative=%s"
                % (
                    self.recovery_current_offset_deg,
                    len(self.motion_tubes),
                    len(feas),
                    str(has_representative),
                )
            )

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

            self._log_throttle(
                "warn",
                "recovery_recompute_vfh",
                0.5,
                "RECOVERY no representative tube after %.0f deg, recomputing VFH from latest scan"
                % self.recovery_current_offset_deg
            )
            self.vfh_recovery_force_explore = True
            self._start_recovery_scan(now)
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
        if (not self.motion_tubes and not self.path_trace) or not self.current_pose or self._is_shutting_down:
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

        if self.enable_path_trace_marker and len(self.path_trace) >= 2:
            path_marker = Marker()
            path_marker.header = header
            self._init_marker_pose(path_marker)
            path_marker.ns = "path_trace"
            path_marker.id = mid
            mid += 1
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.scale.x = 0.045
            path_marker.color = ColorRGBA(r=1.0, g=0.45, b=0.05, a=0.95)
            path_marker.points = [
                Point(x=float(px), y=float(py), z=Z + 0.02)
                for _, px, py in self.path_trace
            ]
            ma.markers.append(path_marker)

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
            for s in t.centerline_samples:
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
            for s in t.centerline_samples:
                sx, sy_local = float(s[0]), float(s[1])
                ox = rx + (sx * cy - sy_local * sy)
                oy = ry + (sx * sy + sy_local * cy)
                pts.append(Point(x=ox, y=oy, z=Z))
            m.points = pts
            ma.markers.append(m)

            if len(t.centerline_samples) > 0:
                end = t.centerline_samples[-1]
                sx, sy_local = float(end[0]), float(end[1])
                ox = rx + (sx * cy - sy_local * sy)
                oy = ry + (sx * sy + sy_local * cy)

                txt = Marker()
                txt.header = header
                self._init_marker_pose(txt)
                txt.ns = "motion_tube_labels"
                txt.id = mid
                mid += 1
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose.position.x = ox
                txt.pose.position.y = oy
                txt.pose.position.z = Z + 0.45
                txt.scale.z = 0.24
                txt.color = ColorRGBA(r=0.0, g=0.95, b=1.0, a=1.0)
                txt.text = "%s\ncost=%.2f  w=%.2f  T=%.1f" % (
                    t.group_name,
                    t.cost,
                    t.w,
                    t.T,
                )
                ma.markers.append(txt)

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
            "cmd=(%.2f, %.2f) sel=[%s]"
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
