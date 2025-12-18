#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from collections import deque
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
try:
    from nav2_msgs.action import NavigateToPose
    HAVE_NAV2 = True
except ImportError:
    HAVE_NAV2 = False
    NavigateToPose = None

class TemplateType(object):
    MOVE_STRAIGHT = 0
    STEER_LEFT = 1
    STEER_RIGHT = 2

class RobotGeometry(object):
    def __init__(self):
        self.length = 0.43
        self.width = 0.37
        self.radius = math.hypot(self.length/2.0, self.width/2.0)

class MotionTube(object):
    def __init__(self, template_type, v, w, T, samples, beam_indices, arc_len):
        self.template_type = template_type
        self.v = v
        self.w = w
        self.T = T
        self.samples = samples
        self.beam_indices = beam_indices
        self.arc_len = arc_len
        self.cost = float('inf')
        self.is_feasible = False
        self.goal_progress = 0.0
        self.min_clearance = float('inf')
        self.heading_after = 0.0
        self.obstacle_penalty = 0.0

class GoalOrientedMotionTubePlanner(Node):
    def __init__(self):
        super().__init__('goal_oriented_motion_tube_planner')
        
        self.declare_parameter("scan_topic", "/j100_0896/scan")
        self.declare_parameter("odom_topic", "/j100_0896/odometry/filtered")

        self.declare_parameter("safety_margin", 0.05)
        self.declare_parameter("critical_pad", 0.06)
        self.declare_parameter("warning_pad", 0.4)
        self.declare_parameter("max_v", 1.4)
        self.declare_parameter("max_w", 1.2)
        self.declare_parameter("velocity_layers", [0.3, 0.6, 0.9, 1.2])
        self.declare_parameter("num_angular_samples", 25)
        self.declare_parameter("base_to_laser_yaw", 0.0)
        self.declare_parameter("commit_time", 0.25)
        self.declare_parameter("switch_hysteresis", 0.04)
        self.declare_parameter("deadband_w", 0.05)
        self.declare_parameter("cmd_smooth_alpha", 0.35)
        self.declare_parameter("min_forward_scale", 0.25)
        self.declare_parameter("time_horizons", [0.8, 1.6, 2.4])

        self.declare_parameter("recovery_duration", 12.0)
        self.declare_parameter("recovery_speed", 0.6)

        self.declare_parameter("recovery_turn_seconds", 2.6) 
        self.declare_parameter("recovery_pause_seconds", 2.0)

        self.latest_scan = None
        self.current_pose = None

        self.sensor_config = {
            'min_angle': -np.pi,
            'max_angle':  np.pi,
            'angle_increment': 0.01,
            'range_min': 0.1,
            'range_max': 30.0,
            'num_beams': 0,
        }

        self.position_hist = deque(maxlen=80)
        self.last_scan_time = 0.0
        self.last_odom_time = 0.0
        self.planning_enabled = False

        self.scan_topic = self.get_parameter("scan_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value

        qos_scan = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.cmd_pub = self.create_publisher(Twist, '/j100_0896/cmd_vel', 1)
        self.marker_pub = self.create_publisher(MarkerArray, '/j100_0896/motion_tubes', 1)

        self.get_logger().info(f"Subscribing to Scan: {self.scan_topic} (Best Effort)")
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_scan)
        
        self.get_logger().info(f"Subscribing to Odom: {self.odom_topic}")
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos_reliable)

        if HAVE_NAV2:
            self._action_server = ActionServer(
                self,
                NavigateToPose,
                'navigate_to_pose', 
                self.execute_callback,
                goal_callback=self.goal_acceptance_callback,
                cancel_callback=self.cancel_callback,
                callback_group=ReentrantCallbackGroup()
            )
            self.get_logger().info("Nav2 Action Server started at /navigate_to_pose")
        else:
            self.get_logger().warn("Nav2 not installed. Using simple topic for goal.")
            self.create_subscription(PoseStamped, '/j100_0896/goal_pose', self.simple_goal_callback, 10)
            self.create_subscription(PoseStamped, '/move_base_simple/goal', self.simple_goal_callback, 10)

        # Robot Geometry
        self.robot = RobotGeometry()

        # Parameters
        self.recovery_turn_seconds = self.get_parameter("recovery_turn_seconds").value
        self.recovery_pause_seconds = self.get_parameter("recovery_pause_seconds").value
        self.recovery_state = 'TURNING' 
        self.recovery_phase_start_time = 0.0

        self.safety_margin = self.get_parameter("safety_margin").value
        self.critical_distance = self.robot.radius + self.get_parameter("critical_pad").value
        self.warning_distance  = self.critical_distance + self.get_parameter("warning_pad").value

        self.max_v = self.get_parameter("max_v").value
        self.max_w = self.get_parameter("max_w").value
        self.velocity_layers = self.get_parameter("velocity_layers").value
        N_ANG = self.get_parameter("num_angular_samples").value
        self.angular_rates = np.linspace(-self.max_w, self.max_w, N_ANG)

        self.base_to_laser_yaw = self.get_parameter("base_to_laser_yaw").value

        self.commit_time = self.get_parameter("commit_time").value
        self.switch_hysteresis = self.get_parameter("switch_hysteresis").value
        self.deadband_w = self.get_parameter("deadband_w").value

        self.last_selected_tube = None
        self.last_select_stamp = self.get_time_sec()

        self.cmd_v_filt = 0.0
        self.cmd_w_filt = 0.0
        self.alpha = self.get_parameter("cmd_smooth_alpha").value
        self.min_forward_scale = self.get_parameter("min_forward_scale").value

        self.stuck_detection_enabled = False
        self.enable_stuck_after = 8.0
        self.stuck_check_time = 4.0
        self.stuck_threshold = 0.15
        self.recovery_mode = False
        self.recovery_start_time = 0.0
        self.recovery_duration = 2.5

        self.recovery_duration = self.get_parameter("recovery_duration").value
        self.recovery_speed = self.get_parameter("recovery_speed").value
        self.time_horizons = self.get_parameter("time_horizons").value
        self.goal_tolerance = 0.45

        # State
        self.current_goal = None
        self.motion_tubes = []
        self.selected_tube = None
        self.min_cost = 0.0
        self.max_cost = 1.0

        # Timers
        self.timer = self.create_timer(0.15, self.planning_cycle)
        self.diag_timer = self.create_timer(3.0, self.print_diagnostics)

        self.get_logger().info('Motion Tube Planner (ROS 2 Port - j100_0896) initialized')

    # ---------- Helpers for ROS 2 time ----------
    def get_time_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    # ---------- Callbacks ----------
    def scan_callback(self, msg):
        self.latest_scan = msg
        self.last_scan_time = self.get_time_sec()
        if self.sensor_config['num_beams'] == 0:
            inc = msg.angle_increment if abs(msg.angle_increment) > 1e-9 else 1e-3
            self.sensor_config.update({
                'min_angle': msg.angle_min,
                'max_angle': msg.angle_max,
                'angle_increment': inc,
                'range_min': msg.range_min,
                'range_max': msg.range_max,
                'num_beams': len(msg.ranges)
            })
            self.planning_enabled = True
            self.get_logger().info(f"Laser: {len(msg.ranges)} beams, FOV {math.degrees(msg.angle_max - msg.angle_min):.1f} deg")

    def odom_callback(self, msg):
        self.current_pose = PoseStamped()
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.pose.pose
        self.last_odom_time = self.get_time_sec()

        t = self.get_time_sec()
        p = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.position_hist.append((t, p))

        if not self.stuck_detection_enabled and (t - self.last_scan_time) > self.enable_stuck_after:
            self.stuck_detection_enabled = True
            self.get_logger().info("Stuck detection enabled")

    # ---------- Action Server Logic ----------
    def goal_acceptance_callback(self, goal_request):
        self.get_logger().info('Received Goal Request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received Cancel Request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        self.current_goal = PoseStamped()
        self.current_goal.header = goal_handle.request.pose.header
        self.current_goal.pose = goal_handle.request.pose.pose
        
        # Reset state
        self.recovery_mode = False
        self.position_hist.clear()
        self.stuck_detection_enabled = False
        self.last_select_stamp = self.get_time_sec()

        result = NavigateToPose.Result()
        
        rate = self.create_rate(10)
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.cmd_pub.publish(Twist()) # Stop
                self.get_logger().info('Goal Canceled')
                return result

            if self.current_pose is None:
                rate.sleep()
                continue

            # Check if reached
            if self.is_goal_reached():
                self.cmd_pub.publish(Twist()) # Stop
                goal_handle.succeed()
                self.get_logger().info("Goal reached")
                return result
            
            rate.sleep()
        
        return result

    def simple_goal_callback(self, msg):
        self.current_goal = msg
        self.recovery_mode = False
        self.position_hist.clear()
        self.stuck_detection_enabled = False
        self.last_select_stamp = self.get_time_sec()
        self.get_logger().info(f"New goal via topic: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    # ---------- Planning loop ----------
    def is_goal_reached(self):
        if not self.current_goal or not self.current_pose: return False
        dx = self.current_goal.pose.position.x - self.current_pose.pose.position.x
        dy = self.current_goal.pose.position.y - self.current_pose.pose.position.y
        return math.hypot(dx, dy) < self.goal_tolerance

    def planning_cycle(self):
        if not (self.planning_enabled and self.current_goal and self.current_pose and self.latest_scan): return
        if (self.get_time_sec() - self.last_scan_time) > 1.0: return

        if self.stuck_detection_enabled: self.update_stuck()
        self.generate_motion_tubes_dense()
        self.evaluate_tubes_better()
        self.select_best_tube_commit()
        self.publish_commands_smoothed()
        self._publish_tube_markers()

    # ---------- Helpers ----------
    def _pose_xy_yaw(self):
        if not self.current_pose: return 0.0, 0.0, 0.0
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        q = self.current_pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return x, y, yaw

    def _norm_range(self, r):
        if r is None or math.isnan(r) or math.isinf(r) or r <= 0.0:
            return self.sensor_config['range_max']
        return max(self.sensor_config['range_min'], min(r, self.sensor_config['range_max']))

    def angle_to_beam_idx(self, angle_in_base):
        if not self.latest_scan or self.sensor_config['num_beams'] == 0: return 0
        angle_in_laser = self._wrap(angle_in_base + self.base_to_laser_yaw)
        ang_min = self.sensor_config['min_angle']
        ang_max = self.sensor_config['max_angle']
        inc = self.sensor_config['angle_increment']
        n = self.sensor_config['num_beams']

        if inc > 0:
            if angle_in_laser <= ang_min: return 0
            if angle_in_laser >= ang_max: return n - 1
        else:
            if angle_in_laser >= ang_min: return 0
            if angle_in_laser <= ang_max: return n - 1

        idx = int((angle_in_laser - ang_min) / inc)
        return max(0, min(n - 1, idx))

    def get_forward_clearance(self):
        if not self.latest_scan: return float('inf')
        ang_min = self.sensor_config['min_angle']
        inc = self.sensor_config['angle_increment']
        n = self.sensor_config['num_beams']
        forward_laser = self._wrap(-self.base_to_laser_yaw)
        half = math.radians(25.0)
        best = float('inf')
        for i in range(n):
            a = ang_min + i * inc
            if abs(self._wrap(a - forward_laser)) <= half:
                best = min(best, self._norm_range(self.latest_scan.ranges[i]))
        return best

    def _wrap(self, ang):
        while ang > math.pi: ang -= 2*math.pi
        while ang < -math.pi: ang += 2*math.pi
        return ang

    def update_stuck(self):
        if len(self.position_hist) < 2 or self.recovery_mode: return
        t_now = self.get_time_sec()
        pts = [(t, p) for (t, p) in self.position_hist if t >= (t_now - self.stuck_check_time)]
        if len(pts) < 2: return
        dist = 0.0
        for i in range(1, len(pts)):
            dx = pts[i][1][0] - pts[i-1][1][0]
            dy = pts[i][1][1] - pts[i-1][1][1]
            dist += math.hypot(dx, dy)
        if dist < self.stuck_threshold:
            self.recovery_mode = True
            self.recovery_start_time = t_now
            self.get_logger().warn(f"Stuck detected ({dist:.2f}m in {self.stuck_check_time:.1f}s)")
        elif self.recovery_mode and (t_now - self.recovery_start_time) > self.recovery_duration:
            self.recovery_mode = False
            self.get_logger().info("Exit recovery")

    # ---------- Tube generation ----------
    def generate_motion_tubes_dense(self):
        self.motion_tubes = []

        # if forward space is tight, favor shorter T and more turns
        fwd = self.get_forward_clearance()
        horizons = list(self.time_horizons)
        if fwd < 0.8:
            horizons = [self.time_horizons[0], self.time_horizons[1]]
        if self.recovery_mode:
            v_layers = [max(0.2, self.velocity_layers[0]), self.velocity_layers[1]]
            angs = [w for w in self.angular_rates if abs(w) > 0.25]
        else:
            v_layers = list(self.velocity_layers)
            angs = list(self.angular_rates)

        for T in horizons:
            for v in v_layers:
                for w in angs:
                    approx_len = abs(v) * T
                    if fwd < 0.6 and approx_len > (fwd + 0.3): continue
                    tube = self.create_motion_tube(v, w, T)
                    if tube: self.motion_tubes.append(tube)

    def create_motion_tube(self, v, w, T):
        if abs(w) < 0.05: ttype = TemplateType.MOVE_STRAIGHT
        elif w > 0: ttype = TemplateType.STEER_LEFT
        else: ttype = TemplateType.STEER_RIGHT

        samples, beam_indices = [], []
        arc_len = abs(v) * T
        n_samples = int(np.clip(8 + arc_len * 12.0, 12, 48))

        for i in range(1, n_samples+1):
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
            idx = self.angle_to_beam_idx(ang_base + 0.1 * heading)
            beam_indices.append(idx)

        return MotionTube(ttype, v, w, T, samples, beam_indices, arc_len)

    # ---------- Collision check ----------
    def evaluate_tubes_better(self):
        if not self.motion_tubes: return
        yaw = self._pose_xy_yaw()[2]
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        goal_bearing = math.atan2(gy - cy, gx - cx)

        for t in self.motion_tubes:
            t.is_feasible, t.min_clearance, t.obstacle_penalty = self.collision_check_graded(t)
            if not t.is_feasible: continue
            t.heading_after = yaw + t.w * t.T
            t.goal_progress = self.goal_progress_along_tube(t)
            t.cost = self.composite_cost(t, goal_bearing)

        feas = [x for x in self.motion_tubes if x.is_feasible]
        if feas:
            self.min_cost = min(x.cost for x in feas)
            self.max_cost = max(x.cost for x in feas)
        else: self.min_cost = self.max_cost = 0.0

    def collision_check_graded(self, tube):
        if not self.latest_scan: return False, 0.0, 0.0
        ranges = self.latest_scan.ranges
        inc = self.sensor_config['angle_increment']
        n = self.sensor_config['num_beams']
        min_clr = float('inf'); penalty = 0.0; hard = False

        idxs = np.linspace(0, len(tube.samples)-1, num=min(8, len(tube.samples)), dtype=int)
        for k in idxs:
            s = tube.samples[k]
            dist = float(np.linalg.norm(s))
            if dist < 1e-3: continue
            half = math.atan2(self.robot.radius, max(0.1, dist))
            center_idx = tube.beam_indices[k]
            beam_span = max(1, int(half / max(abs(inc), 1e-6)))

            i0 = max(0, center_idx - beam_span)
            i1 = min(n - 1, center_idx + beam_span)

            for j in range(i0, i1+1):
                r = self._norm_range(ranges[j])
                clr = r - dist
                min_clr = min(min_clr, clr)
                if clr < self.critical_distance * 0.6:
                    hard = True
                    penalty += (self.critical_distance*0.6 - clr) * 400.0
                elif clr < self.critical_distance: penalty += (self.critical_distance - clr) * 60.0
                elif clr < self.warning_distance: penalty += (self.warning_distance - clr) * 8.0
        return (not hard), min_clr, penalty

    # ---------- Cost ----------
    def goal_progress_along_tube(self, tube):
        cx, cy, yaw = self._pose_xy_yaw()
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        now = math.hypot(gx - cx, gy - cy)

        if abs(tube.w) < 1e-3:
            fx = cx + tube.v * tube.T * math.cos(yaw)
            fy = cy + tube.v * tube.T * math.sin(yaw)
        else:
            R = tube.v / tube.w
            dth = tube.w * tube.T
            fx = cx + R * (math.sin(yaw + dth) - math.sin(yaw))
            fy = cy - R * (math.cos(yaw + dth) - math.cos(yaw))
        after = math.hypot(gx - fx, gy - fy)
        return max(0.0, now - after)

    def composite_cost(self, t, goal_bearing):
        w_progress   = 20.0
        w_heading    =  6.0
        w_obstacle   =  1.0
        w_curvature  =  0.7
        w_length     =  0.2
        w_speed      =  1.5

        # 1) Progress
        c = 0.0
        c -= w_progress * t.goal_progress

        # 2) Heading alignment at end of rollout
        heading_err = abs(self._wrap(t.heading_after - goal_bearing))
        c += w_heading * heading_err

        # 3) Obstacle penalty
        c += w_obstacle * t.obstacle_penalty

        # 4) Curvature penalty
        c += w_curvature * abs(t.w)

        # 5) Path length
        c += w_length * t.arc_len

        # 6) Speed bonus
        c -= w_speed * t.v

        # Recovery encourage turning
        if self.recovery_mode and abs(t.w) > 0.25: c -= 5.0

        return c

    def select_best_tube_commit(self):
            feas = [t for t in self.motion_tubes if t.is_feasible]

            if self.recovery_mode and feas:
                self.recovery_mode = False
                self.get_logger().info("Found path during recovery, stopping rotation!")

            if not feas:
                self.selected_tube = None
                
                if not self.recovery_mode:
                    self.recovery_mode = True
                    self.recovery_start_time = self.get_time_sec()
                    self.recovery_state = 'TURNING' 
                    self.recovery_phase_start_time = self.get_time_sec()
                    self.get_logger().warn("No feasible tubes! Forcing recovery mode (Rotation).")
                # =========================================================
                return

            best = min(feas, key=lambda z: z.cost)
            now = self.get_time_sec()
            if self.last_selected_tube and (now - self.last_select_stamp) < self.commit_time:
                self.selected_tube = self.last_selected_tube
                return

            if self.last_selected_tube and (self.last_selected_tube in feas):
                delta = best.cost - self.last_selected_tube.cost
                if delta > -self.switch_hysteresis:
                    self.selected_tube = self.last_selected_tube
                    return

            self.selected_tube = best
            self.last_selected_tube = best
            self.last_select_stamp = now

    # ---------- Commands ----------
    def publish_commands_smoothed(self):
        cmd = Twist()
        if self.selected_tube:
            v = max(0.0, min(self.max_v, self.selected_tube.v))
            w = max(-self.max_w, min(self.max_w, self.selected_tube.w))

            if abs(w) < self.deadband_w: w = 0.0
            fwd = self.get_forward_clearance()
            if fwd < (self.warning_distance + 0.3):
                scale = max(self.min_forward_scale, min(1.0, fwd / (self.warning_distance + 0.3)))
                v *= scale

            self.cmd_v_filt = self.alpha * v + (1 - self.alpha) * self.cmd_v_filt
            self.cmd_w_filt = self.alpha * w + (1 - self.alpha) * self.cmd_w_filt

            cmd.linear.x = float(self.cmd_v_filt)
            cmd.angular.z = float(self.cmd_w_filt)
        else:
            if self.recovery_mode:
                cmd.linear.x = 0.0

                now = self.get_time_sec()
                phase_time = now - self.recovery_phase_start_time
                
                if self.recovery_state == 'TURNING':
                    cmd.angular.z = self.recovery_speed
                    if phase_time > self.recovery_turn_seconds:
                        self.recovery_state = 'PAUSED'
                        self.recovery_phase_start_time = now
                        self.get_logger().info("Recovery: Pausing to scan...")
                        
                elif self.recovery_state == 'PAUSED':
                    cmd.angular.z = 0.0
                    if phase_time > self.recovery_pause_seconds:
                        self.recovery_state = 'TURNING'
                        self.recovery_phase_start_time = now
                        self.get_logger().info("Recovery: Turning again...")
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)
        
    # ---------- Viz ----------
    def _tube_color(self, tube):
        if tube is self.selected_tube: return ColorRGBA(r=0.00, g=0.95, b=1.00, a=1.0)
        if not tube.is_feasible: return ColorRGBA(r=0.95, g=0.15, b=0.15, a=0.35)
        try: c = max(0.0, min(1.0, (tube.cost - self.min_cost) / (self.max_cost - self.min_cost + 1e-6)))
        except: c = 0.5
        return ColorRGBA(r=1.0*c, g=1.0, b=0.0, a=0.9)

    def _publish_tube_markers(self):
        if not self.motion_tubes or not self.current_pose: return
        rx, ry, ryaw = self._pose_xy_yaw()
        cy, sy = math.cos(ryaw), math.sin(ryaw)
        ma = MarkerArray()
        frame = (self.current_pose.header.frame_id or "odom")
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame

        wipe = Marker()
        wipe.header = header; wipe.ns = "motion_tubes"; wipe.id = 0
        wipe.action = Marker.DELETEALL
        ma.markers.append(wipe)

        mid = 1; Z = 0.03; BASE_W = 0.05; SEL_W = 0.09
        for t in self.motion_tubes:
            m = Marker()
            m.header = header; m.ns = "motion_tubes"; m.id = mid; mid += 1
            m.type = Marker.LINE_STRIP; m.action = Marker.ADD
            m.scale.x = BASE_W; m.color = self._tube_color(t)
            pts = []
            for s in t.samples:
                sx, sy_local = float(s[0]), float(s[1])
                ox = rx + ( sx * cy - sy_local * sy )
                oy = ry + ( sx * sy + sy_local * cy )
                pts.append(Point(x=ox, y=oy, z=Z))
            m.points = pts
            ma.markers.append(m)

        if self.selected_tube is not None:
            t = self.selected_tube
            m = Marker()
            m.header = header; m.ns = "motion_tubes"; m.id = mid
            m.type = Marker.LINE_STRIP; m.action = Marker.ADD
            m.scale.x = SEL_W; m.color = ColorRGBA(r=0.00, g=0.95, b=1.00, a=1.0)
            pts = []
            for s in t.samples:
                sx, sy_local = float(s[0]), float(s[1])
                ox = rx + ( sx * cy - sy_local * sy )
                oy = ry + ( sx * sy + sy_local * cy )
                pts.append(Point(x=ox, y=oy, z=Z))
            m.points = pts
            ma.markers.append(m)

        self.marker_pub.publish(ma)

    # ---------- Diagnostics ----------
    def print_diagnostics(self):
        feas = [t for t in self.motion_tubes if t.is_feasible]
        fwd = self.get_forward_clearance()
        self.get_logger().info(f"=== PLANNER DIAG === tubes={len(self.motion_tubes)} feas={len(feas)} fwd={fwd:.2f}m")

def main(args=None):
    rclpy.init(args=args)
    planner = GoalOrientedMotionTubePlanner()

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(planner)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        planner.cmd_pub.publish(Twist()) # Stop robot
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()