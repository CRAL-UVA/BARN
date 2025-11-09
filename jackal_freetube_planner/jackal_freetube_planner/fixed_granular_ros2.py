#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import math
import numpy as np
from collections import deque
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header


try:
    from nav2_msgs.action import NavigateToPose
    HAVE_NAV2 = True
except Exception:
    NavigateToPose = None
    HAVE_NAV2 = False




class TemplateType:
    MOVE_STRAIGHT = 0
    STEER_LEFT = 1
    STEER_RIGHT = 2


class RobotGeometry:
    def __init__(self):
        self.length = 0.43
        self.width = 0.37
        self.radius = math.hypot(self.length / 2.0, self.width / 2.0)


class MotionTube:
    def __init__(self, template_type, v, w, T, samples, beam_indices, arc_len):
        self.template_type = template_type
        self.v = v
        self.w = w
        self.T = T
        self.samples: List[np.ndarray] = samples
        self.beam_indices: List[int] = beam_indices
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

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('marker_topic', '/motion_tubes')

        self.declare_parameter('scan_qos_best_effort', False)

        self.declare_parameter('safety_margin', 0.05)
        self.declare_parameter('critical_pad', 0.06)
        self.declare_parameter('warning_pad', 0.4)

        self.declare_parameter('max_v', 1.4)
        self.declare_parameter('max_w', 1.2)
        self.declare_parameter('velocity_layers', [0.3, 0.6, 0.9, 1.2])
        self.declare_parameter('num_angular_samples', 25)
        self.declare_parameter('base_to_laser_yaw', 0.0)

        self.declare_parameter('commit_time', 0.25)
        self.declare_parameter('switch_hysteresis', 0.04)
        self.declare_parameter('deadband_w', 0.05)
        self.declare_parameter('cmd_smooth_alpha', 0.35)
        self.declare_parameter('min_forward_scale', 0.25)

        self.declare_parameter('time_horizons', [0.8, 1.6, 2.4])
        self.declare_parameter('goal_tolerance', 0.45)

        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self.scan_qos_best_effort = bool(self.get_parameter('scan_qos_best_effort').value)

        self.safety_margin = float(self.get_parameter('safety_margin').value)
        self.robot = RobotGeometry()
        self.critical_distance = self.robot.radius + float(self.get_parameter('critical_pad').value)
        self.warning_distance = self.critical_distance + float(self.get_parameter('warning_pad').value)

        self.max_v = float(self.get_parameter('max_v').value)
        self.max_w = float(self.get_parameter('max_w').value)
        self.velocity_layers = list(self.get_parameter('velocity_layers').value)
        N_ANG = int(self.get_parameter('num_angular_samples').value)
        self.angular_rates = np.linspace(-self.max_w, self.max_w, N_ANG)

        self.base_to_laser_yaw = float(self.get_parameter('base_to_laser_yaw').value)

        self.commit_time = float(self.get_parameter('commit_time').value)
        self.switch_hysteresis = float(self.get_parameter('switch_hysteresis').value)
        self.deadband_w = float(self.get_parameter('deadband_w').value)

        self.alpha = float(self.get_parameter('cmd_smooth_alpha').value)
        self.min_forward_scale = float(self.get_parameter('min_forward_scale').value)

        self.time_horizons = list(self.get_parameter('time_horizons').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)

        self.latest_scan: Optional[LaserScan] = None
        self.current_pose: Optional[PoseStamped] = None
        self.position_hist = deque(maxlen=80)

        self.sensor_config = {
            'min_angle': -np.pi,
            'max_angle': np.pi,
            'angle_increment': 0.01,
            'range_min': 0.1,
            'range_max': 30.0,
            'num_beams': 0
        }

        self.last_scan_time = 0.0
        self.last_odom_time = 0.0
        self.planning_enabled = False

        self.last_selected_tube: Optional[MotionTube] = None
        self.last_select_stamp = self.now_sec()

        self.cmd_v_filt = 0.0
        self.cmd_w_filt = 0.0

        self.stuck_detection_enabled = False
        self.enable_stuck_after = 8.0
        self.stuck_check_time = 4.0
        self.stuck_threshold = 0.15
        self.recovery_mode = False
        self.recovery_start_time = 0.0
        self.recovery_duration = 2.5

        self.current_goal: Optional[PoseStamped] = None
        self.motion_tubes: List[MotionTube] = []
        self.selected_tube: Optional[MotionTube] = None
        self.min_cost = 0.0
        self.max_cost = 1.0

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT if self.scan_qos_best_effort else ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 1)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 1)

        self.scan_topic = self._maybe_autodetect(self.scan_topic, [
            '/front/scan', '/laser/scan', '/base_scan', '/scan'
        ], LaserScan)
        self.odom_topic = self._maybe_autodetect(self.odom_topic, [
            '/odometry/filtered', '/odom', '/jackal/odom', '/robot/odom'
        ], Odometry)

        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, scan_qos)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, odom_qos)

        self.create_subscription(PoseStamped, '/freetube/goal', self.goal_topic_callback, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_topic_callback, 10)  # RViz

        if HAVE_NAV2:
            self.action_server = ActionServer(
                self,
                NavigateToPose,
                '/navigate_to_pose',
                execute_callback=self.nav2_execute_cb,
                goal_callback=self.nav2_goal_cb,
                cancel_callback=self.nav2_cancel_cb
            )
            self.get_logger().info('NavigateToPose ActionServer started at /navigate_to_pose')
        else:
            self.action_server = None
            self.get_logger().warn('nav2_msgs not found; using /freetube/goal or /move_base_simple/goal to set targets.')

        self.create_timer(0.15, self.planning_cycle) 
        self.create_timer(3.0, self.print_diagnostics)

        self.get_logger().info(f'Using scan: {self.scan_topic}, odom: {self.odom_topic}, cmd_vel: {self.cmd_vel_topic}')
        self.get_logger().info('Motion Tube Planner (ROS2 Humble) initialized')

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _wrap(self, ang: float) -> float:
        while ang > math.pi:
            ang -= 2 * math.pi
        while ang < -math.pi:
            ang += 2 * math.pi
        return ang

    def _pose_xy_yaw(self) -> Tuple[float, float, float]:
        if self.current_pose is None:
            return 0.0, 0.0, 0.0
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        q = self.current_pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return x, y, yaw

    def _norm_range(self, r: Optional[float]) -> float:
        if r is None or math.isnan(r) or math.isinf(r) or r <= 0.0:
            return self.sensor_config['range_max']
        return max(self.sensor_config['range_min'], min(r, self.sensor_config['range_max']))

    def _maybe_autodetect(self, param_value: str, candidates: List[str], msg_type):
        if param_value and param_value != 'auto':
            return param_value

        self.get_logger().info(f'Auto-detecting {msg_type.__name__} topic ...')
        got_flag = {'ok': False}

        for topic in candidates:
            got_flag['ok'] = False

            def once(_msg):
                got_flag['ok'] = True

            temp_sub = self.create_subscription(msg_type, topic, once, 10)
            start = self.now_sec()
            while (self.now_sec() - start) < 2.0 and not got_flag['ok']:
                rclpy.spin_once(self, timeout_sec=0.05)
            self.destroy_subscription(temp_sub)

            if got_flag['ok']:
                self.get_logger().info(f'Using detected {msg_type.__name__} topic: {topic}')
                return topic

        default = candidates[-1] if candidates else ''
        self.get_logger().warn(f'Failed to auto-detect {msg_type.__name__}; fallback to {default}')
        return default

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self.last_scan_time = self.now_sec()
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
            fov_deg = math.degrees(msg.angle_max - msg.angle_min)
            self.get_logger().info(f'Laser: {len(msg.ranges)} beams, FOV {fov_deg:.1f}°, inc={inc:.5f} rad')

    def odom_callback(self, msg: Odometry):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self.current_pose = ps
        self.last_odom_time = self.now_sec()

        t = self.now_sec()
        p = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.position_hist.append((t, p))

        if not self.stuck_detection_enabled and (t - self.last_scan_time) > self.enable_stuck_after:
            self.stuck_detection_enabled = True
            self.get_logger().info('Stuck detection enabled')

    def goal_topic_callback(self, goal_msg: PoseStamped):
        self.current_goal = goal_msg
        self.recovery_mode = False
        self.position_hist.clear()
        self.stuck_detection_enabled = False
        self.last_select_stamp = self.now_sec()
        self.get_logger().info(
            f'New goal (topic): ({goal_msg.pose.position.x:.2f}, {goal_msg.pose.position.y:.2f})'
        )

    def nav2_goal_cb(self, _goal_request):
        return GoalResponse.ACCEPT

    def nav2_cancel_cb(self, _goal_handle):
        return CancelResponse.ACCEPT

    async def nav2_execute_cb(self, goal_handle):
        goal = goal_handle.request
        self.current_goal = PoseStamped()
        self.current_goal.header = goal.pose.header
        self.current_goal.pose = goal.pose.pose
        self.recovery_mode = False
        self.position_hist.clear()
        self.stuck_detection_enabled = False
        self.last_select_stamp = self.now_sec()
        self.get_logger().info(
            f'New goal (action): ({self.current_goal.pose.position.x:.2f}, {self.current_goal.pose.position.y:.2f})'
        )

        result = NavigateToPose.Result()
        rate = rclpy.rate.Rate(10.0, self.get_clock())
        while rclpy.ok():
            if self.is_goal_reached():
                goal_handle.succeed()
                self.get_logger().info('Goal reached (action)')
                break
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().warn('Goal canceled')
                break
            await rate.sleep()

        return result

    def is_goal_reached(self) -> bool:
        if self.current_goal is None or self.current_pose is None:
            return False
        dx = self.current_goal.pose.position.x - self.current_pose.pose.position.x
        dy = self.current_goal.pose.position.y - self.current_pose.pose.position.y
        return math.hypot(dx, dy) < self.goal_tolerance

    def planning_cycle(self):
        if not (self.planning_enabled and self.current_goal and self.current_pose and self.latest_scan):
            return
        if (self.now_sec() - self.last_scan_time) > 1.0:
            return

        if self.stuck_detection_enabled:
            self.update_stuck()
        self.generate_motion_tubes_dense()
        self.evaluate_tubes_better()
        self.select_best_tube_commit()
        self.publish_commands_smoothed()
        self._publish_tube_markers()

    def angle_to_beam_idx(self, angle_in_base: float) -> int:
        if (self.latest_scan is None) or (self.sensor_config['num_beams'] == 0):
            return 0
        angle_in_laser = self._wrap(angle_in_base + self.base_to_laser_yaw)
        ang_min = self.sensor_config['min_angle']
        ang_max = self.sensor_config['max_angle']
        inc = self.sensor_config['angle_increment']
        n = self.sensor_config['num_beams']

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

    def get_forward_clearance(self) -> float:
        if self.latest_scan is None:
            return float('inf')
        ang_min = self.sensor_config['min_angle']
        inc = self.sensor_config['angle_increment']
        n = self.sensor_config['num_beams']
        forward_laser = self._wrap(-self.base_to_laser_yaw)
        half = math.radians(25.0)
        best = float('inf')
        ranges = self.latest_scan.ranges
        for i in range(n):
            a = ang_min + i * inc
            if abs(self._wrap(a - forward_laser)) <= half:
                best = min(best, self._norm_range(ranges[i]))
        return best

    def update_stuck(self):
        if len(self.position_hist) < 2 or self.recovery_mode:
            return
        t_now = self.now_sec()
        pts = [(t, p) for (t, p) in self.position_hist if t >= (t_now - self.stuck_check_time)]
        if len(pts) < 2:
            return
        dist = 0.0
        for i in range(1, len(pts)):
            dx = pts[i][1][0] - pts[i - 1][1][0]
            dy = pts[i][1][1] - pts[i - 1][1][1]
            dist += math.hypot(dx, dy)
        if dist < self.stuck_threshold:
            self.recovery_mode = True
            self.recovery_start_time = t_now
            self.get_logger().warn(f'Stuck detected ({dist:.2f}m in {self.stuck_check_time:.1f}s)')
        elif self.recovery_mode and (t_now - self.recovery_start_time) > self.recovery_duration:
            self.recovery_mode = False
            self.get_logger().info('Exit recovery')

    def generate_motion_tubes_dense(self):
        self.motion_tubes = []

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
                    if fwd < 0.6 and approx_len > (fwd + 0.3):
                        continue
                    tube = self.create_motion_tube(v, w, T)
                    if tube:
                        self.motion_tubes.append(tube)

    def create_motion_tube(self, v, w, T) -> MotionTube:
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
            idx = self.angle_to_beam_idx(ang_base + 0.1 * heading)
            beam_indices.append(idx)

        return MotionTube(ttype, v, w, T, samples, beam_indices, arc_len)

    def evaluate_tubes_better(self):
        if not self.motion_tubes:
            return
        yaw = self._pose_xy_yaw()[2]
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        goal_bearing = math.atan2(gy - cy, gx - cx)

        for t in self.motion_tubes:
            t.is_feasible, t.min_clearance, t.obstacle_penalty = self.collision_check_graded(t)
            if not t.is_feasible:
                continue
            t.heading_after = yaw + t.w * t.T
            t.goal_progress = self.goal_progress_along_tube(t)
            t.cost = self.composite_cost(t, goal_bearing)

        feas = [x for x in self.motion_tubes if x.is_feasible]
        if feas:
            self.min_cost = min(x.cost for x in feas)
            self.max_cost = max(x.cost for x in feas)
        else:
            self.min_cost = self.max_cost = 0.0

    def collision_check_graded(self, tube: MotionTube):
        if self.latest_scan is None:
            return False, 0.0, 0.0
        ranges = self.latest_scan.ranges
        inc = self.sensor_config['angle_increment']
        n = self.sensor_config['num_beams']
        min_clr = float('inf')
        penalty = 0.0
        hard = False

        idxs = np.linspace(0, len(tube.samples) - 1, num=min(8, len(tube.samples)), dtype=int)
        for k in idxs:
            s = tube.samples[k]
            dist = float(np.linalg.norm(s))
            if dist < 1e-3:
                continue
            half = math.atan2(self.robot.radius, max(0.1, dist))
            center_idx = tube.beam_indices[k]
            beam_span = max(1, int(half / max(abs(inc), 1e-6)))

            i0 = max(0, center_idx - beam_span)
            i1 = min(n - 1, center_idx + beam_span)

            for j in range(i0, i1 + 1):
                r = self._norm_range(ranges[j])
                clr = r - dist
                min_clr = min(min_clr, clr)
                if clr < self.critical_distance * 0.6:
                    hard = True
                    penalty += (self.critical_distance * 0.6 - clr) * 400.0
                elif clr < self.critical_distance:
                    penalty += (self.critical_distance - clr) * 60.0
                elif clr < self.warning_distance:
                    penalty += (self.warning_distance - clr) * 8.0
        return (not hard), min_clr, penalty

    def goal_progress_along_tube(self, tube: MotionTube) -> float:
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

    def composite_cost(self, t: MotionTube, goal_bearing: float) -> float:
        w_progress = 20.0
        w_heading = 6.0
        w_obstacle = 1.0
        w_curvature = 0.7
        w_length = 0.2
        w_speed = 1.5

        c = 0.0
        c -= w_progress * t.goal_progress

        heading_err = abs(self._wrap(t.heading_after - goal_bearing))
        c += w_heading * heading_err

        c += w_obstacle * t.obstacle_penalty
        c += w_curvature * abs(t.w)
        c += w_length * t.arc_len
        c -= w_speed * t.v

        if self.recovery_mode and abs(t.w) > 0.25:
            c -= 5.0

        return c

    def select_best_tube_commit(self):
        feas = [t for t in self.motion_tubes if t.is_feasible]
        if not feas:
            self.selected_tube = None
            return

        best = min(feas, key=lambda z: z.cost)
        now = self.now_sec()
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

    def publish_commands_smoothed(self):
        cmd = Twist()
        if self.selected_tube:
            v = max(0.0, min(self.max_v, self.selected_tube.v))
            w = max(-self.max_w, min(self.max_w, self.selected_tube.w))

            if abs(w) < self.deadband_w:
                w = 0.0
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
                cmd.linear.x = 0.05
                cmd.angular.z = 0.5
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    def _tube_color(self, tube: MotionTube) -> ColorRGBA:
        if tube is self.selected_tube:
            return ColorRGBA(r=0.00, g=0.95, b=1.00, a=1.0)
        if not tube.is_feasible:
            return ColorRGBA(r=0.95, g=0.15, b=0.15, a=0.35)
        try:
            c = max(0.0, min(1.0, (tube.cost - self.min_cost) / (self.max_cost - self.min_cost + 1e-6)))
        except Exception:
            c = 0.5
        return ColorRGBA(r=float(1.0 * c), g=1.0, b=0.0, a=0.9)

    def _publish_tube_markers(self):
        if not self.motion_tubes or self.current_pose is None:
            return
        rx, ry, ryaw = self._pose_xy_yaw()
        cy, sy = math.cos(ryaw), math.sin(ryaw)

        frame = (self.current_pose.header.frame_id or 'odom')
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame

        ma = MarkerArray()

        wipe = Marker()
        wipe.header = header
        wipe.ns = 'motion_tubes'
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
            m.ns = 'motion_tubes'
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
            m.ns = 'motion_tubes'
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
        fwd = self.get_forward_clearance()
        self.get_logger().info(f'PLANNER: tubes={len(self.motion_tubes)} feas={len(feas)} fwd={fwd:.2f}m')


def main():
    rclpy.init()
    node = GoalOrientedMotionTubePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()