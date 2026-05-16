#!/usr/bin/env python3
#
# Copyright 2026 - TurtleBot3 Yellow Lane Follower
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32, Int8


class LaneControllerNode(Node):
    """
    PID controller that keeps the TurtleBot3 centred between two yellow lanes.

    Subscribes to /lane/center_error (std_msgs/Float32):
        0.0  → robot is centred
        +1.0 → robot is at the far right of the lane  → turn left
        -1.0 → robot is at the far left of the lane   → turn right
        NaN  → lane completely lost                   → stop immediately

    Publishes to /cmd_vel (geometry_msgs/Twist).

    PID features
    ------------
    - Actual dt from ROS clock (not assumed constant)
    - Anti-windup clamp on the integral term (ki_max)
    - Motor dead-zone: suppresses angular commands too small to move the
      hardware, avoiding PWM buzz at near-zero corrections
    - Quadratic speed reduction: robot slows much more aggressively in
      tight turns than a linear formula would
    - Safety timeout: publishes a zero Twist if no valid error is received
      for safety_timeout seconds (e.g. node crash, WiFi dropout)

    Parameters (see param/lane_controller.yaml for defaults and tuning notes)
    ----------
    kp, ki, kd, ki_max,
    base_speed, max_angular_vel, speed_reduction_factor,
    angular_deadzone, safety_timeout
    """

    def __init__(self):
        super().__init__('lane_controller_node')

        # ── parameters ────────────────────────────────────────────────────
        self.declare_parameter('kp', 1.2)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.3)
        self.declare_parameter('ki_max', 0.5)
        self.declare_parameter('base_speed', 0.10)
        self.declare_parameter('max_angular_vel', 1.5)
        self.declare_parameter('speed_reduction_factor', 1.5)
        self.declare_parameter('angular_deadzone', 0.05)
        self.declare_parameter('safety_timeout', 2.0)

        # Forced left-turn parameters (special track section)
        self.declare_parameter('turn_left_speed',    0.05)   # m/s forward
        self.declare_parameter('turn_left_angular',  1.0)    # rad/s (positive=left)
        self.declare_parameter('turn_left_duration', 2.0)    # seconds

        # NaN-coast: when the lane detector publishes NaN (lane lost), keep
        # turning in the last known direction for this many seconds at
        # `nan_coast_speed` instead of stopping immediately. Prevents the
        # robot from giving up mid-corner.
        self.declare_parameter('nan_coast_duration', 1.5)
        self.declare_parameter('nan_coast_speed', 0.05)
        # Minimum magnitude of angular_z used during a coast when the last
        # PID output was tiny — biases the robot to keep curving.
        self.declare_parameter('nan_coast_min_angular', 0.6)

        # Parking: when zone_state=PARKING, first STEER toward the red blob's
        # centroid for `parking_align_duration` seconds at low speed, then
        # decelerate to zero over `parking_brake_duration` seconds.
        self.declare_parameter('parking_align_duration', 0.8)
        self.declare_parameter('parking_align_speed', 0.05)
        self.declare_parameter('parking_align_kp', 1.5)
        self.declare_parameter('parking_brake_duration', 1.0)

        self._read_parameters()

        # ── PID state ─────────────────────────────────────────────────────
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.last_msg_time = None

        # Last valid command sent — used for NaN-coast and gentle stops.
        self.last_angular_z = 0.0
        self.last_linear_x  = 0.0

        # NaN-coast state
        self._nan_coast_start: rclpy.time.Time | None = None

        # Parking state
        self._parking = False
        self._parking_start: rclpy.time.Time | None = None
        # Latest normalised red-marker centroid x ([-1,1] or NaN if unseen)
        self._red_centroid_x: float = float('nan')

        # ── Turn-left state ───────────────────────────────────────────────
        self._turning = False        # True while executing forced left turn
        self._turn_start: rclpy.time.Time | None = None

        # ── publishers / subscribers ──────────────────────────────────────
        qos = QoSProfile(depth=10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

        self.error_sub = self.create_subscription(
            Float32,
            '/lane/center_error',
            self.error_callback,
            qos,
        )

        self.zone_sub = self.create_subscription(
            Int8,
            '/lane/zone_state',
            self.zone_state_callback,
            qos,
        )

        self.red_centroid_sub = self.create_subscription(
            Float32,
            '/lane/red_centroid',
            self.red_centroid_callback,
            qos,
        )

        # Safety watchdog fires every 0.5 s
        self.safety_timer = self.create_timer(0.5, self.safety_timer_callback)

        self.get_logger().info('LaneControllerNode started')
        self.get_logger().info(
            f'  PID: Kp={self.kp}  Ki={self.ki}  Kd={self.kd}'
        )
        self.get_logger().info(
            f'  base_speed={self.base_speed} m/s  '
            f'max_angular={self.max_angular_vel} rad/s'
        )

    # ─────────────────────────────────────────────────────────────────────
    # Parameter helpers
    # ─────────────────────────────────────────────────────────────────────

    def _read_parameters(self):
        self.kp = float(self.get_parameter('kp').value)
        self.ki = float(self.get_parameter('ki').value)
        self.kd = float(self.get_parameter('kd').value)
        self.ki_max = float(self.get_parameter('ki_max').value)
        self.base_speed = float(self.get_parameter('base_speed').value)
        self.max_angular_vel = float(self.get_parameter('max_angular_vel').value)
        self.speed_reduction_factor = float(
            self.get_parameter('speed_reduction_factor').value)
        self.angular_deadzone = float(self.get_parameter('angular_deadzone').value)
        self.safety_timeout = float(self.get_parameter('safety_timeout').value)
        self.turn_left_speed    = float(self.get_parameter('turn_left_speed').value)
        self.turn_left_angular  = float(self.get_parameter('turn_left_angular').value)
        self.turn_left_duration = float(self.get_parameter('turn_left_duration').value)
        self.nan_coast_duration    = float(self.get_parameter('nan_coast_duration').value)
        self.nan_coast_speed       = float(self.get_parameter('nan_coast_speed').value)
        self.nan_coast_min_angular = float(self.get_parameter('nan_coast_min_angular').value)
        self.parking_align_duration = float(
            self.get_parameter('parking_align_duration').value)
        self.parking_align_speed = float(
            self.get_parameter('parking_align_speed').value)
        self.parking_align_kp = float(
            self.get_parameter('parking_align_kp').value)
        self.parking_brake_duration = float(
            self.get_parameter('parking_brake_duration').value)

    # ─────────────────────────────────────────────────────────────────────
    # Error callback
    # ─────────────────────────────────────────────────────────────────────

    def error_callback(self, msg: Float32):
        now = self.get_clock().now()
        self.last_msg_time = now

        # Parking has highest priority — brake to zero, then hold.
        if self._parking:
            self._drive_parking_brake(now)
            return

        # During a forced left turn, ignore the error signal
        if self._turning:
            elapsed = (now - self._turn_start).nanoseconds * 1e-9
            if elapsed < self.turn_left_duration:
                self._publish_twist(self.turn_left_speed, self.turn_left_angular)
                return
            else:
                # Turn complete — resume PID
                self._turning = False
                self._reset_pid()
                self.get_logger().info('Left turn complete — resuming PID')

        error = float(msg.data)

        # NaN → lane lost. Coast in the last known direction for a short
        # window before giving up. This stops the robot from freezing mid
        # corner / on brief blind spots.
        if math.isnan(error):
            if self._nan_coast_start is None:
                self._nan_coast_start = now
            elapsed = (now - self._nan_coast_start).nanoseconds * 1e-9
            if elapsed < self.nan_coast_duration:
                # Bias angular toward the last commanded sign with a floor —
                # tiny last_angular_z would otherwise just stop the robot
                # straight into whatever wall it lost the lane next to.
                sign = 1.0 if self.last_angular_z >= 0.0 else -1.0
                a = self.last_angular_z
                if abs(a) < self.nan_coast_min_angular:
                    a = sign * self.nan_coast_min_angular
                a = max(-self.max_angular_vel, min(self.max_angular_vel, a))
                self._publish_twist(self.nan_coast_speed, a)
                self.get_logger().warn(
                    f'Lane lost — coasting ({elapsed:.2f}s, ω={a:+.2f})',
                    throttle_duration_sec=1.0,
                )
                return
            # Coast window expired — full stop.
            self._publish_stop()
            self.get_logger().warn(
                'Lane lost (NaN) — coast expired, robot stopped',
                throttle_duration_sec=2.0,
            )
            self._reset_pid()
            return

        # Valid error received → reset the NaN-coast timer.
        self._nan_coast_start = None

        # ── dt ────────────────────────────────────────────────────────────
        if self.prev_time is None:
            dt = 0.05   # assume 20 Hz on first tick
        else:
            dt = (now - self.prev_time).nanoseconds * 1e-9
            # Guard: reject implausible dt values
            if dt <= 0.0 or dt > 1.0:
                dt = 0.05
        self.prev_time = now

        # ── PID computation ───────────────────────────────────────────────
        self.integral += error * dt
        # Anti-windup: clamp integral so accumulated offset doesn't overshoot
        self.integral = max(-self.ki_max, min(self.ki_max, self.integral))

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        # Positive error → robot too far right → need negative angular_z (turn left)
        angular_z = -(
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        # Clamp to hardware-safe range
        angular_z = max(-self.max_angular_vel,
                        min(self.max_angular_vel, angular_z))

        # Motor dead-zone: very small corrections are below the motor's
        # minimum effective PWM and just cause buzz without movement
        if 0.0 < abs(angular_z) < self.angular_deadzone:
            angular_z = 0.0

        # ── Speed reduction on curves (quadratic) ────────────────────────
        # error² grows much faster than |error|, so the robot slows
        # aggressively in tight turns while barely slowing on gentle drifts.
        speed_scale = max(0.3, 1.0 - (error ** 2) * self.speed_reduction_factor)
        linear_x = self.base_speed * speed_scale

        self._publish_twist(linear_x, angular_z)

    # ─────────────────────────────────────────────────────────────────────
    # Safety watchdog
    # ─────────────────────────────────────────────────────────────────────

    def safety_timer_callback(self):
        """Stop the robot if the lane detector has gone silent."""
        if self.last_msg_time is None:
            # No message ever received – stay stopped
            self._publish_stop()
            return

        now = self.get_clock().now()
        elapsed = (now - self.last_msg_time).nanoseconds * 1e-9

        if elapsed > self.safety_timeout:
            self._publish_stop()
            self.get_logger().warn(
                f'No lane error for {elapsed:.1f}s – safety stop',
                throttle_duration_sec=5.0,
            )
            self._reset_pid()

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _publish_twist(self, linear_x: float, angular_z: float):
        twist = Twist()
        twist.linear.x  = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)
        self.last_linear_x  = float(linear_x)
        self.last_angular_z = float(angular_z)

    def _publish_stop(self):
        self.cmd_vel_pub.publish(Twist())
        self.last_linear_x  = 0.0
        self.last_angular_z = 0.0

    def _drive_parking_brake(self, now):
        """
        Two-phase parking:
          1. Align — steer toward the red marker's centroid at low speed
             so we end up centred laterally on it.
          2. Brake — linearly decay linear_x to zero.
        """
        elapsed = (now - self._parking_start).nanoseconds * 1e-9

        # ── Phase 1: align toward red centroid ───────────────────────────
        if elapsed < self.parking_align_duration:
            cx = self._red_centroid_x
            if math.isnan(cx):
                # Red briefly out of view — crawl straight at align speed
                # until either it returns or we move to braking.
                self._publish_twist(self.parking_align_speed, 0.0)
                return
            # cx > 0 → marker to the right → turn right → negative angular_z
            angular_z = -self.parking_align_kp * cx
            angular_z = max(-self.max_angular_vel,
                            min(self.max_angular_vel, angular_z))
            self._publish_twist(self.parking_align_speed, angular_z)
            return

        # ── Phase 2: brake ───────────────────────────────────────────────
        brake_elapsed = elapsed - self.parking_align_duration
        if brake_elapsed >= self.parking_brake_duration:
            self._publish_stop()
            return
        ratio = 1.0 - (brake_elapsed / self.parking_brake_duration)
        linear_x = self.parking_align_speed * ratio
        self._publish_twist(linear_x, 0.0)

    def red_centroid_callback(self, msg: Float32):
        self._red_centroid_x = float(msg.data)

    def _reset_pid(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    # ─────────────────────────────────────────────────────────────────────
    # Zone state callback
    # ─────────────────────────────────────────────────────────────────────

    def zone_state_callback(self, msg: Int8):
        NORMAL    = 0
        TURN_LEFT = 2
        PARKING   = 3
        if msg.data == PARKING and not self._parking:
            self._parking = True
            self._parking_start = self.get_clock().now()
            self._turning = False
            self._reset_pid()
            self.get_logger().info(
                f'PARKING received — braking over {self.parking_brake_duration}s')
            return
        if msg.data == TURN_LEFT and not self._turning and not self._parking:
            self._turning = True
            self._turn_start = self.get_clock().now()
            self._reset_pid()
            self.get_logger().info(
                f'TURN_LEFT received — turning until yellow returns '
                f'(safety timeout {self.turn_left_duration}s)')
        elif msg.data == NORMAL and self._turning:
            # Detector saw yellow again → close the loop, exit the forced turn
            # immediately instead of waiting for the open-loop timeout.
            self._turning = False
            self._reset_pid()
            self.get_logger().info('Turn closed-loop: yellow re-acquired, resuming PID')


def main(args=None):
    rclpy.init(args=args)
    node = LaneControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
