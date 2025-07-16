import math

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class WallFollow(Node):
    """Implement Wall Following on the car."""

    def __init__(self):
        super().__init__('wall_follow_node')

        # --- PARAMETERS (tune these!) ---
        self.desired_dist = 1.0             # Desired distance to the wall (meters)
        self.theta = math.radians(45.0)     # Angle between beams (radians)
        self.lookahead_dist = 1.0           # Lookahead distance L (meters)

        # PID gains
        self.kp = 1.0
        self.ki = 0.0
        self.kd = 0.1

        # PID state
        self.integral = 0.0
        self.prev_error = 0.0

        # Topics
        scan_topic = '/scan'
        drive_topic = '/drive'

        # Subscriber & publisher
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10)

        self.get_logger().info('WallFollow node initialized')

    def get_range(self, scan: LaserScan, angle: float) -> float:
        """
        Return the LiDAR range at a given angle (in the car frame),
        handling NaNs and infinities.
        """
        # Compute the index into the ranges array
        idx = int((angle - scan.angle_min) / scan.angle_increment)
        idx = max(0, min(idx, len(scan.ranges) - 1))
        r = scan.ranges[idx]
        # Replace invalid readings with max range
        if not math.isfinite(r):
            return scan.range_max
        return r

    def get_error(self, scan: LaserScan) -> float:
        """
        Compute the wall‐following error:
          e = desired_dist − D_{t+1},
        where D_{t+1} = D_t + L * sin(alpha)
        and alpha = arctan((a cosθ − b) / (a sinθ)) with
          a = scan at (−90° + θ), b = scan at −90°.

        :contentReference[oaicite:0]{index=0}
        """
        # b: directly to the right (−90°)
        b = self.get_range(scan, -math.pi / 2)
        # a: beam at −90°+θ
        a = self.get_range(scan, -math.pi / 2 + self.theta)

        # Compute orientation α of car relative to wall
        alpha = math.atan2(
            a * math.cos(self.theta) - b,
            a * math.sin(self.theta)
        )
        # Current distance D_t
        d_t = b * math.cos(alpha)
        # Projected distance D_{t+1}
        d_t1 = d_t + self.lookahead_dist * math.sin(alpha)

        # Error is how far off we are from the set‐point
        return self.desired_dist - d_t1

    def pid_control(self, error: float):
        """
        Run a PID controller on the error to compute a steering angle,
        then choose speed based on that angle and publish.
        """
        # Integral and derivative terms
        self.integral += error
        derivative = error - self.prev_error

        # PID steering output
        angle = (self.kp * error +
                 self.ki * self.integral +
                 self.kd * derivative)

        self.prev_error = error

        # Speed scheduling based on steering angle
        # 0–10°: 1.5 m/s, 10–20°: 1.0 m/s, else 0.5 m/s :contentReference[oaicite:1]{index=1}
        abs_ang = abs(angle)
        if abs_ang < math.radians(10):
            speed = 1.5
        elif abs_ang < math.radians(20):
            speed = 1.0
        else:
            speed = 0.5

        # Publish drive command
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.drive.steering_angle = angle
        cmd.drive.speed = speed
        self.drive_pub.publish(cmd)

    def scan_callback(self, msg: LaserScan):
        """LaserScan callback: compute error and invoke PID."""
        err = self.get_error(msg)
        self.pid_control(err)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
