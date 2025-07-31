#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include <cmath>
#include <algorithm>

class WallFollow : public rclcpp::Node {
public:
  WallFollow()
  : Node("wall_follow_node"),
    kp_(1.0),    // tune these gains
    kd_(0.1),
    ki_(0.01),
    servo_offset_(0.0),
    prev_error_(0.0),
    integral_(0.0)
  {
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", rclcpp::QoS(10),
      std::bind(&WallFollow::scan_callback, this, std::placeholders::_1)
    );

    drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      "/drive", rclcpp::QoS(10)
    );
  }

private:
  // PID terms
  double kp_, kd_, ki_, servo_offset_, prev_error_, integral_;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

  double get_range(
    const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
    double angle
  ) {
    int idx = int((angle - scan->angle_min) / scan->angle_increment);
    idx = std::clamp(idx, 0, int(scan->ranges.size()) - 1);
    double r = scan->ranges[idx];
    if (std::isnan(r) || std::isinf(r)) {
      r = scan->range_max;
    }
    return r;
  }

  double compute_error(
    const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
    double desired_dist
  ) {
    // measure at 90° and 45° to the left
    const double theta = M_PI / 4.0;
    double a = get_range(scan, M_PI/2.0 - theta);
    double b = get_range(scan, M_PI/2.0);

    // wall angle and perpendicular distance
    double alpha = std::atan((a * std::cos(theta) - b) / (a * std::sin(theta)));
    double curr_dist = b * std::cos(alpha);

    return curr_dist - desired_dist;
  }

  void pid_control(double error, double speed) {
    integral_  += error;
    double derivative = error - prev_error_;
    double output = kp_ * error + ki_ * integral_ + kd_ * derivative;
    prev_error_ = error;

    double steer = -output + servo_offset_;
    // clamp to ±24°
    steer = std::clamp(steer, -0.418, 0.418);

    auto msg = ackermann_msgs::msg::AckermannDriveStamped();
    msg.drive.speed          = speed;
    msg.drive.steering_angle = steer;
    drive_pub_->publish(msg);
  }

  void scan_callback(
    const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg
  ) {
    const double desired_distance = 1.0;  // meters from wall
    double error = compute_error(scan_msg, desired_distance);
    double speed = 1.0;                   // constant forward speed
    pid_control(error, speed);
  }
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WallFollow>());
  rclcpp::shutdown();
  return 0;
}
