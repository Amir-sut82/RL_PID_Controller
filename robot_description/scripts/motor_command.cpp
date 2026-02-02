#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <cmath>

class MotorCommandNode : public rclcpp::Node
{
public:
    MotorCommandNode()
    : Node("motor_command_node"),
      wheel_radius_(0.1),     
      wheel_separation_(0.46)  
    {
        // Publishers
        left_motor_rpm_pub_  = this->create_publisher<std_msgs::msg::Float64>("/left_motor_rpm", 10);
        right_motor_rpm_pub_ = this->create_publisher<std_msgs::msg::Float64>("/right_motor_rpm", 10);

        // Subscriber to /cmd_vel (Twist)
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel",
            10,
            std::bind(&MotorCommandNode::cmdVelCallback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "Motor Command Node started.");
    }

private:
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        double vx = msg->linear.x;     
        double wz = msg->angular.z;    

        RCLCPP_INFO(this->get_logger(), "Received cmd_vel → vx: %f , wz: %f", vx, wz);

        double left_motor_rad_per_sec  = (vx - (wheel_separation_ * wz) / 2.0) / wheel_radius_;
        double right_motor_rad_per_sec = (vx + (wheel_separation_ * wz) / 2.0) / wheel_radius_;

        RCLCPP_INFO(this->get_logger(), "Left rad/s: %f , Right rad/s: %f",
                    left_motor_rad_per_sec, right_motor_rad_per_sec);

        // rad/s to RPM conversion
        double left_motor_rpm  = left_motor_rad_per_sec * (60.0 / (2.0 * M_PI)) ;
        double right_motor_rpm = right_motor_rad_per_sec * (60.0 / (2.0 * M_PI)) ;
        // (60.0 / (2.0 * M_PI))

        RCLCPP_INFO(this->get_logger(), "Left RPM: %f , Right RPM: %f",
                    left_motor_rpm, right_motor_rpm);

        std_msgs::msg::Float64 left_msg;
        std_msgs::msg::Float64 right_msg;

        left_msg.data  = left_motor_rpm;
        right_msg.data = right_motor_rpm;

        left_motor_rpm_pub_->publish(left_msg);
        right_motor_rpm_pub_->publish(right_msg);
    }

    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_motor_rpm_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr right_motor_rpm_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

    double wheel_radius_;
    double wheel_separation_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MotorCommandNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
