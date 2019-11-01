#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>
#include "boost/thread/mutex.hpp"
#include "boost/thread/thread.hpp"
#include "ros/console.h"

class TurtlebotTeleop
{
public:
  TurtlebotTeleop();

private:
  void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
  void publish();

  ros::NodeHandle ph_, nh_;

  int linear_, angular_, deadman_axis_;
  double l_scale_, a_scale_;
  ros::Publisher vel_pub_;
  ros::Subscriber joy_sub_;

  geometry_msgs::Twist last_published_;
  boost::mutex publish_mutex_;
  bool zero_twist_published_;
  ros::Timer timer_;

};

TurtlebotTeleop::TurtlebotTeleop():
  ph_("~"),
  linear_(4),
  angular_(0), // no use here
  l_scale_(0.8),
  a_scale_(0.4)
{
  ph_.param("axis_deadman", deadman_axis_, deadman_axis_);
  ph_.param("scale_angular", a_scale_, a_scale_);
  ph_.param("scale_linear", l_scale_, l_scale_);

  zero_twist_published_ = false;

  vel_pub_ = ph_.advertise<geometry_msgs::Twist>("/teleop_velocity_smoother/raw_cmd_vel", 1, true);
  joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("/joy", 10, &TurtlebotTeleop::joyCallback, this);

  timer_ = nh_.createTimer(ros::Duration(0.1), boost::bind(&TurtlebotTeleop::publish, this));
}

void TurtlebotTeleop::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{ 
  geometry_msgs::Twist vel;
  vel.angular.z = a_scale_*(joy->axes[5] - joy->axes[2]) / 2.0;
  vel.linear.x = l_scale_*joy->axes[linear_];
  // To disable moving backward
  //if(vel.linear.x < 0.f) vel.linear.x = 0.f;

  last_published_ = vel;
}

void TurtlebotTeleop::publish()
{
  boost::mutex::scoped_lock lock(publish_mutex_);

  vel_pub_.publish(last_published_);
  zero_twist_published_=false;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ps4_joy");
  TurtlebotTeleop turtlebot_teleop;

  ros::spin();
}

