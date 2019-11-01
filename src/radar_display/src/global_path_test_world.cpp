#include <ros/ros.h> 
#include <std_msgs/Float64MultiArray.h> 
#include <std_msgs/Float64.h> 
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Point.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <gazebo_msgs/ModelStates.h>
#include <geometry_msgs/Twist.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;


/* Values about map and route */
const double points[16][2] = {{-15, -10.5}, {-6.5, -10.5}, {5, -10.5}, {14.5, -10.5}, 
                            {-15, -3.5}, {-6.5, -3.5}, {5, -3.5}, {14.5, -3.5},
                            {-15, 4.5}, {-6.5, 4.5}, {5, 3.25}, {14.5, 3.25}, 
                            {-15, 10.5}, {-6.5, 10.5}, {5, 10.5}, {14.5, 10.5}};

const double spawn_position[2] = {-15, -7.5};

// const int point_num = 14;
// const int route[point_num] ={4, 5, 1, 2, 3, 7, 6, 10, 11, 15, 14, 13, 9, 8}; 
const int point_num = 22;
const int route[point_num] ={4, 5, 1, 2, 10, 11, 7, 6, 14, 13, 9, 8, 12, 13, 5, 6, 2, 3, 15, 12, 0, 1}; 

const double close_dist = 1.8;  // 1.5


/* Variables */
double position[3]={0.0, 0.0, 0.0};
double angle[3] = {0.0, 0.0, 0.0};

ofstream path_file;
int write_interp = 5;
int write_flag = 0;

ofstream control_signal_file;

void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, cv::Scalar color, int thickness, int lineType)
{
    const double PI = 3.1415926;
    Point arrow;
    double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
    line(img, pStart, pEnd, color, thickness, lineType);
    arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);

    arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);

    line(img, pEnd, arrow, color, thickness, lineType);

    arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);

    arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);

    line(img, pEnd, arrow, color, thickness, lineType);
}


void groundtruthCallback(const gazebo_msgs::ModelStates& msg)
{
    int num_model = msg.name.size();
    if (std::strcmp(msg.name[num_model-1].c_str(),"mobile_base")){
        cout << "Attention! Got the wrong object from model_states!! not the mobile base!!"<<endl 
             << "Object got from model_states: " << msg.name[num_model-1].c_str() << endl;
    }

    position[0] = msg.pose[num_model-1].position.x;
    position[1] = msg.pose[num_model-1].position.y;
    position[2] = msg.pose[num_model-1].position.z;

    float q0 = msg.pose[num_model-1].orientation.x;
    float q1 = msg.pose[num_model-1].orientation.y;
    float q2 = msg.pose[num_model-1].orientation.z;
    float q3 = msg.pose[num_model-1].orientation.w;

    /* Pitch roll may be needed for MAVs */
    angle[2] = atan2(2*q3*q2 + 2*q0*q1, -2*q1*q1 - 2*q2*q2 + 1);  // Yaw

    write_flag ++;
    if(write_flag >= write_interp)
    {
        write_flag = 0;
        path_file<<position[0]<<","<<position[1]<<","<<endl;
    }

}

void controlSignalCallback(const geometry_msgs::Twist& msg)
{
    control_signal_file << msg.linear.x <<","<<msg.angular.z<<","<<endl;
}


int main(int argc, char **argv) 
{ 
    ros::init(argc,argv,"global_path_test_world"); 
    ros::NodeHandle n; 

    // ros::Subscriber yaw_sub= n.subscribe("/odom",1,odometryCallback); 
    ros::Subscriber yaw_sub= n.subscribe("/gazebo/model_states",1,groundtruthCallback);
    path_file.open("/home/ubuntu/chg_workspace/log/testing/path1.csv");
    ros::Subscriber control_signal_sub = n.subscribe("/mobile_base/commands/velocity", 1, controlSignalCallback);
    control_signal_file.open("/home/ubuntu/chg_workspace/log/testing/control1.csv");

    namedWindow( "Compass", CV_WINDOW_AUTOSIZE );
    namedWindow( "Command", CV_WINDOW_AUTOSIZE );

    ros::Publisher target_pos_pub = n.advertise<geometry_msgs::Point>("/radar/target_point", 1);  // Gloabl coordinate, not robot odom coord
    ros::Publisher current_pos_pub = n.advertise<geometry_msgs::Point>("/radar/current_point", 1); // Gloabl coordinate, not robot odom coord
    ros::Publisher target_yaw_pub = n.advertise<std_msgs::Float64>("/radar/target_yaw", 1); // Gloabl coordinate, same with robot odom coord
    ros::Publisher current_yaw_pub = n.advertise<std_msgs::Float64>("/radar/current_yaw", 1);
    ros::Publisher delt_yaw_pub = n.advertise<std_msgs::Float64>("/radar/delt_yaw", 1);
    ros::Publisher stop_pub = n.advertise<std_msgs::Float64>("/stop_recording", 1);
    ros::Publisher direction_pub = n.advertise<std_msgs::Float64MultiArray>("/radar/direction", 1);

    std_msgs::Float64 target_yaw;
    std_msgs::Float64 current_yaw;
    std_msgs::Float64 delt_yaw;
    std_msgs::Float64MultiArray direction;

    geometry_msgs::Point target_point;
    geometry_msgs::Point current_point;

    int route_point_counter = 0;

    double target_x = points[route[route_point_counter]][0];
    double target_y = points[route[route_point_counter]][1];

    ros::Rate loop_rate(20);

    while(ros::ok())
    {
     /* Close detection */
     double dist_x = sqrt((target_x - position[0])*(target_x - position[0]) + (target_y - position[1])*(target_y - position[1]));
     if(dist_x < close_dist) 
     {
         route_point_counter += 1;
         if(route_point_counter >= point_num)
         {
             std::cout<< " You achieved the target!! Mission completed!!" << std::endl;
             break;
         }

         target_x = points[route[route_point_counter]][0];
         target_y = points[route[route_point_counter]][1];
     }

     /* Calculate target yaw */
     double delt_x = target_x - position[0];
     double delt_y = target_y - position[1];

     double yaw_t = atan2(delt_y, delt_x);

     /* Calculate delt yaw */
     double delt_yaw_value = 0.0;

        double delt_yaw_direct = yaw_t - angle[2];
        double delt_yaw_direct_abs = std::fabs(delt_yaw_direct);
        double sup_yaw_direct_abs = 2*M_PI - delt_yaw_direct_abs;

        if(delt_yaw_direct_abs < sup_yaw_direct_abs)
            delt_yaw_value = delt_yaw_direct;
        else
            delt_yaw_value = - sup_yaw_direct_abs * delt_yaw_direct / delt_yaw_direct_abs;

        /* Calculate diraction */
        direction.data.clear();
        int command_type = 0;
        if(delt_yaw_value > -M_PI/6.0 && delt_yaw_value < M_PI/6.0)  // Move forward
        {
            direction.data.push_back(1.0);
            direction.data.push_back(0.0);
            direction.data.push_back(0.0);
            direction.data.push_back(0.0);
            command_type = 0;
        }
        else if(delt_yaw_value >= -5*M_PI/6.0 && delt_yaw_value <= -M_PI/6.0)  // Turn right
        {
            direction.data.push_back(0.0);
            direction.data.push_back(0.0);
            direction.data.push_back(0.0);
            direction.data.push_back(1.0);
            command_type = 3;
        }
        else if(delt_yaw_value >= M_PI/6.0 && delt_yaw_value <= 5*M_PI/6.0) // Turn left
        {
            direction.data.push_back(0.0);
            direction.data.push_back(0.0);
            direction.data.push_back(1.0);
            direction.data.push_back(0.0);
            command_type = 2;
        }
        else  // Move backward
        {
            direction.data.push_back(0.0);
            direction.data.push_back(1.0);
            direction.data.push_back(0.0);
            direction.data.push_back(0.0);
            command_type = 1;
        }
        direction_pub.publish(direction);

     /* Update and publish*/
     target_point.x = target_x;
     target_point.y = target_y;
     target_point.z = 0.0;
     target_pos_pub.publish(target_point);

     current_point.x = position[0];
     current_point.y = position[1];
     current_point.z = 0.0;
     current_pos_pub.publish(current_point);

     target_yaw.data = yaw_t;
     target_yaw_pub.publish(target_yaw);

     current_yaw.data = angle[2];
     current_yaw_pub.publish(current_yaw);

     delt_yaw.data = delt_yaw_value;
     delt_yaw_pub.publish(delt_yaw);


     /* Convert to body coordinate */
     double suggested_body_x = delt_x * cos(angle[2]) + delt_y * sin(angle[2]); 
     double suggested_body_y = -delt_x * sin(angle[2]) + delt_y * cos(angle[2]);

     /* Draw radar */
     Mat img(300, 300, CV_8UC3, Scalar(0,0,0));
     Point p(150, 150);
     circle(img, p, 60, Scalar(0, 255, 0), 10);
     circle(img, p, 5, Scalar(0, 0, 255), 3);
     line(img, Point(150, 270), Point(150, 30), Scalar(255, 20, 0), 3);
     line(img, Point(140, 40), Point(150, 30), Scalar(255, 20, 0), 3);
     line(img, Point(160, 40), Point(150, 30), Scalar(255, 20, 0), 3);

     Point p_dsr( -suggested_body_y* 140 + 150 , -suggested_body_x * 140 + 150);
     line(img, p, p_dsr, Scalar(0, 0, 255), 4);

     imshow("Compass", img);

     /* Draw command */
     Mat img2(300, 300, CV_8UC3, Scalar(0,0,0));
     Point pStart(150, 150);
     Point pEnd;
     if(command_type == 0){
         pEnd.x = 150;
         pEnd.y = 50;
     }
     else if(command_type == 1){
         pEnd.x = 150;
         pEnd.y = 250;
     }
     else if(command_type == 2){
         pEnd.x = 50;
         pEnd.y = 150;
     }
     else{
         pEnd.x = 250;
         pEnd.y = 150;
     }
     drawArrow(img2, pStart, pEnd, 25, 30, Scalar(0, 0, 255), 3, 4);     

     imshow("Command", img2);
     waitKey(5);


     ros::spinOnce(); 
     loop_rate.sleep();
    }

    std_msgs::Float64 stop_flag;
    stop_flag.data = 1.0;


    int flag_pub_counter = 0;
    while(ros::ok() && flag_pub_counter < 20)
    {
        flag_pub_counter ++;
        stop_pub.publish(stop_flag);
        ros::spinOnce(); 
        loop_rate.sleep();
    }
    
    return 0;
} 

