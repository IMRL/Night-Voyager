/* 
 * Night-Voyager: Consistent and Efficient Nocturnal Vision-Aided State Estimation in Object Maps
 * Copyright (C) 2025 Night-Voyager Contributors
 * 
 * For commercial use, please contact Tianxiao Gao at <ga0.tianxiao@connect.um.edu.mo>
 * or Mingle Zhao at <zhao.mingle@connect.um.edu.mo>
 * 
 * This file is subject to the terms and conditions outlined in the 'LICENSE' file,
 * which is included as part of this source code package.
 */
#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "core/CommonLib.h"
#include "core/NightVoyagerOptions.h"
#include "visualizer/CameraPoseVisualization.h"
#include <atomic>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <deque>
#include <fstream>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <image_transport/image_transport.h>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <thread>
#include <visualization_msgs/MarkerArray.h>

using namespace std;

namespace night_voyager {
class VILManager;
class CameraPoseVisualization;

class Visualizer {
  public:
    Visualizer(ros::NodeHandle &nh, const NightVoyagerOptions &options, shared_ptr<VILManager> sys);

    void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg);

    void img_cbk(const sensor_msgs::CompressedImage::ConstPtr &msg);

    void box_cbk(const night_voyager::BoundingBoxes::ConstPtr &msg);

    void odom_cbk(const nav_msgs::Odometry::ConstPtr &msg);

    void run();

    void visualize(const PackData &message, double time_cost);

    void publish_images(const PackData &message);

    void publish_detection(const PackData &message);

    void publish_images_tracking_recover(const PackData &message);

    void publish_other_images_tracking_recover(const PackData &message);

    void publish_state();

    void publish_hz(double time_cost);

    void publish_state_tracking_recover();

    void publish_features();

    void publish_odometry();

    void publish_cam_path();

    void publish_features_tracking_recover();

    void publish_prior_poses();

    void publish_initialization_regions();

    void publish_near_prior_poses();

    void publish_global_pointcloud();

    void publish_matched_pointcloud();

    void publish_all_possible_paths();

    void save_total_state_to_file();

    void save_total_state_to_file_tracking_recover();

    sensor_msgs::PointCloud2 get_ros_pointcloud_inMAP(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &feats);

    sensor_msgs::PointCloud2 get_ros_pointcloud_inMAP_tracking_recover(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &feats);

    double time_before_initialized = 0, time_after_initialized = 0;
    int n_before_initialized = 1, n_after_initialized = 1;

  private:
    shared_ptr<VILManager> app;
    mutex camera_queue_mtx;
    mutex odom_queue_mtx;
    mutex box_queue_mtx;
    // Thread atomics
    atomic<bool> thread_update_running;

    // For path viz
    unsigned int poses_seq_imu = 0;
    unsigned int poses_seq_vins = 0, poses_seq_open_vins = 0, poses_seq_night_rider = 0, poses_seq_gt = 0;
    unsigned int num_hz = 0;
    std::vector<geometry_msgs::PoseStamped> poses_imu;
    std::vector<geometry_msgs::PoseStamped> poses_vins, poses_open_vins, poses_night_rider, poses_gt;

    // deque<sensor_msgs::ImuConstPtr> imu_buffer;
    deque<ImuData> imu_buffer;
    deque<CameraData> img_buffer;
    deque<OdomData> odom_buffer;
    deque<BoxData> box_buffer;

    ros::Subscriber sub_img;
    ros::Subscriber sub_imu;
    ros::Subscriber sub_box;
    ros::Subscriber sub_odom;

    image_transport::Publisher it_pub_tracks, it_pub_loop_img_depth, it_pub_loop_img_depth_color, it_pub_streetlight_matches, it_pub_detection, it_pub_projection,
        it_pub_projection_tracking_recover, it_pub_hz;
    ros::Publisher pub_poseimu, pub_pathimu, pub_paths_tracking_recover;
    ros::Publisher pub_points_msckf, pub_points_slam, pub_points_streetlight;
    ros::Publisher pub_prior_map, pub_prior_poss, pub_prior_quats, pub_init_regions, pub_near_prior_cloud, pub_view_region_bound, pub_view_region_fill, pub_odometer;
    ros::Publisher pub_night_voyager_cam;

    bool save_total_state, save_time_consume;
    ofstream of_state_est, of_state_std, of_state_tum_loc, of_state_tum_global, of_state_loc_part, of_state_rel_part, of_state_global_part;

    ofstream of_state_est_tracking_recover, of_state_std_tracking_recover, of_state_tum_loc_tracking_recover, of_state_tum_global_tracking_recover;

    bool last_tracking_lost;

    double last_timestamp_imu;
    double last_timestamp_img;
    double last_timestamp_odom;
    double last_timestamp_box;

    double accel_norm;

    double last_visualization_timestamp, last_visualization_timestamp_image;

    visualization_msgs::MarkerArray marker_array;

    CameraPoseVisualization night_voyager_cam;
};
} // namespace night_voyager
#endif