/* 
 * Night-Voyager: Consistent and Efficient Nocturnal Vision-Aided State Estimation in Object Maps
 * Copyright (C) 2025 Night-Voyager Contributors
 * 
 * For technical issues and support, please contact Tianxiao Gao at <ga0.tianxiao@connect.um.edu.mo>
 * or Mingle Zhao at <zhao.mingle@connect.um.edu.mo>. For commercial use, please contact Prof. Hui Kong at <huikong@um.edu.mo>.
 * 
 * This file is subject to the terms and conditions outlined in the 'LICENSE' file,
 * which is included as part of this source code package.
 */
#include "visualizer/Visualizer.h"
#include "msckf_iekf/State.h"
#include "msckf_iekf/UpdaterMSCKF.h"
#include "msckf_iekf/UpdaterSLAM.h"
#include "msckf_iekf/VILManager.h"
#include "prior_pose/PriorPoseManager.h"
#include "streetlight_matcher/PcdManager.h"
#include <cv_bridge/cv_bridge.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <random>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

namespace night_voyager {
Visualizer::Visualizer(ros::NodeHandle &nh, const NightVoyagerOptions &options, shared_ptr<VILManager> sys) : app(sys) {
    sub_img = nh.subscribe(options.img_topic, 1000, &Visualizer::img_cbk, this);
    sub_imu = nh.subscribe(options.imu_topic, 1000, &Visualizer::imu_cbk, this);
    sub_odom = nh.subscribe(options.odom_topic, 1000, &Visualizer::odom_cbk, this);
    sub_box = nh.subscribe(options.box_topic, 1000, &Visualizer::box_cbk, this);

    // Create image transport
    image_transport::ImageTransport it(nh);

    // Our tracking image
    it_pub_tracks = it.advertise("trackhist", 2);
    it_pub_streetlight_matches = it.advertise("streetlight_matches", 2);
    it_pub_detection = it.advertise("streetlight_detction_dlbi", 2);
    it_pub_projection = it.advertise("streetlight_projection", 2);
    it_pub_projection_tracking_recover = it.advertise("tracking_recover_projection", 2);
    it_pub_hz = it.advertise("frequency", 2);

    pub_poseimu = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("poseimu", 2);
    pub_pathimu = nh.advertise<nav_msgs::Path>("pathimu", 2);
    pub_paths_tracking_recover = nh.advertise<visualization_msgs::MarkerArray>("tracking_recover_paths", 100);

    pub_points_msckf = nh.advertise<sensor_msgs::PointCloud2>("points_msckf", 2);
    pub_points_slam = nh.advertise<sensor_msgs::PointCloud2>("points_slam", 2);
    pub_points_streetlight = nh.advertise<sensor_msgs::PointCloud2>("points_streetlight", 1000);
    pub_prior_map = nh.advertise<sensor_msgs::PointCloud2>("prior_map", 1000);
    pub_prior_poss = nh.advertise<visualization_msgs::MarkerArray>("prior_poss", 10000);
    pub_init_regions = nh.advertise<visualization_msgs::MarkerArray>("init_regions", 10000);
    pub_view_region_bound = nh.advertise<visualization_msgs::Marker>("view_bound_region", 10000);
    pub_view_region_fill = nh.advertise<visualization_msgs::Marker>("view_fill_region", 10000);
    pub_odometer = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_near_prior_cloud = nh.advertise<sensor_msgs::PointCloud2>("near_prior_cloud", 10);

    pub_night_voyager_cam = nh.advertise<visualization_msgs::MarkerArray>("/night_voyager_cam", 100000);
    night_voyager_cam.setRGBA(0, 0.67, 1, 1);

    save_total_state = options.save_total_state;
    save_time_consume = options.save_time_consume;
    if (options.save_total_state) {
        cout << endl << options.of_state_est << endl << options.of_state_std << endl << options.of_state_tum_loc << endl << options.of_state_tum_global << endl;
        // If it exists, then delete it
        if (boost::filesystem::exists(options.of_state_est))
            boost::filesystem::remove(options.of_state_est);
        if (boost::filesystem::exists(options.of_state_std))
            boost::filesystem::remove(options.of_state_std);
        if (boost::filesystem::exists(options.of_state_tum_loc))
            boost::filesystem::remove(options.of_state_tum_loc);
        if (boost::filesystem::exists(options.of_state_tum_global))
            boost::filesystem::remove(options.of_state_tum_global);

        boost::filesystem::path last_level = boost::filesystem::path(options.of_state_est.c_str()).filename();
        last_level = "tracking_recover_" + last_level.string();
        boost::filesystem::path of_state_est_tracking_recover_path = boost::filesystem::path(options.of_state_tracking_recover.c_str()) / last_level;
        if (boost::filesystem::exists(of_state_est_tracking_recover_path)) {
            boost::filesystem::remove(of_state_est_tracking_recover_path);
        }

        last_level = boost::filesystem::path(options.of_state_std.c_str()).filename();
        last_level = "tracking_recover_" + last_level.string();
        boost::filesystem::path of_state_std_tracking_recover_path = boost::filesystem::path(options.of_state_tracking_recover.c_str()) / last_level;
        if (boost::filesystem::exists(of_state_std_tracking_recover_path)) {
            boost::filesystem::remove(of_state_std_tracking_recover_path);
        }

        last_level = boost::filesystem::path(options.of_state_tum_loc.c_str()).filename();
        last_level = "tracking_recover_" + last_level.string();
        boost::filesystem::path of_state_tum_loc_tracking_recover_path = boost::filesystem::path(options.of_state_tracking_recover.c_str()) / last_level;
        if (boost::filesystem::exists(of_state_tum_loc_tracking_recover_path)) {
            boost::filesystem::remove(of_state_tum_loc_tracking_recover_path);
        }

        last_level = boost::filesystem::path(options.of_state_tum_global.c_str()).filename();
        last_level = "tracking_recover_" + last_level.string();
        boost::filesystem::path of_state_tum_global_tracking_recover_path = boost::filesystem::path(options.of_state_tracking_recover.c_str()) / last_level;
        if (boost::filesystem::exists(of_state_tum_global_tracking_recover_path)) {
            boost::filesystem::remove(of_state_tum_global_tracking_recover_path);
        }

        // Create folder path to this location if not exists
        boost::filesystem::create_directories(boost::filesystem::path(options.of_state_est.c_str()).parent_path());
        boost::filesystem::create_directories(boost::filesystem::path(options.of_state_std.c_str()).parent_path());
        boost::filesystem::create_directories(boost::filesystem::path(options.of_state_tum_loc.c_str()).parent_path());
        boost::filesystem::create_directories(boost::filesystem::path(options.of_state_tum_global.c_str()).parent_path());
        boost::filesystem::create_directories(boost::filesystem::path(options.of_state_tracking_recover.c_str()));

        // Open the files
        of_state_est.open(options.of_state_est.c_str());
        of_state_std.open(options.of_state_std.c_str());
        of_state_tum_loc.open(options.of_state_tum_loc.c_str());
        of_state_tum_global.open(options.of_state_tum_global.c_str());
        of_state_est_tracking_recover.open(of_state_est_tracking_recover_path.c_str());
        of_state_std_tracking_recover.open(of_state_std_tracking_recover_path.c_str());
        of_state_tum_loc_tracking_recover.open(of_state_tum_loc_tracking_recover_path.c_str());
        of_state_tum_global_tracking_recover.open(of_state_tum_global_tracking_recover_path.c_str());
        of_state_est << "# timestamp(s) q p v bg ba q_MAPtoLOC t_MAPtoLOC cam_imu_dt cam_k cam_d cam_rot cam_trans" << std::endl;
        of_state_std << "# timestamp(s) q p v bg ba q_MAPtoLOC t_MAPtoLOC cam_imu_dt cam_k cam_d cam_rot cam_trans" << std::endl;
    }

    last_timestamp_img = 0.0;
    last_timestamp_imu = 0.0;
    last_timestamp_odom = 0.0;
    last_timestamp_box = 0.0;

    accel_norm = options.accel_norm;

    last_tracking_lost = false;
}

void Visualizer::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg) {
    double timestamp = msg->header.stamp.toSec();
    if (timestamp < last_timestamp_imu) {
        ROS_WARN("imu loop back");
        std::exit(EXIT_FAILURE);
        return;
    }
    last_timestamp_imu = timestamp;

    ImuData message;
    message.timestamp = msg->header.stamp.toSec();
    message.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    message.am << msg->linear_acceleration.x * accel_norm, msg->linear_acceleration.y * accel_norm, msg->linear_acceleration.z * accel_norm;

    imu_buffer.push_back(message);
}

void Visualizer::img_cbk(const sensor_msgs::CompressedImage::ConstPtr &msg) {
    if (msg->header.stamp.toSec() < last_timestamp_img) {
        ROS_ERROR("img loop back, clear buffer");
        std::exit(EXIT_FAILURE);
        return;
    }
    last_timestamp_img = msg->header.stamp.toSec();

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("fail to load img_msg");
        std::exit(EXIT_FAILURE);
        return;
    }

    CameraData message;
    message.timestamp = cv_ptr->header.stamp.toSec();
    message.image = cv_ptr->image;
    message.color_image = cv_ptr->image;

    // append it to our queue of images
    img_buffer.push_back(message);
}

void Visualizer::odom_cbk(const nav_msgs::Odometry::ConstPtr &msg) {

    double timestamp = msg->header.stamp.toSec();
    if (timestamp < last_timestamp_odom) {
        ROS_WARN("odom loop back, clear buffer");
        std::exit(EXIT_FAILURE);
        return;
    }
    last_timestamp_odom = timestamp;

    OdomData message;
    message.timestamp = timestamp;
    message.vm = Eigen::Vector3d(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);

    odom_buffer.push_back(message);
}

void Visualizer::box_cbk(const night_voyager::BoundingBoxes::ConstPtr &msg) {

    double timestamp = msg->header.stamp.toSec();
    if (timestamp < last_timestamp_box) {
        ROS_WARN("box loop back, clear buffer");
        std::exit(EXIT_FAILURE);
        return;
    }
    last_timestamp_box = timestamp;

    BoxData message;
    message.timestamp = timestamp;
    for (const auto box : msg->bounding_boxes) {
        cv::Rect rect(cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax));
        message.rects.push_back(rect);
    }

    app->feed_measurement_box(message);
}

void Visualizer::run() {

    // First we need to synchronize and package data
    if (img_buffer.empty() || imu_buffer.empty() || odom_buffer.empty()) {
        return;
    }

    double timestamp = img_buffer.front().timestamp;
    // We need to ensure at lease one IMU frame after camera for interpolation
    if (imu_buffer.back().timestamp < timestamp) {
        return;
    }

    PackData pack;
    while ((!imu_buffer.empty()) && (imu_buffer.front().timestamp < timestamp)) {
        pack.imus.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }
    // We also save the one IMU frame after camera
    while (pack.imus.empty() || pack.imus.back().timestamp <= timestamp) {
        if (!imu_buffer.empty()) {
            pack.imus.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }
    }
    assert(pack.imus.back().timestamp > timestamp);

    while ((!odom_buffer.empty()) && (odom_buffer.front().timestamp < timestamp)) {
        pack.odoms.push_back(odom_buffer.front());
        odom_buffer.pop_front();
        if (odom_buffer.empty())
            break;
    }

    // Finally save and process the image
    cv::cvtColor(img_buffer.front().image, img_buffer.front().image, cv::COLOR_BGR2GRAY);
    pack.cam = img_buffer.front();
    img_buffer.pop_front();

    // Feed the packed data into our VILManager
    auto rT1 = boost::posix_time::microsec_clock::local_time();
    app->feed_measurement_all(pack);
    auto rT2 = boost::posix_time::microsec_clock::local_time();

    if (app->initialized_map()) {
        double time_cost = (rT2 - rT1).total_microseconds() * 1e-3;
        time_after_initialized += (time_cost - time_after_initialized) / n_after_initialized;
        ++n_after_initialized;
        visualize(pack, time_cost * 1e-3);
    } else {
        double time_cost = (rT2 - rT1).total_microseconds() * 1e-3;
        time_before_initialized += (time_cost - time_before_initialized) / n_before_initialized;
        ++n_before_initialized;
        visualize(pack, time_cost * 1e-3);
    }
    PRINT_INFO("%.4f seconds to run Night-Voyager before initialization, %.4f seconds to run Night-Voyager after initialization\n", time_before_initialized,
               time_after_initialized);
}

void Visualizer::visualize(const PackData &message, double time_cost) {

    // Publish prior map and prior poses
    publish_global_pointcloud();
    publish_prior_poses();

    if (!app->is_tracking_lost()) {
        if (!marker_array.markers.empty())
            marker_array.markers.clear();
        // Return if we have already visualized
        if (last_visualization_timestamp == app->get_state()->_timestamp && app->initialized_imu()) {
            PRINT_INFO(YELLOW "Last timestamp equals to current timestamp. There might be something wrong.\n" RESET);
            return;
        }
        last_visualization_timestamp = app->get_state()->_timestamp;

        // Publish matched pointcloud and near prior poses
        if (app->initialized_map()) {
            publish_matched_pointcloud();
            publish_near_prior_poses();
        }

        publish_images(message);
        publish_detection(message);
        // publish_hz(time_cost);

        // Return if we have not inited
        if (!app->initialized_imu() || !app->initialized_map())
            return;

        // publish state
        publish_state();

        // publish camera markarray
        publish_cam_path();

        // publish points
        publish_features();

        publish_odometry();

        if (save_total_state) {
            save_total_state_to_file();
        }

        last_tracking_lost = false;
    }

    else {
        last_visualization_timestamp = app->get_state_tracking_recover()->_timestamp;

        publish_matched_pointcloud();
        publish_near_prior_poses();

        publish_all_possible_paths();

        publish_images_tracking_recover(message);

        publish_other_images_tracking_recover(message);

        // publish_state_tracking_recover();

        publish_features_tracking_recover();

        if (save_total_state) {
            save_total_state_to_file_tracking_recover();
        }

        last_tracking_lost = true;
    }
}

void Visualizer::publish_hz(double time_cost) {
    // Check if we have subscribers
    ++num_hz;
    if (num_hz != 30)
        return;
    else
        num_hz = 0;
    if (it_pub_hz.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for feature image.\n");
        return;
    }

    // Get our image of history tracks
    cv::Mat colorImage(640, 1920, CV_8UC3, cv::Scalar(255, 255, 255));
    string text = "FPS: " + to_string(int(1.0f / time_cost));
    cv::putText(colorImage, text, cv::Point(0, colorImage.rows / 2), cv::FONT_HERSHEY_DUPLEX, 10.0, cv::Scalar(0, 0, 0), 13);

    // Create our message
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "cam0";
    sensor_msgs::ImagePtr hz_msg = cv_bridge::CvImage(header, "bgr8", colorImage).toImageMsg();

    // Publish
    it_pub_hz.publish(hz_msg);
}

void Visualizer::save_total_state_to_file() {

    if (last_tracking_lost) {
        std::map<double, Eigen::Matrix3d> rots_loc = app->get_rots_loc_in_tracking_lost();
        std::map<double, Eigen::Vector3d> poss_loc = app->get_poss_loc_in_tracking_lost();
        std::map<double, Eigen::Matrix3d> rots_global = app->get_rots_global_in_tracking_lost();
        std::map<double, Eigen::Vector3d> poss_global = app->get_poss_global_in_tracking_lost();

        for (const auto &pair : rots_loc) {

            Eigen::Matrix3d R_IitoMAP = rots_global[pair.first].transpose() * pair.second;
            Eigen::Vector3d p_IinG = poss_loc[pair.first];
            Eigen::Vector3d p_IiinMAP = rots_global[pair.first].transpose() * (p_IinG - poss_global[pair.first]);
            Eigen::Quaterniond ori_IitoG = Eigen::Quaterniond(pair.second);
            Eigen::Quaterniond ori_IitoMAP = Eigen::Quaterniond(R_IitoMAP);

            of_state_tum_loc.precision(10);
            of_state_tum_loc.setf(std::ios::fixed, std::ios::floatfield);
            of_state_tum_loc << pair.first << " ";
            of_state_tum_loc.precision(6);
            of_state_tum_loc << p_IinG.x() << " " << p_IinG.y() << " " << p_IinG.z() << " ";
            of_state_tum_loc << ori_IitoG.x() << " " << ori_IitoG.y() << " " << ori_IitoG.z() << " " << ori_IitoG.w();

            of_state_tum_global.precision(10);
            of_state_tum_global.setf(std::ios::fixed, std::ios::floatfield);
            of_state_tum_global << pair.first << " ";
            of_state_tum_global.precision(6);
            of_state_tum_global << p_IiinMAP.x() << " " << p_IiinMAP.y() << " " << p_IiinMAP.z() << " ";
            of_state_tum_global << ori_IitoMAP.x() << " " << ori_IitoMAP.y() << " " << ori_IitoMAP.z() << " " << ori_IitoMAP.w();

            of_state_tum_loc << endl;
            of_state_tum_global << endl;
        }
    }

    auto state = app->get_state();
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double timestamp_inI = state->_timestamp;

    // Get the covariance of the whole system
    Eigen::MatrixXd cov = StateHelper::get_full_covariance(state);

    // STATE: Write the current state to file
    of_state_est.precision(5);
    of_state_est.setf(std::ios::fixed, std::ios::floatfield);
    of_state_est << timestamp_inI << " ";
    of_state_est.precision(6);
    of_state_est << state->_imu->quat().x() << " " << state->_imu->quat().y() << " " << state->_imu->quat().z() << " " << state->_imu->quat().w() << " ";
    of_state_est << state->_imu->pos()(0) << " " << state->_imu->pos()(1) << " " << state->_imu->pos()(2) << " ";
    of_state_est << state->_imu->vel()(0) << " " << state->_imu->vel()(1) << " " << state->_imu->vel()(2) << " ";
    of_state_est << state->_imu->bias_g()(0) << " " << state->_imu->bias_g()(1) << " " << state->_imu->bias_g()(2) << " ";
    of_state_est << state->_imu->bias_a()(0) << " " << state->_imu->bias_a()(1) << " " << state->_imu->bias_a()(2) << " ";
    if (app->initialized_map()) {
        of_state_est << state->_pose_MAPtoLOC->quat().x() << " " << state->_pose_MAPtoLOC->quat().y() << " " << state->_pose_MAPtoLOC->quat().z() << " "
                     << state->_pose_MAPtoLOC->quat().w() << " ";
        of_state_est << state->_pose_MAPtoLOC->pos()(0) << " " << state->_pose_MAPtoLOC->pos()(1) << " " << state->_pose_MAPtoLOC->pos()(2) << " ";
    }

    of_state_tum_loc.precision(10);
    of_state_tum_loc.setf(std::ios::fixed, std::ios::floatfield);
    of_state_tum_loc << timestamp_inI << " ";
    of_state_tum_loc.precision(8);
    of_state_tum_loc << state->_imu->pos()(0) << " " << state->_imu->pos()(1) << " " << state->_imu->pos()(2) << " ";
    of_state_tum_loc << state->_imu->quat().x() << " " << state->_imu->quat().y() << " " << state->_imu->quat().z() << " " << state->_imu->quat().w();

    if (app->initialized_map()) {
        Eigen::Matrix3d R_IitoMAP = state->_pose_MAPtoLOC->Rot().transpose() * state->_imu->Rot();
        Eigen::Vector3d p_IiinMAP = state->_pose_MAPtoLOC->Rot().transpose() * (state->_imu->pos() - state->_pose_MAPtoLOC->pos());
        Eigen::Quaterniond q_IitoMAP = Eigen::Quaterniond(R_IitoMAP);

        of_state_tum_global.precision(10);
        of_state_tum_global.setf(std::ios::fixed, std::ios::floatfield);
        of_state_tum_global << timestamp_inI << " ";
        of_state_tum_global.precision(8);
        of_state_tum_global << p_IiinMAP(0) << " " << p_IiinMAP(1) << " " << p_IiinMAP(2) << " ";
        of_state_tum_global << q_IitoMAP.x() << " " << q_IitoMAP.y() << " " << q_IitoMAP.z() << " " << q_IitoMAP.w();
    }

    // STATE: Write current uncertainty to file
    of_state_std.precision(5);
    of_state_std.setf(std::ios::fixed, std::ios::floatfield);
    of_state_std << timestamp_inI << " ";
    of_state_std.precision(8);
    int id = state->_imu->R()->id();
    of_state_std << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
    id = state->_imu->p()->id();
    of_state_std << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
    id = state->_imu->v()->id();
    of_state_std << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
    id = state->_imu->bg()->id();
    of_state_std << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
    id = state->_imu->ba()->id();
    of_state_std << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
    if (app->initialized_map()) {
        id = state->_pose_MAPtoLOC->R()->id();
        of_state_std << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
        id = state->_pose_MAPtoLOC->p()->id();
        of_state_std << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
    }

    // Intrinsics values
    of_state_est << state->_cam_intrinsics->value()(0) << " " << state->_cam_intrinsics->value()(1) << " " << state->_cam_intrinsics->value()(2) << " "
                 << state->_cam_intrinsics->value()(3) << " ";
    of_state_est << state->_cam_intrinsics->value()(4) << " " << state->_cam_intrinsics->value()(5) << " " << state->_cam_intrinsics->value()(6) << " "
                 << state->_cam_intrinsics->value()(7) << " ";
    // Rotation and position
    of_state_est << state->_calib_IMUtoCAM->value()(0) << " " << state->_calib_IMUtoCAM->value()(1) << " " << state->_calib_IMUtoCAM->value()(2) << " "
                 << state->_calib_IMUtoCAM->value()(3) << " ";
    of_state_est << state->_calib_IMUtoCAM->value()(4) << " " << state->_calib_IMUtoCAM->value()(5) << " " << state->_calib_IMUtoCAM->value()(6) << " ";

    // Done with the estimates!
    of_state_est << endl;
    of_state_std << endl;
    of_state_tum_loc << endl;
    of_state_tum_global << endl;
}

void Visualizer::save_total_state_to_file_tracking_recover() {
    of_state_est << "tracking_recovering..." << endl;
    of_state_std << "tracking_recovering..." << endl;

    std::vector<std::shared_ptr<State>> states = app->get_all_states_tracking_recover();
    for (size_t i = 0; i < states.size(); ++i) {
        // We want to publish in the IMU clock frame
        // The timestamp in the state will be the last camera time
        double timestamp_inI = states[i]->_timestamp;

        // Get the covariance of the whole system
        Eigen::MatrixXd cov = StateHelper::get_full_covariance(states[i]);

        // STATE: Write the current state to file
        of_state_est_tracking_recover << "state: " << i << " ";
        of_state_est_tracking_recover.precision(5);
        of_state_est_tracking_recover.setf(std::ios::fixed, std::ios::floatfield);
        of_state_est_tracking_recover.precision(6);
        of_state_est_tracking_recover << states[i]->_imu->quat().x() << " " << states[i]->_imu->quat().y() << " " << states[i]->_imu->quat().z() << " "
                                      << states[i]->_imu->quat().w() << " ";
        of_state_est_tracking_recover << states[i]->_imu->pos()(0) << " " << states[i]->_imu->pos()(1) << " " << states[i]->_imu->pos()(2) << " ";
        of_state_est_tracking_recover << states[i]->_imu->vel()(0) << " " << states[i]->_imu->vel()(1) << " " << states[i]->_imu->vel()(2) << " ";
        of_state_est_tracking_recover << states[i]->_imu->bias_g()(0) << " " << states[i]->_imu->bias_g()(1) << " " << states[i]->_imu->bias_g()(2) << " ";
        of_state_est_tracking_recover << states[i]->_imu->bias_a()(0) << " " << states[i]->_imu->bias_a()(1) << " " << states[i]->_imu->bias_a()(2) << " ";
        of_state_est_tracking_recover << states[i]->_pose_MAPtoLOC->quat().x() << " " << states[i]->_pose_MAPtoLOC->quat().y() << " " << states[i]->_pose_MAPtoLOC->quat().z()
                                      << " " << states[i]->_pose_MAPtoLOC->quat().w() << " ";
        of_state_est_tracking_recover << states[i]->_pose_MAPtoLOC->pos()(0) << " " << states[i]->_pose_MAPtoLOC->pos()(1) << " " << states[i]->_pose_MAPtoLOC->pos()(2)
                                      << " ";
        // Intrinsics values
        of_state_est_tracking_recover << states[i]->_cam_intrinsics->value()(0) << " " << states[i]->_cam_intrinsics->value()(1) << " "
                                      << states[i]->_cam_intrinsics->value()(2) << " " << states[i]->_cam_intrinsics->value()(3) << " ";
        of_state_est_tracking_recover << states[i]->_cam_intrinsics->value()(4) << " " << states[i]->_cam_intrinsics->value()(5) << " "
                                      << states[i]->_cam_intrinsics->value()(6) << " " << states[i]->_cam_intrinsics->value()(7) << " ";
        // Rotation and position
        of_state_est_tracking_recover << states[i]->_calib_IMUtoCAM->value()(0) << " " << states[i]->_calib_IMUtoCAM->value()(1) << " "
                                      << states[i]->_calib_IMUtoCAM->value()(2) << " " << states[i]->_calib_IMUtoCAM->value()(3) << " ";
        of_state_est_tracking_recover << states[i]->_calib_IMUtoCAM->value()(4) << " " << states[i]->_calib_IMUtoCAM->value()(5) << " "
                                      << states[i]->_calib_IMUtoCAM->value()(6) << " ";

        of_state_tum_loc_tracking_recover << "state: " << i << " ";
        of_state_tum_loc_tracking_recover.precision(10);
        of_state_tum_loc_tracking_recover.setf(std::ios::fixed, std::ios::floatfield);
        of_state_tum_loc_tracking_recover << timestamp_inI << " ";
        of_state_tum_loc_tracking_recover.precision(6);
        of_state_tum_loc_tracking_recover << states[i]->_imu->pos()(0) << " " << states[i]->_imu->pos()(1) << " " << states[i]->_imu->pos()(2) << " ";
        of_state_tum_loc_tracking_recover << states[i]->_imu->quat().x() << " " << states[i]->_imu->quat().y() << " " << states[i]->_imu->quat().z() << " "
                                          << states[i]->_imu->quat().w();

        of_state_tum_global_tracking_recover << "state: " << i << " ";
        Eigen::Matrix3d R_IitoMAP = states[i]->_pose_MAPtoLOC->Rot().transpose() * states[i]->_imu->Rot();
        Eigen::Vector3d p_IiinMAP = states[i]->_pose_MAPtoLOC->Rot().transpose() * (states[i]->_imu->pos() - states[i]->_pose_MAPtoLOC->pos());
        Eigen::Quaterniond q_IitoMAP = Eigen::Quaterniond(R_IitoMAP);
        of_state_tum_global_tracking_recover.precision(10);
        of_state_tum_global_tracking_recover.setf(std::ios::fixed, std::ios::floatfield);
        of_state_tum_global_tracking_recover << timestamp_inI << " ";
        of_state_tum_global_tracking_recover.precision(6);
        of_state_tum_global_tracking_recover << p_IiinMAP(0) << " " << p_IiinMAP(1) << " " << p_IiinMAP(2) << " ";
        of_state_tum_global_tracking_recover << q_IitoMAP.x() << " " << q_IitoMAP.y() << " " << q_IitoMAP.z() << " " << q_IitoMAP.w();

        // STATE: Write current uncertainty to file
        of_state_std_tracking_recover << "state: " << i << " ";
        of_state_std_tracking_recover.precision(5);
        of_state_std_tracking_recover.setf(std::ios::fixed, std::ios::floatfield);
        of_state_std_tracking_recover << timestamp_inI << " ";
        of_state_std_tracking_recover.precision(6);
        int id = states[i]->_imu->R()->id();
        of_state_std_tracking_recover << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
        id = states[i]->_imu->p()->id();
        of_state_std_tracking_recover << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
        id = states[i]->_imu->v()->id();
        of_state_std_tracking_recover << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
        id = states[i]->_imu->bg()->id();
        of_state_std_tracking_recover << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
        id = states[i]->_imu->ba()->id();
        of_state_std_tracking_recover << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
        id = states[i]->_pose_MAPtoLOC->R()->id();
        of_state_std_tracking_recover << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
        id = states[i]->_pose_MAPtoLOC->p()->id();
        of_state_std_tracking_recover << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";

        of_state_est_tracking_recover << endl;
        of_state_std_tracking_recover << endl;
        of_state_tum_loc_tracking_recover << endl;
        of_state_tum_global_tracking_recover << endl;
    }

    // Done with the estimates!
    of_state_est_tracking_recover << endl << endl;
    of_state_std_tracking_recover << endl << endl;
    of_state_tum_loc_tracking_recover << endl << endl;
    of_state_tum_global_tracking_recover << endl << endl;
}

void Visualizer::publish_images(const PackData &message) {
    // Return if we have already visualized
    if (app->get_state() == nullptr)
        return;

    if (last_visualization_timestamp_image == app->get_state()->_timestamp && app->initialized_imu())
        return;
    last_visualization_timestamp_image = app->get_state()->_timestamp;

    // Check if we have subscribers
    if (it_pub_tracks.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for feature image.\n");
        return;
    }

    // Get our image of history tracks
    cv::Mat img_history = message.cam.color_image.clone();
    if (!app->get_historical_viz_image(img_history)) {
        return;
    }

    // Create our message
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "cam0";
    sensor_msgs::ImagePtr exl_track_msg = cv_bridge::CvImage(header, "bgr8", img_history).toImageMsg();

    // Publish
    it_pub_tracks.publish(exl_track_msg);

    // Check if we have subscribers
    if (it_pub_streetlight_matches.getNumSubscribers() == 0 || it_pub_projection.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for matches and projection images.\n");
        return;
    }

    if (!app->initialized_map()) {
        // Get our image of streetlight detections
        cv::Mat img_streetlight = message.cam.color_image.clone();
        if (!app->get_streetlight_viz_image_not_init(img_streetlight)) {
            return;
        }
        // Create our message
        sensor_msgs::ImagePtr exl_st_msg = cv_bridge::CvImage(header, "bgr8", img_streetlight).toImageMsg();

        // Publish
        it_pub_streetlight_matches.publish(exl_st_msg);
    } else {
        cv::Mat img_streetlight1 = message.cam.color_image.clone();
        cv::Mat img_streetlight2 = message.cam.color_image.clone();
        if (!app->get_streetlight_viz_image_init(img_streetlight1, img_streetlight2)) {
            return;
        }

        // Create our message
        sensor_msgs::ImagePtr exl_st_msg1 = cv_bridge::CvImage(header, "bgr8", img_streetlight1).toImageMsg();
        sensor_msgs::ImagePtr exl_st_msg2 = cv_bridge::CvImage(header, "bgr8", img_streetlight2).toImageMsg();

        // Publish
        it_pub_streetlight_matches.publish(exl_st_msg1);
        it_pub_projection.publish(exl_st_msg2);
    }
}

void Visualizer::publish_detection(const PackData &message) {
    // Return if we have already visualized
    if (app->get_state() == nullptr)
        return;

    // Create our message
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "cam0";

    // Check if we have subscribers
    if (it_pub_detection.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for detection images.\n");
        return;
    }

    // Get our image of streetlight detections
    cv::Mat img_streetlight = message.cam.color_image.clone();
    if (!app->get_streetlight_detection_viz_image(img_streetlight)) {
        return;
    }
    // Create our message
    sensor_msgs::ImagePtr exl_st_msg = cv_bridge::CvImage(header, "bgr8", img_streetlight).toImageMsg();

    // Publish
    it_pub_detection.publish(exl_st_msg);
}

void Visualizer::publish_images_tracking_recover(const PackData &message) {
    // Return if we have already visualized
    if (app->get_state_tracking_recover() == nullptr)
        return;

    last_visualization_timestamp_image = app->get_state_tracking_recover()->_timestamp;

    // Check if we have subscribers
    if (it_pub_tracks.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for feature image.\n");
        return;
    }

    // Get our image of history tracks
    cv::Mat img_history = message.cam.color_image.clone();
    if (!app->get_historical_viz_image(img_history)) {
        return;
    }

    // Create our message
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "cam0";
    sensor_msgs::ImagePtr exl_track_msg = cv_bridge::CvImage(header, "bgr8", img_history).toImageMsg();

    // Publish
    it_pub_tracks.publish(exl_track_msg);

    // Check if we have subscribers
    if (it_pub_streetlight_matches.getNumSubscribers() == 0 || it_pub_projection.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for detection and projection images.\n");
        return;
    }

    cv::Mat img_streetlight1 = message.cam.color_image.clone();
    cv::Mat img_streetlight2 = message.cam.color_image.clone();
    if (!app->get_streetlight_viz_image_tracking_recover(img_streetlight1, img_streetlight2)) {
        return;
    }

    // Create our message
    sensor_msgs::ImagePtr exl_st_msg1 = cv_bridge::CvImage(header, "bgr8", img_streetlight1).toImageMsg();
    sensor_msgs::ImagePtr exl_st_msg2 = cv_bridge::CvImage(header, "bgr8", img_streetlight2).toImageMsg();

    // Publish
    it_pub_streetlight_matches.publish(exl_st_msg1);
    it_pub_projection.publish(exl_st_msg2);
}

void Visualizer::publish_other_images_tracking_recover(const PackData &message) {
    // Return if we have already visualized
    if (app->get_state_tracking_recover() == nullptr)
        return;

    last_visualization_timestamp_image = app->get_state_tracking_recover()->_timestamp;

    // Check if we have subscribers
    if (it_pub_projection_tracking_recover.getNumSubscribers() == 0 || it_pub_projection_tracking_recover.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for detection and projection images.\n");
        return;
    }

    cv::Mat img_streetlight = message.cam.color_image.clone();
    if (!app->get_other_viz_image_tracking_recover(img_streetlight)) {
        return;
    }

    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "cam0";

    // Create our message
    sensor_msgs::ImagePtr exl_st_msg = cv_bridge::CvImage(header, "bgr8", img_streetlight).toImageMsg();

    // Publish
    it_pub_projection_tracking_recover.publish(exl_st_msg);
}

void Visualizer::publish_state() {

    if (last_tracking_lost) {
        std::map<double, Eigen::Matrix3d> rots_loc = app->get_rots_loc_in_tracking_lost();
        std::map<double, Eigen::Vector3d> poss_loc = app->get_poss_loc_in_tracking_lost();
        std::map<double, Eigen::Matrix3d> rots_global = app->get_rots_global_in_tracking_lost();
        std::map<double, Eigen::Vector3d> poss_global = app->get_poss_global_in_tracking_lost();

        for (const auto &pair : rots_loc) {

            Eigen::Matrix3d R_IitoMAP = rots_global[pair.first].transpose() * rots_loc[pair.first];
            Eigen::Vector3d p_IiinMAP = rots_global[pair.first].transpose() * (poss_loc[pair.first] - poss_global[pair.first]);
            Eigen::Quaterniond ori_IitoMAP = Eigen::Quaterniond(R_IitoMAP);

            geometry_msgs::PoseStamped posetemp;
            posetemp.header.stamp = ros::Time(pair.first);
            posetemp.header.seq = poses_seq_imu;
            posetemp.header.frame_id = "global";
            posetemp.pose.orientation.x = ori_IitoMAP.x();
            posetemp.pose.orientation.y = ori_IitoMAP.y();
            posetemp.pose.orientation.z = ori_IitoMAP.z();
            posetemp.pose.orientation.w = ori_IitoMAP.w();
            posetemp.pose.position.x = p_IiinMAP.x();
            posetemp.pose.position.y = p_IiinMAP.y();
            posetemp.pose.position.z = p_IiinMAP.z();

            poses_imu.push_back(posetemp);
            // Create our path (imu)
            // NOTE: We downsample the number of poses as needed to prevent rviz crashes
            // NOTE: https://github.com/ros-visualization/rviz/issues/1107
            nav_msgs::Path arrIMU;
            arrIMU.header.stamp = ros::Time(pair.first);
            arrIMU.header.seq = poses_seq_imu;
            arrIMU.header.frame_id = "global";
            for (size_t i = 0; i < poses_imu.size(); i += std::floor((double)poses_imu.size() / 16384.0) + 1) {
                arrIMU.poses.push_back(poses_imu.at(i));
            }
            pub_pathimu.publish(arrIMU);

            // Move them forward in time
            poses_seq_imu++;
        }

        for (size_t i = 0; i < 10; ++i) {
            visualization_msgs::Marker marker;
            marker.id = 2 * i;
            marker.action = visualization_msgs::Marker::DELETE;
            pub_view_region_bound.publish(marker);

            marker.id = 2 * i + 1;
            marker.action = visualization_msgs::Marker::DELETE;
            pub_view_region_fill.publish(marker);
        }
    }

    // Get the current state
    std::shared_ptr<State> state = app->get_state();

    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double timestamp_inI = state->_timestamp;

    // If initialized_map is true, we obtain the global to local pose, the global path can be visualized
    if (app->initialized_map()) {
        Eigen::Matrix3d R_MAPtoLOC = state->_pose_MAPtoLOC->Rot();
        Eigen::Vector3d p_MAPtoLOC = state->_pose_MAPtoLOC->pos();
        Eigen::Matrix3d R_IitoG = state->_imu->Rot();
        Eigen::Vector3d p_IiinG = state->_imu->pos();
        Eigen::Quaterniond ori_IitoMAP = Eigen::Quaterniond(R_MAPtoLOC.transpose() * R_IitoG);
        Eigen::Vector3d p_IiinMAP = R_MAPtoLOC.transpose() * (p_IiinG - p_MAPtoLOC);

        // Create pose of IMU (note we use the bag time)
        geometry_msgs::PoseWithCovarianceStamped poseIinM;
        poseIinM.header.stamp = ros::Time(timestamp_inI);
        poseIinM.header.seq = poses_seq_imu;
        poseIinM.header.frame_id = "global";
        poseIinM.pose.pose.orientation.x = ori_IitoMAP.x();
        poseIinM.pose.pose.orientation.y = ori_IitoMAP.y();
        poseIinM.pose.pose.orientation.z = ori_IitoMAP.z();
        poseIinM.pose.pose.orientation.w = ori_IitoMAP.w();
        poseIinM.pose.pose.position.x = p_IiinMAP.x();
        poseIinM.pose.pose.position.y = p_IiinMAP.y();
        poseIinM.pose.pose.position.z = p_IiinMAP.z();

        pub_poseimu.publish(poseIinM);

        //=========================================================
        //=========================================================

        // Append to our pose vector
        geometry_msgs::PoseStamped posetemp;
        posetemp.header = poseIinM.header;
        posetemp.pose = poseIinM.pose.pose;
        poses_imu.push_back(posetemp);

        // Create our path (imu)
        // NOTE: We downsample the number of poses as needed to prevent rviz crashes
        // NOTE: https://github.com/ros-visualization/rviz/issues/1107
        nav_msgs::Path arrIMU;
        arrIMU.header.stamp = ros::Time::now();
        arrIMU.header.seq = poses_seq_imu;
        arrIMU.header.frame_id = "global";
        for (size_t i = 0; i < poses_imu.size(); i += std::floor((double)poses_imu.size() / 16384.0) + 1) {
            arrIMU.poses.push_back(poses_imu.at(i));
        }
        pub_pathimu.publish(arrIMU);

        // Move them forward in time
        poses_seq_imu++;

        Eigen::Quaterniond ori_OtoMAP = Eigen::Quaterniond(ori_IitoMAP.toRotationMatrix() * state->_calib_IMUtoOdom->Rot().transpose());
        geometry_msgs::Pose robot_pose;
        robot_pose.orientation.w = ori_OtoMAP.w();
        robot_pose.orientation.x = ori_OtoMAP.x();
        robot_pose.orientation.y = ori_OtoMAP.y();
        robot_pose.orientation.z = ori_OtoMAP.z();
        robot_pose.position.x = p_IiinMAP.x();
        robot_pose.position.y = p_IiinMAP.y();
        robot_pose.position.z = p_IiinMAP.z();

        double alpha = 7 * M_PI / 36;

        int radius = 75, segments = 100;
        double start_angle = -alpha, end_angle = alpha;
        visualization_msgs::MarkerArray circle_marker_array;
        visualization_msgs::Marker circle_fill_marker;
        visualization_msgs::Marker circle_bound_marker;
        circle_bound_marker.header.frame_id = "global";
        circle_bound_marker.header.stamp = ros::Time::now();
        circle_bound_marker.id = 0;
        circle_bound_marker.type = visualization_msgs::Marker::LINE_STRIP;
        circle_bound_marker.action = visualization_msgs::Marker::ADD;

        circle_bound_marker.scale.x = 1.0;
        circle_bound_marker.color.r = 0.612f;
        circle_bound_marker.color.g = 0.769f;
        circle_bound_marker.color.b = 0.902f;
        circle_bound_marker.color.a = 0.5f;

        double angle_increment = (end_angle - start_angle) / segments;

        for (int j = 0; j <= segments; ++j) {
            float angle = start_angle + j * angle_increment;
            float x = radius * cos(angle);
            float y = radius * sin(angle);
            geometry_msgs::Point p;
            p.x = x;
            p.y = y;
            p.z = 0;
            circle_bound_marker.points.push_back(p);
        }
        circle_bound_marker.pose = robot_pose;
        pub_view_region_bound.publish(circle_bound_marker);

        circle_fill_marker.header.frame_id = "global";
        circle_fill_marker.header.stamp = ros::Time::now();
        circle_fill_marker.id = 1;
        circle_fill_marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
        circle_fill_marker.action = visualization_msgs::Marker::ADD;

        circle_fill_marker.scale.x = 1.0;
        circle_fill_marker.scale.y = 1.0;
        circle_fill_marker.scale.z = 1.0;
        circle_fill_marker.color.r = 0.612f;
        circle_fill_marker.color.g = 0.769f;
        circle_fill_marker.color.b = 0.902f;
        circle_fill_marker.color.a = 1.0f;

        for (int j = 0; j <= segments; ++j) {
            double angle1 = start_angle + j * angle_increment;
            double angle2 = start_angle + (j + 1) * angle_increment;

            geometry_msgs::Point p0, p1, p2;
            p0.x = 0.0;
            p0.y = 0.0;
            p0.z = 0.0;

            p1.x = radius * cos(angle1);
            p1.y = radius * sin(angle1);
            p1.z = 0.0;

            p2.x = radius * cos(angle2);
            p2.y = radius * sin(angle2);
            p2.z = 0.0;

            circle_fill_marker.points.push_back(p0);
            circle_fill_marker.points.push_back(p1);
            circle_fill_marker.points.push_back(p2);

            std_msgs::ColorRGBA fill_color;
            fill_color.r = 0.612f;
            fill_color.g = 0.769f;
            fill_color.b = 0.902f;
            fill_color.a = 0.2;

            circle_fill_marker.colors.push_back(fill_color);
            circle_fill_marker.colors.push_back(fill_color);
            circle_fill_marker.colors.push_back(fill_color);
        }
        circle_fill_marker.pose = robot_pose;
        pub_view_region_fill.publish(circle_fill_marker);
    }
}

void Visualizer::publish_state_tracking_recover() {
    // Get the current state
    std::shared_ptr<State> state = app->get_state_tracking_recover();

    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double timestamp_inI = state->_timestamp;

    Eigen::Matrix3d R_IitoG = state->_imu->Rot();
    Eigen::Vector3d p_IiinG = state->_imu->pos();
    Eigen::Quaterniond ori_IitoG = Eigen::Quaterniond(R_IitoG);

    // Create pose of IMU (note we use the bag time)
    geometry_msgs::PoseWithCovarianceStamped poseIinG;
    poseIinG.header.stamp = ros::Time(timestamp_inI);
    poseIinG.header.seq = poses_seq_imu;
    poseIinG.header.frame_id = "global";
    poseIinG.pose.pose.orientation.x = ori_IitoG.x();
    poseIinG.pose.pose.orientation.y = ori_IitoG.y();
    poseIinG.pose.pose.orientation.z = ori_IitoG.z();
    poseIinG.pose.pose.orientation.w = ori_IitoG.w();
    poseIinG.pose.pose.position.x = p_IiinG.x();
    poseIinG.pose.pose.position.y = p_IiinG.y();
    poseIinG.pose.pose.position.z = p_IiinG.z();

    pub_poseimu.publish(poseIinG);

    //=========================================================
    //=========================================================

    // Append to our pose vector
    geometry_msgs::PoseStamped posetemp;
    posetemp.header = poseIinG.header;
    posetemp.pose = poseIinG.pose.pose;
    poses_imu.push_back(posetemp);

    // Create our path (imu)
    // NOTE: We downsample the number of poses as needed to prevent rviz crashes
    // NOTE: https://github.com/ros-visualization/rviz/issues/1107
    nav_msgs::Path arrIMU;
    arrIMU.header.stamp = ros::Time::now();
    arrIMU.header.seq = poses_seq_imu;
    arrIMU.header.frame_id = "global";
    for (size_t i = 0; i < poses_imu.size(); i += std::floor((double)poses_imu.size() / 16384.0) + 1) {
        arrIMU.poses.push_back(poses_imu.at(i));
    }
    pub_pathimu.publish(arrIMU);

    // Move them forward in time
    poses_seq_imu++;
}

void Visualizer::publish_features() {

    // Check if we have subscribers
    if (pub_points_msckf.getNumSubscribers() == 0 && pub_points_slam.getNumSubscribers() == 0)
        return;

    // Get our good MSCKF features
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> feats_msckf = app->get_good_features_MSCKF();
    sensor_msgs::PointCloud2 cloud = get_ros_pointcloud_inMAP(feats_msckf);
    pub_points_msckf.publish(cloud);

    // Get our good SLAM features
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> feats_slam = app->get_features_SLAM();
    sensor_msgs::PointCloud2 cloud_SLAM = get_ros_pointcloud_inMAP(feats_slam);
    pub_points_slam.publish(cloud_SLAM);
}

void Visualizer::publish_odometry() {
    nav_msgs::Odometry odom;
    odom.header.frame_id = "global";
    odom.child_frame_id = "body";

    // Get the current state
    std::shared_ptr<State> state = app->get_state();

    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double timestamp_inI = state->_timestamp;
    odom.header.stamp = ros::Time(timestamp_inI);

    // If initialized_map is true, we obtain the global to local pose, the global path can be visualized
    Eigen::Matrix3d R_MAPtoLOC = state->_pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPtoLOC = state->_pose_MAPtoLOC->pos();
    Eigen::Matrix3d R_IitoG = state->_imu->Rot();
    Eigen::Vector3d p_IiinG = state->_imu->pos();
    Eigen::Quaterniond ori_IitoMAP = Eigen::Quaterniond(R_MAPtoLOC.transpose() * R_IitoG);
    Eigen::Vector3d p_IiinMAP = R_MAPtoLOC.transpose() * (p_IiinG - p_MAPtoLOC);

    odom.pose.pose.position.x = p_IiinMAP.x();
    odom.pose.pose.position.y = p_IiinMAP.y();
    odom.pose.pose.position.z = p_IiinMAP.z();
    odom.pose.pose.orientation.x = ori_IitoMAP.x();
    odom.pose.pose.orientation.y = ori_IitoMAP.y();
    odom.pose.pose.orientation.z = ori_IitoMAP.z();
    odom.pose.pose.orientation.w = ori_IitoMAP.w();
    pub_odometer.publish(odom);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z));
    q.setW(odom.pose.pose.orientation.w);
    q.setX(odom.pose.pose.orientation.x);
    q.setY(odom.pose.pose.orientation.y);
    q.setZ(odom.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odom.header.stamp, "global", "body"));
}

void Visualizer::publish_cam_path() {
    // Append to our pose vector
    geometry_msgs::PoseStamped posetemp;
    posetemp.header.frame_id = "global";
    posetemp.header.stamp = ros::Time::now();

    // Get the current state
    std::shared_ptr<State> state = app->get_state();
    Eigen::Matrix3d R_MAPtoLOC = state->_pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPtoLOC = state->_pose_MAPtoLOC->pos();
    Eigen::Matrix3d R_IitoG = state->_imu->Rot();
    Eigen::Vector3d p_IiinG = state->_imu->pos();
    Eigen::Quaterniond ori_IitoMAP = Eigen::Quaterniond(R_MAPtoLOC.transpose() * R_IitoG);
    Eigen::Vector3d p_IiinMAP = R_MAPtoLOC.transpose() * (p_IiinG - p_MAPtoLOC);
    Eigen::Quaterniond ori_OtoMAP = (ori_IitoMAP * Eigen::Quaterniond(state->_calib_IMUtoCAM->Rot().transpose())).normalized();

    posetemp.pose.position.x = p_IiinMAP.x();
    posetemp.pose.position.y = p_IiinMAP.y();
    posetemp.pose.position.z = p_IiinMAP.z();
    posetemp.pose.orientation.x = ori_OtoMAP.x();
    posetemp.pose.orientation.y = ori_OtoMAP.y();
    posetemp.pose.orientation.z = ori_OtoMAP.z();
    posetemp.pose.orientation.w = ori_OtoMAP.w();
    night_voyager_cam.reset();
    night_voyager_cam.add_pose(Eigen::Vector3d(posetemp.pose.position.x, posetemp.pose.position.y, posetemp.pose.position.z),
                               Eigen::Quaterniond(posetemp.pose.orientation.w, posetemp.pose.orientation.x, posetemp.pose.orientation.y, posetemp.pose.orientation.z));
    night_voyager_cam.publish_by(pub_night_voyager_cam, state->_timestamp);
}

void Visualizer::publish_features_tracking_recover() {

    // Check if we have subscribers
    if (pub_points_msckf.getNumSubscribers() == 0)
        return;

    // Get our good MSCKF features
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> feats_msckf = app->get_good_features_MSCKF();
    sensor_msgs::PointCloud2 cloud = get_ros_pointcloud_inMAP_tracking_recover(feats_msckf);
    pub_points_msckf.publish(cloud);
}

sensor_msgs::PointCloud2 Visualizer::get_ros_pointcloud_inMAP(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &feats) {

    // Get the current state
    std::shared_ptr<State> state = app->get_state();

    // If initialized_map is true, we obtain the global to local pose, the features in the map can be visualized
    if (app->initialized_map()) {
        Eigen::Matrix3d R_LOCtoMAP = state->_pose_MAPtoLOC->Rot().transpose();
        Eigen::Vector3d p_MAPtoLOC = state->_pose_MAPtoLOC->pos();

        // Declare message and sizes
        sensor_msgs::PointCloud2 cloud;
        cloud.header.frame_id = "global";
        cloud.header.stamp = ros::Time::now();
        cloud.width = feats.size();
        cloud.height = 1;
        cloud.is_bigendian = false;
        cloud.is_dense = false; // there may be invalid points

        // Setup pointcloud fields
        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(feats.size());

        // Iterators
        sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");

        // Fill our iterators
        for (const auto &pt : feats) {
            Eigen::Vector3d pf_MAP = R_LOCtoMAP * (pt - p_MAPtoLOC);
            *out_x = (float)pf_MAP(0);
            ++out_x;
            *out_y = (float)pf_MAP(1);
            ++out_y;
            *out_z = (float)pf_MAP(2);
            ++out_z;
        }

        return cloud;
    }
    // Else we visualize the features in local frame
    else {
        // Declare message and sizes
        sensor_msgs::PointCloud2 cloud;
        cloud.header.frame_id = "global";
        cloud.header.stamp = ros::Time::now();
        cloud.width = feats.size();
        cloud.height = 1;
        cloud.is_bigendian = false;
        cloud.is_dense = false; // there may be invalid points

        // Setup pointcloud fields
        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(feats.size());

        // Iterators
        sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");

        // Fill our iterators
        for (const auto &pt : feats) {
            *out_x = (float)pt(0);
            ++out_x;
            *out_y = (float)pt(1);
            ++out_y;
            *out_z = (float)pt(2);
            ++out_z;
        }

        return cloud;
    }
}

sensor_msgs::PointCloud2 Visualizer::get_ros_pointcloud_inMAP_tracking_recover(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &feats) {

    // Get the current state
    std::shared_ptr<State> state = app->get_state_tracking_recover();

    Eigen::Matrix3d R_LOCtoMAP = state->_pose_MAPtoLOC->Rot().transpose();
    Eigen::Vector3d p_MAPtoLOC = state->_pose_MAPtoLOC->pos();

    // Declare message and sizes
    sensor_msgs::PointCloud2 cloud;
    cloud.header.frame_id = "global";
    cloud.header.stamp = ros::Time::now();
    cloud.width = feats.size();
    cloud.height = 1;
    cloud.is_bigendian = false;
    cloud.is_dense = false; // there may be invalid points

    // Setup pointcloud fields
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(feats.size());

    // Iterators
    sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");

    // Fill our iterators
    for (const auto &pt : feats) {
        Eigen::Vector3d pf_MAP = R_LOCtoMAP * (pt - p_MAPtoLOC);
        *out_x = (float)pf_MAP(0);
        ++out_x;
        *out_y = (float)pf_MAP(1);
        ++out_y;
        *out_z = (float)pf_MAP(2);
        ++out_z;
    }

    return cloud;
}

void Visualizer::publish_prior_poses() {

    // Get the prior pose manager
    std::shared_ptr<PriorPoseManager> prpose = app->get_prpose_manager();

    // Check if we have subscribers
    if (pub_prior_poss.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for prior poses.\n");
        return;
    }

    for (size_t i = 0; i < prpose->plane_marker_array.markers.size(); i++) {
        prpose->plane_marker_array.markers[i].header.stamp = ros::Time::now();
    }

    pub_prior_poss.publish(prpose->plane_marker_array);
}

void Visualizer::publish_near_prior_poses() {

    // Get the prior pose manager
    std::shared_ptr<PriorPoseManager> prpose = app->get_prpose_manager();

    // Check if we have subscribers
    if (pub_near_prior_cloud.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for prior poses.\n");
        return;
    }

    sensor_msgs::PointCloud2 near_prior_pose_cloud;
    pcl::toROSMsg(*prpose->near_prior_pose_cloud, near_prior_pose_cloud);
    near_prior_pose_cloud.header.frame_id = "global";
    near_prior_pose_cloud.header.stamp = ros::Time::now();
    pub_near_prior_cloud.publish(near_prior_pose_cloud);
}

void Visualizer::publish_initialization_regions() {

    // Get the prior pose manager
    std::shared_ptr<PriorPoseManager> prpose = app->get_prpose_manager();

    // Check if we have subscribers
    if (pub_init_regions.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for prior poses.\n");
        return;
    }

    for (size_t i = 0; i < prpose->circle_marker_array.markers.size(); i++) {
        prpose->circle_marker_array.markers[i].header.stamp = ros::Time::now();
    }

    pub_init_regions.publish(prpose->circle_marker_array);
}

void Visualizer::publish_global_pointcloud() {

    // Get the pcd manager
    std::shared_ptr<PcdManager> pcd = app->get_pcd_manager();

    // Check if we have subscribers
    if (pub_prior_map.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for prior map.\n");
        return;
    }

    pcd->streetlight_cloud_ros.header.frame_id = "global";
    pcd->streetlight_cloud_ros.header.stamp = ros::Time::now();
    pub_prior_map.publish(pcd->streetlight_cloud_ros);
}

void Visualizer::publish_matched_pointcloud() {

    // Get the current pcd manager
    std::shared_ptr<PcdManager> pcd = app->get_pcd_manager();

    if (pub_points_streetlight.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for matched streetlight clusters.\n");
        return;
    }

    sensor_msgs::PointCloud2 matched_cloud;
    pcl::toROSMsg(*pcd->matched_streetlight_cloud, matched_cloud);
    matched_cloud.header.frame_id = "global";
    matched_cloud.header.stamp = ros::Time::now();
    pub_points_streetlight.publish(matched_cloud);
}

void Visualizer::publish_all_possible_paths() {

    std::vector<std::shared_ptr<State>> states = app->get_all_states_tracking_recover();

    if (pub_paths_tracking_recover.getNumSubscribers() == 0) {
        PRINT_INFO(WHITE "No subscribers for tracking recovering paths.\n");
        return;
    }

    if (marker_array.markers.empty()) {
        marker_array.markers.resize(states.size());
    }

    auto createColor = [](float r, float g, float b, float a) {
        std_msgs::ColorRGBA color;
        color.r = r;
        color.g = g;
        color.b = b;
        color.a = a;
        return color;
    };

    vector<std_msgs::ColorRGBA> colors = {
        createColor(0.56, 0.93, 0.56, 1.0), // Light Green
        createColor(0.68, 0.85, 0.9, 1.0),  // Light Blue
        createColor(0.86, 0.08, 0.24, 1.0), // Crimson
        createColor(1.0, 0.75, 0.8, 1.0),   // Pink
        createColor(0.5, 0.5, 0.0, 1.0),    // Olive
        createColor(0.0, 0.0, 0.55, 1.0),   // Dark Blue
        createColor(0.65, 0.16, 0.16, 1.0), // Brown
        createColor(0.0, 0.39, 0.0, 1.0),   // Dark Green
        createColor(1.0, 0.55, 0.0, 1.0),   // Dark Orange
        createColor(1.0, 0.84, 0.0, 1.0)    // Gold
    };

    for (size_t i = 0; i < states.size(); ++i) {
        visualization_msgs::Marker &marker = marker_array.markers[i];
        marker.header.frame_id = "global";
        marker.header.stamp = ros::Time::now();

        marker.id = i;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.2;
        marker.color = colors[i];

        geometry_msgs::Point new_point;
        Eigen::Matrix3d R_ItoMAP = states[i]->_pose_MAPtoLOC->Rot().transpose() * states[i]->_imu->Rot();
        Eigen::Vector3d p_IinMAP = states[i]->_pose_MAPtoLOC->Rot().transpose() * (states[i]->_imu->pos() - states[i]->_pose_MAPtoLOC->pos());
        new_point.x = p_IinMAP.x();
        new_point.y = p_IinMAP.y();
        new_point.z = p_IinMAP.z();
        marker.points.push_back(new_point);

        marker_array.markers[i] = marker;

        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(p_IinMAP.x(), p_IinMAP.y(), p_IinMAP.z()));
        Eigen::Quaterniond q_ItoMAP(R_ItoMAP);
        q.setW(q_ItoMAP.w());
        q.setX(q_ItoMAP.x());
        q.setY(q_ItoMAP.y());
        q.setZ(q_ItoMAP.z());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "global", "body" + to_string(i)));

        Eigen::Quaterniond ori_OtoMAP = Eigen::Quaterniond(R_ItoMAP * states[i]->_calib_IMUtoOdom->Rot().transpose());
        geometry_msgs::Pose robot_pose;
        robot_pose.orientation.w = ori_OtoMAP.w();
        robot_pose.orientation.x = ori_OtoMAP.x();
        robot_pose.orientation.y = ori_OtoMAP.y();
        robot_pose.orientation.z = ori_OtoMAP.z();
        robot_pose.position.x = p_IinMAP.x();
        robot_pose.position.y = p_IinMAP.y();
        robot_pose.position.z = p_IinMAP.z();

        double alpha = 7 * M_PI / 36;

        int radius = 25, segments = 100;
        double start_angle = -alpha, end_angle = alpha;
        visualization_msgs::MarkerArray circle_marker_array;
        visualization_msgs::Marker circle_fill_marker;
        visualization_msgs::Marker circle_bound_marker;
        circle_bound_marker.header.frame_id = "global";
        circle_bound_marker.header.stamp = ros::Time::now();
        circle_bound_marker.id = 2 * i;
        circle_bound_marker.type = visualization_msgs::Marker::LINE_STRIP;
        circle_bound_marker.action = visualization_msgs::Marker::ADD;

        circle_bound_marker.scale.x = 1.0;
        // circle_marker.scale.y = 1.0;
        // circle_marker.scale.z = 1.0;
        // circle_bound_marker.color.r = float(i) / states.size();
        // circle_bound_marker.color.g = 1.0f - float(i) / states.size();
        // circle_bound_marker.color.b = 1.0f - float(i) / states.size();
        // circle_bound_marker.color.a = 0.5f;
        circle_bound_marker.color = colors[i];

        double angle_increment = (end_angle - start_angle) / segments;

        for (int j = 0; j <= segments; ++j) {
            float angle = start_angle + j * angle_increment;
            float x = radius * cos(angle);
            float y = radius * sin(angle);
            geometry_msgs::Point p;
            p.x = x;
            p.y = y;
            p.z = 0;
            circle_bound_marker.points.push_back(p);
        }
        circle_bound_marker.pose = robot_pose;
        pub_view_region_bound.publish(circle_bound_marker);
        // circle_marker_array.markers.push_back(circle_bound_marker);

        circle_fill_marker.header.frame_id = "global";
        circle_fill_marker.header.stamp = ros::Time::now();
        circle_fill_marker.id = 2 * i + 1;
        circle_fill_marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
        circle_fill_marker.action = visualization_msgs::Marker::ADD;

        circle_fill_marker.scale.x = 1.0;
        circle_fill_marker.scale.y = 1.0;
        circle_fill_marker.scale.z = 1.0;
        circle_fill_marker.color = colors[i];
        circle_fill_marker.color.a = 0.5;

        for (int j = 0; j <= segments; ++j) {
            double angle1 = start_angle + j * angle_increment;
            double angle2 = start_angle + (j + 1) * angle_increment;

            geometry_msgs::Point p0, p1, p2;
            p0.x = 0.0;
            p0.y = 0.0;
            p0.z = 0.0;

            p1.x = radius * cos(angle1);
            p1.y = radius * sin(angle1);
            p1.z = 0.0;

            p2.x = radius * cos(angle2);
            p2.y = radius * sin(angle2);
            p2.z = 0.0;

            circle_fill_marker.points.push_back(p0);
            circle_fill_marker.points.push_back(p1);
            circle_fill_marker.points.push_back(p2);

            std_msgs::ColorRGBA fill_color;
            fill_color = colors[i];
            fill_color.a = 0.5;
            // fill_color.r = 0.612f;
            // fill_color.g = 0.769f;
            // fill_color.b = 0.902f;
            // fill_color.a = 0.2;

            circle_fill_marker.colors.push_back(fill_color);
            circle_fill_marker.colors.push_back(fill_color);
            circle_fill_marker.colors.push_back(fill_color);
        }
        circle_fill_marker.pose = robot_pose;
        pub_view_region_fill.publish(circle_fill_marker);
    }

    pub_paths_tracking_recover.publish(marker_array);
}
} // namespace night_voyager