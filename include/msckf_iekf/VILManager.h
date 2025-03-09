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
#ifndef VILMANAGER_H
#define VILMANAGER_H

#include "core/CommonLib.h"
#include "core/NightVoyagerOptions.h"
#include "msckf_iekf/Propagator.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace night_voyager {

class TrackKLT;
class PcdManager;
class PriorPoseManager;
class UpdaterMSCKF;
class UpdaterSLAM;
class UpdaterOdom;
class UpdaterMAP;
class UpdaterZeroVelocity;
class UpdaterPlane;
class Propagator;
class State;
class InertialInitializer;
class TrackingRecover;

class VILManager {
  public:
    VILManager(NightVoyagerOptions &options, std::shared_ptr<PcdManager> pcd, std::shared_ptr<PriorPoseManager> prpose);

    void feed_measurement_all(const PackData &message);

    void feed_measurement_box(const BoxData &message);

    bool try_to_initialize_imu(const PackData &message);

    bool try_to_initialize_map(const PackData &message);

    void init_thread_imu();

    void init_thread_map(const PackData &message, const Eigen::Matrix4d &T_ItoC, const Eigen::Matrix4d &T_ItoG,
                         const Eigen::MatrixXd &covariance_viwo_pose);

    void postprocess_after_initviwo();

    void do_feature_propagate_update(const PackData &message, const ImuData &last_imu);

    void do_feature_propagate_update_tracking_recover(const PackData &message, const ImuData &last_imu);

    /// Get a nice visualization image of what tracks we have
    bool get_historical_viz_image(cv::Mat &img_history);

    /// Get a nice visualization image of what matched streetlight detections we have
    bool get_streetlight_viz_image_init(cv::Mat &img_streetlight1, cv::Mat &img_streetlight2);

    bool get_streetlight_viz_image_tracking_recover(cv::Mat &img_streetlight1, cv::Mat &img_streetlight2);

    bool get_other_viz_image_tracking_recover(cv::Mat &img_streetlight);

    /// Get a nice visualization image of what streetlight detections we have
    bool get_streetlight_viz_image_not_init(cv::Mat &img_streetlight);

    bool get_streetlight_detection_viz_image(cv::Mat &img_streetlight);

    /// Accessor to get the current state
    std::shared_ptr<State> get_state() { return state; }

    /// Accessor to get the tracking recover state
    std::shared_ptr<State> get_state_tracking_recover();

    std::vector<std::shared_ptr<State>> get_all_states_tracking_recover();

    /// Accessor to get the current pcdmanager
    std::shared_ptr<PcdManager> get_pcd_manager() { return _pcd; }

    /// Accessor to get the current prior pose manager
    std::shared_ptr<PriorPoseManager> get_prpose_manager() { return _prpose; }

    /// If we are initialized or not (IMU)
    bool initialized_imu() { return is_initialized_imu; } // && timelastupdate != -1;

    /// If we are initialized or not
    bool initialized_map() { return is_initialized_map; }

    /// Returns 3d SLAM features in the global frame
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> get_features_SLAM();

    /// Returns 3d features used in the last update in global frame
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> get_good_features_MSCKF() { return good_features_MSCKF; }

    bool is_tracking_lost();

    std::map<double, Eigen::Matrix3d> get_rots_global_in_tracking_lost();

    std::map<double, Eigen::Vector3d> get_poss_global_in_tracking_lost();

    std::map<double, Eigen::Matrix3d> get_rots_loc_in_tracking_lost();

    std::map<double, Eigen::Vector3d> get_poss_loc_in_tracking_lost();

    bool no_streetlight;

    double time_no_map;

  protected:
    NightVoyagerOptions options;

    /// Our master state object :D
    std::shared_ptr<State> state;

    std::shared_ptr<TrackingRecover> tracking_recover;

    std::shared_ptr<PcdManager> _pcd;
    std::shared_ptr<PriorPoseManager> _prpose;

    shared_ptr<Propagator> propagator;

    // Our sparse feature tracker (klt)
    shared_ptr<TrackKLT> trackFEATS;

    /// State initializer
    std::shared_ptr<InertialInitializer> initializer;

    /// Our MSCKF feature updater
    std::shared_ptr<UpdaterMSCKF> updaterMSCKF;

    /// Our SLAM feature updater
    std::shared_ptr<UpdaterSLAM> updaterSLAM;

    /// Our Odom updater
    std::shared_ptr<UpdaterOdom> updaterOdom;

    /// Our streetlight feature updater
    std::shared_ptr<UpdaterMAP> updaterMAP;

    /// Our plane updater
    std::shared_ptr<UpdaterPlane> updaterPlane;

    /// Our zero velocity tracker
    std::shared_ptr<UpdaterZeroVelocity> updaterZUPT;

    bool is_initialized_map = false, is_initialized_imu = false;

    atomic<bool> thread_init_imu_running, thread_init_imu_success, thread_init_map_running, thread_init_map_success;

    // double timelastupdate = -1;

    // If we did a zero velocity update
    bool did_zupt_update = false;
    bool has_moved_since_zupt = false;

    /// This is the queue of measurement times that have come in since we starting doing initialization
    /// After we initialize, we will want to prop & update to the latest timestamp quickly
    std::vector<CameraData> camera_queue_init;
    std::vector<OdomData> odom_queue_init;
    std::vector<PackData> pack_imu_queue_init;
    std::mutex camera_queue_init_mtx;
    std::mutex odom_queue_init_mtx;
    std::mutex pack_imu_queue_init_mtx;

    boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7, rT8, rT9, rT10, rT11, rT12;

    // Startup time of the filter
    double startup_time = -1;

    // Good features that where used in the last update (used in visualization)
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> good_features_MSCKF;
};
} // namespace night_voyager
#endif