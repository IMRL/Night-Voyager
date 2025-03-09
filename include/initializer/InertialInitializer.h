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
#ifndef INERTIAL_INITIALIZER_H
#define INERTIAL_INITIALIZER_H

#include "core/CommonLib.h"
#include "core/NightVoyagerOptions.h"
#include <mutex>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <unordered_map>

namespace night_voyager {
class IMU;
class PoseHamilton;
class Type;
class State;
class FeatureDatabase;
class StaticInitializer;
class PcdManager;
class PriorPoseManager;

class InertialInitializer {
  public:
    struct Tuple {
        /// Indices of streetlight observations
        int o1, o2, o3;
    };

    struct TupleScore {
        /// Indices of streetlight clusters
        int l1, l2, l3;

        /// Score of this combination
        double score = -1;

        /// 2D pose estimated from this combination, p_CinMAP, theta_CtoMAP
        Eigen::Vector3d pose;

        /// Tuple streetlight matches
        std::vector<int> matches;
    };

    struct PoseScore {
        double score = -10000;
        double subscore = -10000;

        Eigen::Matrix3d R;
        Eigen::Vector3d p;
        std::vector<int> matches;
        int region;
    };

    /**
     * @brief Default constructor
     * @param options Parameters loaded from either ROS or CMDLINE
     * @param db Feature tracker database with all features in it
     */
    explicit InertialInitializer(NightVoyagerOptions &options, std::shared_ptr<FeatureDatabase> db, std::shared_ptr<PcdManager> pcd,
                                 std::shared_ptr<PriorPoseManager> prpose, std::shared_ptr<CamBase> &camera_intrinsic);

    /**
     * @brief Feed function for inertial data
     * @param message Contains our timestamp and inertial information
     * @param oldest_time Time that we can discard measurements before
     */
    void feed_imu(const ImuData &message, double oldest_time = -1);

    /**
     * @brief Feed function for Box data
     * @param message Contains our timestamp and box information
     * @param oldest_time Time that we can discard measurements before
     */
    void feed_box(const BoxData &message, double oldest_time = -1);

    /**
     * @brief Try to get the initialized system
     *
     *
     * @m_class{m-note m-warning}
     *
     * @par Processing Cost
     * This is a serial process that can take on orders of seconds to complete.
     * If you are a real-time application then you will likely want to call this from
     * a async thread which allows for this to process in the background.
     * The features used are cloned from the feature database thus should be thread-safe
     * to continue to append new feature tracks to the database.
     *
     * @param[out] timestamp Timestamp we have initialized the state at
     * @param[out] covariance Calculated covariance of the returned state
     * @param[out] order Order of the covariance matrix
     * @param[out] t_imu Our imu type (need to have correct ids)
     * @return True if we have successfully initialized our system
     */
    void initialize_viwo(double &timestamp, Eigen::MatrixXd &covariance_viwo, std::vector<std::shared_ptr<Type>> &order_viwo,
                         std::shared_ptr<IMU> t_imu);

    void initialize_map(const PackData &pack_queue, const Eigen::Matrix4d &T_ItoC, const Eigen::Matrix4d &T_ItoG,
                        const Eigen::MatrixXd &covariance_viwo_pose, Eigen::MatrixXd &T_MAPtoLOC, Eigen::MatrixXd &covariance_map);

    void process_box(const CameraData &img, BoxData &box);

    void display_streetlights(cv::Mat &img_out, string overlay);

    bool viwo_initialized() { return initialized_viwo; }

    bool map_initialized() { return initialized_map; }

  protected:
    /// Initialization parameters
    InertialInitializerOptions _options;

    /// Feature tracker database with all features in it
    std::shared_ptr<FeatureDatabase> _db;

    /// PcdManager with all map points
    std::shared_ptr<PcdManager> _pcd;

    /// PriorposeManager with downsampled poses
    std::shared_ptr<PriorPoseManager> _prpose;

    bool initialized_viwo, initialized_map;

    std::shared_ptr<CamBase> camera_calib;

    /// Our history of IMU messages (time, angular, linear)
    std::shared_ptr<std::vector<ImuData>> imu_data;

    /// Our history of box messages
    std::vector<BoxData> box_data;
    std::mutex box_data_mtx;

    /// Static initialization helper class
    std::shared_ptr<StaticInitializer> init_static;

    /// Used to store the selected box measruement (static initializer)
    BoxData selected_box_data;
    BoxData unselected_box_data;

    std::vector<Eigen::Vector2d> region_samples;
    std::vector<std::vector<int>> stidx_in_regions;
    std::vector<std::vector<TupleScore>> combs_in_region;
};
} // namespace night_voyager

#endif