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
#ifndef STATE_H
#define STATE_H

#include "core/IMU.h"
#include "core/LandMark.h"
#include "core/NightVoyagerOptions.h"
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace night_voyager {

class State {

  public:
    /**
     * @brief Default Constructor (will initialize variables to defaults)
     * @param options Options structure containing filter options
     */
    State(const NightVoyagerOptions &options) {
        // Save our options
        _options = options.state_options;

        // Append the imu to the state and covariance
        int current_id = 0;
        _imu = std::make_shared<IMU>();
        _imu->set_local_id(current_id);
        _variables.push_back(_imu);
        current_id += _imu->size();

        _pose_MAPtoLOC = std::make_shared<PoseHamilton>();
        _pose_MAPtoLOC->set_local_id(current_id);
        _variables.push_back(_pose_MAPtoLOC);
        current_id += _pose_MAPtoLOC->size();

        _cam_intrinsics = std::make_shared<Vec>(8);
        _calib_IMUtoCAM = std::make_shared<PoseHamilton>();
        _calib_IMUtoOdom = std::make_shared<PoseHamilton>();

        // Finally initialize our covariance to small value
        _Cov = std::pow(1e-3, 2) * Eigen::MatrixXd::Identity(current_id, current_id);

        _zupt = false;
        _kf = options.state_options.kf;
        _feature_in_clone = false;
        _feature_in_rel_group = false;
        _pseudo_map_anchor_time = -1;
    };

    State(const State &other) {
        _options = other._options;

        _timestamp = other._timestamp;

        _pseudo_map_anchor_time = other._pseudo_map_anchor_time;

        std::shared_ptr<Type> new_clone = nullptr;

        _imu = std::make_shared<IMU>();
        new_clone = other._imu->clone();
        new_clone->set_local_id(other._imu->id());
        _imu = std::dynamic_pointer_cast<IMU>(new_clone);
        if (_imu == nullptr) {
            PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
            std::exit(EXIT_FAILURE);
        }
        _variables.push_back(_imu);

        _pose_MAPtoLOC = std::make_shared<PoseHamilton>();
        new_clone = other._pose_MAPtoLOC->clone();
        new_clone->set_local_id(other._pose_MAPtoLOC->id());
        _pose_MAPtoLOC = std::dynamic_pointer_cast<PoseHamilton>(new_clone);
        if (_pose_MAPtoLOC == nullptr) {
            PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
            std::exit(EXIT_FAILURE);
        }
        _variables.push_back(_pose_MAPtoLOC);

        _cam_intrinsics = std::make_shared<Vec>(8);
        new_clone = other._cam_intrinsics->clone();
        _cam_intrinsics = std::dynamic_pointer_cast<Vec>(new_clone);
        if (_cam_intrinsics == nullptr) {
            PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
            std::exit(EXIT_FAILURE);
        }

        _calib_IMUtoCAM = std::make_shared<PoseHamilton>();
        new_clone = other._calib_IMUtoCAM->clone();
        _calib_IMUtoCAM = std::dynamic_pointer_cast<PoseHamilton>(new_clone);
        if (_calib_IMUtoCAM == nullptr) {
            PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
            std::exit(EXIT_FAILURE);
        }

        _calib_IMUtoOdom = std::make_shared<PoseHamilton>();
        new_clone = other._calib_IMUtoOdom->clone();
        _calib_IMUtoOdom = std::dynamic_pointer_cast<PoseHamilton>(new_clone);
        if (_calib_IMUtoOdom == nullptr) {
            PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
            std::exit(EXIT_FAILURE);
        }

        _cam_intrinsics_camera = other._cam_intrinsics_camera;

        for (auto &pair : other._clones_IMU) {
            std::shared_ptr<PoseHamilton> posetemp = std::make_shared<PoseHamilton>();
            new_clone = pair.second->clone();
            new_clone->set_local_id(pair.second->id());
            posetemp = std::dynamic_pointer_cast<PoseHamilton>(new_clone);
            if (posetemp == nullptr) {
                PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
                std::exit(EXIT_FAILURE);
            }
            _clones_IMU[pair.first] = posetemp;
            _variables.push_back(_clones_IMU[pair.first]);
        }

        for (auto &pair : other._features_SLAM) {
            std::shared_ptr<Landmark> feattemp = std::make_shared<Landmark>(3);
            new_clone = pair.second->clone();
            new_clone->set_local_id(pair.second->id());
            feattemp = std::dynamic_pointer_cast<Landmark>(new_clone);
            if (feattemp == nullptr) {
                PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
                std::exit(EXIT_FAILURE);
            }
            feattemp->_featid = pair.second->_featid;
            feattemp->_anchor_clone_timestamp = pair.second->_anchor_clone_timestamp;
            feattemp->pseudo_anchor_clone_timestamp = pair.second->pseudo_anchor_clone_timestamp;
            feattemp->should_marg = pair.second->should_marg;
            _features_SLAM[pair.first] = feattemp;
            _variables.push_back(_features_SLAM[pair.first]);
        }
        _zupt = other._zupt;
        _Cov = other._Cov;
        _feature_in_clone = other._feature_in_clone;
        _feature_in_rel_group = other._feature_in_rel_group;
        _kf = other._kf;
    }

    ~State() {}

    /**
     * @brief Will return the timestep that we will marginalize next.
     * As of right now, since we are using a sliding window, this is the oldest clone.
     * But if you wanted to do a keyframe system, you could selectively marginalize clones.
     * @return timestep of clone we will marginalize
     */
    double margtimestep() {
        std::lock_guard<std::mutex> lock(_mutex_state);
        double time = INFINITY;
        for (const auto &clone_imu : _clones_IMU) {
            if (clone_imu.first < time) {
                time = clone_imu.first;
            }
        }
        return time;
    }

    /**
     * @brief Calculates the current max size of the covariance
     * @return Size of the current covariance matrix
     */
    int max_covariance_size() { return (int)_Cov.rows(); }

    Eigen::MatrixXd get_covariance() { return _Cov; }

    /// Mutex for locking access to the state
    std::mutex _mutex_state;

    /// Current timestamp (should be the last update time in camera clock frame!)
    double _timestamp = -1;

    /// Struct containing filter options
    StateOptions _options;

    /// Pointer to the "active" IMU state (R_ItoG, p_IinG, v_IinG, bg, ba)
    std::shared_ptr<IMU> _imu;

    /// Our current set of SLAM features (3d positions)
    std::unordered_map<size_t, std::shared_ptr<Landmark>> _features_SLAM;

    /// Our map frame to local frame
    std::shared_ptr<PoseHamilton> _pose_MAPtoLOC;

    /// Calibration pose for camera (R_ItoC, p_IinC)
    std::shared_ptr<PoseHamilton> _calib_IMUtoCAM;

    /// Calibration pose for odom (R_ItoO, p_IinC)
    std::shared_ptr<PoseHamilton> _calib_IMUtoOdom;

    /// Camera intrinsics camera objects
    std::shared_ptr<CamBase> _cam_intrinsics_camera;

    /// Camera intrinsics
    std::shared_ptr<Vec> _cam_intrinsics;

    /// Map between imaging times and clone poses (q_GtoIi, p_IiinG)
    std::map<double, std::shared_ptr<PoseHamilton>> _clones_IMU;

    bool _zupt;

    KFCLASS _kf;
    bool _feature_in_clone;
    bool _feature_in_rel_group;
    bool _feature_no_group;

    double _pseudo_map_anchor_time;

  private:
    // Define that the state helper is a friend class of this class
    // This will allow it to access the below functions which should normally not be called
    // This prevents a developer from thinking that the "insert clone" will actually correctly add it to the covariance
    friend class StateHelper;

    /// Covariance of all active variables
    Eigen::MatrixXd _Cov;

    /// Vector of variables
    std::vector<std::shared_ptr<Type>> _variables;
};
} // namespace night_voyager
#endif