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
#include "msckf_iekf/VILManager.h"
#include "core/LandMark.h"
#include "core/Type.h"
#include "feature_tracker/Feature.h"
#include "feature_tracker/FeatureDatabase.h"
#include "feature_tracker/TrackKLT.h"
#include "initializer/InertialInitializer.h"
#include "msckf_iekf/State.h"
#include "msckf_iekf/StateHelper.h"
#include "msckf_iekf/UpdaterMAP.h"
#include "msckf_iekf/UpdaterMSCKF.h"
#include "msckf_iekf/UpdaterOdom.h"
#include "msckf_iekf/UpdaterPlane.h"
#include "msckf_iekf/UpdaterSLAM.h"
#include "msckf_iekf/UpdaterZeroVelocity.h"
#include "tracking_recover/TrackingRecover.h"
#include <chrono>
#include <random>
#include <tbb/tbb.h>
#include <thread>

namespace night_voyager {
VILManager::VILManager(NightVoyagerOptions &options, std::shared_ptr<PcdManager> pcd, std::shared_ptr<PriorPoseManager> prpose) : _pcd(pcd), _prpose(prpose) {

    this->options = options;

    // Create the state!!
    state = std::make_shared<State>(options);

    // This will globally set the thread count we will use
    // -1 will reset to the system default threading (usually the num of cores)
    cv::setNumThreads(16);
    cv::setRNGSeed(0);

    // Loop through and load each of the cameras
    state->_cam_intrinsics_camera = options.camera_intrinsic;
    state->_cam_intrinsics->set_value(options.camera_intrinsic->get_value());
    state->_calib_IMUtoCAM->set_value(options.camera_imu_extrinsic);
    state->_calib_IMUtoOdom->set_value(options.odom_imu_extrinsic);

    trackFEATS = std::make_shared<TrackKLT>(options, state->_cam_intrinsics_camera);
    // Our state initialize
    initializer = std::make_shared<InertialInitializer>(options, trackFEATS->get_feature_database(), pcd, prpose, options.camera_intrinsic);
    if (options.try_zupt) {
        updaterZUPT = std::make_shared<UpdaterZeroVelocity>(options, trackFEATS->get_feature_database(), propagator);
    }

    // Initialize our state propagator
    propagator = std::make_shared<Propagator>(options);

    // Make the updater!
    updaterMSCKF = std::make_shared<UpdaterMSCKF>(options);
    updaterSLAM = std::make_shared<UpdaterSLAM>(options);
    updaterOdom = std::make_shared<UpdaterOdom>(options);
    updaterPlane = std::make_shared<UpdaterPlane>(options, prpose);
    updaterMAP = std::make_shared<UpdaterMAP>(options, pcd, prpose, state->_cam_intrinsics_camera);
    tracking_recover = std::make_shared<TrackingRecover>(options, pcd, state->_cam_intrinsics_camera, prpose);

    thread_init_imu_running = false;
    thread_init_imu_success = false;
    thread_init_map_running = false;
    thread_init_map_success = false;
    is_initialized_map = initializer->map_initialized();
    if (state->_kf == KFCLASS::IKF_CHANGEGROUP) {
        state->_feature_in_clone = true;
        state->_feature_in_rel_group = false;
    } else {
        state->_feature_in_clone = false;
        state->_feature_in_rel_group = false;
        if (state->_kf == KFCLASS::IKF_NOGROUP)
            state->_feature_no_group = true;
        else
            state->_feature_no_group = false;
    }
}

void VILManager::feed_measurement_all(const PackData &message) {

    // The oldest time we need IMU with is the last clone
    // We shouldn't really need the whole window, but if we go backwards in time we will
    double oldest_time = state->margtimestep();
    if (oldest_time > state->_timestamp) {
        oldest_time = -1;
    }

    for (const auto imu : message.imus) {

        if (imu.timestamp == message.imus.back().timestamp)
            break;

        if (!is_initialized_imu) {
            oldest_time = imu.timestamp - options.init_options.init_window_time - 0.10;
        }

        propagator->feed_imu(imu, oldest_time);

        // Push back to our initializer
        if (!is_initialized_imu) {
            initializer->feed_imu(imu, -1);
        }

        // Push back to the zero velocity updater if it is enabled
        // No need to push back if we are just doing the zv-update at the begining and we have moved
        if (updaterZUPT != nullptr && (!options.zupt_only_at_beginning || !has_moved_since_zupt)) {
            updaterZUPT->feed_imu(imu, oldest_time);
        }
    }

    rT1 = boost::posix_time::microsec_clock::local_time();
    trackFEATS->feed_new_camera(message.cam);
    rT2 = boost::posix_time::microsec_clock::local_time();

    double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);

    if (!is_initialized_imu) {
        is_initialized_imu = try_to_initialize_imu(message);
        if (!is_initialized_imu)
            return;
    }

    if (tracking_recover->is_tracking_lost() && is_initialized_map) {
        // Check if we should do zero-velocity, if so update the state with it
        // Note that in the case that we only use in the beginning initialization phase
        // If we have since moved, then we should never try to do a zero velocity update!
        if (is_initialized_imu && updaterZUPT != nullptr && (!options.zupt_only_at_beginning || !has_moved_since_zupt)) {
            // If the same state time, use the previous timestep decision
            // Todo Use odom data to update state in ZUPT
            std::vector<bool> did_zupt_updates(tracking_recover->states.size(), false);
            if (state->_timestamp != message.cam.timestamp) {
                tbb::parallel_for(tbb::blocked_range<size_t>(0, tracking_recover->states.size()), [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        did_zupt_updates[i] = updaterZUPT->try_update(tracking_recover->states[i], message);
                    }
                });
            }
            for (size_t i = 0; i < did_zupt_updates.size(); ++i) {
                if (i == 0) {
                    did_zupt_update = did_zupt_updates[i];
                } else {
                    did_zupt_update = did_zupt_update && did_zupt_updates[i];
                }
            }
            if (did_zupt_update) {
                assert(state->_timestamp == message.cam.timestamp);
                propagator->clean_old_imu_measurements(message.cam.timestamp - 0.10);
                updaterZUPT->clean_old_imu_measurements(message.cam.timestamp - 0.10);
                updaterMAP->clean_old_box_measurements(message.cam.timestamp - 0.10);
                return;
            }
        }

        // We first process Odom data
        const vector<OdomData> &odoms = message.odoms;
        const ImuData &last_imu = message.imus.back();

        if (options.do_update_odom) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, tracking_recover->states.size()), [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {

                    // We first process the odom data
                    for (size_t j = 0; j < odoms.size(); ++j) {
                        // We need to first ignore the measurements before current state.
                        // This situation can only happen on the first data exactly after intialization
                        if (odoms[j].timestamp < tracking_recover->states[i]->_timestamp)
                            continue;
                        propagator->propagate(tracking_recover->states[i], odoms[j].timestamp, last_imu);
                        updaterOdom->update(tracking_recover->states[i], odoms[j]);
                    }
                }
            });
        }
        rT3 = boost::posix_time::microsec_clock::local_time();
        double time_odom = (rT3 - rT2).total_microseconds() * 1e-6;
        PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for odom update [tracking_recovering] \n" RESET, time_odom);
        do_feature_propagate_update_tracking_recover(message, last_imu);
    } else {
        // Check if we should do zero-velocity, if so update the state with it
        // Note that in the case that we only use in the beginning initialization phase
        // If we have since moved, then we should never try to do a zero velocity update!
        if (is_initialized_imu && updaterZUPT != nullptr && (!options.zupt_only_at_beginning || !has_moved_since_zupt)) {
            // If the same state time, use the previous timestep decision
            // Todo Use odom data to update state in ZUPT
            if (state->_timestamp != message.cam.timestamp) {
                did_zupt_update = updaterZUPT->try_update(state, message);
            }
            if (did_zupt_update) {
                assert(state->_timestamp == message.cam.timestamp);
                propagator->clean_old_imu_measurements(message.cam.timestamp - 0.10);
                updaterZUPT->clean_old_imu_measurements(message.cam.timestamp - 0.10);
                updaterMAP->clean_old_box_measurements(message.cam.timestamp - 0.10);
                if (!is_initialized_map) {
                    is_initialized_map = try_to_initialize_map(message);
                }
                return;
            }
        }

        // We first process Odom data
        const vector<OdomData> &odoms = message.odoms;
        const ImuData &last_imu = message.imus.back();

        // We first process the odom data
        if (options.do_update_odom) {
            for (size_t i = 0; i < odoms.size(); ++i) {
                // We need to first ignore the measurements before current state.
                // This situation can only happen on the first data exactly after intialization
                if (odoms[i].timestamp < state->_timestamp)
                    continue;
                propagator->propagate(state, odoms[i].timestamp, last_imu);
                updaterOdom->update(state, odoms[i]);
                // We need to wait for enough clones for plane constraint
                // if(options.do_update_plane && state->_clones_IMU.size() >= state->_options.max_clone_size)
                //     updaterPlane->update(state);
            }
        }
        rT3 = boost::posix_time::microsec_clock::local_time();
        double time_odom = (rT3 - rT2).total_microseconds() * 1e-6;
        PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for odom update \n" RESET, time_odom);

        do_feature_propagate_update(message, last_imu);
    }
}

void VILManager::feed_measurement_box(const BoxData &message) {

    // The oldest time we need IMU with is the last clone
    // We shouldn't really need the whole window, but if we go backwards in time we will
    double oldest_time = state->margtimestep();
    if (oldest_time > state->_timestamp) {
        oldest_time = -1;
    }
    if (!is_initialized_imu) {
        oldest_time = message.timestamp - options.init_options.init_window_time - 0.10;
    }
    updaterMAP->feed_box(message, oldest_time);

    // Push back to our initializer
    if (!is_initialized_map) {
        initializer->feed_box(message, oldest_time);
    }
}

bool VILManager::try_to_initialize_imu(const PackData &message) {

    // Directly return if the initialization thread is running
    // Note that we lock on the queue since we could have finished an update
    // And are using this queue to propagate the state forward. We should wait in this case
    if (thread_init_imu_running) {
        std::lock_guard<std::mutex> lck(pack_imu_queue_init_mtx);
        pack_imu_queue_init.emplace_back(message);
        return false;
    }

    // If the thread was a success, then return success!
    if (thread_init_imu_success) {
        return true;
    }

    // Run the initialization in a second thread so it can go as slow as it desires
    thread_init_imu_running = true;
    std::thread thread([&] { init_thread_imu(); });
    thread.detach();
    // cout << "camera: hehe" << endl;
    return false;
}

void VILManager::init_thread_imu() {

    // Returns from our imu initializer
    double timestamp;
    Eigen::MatrixXd covariance_viwo;
    std::vector<std::shared_ptr<Type>> order_viwo;
    auto init_rT1 = boost::posix_time::microsec_clock::local_time();

    initializer->initialize_viwo(timestamp, covariance_viwo, order_viwo, state->_imu);
    if (initializer->viwo_initialized()) {
        StateHelper::set_initial_covariance(state, covariance_viwo, order_viwo);

        // If successfully initialize IMU, set the state time
        state->_timestamp = timestamp;
        startup_time = timestamp;

        // Upate timestamp to the newest camera/odom data
        auto init_rT2 = boost::posix_time::microsec_clock::local_time();
        PRINT_INFO(GREEN "[init-imu]: successful IMU initialization in %.4f seconds\n" RESET, (init_rT2 - init_rT1).total_microseconds() * 1e-6);
        postprocess_after_initviwo();

        trackFEATS->set_num_features(options.num_features);
        thread_init_imu_success = true;
        thread_init_imu_running = false;
        return;
    } else {
        auto init_rT2 = boost::posix_time::microsec_clock::local_time();
        PRINT_INFO(YELLOW "[init-imu]: failed IMU initialization in %.4f seconds\n" RESET, (init_rT2 - init_rT1).total_microseconds() * 1e-6);
        thread_init_imu_success = false;
        thread_init_imu_running = false;
        return;
    }
}

bool VILManager::try_to_initialize_map(const PackData &message) {

    // Directly return if the initialization thread is running
    // Note that we lock on the queue since we could have finished an update
    // And are using this queue to propagate the state forward. We should wait in this case
    if (thread_init_map_running) {
        return false;
    }

    // If the thread was a success, then return success!
    if (thread_init_map_success) {
        return true;
    }

    assert(state->_timestamp == message.cam.timestamp);
    Eigen::Matrix4d T_ItoC = state->_calib_IMUtoCAM->value();
    Eigen::Matrix4d T_ItoG = state->_imu->pose()->value();
    std::vector<std::shared_ptr<Type>> order_viwo_pose = {state->_imu->pose()};
    Eigen::MatrixXd covariance_viwo_pose = StateHelper::get_marginal_covariance(state, order_viwo_pose);

    // Run the initialization in a second thread so it can go as slow as it desires
    thread_init_map_running = true;
    std::thread thread([&, T_ItoC, T_ItoG, covariance_viwo_pose] { init_thread_map(message, T_ItoC, T_ItoG, covariance_viwo_pose); });
    thread.detach();
    // cout << "camera: hehe" << endl;
    return false;
}

void VILManager::init_thread_map(const PackData &message, const Eigen::Matrix4d &T_ItoC, const Eigen::Matrix4d &T_ItoG, const Eigen::MatrixXd &covariance_viwo_pose) {

    // Returns from our initializer
    auto init_rT1 = boost::posix_time::microsec_clock::local_time();

    Eigen::MatrixXd T_MAPtoLOC;
    Eigen::MatrixXd covariance_map = Eigen::MatrixXd::Identity(6, 6);
    initializer->initialize_map(message, T_ItoC, T_ItoG, covariance_viwo_pose, T_MAPtoLOC, covariance_map);
    if (initializer->map_initialized()) {
        // Upate timestamp to the newest camera/odom data
        state->_pose_MAPtoLOC->set_value(T_MAPtoLOC);
        std::vector<std::shared_ptr<Type>> order_map;
        order_map.emplace_back(state->_pose_MAPtoLOC);
        StateHelper::set_initial_covariance(state, covariance_map, order_map);

        auto init_rT2 = boost::posix_time::microsec_clock::local_time();
        PRINT_INFO(YELLOW "[init-map]: successful MAP initialization in %.4f seconds\n" RESET, (init_rT2 - init_rT1).total_microseconds() * 1e-6);
        PRINT_INFO(GREEN "[init-map]: orientation of map in local frame = %.4f, %.4f, %.4f, %.4f\n" RESET, state->_pose_MAPtoLOC->quat().w(),
                   state->_pose_MAPtoLOC->quat().x(), state->_pose_MAPtoLOC->quat().y(), state->_pose_MAPtoLOC->quat().z());
        PRINT_INFO(GREEN "[init-map]: position of map in local frame = %.4f, %.4f, %.4f\n" RESET, state->_pose_MAPtoLOC->pos()(0), state->_pose_MAPtoLOC->pos()(1),
                   state->_pose_MAPtoLOC->pos()(2));

        thread_init_map_success = true;
        thread_init_map_running = false;
        return;
    } else {
        auto init_rT2 = boost::posix_time::microsec_clock::local_time();
        PRINT_INFO(YELLOW "[init-map]: failed MAP initialization in %.4f seconds\n" RESET, (init_rT2 - init_rT1).total_microseconds() * 1e-6);
        thread_init_map_success = false;
        thread_init_map_running = false;
        return;
    }
}

void VILManager::postprocess_after_initviwo() {

    // Cleanup any features older than the initialization time
    // NOTE: we will split the total number of features over all cameras uniformly
    trackFEATS->get_feature_database()->cleanup_measurements(state->_timestamp);

    // Dynamic initialization is not supported yet
    if (state->_imu->vel().norm() > options.zupt_max_velocity) {
        has_moved_since_zupt = true;
        PRINT_ERROR(RED "[init-imu]: IMU moves when performing static IMU initialization!\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    // Else we are good to go, print out our stats
    PRINT_INFO(GREEN "[init-imu]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, state->_imu->quat().w(), state->_imu->quat().x(), state->_imu->quat().y(),
               state->_imu->quat().z());
    PRINT_INFO(GREEN "[init-imu]: bias gyro = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2));
    PRINT_INFO(GREEN "[init-imu]: velocity = %.4f, %.4f, %.4f\n" RESET, state->_imu->vel()(0), state->_imu->vel()(1), state->_imu->vel()(2));
    PRINT_INFO(GREEN "[init-imu]: bias accel = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_a()(0), state->_imu->bias_a()(1), state->_imu->bias_a()(2));
    PRINT_INFO(GREEN "[init-imu]: position = %.4f, %.4f, %.4f\n" RESET, state->_imu->pos()(0), state->_imu->pos()(1), state->_imu->pos()(2));

    // Remove any camera times that are order then the initialized time
    // This can happen if the initialization has taken a while to perform
    std::lock_guard<std::mutex> lck(pack_imu_queue_init_mtx);
    std::deque<PackData> pack_timestamps_to_init;
    for (size_t i = 0; i < pack_imu_queue_init.size(); i++) {
        if (pack_imu_queue_init.at(i).cam.timestamp > state->_timestamp) {
            pack_timestamps_to_init.emplace_back(pack_imu_queue_init.at(i));
        }
    }

    double timestamp = state->_timestamp;
    while (!pack_timestamps_to_init.empty()) {

        const vector<OdomData> &odoms = pack_timestamps_to_init.at(0).odoms;
        const ImuData &imu = pack_timestamps_to_init.at(0).imus.back();

        // We first process the odom data
        for (size_t i = 0; i < odoms.size(); ++i) {
            // We need to first ignore the measurements before current state.
            // This situation can only happen on the first data exactly after intialization
            if (odoms[i].timestamp < state->_timestamp)
                continue;
            propagator->propagate(state, odoms[i].timestamp, imu);
            updaterOdom->update(state, odoms[i]);
        }

        CameraData &cam = pack_timestamps_to_init.at(0).cam;
        propagator->propagate(state, cam.timestamp, imu);
        StateHelper::augment_clone(state);
        StateHelper::marginalize_old_clone(state);
        pack_timestamps_to_init.pop_front();
    }
    PRINT_DEBUG(YELLOW "[init-imu]: moved the state forward %.2f seconds\n" RESET, state->_timestamp - timestamp);
    pack_imu_queue_init.clear();
}

void VILManager::do_feature_propagate_update(const PackData &pack_data, const ImuData &last_imu) {
    //===================================================================================
    // State propagation, and clone augmentation
    //===================================================================================

    const CameraData &message = pack_data.cam;
    // Return if the camera measurement is out of order
    if (state->_timestamp > message.timestamp) {
        PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET, (message.timestamp - state->_timestamp));
        return;
    }

    // Propagate the state forward to the current update time
    // Also augment it with a new clone!
    if (state->_timestamp != message.timestamp) {
        propagator->propagate(state, message.timestamp, last_imu);
        StateHelper::augment_clone(state);
    }
    rT4 = boost::posix_time::microsec_clock::local_time();

    double time_prop = (rT4 - rT3).total_microseconds() * 1e-6;
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation\n" RESET, time_prop);

    // Return if we where unable to propagate
    if (state->_timestamp != message.timestamp) {
        PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
        PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state->_timestamp);
        return;
    }

    // If we have not reached max clones, check if map-based information is available
    // We can start processing things when we have at least 5 clones since we can start triangulating things...
    if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, 5)) {
        PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(), std::min(state->_options.max_clone_size, 5));
    } else if (options.do_update_msckf) {
        //===================================================================================
        // MSCKF features and KLT tracks that are SLAM features
        //===================================================================================

        // Now, lets get all features that should be used for an update that are lost in the newest frame
        // We explicitly request features that have not been deleted (used) in another update step
        std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
        feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(state->_timestamp, false, true);

        // Don't need to get the oldest features until we reach our max number of clones
        if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5) {
            feats_marg = trackFEATS->get_feature_database()->features_containing(state->margtimestep(), false, true);
        }

        // We also need to make sure that the max tracks does not contain any lost features
        // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
        auto it1 = feats_lost.begin();
        while (it1 != feats_lost.end()) {
            if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {
                PRINT_WARNING(YELLOW "FOUND FEATURE THAT WAS IN BOTH feats_lost and feats_marg!!!!!!\n" RESET);
                it1 = feats_lost.erase(it1);
            } else {
                it1++;
            }
        }

        // Find tracks that have reached max length, these can be made into SLAM features
        std::vector<std::shared_ptr<Feature>> feats_maxtracks;
        auto it2 = feats_marg.begin();
        while (it2 != feats_marg.end()) {
            // See if any of our camera's reached max track
            bool reached_max = false;
            if ((int)(*it2)->timestamps.size() > state->_options.max_clone_size) {
                reached_max = true;
            }
            // If max track, then add it to our possible slam feature list
            if (reached_max) {
                feats_maxtracks.push_back(*it2);
                it2 = feats_marg.erase(it2);
            } else {
                it2++;
            }
        }

        // Append a new SLAM feature if we have the room to do so
        // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
        if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= options.state_options.dt_slam_delay &&
            (int)state->_features_SLAM.size() < state->_options.max_slam_features && options.do_update_slam) {
            // Get the total amount to add, then the max amount that we can add given our marginalize feature array
            int amount_to_add = (state->_options.max_slam_features) - (int)state->_features_SLAM.size();
            int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
            // If we have at least 1 that we can add, lets add it!
            // Note: we remove them from the feat_marg array since we don't want to reuse information...
            if (valid_amount > 0) {
                feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
                feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
            }
        }

        // Loop through current SLAM features, we have tracks of them, grab them for this update!
        // NOTE: if we have a slam feature that has lost tracking, then we should marginalize it out
        // NOTE: we only enforce this if the current camera message is where the feature was seen from
        // NOTE: we will also marginalize SLAM features if they have failed their update a couple times in a row
        for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : state->_features_SLAM) {
            std::shared_ptr<Feature> feat = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
            if (feat != nullptr) {
                feats_slam.push_back(feat);
                if (state->_feature_in_clone) {
                    feat->anchor_clone_timestamp = landmark.second->_anchor_clone_timestamp;
                    feat->pseudo_anchor_clone_timestamp = landmark.second->pseudo_anchor_clone_timestamp;
                }
            }
            if (feat == nullptr || landmark.second->update_fail_count > 1)
                landmark.second->should_marg = true;
        }

        // Lets marginalize out all old SLAM features here
        // These are ones that where not successfully tracked into the current frame
        // We do *NOT* marginalize out our aruco tags landmarks
        StateHelper::marginalize_slam(state);

        // Separate our SLAM features into new ones, and old ones
        std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
        for (size_t i = 0; i < feats_slam.size(); i++) {
            if (state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_features_SLAM.end()) {
                feats_slam_UPDATE.push_back(feats_slam.at(i));
                PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d measurements)\n", (int)feats_slam.at(i)->featid, (int)feats_slam.at(i)->timestamps.size());
            } else {
                feats_slam_DELAYED.push_back(feats_slam.at(i));
                PRINT_DEBUG("[UPDATE-SLAM]: new feature ready %d (%d measurements)\n", (int)feats_slam.at(i)->featid, (int)feats_slam.at(i)->timestamps.size());
            }
        }

        // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
        std::vector<std::shared_ptr<Feature>> featsup_MSCKF = feats_lost;
        featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
        featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

        PRINT_INFO(WHITE "Collect %d MSCKF features totally\n" RESET, (int)featsup_MSCKF.size());
        //===================================================================================
        // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
        //===================================================================================

        // Sort based on track length
        // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
        // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
        auto compare_feat = [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
            size_t asize = 0;
            size_t bsize = 0;
            asize += a->timestamps.size();
            bsize += b->timestamps.size();
            return asize < bsize;
        };
        std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

        // Pass them to our MSCKF updater
        // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
        // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
        if ((int)featsup_MSCKF.size() > state->_options.max_msckf_in_update)
            featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - state->_options.max_msckf_in_update);
        updaterMSCKF->update(state, featsup_MSCKF);
        rT5 = boost::posix_time::microsec_clock::local_time();

        // Perform SLAM delay init and update
        // NOTE: that we provide the option here to do a *sequential* update
        // NOTE: this will be a lot faster but won't be as accurate.
        if (options.do_update_slam) {
            // Perform SLAM delay init and update
            // NOTE: that we provide the option here to do a *sequential* update
            // NOTE: this will be a lot faster but won't be as accurate.
            std::vector<std::shared_ptr<Feature>> feats_slam_UPDATE_TEMP;
            PRINT_INFO(WHITE "Collect %d SLAM features totally\n" RESET, (int)feats_slam_UPDATE.size());
            while (!feats_slam_UPDATE.empty()) {
                // Get sub vector of the features we will update with
                std::vector<std::shared_ptr<Feature>> featsup_TEMP;
                featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATE.begin(),
                                    feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
                feats_slam_UPDATE.erase(feats_slam_UPDATE.begin(), feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
                // Do the update
                updaterSLAM->update(state, featsup_TEMP);
                feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
            }
            feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
            rT6 = boost::posix_time::microsec_clock::local_time();
            updaterSLAM->delayed_init(state, feats_slam_DELAYED);
            rT7 = boost::posix_time::microsec_clock::local_time();
        }

        //===================================================================================
        // Update our visualization feature set, and clean up the old features
        //===================================================================================
        // Clear the MSCKF features only on the base camera
        // Thus we should be able to visualize the other unique camera stream
        // MSCKF features as they will also be appended to the vector
        good_features_MSCKF.clear();

        // Save all the MSCKF features used in the update
        for (auto const &feat : featsup_MSCKF) {
            good_features_MSCKF.push_back(feat->p_FinG);
            feat->to_delete = true;
        }

        //===================================================================================
        // Cleanup, marginalize out what we don't need any more...
        //===================================================================================

        // Remove features that where used for the update from our extractors at the last timestep
        // This allows for measurements to be used in the future if they failed to be used this time
        // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
        trackFEATS->get_feature_database()->cleanup();

        // First do anchor change if we are about to lose an anchor pose
        // updaterSLAM->change_pseudo_anchors(state);

        // Cleanup any features older than the marginalization time
        if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
            trackFEATS->get_feature_database()->cleanup_measurements(state->margtimestep());
        }

        rT8 = boost::posix_time::microsec_clock::local_time();

        double time_msckf = (rT5 - rT4).total_microseconds() * 1e-6;
        double time_slam_update = (rT6 - rT5).total_microseconds() * 1e-6;
        double time_slam_delay = (rT7 - rT6).total_microseconds() * 1e-6;
        double time_marg = (rT8 - rT7).total_microseconds() * 1e-6;

        PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats)\n" RESET, time_msckf, (int)featsup_MSCKF.size());
        if (state->_options.max_slam_features > 0) {
            PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM update (%d feats)\n" RESET, time_slam_update, (int)state->_features_SLAM.size());
            PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM delayed init (%d feats)\n" RESET, time_slam_delay, (int)feats_slam_DELAYED.size());
        }
        PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state)\n" RESET, time_marg, (int)state->_clones_IMU.size());
    }

    has_moved_since_zupt = true;

    if (is_initialized_imu && !is_initialized_map) {
        is_initialized_map = try_to_initialize_map(pack_data);
        // Finally marginalize the oldest clone if needed
        if ((state->_feature_in_clone) && (int)state->_clones_IMU.size() > state->_options.max_clone_size)
            StateHelper::change_state(state);
        StateHelper::marginalize_old_clone(state);
        return;
    }

    //===================================================================================
    // Use streetlights to update our state
    //===================================================================================
    // Wait for the detection image coming

    if (options.do_update_map) {
        rT9 = boost::posix_time::microsec_clock::local_time();
        bool street_come = true;
        while (!updaterMAP->findDetection(message.timestamp)) {
            PRINT_INFO(WHITE "[streetlight-detection]: waiting for streetlight detections\n" RESET);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            // boost::posix_time::ptime rT9_1 = boost::posix_time::microsec_clock::local_time();
            // if((rT9_1 - rT9).total_microseconds() * 1e-3 > 15){
            //     street_come = false;
            //     break;
            //     // std::exit(EXIT_FAILURE);
            // }
        }
        rT10 = boost::posix_time::microsec_clock::local_time();
        if (!street_come) {
            PRINT_WARNING(RED "[streetlight-detection]: no streetlight detections, pass this update!\n" RESET);
            updaterMAP->update_for_hist_features(state, message.timestamp);
            // if (options.do_update_plane && state->_clones_IMU.size() >= state->_options.max_clone_size){
            //     // updaterPlane->update(state);
            //     updaterPlane->update_with_prior(state);
            // }
            // return;
        } else {
            PRINT_INFO(WHITE "[streetlight-detection]: find streetlight detections, begin to map update\n" RESET);
            // We first update the history features to smooth the state
            updaterMAP->update_for_hist_features(state, message.timestamp);

            // We have obtained the deep learning-based detection, begin to match
            updaterMAP->preprocess(message);

            // The first match is for deep-learning based streetlight detections
            updaterMAP->match_and_update_dl_features(state, message.timestamp);
            updaterMAP->update_for_dl_features(state, message.timestamp);

            // The second match if for binary-segmentation based streetlight detections
            if (options.use_match_extension) {
                updaterMAP->match_and_update_bi_features(message, state);
                updaterMAP->update_for_bi_features(state, message.timestamp);
            }

            // Finally update our pcd manager for visualization
            updaterMAP->updatepcd();

            // If we have a match, we will change the group of the feature from the clone to the relative group
            if (updaterMAP->exist_matches() && !state->_feature_in_rel_group && state->_feature_in_clone && state->_kf == KFCLASS::IKF_CHANGEGROUP) {
                StateHelper::change_group(state, false);
                state->_feature_in_rel_group = true;
                state->_feature_in_clone = false;
            }
            // If we have no match, we will change the group of the feature from the relative group to the clone
            else if (!updaterMAP->exist_matches() && state->_feature_in_rel_group && !state->_feature_in_clone && state->_kf == KFCLASS::IKF_CHANGEGROUP) {
                StateHelper::change_group(state, true);
                state->_feature_in_rel_group = false;
                state->_feature_in_clone = true;
            }

            if (updaterMAP->matches_num() == 0) {
                // no match in short rotation and position, we will increase the chi2_multipler to permit more matches
                if (tracking_recover->check_tracking_status() == TrackingStatus::TRACKINGSUCCESS) {
                    tracking_recover->change_tracking_status(TrackingStatus::TRACKINGMAYLOST);
                    updaterMAP->set_chi2_multipler_dl(options.tracking_recover_options.chi2_multipler_dl_maylost);
                    updaterMAP->set_chi2_multipler_bi(options.tracking_recover_options.chi2_multipler_bi_maylost);
                }
                // no match in long rotation and position, we will not permit any matches since there may exist wrong matches
                else if (tracking_recover->check_tracking_status() == TrackingStatus::TRACKINGMAYLOST ||
                         tracking_recover->check_tracking_status() == TrackingStatus::TRACKINGUNSTABLESUCCESS) {
                    if (!tracking_recover->check_transform_th())
                        tracking_recover->trans_after_nomatch(state->_imu->Rot(), state->_imu->pos());
                    if (tracking_recover->check_transform_th()) {
                        updaterMAP->set_chi2_multipler_dl(0.0);
                        updaterMAP->set_chi2_multipler_bi(0.0);
                        BoxData st_boxes = updaterMAP->get_cur_stbox();
                        if (!st_boxes.rects.empty() && tracking_recover->traverse_all_situations(state, message, st_boxes, updaterMAP->get_st_database())) {
                            tracking_recover->change_tracking_status(TrackingStatus::TRACKINGLOST);
                            updaterMAP->set_chi2_multipler_dl(options.tracking_recover_options.chi2_multipler_dl_lost);
                            updaterMAP->set_chi2_multipler_bi(options.tracking_recover_options.chi2_multipler_bi_lost);
                        }
                    }
                }
            }
            // only 1 match exists, we will also increase the chi2_multipler to permit more matches, if it lasts for a long movement, the chi2 will be
            // back
            else if (updaterMAP->matches_num() == 1) {
                updaterMAP->set_chi2_multipler_dl(options.tracking_recover_options.chi2_multipler_dl_maylost);
                updaterMAP->set_chi2_multipler_bi(options.tracking_recover_options.chi2_multipler_bi_maylost);
                if (tracking_recover->check_tracking_status() == TrackingStatus::TRACKINGMAYLOST)
                    ;
                tracking_recover->clear_dist();
            } else if (updaterMAP->matches_num() >= 2) {
                updaterMAP->set_chi2_multipler_dl(options.map_options.chi2_multipler_dl);
                updaterMAP->set_chi2_multipler_bi(options.map_options.chi2_multipler_bi);
                if (tracking_recover->check_tracking_status() == TrackingStatus::TRACKINGMAYLOST ||
                    tracking_recover->check_tracking_status() == TrackingStatus::TRACKINGUNSTABLESUCCESS) {
                    tracking_recover->change_tracking_status(TrackingStatus::TRACKINGSUCCESS);
                }
            }

            if (updaterMAP->exist_matches()) {
                // We need to wait for enough clones for plane constraint
                if (options.do_update_plane && state->_clones_IMU.size() >= state->_options.max_clone_size) {
                    // updaterPlane->update(state);
                    rT11 = boost::posix_time::microsec_clock::local_time();
                    updaterPlane->update_with_prior(state);
                    rT12 = boost::posix_time::microsec_clock::local_time();
                    double time_plane = (rT12 - rT11).total_microseconds() * 1e-6;
                    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for plane constraint update \n" RESET, time_plane);
                }
            }

            double time_wait = (rT10 - rT9).total_microseconds() * 1e-6;
            PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for waiting detection \n" RESET, time_wait);

            rT11 = boost::posix_time::microsec_clock::local_time();
            double time_map = (rT11 - rT10).total_microseconds() * 1e-6;
            PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for prior map update \n" RESET, time_map);
        }
    }

    // // //===================================================================================
    // // // Debug info, and stats tracking
    // // //===================================================================================

    // timelastupdate = message.timestamp;
    // Finally marginalize the oldest clone if needed
    if ((state->_feature_in_clone) && (int)state->_clones_IMU.size() > state->_options.max_clone_size)
        StateHelper::change_state(state);
    StateHelper::marginalize_old_clone(state);

    // // Debug, print our current state
    PRINT_INFO("q_ItoG = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f\n", state->_imu->quat().w(), state->_imu->quat().x(), state->_imu->quat().y(), state->_imu->quat().z(),
               state->_imu->pos()(0), state->_imu->pos()(1), state->_imu->pos()(2));
    PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2), state->_imu->bias_a()(0),
               state->_imu->bias_a()(1), state->_imu->bias_a()(2));
    PRINT_INFO("q_MAPtoLOC = %.3f,%.3f,%.3f,%.3f | p_MAPinLOC = %.3f,%.3f,%.3f\n", state->_pose_MAPtoLOC->quat().w(), state->_pose_MAPtoLOC->quat().x(),
               state->_pose_MAPtoLOC->quat().y(), state->_pose_MAPtoLOC->quat().z(), state->_pose_MAPtoLOC->pos()(0), state->_pose_MAPtoLOC->pos()(1),
               state->_pose_MAPtoLOC->pos()(2));
}

void VILManager::do_feature_propagate_update_tracking_recover(const PackData &pack_data, const ImuData &last_imu) {
    //===================================================================================
    // State propagation, and clone augmentation
    //===================================================================================

    const CameraData &message = pack_data.cam;
    // Return if the camera measurement is out of order
    for (size_t i = 0; i < tracking_recover->states.size(); ++i) {
        if (tracking_recover->states[i]->_timestamp > message.timestamp) {
            PRINT_ERROR(RED "image received out of order, unable to do anything (prop dt = %3f) [tracking_recovering]\n" RESET,
                        (message.timestamp - tracking_recover->states[i]->_timestamp));
            std::exit(EXIT_FAILURE);
        }
    }

    // Propagate the state forward to the current update time
    // Also augment it with a new clone!
    // NOTE: if the state is already at the given time (can happen in sim)
    // NOTE: then no need to prop since we already are at the desired timestep
    if (tracking_recover->states[0]->_timestamp != message.timestamp) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, tracking_recover->states.size()), [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                propagator->propagate(tracking_recover->states[i], message.timestamp, last_imu);
                StateHelper::augment_clone(tracking_recover->states[i]);
            }
        });
    }
    rT4 = boost::posix_time::microsec_clock::local_time();

    double time_prop = (rT4 - rT3).total_microseconds() * 1e-6;
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation [tracking_recovering]\n" RESET, time_prop);

    // Return if we where unable to propagate
    for (size_t i = 0; i < tracking_recover->states.size(); ++i) {
        if (tracking_recover->states[i]->_timestamp != message.timestamp) {
            PRINT_ERROR(RED "[PROP]: Propagator unable to propagate the state forward in time! [tracking_recovering]\n" RESET);
            PRINT_ERROR(RED "[PROP]: It has been %.3f since last time we propagated [tracking_recovering]\n" RESET,
                        message.timestamp - tracking_recover->states[i]->_timestamp);
            std::exit(EXIT_FAILURE);
        }

        if ((int)tracking_recover->states[i]->_clones_IMU.size() < std::min(tracking_recover->states[i]->_options.max_clone_size, 5)) {
            PRINT_ERROR(RED "Not enough clone states (%d of %d). Can not transformed to tracking_recovering state\n", (int)tracking_recover->states[i]->_clones_IMU.size(),
                        std::min(tracking_recover->states[i]->_options.max_clone_size, 5));
        }
    }

    //===================================================================================
    // MSCKF features and KLT tracks that are SLAM features
    //===================================================================================

    // Now, lets get all features that should be used for an update that are lost in the newest frame
    // We explicitly request features that have not been deleted (used) in another update step
    if (options.do_update_msckf) {
        std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
        feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(tracking_recover->states[0]->_timestamp, false, true);

        // Don't need to get the oldest features until we reach our max number of clones
        if ((int)tracking_recover->states[0]->_clones_IMU.size() > tracking_recover->states[0]->_options.max_clone_size ||
            (int)tracking_recover->states[0]->_clones_IMU.size() > 5) {
            feats_marg = trackFEATS->get_feature_database()->features_containing(tracking_recover->states[0]->margtimestep(), false, true);
        }

        // We also need to make sure that the max tracks does not contain any lost features
        // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
        auto it1 = feats_lost.begin();
        while (it1 != feats_lost.end()) {
            if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {
                PRINT_WARNING(YELLOW "FOUND FEATURE THAT WAS IN BOTH feats_lost and feats_marg!!!!!!\n" RESET);
                it1 = feats_lost.erase(it1);
            } else {
                it1++;
            }
        }

        // Find tracks that have reached max length, these can be made into SLAM features
        std::vector<std::shared_ptr<Feature>> feats_maxtracks;
        auto it2 = feats_marg.begin();
        while (it2 != feats_marg.end()) {
            // See if any of our camera's reached max track
            bool reached_max = false;
            if ((int)(*it2)->timestamps.size() > tracking_recover->states[0]->_options.max_clone_size) {
                reached_max = true;
            }
            // If max track, then add it to our possible slam feature list
            if (reached_max) {
                feats_maxtracks.push_back(*it2);
                it2 = feats_marg.erase(it2);
            } else {
                it2++;
            }
        }

        std::vector<std::vector<std::shared_ptr<Feature>>> feats_slams;
        std::vector<std::vector<std::shared_ptr<Feature>>> feats_slam_UPDATEs, feats_slam_DELAYEDs;
        feats_slam_UPDATEs.resize(tracking_recover->states.size());
        feats_slam_DELAYEDs.resize(tracking_recover->states.size());
        // Append a new SLAM feature if we have the room to do so
        // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
        if (tracking_recover->states[0]->_options.max_slam_features > 0 && message.timestamp - startup_time >= options.state_options.dt_slam_delay &&
            (int)tracking_recover->states[0]->_features_SLAM.size() < tracking_recover->states[0]->_options.max_slam_features && options.do_update_slam) {
            // Get the total amount to add, then the max amount that we can add given our marginalize feature array
            int amount_to_add = (tracking_recover->states[0]->_options.max_slam_features) - (int)tracking_recover->states[0]->_features_SLAM.size();
            int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
            // If we have at least 1 that we can add, lets add it!
            // Note: we remove them from the feat_marg array since we don't want to reuse information...
            if (valid_amount > 0) {
                feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
                feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
            }

            feats_slams.resize(tracking_recover->states.size());
            for (size_t i = 0; i < tracking_recover->states.size(); ++i) {
                feats_slams[i].resize(feats_slam.size());
                for (size_t j = 0; j < feats_slam.size(); ++j) {
                    feats_slams[i][j] = std::make_shared<Feature>(*feats_slam[j]);
                }
            }
        }

        for (size_t i = 0; i < tracking_recover->states.size(); ++i) {
            // Loop through current SLAM features, we have tracks of them, grab them for this update!
            // NOTE: if we have a slam feature that has lost tracking, then we should marginalize it out
            // NOTE: we only enforce this if the current camera message is where the feature was seen from
            // NOTE: we will also marginalize SLAM features if they have failed their update a couple times in a row
            for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : tracking_recover->states[i]->_features_SLAM) {
                std::shared_ptr<Feature> feat = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
                if (feat != nullptr) {
                    std::shared_ptr<Feature> feat_clone = std::make_shared<Feature>(*feat);
                    feats_slams[i].push_back(feat_clone);
                    if (state->_feature_in_clone) {
                        feat_clone->anchor_clone_timestamp = landmark.second->_anchor_clone_timestamp;
                        feat_clone->pseudo_anchor_clone_timestamp = landmark.second->pseudo_anchor_clone_timestamp;
                    }
                }
                if (feat == nullptr || landmark.second->update_fail_count > 1)
                    landmark.second->should_marg = true;
            }

            // Lets marginalize out all old SLAM features here
            // These are ones that where not successfully tracked into the current frame
            // We do *NOT* marginalize out our aruco tags landmarks
            StateHelper::marginalize_slam(tracking_recover->states[i]);

            // Separate our SLAM features into new ones, and old ones
            std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
            for (size_t j = 0; j < feats_slams[i].size(); j++) {
                if (tracking_recover->states[i]->_features_SLAM.find(feats_slams[i].at(j)->featid) != tracking_recover->states[i]->_features_SLAM.end()) {
                    feats_slam_UPDATE.push_back(feats_slams[i].at(j));
                    // PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d measurements)\n",
                    // (int)feats_slams[i].at(j)->featid,(int)feats_slams[i].at(j)->timestamps.size());
                } else {
                    bool exist = false;
                    for (size_t k = 0; k < feats_slam_DELAYED.size(); k++) {
                        if (feats_slam_DELAYED[k]->featid == feats_slams[i].at(j)->featid) {
                            exist = true;
                            break;
                        }
                    }
                    if (!exist)
                        feats_slam_DELAYED.push_back(feats_slams[i].at(j));
                }
            }

            feats_slam_UPDATEs[i].resize(feats_slam_UPDATE.size());
            feats_slam_DELAYEDs[i].resize(feats_slam_DELAYED.size());
            for (size_t j = 0; j < feats_slam_UPDATE.size(); ++j) {
                feats_slam_UPDATEs[i][j] = std::make_shared<Feature>(*feats_slam_UPDATE[j]);
            }
            for (size_t j = 0; j < feats_slam_DELAYED.size(); ++j) {
                feats_slam_DELAYEDs[i][j] = std::make_shared<Feature>(*feats_slam_DELAYED[j]);
            }
        }

        // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
        std::vector<std::shared_ptr<Feature>> featsup_MSCKF = feats_lost;
        featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
        featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

        PRINT_INFO(WHITE "Collect %d MSCKF features totally [tracking_recovering]\n" RESET, (int)featsup_MSCKF.size());
        //===================================================================================
        // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
        //===================================================================================

        // Sort based on track length
        // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
        // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
        auto compare_feat = [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
            size_t asize = 0;
            size_t bsize = 0;
            asize += a->timestamps.size();
            bsize += b->timestamps.size();
            return asize < bsize;
        };
        std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

        // Pass them to our MSCKF updater
        // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
        // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
        if ((int)featsup_MSCKF.size() > tracking_recover->states[0]->_options.max_msckf_in_update)
            featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - state->_options.max_msckf_in_update);

        std::vector<std::vector<std::shared_ptr<Feature>>> featsup_MSCKFs;
        featsup_MSCKFs.resize(tracking_recover->states.size());
        for (size_t i = 0; i < featsup_MSCKFs.size(); ++i) {
            featsup_MSCKFs[i].resize(featsup_MSCKF.size());
            for (size_t j = 0; j < featsup_MSCKF.size(); ++j) {
                featsup_MSCKFs[i][j] = std::make_shared<Feature>(*featsup_MSCKF[j]);
            }
        }
        tbb::parallel_for(tbb::blocked_range<size_t>(0, tracking_recover->states.size()), [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                updaterMSCKF->update(tracking_recover->states[i], featsup_MSCKFs[i]);
            }
        });

        rT5 = boost::posix_time::microsec_clock::local_time();

        // Perform SLAM delay init and update
        // NOTE: that we provide the option here to do a *sequential* update
        // NOTE: this will be a lot faster but won't be as accurate.
        if (options.do_update_slam) {
            // Perform SLAM delay init and update
            // NOTE: that we provide the option here to do a *sequential* update
            // NOTE: this will be a lot faster but won't be as accurate.
            for (size_t i = 0; i < tracking_recover->states.size(); ++i) {
                std::vector<std::shared_ptr<Feature>> feats_slam_UPDATE_TEMP;
                PRINT_INFO(WHITE "Collect %d SLAM features totally\n" RESET, (int)feats_slam_UPDATEs[i].size());
                while (!feats_slam_UPDATEs[i].empty()) {
                    // Get sub vector of the features we will update with
                    std::vector<std::shared_ptr<Feature>> featsup_TEMP;
                    featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATEs[i].begin(),
                                        feats_slam_UPDATEs[i].begin() + std::min(tracking_recover->states[i]->_options.max_slam_in_update, (int)feats_slam_UPDATEs[i].size()));
                    feats_slam_UPDATEs[i].erase(feats_slam_UPDATEs[i].begin(),
                                                feats_slam_UPDATEs[i].begin() +
                                                    std::min(tracking_recover->states[i]->_options.max_slam_in_update, (int)feats_slam_UPDATEs[i].size()));
                    // Do the update
                    updaterSLAM->update(tracking_recover->states[i], featsup_TEMP);
                    feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
                }
                PRINT_INFO(WHITE "%d SLAM features totally\n" RESET, (int)tracking_recover->states[i]->_features_SLAM.size());
                feats_slam_UPDATEs[i] = feats_slam_UPDATE_TEMP;
                updaterSLAM->delayed_init(tracking_recover->states[i], feats_slam_DELAYEDs[i]);
                PRINT_INFO(WHITE "%d SLAM features totally\n" RESET, (int)tracking_recover->states[i]->_features_SLAM.size());
            }
            rT6 = boost::posix_time::microsec_clock::local_time();
        }

        //===================================================================================
        // Update our visualization feature set, and clean up the old features
        //===================================================================================
        // Clear the MSCKF features only on the base camera
        // Thus we should be able to visualize the other unique camera stream
        // MSCKF features as they will also be appended to the vector
        good_features_MSCKF.clear();

        // Save all the MSCKF features used in the update
        for (auto const &feat : featsup_MSCKFs[0]) {
            good_features_MSCKF.push_back(feat->p_FinG);
            feat->to_delete = true;
        }

        //===================================================================================
        // Cleanup, marginalize out what we don't need any more...
        //===================================================================================

        // Remove features that where used for the update from our extractors at the last timestep
        // This allows for measurements to be used in the future if they failed to be used this time
        // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
        trackFEATS->get_feature_database()->cleanup();

        // Cleanup any features older than the marginalization time
        if ((int)tracking_recover->states[0]->_clones_IMU.size() > tracking_recover->states[0]->_options.max_clone_size) {
            trackFEATS->get_feature_database()->cleanup_measurements(tracking_recover->states[0]->margtimestep());
        }

        double time_msckf = (rT5 - rT4).total_microseconds() * 1e-6;
        PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats) [tracking recovering]\n" RESET, time_msckf, (int)featsup_MSCKF.size());
    }

    // Finally marginalize the oldest clone if needed
    rT5 = boost::posix_time::microsec_clock::local_time();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, tracking_recover->states.size()), [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
            if ((tracking_recover->states[i]->_feature_in_clone) &&
                (int)tracking_recover->states[i]->_clones_IMU.size() > tracking_recover->states[i]->_options.max_clone_size)
                StateHelper::change_state(tracking_recover->states[i]);
            StateHelper::marginalize_old_clone(tracking_recover->states[i]);
        }
    });
    rT6 = boost::posix_time::microsec_clock::local_time();

    double time_marg = (rT6 - rT5).total_microseconds() * 1e-6;

    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state) [tracking recovering]\n" RESET, time_marg,
                (int)tracking_recover->states[0]->_clones_IMU.size());

    has_moved_since_zupt = true;

    assert(is_initialized_imu && is_initialized_map);

    //===================================================================================
    // Use streetlights to update our state
    //===================================================================================
    // Wait for the detection image coming
    bool street_come = true;
    while (!updaterMAP->findDetection(message.timestamp)) {
        PRINT_INFO(WHITE "[streetlight-detection]: waiting for streetlight detections [tracking recovering]\n" RESET);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        // boost::posix_time::ptime rT9_1 = boost::posix_time::microsec_clock::local_time();
        // if((rT9_1 - rT9).total_microseconds() * 1e-3 > 15){
        //     street_come = false;
        //     break;
        //     // std::exit(EXIT_FAILURE);
        // }
    }
    rT7 = boost::posix_time::microsec_clock::local_time();
    if (!street_come) {
        PRINT_WARNING(RED "[streetlight-detection]: no streetlight detections, pass this update! [tracking recovering]\n" RESET);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, tracking_recover->states.size()), [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                updaterMAP->update_for_hist_features_tracking_recover(tracking_recover->states[i], message.timestamp, tracking_recover->st_databases[i]);
                // if (options.do_update_plane && state->_clones_IMU.size() >= state->_options.max_clone_size){
                //     // updaterPlane->update(state);
                //     updaterPlane->update_with_prior(state);
                // }
            }
        });
        return;
    } else {
        PRINT_INFO(WHITE "[streetlight-detection]: find streetlight detections, begin to map update [tracking recovering]\n" RESET);
    }

    // We have obtained the deep learning-based detection, begin to match
    BoxData cur_box = updaterMAP->preprocess(message);

    if (tracking_recover->check_if_select(cur_box)) {
        std::vector<STMatch> matches = tracking_recover->select_best_state(message, cur_box);
        if (!matches.empty()) {
            state = tracking_recover->states[1];
            updaterMAP->set_st_database(tracking_recover->st_databases[1]);
            updaterMAP->set_cur_dl_matches(matches);
            if (matches.size() == 1) {
                tracking_recover->change_tracking_status(TrackingStatus::TRACKINGUNSTABLESUCCESS);
                updaterMAP->set_chi2_multipler_dl(options.tracking_recover_options.chi2_multipler_dl_maylost);
                updaterMAP->set_chi2_multipler_bi(options.tracking_recover_options.chi2_multipler_bi_maylost);
            } else {
                tracking_recover->change_tracking_status(TrackingStatus::TRACKINGSUCCESS);
                updaterMAP->set_chi2_multipler_dl(options.map_options.chi2_multipler_dl);
                updaterMAP->set_chi2_multipler_bi(options.map_options.chi2_multipler_bi);
            }

            // The second match if for binary-segmentation based streetlight detections
            updaterMAP->match_and_update_bi_features(message, state);
            updaterMAP->update_for_bi_features(state, message.timestamp);

            updaterMAP->updatepcd();

            // We need to wait for enough clones for plane constraint
            if (options.do_update_plane && state->_clones_IMU.size() >= state->_options.max_clone_size) {
                // updaterPlane->update(state);
                updaterPlane->update_with_prior(state);
            }

            rT8 = boost::posix_time::microsec_clock::local_time();
            // // //===================================================================================
            // // // Debug info, and stats tracking
            // // //===================================================================================

            // // // Get timing statitics information
            double time_wait = (rT7 - rT6).total_microseconds() * 1e-6;
            double time_map = (rT8 - rT7).total_microseconds() * 1e-6;

            // // Timing information
            PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for waiting detection [tracking recovering]\n" RESET, time_wait);
            PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for prior map update and prior pose update [tracking recovering] \n" RESET, time_map);
            // PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for the whole state update \n" RESET, time_total);

            // timelastupdate = message.timestamp;

            if (state->_kf == KFCLASS::IKF_CHANGEGROUP && !state->_feature_in_clone && state->_feature_in_rel_group) {
                StateHelper::change_group(state, false);
                state->_feature_in_rel_group = true;
                state->_feature_in_clone = false;
            }

            // // Debug, print our current state
            PRINT_INFO("q_ItoG = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f\n", state->_imu->quat().w(), state->_imu->quat().x(), state->_imu->quat().y(),
                       state->_imu->quat().z(), state->_imu->pos()(0), state->_imu->pos()(1), state->_imu->pos()(2));
            PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2), state->_imu->bias_a()(0),
                       state->_imu->bias_a()(1), state->_imu->bias_a()(2));
            PRINT_INFO("q_MAPtoLOC = %.3f,%.3f,%.3f,%.3f | p_MAPinLOC = %.3f,%.3f,%.3f\n", state->_pose_MAPtoLOC->quat().w(), state->_pose_MAPtoLOC->quat().x(),
                       state->_pose_MAPtoLOC->quat().y(), state->_pose_MAPtoLOC->quat().z(), state->_pose_MAPtoLOC->pos()(0), state->_pose_MAPtoLOC->pos()(1),
                       state->_pose_MAPtoLOC->pos()(2));
            return;
        }
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, tracking_recover->states.size()), [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
            updaterMAP->update_for_hist_features_tracking_recover(tracking_recover->states[i], message.timestamp, tracking_recover->st_databases[i]);
            vector<STMatch> cur_dl_matches =
                updaterMAP->match_and_update_for_dl_features_tracking_recover(tracking_recover->states[i], message.timestamp, tracking_recover->st_databases[i],
                                                                              options.tracking_recover_options.z_th, options.tracking_recover_options.extend);
            if (i == 0) {
                // Finally update our pcd manager for visualization
                updaterMAP->updatepcd_tracking_recover(cur_dl_matches);
                updaterMAP->set_cur_dl_matches(cur_dl_matches);
            }

            // We need to wait for enough clones for plane constraint
            if (options.do_update_plane && !cur_dl_matches.empty()) {
                // updaterPlane->update(tracking_recover->states[i]);
                std::map<int, bool> near_indices_to_store = updaterPlane->update_with_prior_tracking_recover(tracking_recover->states[i]);
                if (i == 0) {
                    updaterPlane->add_near_pose(near_indices_to_store);
                }
            }
        }
    });
    tracking_recover->store_poses();

    rT8 = boost::posix_time::microsec_clock::local_time();
    // // //===================================================================================
    // // // Debug info, and stats tracking
    // // //===================================================================================

    // // // Get timing statitics information
    double time_wait = (rT7 - rT6).total_microseconds() * 1e-6;
    double time_map = (rT8 - rT7).total_microseconds() * 1e-6;

    // // Timing information
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for waiting detection [tracking recovering]\n" RESET, time_wait);
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for prior map update and prior pose update [tracking recovering] \n" RESET, time_map);
    // PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for the whole state update \n" RESET, time_total);

    // // Debug, print our current state
    PRINT_INFO("q_ItoG = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f\n", tracking_recover->states[0]->_imu->quat().w(), tracking_recover->states[0]->_imu->quat().x(),
               tracking_recover->states[0]->_imu->quat().y(), tracking_recover->states[0]->_imu->quat().z(), tracking_recover->states[0]->_imu->pos()(0),
               tracking_recover->states[0]->_imu->pos()(1), tracking_recover->states[0]->_imu->pos()(2));
    PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", tracking_recover->states[0]->_imu->bias_g()(0), tracking_recover->states[0]->_imu->bias_g()(1),
               tracking_recover->states[0]->_imu->bias_g()(2), tracking_recover->states[0]->_imu->bias_a()(0), tracking_recover->states[0]->_imu->bias_a()(1),
               tracking_recover->states[0]->_imu->bias_a()(2));
    PRINT_INFO("q_MAPtoLOC = %.3f,%.3f,%.3f,%.3f | p_MAPinLOC = %.3f,%.3f,%.3f\n", tracking_recover->states[0]->_pose_MAPtoLOC->quat().w(),
               tracking_recover->states[0]->_pose_MAPtoLOC->quat().x(), tracking_recover->states[0]->_pose_MAPtoLOC->quat().y(),
               tracking_recover->states[0]->_pose_MAPtoLOC->quat().z(), tracking_recover->states[0]->_pose_MAPtoLOC->pos()(0),
               tracking_recover->states[0]->_pose_MAPtoLOC->pos()(1), tracking_recover->states[0]->_pose_MAPtoLOC->pos()(2));
    return;
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VILManager::get_features_SLAM() {
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> slam_feats;
    for (auto &f : state->_features_SLAM) {
        slam_feats.push_back(f.second->get_xyz());
    }
    return slam_feats;
}

bool VILManager::get_historical_viz_image(cv::Mat &img_history) {
    // Return if not ready yet
    if (state == nullptr || (trackFEATS == nullptr))
        return false;

    // Build an id-list of what features we should highlight (i.e. SLAM)
    std::vector<size_t> highlighted_ids;
    for (const auto &feat : state->_features_SLAM) {
        highlighted_ids.push_back(feat.first);
    }

    // Text we will overlay if needed
    std::string overlay = (did_zupt_update) ? "Zupt" : "Features";
    // overlay = (!is_initialized_imu) ? "IMU_init" : overlay;

    // Get the current active tracks
    trackFEATS->display_history(img_history, 255, 255, 0, 255, 255, 255, highlighted_ids, overlay);

    // Finally return
    return true;
}

bool VILManager::get_streetlight_viz_image_init(cv::Mat &img_streetlight1, cv::Mat &img_streetlight2) {
    // Return if not ready yet
    if (state == nullptr || updaterMAP == nullptr)
        return false;

    // Text we will overlay if needed
    std::string overlay1 = (did_zupt_update) ? "Zupt" : "Streetlight Matches";
    std::string overlay2 = (did_zupt_update) ? "Zupt" : "Streetlight Projection";

    // Get the current matched streetlight detections and centers
    updaterMAP->display_streetlights(state, img_streetlight1, img_streetlight2, overlay1, overlay2);
    return true;
}

bool VILManager::get_streetlight_viz_image_not_init(cv::Mat &img_streetlight) {
    // Return if not ready yet
    if (state == nullptr)
        return false;

    // Text we will overlay if needed
    string overlay = "Initializing";

    // Get the current streetlight detections and centers
    initializer->display_streetlights(img_streetlight, overlay);
    return true;
}

bool VILManager::get_streetlight_viz_image_tracking_recover(cv::Mat &img_streetlight1, cv::Mat &img_streetlight2) {
    // Return if not ready yet
    if (tracking_recover->states[0] == nullptr || updaterMAP == nullptr)
        return false;

    // Text we will overlay if needed
    // std::string overlay = "tracking_recover";
    std::string overlay = "Tracking_recovering";

    // Get the current matched streetlight detections and centers
    updaterMAP->display_streetlights_tracking_recover(tracking_recover->states[0], img_streetlight1, img_streetlight2, overlay);
    return true;
}

bool VILManager::get_other_viz_image_tracking_recover(cv::Mat &img_streetlight) {
    // Return if not ready yet
    if (tracking_recover->states[0] == nullptr || updaterMAP == nullptr)
        return false;

    // Get the current matched streetlight detections and centers
    updaterMAP->display_other_streetlights_tracking_recover(tracking_recover->states, img_streetlight);
    return true;
}

bool VILManager::get_streetlight_detection_viz_image(cv::Mat &img_streetlight) {
    // Return if not ready yet
    if (state == nullptr)
        return false;

    // Text we will overlay if needed
    // std::string overlay = (did_zupt_update) ? "zvupt" : "";
    std::string overlay = "Streetlight Detections";

    // Get the current streetlight detections and centers
    updaterMAP->display_streetlights_detection(img_streetlight, overlay);
    return true;
}

std::shared_ptr<State> VILManager::get_state_tracking_recover() { return tracking_recover->states[0]; }

std::vector<std::shared_ptr<State>> VILManager::get_all_states_tracking_recover() { return tracking_recover->states; }

std::map<double, Eigen::Matrix3d> VILManager::get_rots_loc_in_tracking_lost() { return tracking_recover->rots_loc[1]; }

std::map<double, Eigen::Matrix3d> VILManager::get_rots_global_in_tracking_lost() { return tracking_recover->rots_global[1]; }

std::map<double, Eigen::Vector3d> VILManager::get_poss_loc_in_tracking_lost() { return tracking_recover->poss_loc[1]; }

std::map<double, Eigen::Vector3d> VILManager::get_poss_global_in_tracking_lost() { return tracking_recover->poss_global[1]; }

bool VILManager::is_tracking_lost() { return tracking_recover->is_tracking_lost(); }

} // namespace night_voyager