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
#ifndef UPDATER_ZEROVELOCITY_H
#define UPDATER_ZEROVELOCITY_H

#include "core/CommonLib.h"
#include "core/NightVoyagerOptions.h"
#include "utils/Transform.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <memory>

namespace night_voyager {

class Feature;
class FeatureDatabase;
class State;
class Propagator;
class Landmark;
class NoiseManager;

class UpdaterZeroVelocity {
  public:
    /**
     * @brief Default constructor for our zero velocity detector and updater.
     * @param options Updater options (chi2 multiplier)
     * @param noises imu noise characteristics (continuous time)
     * @param db Feature tracker database with all features in it
     * @param prop Propagator class object which can predict the state forward in time
     * @param gravity_mag Global gravity magnitude of the system (normally 9.81)
     * @param zupt_max_velocity Max velocity we should consider to do a update with
     * @param zupt_noise_multiplier Multiplier of our IMU noise matrix (default should be 1.0)
     * @param zupt_max_disparity Max disparity we should consider to do a update with
     */
    UpdaterZeroVelocity(const NightVoyagerOptions &options, std::shared_ptr<FeatureDatabase> db, std::shared_ptr<Propagator> prop);

    /**
     * @brief Feed function for inertial data
     * @param message Contains our timestamp and inertial information
     * @param oldest_time Time that we can discard measurements before
     */
    void feed_imu(const ImuData &message, double oldest_time = -1) {

        // Append it to our vector
        imu_data.emplace_back(message);

        // Sort our imu data (handles any out of order measurements)
        // std::sort(imu_data.begin(), imu_data.end(), [](const IMUDATA i, const IMUDATA j) {
        //    return i.timestamp < j.timestamp;
        //});

        // Clean old measurements
        // std::cout << "ZVUPT: imu_data.size() " << imu_data.size() << std::endl;
        clean_old_imu_measurements(oldest_time - 0.10);
    }

    /**
     * @brief This will remove any IMU measurements that are older then the given measurement time
     * @param oldest_time Time that we can discard measurements before (in IMU clock)
     */
    void clean_old_imu_measurements(double oldest_time) {
        if (oldest_time < 0)
            return;
        auto it0 = imu_data.begin();
        while (it0 != imu_data.end()) {
            if (it0->timestamp < oldest_time) {
                it0 = imu_data.erase(it0);
            } else {
                it0++;
            }
        }
    }

    /**
     * @brief Will first detect if the system is zero velocity, then will update.
     * @param state State of the filter
     * @param timestamp Next camera or odom timestamp we want to see if we should propagate to.
     * @return True if the system is currently at zero velocity
     */
    bool try_update(std::shared_ptr<State> state, const PackData &message);

  protected:
    UpdaterOptions _options;

    /// Container for the imu noise values
    NoiseManager _noises;

    /// Feature tracker database with all features in it
    std::shared_ptr<FeatureDatabase> _db;

    /// Our propagator!
    std::shared_ptr<Propagator> _prop;

    /// Max velocity (m/s) that we should consider a zupt with(for odom)
    double _zupt_max_velocity_odom = 0.05;

    /// Max velocity (m/s) that we should consider a zupt with
    double _zupt_max_velocity = 1.0;

    /// Multiplier of our IMU noise matrix (default should be 1.0)
    double _zupt_noise_multiplier = 1.0;

    /// Max disparity (pixels) that we should consider a zupt with
    double _zupt_max_disparity = 1.0;

    /// Gravity vector
    Eigen::Vector3d _gravity;

    /// Chi squared 95th percentile table (lookup would be size of residual)
    std::map<int, double> chi_squared_table;

    /// Our history of IMU messages (time, angular, linear)
    std::vector<ImuData> imu_data;

    /// Last timestamp we did zero velocity update with
    double last_zupt_state_timestamp = 0.0;

    /// Number of times we have called update
    int last_zupt_count = 0;
};
} // namespace night_voyager
#endif