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
#ifndef STATICINITIALIZER_H
#define STATICINITIALIZER_H

#include "core/CommonLib.h"
#include "core/NightVoyagerOptions.h"

namespace night_voyager {
class IMU;
class Type;
class FeatureDatabase;

class StaticInitializer {
  public:
    /**
     * @brief Default constructor
     * @param options Parameters loaded from either ROS or CMDLINE
     * @param db Feature tracker database with all features in it
     * @param imu_data_ Shared pointer to our IMU vector of historical information
     */
    explicit StaticInitializer(InertialInitializerOptions &options, std::shared_ptr<FeatureDatabase> db,
                               std::shared_ptr<std::vector<ImuData>> imu_data_)
        : _options(options), _db(db), imu_data(imu_data_) {}

    /**
     * @brief Try to get the initialized system using just the imu
     *
     * This will check if we have had a large enough jump in our acceleration.
     * If we have then we will use the period of time before this jump to initialize the state.
     * This assumes that our imu is sitting still and is not moving (so this would fail if we are experiencing constant acceleration).
     *
     * In the case that we do not wait for a jump (i.e. `wait_for_jerk` is false), then the system will try to initialize as soon as possible.
     * This is only recommended if you have zero velocity update enabled to handle the stationary cases.
     * To initialize in this case, we need to have the average angular variance be below the set threshold (i.e. we need to be stationary).
     *
     * @param[out] timestamp Timestamp we have initialized the state at
     * @param[out] covariance Calculated covariance of the returned state
     * @param[out] order Order of the covariance matrix
     * @param[out] t_imu Our imu type element
     * @return True if we have successfully initialized our system
     */
    bool initialize(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<Type>> &order, std::shared_ptr<IMU> t_imu);

  private:
    /// Initialization parameters
    InertialInitializerOptions _options;

    /// Feature tracker database with all features in it
    std::shared_ptr<FeatureDatabase> _db;

    /// Our history of IMU messages (time, angular, linear)
    std::shared_ptr<std::vector<ImuData>> imu_data;
};
} // namespace night_voyager
#endif