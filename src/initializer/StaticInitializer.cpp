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
#include "initializer/StaticInitializer.h"
#include "core/IMU.h"
#include "initializer/InitializerHelper.h"
#include "utils/Print.h"

namespace night_voyager {
bool StaticInitializer::initialize(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<Type>> &order,
                                   std::shared_ptr<IMU> t_imu) {
    // Return if we don't have any measurements
    if (imu_data->size() < 2) {
        return false;
    }

    // Newest and oldest imu timestamp
    // double newesttime = imu_data->at(imu_data->size() - 1).timestamp;
    double newesttime = imu_data->at(imu_data->size() - 1).timestamp;
    double oldesttime = imu_data->at(0).timestamp;

    // Return if we don't have enough for two windows
    if (newesttime - oldesttime < _options.init_window_time) {
        PRINT_INFO(YELLOW "[init-s]: unable to select window of IMU readings, not enough readings\n" RESET);
        return false;
    }

    Eigen::Vector3d a_avg = Eigen::Vector3d::Zero();
    Eigen::Vector3d w_avg = Eigen::Vector3d::Zero();
    for (const ImuData &data : *imu_data) {
        a_avg += data.am;
        w_avg += data.wm;
    }
    a_avg = a_avg / imu_data->size();
    w_avg = w_avg / imu_data->size();

    double a_var = 0;
    for (const ImuData &data : *imu_data) {
        a_var += (data.am - a_avg).dot(data.am - a_avg);
    }
    a_var = std::sqrt(a_var / ((int)imu_data->size() - 1));

    // If it is above the threshold and we are not waiting for a jerk
    // Then we are not stationary (i.e. moving) so we should wait till we are
    if (a_var > _options.init_imu_thresh) {
        PRINT_INFO(YELLOW "[init-s]: to much IMU excitation, above threshold %.3f > %.3f\n" RESET, a_var, _options.init_imu_thresh);
        return false;
    }

    // Get rotation with z axis aligned with -g (z_in_G=0,0,1)
    Eigen::Vector3d z_axis = a_avg / a_avg.norm();
    // InitializerHelper::gram_schmidt(z_axis, Ro);
    // Create an x_axis
    Eigen::Vector3d e_1(1, 0, 0);

    // Make x_axis perpendicular to z
    // as z_axis and e_1 are norm vector, so , z_axis.transpose()*e_1 is the length of e1 project into z_axis
    // then it is multiplied by z_axis (which is the direction of z_axis), so that e1, z_axis*z_axis.transpose()*e_1 and
    // e1-z_axis*z_axis.transpose()*e_1 compose a right-angled triangle. so x_axis is perpendicular to z
    Eigen::Vector3d x_axis = e_1 - z_axis * z_axis.transpose() * e_1;
    x_axis = x_axis / x_axis.norm();

    // Get y from the cross product of these two
    Eigen::Vector3d y_axis = skew(z_axis) * x_axis;
    y_axis = y_axis / y_axis.norm();

    // From these axes get rotation
    // Pose_IMU=R_GtoI*Pose_G.
    // we assume the Pose_G is Identity, so Pose_IMU(i.e. Ro) is R_GtoI.
    Eigen::Matrix<double, 3, 3> Ro;
    Ro.block(0, 0, 3, 1) = x_axis;
    Ro.block(0, 1, 3, 1) = y_axis;
    Ro.block(0, 2, 3, 1) = z_axis;

    // Set our biases equal to our noise (subtract our gravity from accelerometer bias)
    Eigen::Vector3d gravity_inG;
    gravity_inG << 0.0, 0.0, _options.gravity_mag;
    Eigen::Vector3d bg = w_avg;
    Eigen::Vector3d ba = a_avg - Ro * gravity_inG;

    // Set our state variables
    timestamp = newesttime;
    Eigen::MatrixXd imu_state = Eigen::MatrixXd::Identity(5, 7);
    imu_state.block(0, 0, 3, 3) = Ro.transpose();
    imu_state.block(0, 5, 3, 1) = bg;
    imu_state.block(0, 6, 3, 1) = ba;
    assert(t_imu != nullptr);
    t_imu->set_value(imu_state);

    // Create base covariance and its covariance ordering
    order.clear();
    order.push_back(t_imu);
    covariance = std::pow(0.02, 2) * Eigen::MatrixXd::Identity(t_imu->size(), t_imu->size());
    covariance.block(0, 0, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity();  // q0.05
    covariance.block(3, 3, 3, 3) = std::pow(0.005, 2) * Eigen::Matrix3d::Identity(); // p0.005
    covariance.block(6, 6, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity();  // v0.05 (static)

    return true;
}
} // namespace night_voyager