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
#ifndef INITIALIZER_HELPER_H
#define INITIALIZER_HELPER_H

#include "core/CommonLib.h"
#include <Eigen/Core>

namespace night_voyager {
class InitializerHelper {
  public:
    /**
     * @brief Nice helper function that will linearly interpolate between two imu messages.
     *
     * This should be used instead of just "cutting" imu messages that bound the camera times
     * Give better time offset if we use this function, could try other orders/splines if the imu is slow.
     *
     * @param imu_1 imu at begining of interpolation interval
     * @param imu_2 imu at end of interpolation interval
     * @param timestamp Timestamp being interpolated to
     */
    static ImuData interpolate_data(const ImuData &imu_1, const ImuData &imu_2, double timestamp) {
        double lambda = (timestamp - imu_1.timestamp) / (imu_2.timestamp - imu_1.timestamp);
        ImuData data;
        data.timestamp = timestamp;
        data.am = (1 - lambda) * imu_1.am + lambda * imu_2.am;
        data.wm = (1 - lambda) * imu_1.wm + lambda * imu_2.wm;
        return data;
    }

    /**
     * @brief Helper function that given current imu data, will select imu readings between the two times.
     *
     * This will create measurements that we will integrate with, and an extra measurement at the end.
     * We use the @ref interpolate_data() function to "cut" the imu readings at the beginning and end of the integration.
     * The timestamps passed should already take into account the time offset values.
     *
     * @param imu_data_tmp IMU data we will select measurements from
     * @param time0 Start timestamp
     * @param time1 End timestamp
     * @return Vector of measurements (if we could compute them)
     */
    static std::vector<ImuData> select_imu_readings(const std::vector<ImuData> &imu_data_tmp, double time0, double time1) {
        // Our vector imu readings
        std::vector<ImuData> prop_data;

        // Ensure we have some measurements in the first place!
        if (imu_data_tmp.empty()) {
            return prop_data;
        }

        // Loop through and find all the needed measurements to propagate with
        // Note we split measurements based on the given state time, and the update timestamp
        for (size_t i = 0; i < imu_data_tmp.size() - 1; i++) {

            // START OF THE INTEGRATION PERIOD
            if (imu_data_tmp.at(i + 1).timestamp > time0 && imu_data_tmp.at(i).timestamp < time0) {
                ImuData data = interpolate_data(imu_data_tmp.at(i), imu_data_tmp.at(i + 1), time0);
                prop_data.push_back(data);
                continue;
            }

            // MIDDLE OF INTEGRATION PERIOD
            if (imu_data_tmp.at(i).timestamp >= time0 && imu_data_tmp.at(i + 1).timestamp <= time1) {
                prop_data.push_back(imu_data_tmp.at(i));
                continue;
            }

            // END OF THE INTEGRATION PERIOD
            if (imu_data_tmp.at(i + 1).timestamp > time1) {
                if (imu_data_tmp.at(i).timestamp > time1 && i == 0) {
                    break;
                } else if (imu_data_tmp.at(i).timestamp > time1) {
                    ImuData data = interpolate_data(imu_data_tmp.at(i - 1), imu_data_tmp.at(i), time1);
                    prop_data.push_back(data);
                } else {
                    prop_data.push_back(imu_data_tmp.at(i));
                }
                if (prop_data.at(prop_data.size() - 1).timestamp != time1) {
                    ImuData data = interpolate_data(imu_data_tmp.at(i), imu_data_tmp.at(i + 1), time1);
                    prop_data.push_back(data);
                }
                break;
            }
        }

        // Check that we have at least one measurement to propagate with
        if (prop_data.empty()) {
            return prop_data;
        }

        // Loop through and ensure we do not have an zero dt values
        // This would cause the noise covariance to be Infinity
        for (size_t i = 0; i < prop_data.size() - 1; i++) {
            if (std::abs(prop_data.at(i + 1).timestamp - prop_data.at(i).timestamp) < 1e-12) {
                prop_data.erase(prop_data.begin() + i);
                i--;
            }
        }

        // Success :D
        return prop_data;
    }

    /**
     * @brief Given a gravity vector, compute the rotation from the inertial reference frame to this vector.
     *
     * The key assumption here is that our gravity is along the vertical direction in the inertial frame.
     * We can take this vector (z_in_G=0,0,1) and find two arbitrary tangent directions to it.
     * https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
     *
     * @param gravity_inI Gravity in our sensor frame
     * @param R_GtoI Rotation from the arbitrary inertial reference frame to this gravity vector
     */
    static void gram_schmidt(const Eigen::Vector3d &gravity_inI, Eigen::Matrix3d &R_GtoI) {

        // This will find an orthogonal vector to gravity which is our local z-axis
        // We need to ensure we normalize after each one such that we obtain unit vectors
        Eigen::Vector3d z_axis = gravity_inI / gravity_inI.norm();
        Eigen::Vector3d x_axis, y_axis;
        Eigen::Vector3d e_1(1.0, 0.0, 0.0);
        Eigen::Vector3d e_2(0.0, 1.0, 0.0);
        double inner1 = e_1.dot(z_axis) / z_axis.norm();
        double inner2 = e_2.dot(z_axis) / z_axis.norm();
        if (fabs(inner1) < fabs(inner2)) {
            x_axis = z_axis.cross(e_1);
            x_axis = x_axis / x_axis.norm();
            y_axis = z_axis.cross(x_axis);
            y_axis = y_axis / y_axis.norm();
        } else {
            x_axis = z_axis.cross(e_2);
            x_axis = x_axis / x_axis.norm();
            y_axis = z_axis.cross(x_axis);
            y_axis = y_axis / y_axis.norm();
        }

        // Original method
        // https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
        // x_axis = e_1 - z_axis * z_axis.transpose() * e_1;
        // x_axis = x_axis / x_axis.norm();
        // y_axis = skew_x(z_axis) * x_axis;
        // y_axis = y_axis / y_axis.norm();

        // Rotation from our global (where gravity is only along the z-axis) to the local one
        R_GtoI.block(0, 0, 3, 1) = x_axis;
        R_GtoI.block(0, 1, 3, 1) = y_axis;
        R_GtoI.block(0, 2, 3, 1) = z_axis;

        // cout << "R_GtoI: " << R_GtoI << endl;
    }
};
} // namespace night_voyager
#endif