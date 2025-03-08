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
#include "msckf_iekf/Propagator.h"
#include "msckf_iekf/State.h"
#include "msckf_iekf/StateHelper.h"
#include "utils/Print.h"
#include "utils/Transform.h"

namespace night_voyager {

void Propagator::propagate(std::shared_ptr<State> state, double timestamp, const ImuData &last_imu) {

    // We should crash if we are trying to propagate backwards
    if (state->_timestamp > timestamp) {
        PRINT_ERROR(RED "Propagator::propagate_and_clone(): Propagation called trying to propagate backwards in time!!!!\n" RESET);
        PRINT_ERROR(RED "Propagator::propagate_and_clone(): desired propagation = %.4f\n" RESET, (timestamp - state->_timestamp));
        std::exit(EXIT_FAILURE);
    }

    // The timestamp of final component of prop_imus must be larger than that of cam
    std::vector<ImuData> prop_data;
    double time0 = state->_timestamp;
    prop_data = Propagator::select_imu_readings(imu_data, time0, timestamp, last_imu);
    if (prop_data.empty()) {
        PRINT_ERROR(RED "Propagator::propagate_and_clone(): No IMU data is selected for propagation!!\n" RESET);
        std::exit(EXIT_FAILURE);
    } else {
        // We are going to sum up all the state transition matrices, so we can do a single large multiplication at the end
        int size_imu = state->_imu->size();
        assert(size_imu == 15);
        int size_landmarks = 0;
        if (state->_kf == KFCLASS::IKF_IMUGROUP)
            size_landmarks = state->_features_SLAM.size() * 3;

        int prop_size = size_imu + size_landmarks;
        Eigen::MatrixXd Phi_summed = Eigen::MatrixXd::Identity(prop_size, prop_size);
        Eigen::MatrixXd Qd_summed = Eigen::MatrixXd::Zero(prop_size, prop_size);

        double dt_summed = 0;

        // Loop through all IMU messages, and use them to move the state forward in time
        // This uses the zero'th order quat, and then constant acceleration discrete
        if (prop_data.size() > 1) {
            for (size_t i = 0; i < prop_data.size() - 1; i++) {

                // Get the next state Jacobian and noise Jacobian for this IMU reading
                Eigen::MatrixXd F, Qdi;
                if (state->_kf == KFCLASS::MSCKF)
                    predict_and_compute_MSCKF(state, prop_data.at(i), prop_data.at(i + 1), F, Qdi);
                else
                    predict_and_compute(state, prop_data.at(i), prop_data.at(i + 1), F, Qdi);

                // Next we should propagate our IMU covariance
                // Pii' = F*Pii*F.transpose() + G*Q*G.transpose()
                // Pci' = F*Pci and Pic' = Pic*F.transpose()
                // NOTE: Here we are summing the state transition F so we can do a single mutiplication later
                // NOTE: Phi_summed = Phi_i*Phi_summed
                // NOTE: Q_summed = Phi_i*Q_summed*Phi_i^T + G*Q_i*G^T
                Phi_summed = F * Phi_summed;
                Qd_summed = F * Qd_summed * F.transpose() + Qdi;
                Qd_summed = 0.5 * (Qd_summed + Qd_summed.transpose());
                dt_summed += prop_data.at(i + 1).timestamp - prop_data.at(i).timestamp;
            }
        }
        assert(std::abs((timestamp - time0) - dt_summed) < 1e-4);

        std::vector<std::shared_ptr<Type>> Phi_order;
        Phi_order.push_back(state->_imu);

        if (state->_kf == KFCLASS::IKF_IMUGROUP) {
            if (!state->_features_SLAM.empty()) {
                auto iter = state->_features_SLAM.begin();
                while (iter != state->_features_SLAM.end()) {
                    Phi_order.push_back(iter->second);
                    iter++;
                }
            }
        }

        StateHelper::EKFPropagation(state, Phi_order, Phi_order, Phi_summed, Qd_summed);

        // Set timestamp data
        state->_timestamp = timestamp;
    }
    return;
}

std::vector<ImuData> Propagator::select_imu_readings(const std::vector<ImuData> &imu_data, double time0, double time1, const ImuData &last_imu) {
    // Our vector imu readings
    std::vector<ImuData> prop_data;

    // Ensure we have some measurements in the first place!
    if (imu_data.empty()) {
        PRINT_WARNING(YELLOW "Propagator::select_imu_readings(): No IMU measurements. IMU-CAMERA are likely messed up!!!\n" RESET);
        return prop_data;
    }

    if (imu_data[imu_data.size() - 1].timestamp <= time0) {
        PRINT_INFO(
            YELLOW
            "Propagator::select_imu_readings(): All timestamps of IMU measurements (%.10f) are less than current state timestamp (%.10f)!!!\n" RESET,
            imu_data[imu_data.size() - 1].timestamp, time0);
        if (last_imu.timestamp <= time0)
            return prop_data;
    }

    // Loop through and find all the needed measurements to propagate with
    // Note we split measurements based on the given state time, and the update timestamp
    for (size_t i = 0; i < imu_data.size(); i++) {

        // START OF THE INTEGRATION PERIOD
        // If the next timestamp is greater then our current state time
        // And the current is not greater then it yet...
        // Then we should "split" our current IMU measurement
        ImuData next_data;
        bool is_last_imu;
        if (i >= imu_data.size() - 1) {
            next_data = last_imu;
            is_last_imu = true;
        } else {
            next_data = imu_data[i + 1];
            is_last_imu = false;
        }

        if (next_data.timestamp > time0 && imu_data.at(i).timestamp <= time0) {
            ImuData data = Propagator::interpolate_data(imu_data.at(i), next_data, time0);
            prop_data.push_back(data);
            // PRINT_DEBUG("propagation #%d = CASE 1 = %.3f => %.3f\n", (int)i, data.timestamp - prop_data.at(0).timestamp,
            //             time0 - prop_data.at(0).timestamp);
            continue;
        }

        // MIDDLE OF INTEGRATION PERIOD
        // If our imu measurement is right in the middle of our propagation period
        // Then we should just append the whole measurement time to our propagation vector
        if (imu_data.at(i).timestamp >= time0 && next_data.timestamp <= time1) {
            if (is_last_imu) {
                PRINT_ERROR(RED "Propagator::select_and_feed_imu_readings(): the timestamp of the last IMU frame (%.10f) is less than current "
                                "measurment timestamp (%.10f)!!!\n" RESET,
                            next_data.timestamp, time1);
                std::exit(EXIT_FAILURE);
            }
            prop_data.push_back(imu_data.at(i));
            // PRINT_DEBUG("propagation #%d = CASE 2 = %.3f\n", (int)i, imu_data.at(i).timestamp - prop_data.at(0).timestamp);
            continue;
        }

        // END OF THE INTEGRATION PERIOD
        // If the current timestamp is greater then our update time
        // We should just "split" the NEXT IMU measurement to the update time,
        // NOTE: we add the current time, and then the time at the end of the interval (so we can get a dt)
        // NOTE: we also break out of this loop, as this is the last IMU measurement we need!
        if (next_data.timestamp > time1) {
            // If we have a very low frequency IMU then, we could have only recorded the first integration (i.e. case 1) and nothing else
            // In this case, both the current IMU measurement and the next is greater than the desired intepolation, thus we should just cut the
            // current at the desired time Else, we have hit CASE2 and this IMU measurement is not past the desired propagation time, thus add the
            // whole IMU reading
            if (imu_data.at(i).timestamp > time1 && i == 0) {
                // This case can happen if we don't have any imu data that has occured before the startup time
                // This means that either we have dropped IMU data, or we have not gotten enough.
                // In this case we can't propgate forward in time, so there is not that much we can do.
                break;
            } else if (imu_data.at(i).timestamp > time1) {
                ImuData data = interpolate_data(imu_data.at(i - 1), imu_data.at(i), time1);
                prop_data.push_back(data);
                // PRINT_DEBUG("propagation #%d = CASE 3.1 = %.3f => %.3f\n", (int)i, imu_data.at(i).timestamp - prop_data.at(0).timestamp,
                //             imu_data.at(i).timestamp - time0);
            } else {
                prop_data.push_back(imu_data.at(i));
                // PRINT_DEBUG("propagation #%d = CASE 3.2 = %.3f => %.3f\n", (int)i, imu_data.at(i).timestamp - prop_data.at(0).timestamp,
                //             imu_data.at(i).timestamp - time0);
            }
            // If the added IMU message doesn't end exactly at the camera time
            // Then we need to add another one that is right at the ending time
            if (prop_data.at(prop_data.size() - 1).timestamp != time1) {
                ImuData data = interpolate_data(imu_data.at(i), next_data, time1);
                prop_data.push_back(data);
                // PRINT_DEBUG("propagation #%d = CASE 3.3 = %.3f => %.3f\n", (int)i, data.timestamp - prop_data.at(0).timestamp,
                //             data.timestamp - time0);
            }
            break;
        }
    }

    // Check that we have at least one measurement to propagate with
    if (prop_data.empty()) {
        PRINT_WARNING(
            YELLOW
            "Propagator::select_imu_readings(): No IMU measurements to propagate with (%d of 2). IMU-CAMERA/ODOM are likely messed up!!!\n" RESET,
            (int)prop_data.size());
        return prop_data;
    }

    // If we did not reach the whole integration period
    // (i.e., the last inertial measurement we have is smaller then the time we want to reach)
    // Then we should just "stretch" the last measurement to be the whole period
    // TODO: this really isn't that good of logic, we should fix this so the above logic is exact!
    // No exterporlate
    if (prop_data.at(prop_data.size() - 1).timestamp != time1) {
        PRINT_DEBUG(YELLOW "Propagator::select_imu_readings(): Missing inertial measurements to propagate with (%f sec missing)!\n" RESET,
                    (time1 - imu_data.at(imu_data.size() - 1).timestamp));
        // ImuData data = interpolate_data(imu_data.at(imu_data.size() - 2), imu_data.at(imu_data.size() - 1), time1);
        ImuData data = interpolate_data(imu_data.at(imu_data.size() - 1), last_imu, time1);
        prop_data.push_back(data);
        // PRINT_DEBUG("propagation #%d = CASE 3.4 = %.3f => %.3f\n", (int)(imu_data.size() - 2), data.timestamp - prop_data.at(0).timestamp,
        // data.timestamp - time0);
    }

    // Loop through and ensure we do not have any zero dt values
    // This would cause the noise covariance to be Infinity
    // TODO: we should actually fix this by properly implementing this function and doing unit tests on it...
    for (size_t i = 0; i < prop_data.size() - 1; i++) {
        if (std::abs(prop_data.at(i + 1).timestamp - prop_data.at(i).timestamp) < 1e-12) {
            PRINT_WARNING(YELLOW "Propagator::select_imu_readings(): Zero DT between IMU reading %d and %d, removing it!\n" RESET, (int)i,
                          (int)(i + 1));
            prop_data.erase(prop_data.begin() + i);
            i--;
        }
    }

    // Check that we have at least one measurement to propagate with
    if (prop_data.size() < 2) {
        PRINT_WARNING(
            YELLOW
            "Propagator::select_imu_readings(): No IMU measurements to propagate with (%d of 2). IMU-CAMERA/ODOM are likely messed up!!!\n" RESET,
            (int)prop_data.size());
        return prop_data;
    }

    // Success :D
    return prop_data;
}

void Propagator::predict_mean_analytic(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                       Eigen::Matrix3d &new_R, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p, Eigen::Matrix<double, 3, 18> &Xi_sum) {

    // Pre-compute things
    Eigen::Matrix3d R_ktoG = state->_imu->Rot();
    Eigen::Matrix3d R_k1tok = Xi_sum.block(0, 0, 3, 3);
    Eigen::Matrix3d Xi_1 = Xi_sum.block(0, 3, 3, 3);
    Eigen::Matrix3d Xi_2 = Xi_sum.block(0, 6, 3, 3);

    // Use our integrated Xi's to move the state forward
    new_R = R_ktoG * R_k1tok;
    new_v = state->_imu->vel() + R_ktoG * Xi_1 * a_hat - gravity * dt;
    new_p = state->_imu->pos() + state->_imu->vel() * dt + R_ktoG * Xi_2 * a_hat - 0.5 * gravity * dt * dt;
}

void Propagator::predict_mean_rk4(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat1, const Eigen::Vector3d &a_hat1,
                                  const Eigen::Vector3d &w_hat2, const Eigen::Vector3d &a_hat2, Eigen::Matrix3d &new_R, Eigen::Vector3d &new_v,
                                  Eigen::Vector3d &new_p) {

    // Pre-compute things
    Eigen::Vector3d w_hat = w_hat1;
    Eigen::Vector3d a_hat = a_hat1;
    Eigen::Vector3d w_alpha = (w_hat2 - w_hat1) / dt;
    Eigen::Vector3d a_jerk = (a_hat2 - a_hat1) / dt;

    // y0 ================
    Eigen::Vector4d q_0 = Eigen::Quaterniond(state->_imu->Rot().transpose()).coeffs();
    Eigen::Vector3d p_0 = state->_imu->pos();
    Eigen::Vector3d v_0 = state->_imu->vel();

    // k1 ================
    Eigen::Vector4d dq_0 = {0, 0, 0, 1};
    Eigen::Vector4d q0_dot = 0.5 * Omega(w_hat) * dq_0;
    Eigen::Vector3d p0_dot = v_0;
    Eigen::Matrix3d R_Gto0 = quat_2_Rot(quat_multiply(dq_0, q_0));
    Eigen::Vector3d v0_dot = R_Gto0.transpose() * a_hat - gravity;

    Eigen::Vector4d k1_q = q0_dot * dt;
    Eigen::Vector3d k1_p = p0_dot * dt;
    Eigen::Vector3d k1_v = v0_dot * dt;

    // k2 ================
    w_hat += 0.5 * w_alpha * dt;
    a_hat += 0.5 * a_jerk * dt;

    Eigen::Vector4d dq_1 = quatnorm(dq_0 + 0.5 * k1_q);
    // Eigen::Vector3d p_1 = p_0+0.5*k1_p;
    Eigen::Vector3d v_1 = v_0 + 0.5 * k1_v;

    Eigen::Vector4d q1_dot = 0.5 * Omega(w_hat) * dq_1;
    Eigen::Vector3d p1_dot = v_1;
    Eigen::Matrix3d R_Gto1 = quat_2_Rot(quat_multiply(dq_1, q_0));
    Eigen::Vector3d v1_dot = R_Gto1.transpose() * a_hat - gravity;

    Eigen::Vector4d k2_q = q1_dot * dt;
    Eigen::Vector3d k2_p = p1_dot * dt;
    Eigen::Vector3d k2_v = v1_dot * dt;

    // k3 ================
    Eigen::Vector4d dq_2 = quatnorm(dq_0 + 0.5 * k2_q);
    // Eigen::Vector3d p_2 = p_0+0.5*k2_p;
    Eigen::Vector3d v_2 = v_0 + 0.5 * k2_v;

    Eigen::Vector4d q2_dot = 0.5 * Omega(w_hat) * dq_2;
    Eigen::Vector3d p2_dot = v_2;
    Eigen::Matrix3d R_Gto2 = quat_2_Rot(quat_multiply(dq_2, q_0));
    Eigen::Vector3d v2_dot = R_Gto2.transpose() * a_hat - gravity;

    Eigen::Vector4d k3_q = q2_dot * dt;
    Eigen::Vector3d k3_p = p2_dot * dt;
    Eigen::Vector3d k3_v = v2_dot * dt;

    // k4 ================
    w_hat += 0.5 * w_alpha * dt;
    a_hat += 0.5 * a_jerk * dt;

    Eigen::Vector4d dq_3 = quatnorm(dq_0 + k3_q);
    // Eigen::Vector3d p_3 = p_0+k3_p;
    Eigen::Vector3d v_3 = v_0 + k3_v;

    Eigen::Vector4d q3_dot = 0.5 * Omega(w_hat) * dq_3;
    Eigen::Vector3d p3_dot = v_3;
    Eigen::Matrix3d R_Gto3 = quat_2_Rot(quat_multiply(dq_3, q_0));
    Eigen::Vector3d v3_dot = R_Gto3.transpose() * a_hat - gravity;

    Eigen::Vector4d k4_q = q3_dot * dt;
    Eigen::Vector3d k4_p = p3_dot * dt;
    Eigen::Vector3d k4_v = v3_dot * dt;

    // y+dt ================
    Eigen::Vector4d dq = quatnorm(dq_0 + (1.0 / 6.0) * k1_q + (1.0 / 3.0) * k2_q + (1.0 / 3.0) * k3_q + (1.0 / 6.0) * k4_q);
    Eigen::Vector4d new_q = quat_multiply(dq, q_0);
    new_R = Eigen::Quaterniond(new_q(3), new_q(0), new_q(1), new_q(2)).toRotationMatrix().transpose();
    new_p = p_0 + (1.0 / 6.0) * k1_p + (1.0 / 3.0) * k2_p + (1.0 / 3.0) * k3_p + (1.0 / 6.0) * k4_p;
    new_v = v_0 + (1.0 / 6.0) * k1_v + (1.0 / 3.0) * k2_v + (1.0 / 3.0) * k3_v + (1.0 / 6.0) * k4_v;
}

void Propagator::compute_Xi_sum(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                Eigen::Matrix<double, 3, 18> &Xi_sum) {

    // Decompose our angular velocity into a direction and amount
    double w_norm = w_hat.norm();
    double d_th = w_norm * dt;
    Eigen::Vector3d k_hat = Eigen::Vector3d::Zero();
    if (w_norm > 1e-12) {
        k_hat = w_hat / w_norm;
    }

    // Compute useful identities used throughout
    Eigen::Matrix3d I_3x3 = Eigen::Matrix3d::Identity();
    double d_t2 = std::pow(dt, 2);
    double d_t3 = std::pow(dt, 3);
    double w_norm2 = std::pow(w_norm, 2);
    double w_norm3 = std::pow(w_norm, 3);
    double cos_dth = std::cos(d_th);
    double sin_dth = std::sin(d_th);
    double d_th2 = std::pow(d_th, 2);
    double d_th3 = std::pow(d_th, 3);
    Eigen::Matrix3d sK = skew(k_hat);
    Eigen::Matrix3d sK2 = sK * sK;
    Eigen::Matrix3d sA = skew(a_hat);

    // Integration components will be used later
    Eigen::Matrix3d R_k1tok, Xi_1, Xi_2, Jr_k1tok, Xi_3, Xi_4;
    R_k1tok = Exp_SO3(w_hat * dt);
    Jr_k1tok = Jr_SO3(w_hat * dt);

    // Now begin the integration of each component
    // Based on the delta theta, let's decide which integration will be used
    bool small_w = (w_norm < 1.0 / 180 * M_PI / 2);
    if (!small_w) {

        // first order rotation integration with constant omega
        Xi_1 = I_3x3 * dt + (1.0 - cos_dth) / w_norm * sK + (dt - sin_dth / w_norm) * sK2;

        // second order rotation integration with constant omega
        Xi_2 = 1.0 / 2 * d_t2 * I_3x3 + (d_th - sin_dth) / w_norm2 * sK + (1.0 / 2 * d_t2 - (1.0 - cos_dth) / w_norm2) * sK2;

        // first order integration with constant omega and constant acc
        Xi_3 = 1.0 / 2 * d_t2 * sA + (sin_dth - d_th) / w_norm2 * sA * sK + (sin_dth - d_th * cos_dth) / w_norm2 * sK * sA +
               (1.0 / 2 * d_t2 - (1.0 - cos_dth) / w_norm2) * sA * sK2 +
               (1.0 / 2 * d_t2 + (1.0 - cos_dth - d_th * sin_dth) / w_norm2) * (sK2 * sA + k_hat.dot(a_hat) * sK) -
               (3 * sin_dth - 2 * d_th - d_th * cos_dth) / w_norm2 * k_hat.dot(a_hat) * sK2;

        // second order integration with constant omega and constant acc
        Xi_4 = 1.0 / 6 * d_t3 * sA + (2 * (1.0 - cos_dth) - d_th2) / (2 * w_norm3) * sA * sK +
               ((2 * (1.0 - cos_dth) - d_th * sin_dth) / w_norm3) * sK * sA + ((sin_dth - d_th) / w_norm3 + d_t3 / 6) * sA * sK2 +
               ((d_th - 2 * sin_dth + 1.0 / 6 * d_th3 + d_th * cos_dth) / w_norm3) * (sK2 * sA + k_hat.dot(a_hat) * sK) +
               (4 * cos_dth - 4 + d_th2 + d_th * sin_dth) / w_norm3 * k_hat.dot(a_hat) * sK2;

    } else {

        // first order rotation integration with constant omega
        Xi_1 = dt * (I_3x3 + sin_dth * sK + (1.0 - cos_dth) * sK2);

        // second order rotation integration with constant omega
        Xi_2 = 1.0 / 2 * dt * Xi_1;

        // first order integration with constant omega and constant acc
        Xi_3 = 1.0 / 2 * d_t2 *
               (sA + sin_dth * (-sA * sK + sK * sA + k_hat.dot(a_hat) * sK2) + (1.0 - cos_dth) * (sA * sK2 + sK2 * sA + k_hat.dot(a_hat) * sK));

        // second order integration with constant omega and constant acc
        Xi_4 = 1.0 / 3 * dt * Xi_3;
    }

    // Store the integrated parameters
    Xi_sum.setZero();
    Xi_sum.block(0, 0, 3, 3) = R_k1tok;
    Xi_sum.block(0, 3, 3, 3) = Xi_1;
    Xi_sum.block(0, 6, 3, 3) = Xi_2;
    Xi_sum.block(0, 9, 3, 3) = Jr_k1tok;
    Xi_sum.block(0, 12, 3, 3) = Xi_3;
    Xi_sum.block(0, 15, 3, 3) = Xi_4;
}

void Propagator::compute_F_and_G_analytic(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                          const Eigen::Matrix3d &new_R, const Eigen::Vector3d &new_v, const Eigen::Vector3d &new_p,
                                          const Eigen::Matrix<double, 3, 18> &Xi_sum, Eigen::MatrixXd &F, Eigen::MatrixXd &G) {
    // Get the locations of each entry of the imu state
    int local_size = 0;
    int th_id = local_size;
    local_size += state->_imu->R()->size();
    int p_id = local_size;
    local_size += state->_imu->p()->size();
    int v_id = local_size;
    local_size += state->_imu->v()->size();
    int bg_id = local_size;
    local_size += state->_imu->bg()->size();
    int ba_id = local_size;
    local_size += state->_imu->ba()->size();

    // The change in the orientation from the end of the last prop to the current prop
    // This is needed since we need to include the "k-th" updated orientation information
    Eigen::Matrix3d R_k = state->_imu->Rot();

    Eigen::Matrix3d Xi_1 = Xi_sum.block(0, 3, 3, 3);
    Eigen::Matrix3d Xi_2 = Xi_sum.block(0, 6, 3, 3);
    Eigen::Matrix3d Jr_k1tok = Xi_sum.block(0, 9, 3, 3);
    Eigen::Matrix3d Xi_3 = Xi_sum.block(0, 12, 3, 3);
    Eigen::Matrix3d Xi_4 = Xi_sum.block(0, 15, 3, 3);

    // for th
    F.block(th_id, th_id, 3, 3) = Eigen::Matrix3d::Identity();
    F.block(p_id, th_id, 3, 3) = -0.5 * skew(gravity) * dt * dt;
    F.block(v_id, th_id, 3, 3) = -skew(gravity) * dt;

    // for p
    F.block(p_id, p_id, 3, 3).setIdentity();

    // for v
    F.block(p_id, v_id, 3, 3) = Eigen::Matrix3d::Identity() * dt;
    F.block(v_id, v_id, 3, 3).setIdentity();

    // for bg
    Eigen::Matrix3d tmp = new_R * Jr_k1tok * dt;
    F.block(th_id, bg_id, 3, 3) = -tmp;
    F.block(p_id, bg_id, 3, 3) = -skew(new_p) * tmp + R_k * Xi_4;
    F.block(v_id, bg_id, 3, 3) = -skew(new_v) * tmp + R_k * Xi_3;
    F.block(bg_id, bg_id, 3, 3).setIdentity();

    // for ba
    F.block(p_id, ba_id, 3, 3) = -R_k * Xi_2;
    F.block(v_id, ba_id, 3, 3) = -R_k * Xi_1;
    F.block(ba_id, ba_id, 3, 3).setIdentity();

    // construct the G part
    G.block(th_id, th_id, 9, 6) = F.block(0, bg_id, 9, 6);
    G.block(bg_id, 6, 3, 3) = dt * Eigen::Matrix3d::Identity();
    G.block(ba_id, 9, 3, 3) = dt * Eigen::Matrix3d::Identity();

    if (!state->_features_SLAM.empty() && state->_kf == KFCLASS::IKF_IMUGROUP) {
        auto iter = state->_features_SLAM.begin();
        while (iter != state->_features_SLAM.end()) {
            int pf_id = local_size;
            local_size += iter->second->size();
            // cout << iter->second->get_xyz().transpose() << endl;
            F.block(pf_id, bg_id, 3, 3) = -skew(iter->second->get_xyz()) * tmp;
            F.block(pf_id, pf_id, 3, 3).setIdentity();
            G.block(pf_id, th_id, 3, 3) = F.block(pf_id, bg_id, 3, 3);
            ++iter;
        }
    }

    // if (state->_transform_as_group){
    //     int p_trans_id = local_size;
    //     local_size += state->_pose_MAPtoLOC->p()->size();
    //     F.block(p_trans_id, bg_id, 3, 3) = - skew(state->_pose_MAPtoLOC->pos()) * tmp;
    //     F.block(p_trans_id, p_trans_id, 3, 3).setIdentity();
    //     G.block(p_trans_id, th_id, 3, 3) = F.block(p_trans_id, bg_id, 3, 3);
    // }

    // if (state->_multistate_as_group){
    //     auto iter = state->_clones_IMU.begin();
    //     while (iter != state->_clones_IMU.end())
    //     {
    //         int p_multi_id = local_size;
    //         local_size += iter->second->p()->size();
    //         F.block(p_multi_id, bg_id, 3, 3) = -skew(iter->second->pos()) * tmp;
    //         F.block(p_multi_id, p_multi_id, 3, 3).setIdentity();
    //         G.block(p_multi_id, th_id, 3, 3) = F.block(p_multi_id, bg_id, 3, 3);
    //         ++iter;
    //     }
    // }
}

void Propagator::compute_F_and_G_analytic_MSCKF(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                                const Eigen::Matrix3d &new_R, const Eigen::Vector3d &new_v, const Eigen::Vector3d &new_p,
                                                const Eigen::Matrix<double, 3, 18> &Xi_sum, Eigen::MatrixXd &F, Eigen::MatrixXd &G) {
    // Get the locations of each entry of the imu state
    int local_size = 0;
    int th_id = local_size;
    local_size += state->_imu->R()->size();
    int p_id = local_size;
    local_size += state->_imu->p()->size();
    int v_id = local_size;
    local_size += state->_imu->v()->size();
    int bg_id = local_size;
    local_size += state->_imu->bg()->size();
    int ba_id = local_size;
    local_size += state->_imu->ba()->size();

    // The change in the orientation from the end of the last prop to the current prop
    // This is needed since we need to include the "k-th" updated orientation information
    Eigen::Matrix3d R_k = state->_imu->Rot();
    Eigen::Vector3d v_k = state->_imu->vel();
    Eigen::Vector3d p_k = state->_imu->pos();

    Eigen::Matrix3d Xi_1 = Xi_sum.block(0, 3, 3, 3);
    Eigen::Matrix3d Xi_2 = Xi_sum.block(0, 6, 3, 3);
    Eigen::Matrix3d Jr_k1tok = Xi_sum.block(0, 9, 3, 3);
    Eigen::Matrix3d Xi_3 = Xi_sum.block(0, 12, 3, 3);
    Eigen::Matrix3d Xi_4 = Xi_sum.block(0, 15, 3, 3);

    Eigen::Matrix3d dR_k1tok = new_R.transpose() * R_k;
    // for th
    F.block(th_id, th_id, 3, 3) = dR_k1tok;
    F.block(p_id, th_id, 3, 3) = -skew(new_p - p_k - v_k * dt + 0.5 * gravity * dt * dt) * R_k;
    F.block(v_id, th_id, 3, 3) = -skew(new_v - v_k + gravity * dt) * R_k;

    // for p
    F.block(p_id, p_id, 3, 3).setIdentity();

    // for v
    F.block(p_id, v_id, 3, 3) = Eigen::Matrix3d::Identity() * dt;
    F.block(v_id, v_id, 3, 3).setIdentity();

    // for bg
    F.block(th_id, bg_id, 3, 3) = -Jr_k1tok * dt;
    F.block(p_id, bg_id, 3, 3) = R_k * Xi_4;
    F.block(v_id, bg_id, 3, 3) = R_k * Xi_3;
    F.block(bg_id, bg_id, 3, 3).setIdentity();

    // for ba
    F.block(p_id, ba_id, 3, 3) = -R_k * Xi_2;
    F.block(v_id, ba_id, 3, 3) = -R_k * Xi_1;
    F.block(ba_id, ba_id, 3, 3).setIdentity();

    // construct the G part
    G.block(th_id, th_id, 9, 6) = F.block(0, bg_id, 9, 6);
    G.block(bg_id, 6, 3, 3) = dt * Eigen::Matrix3d::Identity();
    G.block(ba_id, 9, 3, 3) = dt * Eigen::Matrix3d::Identity();
}

void Propagator::compute_F_and_G_Gamma(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                       Eigen::Matrix3d &new_R, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p, Eigen::MatrixXd &F,
                                       Eigen::MatrixXd &G) {

    Eigen::Vector3d phi = w_hat * dt;
    Eigen::Matrix3d G0 = Gamma_SO3(phi, 0);
    Eigen::Matrix3d G1 = Gamma_SO3(phi, 1);
    Eigen::Matrix3d G2 = Gamma_SO3(phi, 2);
    Eigen::Matrix3d G0t = G0.transpose(); //转置
    // Eigen::Matrix3d G1t = G1.transpose();
    Eigen::Matrix3d G2t = G2.transpose();
    Eigen::Matrix3d G3t = Gamma_SO3(-phi, 3);

    int dimP = state->_imu->size();
    int dimTheta = state->_imu->bg()->size() + state->_imu->ba()->size();
    // int dimX = dimP - dimTheta;
    F = Eigen::MatrixXd::Identity(dimP, dimP);
    // Compute the complicated bias terms (derived for the left invariant case)
    Eigen::Matrix3d ax = skew(a_hat);
    Eigen::Matrix3d wx = skew(w_hat);
    Eigen::Matrix3d wx2 = wx * wx;
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double theta = w_hat.norm();
    double theta2 = theta * theta;
    double theta3 = theta2 * theta;
    double theta4 = theta3 * theta;
    double theta5 = theta4 * theta;
    double theta6 = theta5 * theta;
    double theta7 = theta6 * theta;
    double thetadt = theta * dt;
    double thetadt2 = thetadt * thetadt;
    double thetadt3 = thetadt2 * thetadt;
    double sinthetadt = sin(thetadt);
    double costhetadt = cos(thetadt);
    double sin2thetadt = sin(2 * thetadt);
    double cos2thetadt = cos(2 * thetadt);
    double thetadtcosthetadt = thetadt * costhetadt;
    double thetadtsinthetadt = thetadt * sinthetadt;

    Eigen::Matrix3d Phi25L = G0t * (ax * G2t * dt2 + ((sinthetadt - thetadtcosthetadt) / (theta3)) * (wx * ax) -
                                    ((cos2thetadt - 4 * costhetadt + 3) / (4 * theta4)) * (wx * ax * wx) +
                                    ((4 * sinthetadt + sin2thetadt - 4 * thetadtcosthetadt - 2 * thetadt) / (4 * theta5)) * (wx * ax * wx2) +
                                    ((thetadt2 - 2 * thetadtsinthetadt - 2 * costhetadt + 2) / (2 * theta4)) * (wx2 * ax) -
                                    ((6 * thetadt - 8 * sinthetadt + sin2thetadt) / (4 * theta5)) * (wx2 * ax * wx) +
                                    ((2 * thetadt2 - 4 * thetadtsinthetadt - cos2thetadt + 1) / (4 * theta6)) * (wx2 * ax * wx2));

    Eigen::Matrix3d Phi35L =
        G0t * (ax * G3t * dt3 - ((thetadtsinthetadt + 2 * costhetadt - 2) / (theta4)) * (wx * ax) -
               ((6 * thetadt - 8 * sinthetadt + sin2thetadt) / (8 * theta5)) * (wx * ax * wx) -
               ((2 * thetadt2 + 8 * thetadtsinthetadt + 16 * costhetadt + cos2thetadt - 17) / (8 * theta6)) * (wx * ax * wx2) +
               ((thetadt3 + 6 * thetadt - 12 * sinthetadt + 6 * thetadtcosthetadt) / (6 * theta5)) * (wx2 * ax) -
               ((6 * thetadt2 + 16 * costhetadt - cos2thetadt - 15) / (8 * theta6)) * (wx2 * ax * wx) +
               ((4 * thetadt3 + 6 * thetadt - 24 * sinthetadt - 3 * sin2thetadt + 24 * thetadtcosthetadt) / (24 * theta7)) * (wx2 * ax * wx2));

    const double tolerance = 1e-6;
    if (theta < tolerance) {
        Phi25L = (1 / 2) * ax * dt2;
        Phi35L = (1 / 6) * ax * dt3;
    }

    Eigen::Matrix3d gx = skew(gravity);
    Eigen::Matrix3d R = state->_imu->Rot();
    Eigen::Vector3d v = state->_imu->vel();
    Eigen::Vector3d p = state->_imu->pos();
    Eigen::Matrix3d RG0 = R * G0;
    Eigen::Matrix3d RG1dt = R * G1 * dt;
    Eigen::Matrix3d RG2dt2 = R * G2 * dt2;

    new_R = RG0;
    new_v = v + RG1dt * a_hat - gravity * dt;
    new_p = p + v * dt + RG2dt2 * a_hat - 0.5 * gravity * dt2;

    F.block<3, 3>(3, 0) = -0.5 * gx * dt2;                                   // Phi_21
    F.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;                  // Phi_23
    F.block<3, 3>(6, 0) = -gx * dt;                                          // Phi_31
    F.block<3, 3>(0, dimP - dimTheta) = -RG1dt;                              // Phi_15
    F.block<3, 3>(3, dimP - dimTheta) = -skew(new_p) * RG1dt + RG0 * Phi35L; // Phi_25
    F.block<3, 3>(6, dimP - dimTheta) = -skew(new_v) * RG1dt + RG0 * Phi25L; // Phi_35
    F.block<3, 3>(3, dimP - dimTheta + 3) = -RG2dt2;                         // Phi_26
    F.block<3, 3>(6, dimP - dimTheta + 3) = -RG1dt;                          // Phi_36

    G = Eigen::MatrixXd::Identity(dimP, dimP);
    G.block(0, 0, dimP - dimTheta, dimP - dimTheta) = Adjoint_SEK3(state->_imu->value().block(0, 0, 5, 5));
}

void Propagator::predict_and_compute(std::shared_ptr<State> state, const ImuData &data_minus, const ImuData &data_plus, Eigen::MatrixXd &F,
                                     Eigen::MatrixXd &Qd) {
    // Time elapsed over interval
    double dt = data_plus.timestamp - data_minus.timestamp;
    // assert(data_plus.timestamp>data_minus.timestamp);

    // cout << "am: " << data_minus.am.transpose() << endl;
    // cout << "omega_m: " << data_minus.wm.transpose() << endl;

    // Corrected imu acc measurements with our current biases
    Eigen::Vector3d a_hat1 = data_minus.am - state->_imu->bias_a();
    Eigen::Vector3d a_hat2 = data_plus.am - state->_imu->bias_a();
    Eigen::Vector3d a_hat_avg = .5 * (a_hat1 + a_hat2);

    // Corrected imu gyro measurements with our current biases and gravity sensitivity
    Eigen::Vector3d w_hat1 = data_minus.wm - state->_imu->bias_g();
    Eigen::Vector3d w_hat2 = data_plus.wm - state->_imu->bias_g();
    Eigen::Vector3d w_hat_avg = .5 * (w_hat1 + w_hat2);

    // Pre-compute some analytical values for the mean and covariance integration
    Eigen::Matrix<double, 3, 18> Xi_sum = Eigen::Matrix<double, 3, 18>::Zero(3, 18);

    compute_Xi_sum(state, dt, w_hat_avg, a_hat_avg, Xi_sum);

    // Compute the new state mean value
    Eigen::Matrix3d new_R;
    Eigen::Vector3d new_v, new_p;

    predict_mean_analytic(state, dt, w_hat_avg, a_hat_avg, new_R, new_v, new_p, Xi_sum);
    // predict_mean_rk4(state, dt, w_hat1, a_hat1, w_hat2, a_hat2, new_R, new_v, new_p);

    // Allocate state transition and continuous-time noise Jacobian
    int prop_size = 15;
    if (!state->_features_SLAM.empty() && state->_kf == KFCLASS::IKF_IMUGROUP)
        prop_size += 3 * state->_features_SLAM.size();
    // if (state->_transform_as_group)
    //     prop_size += 3;
    // if (state->_multistate_as_group)
    //     prop_size += 3 * state->_clones_IMU.size();

    F = Eigen::MatrixXd::Zero(prop_size, prop_size);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(prop_size, 12);
    compute_F_and_G_analytic(state, dt, w_hat_avg, a_hat_avg, new_R, new_v, new_p, Xi_sum, F, G);

    // Construct our discrete noise covariance matrix
    // Note that we need to convert our continuous time noises to discrete
    // Equations (129) amd (130) of Trawny tech report
    Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
    Qc.block(0, 0, 3, 3) = noises.sigma_w_2_eig / dt;
    Qc.block(3, 3, 3, 3) = noises.sigma_a_2_eig / dt;
    Qc.block(6, 6, 3, 3) = noises.sigma_wb_2_eig / dt;
    Qc.block(9, 9, 3, 3) = noises.sigma_ab_2_eig / dt;

    // Compute the noise injected into the state over the interval
    Qd = Eigen::MatrixXd::Zero(prop_size, prop_size);
    Qd = G * Qc * G.transpose();
    Qd = 0.5 * (Qd + Qd.transpose());

    // Now replace imu estimate with propagated values
    Eigen::Matrix<double, 5, 7> imu_x = state->_imu->value();
    imu_x.block(0, 0, 3, 3) = new_R;
    imu_x.block(0, 3, 3, 1) = new_p;
    imu_x.block(0, 4, 3, 1) = new_v;
    state->_imu->set_value(imu_x);
}

// void Propagator::predict_and_compute(std::shared_ptr<State> state, const ImuData &data_minus, const ImuData &data_plus,
//                                       Eigen::MatrixXd &F, Eigen::MatrixXd &Qd){

//     // Time elapsed over interval
//     double dt = data_plus.timestamp - data_minus.timestamp;
//     // assert(data_plus.timestamp>data_minus.timestamp);

//     // Corrected imu acc measurements with our current biases
//     Eigen::Vector3d a_hat1 = data_minus.am - state->_imu->bias_a();
//     Eigen::Vector3d a_hat2 = data_plus.am - state->_imu->bias_a();
//     Eigen::Vector3d a_hat_avg = .5 * (a_hat1 + a_hat2);

//     // Corrected imu gyro measurements with our current biases and gravity sensitivity
//     Eigen::Vector3d w_hat1 = data_minus.wm - state->_imu->bias_g();
//     Eigen::Vector3d w_hat2 = data_plus.wm - state->_imu->bias_g();
//     Eigen::Vector3d w_hat_avg = .5 * (w_hat1 + w_hat2);

//     Eigen::Matrix3d new_R;
//     Eigen::Vector3d new_v;
//     Eigen::Vector3d new_p;

//     Eigen::MatrixXd G;
//     compute_F_and_G_Gamma(state, dt, w_hat_avg, a_hat_avg, new_R, new_v, new_p, F, G);

//     // Continuous noise covariance
//     Eigen::MatrixXd Qc = Eigen::MatrixXd::Zero(15 , 15);
//     Qc.block<3,3>(0,0) = noises.sigma_w_2_eig;
//     Qc.block<3,3>(3,3) = noises.sigma_a_2_eig;
//     Qc.block<3,3>(9,9) = noises.sigma_wb_2_eig;
//     Qc.block<3,3>(12,12) = noises.sigma_ab_2_eig;
//     Eigen::MatrixXd PhiG = F * G;
//     Qd = PhiG * Qc * PhiG.transpose() * dt; // Approximated discretized noise matrix

//     Eigen::Matrix<double, 5, 7> imu_x = state->_imu->value();
//     imu_x.block(0, 0, 3, 3) = new_R;
//     imu_x.block(0, 3, 3, 1) = new_p;
//     imu_x.block(0, 4, 3, 1) = new_v;
//     state->_imu->set_value(imu_x);
// }

void Propagator::predict_and_compute_MSCKF(std::shared_ptr<State> state, const ImuData &data_minus, const ImuData &data_plus, Eigen::MatrixXd &F,
                                           Eigen::MatrixXd &Qd) {
    // Time elapsed over interval
    double dt = data_plus.timestamp - data_minus.timestamp;
    // assert(data_plus.timestamp>data_minus.timestamp);

    // Corrected imu acc measurements with our current biases
    Eigen::Vector3d a_hat1 = data_minus.am - state->_imu->bias_a();
    Eigen::Vector3d a_hat2 = data_plus.am - state->_imu->bias_a();
    Eigen::Vector3d a_hat_avg = .5 * (a_hat1 + a_hat2);

    // Corrected imu gyro measurements with our current biases and gravity sensitivity
    Eigen::Vector3d w_hat1 = data_minus.wm - state->_imu->bias_g();
    Eigen::Vector3d w_hat2 = data_plus.wm - state->_imu->bias_g();
    Eigen::Vector3d w_hat_avg = .5 * (w_hat1 + w_hat2);

    Eigen::Matrix3d new_R;
    Eigen::Vector3d new_v;
    Eigen::Vector3d new_p;

    // Pre-compute some analytical values for the mean and covariance integration
    Eigen::Matrix<double, 3, 18> Xi_sum = Eigen::Matrix<double, 3, 18>::Zero(3, 18);

    compute_Xi_sum(state, dt, w_hat_avg, a_hat_avg, Xi_sum);

    // Compute the new state mean value
    predict_mean_analytic(state, dt, w_hat_avg, a_hat_avg, new_R, new_v, new_p, Xi_sum);

    // Allocate state transition and continuous-time noise Jacobian
    F = Eigen::MatrixXd::Zero(15, 15);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(15, 12);

    compute_F_and_G_analytic_MSCKF(state, dt, w_hat_avg, a_hat_avg, new_R, new_v, new_p, Xi_sum, F, G);

    // Construct our discrete noise covariance matrix
    // Note that we need to convert our continuous time noises to discrete
    // Equations (129) amd (130) of Trawny tech report
    Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
    Qc.block(0, 0, 3, 3) = noises.sigma_w_2_eig / dt;
    Qc.block(3, 3, 3, 3) = noises.sigma_a_2_eig / dt;
    Qc.block(6, 6, 3, 3) = noises.sigma_wb_2_eig / dt;
    Qc.block(9, 9, 3, 3) = noises.sigma_ab_2_eig / dt;

    // Compute the noise injected into the state over the interval
    Qd = Eigen::MatrixXd::Zero(15, 15);
    Qd = G * Qc * G.transpose();
    Qd = 0.5 * (Qd + Qd.transpose());

    // Now replace imu estimate with propagated values
    Eigen::Matrix<double, 5, 7> imu_x = state->_imu->value();
    imu_x.block(0, 0, 3, 3) = new_R;
    imu_x.block(0, 3, 3, 1) = new_p;
    imu_x.block(0, 4, 3, 1) = new_v;
    state->_imu->set_value(imu_x);
}

} // namespace night_voyager