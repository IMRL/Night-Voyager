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
#include "msckf_iekf/UpdaterZeroVelocity.h"
#include "feature_tracker/FeatureDatabase.h"
#include "feature_tracker/FeatureHelper.h"
#include "msckf_iekf/Propagator.h"
#include "msckf_iekf/State.h"
#include "msckf_iekf/StateHelper.h"
#include "msckf_iekf/UpdaterHelper.h"
#include "utils/Print.h"

namespace night_voyager {
UpdaterZeroVelocity::UpdaterZeroVelocity(const NightVoyagerOptions &options, std::shared_ptr<FeatureDatabase> db, std::shared_ptr<Propagator> prop)
    : _options(options.zupt_options), _noises(options.noises), _db(db), _prop(prop), _zupt_max_velocity_odom(options.zupt_max_velocity_odom),
      _zupt_max_velocity(options.zupt_max_velocity), _zupt_noise_multiplier(options.zupt_noise_multiplier), _zupt_max_disparity(options.zupt_max_disparity) {
    // Gravity
    _gravity << 0.0, 0.0, options.gravity_mag;

    // Initialize the chi squared test table with confidence level 0.95
    // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
    for (int i = 1; i < 1000; i++) {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
    }
}

bool UpdaterZeroVelocity::try_update(std::shared_ptr<State> state, const PackData &message) {

    // Return if we don't have any imu data yet
    if (imu_data.empty()) {
        last_zupt_state_timestamp = 0.0;
        return false;
    }

    // Return if the state is already at the desired time
    if (state->_timestamp == message.cam.timestamp) {
        last_zupt_state_timestamp = 0.0;
        return false;
    }

    // First use odom data to check
    const vector<OdomData> &odoms = message.odoms;
    for (size_t i = 0; i < odoms.size(); ++i) {
        if (odoms[i].vm.norm() >= _zupt_max_velocity_odom) {
            last_zupt_state_timestamp = 0.0;
            return false;
        }
    }

    // First lets construct an IMU vector of measurements we need
    double time0 = state->_timestamp;
    double time1 = message.cam.timestamp;

    // Select bounding inertial measurements
    std::vector<ImuData> imu_recent = Propagator::select_imu_readings(imu_data, time0, time1, message.imus.back());

    // Check that we have at least one measurement to propagate with
    if (imu_recent.size() < 2) {
        PRINT_ERROR(RED "[ZUPT]: There are no IMU data to check for zero velocity with!!\n" RESET);
        last_zupt_state_timestamp = 0.0;
        std::exit(EXIT_FAILURE);
        // return false;
    }

    // If we should integrate the acceleration and say the velocity should be zero
    // Also if we should still inflate the bias based on their random walk noises
    bool integrated_accel_constraint = false; // untested
    bool model_time_varying_bias = true;
    bool override_with_disparity_check = false;

    // Order of our Jacobian
    std::vector<std::shared_ptr<Type>> Hx_order;
    Hx_order.push_back(state->_imu->R());
    Hx_order.push_back(state->_imu->bg());
    Hx_order.push_back(state->_imu->ba());
    if (integrated_accel_constraint) {
        Hx_order.push_back(state->_imu->v());
    }

    // Large final matrices used for update (we will compress these)
    int h_size = (integrated_accel_constraint) ? 12 : 9;
    int m_size = 6 * ((int)imu_recent.size() - 1);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m_size, h_size);
    Eigen::VectorXd res = Eigen::VectorXd::Zero(m_size);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(m_size, m_size);

    // Loop through all our IMU and construct the residual and Jacobian
    // TODO: should add jacobians here in respect to IMU intrinsics!!
    // State order is: [R_ItoG, bg, ba, v_IinG]
    // Measurement order is: [w_true = 0, a_true = 0 or v_k+1 = 0]
    // w_true = w_m - bw - nw
    // a_true = a_m - ba - R^T*g - na
    // v_true = v_k - g*dt + R*(a_m - ba - na)*dt
    double dt_summed = 0;
    for (size_t i = 0; i < imu_recent.size() - 1; i++) {

        // Precomputed values
        double dt = imu_recent.at(i + 1).timestamp - imu_recent.at(i).timestamp;
        Eigen::Vector3d a_hat = imu_recent.at(i).am - state->_imu->bias_a();
        Eigen::Vector3d w_hat = imu_recent.at(i).wm - state->_imu->bias_g();

        // Measurement residual (true value is zero)
        res.block(6 * i + 0, 0, 3, 1) = -w_hat;
        if (!integrated_accel_constraint) {
            res.block(6 * i + 3, 0, 3, 1) = -(a_hat - state->_imu->Rot().transpose() * _gravity);
        } else {
            res.block(6 * i + 3, 0, 3, 1) = -(state->_imu->vel() - _gravity * dt + state->_imu->Rot() * a_hat * dt);
        }

        // Measurement Jacobian
        Eigen::Matrix3d R_GtoI_jacob = state->_imu->Rot().transpose();
        H.block(6 * i + 0, 3, 3, 3) = -Eigen::Matrix3d::Identity();
        if (!integrated_accel_constraint) {
            if (state->_kf == KFCLASS::MSCKF)
                H.block(6 * i + 3, 0, 3, 3) = -skew(R_GtoI_jacob * _gravity);
            else
                H.block(6 * i + 3, 0, 3, 3) = -R_GtoI_jacob * skew(_gravity);
            H.block(6 * i + 3, 6, 3, 3) = -Eigen::Matrix3d::Identity();
        } else {
            if (state->_kf == KFCLASS::MSCKF)
                H.block(6 * i + 3, 0, 3, 3) = -state->_imu->Rot() * skew(a_hat) * dt;
            else
                H.block(6 * i + 3, 0, 3, 3) = -skew(state->_imu->Rot() * a_hat) * dt - skew(state->_imu->vel());
            H.block(6 * i + 3, 6, 3, 3) = state->_imu->Rot() * dt;
            H.block(6 * i + 3, 9, 3, 3) = Eigen::Matrix3d::Identity();
        }

        R.block(6 * i + 0, 6 * i + 0, 3, 3) = _noises.sigma_w_2_eig / dt;
        if (!integrated_accel_constraint) {
            R.block(6 * i + 3, 6 * i + 3, 3, 3) = _noises.sigma_a_2_eig / dt;
        } else {
            R.block(6 * i + 3, 6 * i + 3, 3, 3) = _noises.sigma_a_2_eig * dt;
        }

        dt_summed += dt;
    }

    // Multiply our noise matrix by a fixed amount
    // We typically need to treat the IMU as being "worst" to detect / not become over confident
    R *= _zupt_noise_multiplier;

    // Next propagate the biases forward in time
    // NOTE: G*Qd*G^t = dt*Qd*dt = dt*(1/dt*Qc)*dt = dt*Qc
    Eigen::MatrixXd Q_bias = Eigen::MatrixXd::Identity(6, 6);
    Q_bias.block(0, 0, 3, 3) = dt_summed * _noises.sigma_wb_2_eig;
    Q_bias.block(3, 3, 3, 3) = dt_summed * _noises.sigma_ab_2_eig;

    // Chi2 distance check
    // NOTE: we also append the propagation we "would do before the update" if this was to be accepted (just the bias evolution)
    // NOTE: we don't propagate first since if we fail the chi2 then we just want to return and do normal logic
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
    if (model_time_varying_bias) {
        P_marg.block(3, 3, 6, 6) += Q_bias;
    }
    Eigen::MatrixXd S = H * P_marg * H.transpose() + R;
    double chi2 = res.dot(S.llt().solve(res));

    // Get our threshold (we precompute up to 1000 but handle the case that it is more)
    double chi2_check;
    if (res.rows() < 1000) {
        chi2_check = chi_squared_table[res.rows()];
    } else {
        boost::math::chi_squared chi_squared_dist(res.rows());
        chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
        PRINT_WARNING(YELLOW "[ZUPT]: chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
    }

    // Check if the image disparity
    bool disparity_passed = false;
    if (override_with_disparity_check) {

        // Get the disparity statistics from this image to the previous
        double time0_cam = state->_timestamp;
        double time1_cam = message.cam.timestamp;
        int num_features = 0;
        double disp_avg = 0.0;
        double disp_var = 0.0;
        FeatureHelper::compute_disparity(_db, time0_cam, time1_cam, disp_avg, disp_var, num_features);

        // Check if this disparity is enough to be classified as moving
        disparity_passed = (disp_avg < _zupt_max_disparity && num_features > 20);
        if (disparity_passed) {
            PRINT_INFO(CYAN "[ZUPT]: passed disparity (%.3f < %.3f, %d features)\n" RESET, disp_avg, _zupt_max_disparity, (int)num_features);
        } else {
            PRINT_DEBUG(YELLOW "[ZUPT]: failed disparity (%.3f > %.3f, %d features)\n" RESET, disp_avg, _zupt_max_disparity, (int)num_features);
        }
    }

    // Check if we are currently zero velocity
    // We need to pass the chi2 and not be above our velocity threshold
    if (!disparity_passed && (chi2 > _options.chi2_multipler * chi2_check || state->_imu->vel().norm() > _zupt_max_velocity)) {
        last_zupt_state_timestamp = 0.0;
        last_zupt_count = 0;
        PRINT_DEBUG(YELLOW "[ZUPT]: rejected |v_IinG| = %.3f (chi2 %.3f > %.3f)\n" RESET, state->_imu->vel().norm(), chi2, _options.chi2_multipler * chi2_check);
        return false;
    }
    PRINT_INFO(CYAN "[ZUPT]: accepted |v_IinG| = %.3f (chi2 %.3f < %.3f)\n" RESET, state->_imu->vel().norm(), chi2, _options.chi2_multipler * chi2_check);

    // Do our update, only do this update if we have previously detected
    // If we have succeeded, then we should remove the current timestamp feature tracks
    // This is because we will not clone at this timestep and instead do our zero velocity update
    // NOTE: We want to keep the tracks from the second time we have called the zv-upt since this won't have a clone
    // NOTE: All future times after the second call to this function will also *not* have a clone, so we can remove those
    if (last_zupt_count >= 2) {
        _db->cleanup_measurements_exact(last_zupt_state_timestamp);
    }

    // Else we are good, update the system
    // 1) update with our IMU measurements directly
    // 2) propagate and then explicitly say that our ori, pos, and vel should be zero

    // Next propagate the biases forward in time
    // NOTE: G*Qd*G^t = dt*Qd*dt = dt*Qc
    // TODO check the propagation
    if (model_time_varying_bias) {
        int size = state->_imu->size();
        Eigen::MatrixXd Phi_bias = Eigen::MatrixXd::Identity(size, size);
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(size, size);
        Q.block(9, 9, 6, 6) = Q_bias;
        std::vector<std::shared_ptr<Type>> Phi_order;
        Phi_order.push_back(state->_imu);
        state->_zupt = true;
        // Phi_order.push_back(state->_imu->ba());
        StateHelper::EKFPropagation(state, Phi_order, Phi_order, Phi_bias, Q);
    }

    // Finally move the state time forward
    StateHelper::EKFUpdate(state, Hx_order, H, res, R);
    state->_timestamp = message.cam.timestamp;
    state->_zupt = false;
    // Finally return
    last_zupt_state_timestamp = message.cam.timestamp;
    last_zupt_count++;
    return true;
}
} // namespace night_voyager