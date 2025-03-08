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
#ifndef PROPAGATOR_H
#define PROPAGATOR_H

#include "core/CommonLib.h"
#include "core/NightVoyagerOptions.h"
#include "msckf_iekf/StateHelper.h"
#include <atomic>
#include <memory>
#include <mutex>

namespace night_voyager {
class State;

class Propagator {
  public:
    Propagator(const NightVoyagerOptions &options) {
        noises = options.noises;
        gravity << 0.0, 0.0, options.gravity_mag;
    }

    /**
     * @brief Stores incoming inertial readings
     * @param message Contains our timestamp and inertial information
     * @param oldest_time Time that we can discard measurements before (in IMU clock)
     */
    void feed_imu(const ImuData &message, double oldest_time = -1) {

        // Append it to our vector
        std::lock_guard<std::mutex> lck(imu_data_mtx);
        imu_data.emplace_back(message);

        // Clean old measurements
        // std::cout << "PROP: imu_data.size() " << imu_data.size() << std::endl;
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
     * @brief Propagate state up to timestamp of package data and then clone
     *
     * This will first collect all imu readings that occured between the
     * *current* state time and the new time we want the state to be at.
     * If we don't have any imu readings we will try to extrapolate into the future.
     * After propagating the mean and covariance using our dynamics,
     * We clone the current imu pose as a new clone in our state.
     *
     * @param state Pointer to state
     * @param timestamp Time to propagate to and clone at (CAM clock frame)
     * @param prop_imus Imu data to propagate
     */
    void propagate(std::shared_ptr<State> state, double timestamp, const ImuData &last_imu);

    /**
     * @brief Helper function that given current imu data, will select imu readings between the two times.
     *
     * This will create measurements that we will integrate with, and an extra measurement at the end.
     * We use the @ref interpolate_data() function to "cut" the imu readings at the begining and end of the integration.
     * The timestamps passed should already take into account the time offset values.
     *
     * @param imu_data IMU data we will select measurements from
     * @param time0 Start timestamp
     * @param time1 End timestamp
     * @param last_imu The last imu data used for interpolation
     * @return Vector of measurements (if we could compute them)
     */
    static std::vector<ImuData> select_imu_readings(const std::vector<ImuData> &imu_data, double time0, double time1, const ImuData &last_imu);

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
        // time-distance lambda
        double lambda = (timestamp - imu_1.timestamp) / (imu_2.timestamp - imu_1.timestamp);
        // PRINT_DEBUG("lambda - %d\n", lambda);
        // interpolate between the two times
        ImuData data;
        data.timestamp = timestamp;
        data.am = (1 - lambda) * imu_1.am + lambda * imu_2.am;
        data.wm = (1 - lambda) * imu_1.wm + lambda * imu_2.wm;
        return data;
    }

  protected:
    /**
     * @brief Analytically compute the integration components based on ACI^2
     *
     * See the @ref analytical_prop page and @ref analytical_integration_components for details.
     * For computing Xi_1, Xi_2, Xi_3 and Xi_4 we have:
     *
     * \f{align*}{
     * \boldsymbol{\Xi}_1 & = \mathbf{I}_3 \delta t + \frac{1 - \cos (\hat{\omega} \delta t)}{\hat{\omega}} \lfloor \hat{\mathbf{k}} \rfloor
     * + \left(\delta t  - \frac{\sin (\hat{\omega} \delta t)}{\hat{\omega}}\right) \lfloor \hat{\mathbf{k}} \rfloor^2 \\
     * \boldsymbol{\Xi}_2 & = \frac{1}{2} \delta t^2 \mathbf{I}_3 +
     * \frac{\hat{\omega} \delta t - \sin (\hat{\omega} \delta t)}{\hat{\omega}^2}\lfloor \hat{\mathbf{k}} \rfloor
     * + \left( \frac{1}{2} \delta t^2 - \frac{1  - \cos (\hat{\omega} \delta t)}{\hat{\omega}^2} \right) \lfloor \hat{\mathbf{k}} \rfloor ^2
     * \\ \boldsymbol{\Xi}_3  &= \frac{1}{2}\delta t^2  \lfloor \hat{\mathbf{a}} \rfloor
     * + \frac{\sin (\hat{\omega} \delta t_i) - \hat{\omega} \delta t }{\hat{\omega}^2} \lfloor\hat{\mathbf{a}} \rfloor \lfloor
     * \hat{\mathbf{k}} \rfloor
     * + \frac{\sin (\hat{\omega} \delta t) - \hat{\omega} \delta t \cos (\hat{\omega} \delta t)  }{\hat{\omega}^2}
     * \lfloor \hat{\mathbf{k}} \rfloor\lfloor\hat{\mathbf{a}} \rfloor
     * + \left( \frac{1}{2} \delta t^2 - \frac{1 - \cos (\hat{\omega} \delta t)}{\hat{\omega}^2} \right) 	\lfloor\hat{\mathbf{a}} \rfloor
     * \lfloor \hat{\mathbf{k}} \rfloor ^2
     * + \left(
     * \frac{1}{2} \delta t^2 + \frac{1 - \cos (\hat{\omega} \delta t) - \hat{\omega} \delta t \sin (\hat{\omega} \delta t) }{\hat{\omega}^2}
     *  \right)
     *  \lfloor \hat{\mathbf{k}} \rfloor ^2 \lfloor\hat{\mathbf{a}} \rfloor
     *  + \left(
     *  \frac{1}{2} \delta t^2 + \frac{1 - \cos (\hat{\omega} \delta t) - \hat{\omega} \delta t \sin (\hat{\omega} \delta t) }{\hat{\omega}^2}
     *  \right)  \hat{\mathbf{k}}^{\top} \hat{\mathbf{a}} \lfloor \hat{\mathbf{k}} \rfloor
     *  - \frac{ 3 \sin (\hat{\omega} \delta t) - 2 \hat{\omega} \delta t - \hat{\omega} \delta t \cos (\hat{\omega} \delta t)
     * }{\hat{\omega}^2} \hat{\mathbf{k}}^{\top} \hat{\mathbf{a}} \lfloor \hat{\mathbf{k}} \rfloor ^2  \\
     * \boldsymbol{\Xi}_4 & = \frac{1}{6}\delta
     * t^3 \lfloor\hat{\mathbf{a}} \rfloor
     * + \frac{2(1 - \cos (\hat{\omega} \delta t)) - (\hat{\omega}^2 \delta t^2)}{2 \hat{\omega}^3}
     *  \lfloor\hat{\mathbf{a}} \rfloor \lfloor \hat{\mathbf{k}} \rfloor
     *  + \left(
     *  \frac{2(1- \cos(\hat{\omega} \delta t)) - \hat{\omega} \delta t \sin (\hat{\omega} \delta t)}{\hat{\omega}^3}
     *  \right)
     *  \lfloor \hat{\mathbf{k}} \rfloor\lfloor\hat{\mathbf{a}} \rfloor
     *  + \left(
     *  \frac{\sin (\hat{\omega} \delta t) - \hat{\omega} \delta t}{\hat{\omega}^3} +
     *  \frac{\delta t^3}{6}
     *  \right)
     *  \lfloor\hat{\mathbf{a}} \rfloor \lfloor \hat{\mathbf{k}} \rfloor^2
     *  +
     *  \frac{\hat{\omega} \delta t - 2 \sin(\hat{\omega} \delta t) + \frac{1}{6}(\hat{\omega} \delta t)^3 + \hat{\omega} \delta t
     * \cos(\hat{\omega} \delta t)}{\hat{\omega}^3} \lfloor \hat{\mathbf{k}} \rfloor^2\lfloor\hat{\mathbf{a}} \rfloor
     *  +
     *  \frac{\hat{\omega} \delta t - 2 \sin(\hat{\omega} \delta t) + \frac{1}{6}(\hat{\omega} \delta t)^3 + \hat{\omega} \delta t
     * \cos(\hat{\omega} \delta t)}{\hat{\omega}^3} \hat{\mathbf{k}}^{\top} \hat{\mathbf{a}} \lfloor \hat{\mathbf{k}} \rfloor
     *  +
     *  \frac{4 \cos(\hat{\omega} \delta t) - 4 + (\hat{\omega} \delta t)^2 + \hat{\omega} \delta t \sin(\hat{\omega} \delta t) }
     *  {\hat{\omega}^3}
     *  \hat{\mathbf{k}}^{\top} \hat{\mathbf{a}} \lfloor \hat{\mathbf{k}} \rfloor^2
     * \f}
     *
     * @param state Pointer to state
     * @param dt Time we should integrate over
     * @param w_hat Angular velocity with bias removed
     * @param a_hat Linear acceleration with bias removed
     * @param Xi_sum All the needed integration components, including R_k, Xi_1, Xi_2, Jr, Xi_3, Xi_4 in order
     */
    void compute_Xi_sum(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                        Eigen::Matrix<double, 3, 18> &Xi_sum);

    /**
     * @brief Analytically predict IMU mean based on ACI^2
     *
     * See the @ref analytical_prop page for details.
     *
     * \f{align*}{
     * {}^{I_{k+1}}_G\hat{\mathbf{R}} & \simeq  \Delta \mathbf{R}_k {}^{I_k}_G\hat{\mathbf{R}}  \\
     * {}^G\hat{\mathbf{p}}_{I_{k+1}} & \simeq {}^{G}\hat{\mathbf{p}}_{I_k} + {}^G\hat{\mathbf{v}}_{I_k}\delta t_k  +
     * {}^{I_k}_G\hat{\mathbf{R}}^\top  \Delta \hat{\mathbf{p}}_k - \frac{1}{2}{}^G\mathbf{g}\delta t^2_k \\
     * {}^G\hat{\mathbf{v}}_{I_{k+1}} & \simeq  {}^{G}\hat{\mathbf{v}}_{I_k} + {}^{I_k}_G\hat{\mathbf{R}}^\top + \Delta \hat{\mathbf{v}}_k -
     * {}^G\mathbf{g}\delta t_k
     * \f}
     *
     * @param state Pointer to state
     * @param dt Time we should integrate over
     * @param w_hat Angular velocity with bias removed
     * @param a_hat Linear acceleration with bias removed
     * @param new_R The resulting new orientation after integration
     * @param new_v The resulting new velocity after integration
     * @param new_p The resulting new position after integration
     * @param Xi_sum All the needed integration components, including R_k, Xi_1, Xi_2, Jr, Xi_3, Xi_4
     */
    void predict_mean_analytic(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                               Eigen::Matrix3d &new_R, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p, Eigen::Matrix<double, 3, 18> &Xi_sum);

    void predict_mean_rk4(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat1, const Eigen::Vector3d &a_hat1,
                          const Eigen::Vector3d &w_hat2, const Eigen::Vector3d &a_hat2, Eigen::Matrix3d &new_R, Eigen::Vector3d &new_v,
                          Eigen::Vector3d &new_p);

    /**
     * @brief Propagates the state forward using the imu data and computes the noise covariance and state-transition
     * matrix of this interval.
     *
     * This function can be replaced with analytical/numerical integration or when using a different state representation.
     * This contains our state transition matrix along with how our noise evolves in time.
     * If you have other state variables besides the IMU that evolve you would add them here.
     * See the @ref propagation_discrete page for details on how discrete model was derived.
     * See the @ref propagation_analytical page for details on how analytic model was derived.
     *
     * @param state Pointer to state
     * @param data_minus imu readings at beginning of interval
     * @param data_plus imu readings at end of interval
     * @param F State-transition matrix over the interval
     * @param Qd Discrete-time noise covariance over the interval
     */
    void predict_and_compute(std::shared_ptr<State> state, const ImuData &data_minus, const ImuData &data_plus, Eigen::MatrixXd &F,
                             Eigen::MatrixXd &Qd);

    void predict_and_compute_MSCKF(std::shared_ptr<State> state, const ImuData &data_minus, const ImuData &data_plus, Eigen::MatrixXd &F,
                                   Eigen::MatrixXd &Qd);

    /**
     * @brief Analytically compute state transition matrix F and noise Jacobian G based on ACI^2
     *
     * This function is for analytical integration of the linearized error-state.
     * This contains our state transition matrix and noise Jacobians.
     * If you have other state variables besides the IMU that evolve you would add them here.
     * See the @ref analytical_linearization page for details on how this was derived.
     *
     * @param state Pointer to state
     * @param dt Time we should integrate over
     * @param w_hat Angular velocity with bias removed
     * @param a_hat Linear acceleration with bias removed
     * @param w_uncorrected Angular velocity in acc frame with bias and gravity sensitivity removed
     * @param new_q The resulting new orientation after integration
     * @param new_v The resulting new velocity after integration
     * @param new_p The resulting new position after integration
     * @param Xi_sum All the needed integration components, including R_k, Xi_1, Xi_2, Jr, Xi_3, Xi_4
     * @param F State transition matrix
     * @param G Noise Jacobian
     */
    void compute_F_and_G_analytic(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                  const Eigen::Matrix3d &new_R, const Eigen::Vector3d &new_v, const Eigen::Vector3d &new_p,
                                  const Eigen::Matrix<double, 3, 18> &Xi_sum, Eigen::MatrixXd &F, Eigen::MatrixXd &G);

    void compute_F_and_G_analytic_MSCKF(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                        const Eigen::Matrix3d &new_R, const Eigen::Vector3d &new_v, const Eigen::Vector3d &new_p,
                                        const Eigen::Matrix<double, 3, 18> &Xi_sum, Eigen::MatrixXd &F, Eigen::MatrixXd &G);

    void compute_F_and_G_Gamma(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                               Eigen::Matrix3d &new_R, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p, Eigen::MatrixXd &F, Eigen::MatrixXd &G);

    Eigen::Vector3d sigma_w2;
    Eigen::Vector3d sigma_a2;
    Eigen::Vector3d sigma_wb2;
    Eigen::Vector3d sigma_ab2;
    Eigen::Vector3d gravity;

    /// Our history of IMU messages (time, angular, linear)
    std::vector<ImuData> imu_data;
    std::mutex imu_data_mtx;

    /// Container for the noise values
    NoiseManager noises;
};
} // namespace night_voyager
#endif