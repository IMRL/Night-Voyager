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
#include "msckf_iekf/UpdaterOdom.h"
#include "msckf_iekf/State.h"
#include "msckf_iekf/StateHelper.h"
#include "msckf_iekf/UpdaterHelper.h"
#include <boost/math/distributions/chi_squared.hpp>

namespace night_voyager {

UpdaterOdom::UpdaterOdom(NightVoyagerOptions &options) {

    _options = options.odom_options;

    // Initialize the chi squared test table with confidence level 0.95
    // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
    for (int i = 1; i < 500; i++) {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
    }
}

void UpdaterOdom::update(std::shared_ptr<State> state, const OdomData &odom) {
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    Eigen::MatrixXd R;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // cout << "odom_vel: " << odom.vm.transpose() << endl;

    // Get the Jacobian for this odom
    if (state->_kf == KFCLASS::MSCKF)
        UpdaterHelper::get_odom_jacobian_full_msckf(state, odom, res, H_x, Hx_order, R);
    else
        UpdaterHelper::get_odom_jacobian_full(state, odom, res, H_x, Hx_order, R);

    R = _options.odom_cov_eig;

    // Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
    // Eigen::MatrixXd S = H_x*P_marg*H_x.transpose() + R; //HPH^T+R
    // double chi2 = res.dot(S.llt().solve(res)); // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
    //                                             // the result is the Mahalanobis distance

    // // Get our threshold (we precompute up to 500 but handle the case that it is more)
    // double chi2_check;

    // if (res.rows() < 500) {
    //     chi2_check = chi_squared_table[res.rows()];
    // }
    // else {
    //     boost::math::chi_squared chi_squared_dist(res.rows());
    //     chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
    //     PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
    // }

    // // Check if we should delete or not
    // if (chi2 > _options.chi2_multipler * chi2_check) {
    //     PRINT_DEBUG("Odom measurement, chi2 = %f > %f, PASS this update\n", chi2, _options.chi2_multipler*chi2_check);
    //     std::stringstream ss;
    //     ss << "res = " << std::endl << res.transpose() << std::endl;
    //     PRINT_DEBUG(ss.str().c_str());
    // }

    StateHelper::EKFUpdate(state, Hx_order, H_x, res, R);
}

} // namespace night_voyager