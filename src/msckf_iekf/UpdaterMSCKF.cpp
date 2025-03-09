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
#include "msckf_iekf/UpdaterMSCKF.h"
#include "core/LandMark.h"
#include "feature_tracker/Feature.h"
#include "feature_tracker/FeatureInitializer.h"
#include "msckf_iekf/State.h"
#include "msckf_iekf/StateHelper.h"
#include "msckf_iekf/UpdaterHelper.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>

namespace night_voyager {
UpdaterMSCKF::UpdaterMSCKF(NightVoyagerOptions &options) : _options(options.msckf_options) {
    // Save our raw pixel noise squared
    _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);

    // Save our feature initializer
    initializer_feat = std::shared_ptr<FeatureInitializer>(new FeatureInitializer(options.feat_init_options));

    // Initialize the chi squared test table with confidence level 0.95
    // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
    for (int i = 1; i < 500; i++) {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
    }
}

void UpdaterMSCKF::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {
    // Return if no features
    if (feature_vec.empty())
        return;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
    rT0 = boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    std::vector<double> clonetimes;
    for (const auto &clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // cout << "feature_vec: " << feature_vec.size() << endl;

    // 1. Clean all feature measurements and make sure they all have valid clone times
    auto it0 = feature_vec.begin();
    while (it0 != feature_vec.end()) {

        // Clean the feature
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements
        int ct_meas = (*it0)->timestamps.size();

        // Remove if we don't have enough
        if (ct_meas < 2) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }
    }
    rT1 = boost::posix_time::microsec_clock::local_time();
    // cout << "feature_vec: " << feature_vec.size() << endl;

    // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cam; //(R_CG, p_GC)
    for (const auto &clone_imu : state->_clones_IMU) {

        // Get current camera pose
        Eigen::Matrix<double, 3, 3> R_GtoCi = state->_calib_IMUtoCAM->Rot() * clone_imu.second->Rot().transpose();
        Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * state->_calib_IMUtoCAM->pos();

        // Append to our map
        clones_cam.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }

    // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
    auto it1 = feature_vec.begin();
    while (it1 != feature_vec.end()) {

        // Triangulate the feature and remove if it fails
        bool success_tri = true;
        if (initializer_feat->config().triangulate_1d) {
            success_tri = initializer_feat->single_triangulation_1d(*it1, clones_cam);
        } else {
            success_tri = initializer_feat->single_triangulation(*it1, clones_cam);
        }

        // Gauss-newton refine the feature
        bool success_refine = true;
        if (initializer_feat->config().refine_features) {
            success_refine = initializer_feat->single_gaussnewton(*it1, clones_cam);
        }

        // Remove the feature if not a success
        if (!success_tri || !success_refine) {
            (*it1)->to_delete = true;
            it1 = feature_vec.erase(it1);
            continue;
        }
        it1++;
    }
    rT2 = boost::posix_time::microsec_clock::local_time();

    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    for (size_t i = 0; i < feature_vec.size(); i++) {
        max_meas_size += 2 * feature_vec.at(i)->timestamps.size();
    }

    // Calculate max possible state size (i.e. the size of our covariance)
    size_t max_hx_size = state->max_covariance_size();
    for (auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }

    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
    std::vector<std::shared_ptr<Type>> Hx_order_big;
    size_t ct_jacob = 0;
    size_t ct_meas = 0;

    // 4. Compute linear system for each feature, nullspace project, and reject
    auto it2 = feature_vec.begin();
    while (it2 != feature_vec.end()) {

        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        feat.pseudo_anchor_clone_timestamp = (*it2)->pseudo_anchor_clone_timestamp;
        feat.anchor_clone_timestamp = (*it2)->anchor_clone_timestamp;

        feat.p_FinG = (*it2)->p_FinG;

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f;
        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        std::vector<std::shared_ptr<Type>> Hx_order;

        // Get the Jacobian for this feature
        // MSCKF
        if (state->_kf == KFCLASS::MSCKF)
            UpdaterHelper::get_feature_jacobian_full_msckf(state, feat, H_f, H_x, res, Hx_order);
        else {
            // MSC-InEKF-FDRC
            if (state->_feature_in_clone)
                UpdaterHelper::get_feature_jacobian_full_clone_group(state, feat, H_f, H_x, res, Hx_order);
            // MSC-InEKF-FDRC or MSC-InEKF-FDR
            else if (state->_feature_in_rel_group)
                UpdaterHelper::get_feature_jacobian_full_rel_group(state, feat, H_f, H_x, res, Hx_order);
            // MSC-InEKF-FDN
            else if (state->_feature_no_group)
                UpdaterHelper::get_feature_jacobian_full_no_group(state, feat, H_f, H_x, res, Hx_order);
            // MSC-InEKF-FC
            else
                UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);
        }

        // Nullspace project
        UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

        /// Chi2 distance check
        Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
        Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
        S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows()); // HPH^T+R
        double chi2 = res.dot(S.llt().solve(res));                               // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
                                                                                    // the result is the Mahalanobis distance

        // Get our threshold (we precompute up to 500 but handle the case that it is more)
        double chi2_check;
        if (res.rows() < 500) {
            chi2_check = chi_squared_table[res.rows()];
        } else {
            boost::math::chi_squared chi_squared_dist(res.rows());
            chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
            PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
        }

        // Check if we should delete or not
        if (chi2 > _options.chi2_multipler * chi2_check) {
            (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            PRINT_DEBUG("featid = %d\n", feat.featid);
            PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options.chi2_multipler * chi2_check);
            std::stringstream ss;
            ss << "res = " << std::endl << res.transpose() << std::endl;
            PRINT_DEBUG(ss.str().c_str());
            continue;
        }

        // We are good!!! Append to our large H vector
        size_t ct_hx = 0;
        for (const auto &var : Hx_order) {

            // Ensure that this variable is in our Jacobian
            if (Hx_mapping.find(var) == Hx_mapping.end()) {
                Hx_mapping.insert({var, ct_jacob});
                Hx_order_big.push_back(var);
                ct_jacob += var->size();
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
            ct_hx += var->size();
        }

        // Append our residual and move forward
        res_big.block(ct_meas, 0, res.rows(), 1) = res;
        ct_meas += res.rows();
        it2++;
    }
    rT3 = boost::posix_time::microsec_clock::local_time();

    // We have appended all features to our Hx_big, res_big
    // Delete it so we do not reuse information
    for (size_t f = 0; f < feature_vec.size(); f++) {
        feature_vec[f]->to_delete = true;
    }

    // Return if we don't have anything and resize our matrices
    if (ct_meas < 1) {
        return;
    }
    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);

    // 5. Perform measurement compression
    UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    if (Hx_big.rows() < 1) {
        return;
    }
    rT4 = boost::posix_time::microsec_clock::local_time();

    // Our noise is isotropic, so make it here after our compression
    Eigen::MatrixXd R_big = _options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

    // 6. With all good features update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
    rT5 = boost::posix_time::microsec_clock::local_time();

    // Debug print timing information
    PRINT_ALL("[MSCKF-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
    PRINT_ALL("[MSCKF-UP]: %.4f seconds to triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
    PRINT_ALL("[MSCKF-UP]: %.4f seconds create system (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
    PRINT_ALL("[MSCKF-UP]: %.4f seconds compress system\n", (rT4 - rT3).total_microseconds() * 1e-6);
    PRINT_ALL("[MSCKF-UP]: %.4f seconds update state (%d size)\n", (rT5 - rT4).total_microseconds() * 1e-6, (int)res_big.rows());
    PRINT_ALL("[MSCKF-UP]: %.4f seconds total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}
} // namespace night_voyager