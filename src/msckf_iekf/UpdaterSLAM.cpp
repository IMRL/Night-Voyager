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
#include "msckf_iekf/UpdaterSLAM.h"
#include "core/LandMark.h"
#include "feature_tracker/Feature.h"
#include "feature_tracker/FeatureInitializer.h"
#include "msckf_iekf/State.h"
#include "msckf_iekf/StateHelper.h"
#include "msckf_iekf/UpdaterHelper.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>

namespace night_voyager {

UpdaterSLAM::UpdaterSLAM(NightVoyagerOptions &options) : _options_slam(options.slam_options) {

    _options_slam.sigma_pix_sq = std::pow(_options_slam.sigma_pix, 2);

    // Save our feature initializer
    initializer_feat = std::shared_ptr<FeatureInitializer>(new FeatureInitializer(options.feat_init_options));

    // Initialize the chi squared test table with confidence level 0.95
    // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
    for (int i = 1; i < 500; i++) {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
    }
}

void UpdaterSLAM::delayed_init(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

    // Return if no features
    if (feature_vec.empty())
        return;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3;
    rT0 = boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    std::vector<double> clonetimes;
    for (const auto &clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    auto it0 = feature_vec.begin();
    while (it0 != feature_vec.end()) {

        // Clean the feature
        (*it0)->clean_old_measurements(clonetimes);

        // Remove if we don't have enough
        if ((*it0)->timestamps.size() < 2) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }
    }
    rT1 = boost::posix_time::microsec_clock::local_time();

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

    // 4. Compute linear system for each feature, nullspace project, and reject
    auto it2 = feature_vec.begin();
    cout << "num of landmark feats to be initialized : " << feature_vec.size() << endl;
    while (it2 != feature_vec.end()) {
        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        feat.pseudo_anchor_clone_timestamp = (*it2)->pseudo_anchor_clone_timestamp;

        feat.p_FinG = (*it2)->p_FinG;

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f;
        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        std::vector<std::shared_ptr<Type>> Hx_order;

        // Get the Jacobian for this feature
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

        // cout << "Hf: " << H_f << endl;
        // cout << "Hx: " << H_x << endl;
        // cout << "res: " << res << endl;
        // UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

        // Create feature pointer
        auto landmark = std::make_shared<Landmark>(3);
        landmark->_featid = feat.featid;
        landmark->pseudo_anchor_clone_timestamp = feat.pseudo_anchor_clone_timestamp;
        landmark->set_from_xyz(feat.p_FinG);

        // Measurement noise matrix
        double sigma_pix_sq = _options_slam.sigma_pix_sq;
        Eigen::MatrixXd R = sigma_pix_sq * Eigen::MatrixXd::Identity(res.rows(), res.rows());

        // Try to initialize, delete new pointer if we failed
        double chi2_multipler = _options_slam.chi2_multipler;
        if (StateHelper::initialize(state, landmark, Hx_order, H_x, H_f, R, res, chi2_multipler)) {
            // cout << "feat_id: " << (*it2)->featid << endl;
            state->_features_SLAM.insert({(*it2)->featid, landmark});
            (*it2)->to_delete = true;
            it2++;
        } else {
            (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
        }
    }
    rT3 = boost::posix_time::microsec_clock::local_time();

    // cout << state->_imu->Rot() << endl;
    // cout << state->_imu->pos() << endl;

    // // Debug print timing information
    PRINT_INFO("[SLAM-DELAY]: %.4f seconds initialize (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
    if (!feature_vec.empty()) {
        PRINT_ALL("[SLAM-DELAY]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
        PRINT_ALL("[SLAM-DELAY]: %.4f seconds to triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
        PRINT_ALL("[SLAM-DELAY]: %.4f seconds initialize (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
        PRINT_ALL("[SLAM-DELAY]: %.4f seconds total\n", (rT3 - rT1).total_microseconds() * 1e-6);
    }
}

void UpdaterSLAM::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {
    // Return if no features
    if (feature_vec.empty())
        return;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3;
    rT0 = boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    std::vector<double> clonetimes;
    for (const auto &clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    auto it0 = feature_vec.begin();
    while (it0 != feature_vec.end()) {

        // Clean the feature
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements
        int ct_meas = (*it0)->timestamps.size();

        // Remove if we don't have enough
        if (ct_meas < 1) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }
    }
    rT1 = boost::posix_time::microsec_clock::local_time();

    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    for (size_t i = 0; i < feature_vec.size(); i++) {
        max_meas_size += 2 * feature_vec.at(i)->timestamps.size();
    }

    // Calculate max possible state size (i.e. the size of our covariance)
    size_t max_hx_size = state->max_covariance_size();

    // Large Jacobian, residual, and measurement noise of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(max_meas_size, max_meas_size);
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
    std::vector<std::shared_ptr<Type>> Hx_order_big;
    size_t ct_jacob = 0;
    size_t ct_meas = 0;

    // 4. Compute linear system for each feature, nullspace project, and reject
    auto it2 = feature_vec.begin();
    while (it2 != feature_vec.end()) {

        // Ensure we have the landmark and it is the same
        assert(state->_features_SLAM.find((*it2)->featid) != state->_features_SLAM.end());
        assert(state->_features_SLAM.at((*it2)->featid)->_featid == (*it2)->featid);

        // Get our landmark from the state
        std::shared_ptr<Landmark> landmark = state->_features_SLAM.at((*it2)->featid);

        // Convert the state landmark into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        feat.pseudo_anchor_clone_timestamp = (*it2)->pseudo_anchor_clone_timestamp;

        feat.p_FinG = landmark->get_xyz();

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f;
        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        std::vector<std::shared_ptr<Type>> Hx_order;

        // Get the Jacobian for this feature
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

        // Place Jacobians in one big Jacobian, since the landmark is already in our state vector
        Eigen::MatrixXd H_xf = H_x;
        H_xf.conservativeResize(H_x.rows(), H_x.cols() + H_f.cols());
        H_xf.block(0, H_x.cols(), H_x.rows(), H_f.cols()) = H_f;

        // Append to our Jacobian order vector
        std::vector<std::shared_ptr<Type>> Hxf_order = Hx_order;
        Hxf_order.push_back(landmark);

        // Chi2 distance check
        Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hxf_order);
        Eigen::MatrixXd S = H_xf * P_marg * H_xf.transpose();
        double sigma_pix_sq = _options_slam.sigma_pix_sq;
        S.diagonal() += sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
        double chi2 = res.dot(S.llt().solve(res));

        // Get our threshold (we precompute up to 500 but handle the case that it is more)
        double chi2_check;
        if (res.rows() < 500) {
            chi2_check = chi_squared_table[res.rows()];
        } else {
            boost::math::chi_squared chi_squared_dist(res.rows());
            chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
            // PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
        }

        // Check if we should delete or not
        double chi2_multipler = _options_slam.chi2_multipler;
        if (chi2 > chi2_multipler * chi2_check) {
            landmark->update_fail_count++;
            (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            PRINT_DEBUG("featid = %d\n", feat.featid);
            PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options_slam.chi2_multipler * chi2_check);
            std::stringstream ss;
            ss << "res = " << std::endl << res.transpose() << std::endl;
            PRINT_DEBUG(ss.str().c_str());
            continue;
        }

        // We are good!!! Append to our large H vector
        size_t ct_hx = 0;
        for (const auto &var : Hxf_order) {

            // Ensure that this variable is in our Jacobian
            if (Hx_mapping.find(var) == Hx_mapping.end()) {
                Hx_mapping.insert({var, ct_jacob});
                Hx_order_big.push_back(var);
                ct_jacob += var->size();
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas, Hx_mapping[var], H_xf.rows(), var->size()) = H_xf.block(0, ct_hx, H_xf.rows(), var->size());
            ct_hx += var->size();
        }

        // Our isotropic measurement noise
        R_big.block(ct_meas, ct_meas, res.rows(), res.rows()) *= _options_slam.sigma_pix_sq;

        // Append our residual and move forward
        res_big.block(ct_meas, 0, res.rows(), 1) = res;
        ct_meas += res.rows();
        it2++;
    }

    rT2 = boost::posix_time::microsec_clock::local_time();

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
    R_big.conservativeResize(ct_meas, ct_meas);

    // 5. With all good SLAM features update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
    rT3 = boost::posix_time::microsec_clock::local_time();

    // Debug print timing information
    PRINT_ALL("[SLAM-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
    PRINT_ALL("[SLAM-UP]: %.4f seconds creating linear system\n", (rT2 - rT1).total_microseconds() * 1e-6);
    PRINT_ALL("[SLAM-UP]: %.4f seconds to update (%d feats of %d size)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size(), (int)Hx_big.rows());
    PRINT_ALL("[SLAM-UP]: %.4f seconds total\n", (rT3 - rT1).total_microseconds() * 1e-6);
}

} // namespace night_voyager