#include "msckf_iekf/UpdaterPlane.h"
#include "core/LandMark.h"
#include "msckf_iekf/State.h"
#include "msckf_iekf/StateHelper.h"
#include "msckf_iekf/UpdaterHelper.h"
#include "prior_pose/PriorPoseManager.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>

namespace night_voyager {

UpdaterPlane::UpdaterPlane(NightVoyagerOptions &options, std::shared_ptr<PriorPoseManager> prpose) : _options_plane(options.plane_options), _prpose(prpose) {

    for (int i = 1; i < 500; i++) {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
    }

    _Roi = options.Roi;
    _Rio = _Roi.transpose();
}

void UpdaterPlane::update(std::shared_ptr<State> state) {

    // Return if no clone states
    if (state->_clones_IMU.size() < state->_options.max_clone_size)
        return;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2;
    rT0 = boost::posix_time::microsec_clock::local_time();

    Eigen::Matrix3d R_GtoI;
    Eigen::Vector3d p_IinG;
    if (state->_clones_IMU.find(state->_timestamp) == state->_clones_IMU.end()) {
        R_GtoI = state->_imu->Rot().transpose();
        p_IinG = state->_imu->pos().transpose();
    } else {
        R_GtoI = state->_clones_IMU[state->_timestamp]->Rot().transpose();
        p_IinG = state->_clones_IMU[state->_timestamp]->pos();
    }

    // 1. Calculate the max possible measurement size, each plane has two constraints
    size_t max_meas_size = 0;
    for (auto &clone_imu : state->_clones_IMU) {
        if ((clone_imu.second->pos() - p_IinG).norm() <= _options_plane.distance_thresh_loc) {
            max_meas_size += 2;
        }
    }

    size_t max_hx_size = state->max_covariance_size();
    for (auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }

    // Large Jacobian and residual of all plane constraints for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(max_meas_size, max_meas_size);
    std::vector<std::shared_ptr<Type>> Hx_order_big;
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;

    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    std::shared_ptr<PoseHamilton> cur_state;
    if (state->_clones_IMU.find(state->_timestamp) == state->_clones_IMU.end()) {
        cur_state = state->_imu->pose();
    } else {
        cur_state = state->_clones_IMU[state->_timestamp];
    }
    // 2. Compute linear system for each state, nullspace project, and reject
    for (auto &clone_imu : state->_clones_IMU) {

        if (clone_imu.first == state->_timestamp)
            continue;

        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        std::vector<std::shared_ptr<Type>> Hx_order;

        if (state->_kf == KFCLASS::MSCKF)
            UpdaterHelper::get_plane_jacobian_full_msckf(cur_state, clone_imu.second, _Roi, H_x, res, Hx_order);
        else {
            UpdaterHelper::get_plane_jacobian_full(cur_state, clone_imu.second, _Roi, H_x, res, Hx_order);
        }

        Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
        Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
        S.diagonal() += _options_plane.distance_weight_loc_sq * (p_IinG - clone_imu.second->pos()).squaredNorm() * Eigen::VectorXd::Ones(S.rows());
        double chi2 = res.dot(S.llt().solve(res)); // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
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
        // cout << "noise: " << _options_plane.distance_weight_loc_sq * (p_IinG - clone_imu.second->pos()).squaredNorm() << endl;
        // cout << "chi2: " << chi2 << "  " << "chi2_multipler * chi2_check: " << _options_plane.chi2_multipler * chi2_check << endl;
        if (chi2 > _options_plane.chi2_multipler_loc * chi2_check) {
            PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options_plane.chi2_multipler_loc * chi2_check);
            std::stringstream ss;
            ss << "res = " << std::endl << res.transpose() << std::endl;
            PRINT_DEBUG(ss.str().c_str());
            continue;
        }

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

        res_big.block(ct_meas, 0, res.rows(), 1) = res;
        R_big.block(ct_meas, ct_meas, res.rows(), res.rows()) =
            _options_plane.distance_weight_loc_sq * (p_IinG - clone_imu.second->pos()).squaredNorm() * Eigen::MatrixXd::Identity(res.rows(), res.rows());
        ct_meas += res.rows();
    }
    rT1 = boost::posix_time::microsec_clock::local_time();

    // Return if we don't have anything and resize our matrices
    if (ct_meas < 2) {
        return;
    }

    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    R_big.conservativeResize(ct_meas, ct_meas);

    // 4. Update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
    rT2 = boost::posix_time::microsec_clock::local_time();

    PRINT_ALL("[PLANE-UP]: %.4f seconds to create plane constraint-based system\n", (rT1 - rT0).total_microseconds() * 1e-6);
    PRINT_ALL("[PLANE-UP]: %.4f seconds to update plane constraints\n", (rT2 - rT1).total_microseconds() * 1e-6);
}

void UpdaterPlane::update_with_prior(std::shared_ptr<State> state) {

    // Return if no clone states
    if (state->_clones_IMU.size() < state->_options.max_clone_size)
        return;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
    rT0 = boost::posix_time::microsec_clock::local_time();

    // 1. Search prior poses close to current poses
    Eigen::Matrix3d R_MAPtoLOC = state->_pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinLOC = state->_pose_MAPtoLOC->pos();

    std::unordered_map<std::shared_ptr<Type>, std::pair<v_quatd, v_vec3d>> state_prior_poses;
    // The indices are used for visualizing the valid searched prior poses
    std::unordered_map<std::shared_ptr<Type>, std::vector<int>> state_indices;
    std::unordered_map<std::shared_ptr<Type>, std::vector<bool>> state_priors_to_store;

    auto clone_imu = state->_clones_IMU.at(state->_timestamp);
    Eigen::Vector3d p_IiinMAP = R_MAPtoLOC.transpose() * (clone_imu->pos() - p_MAPinLOC);

    // We find close prior poses for each pose in the sliding window
    v_vec3d prior_poss;
    v_quatd prior_quats;
    std::vector<int> indices;
    std::vector<bool> to_store;
    _prpose->radiusSearch(p_IiinMAP, prior_poss, prior_quats, indices, _options_plane.distance_thresh_prior);

    if (state_prior_poses.find(clone_imu) == state_prior_poses.end())
        state_prior_poses.insert({clone_imu, std::make_pair(prior_quats, prior_poss)});
    if (state_indices.find(clone_imu) == state_indices.end())
        state_indices.insert({clone_imu, indices});
    if (state_priors_to_store.find(clone_imu) == state_priors_to_store.end()) {
        to_store.resize(indices.size());
        std::fill(to_store.begin(), to_store.end(), true);
        state_priors_to_store.insert({clone_imu, to_store});
    }

    rT1 = boost::posix_time::microsec_clock::local_time();

    // 2. Calculate the max possible measurement size, each plane has two constraints
    size_t max_meas_size = 0;
    // for (auto &clone_imu : state->_clones_IMU){
    //     if(state_indices.find(clone_imu.second) != state_indices.end()){
    //         max_meas_size += 2 * state_indices[clone_imu.second].size();
    //     }
    // }
    if (state_indices.find(clone_imu) != state_indices.end()) {
        max_meas_size += 2 * state_indices[clone_imu].size();
    }

    size_t max_hx_size = state->max_covariance_size();
    for (auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }

    // Large Jacobian and residual of all plane constraints for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(max_meas_size, max_meas_size);
    std::vector<std::shared_ptr<Type>> Hx_order_big;
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;

    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    // 3. Compute linear system for each state, nullspace project, and reject
    for (size_t i = 0; i < state_indices[clone_imu].size(); ++i) {
        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        std::vector<std::shared_ptr<Type>> Hx_order;

        Eigen::Matrix3d prior_rot = state_prior_poses[clone_imu].first[i].toRotationMatrix();
        Eigen::Vector3d prior_pos = state_prior_poses[clone_imu].second[i];
        if (state->_kf == KFCLASS::MSCKF)
            UpdaterHelper::get_plane_jacobian_full_msckf(clone_imu, state->_pose_MAPtoLOC, prior_rot, prior_pos, _Roi, H_x, res, Hx_order);
        else {
            UpdaterHelper::get_plane_jacobian_full(clone_imu, state->_pose_MAPtoLOC, prior_rot, prior_pos, _Roi, H_x, res, Hx_order);
        }
        Eigen::Vector3d pos_in_loc = state->_pose_MAPtoLOC->Rot() * prior_pos + state->_pose_MAPtoLOC->pos();
        // Eigen::Vector3d pos_in_loc = state->_pose_MAPtoLOC->Rot().transpose() * (clone_imu->pos() - state->_pose_MAPtoLOC->pos());

        Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
        Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
        S.diagonal() += _options_plane.distance_weight_prior_sq * Eigen::VectorXd::Ones(S.rows());
        double chi2 = res.dot(S.llt().solve(res)); // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
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
        if (chi2 > _options_plane.chi2_multipler_prior * chi2_check) {
            PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options_plane.chi2_multipler_prior * chi2_check);
            std::stringstream ss;
            ss << "res = " << std::endl << res.transpose() << std::endl;
            PRINT_DEBUG(ss.str().c_str());
            state_priors_to_store[clone_imu][i] = false;
            continue;
        }
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

        res_big.block(ct_meas, 0, res.rows(), 1) = res;
        // R_big.block(ct_meas, ct_meas, res.rows(), res.rows()) = std::pow(_options_plane.sigma_pos, 2) * Eigen::MatrixXd::Identity(res.rows(), res.rows());
        R_big.block(ct_meas, ct_meas, res.rows(), res.rows()) =
            _options_plane.distance_weight_prior_sq * (clone_imu->pos() - pos_in_loc).squaredNorm() * Eigen::MatrixXd::Identity(res.rows(), res.rows());
        ct_meas += res.rows();
    }
    rT2 = boost::posix_time::microsec_clock::local_time();

    // Return if we don't have anything and resize our matrices
    if (ct_meas < 1) {
        return;
    }

    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    R_big.conservativeResize(ct_meas, ct_meas);

    // 4. Update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
    rT3 = boost::posix_time::microsec_clock::local_time();

    // 5. Update the near_prior_pose_markers
    std::map<int, bool> near_indices_to_store;
    for (const auto &pair : state_indices) {
        assert(state_priors_to_store.find(pair.first) != state_priors_to_store.end());
        assert(state_priors_to_store[pair.first].size() == pair.second.size());

        for (size_t i = 0; i < pair.second.size(); ++i) {
            if (near_indices_to_store.find(pair.second[i]) != near_indices_to_store.end()) {
                near_indices_to_store[pair.second[i]] = near_indices_to_store[pair.second[i]] || state_priors_to_store[pair.first][i];
            } else {
                near_indices_to_store.insert({pair.second[i], state_priors_to_store[pair.first][i]});
            }
        }
    }

    if (near_indices_to_store.empty()) {
        PRINT_INFO(WHITE "[radius-search]: No neibouring prior pose found." RESET);
        return;
    }

    add_near_pose(near_indices_to_store);

    PRINT_ALL("[PRIOR-PLANE-UP]: %.4f seconds to search close prior poses\n", (rT1 - rT0).total_microseconds() * 1e-6);
    PRINT_ALL("[PRIOR-PLANE-UP]: %.4f seconds to create prior plane constraint-based system\n", (rT2 - rT1).total_microseconds() * 1e-6);
    PRINT_ALL("[PRIOR-PLANE-UP]: %.4f seconds to update prior plane constraints\n", (rT3 - rT2).total_microseconds() * 1e-6);
}

std::map<int, bool> UpdaterPlane::update_with_prior_tracking_recover(std::shared_ptr<State> state) {

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
    rT0 = boost::posix_time::microsec_clock::local_time();

    // 1. Search prior poses close to current poses
    Eigen::Matrix3d R_MAPtoLOC = state->_pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinLOC = state->_pose_MAPtoLOC->pos();

    std::unordered_map<std::shared_ptr<Type>, std::pair<v_quatd, v_vec3d>> state_prior_poses;
    // The indices are used for visualizing the valid searched prior poses
    std::unordered_map<std::shared_ptr<Type>, std::vector<int>> state_indices;
    std::unordered_map<std::shared_ptr<Type>, std::vector<bool>> state_priors_to_store;

    int idx = 0;
    for (auto &clone_imu : state->_clones_IMU) {

        if (idx % step != 0)
            continue;
        else
            idx++;

        Eigen::Vector3d p_IiinMAP = R_MAPtoLOC.transpose() * (clone_imu.second->pos() - p_MAPinLOC);

        // We find close prior poses for each pose in the sliding window
        v_vec3d prior_poss;
        v_quatd prior_quats;
        std::vector<int> indices;
        std::vector<bool> to_store;
        _prpose->radiusSearch(p_IiinMAP, prior_poss, prior_quats, indices, _options_plane.distance_thresh_prior);

        if (state_prior_poses.find(clone_imu.second) == state_prior_poses.end())
            state_prior_poses.insert({clone_imu.second, std::make_pair(prior_quats, prior_poss)});
        if (state_indices.find(clone_imu.second) == state_indices.end())
            state_indices.insert({clone_imu.second, indices});
        if (state_priors_to_store.find(clone_imu.second) == state_priors_to_store.end()) {
            to_store.resize(indices.size());
            std::fill(to_store.begin(), to_store.end(), true);
            state_priors_to_store.insert({clone_imu.second, to_store});
        }
    }
    rT1 = boost::posix_time::microsec_clock::local_time();

    // 2. Calculate the max possible measurement size, each plane has two constraints
    size_t max_meas_size = 0;
    for (auto &clone_imu : state->_clones_IMU) {
        if (state_indices.find(clone_imu.second) != state_indices.end()) {
            max_meas_size += 2 * state_indices[clone_imu.second].size();
        }
    }

    size_t max_hx_size = state->max_covariance_size();
    for (auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }

    // Large Jacobian and residual of all plane constraints for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(max_meas_size, max_meas_size);
    std::vector<std::shared_ptr<Type>> Hx_order_big;
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;

    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    // 3. Compute linear system for each state, nullspace project, and reject
    for (auto &clone_imu : state->_clones_IMU) {

        if (state_indices.find(clone_imu.second) == state_indices.end())
            continue;

        for (size_t i = 0; i < state_indices[clone_imu.second].size(); ++i) {
            Eigen::MatrixXd H_x;
            Eigen::VectorXd res;
            std::vector<std::shared_ptr<Type>> Hx_order;

            Eigen::Matrix3d prior_rot = state_prior_poses[clone_imu.second].first[i].toRotationMatrix();
            Eigen::Vector3d prior_pos = state_prior_poses[clone_imu.second].second[i];
            // cout << "prior_rot: " << prior_rot << endl;
            // cout << "prior_pos: " << prior_pos.transpose() << endl;
            // cout << "_Roi: " << _Roi << endl;
            if (state->_kf == KFCLASS::MSCKF)
                UpdaterHelper::get_plane_jacobian_full_msckf(clone_imu.second, state->_pose_MAPtoLOC, prior_rot, prior_pos, _Roi, H_x, res, Hx_order);
            else {
                UpdaterHelper::get_plane_jacobian_full(clone_imu.second, state->_pose_MAPtoLOC, prior_rot, prior_pos, _Roi, H_x, res, Hx_order);
            }
            Eigen::Vector3d pos_in_loc = state->_pose_MAPtoLOC->Rot() * prior_pos + state->_pose_MAPtoLOC->pos();

            Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
            Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
            S.diagonal() += _options_plane.distance_weight_prior_sq * (clone_imu.second->pos() - pos_in_loc).squaredNorm() * Eigen::VectorXd::Ones(S.rows());
            double chi2 = res.dot(S.llt().solve(res)); // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
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
            // cout << "noise: " << _options_plane.distance_weight_prior_sq * (clone_imu.second->pos() - pos_in_loc).squaredNorm() << endl;
            // cout << "chi2: " << chi2 << "  " << "chi2_multipler_prior * chi2_check: " << _options_plane.chi2_multipler_prior * chi2_check << endl;
            if (chi2 > _options_plane.chi2_multipler_prior * chi2_check) {
                PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options_plane.chi2_multipler_prior * chi2_check);
                std::stringstream ss;
                ss << "res = " << std::endl << res.transpose() << std::endl;
                PRINT_DEBUG(ss.str().c_str());
                state_priors_to_store[clone_imu.second][i] = false;
                continue;
            }

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

            res_big.block(ct_meas, 0, res.rows(), 1) = res;
            R_big.block(ct_meas, ct_meas, res.rows(), res.rows()) =
                _options_plane.distance_weight_prior_sq * (clone_imu.second->pos() - pos_in_loc).squaredNorm() * Eigen::MatrixXd::Identity(res.rows(), res.rows());
            ct_meas += res.rows();
        }
    }
    rT2 = boost::posix_time::microsec_clock::local_time();

    // Return if we don't have anything and resize our matrices
    if (ct_meas < 2) {
        return std::map<int, bool>();
    }

    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    R_big.conservativeResize(ct_meas, ct_meas);

    // 4. Update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
    rT3 = boost::posix_time::microsec_clock::local_time();

    // 5. Update the near_prior_pose_markers
    std::map<int, bool> near_indices_to_store;
    for (const auto &pair : state_indices) {
        assert(state_priors_to_store.find(pair.first) != state_priors_to_store.end());
        assert(state_priors_to_store[pair.first].size() == pair.second.size());

        for (size_t i = 0; i < pair.second.size(); ++i) {
            if (near_indices_to_store.find(pair.second[i]) != near_indices_to_store.end()) {
                near_indices_to_store[pair.second[i]] = near_indices_to_store[pair.second[i]] || state_priors_to_store[pair.first][i];
            } else {
                near_indices_to_store.insert({pair.second[i], state_priors_to_store[pair.first][i]});
            }
        }
    }

    return near_indices_to_store;
}

void UpdaterPlane::add_near_pose(const std::map<int, bool> &near_indices_to_store) { _prpose->addNearPriorPose(near_indices_to_store); }

} // namespace night_voyager
