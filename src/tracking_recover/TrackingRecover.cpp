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
#include "tracking_recover/TrackingRecover.h"
#include "core/LandMark.h"
#include "msckf_iekf/State.h"
#include "msckf_iekf/StateHelper.h"
#include "msckf_iekf/UpdaterHelper.h"
#include "prior_pose/PriorPoseManager.h"
#include "streetlight_matcher/HungaryEstimator.h"
#include "streetlight_matcher/PcdManager.h"
#include "streetlight_matcher/StreetlightFeature.h"
#include "streetlight_matcher/StreetlightFeatureDatabase.h"
#include "third_party/p3p_solver.h"
#include <Eigen/Geometry>
#include <numeric>
#include <pcl/search/kdtree.h>
#include <tbb/tbb.h>
#include <unordered_set>

namespace night_voyager {

void TrackingRecover::trans_after_nomatch(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos) {
    if (tracking_status != TrackingStatus::TRACKINGMAYLOST)
        return;

    if (nomatch_dist < 0.0 || nomatch_ang < 0.0) {
        nomatch_pos = pos;
        nomatch_rot = rot;
        nomatch_dist = 0.0;
        nomatch_ang = 0.0;
    } else {
        nomatch_dist += (pos - nomatch_pos).norm();
        nomatch_pos = pos;

        Eigen::AngleAxisd angle_axis(nomatch_rot.transpose() * rot);
        nomatch_ang += angle_axis.angle();
        nomatch_rot = rot;
    }

    return;
}

bool TrackingRecover::traverse_all_situations(std::shared_ptr<State> state, const CameraData &img, BoxData &st_boxes, std::shared_ptr<StreetlightFeatureDatabase> st_database,
                                              bool msckf) {

    // The largest box is selected to match all possible streetlights
    Eigen::Matrix3d R_ItoLoc = state->_imu->Rot();
    Eigen::Vector3d p_IinLoc = state->_imu->pos();
    Eigen::Matrix3d R_ItoMAP = state->_pose_MAPtoLOC->Rot().transpose() * state->_imu->Rot();
    Eigen::Vector3d p_IinMAP = state->_pose_MAPtoLOC->Rot().transpose() * (state->_imu->pos() - state->_pose_MAPtoLOC->pos());
    Eigen::Matrix3d R_MAPtoC = state->_calib_IMUtoCAM->Rot() * R_ItoMAP.transpose();
    Eigen::Vector3d p_MAPinC = -R_MAPtoC * p_IinMAP + state->_calib_IMUtoCAM->pos();

    std::vector<ScoreData> STs = select_streetlights(R_MAPtoC, p_MAPinC);

    auto it1 = st_boxes.rects.begin();
    auto it2 = st_boxes.centers.begin();
    auto it3 = st_boxes.centers_norm.begin();
    int edge = 40;
    while (it1 != st_boxes.rects.end()) {
        if (it1->area() < options.area_th || it1->x < edge || it1->y < edge || it1->br().x >= _cam->w() - edge || it1->br().y >= _cam->h() - edge) {
            it1 = st_boxes.rects.erase(it1);
            it2 = st_boxes.centers.erase(it2);
            it3 = st_boxes.centers_norm.erase(it3);
        } else {
            ++it1;
            ++it2;
            ++it3;
        }
    }

    if (st_boxes.rects.empty()) {
        PRINT_INFO(YELLOW "Detection boxes have been removed since they locate on the edge of image!\n");
        return false;
    }
    if (STs.size() == 0) {
        PRINT_INFO(YELLOW "No streetlights nearby!\n");
        return false;
    }

    std::vector<std::vector<int>> sts;
    std::vector<int> permutation;
    std::vector<bool> chosen(STs.size() + st_boxes.rects.size(), false);
    // Generate all possible match combinations
    generatePermutations(permutation, chosen, STs.size() + st_boxes.rects.size(), st_boxes.rects.size(), sts);

    // -1 indicates the streetlight detection has no match
    for (size_t i = 0; i < sts.size(); ++i) {
        for (size_t j = 0; j < sts[i].size(); ++j) {
            if (sts[i][j] >= STs.size()) {
                sts[i][j] = -1;
            }
        }
    }
    // Remove repetitive combinations
    struct Hash {
        size_t operator()(const std::vector<int> &v) const { return boost::hash_range(v.begin(), v.end()); }
    };
    std::unordered_set<std::vector<int>, Hash> unique_vector_set;
    std::vector<std::vector<int>> combs;
    for (const auto &vec : sts) {
        if (unique_vector_set.insert(vec).second) {
            combs.push_back(vec);
        }
    }

    /// Since we have odometer and prior pose, roll, pitch and distance from ground can be well constrained, the projected streetlights will be nearly
    /// horizontal with the observation
    std::vector<cv::Rect> search_rec_st(STs.size());
    std::vector<cv::Rect> aug_obs(st_boxes.rects.size());
    for (size_t i = 0; i < search_rec_st.size(); ++i) {
        search_rec_st[i] = cv::Rect(0, STs[i].pt.y() - options.search_bar_height / 2, _cam->w(), options.search_bar_height);
    }

    int off = 10;
    for (size_t i = 0; i < aug_obs.size(); ++i) {
        aug_obs[i] = cv::Rect(st_boxes.rects[i].x - off, st_boxes.rects[i].y - off, st_boxes.rects[i].width + 2 * off, st_boxes.rects[i].height + 2 * off);
    }

    auto it_comb = combs.begin();
    while (it_comb != combs.end()) {
        bool overlap = true;
        for (size_t i = 0; i < it_comb->size(); ++i) {
            if (it_comb->at(i) < 0)
                continue;
            if ((aug_obs[i] & search_rec_st[it_comb->at(i)]).area() == 0) {
                overlap = false;
                break;
            }
        }
        if (!overlap) {
            it_comb = combs.erase(it_comb);
        } else {
            ++it_comb;
        }
    }
    if (combs.size() == 1)
        return false;

    // Ensure that the combination of no matches is the first one
    for (size_t i = 0; i < combs.size(); ++i) {
        if (std::accumulate(combs[i].begin(), combs[i].end(), 0) == -st_boxes.rects.size()) {
            std::vector<int> tmp = combs[0];
            combs[0] = combs[i];
            combs[i] = tmp;
        }
    }

    size_t max_hx_size = state->max_covariance_size();
    for (auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }

    copy_state_stdatabase(state, combs.size(), st_database);
    std::vector<bool> feasible_states(combs.size(), true);

    std::vector<int> indices;
    std::vector<float> distances;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(_prpose->prior_pose_cloud);

    // tbb::parallel_for(tbb::blocked_range<size_t>(0, combs.size()), [&](const tbb::blocked_range<size_t>& r){
    for (size_t i = 0; i < combs.size(); ++i) {
        // for (size_t i = r.begin(); i != r.end(); ++i){

        int valid_matches = 0;
        for (size_t k = 0; k < combs[i].size(); ++k) {
            if (combs[i][k] >= 0)
                ++valid_matches;
        }

        size_t ct_jacob = 0;
        size_t ct_meas = 0;
        size_t max_meas_size = 2 * valid_matches;

        if (max_meas_size < 2)
            continue;

        Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
        Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
        Eigen::MatrixXd R_big = Eigen::MatrixXd::Zero(max_meas_size, max_meas_size);
        std::vector<std::shared_ptr<Type>> Hx_order_big;
        std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;

        std::vector<std::shared_ptr<StreetlightFeature>> cur_feats;

        for (size_t k = 0; k < combs[i].size(); ++k) {

            if (combs[i][k] < 0)
                continue;

            // Transform the match to feature
            std::shared_ptr<StreetlightFeature> feat = std::make_shared<StreetlightFeature>();
            feat->featid = STs[combs[i][k]].id;
            feat->boxes.push_back(st_boxes.rects[k]);
            feat->uvs.push_back(st_boxes.centers[k]);
            feat->uvs_norm.push_back(st_boxes.centers_norm[k]);
            feat->timestamps.push_back(st_boxes.timestamp);

            Eigen::MatrixXd H_x;
            Eigen::VectorXd res;
            Eigen::MatrixXd R;
            std::vector<std::shared_ptr<Type>> Hx_order;

            if (state->_kf == KFCLASS::MSCKF) {
                UpdaterHelper::get_stfeature_jacobian_full_msckf(states[i], feat, H_x, res, Hx_order, _pcd, R);
            } else {
                UpdaterHelper::get_stfeature_jacobian_full(states[i], feat, H_x, res, Hx_order, _pcd, R);
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

            cur_feats.push_back(feat);
            res_big.block(ct_meas, 0, res.rows(), 1) = res;
            ct_meas += res.rows();
        }

        assert(ct_meas <= max_meas_size);
        assert(ct_jacob <= max_hx_size);
        res_big.conservativeResize(ct_meas, 1);
        Hx_big.conservativeResize(ct_meas, ct_jacob);
        R_big.conservativeResize(ct_meas, ct_meas);

        // our noise is isotropic, so make it here after our compression
        UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
        R_big.conservativeResize(res_big.rows(), res_big.rows());
        R_big = options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

        if (ct_meas < 2) {
            feasible_states[i] = false;
            continue;
        }

        StateHelper::EKFUpdate(states[i], Hx_order_big, Hx_big, res_big, R_big);

        // Todo Prior Pose
        pcl::PointXYZ point3d;
        Eigen::Matrix3d update_R_ItoLoc = states[i]->_imu->Rot();
        Eigen::Vector3d update_p_IinLoc = states[i]->_imu->pos();
        Eigen::Matrix3d update_R_ItoMAP = states[i]->_pose_MAPtoLOC->Rot().transpose() * states[i]->_imu->Rot();
        Eigen::Vector3d update_p_IinMAP = states[i]->_pose_MAPtoLOC->Rot().transpose() * (states[i]->_imu->pos() - states[i]->_pose_MAPtoLOC->pos());
        Eigen::Vector3d e3(0, 0, 1);
        point3d.x = update_p_IinMAP.x(), point3d.y = update_p_IinMAP.y(), point3d.z = update_p_IinMAP.z();

        int num = kdtree.nearestKSearch(point3d, 1, indices, distances);

        if (num == 0 || std::abs(_prpose->prior_pose_cloud->at(indices[0]).z - point3d.z) > options.prior_z_diff_th) {
            cout << "large z" << endl;
            feasible_states[i] = false;
        }

        double delta_position = (update_p_IinMAP - p_IinMAP).norm();
        double delta_loc_position = (update_p_IinLoc - p_IinLoc).norm();
        double delta_rotation = Eigen::AngleAxisd(R_ItoMAP.transpose() * update_R_ItoMAP).angle();
        double delta_loc_rotation = Eigen::AngleAxisd(R_ItoLoc.transpose() * update_R_ItoLoc).angle();
        if ((update_p_IinMAP - p_IinMAP).norm() > 2 || delta_rotation > 0.7 || (update_p_IinLoc - p_IinLoc).norm() > 2 || delta_loc_rotation > 0.7) {
            if (delta_position > 2)
                cout << "large delta position: " << delta_position << endl;
            if (delta_rotation > 0.7)
                cout << "large delta rotation: " << delta_rotation << endl;
            if (delta_loc_position > 2)
                cout << "large delta loc position: " << delta_loc_position << endl;
            if (delta_loc_rotation > 0.7)
                cout << "large delta loc rotation: " << delta_loc_rotation << endl;
            feasible_states[i] = false;
        }

        if (feasible_states[i]) {
            for (const auto &feat : cur_feats) {
                st_databases[i]->update_feature(feat);
            }
        }
    }
    // });

    // Filtered state
    std::vector<std::shared_ptr<State>> new_states;
    std::vector<std::shared_ptr<StreetlightFeatureDatabase>> new_stdatabase;
    // cout << state->_imu->Rot() << endl;
    // cout << state->_imu->pos() << endl;
    for (size_t i = 0; i < states.size(); ++i) {
        // cout << states[i]->_imu->Rot() << endl;
        // cout << states[i]->_imu->pos() << endl;
        if (std::accumulate(combs[i].begin(), combs[i].end(), 0) == -st_boxes.rects.size()) {
            new_states.insert(new_states.begin(), states[i]);
            new_stdatabase.insert(new_stdatabase.begin(), st_databases[i]);
        } else if (feasible_states[i]) {
            new_states.push_back(states[i]);
            new_stdatabase.push_back(st_databases[i]);
        }
    }

    if (new_states.size() == 1) {
        // state = new_states[0];
        // cout << new_states[0]->_imu->Rot() << endl;
        // cout << new_states[0]->_imu->pos() << endl;
        // cout << state->_imu->Rot() << endl;
        // cout << state->_imu->pos() << endl;
        return false;
    }

    states.clear();
    states = new_states;

    st_databases.clear();
    st_databases = new_stdatabase;

    rots_loc.clear();
    poss_loc.clear();
    rots_global.clear();
    poss_global.clear();
    rots_loc.resize(states.size());
    poss_loc.resize(states.size());
    rots_global.resize(states.size());
    poss_global.resize(states.size());
    store_poses();

    return true;
}

void TrackingRecover::copy_state_stdatabase(std::shared_ptr<State> state, int states_num, std::shared_ptr<StreetlightFeatureDatabase> st_database) {
    states.resize(states_num);
    st_databases.resize(states_num);
    states[0] = state;
    st_databases[0] = st_database;
    for (int i = 1; i < states_num; ++i) {
        states[i] = std::make_shared<State>(*state);
        st_databases[i] = std::make_shared<StreetlightFeatureDatabase>(*st_database);
    }
}

bool TrackingRecover::check_if_select(const BoxData &cur_box) {
    if (nomatch_dist < 0.0 || nomatch_ang < 0.0) {
        nomatch_pos = states[0]->_imu->pos();
        nomatch_rot = states[0]->_imu->Rot();
        nomatch_dist = 0.0;
        nomatch_ang = 0.0;
    } else {
        nomatch_dist += (states[0]->_imu->pos() - nomatch_pos).norm();
        nomatch_pos = states[0]->_imu->pos();

        Eigen::AngleAxisd angle_axis(nomatch_rot.transpose() * states[0]->_imu->Rot());
        nomatch_ang += angle_axis.angle();
        nomatch_rot = states[0]->_imu->Rot();
    }

    if ((nomatch_dist > options.lost_dist_th || nomatch_ang > options.lost_ang_th) && cur_box.rects.size() >= 2) {
        return true;
    } else
        return false;
}

std::vector<STMatch> TrackingRecover::select_best_state(const CameraData &img, const BoxData &cur_box) {

    Eigen::Matrix3d R_MAPtoC = states[0]->_calib_IMUtoCAM->Rot() * states[0]->_imu->Rot().transpose() * states[0]->_pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinC = states[0]->_calib_IMUtoCAM->Rot() * (states[0]->_imu->Rot().transpose() * (states[0]->_pose_MAPtoLOC->pos() - states[0]->_imu->pos())) +
                               states[0]->_calib_IMUtoCAM->pos();

    BoxData small_box = process_box(img, cur_box);
    std::vector<ScoreData> STs = select_streetlights(R_MAPtoC, p_MAPinC);

    if (STs.empty())
        return std::vector<STMatch>();

    std::vector<PoseScore> stateMatches(states.size());
    // tbb::parallel_for(tbb::blocked_range<size_t>(0, states.size()), [&](const tbb::blocked_range<size_t>& r){
    for (size_t i = 0; i < states.size(); ++i) {
        // for (size_t i = r.begin(); i != r.end(); ++i){
        Eigen::Matrix3d R_MAPtoCi = states[i]->_calib_IMUtoCAM->Rot() * states[i]->_imu->Rot().transpose() * states[i]->_pose_MAPtoLOC->Rot();
        Eigen::Vector3d p_MAPinCi = states[i]->_calib_IMUtoCAM->Rot() * (states[i]->_imu->Rot().transpose() * (states[i]->_pose_MAPtoLOC->pos() - states[i]->_imu->pos())) +
                                    states[i]->_calib_IMUtoCAM->pos();

        Eigen::MatrixXd error_matrices = Eigen::MatrixXd::Ones(STs.size() + cur_box.centers.size(), STs.size() + cur_box.centers.size());
        stateMatches[i].matches.resize(cur_box.centers.size(), -1);

        std::vector<bool> valid_sts(STs.size(), false);
        std::vector<Eigen::Matrix3d> skew_unit_lines(cur_box.centers.size());
        std::vector<Eigen::Vector3d> unit_st_cams(STs.size());
        for (size_t j = 0; j < cur_box.centers.size(); ++j) {
            Eigen::Vector3d line = Eigen::Vector3d(cur_box.centers_norm[j].x(), cur_box.centers_norm[j].y(), 1);
            Eigen::Matrix3d skew_unit_line = skew(line.normalized());
            skew_unit_lines[j] = skew_unit_line;
        }
        for (size_t k = 0; k < STs.size(); ++k) {
            Eigen::Vector3d st_cam = R_MAPtoCi * _pcd->center_points[STs[k].id] + p_MAPinCi;
            if (st_cam.z() < 0.5 || st_cam.z() > options.z_th)
                continue;

            // double pt_x = _cam->get_fx() * st_cam.x() / st_cam.z() + _cam->get_cx();
            // double pt_y = _cam->get_fy() * st_cam.y() / st_cam.z() + _cam->get_cy();
            // if (pt_x < 0 || pt_x >= _cam->w() || pt_y < 0 || pt_y >= _cam->h()) continue;

            Eigen::Vector3d unit_st_cam = st_cam.normalized();
            unit_st_cams[k] = unit_st_cam;
            valid_sts[k] = true;
        }

        bool is_valid = false;
        for (size_t k = 0; k < STs.size(); ++k) {
            is_valid = is_valid || valid_sts[k];
        }
        if (!is_valid)
            continue;

        for (size_t j = 0; j < cur_box.centers.size(); ++j) {
            for (size_t k = 0; k < STs.size(); ++k) {
                if (valid_sts[k])
                    error_matrices(j, k) = (skew_unit_lines[j] * unit_st_cams[k]).norm();
            }
            for (size_t k = 0; k < cur_box.centers.size(); ++k) {
                error_matrices(j, STs.size() + k) = options.outlier_cost;
            }
        }
        HungaryEstimator estimator(error_matrices);
        cout << "error_matrices: " << error_matrices;
        std::vector<int> result = estimator.solve();

        stateMatches[i].score = 0;
        for (size_t j = 0; j < cur_box.centers.size(); ++j) {
            // if (result[j] < STs.size()){
            //     double inv_z = 1.0 / unit_st_cams[result[j]].z();
            //     Eigen::Vector2f st_pt;
            //     st_pt.x() = _cam->get_fx() * unit_st_cams[result[j]].x() * inv_z + _cam->get_cx();
            //     st_pt.y() = _cam->get_fy() * unit_st_cams[result[j]].y() * inv_z + _cam->get_cy();
            //     stateMatches[i].score -= (st_pt - cur_box.centers[j]).norm();
            // }
            stateMatches[i].matches[j] = result[j] < STs.size() ? STs[result[j]].id : -1;
            stateMatches[i].score -= error_matrices(j, result[j]);
        }
        // int valid_matches = std::count_if(stateMatches[i].matches.begin(), stateMatches[i].matches.end(), [](int x){return x > 0;});
        // stateMatches[i].score = stateMatches[i].score / cur_box.centers.size();

        // Calculate subscore
        std::vector<bool> box_used(small_box.rects.size(), false);
        float subscore = 0.0;
        for (size_t j = 0; j < STs.size(); ++j) {
            Eigen::Vector3d st_cam = R_MAPtoCi * _pcd->center_points[STs[j].id] + p_MAPinCi;
            double inv_z = 1.0 / st_cam.z();
            double st_u = inv_z * _cam->get_fx() * st_cam.x() + _cam->get_cx();
            double st_v = inv_z * _cam->get_fy() * st_cam.y() + _cam->get_cy();
            if (st_cam.z() <= 0.5 || st_u < 0 || st_u >= _cam->w() || st_v < 0 || st_v >= _cam->h())
                continue;

            // Score
            cv::Rect rect(int(st_u) - options.expansion_proj, int(st_v) - options.expansion_proj, options.expansion_proj * 2, options.expansion_proj * 2);
            float min_dist = -1;
            size_t min_id;
            for (size_t k = 0; k < small_box.rects.size(); ++k) {
                if ((rect & small_box.rects[k]).area() > 0 && !box_used[k]) {
                    float dist = std::sqrt(std::pow(st_u - small_box.centers[k].x(), 2) + std::pow(st_v - small_box.centers[k].y(), 2));
                    if (min_dist < 0 || min_dist > dist) {
                        min_dist = dist;
                        min_id = k;
                    }
                }
            }

            if (min_dist > 0) {
                box_used[min_id] = true;
                subscore += std::exp(-0.5 * min_dist / options.variance);
            }
        }

        stateMatches[i].score += options.subscore_weight * subscore;
        cout << "subscore: " << subscore << endl;
    }
    // });

    /// Select the optimal state
    float best_score = -10000;
    int best_idx;
    if (stateMatches.size() == 0) {
        PRINT_ERROR(RED "No state matches, there must be something error.\n" RESET);
        return std::vector<STMatch>();
    }
    for (size_t i = 0; i < stateMatches.size(); ++i) {
        // cout << "stateMatches.score: " << stateMatches[i].score << endl;
        // for (size_t j = 0; j < stateMatches[i].matches.size(); ++j)
        //     cout << "stateMatches.match: " << stateMatches[i].matches[j] << endl;
        if (best_score == -10000 || best_score < stateMatches[i].score) {
            best_idx = i;
            best_score = stateMatches[i].score;
        }
    }

    if (std::accumulate(stateMatches[best_idx].matches.begin(), stateMatches[best_idx].matches.end(), 0) == -stateMatches[best_idx].matches.size())
        return std::vector<STMatch>();

    std::vector<std::shared_ptr<State>> new_states;
    std::vector<std::shared_ptr<StreetlightFeatureDatabase>> new_stdatabases;
    std::vector<std::map<double, Eigen::Matrix3d>> new_rots_loc;
    std::vector<std::map<double, Eigen::Vector3d>> new_poss_loc;
    std::vector<std::map<double, Eigen::Matrix3d>> new_rots_global;
    std::vector<std::map<double, Eigen::Vector3d>> new_poss_global;

    new_states.resize(2);
    new_stdatabases.resize(2);
    new_rots_loc.resize(2);
    new_poss_loc.resize(2);
    new_rots_global.resize(2);
    new_poss_global.resize(2);

    new_states[0] = states[0], new_states[1] = states[best_idx];
    new_stdatabases[0] = st_databases[0], new_stdatabases[1] = st_databases[best_idx];
    new_rots_loc[0] = rots_loc[0], new_rots_loc[1] = rots_loc[best_idx];
    new_poss_loc[0] = poss_loc[0], new_poss_loc[1] = poss_loc[best_idx];
    new_rots_global[0] = rots_global[0], new_rots_global[1] = rots_global[best_idx];
    new_poss_global[0] = poss_global[0], new_poss_global[1] = poss_global[best_idx];

    states.clear();
    st_databases.clear();
    rots_loc.clear();
    poss_loc.clear();
    rots_global.clear();
    poss_global.clear();

    states = new_states;
    st_databases = new_stdatabases;
    rots_loc = new_rots_loc;
    poss_loc = new_poss_loc;
    rots_global = new_rots_global;
    poss_global = new_poss_global;

    std::vector<STMatch> M;
    Eigen::Matrix3d best_R_MAPtoC = states[best_idx]->_calib_IMUtoCAM->Rot() * states[best_idx]->_imu->Rot().transpose() * states[best_idx]->_pose_MAPtoLOC->Rot();
    Eigen::Vector3d best_p_MAPinC =
        states[best_idx]->_calib_IMUtoCAM->Rot() * (states[best_idx]->_imu->Rot().transpose() * (states[best_idx]->_pose_MAPtoLOC->pos() - states[best_idx]->_imu->pos())) +
        states[best_idx]->_calib_IMUtoCAM->pos();

    for (size_t j = 0; j < stateMatches[best_idx].matches.size(); ++j) {
        if (stateMatches[best_idx].matches[j] < 0)
            continue;
        STMatch m;
        m.st_id = stateMatches[best_idx].matches[j];
        m.rect = cur_box.rects[j];
        m.rect_center = cur_box.centers[j];
        m.rect_center_norm = cur_box.centers_norm[j];
        m.st_center_map = _pcd->center_points[m.st_id];
        m.st_center_cam = best_R_MAPtoC * m.st_center_map + best_p_MAPinC;
        M.push_back(m);
    }

    update_select_state(states[1], st_databases[1], M);
    return M;
}

std::vector<ScoreData> TrackingRecover::select_streetlights(const Eigen::Matrix3d &R_MAPtoC, const Eigen::Vector3d &p_MAPinC) {

    std::vector<ScoreData> STs;
    for (size_t i = 0; i < _pcd->center_points.size(); ++i) {
        Eigen::Vector3d cluster_center = _pcd->center_points[i];
        if (isnan(cluster_center.norm()))
            continue;

        Eigen::Vector3d Pc;
        Pc = R_MAPtoC * cluster_center + p_MAPinC;
        double inv_z = 1.0 / Pc.z();

        Eigen::Vector3d pt = inv_z * _cam->get_K_eigen() * Pc;
        // Ensure the streetlight is in the vision range
        if (Pc.z() > -0.5 && Pc.norm() < options.search_dist_scope && pt.x() >= -0.5 * _cam->w() && pt.y() >= -0.5 * _cam->h() && pt.x() < 1.5 * _cam->w() &&
            pt.y() < 1.5 * _cam->h()) {
            // Remove streetlights that overlap in image
            bool overlap = false;
            Eigen::Vector3d l1 = Pc.normalized();
            double n1 = Pc.norm();
            for (size_t j = 0; j < STs.size(); j++) {
                Eigen::Vector3d l2 = STs[j].Pc.normalized();
                if ((l1.transpose() * l2)(0) > 0.9998) {
                    double n2 = STs[j].Pc.norm();
                    overlap = true;
                    // select the closer streetlight
                    if (n2 > n1) {
                        STs[j].id = i;
                        STs[j].Pc = Pc;
                        STs[j].pt = pt.head<2>();
                    }
                    break;
                }
            }
            if (overlap)
                continue;

            ScoreData data;
            data.id = i;
            data.Pc = Pc;
            data.pt = pt.head<2>();
            STs.push_back(data);
        }
    }

    return STs;
}

BoxData TrackingRecover::process_box(const CameraData &img, const BoxData &cur_box) {
    cv::Mat grey_img = img.image;
    cv::Mat bin_img;
    cv::threshold(grey_img, bin_img, options.grey_thresh_low, 255, cv::THRESH_BINARY);

    BoxData boxes;
    vector<vector<cv::Point>> contours;
    cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        int area = rect.width * rect.height;
        if (area < 5)
            continue;

        if (rect.x != 0 && rect.y != 0 && rect.width != bin_img.cols && rect.height != bin_img.rows) {
            boxes.rects.push_back(rect);
            cv::Point2f center(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
            boxes.centers.push_back(Eigen::Vector2f(center.x, center.y));
            cv::Point2f norm_center = _cam->undistort_cv(center);
            boxes.centers_norm.push_back(Eigen::Vector2f(norm_center.x, norm_center.y));
        }
    }

    for (size_t i = 0; i < cur_box.rects.size(); ++i) {
        auto it1 = boxes.rects.begin();
        auto it2 = boxes.centers.begin();
        auto it3 = boxes.centers_norm.begin();

        while (it1 != boxes.rects.end()) {
            if ((*it1 & cur_box.rects[i]).area() > 0) {
                it1 = boxes.rects.erase(it1);
                it2 = boxes.centers.erase(it2);
                it3 = boxes.centers_norm.erase(it3);
            } else {
                ++it1;
                ++it2;
                ++it3;
            }
        }
    }

    return boxes;
}

void TrackingRecover::update_select_state(std::shared_ptr<State> state, std::shared_ptr<StreetlightFeatureDatabase> st_database, const std::vector<STMatch> &matches,
                                          bool msckf) {

    size_t max_hx_size = state->max_covariance_size();
    for (auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }

    /// Update state
    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    size_t max_meas_size = 2 * matches.size();

    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Zero(max_meas_size, max_meas_size);
    std::vector<std::shared_ptr<Type>> Hx_order_big;
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;

    std::vector<std::shared_ptr<StreetlightFeature>> cur_feats;

    for (size_t i = 0; i < matches.size(); ++i) {

        // Transform the match to feature
        std::shared_ptr<StreetlightFeature> feat = std::make_shared<StreetlightFeature>();
        feat->featid = matches[i].st_id;
        feat->boxes.push_back(matches[i].rect);
        feat->uvs.push_back(matches[i].rect_center);
        feat->uvs_norm.push_back(matches[i].rect_center_norm);
        feat->timestamps.push_back(state->_timestamp);

        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        Eigen::MatrixXd R;
        std::vector<std::shared_ptr<Type>> Hx_order;

        if (state->_kf == KFCLASS::MSCKF) {
            UpdaterHelper::get_stfeature_jacobian_full_msckf(state, feat, H_x, res, Hx_order, _pcd, R);
        } else {
            UpdaterHelper::get_stfeature_jacobian_full(state, feat, H_x, res, Hx_order, _pcd, R);
        }
        Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);

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

        cur_feats.push_back(feat);
        res_big.block(ct_meas, 0, res.rows(), 1) = res;
        ct_meas += res.rows();
    }

    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    R_big.conservativeResize(ct_meas, ct_meas);

    // our noise is isotropic, so make it here after our compression
    UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    R_big.conservativeResize(res_big.rows(), res_big.rows());
    R_big = options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);

    for (const auto &feat : cur_feats) {
        st_database->update_feature(feat);
    }
}

void TrackingRecover::generateCombinations(std::vector<int> &combination, int start, int n, int index, int r, std::vector<std::vector<int>> &allCombinations) {
    // 当前组合已满足所需长度，打印它
    if (index == r) {
        allCombinations.push_back(combination);
        return;
    }

    // 递归并在每个深度上添加一个元素
    for (int i = start; i <= n - (r - index); ++i) {
        combination[index] = i;
        generateCombinations(combination, i + 1, n, index + 1, r, allCombinations);
    }
}

void TrackingRecover::generatePermutations(std::vector<int> &permutation, std::vector<bool> &chosen, int n, int r, std::vector<std::vector<int>> &allPermutations) {
    if (permutation.size() == r) {
        allPermutations.push_back(permutation);
        return;
    }

    for (int i = 0; i < n; ++i) {
        if (chosen[i])
            continue;
        chosen[i] = true;
        permutation.push_back(i);
        generatePermutations(permutation, chosen, n, r, allPermutations);
        chosen[i] = false;
        permutation.pop_back();
    }
}

void TrackingRecover::store_poses() {
    for (size_t i = 0; i < states.size(); ++i) {
        double timestamp_inI = states[i]->_timestamp;
        rots_loc[i][timestamp_inI] = states[i]->_imu->Rot();
        poss_loc[i][timestamp_inI] = states[i]->_imu->pos();
        rots_global[i][timestamp_inI] = states[i]->_pose_MAPtoLOC->Rot();
        poss_global[i][timestamp_inI] = states[i]->_pose_MAPtoLOC->pos();
    }
}

} // namespace night_voyager