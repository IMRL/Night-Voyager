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
#include "initializer/InertialInitializer.h"
#include "core/CommonLib.h"
#include "core/IMU.h"
#include "core/Type.h"
#include "feature_tracker/Feature.h"
#include "feature_tracker/FeatureDatabase.h"
#include "initializer/StaticInitializer.h"
#include "msckf_iekf/State.h"
#include "msckf_iekf/StateHelper.h"
#include "prior_pose/PriorPoseManager.h"
#include "streetlight_matcher/PcdManager.h"
#include "third_party/p3p_solver.h"
#include <fstream>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>
#include <queue>
#include <streetlight_matcher/HungaryEstimator.h>
#include <tbb/tbb.h>

namespace night_voyager {
InertialInitializer::InertialInitializer(NightVoyagerOptions &options, std::shared_ptr<FeatureDatabase> db, std::shared_ptr<PcdManager> pcd,
                                         std::shared_ptr<PriorPoseManager> prpose, std::shared_ptr<CamBase> &camera_intrinsic)
    : _options(options.init_options), _db(db), _pcd(pcd), _prpose(prpose), initialized_viwo(false), initialized_map(false),
      camera_calib(camera_intrinsic) {
    // Vector of our IMU data
    imu_data = std::make_shared<std::vector<ImuData>>();

    // Create initializers
    init_static = std::make_shared<StaticInitializer>(_options, _db, imu_data);

    if (options.init_options.need_init_in_map){
        for (size_t i = 0; i < _prpose->downsampled_pose_cloud->size(); ++i) {
            std::vector<int> indices;
            std::vector<float> distances;

            pcl::PointXY point = _prpose->downsampled_pose_cloud->at(i);
            _pcd->radiusSearch2d(Eigen::Vector2d(point.x, point.y), _options.search_dist_scope2, indices, distances);
            region_samples.push_back(Eigen::Vector2d(point.x, point.y));
            stidx_in_regions.push_back(indices);
            PRINT_ALL(WHITE "[Intialization of Initializer]: Prior pose: %f, %f. Find %d streetlights\n", point.x, point.y, indices.size(), RESET);

            // Create the combination of the streetlights
            std::vector<TupleScore> comb_clus;
            for (size_t li = 0; li < indices.size(); ++li) {
                for (size_t lj = 0; lj < indices.size(); ++lj) {
                    if (lj == li)
                        continue;
                    for (size_t lk = 0; lk < indices.size(); ++lk) {
                        if (lk == li || lk == lj)
                            continue;
                        TupleScore comb_clu;
                        comb_clu.l1 = indices[li], comb_clu.l2 = indices[lj], comb_clu.l3 = indices[lk];
                        comb_clus.push_back(comb_clu);
                    }
                }
            }
            std::vector<TupleScore>(comb_clus).swap(comb_clus);
            combs_in_region.push_back(comb_clus);
        }
    }
}

void InertialInitializer::feed_imu(const ImuData &message, double oldest_time) {

    // Append it to our vector
    imu_data->emplace_back(message);

    // Loop through and delete imu messages that are older than our requested time
    // std::cout << "INIT: imu_data.size() " << imu_data->size() << std::endl;
    if (oldest_time != -1) {
        auto it0 = imu_data->begin();
        while (it0 != imu_data->end()) {
            if (message.timestamp < oldest_time) {
                it0 = imu_data->erase(it0);
            } else {
                it0++;
            }
        }
    }
}

void InertialInitializer::feed_box(const BoxData &message, double oldest_time) {

    // Append it to our vector
    box_data.emplace_back(message);

    // Loop through and delete imu messages that are older than our requested time
    // std::cout << "INIT: imu_data.size() " << imu_data->size() << std::endl;
    if (oldest_time != -1) {
        auto it0 = box_data.begin();
        while (it0 != box_data.end()) {
            if (message.timestamp < oldest_time) {
                it0 = box_data.erase(it0);
            } else {
                it0++;
            }
        }
    }
}

void InertialInitializer::initialize_viwo(double &timestamp, Eigen::MatrixXd &covariance_viwo, std::vector<std::shared_ptr<Type>> &order_viwo,
                                          std::shared_ptr<IMU> t_imu) {
    // Get the newest and oldest timestamps we will try to initialize between!
    double newest_cam_time = -1;
    for (auto const &feat : _db->get_internal_data()) {
        auto const times = feat.second->timestamps;
        for (auto const &time : times) {
            newest_cam_time = std::max(newest_cam_time, time);
        }
    }
    double oldest_time = newest_cam_time - _options.init_window_time - 0.10;
    if (newest_cam_time < 0 || oldest_time < 0) {
        PRINT_INFO(YELLOW "[init-s]: Not enough data, return directly\n" RESET);
        initialized_viwo = false;
        return;
    }

    // Remove all measurements that are older then our initialization window
    // Then we will try to use all features that are in the feature database!
    _db->cleanup_measurements(oldest_time);
    auto it_imu = imu_data->begin();
    while (it_imu != imu_data->end() && it_imu->timestamp < oldest_time) {
        it_imu = imu_data->erase(it_imu);
    }

    initialized_viwo = init_static->initialize(timestamp, covariance_viwo, order_viwo, t_imu);
    return;
}

void InertialInitializer::initialize_map(const PackData &pack_data, const Eigen::Matrix4d &T_ItoC, const Eigen::Matrix4d &T_ItoG,
                                         const Eigen::MatrixXd &covariance_viwo_pose, Eigen::MatrixXd &T_MAPtoLOC, Eigen::MatrixXd &covariance_map) {

    if(!_options.need_init_in_map){
        Eigen::Matrix4d T_MAPtoI = Eigen::MatrixXd::Identity(4, 4);
        T_MAPtoI.block(0, 0, 3, 3) = _options.preset_Rwi.transpose();
        T_MAPtoI.block(0, 3, 3, 1) = - _options.preset_Rwi.transpose() * _options.preset_pwi;

        T_MAPtoLOC.setIdentity(4, 4);
        T_MAPtoLOC.block(0, 0, 3, 3) = T_ItoG.block(0, 0, 3, 3) * T_MAPtoI.block(0, 0, 3, 3);
        T_MAPtoLOC.block(0, 3, 3, 1) = T_ItoG.block(0, 0, 3, 3) * T_MAPtoI.block(0, 3, 3, 1) + T_ItoG.block(0, 3, 3, 1);

        covariance_map = std::pow(0.1, 2) * Eigen::MatrixXd::Identity(6, 6);
        covariance_map.block(0, 0, 3, 3) = std::pow(0.01, 2) * Eigen::Matrix3d::Identity(); // q
        covariance_map.block(3, 3, 3, 3) = std::pow(0.01, 2) * Eigen::Matrix3d::Identity(); // p
        initialized_map = true;

        cout << "T_MAPtoLOC: " << T_MAPtoLOC << endl;
        return;
    }

    if (box_data.front().timestamp > pack_data.cam.timestamp) {
        PRINT_ERROR(RED "[init-m]: All the timestamps of predicted streetlight boxes are older than current timestamp of image");
        return;
    }

    std::pair<CameraData, BoxData> good_pair;
    while (1) {
        if (box_data.empty())
            continue;
        if (box_data.back().timestamp >= pack_data.cam.timestamp) {
            std::lock_guard<std::mutex> lck(box_data_mtx);
            bool find_box = false;
            auto erase_idx = box_data.begin();
            auto it_box = box_data.begin();
            while (it_box != box_data.end()) {
                if (it_box->timestamp == pack_data.cam.timestamp) {
                    good_pair = std::make_pair(pack_data.cam, *it_box);
                    find_box = true;
                    erase_idx = it_box;
                    break;
                }
                ++it_box;
            }
            assert(find_box);
            box_data.erase(box_data.begin(), erase_idx);
            break;
        }
    }

    process_box(good_pair.first, good_pair.second);
    if (good_pair.second.rects.size() - _options.preset_outliers < 5) {
        PRINT_INFO(YELLOW "[init-m]: Not enough detection box!");
        return;
    }

    selected_box_data = good_pair.second;

    //===================================================================================
    // We have enough observations, Begin to estimate the initial pose_MAPtoLOC
    //===================================================================================

    std::vector<Tuple> comb_obs;
    Tuple comb_ob;
    comb_ob.o1 = 0, comb_ob.o2 = 1, comb_ob.o3 = 2;
    comb_obs.push_back(comb_ob);
    comb_ob.o1 = 3, comb_ob.o2 = 4, comb_ob.o3 = 5;
    comb_obs.push_back(comb_ob);
    comb_ob.o1 = 0, comb_ob.o2 = 2, comb_ob.o3 = 4;
    comb_obs.push_back(comb_ob);
    comb_ob.o1 = 1, comb_ob.o2 = 3, comb_ob.o3 = 5;
    comb_obs.push_back(comb_ob);
    comb_ob.o1 = 2, comb_ob.o2 = 3, comb_ob.o3 = 4;
    comb_obs.push_back(comb_ob);
    comb_ob.o1 = 1, comb_ob.o2 = 2, comb_ob.o3 = 3;
    comb_obs.push_back(comb_ob);
    // comb_ob.o1 = 0, comb_ob.o2 = 1, comb_ob.o3 = 4;
    // comb_obs.push_back(comb_ob);

    // The whole streelight map is divided into several regions according to the downsampled prior poses
    std::vector<std::vector<TupleScore>> top_tuples;
    std::vector<std::vector<PoseScore>> top_poses_in_region;
    top_tuples.resize(region_samples.size());
    top_poses_in_region.resize(region_samples.size());

    PRINT_INFO(WHITE "[init-m]: Number of regions is %d.\n", region_samples.size());

    for (size_t region_idx = 0; region_idx < region_samples.size(); ++region_idx) {
        PRINT_INFO(WHITE "[init-m]: Begin to estimate the initial pose in region %d. The region center is %.4f, %.4f\n", region_idx,
                   region_samples[region_idx].x(), region_samples[region_idx].y());

        // 1. We first count the whole combinations of observations and clusters
        for (auto &tpScore : combs_in_region[region_idx]) {
            tpScore.score = -10000;
            tpScore.matches.assign(good_pair.second.rects.size(), -1);
        }

        std::vector<int> stidx_in_region = stidx_in_regions[region_idx];
        std::vector<std::vector<TupleScore>> comb_clus(comb_obs.size(), combs_in_region[region_idx]);

        std::vector<std::vector<PoseScore>> poses_score_in_region(comb_clus.size(), std::vector<PoseScore>(combs_in_region[region_idx].size()));

        // 2.For all combinations we caculate the corresponding 2D pose and score

        P3PSolver p3p_solver;
        tbb::parallel_for(
            tbb::blocked_range2d<size_t>(0, comb_obs.size(), 0, combs_in_region[region_idx].size()), [&](const tbb::blocked_range2d<size_t> &r) {
                for (size_t tup_obs_idx = r.rows().begin(); tup_obs_idx != r.rows().end(); ++tup_obs_idx) {
                    // for (size_t tup_obs_idx = 0; tup_obs_idx != comb_obs.size(); ++tup_obs_idx){
                    for (size_t tup_clus_idx = r.cols().begin(); tup_clus_idx != r.cols().end(); ++tup_clus_idx) {
                        // for (size_t tup_clus_idx = 0; tup_clus_idx != combs_in_region[region_idx].size(); ++tup_clus_idx){
                        std::vector<Eigen::Vector3d> x;
                        std::vector<Eigen::Vector3d> X;

                        Eigen::Vector3d xi = Eigen::Vector3d::Ones();
                        xi.x() = good_pair.second.centers_norm[comb_obs[tup_obs_idx].o1].x(),
                        xi.y() = good_pair.second.centers_norm[comb_obs[tup_obs_idx].o1].y();
                        x.push_back(xi.normalized());
                        xi.x() = good_pair.second.centers_norm[comb_obs[tup_obs_idx].o2].x(),
                        xi.y() = good_pair.second.centers_norm[comb_obs[tup_obs_idx].o2].y();
                        x.push_back(xi.normalized());
                        xi.x() = good_pair.second.centers_norm[comb_obs[tup_obs_idx].o3].x(),
                        xi.y() = good_pair.second.centers_norm[comb_obs[tup_obs_idx].o3].y();
                        x.push_back(xi.normalized());

                        X.push_back(_pcd->center_points[comb_clus[tup_obs_idx][tup_clus_idx].l1]);
                        X.push_back(_pcd->center_points[comb_clus[tup_obs_idx][tup_clus_idx].l2]);
                        X.push_back(_pcd->center_points[comb_clus[tup_obs_idx][tup_clus_idx].l3]);

                        std::vector<Eigen::Matrix3d> Rs;
                        std::vector<Eigen::Vector3d> ts;
                        p3p_solver.p3p_ding(x, X, Rs, ts);
                        if (Rs.empty()) {
                            poses_score_in_region[tup_obs_idx][tup_clus_idx].score = -10000;
                            continue;
                        }

                        // We assume the observed streetlight is not too far away from camera center.
                        auto it_R = Rs.begin();
                        auto it_t = ts.begin();
                        while (it_R != Rs.end()) {
                            if (_options.prior_init_available) {
                                if ((*it_t - _options.prior_init).norm() > _options.prior_scale) {
                                    it_R = Rs.erase(it_R);
                                    it_t = ts.erase(it_t);
                                    continue;
                                }
                            }

                            if (std::isnan(it_t->x()) || std::isnan(it_t->y()) || std::isnan(it_t->z()) || std::isnan(it_R->norm())) {
                                it_R = Rs.erase(it_R);
                                it_t = ts.erase(it_t);
                                continue;
                            }
                            Eigen::Vector3d L1_cam = (*it_R).transpose() * (X[0] - *it_t);
                            Eigen::Vector3d L2_cam = (*it_R).transpose() * (X[1] - *it_t);
                            Eigen::Vector3d L3_cam = (*it_R).transpose() * (X[2] - *it_t);

                            if (L1_cam.z() < 0.5 || L2_cam.z() < 0.5 || L3_cam.z() < 0.5 || L1_cam.norm() > _options.max_st_dist ||
                                L2_cam.norm() > _options.max_st_dist || L3_cam.norm() > _options.max_st_dist) { // 55
                                it_R = Rs.erase(it_R);
                                it_t = ts.erase(it_t);
                                continue;
                            }

                            // The solution should be close to the prior pose in the z axis
                            std::vector<int> indices;
                            std::vector<float> distances;
                            pcl::KdTreeFLANN<pcl::PointXY> kdtree;
                            kdtree.setInputCloud(_prpose->prior_2dpose_cloud);
                            pcl::PointXY point2d;
                            point2d.x = it_t->x(), point2d.y = it_t->y();
                            int num = kdtree.nearestKSearch(point2d, 1, indices, distances);
                            if (num == 0 || std::abs(_prpose->prior_pose_cloud->at(indices[0]).z - it_t->z()) > 0.3) {
                                it_R = Rs.erase(it_R);
                                it_t = ts.erase(it_t);
                                continue;
                            }

                            std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> skew_unit_lines;
                            std::vector<int> box_map;
                            std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> unit_st_cams;
                            std::vector<int> st_map;
                            for (size_t i = 0; i < good_pair.second.centers.size(); ++i) {
                                if (i == comb_obs[tup_obs_idx].o1 || i == comb_obs[tup_obs_idx].o2 || i == comb_obs[tup_obs_idx].o3)
                                    continue;
                                Eigen::Vector3d line = Eigen::Vector3d(good_pair.second.centers_norm[i].x(), good_pair.second.centers_norm[i].y(), 1);
                                Eigen::Matrix3d skew_unit_line = skew(line.normalized());
                                skew_unit_lines.push_back(skew_unit_line);
                                box_map.push_back(i);
                            }

                            for (size_t j = 0; j < stidx_in_region.size(); ++j) {
                                if (stidx_in_region[j] == comb_clus[tup_obs_idx][tup_clus_idx].l1 ||
                                    stidx_in_region[j] == comb_clus[tup_obs_idx][tup_clus_idx].l2 ||
                                    stidx_in_region[j] == comb_clus[tup_obs_idx][tup_clus_idx].l3) {
                                    continue;
                                }
                                Eigen::Vector3d st_cam = (*it_R).transpose() * (_pcd->center_points[stidx_in_region[j]] - *it_t);
                                if (st_cam.z() < 0.5 || st_cam.norm() > _options.search_dist_scope1) {
                                    unit_st_cams.push_back(Eigen::Vector3d::Ones());
                                    st_map.push_back(-1);
                                    continue;
                                }

                                double pt_x = camera_calib->get_fx() * st_cam.x() / st_cam.z() + camera_calib->get_cx();
                                double pt_y = camera_calib->get_fy() * st_cam.y() / st_cam.z() + camera_calib->get_cy();
                                if (pt_x < 0 || pt_x >= camera_calib->w() || pt_y < 0 || pt_y >= camera_calib->h()) {
                                    unit_st_cams.push_back(Eigen::Vector3d::Ones());
                                    st_map.push_back(-1);
                                    continue;
                                }

                                Eigen::Vector3d unit_st_cam = st_cam.normalized();
                                unit_st_cams.push_back(unit_st_cam);
                                st_map.push_back(j);
                            }

                            size_t len = skew_unit_lines.size() + unit_st_cams.size();
                            Eigen::MatrixXd error_matrices = Eigen::MatrixXd::Ones(len, len);
                            for (size_t i = 0; i < skew_unit_lines.size(); ++i) {
                                for (size_t j = 0; j < unit_st_cams.size(); ++j) {
                                    if (st_map[j] >= 0) {
                                        error_matrices(i, j) = (skew_unit_lines[i] * unit_st_cams[j]).norm();
                                    }
                                }
                                for (size_t j = 0; j < skew_unit_lines.size(); ++j) {
                                    error_matrices(i, unit_st_cams.size() + j) = _options.sine_th;
                                }
                            }

                            HungaryEstimator estimator(error_matrices);
                            std::vector<int> result = estimator.solve();
                            double totalError = 0.0;
                            std::vector<double> errors(result.size());
                            std::vector<int> matches(good_pair.second.rects.size());
                            for (size_t i = 0; i < result.size(); ++i) {
                                if (i >= skew_unit_lines.size())
                                    break;
                                // totalError += error_matrices(i, result[i]);
                                errors[i] = error_matrices(i, result[i]);
                                matches[box_map[i]] = result[i] < unit_st_cams.size() ? stidx_in_region[st_map[result[i]]] : -1;
                            }

                            std::vector<int> sort_indices(result.size());
                            std::iota(sort_indices.begin(), sort_indices.end(), 0);
                            std::sort(sort_indices.begin(), sort_indices.end(), [&errors](size_t i1, size_t i2) { return errors[i1] < errors[i2]; });
                            // totalError = std::accumulate(errors.begin(), errors.begin() + result.size() - _options.preset_outliers, 0.0);
                            for (int i = 0; i < result.size() - _options.preset_outliers; ++i) {
                                totalError += errors[sort_indices[i]];
                            }
                            for (int i = 1; i <= _options.preset_outliers; ++i) {
                                matches[box_map[sort_indices[result.size() - i]]] = -1;
                            }

                            if (poses_score_in_region[tup_obs_idx][tup_clus_idx].score == -10000 ||
                                poses_score_in_region[tup_obs_idx][tup_clus_idx].score > totalError) {
                                poses_score_in_region[tup_obs_idx][tup_clus_idx].score = totalError;
                                poses_score_in_region[tup_obs_idx][tup_clus_idx].R = *it_R;
                                poses_score_in_region[tup_obs_idx][tup_clus_idx].p = *it_t;
                                poses_score_in_region[tup_obs_idx][tup_clus_idx].matches = matches;
                            }

                            ++it_R;
                            ++it_t;
                        }
                        if (Rs.empty()) {
                            poses_score_in_region[tup_obs_idx][tup_clus_idx].score = -10000;
                            continue;
                        }

                        poses_score_in_region[tup_obs_idx][tup_clus_idx].matches[comb_obs[tup_obs_idx].o1] = comb_clus[tup_obs_idx][tup_clus_idx].l1;
                        poses_score_in_region[tup_obs_idx][tup_clus_idx].matches[comb_obs[tup_obs_idx].o2] = comb_clus[tup_obs_idx][tup_clus_idx].l2;
                        poses_score_in_region[tup_obs_idx][tup_clus_idx].matches[comb_obs[tup_obs_idx].o3] = comb_clus[tup_obs_idx][tup_clus_idx].l3;
                    }
                }
            }); // End TBB

        // 3. For each region, we store the top x combinations (x is a hyper-parameter)
        auto comp = [](const PoseScore &a, const PoseScore &b) { return a.score > b.score; };

        struct Hash {
            std::size_t operator()(const std::vector<int> &v) const { return boost::hash_range(v.begin(), v.end()); }
        };
        std::unordered_map<std::vector<int>, double, Hash> seen;
        std::unordered_map<std::vector<int>, std::pair<int, int>, Hash> seen_pair;

        std::priority_queue<PoseScore, std::vector<PoseScore>, decltype(comp)> pq(comp);

        for (size_t i = 0; i < poses_score_in_region.size(); ++i) {
            for (size_t j = 0; j < poses_score_in_region[i].size(); ++j) {
                if (poses_score_in_region[i][j].score < 0) {
                    continue;
                }
                auto it = seen.find(poses_score_in_region[i][j].matches);
                if (it == seen.end() || (it != seen.end() && poses_score_in_region[i][j].score < it->second)) {
                    seen[poses_score_in_region[i][j].matches] = poses_score_in_region[i][j].score;
                    seen_pair[poses_score_in_region[i][j].matches] = std::make_pair(i, j);
                }
            }
        }
        for (const auto &pair : seen_pair) {
            pq.push(poses_score_in_region[pair.second.first][pair.second.second]);
        }
        std::vector<PoseScore> best_poses_in_region;
        for (int i = 0; i < _options.top_tuples_in_region; ++i) {
            if (pq.empty()) {
                PoseScore pose;
                pose.score = -10000;
                best_poses_in_region.push_back(pose);
                continue;
            }
            best_poses_in_region.push_back(pq.top());
            pq.pop();
        }
        top_poses_in_region[region_idx] = best_poses_in_region;
    } // End prpose

    // 4. 2D pose lifted up to 3D pose, using the nearest prior pose and Hungary algorithm to find matches, the one with the lowest cost is determined
    // as the best
    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, top_poses_in_region.size(), 0, _options.top_tuples_in_region), [&](const tbb::blocked_range2d<size_t> &r) {
            for (size_t region_idx = r.rows().begin(); region_idx < r.rows().end(); ++region_idx) {
                // for (size_t region_idx = 0; region_idx < top_tuples.size(); ++region_idx){
                for (size_t best_pose_i = r.cols().begin(); best_pose_i < r.cols().end(); ++best_pose_i) {
                    // for (size_t best_pose_i = 0; best_pose_i < _options.top_tuples_in_region; ++best_pose_i){
                    if (top_poses_in_region[region_idx][best_pose_i].score < 0)
                        continue;

                    // The initial value is estimated, and it will be input to the solvePnP function as initial values
                    Eigen::Matrix3d R_CtoMAP = top_poses_in_region[region_idx][best_pose_i].R;
                    Eigen::Vector3d p_CinMAP = top_poses_in_region[region_idx][best_pose_i].p;

                    std::vector<cv::Point3f> objectPoints;
                    std::vector<cv::Point2f> imagePoints;
                    for (size_t i = 0; i < top_poses_in_region[region_idx][best_pose_i].matches.size(); ++i) {
                        if (top_poses_in_region[region_idx][best_pose_i].matches[i] > 0) {
                            Eigen::Vector3d Pst_world = _pcd->center_points[top_poses_in_region[region_idx][best_pose_i].matches[i]];
                            objectPoints.push_back(cv::Point3f(Pst_world.x(), Pst_world.y(), Pst_world.z()));
                            imagePoints.push_back(cv::Point2f(good_pair.second.centers[i].x(), good_pair.second.centers[i].y()));
                        }
                    }

                    cv::Mat_<float> R = (cv::Mat_<float>(3, 3) << R_CtoMAP(0, 0), R_CtoMAP(0, 1), R_CtoMAP(0, 2), R_CtoMAP(1, 0), R_CtoMAP(1, 1),
                                         R_CtoMAP(1, 2), R_CtoMAP(2, 0), R_CtoMAP(2, 1), R_CtoMAP(2, 2));
                    cv::Mat_<float> tvec = (cv::Mat_<float>(3, 1) << p_CinMAP.x(), p_CinMAP.y(), p_CinMAP.z());
                    R = R.t();
                    tvec = -R * tvec;
                    cv::Mat rvec;
                    cv::Rodrigues(R, rvec);
                    cv::solvePnP(objectPoints, imagePoints, camera_calib->get_K_opencv(), camera_calib->get_D(), rvec, tvec, true);
                    cv::Rodrigues(rvec, R);
                    R = R.t();
                    tvec = -R * tvec;
                    for (int i = 0; i < 3; ++i) {
                        p_CinMAP(i) = tvec.at<float>(i, 0);
                        for (int j = 0; j < 3; ++j) {
                            R_CtoMAP(i, j) = R.at<float>(i, j);
                        }
                    }

                    std::vector<int> indices;
                    std::vector<float> distances;
                    pcl::KdTreeFLANN<pcl::PointXY> kdtree;
                    kdtree.setInputCloud(_prpose->prior_2dpose_cloud);
                    pcl::PointXY point2d;
                    point2d.x = p_CinMAP.x(), point2d.y = p_CinMAP.y();
                    // cout << "position: " << it_t->transpose() << endl;
                    int num = kdtree.nearestKSearch(point2d, 1, indices, distances);
                    if (num == 0 || std::abs(_prpose->prior_pose_cloud->at(indices[0]).z - p_CinMAP.z()) > 0.3) {
                        top_poses_in_region[region_idx][best_pose_i].score = -10000.0;
                        continue;
                    }

                    bool large_proj = false;
                    for (size_t j = 0; j < top_poses_in_region[region_idx][best_pose_i].matches.size(); ++j) {
                        if (top_poses_in_region[region_idx][best_pose_i].matches[j] > 0) {
                            double min_x = -1, min_y = -1, max_x = -1, max_y = -1;
                            for (const auto &point : _pcd->points[top_poses_in_region[region_idx][best_pose_i].matches[j]]) {
                                Eigen::Vector3d p_cam = R_CtoMAP.transpose() * (point - p_CinMAP);
                                double inv_z = 1.0 / p_cam.z();
                                Eigen::Vector3d pt = camera_calib->get_K_eigen() * inv_z * p_cam;
                                if (min_x < 0 || min_x > pt.x())
                                    min_x = pt.x();
                                if (min_y < 0 || min_y > pt.y())
                                    min_y = pt.y();
                                if (max_x < 0 || max_x < pt.x())
                                    max_x = pt.x();
                                if (max_y < 0 || max_y < pt.y())
                                    max_y = pt.y();
                            }
                            if (int(max_x - min_x) <= 0 || int(max_y - min_y) <= 0) {
                                continue;
                            }
                            cv::Rect proj_rect(int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y));
                            if (float(proj_rect.width) / good_pair.second.rects[j].width > 2.8 ||
                                float(proj_rect.height) / good_pair.second.rects[j].height > 2.8) {
                                large_proj = true;
                                break;
                            }
                        }
                    }
                    if (large_proj) {
                        top_poses_in_region[region_idx][best_pose_i].score = -10000.0;
                        continue;
                    }

                    std::vector<bool> box_used(unselected_box_data.rects.size(), false);
                    float score1 = 0.0, score2 = 0.0;
                    for (size_t j = 0; j < stidx_in_regions[region_idx].size(); ++j) {

                        Eigen::Vector3d st_cam = R_CtoMAP.transpose() * (_pcd->center_points[stidx_in_regions[region_idx][j]] - p_CinMAP);
                        double inv_z = 1.0 / st_cam.z();
                        double st_u = inv_z * camera_calib->get_fx() * st_cam.x() + camera_calib->get_cx();
                        double st_v = inv_z * camera_calib->get_fy() * st_cam.y() + camera_calib->get_cy();

                        // Score
                        auto st_find = std::find(top_poses_in_region[region_idx][best_pose_i].matches.begin(),
                                                 top_poses_in_region[region_idx][best_pose_i].matches.end(), stidx_in_regions[region_idx][j]);
                        if (st_find != top_poses_in_region[region_idx][best_pose_i].matches.end()) {
                            // proj_score
                            int obs_idx = st_find - top_poses_in_region[region_idx][best_pose_i].matches.begin();
                            score1 -= std::sqrt(std::pow(st_u - good_pair.second.centers[obs_idx].x(), 2) +
                                                std::pow(st_v - good_pair.second.centers[obs_idx].y(), 2));
                            continue;
                        }

                        if (st_cam.z() <= 0.5 || st_cam.z() > _options.search_dist_scope2 || st_u < 0 || st_u >= camera_calib->w() || st_v < 0 ||
                            st_v >= camera_calib->h())
                            continue;

                        cv::Rect rect(int(st_u) - _options.expansion_proj, int(st_v) - _options.expansion_proj, _options.expansion_proj * 2,
                                      _options.expansion_proj * 2);
                        // float min_sin_dtheta = -1;
                        float min_dist = -1;
                        size_t min_id;
                        for (size_t k = 0; k < unselected_box_data.rects.size(); ++k) {
                            if ((rect & unselected_box_data.rects[k]).area() > 0 && !box_used[k]) {
                                float dist =
                                    std::pow(st_u - unselected_box_data.centers[k].x(), 2) + std::pow(st_v - unselected_box_data.centers[k].y(), 2);
                                if (min_dist < 0 || min_dist > dist) {
                                    min_dist = dist;
                                    min_id = k;
                                }
                            }
                        }
                        if (min_dist > 0) {
                            box_used[min_id] = true;
                            score2 += std::exp(-0.5 * min_dist / 25);
                        }
                    }
                    for (const auto &match : top_poses_in_region[region_idx][best_pose_i].matches) {
                        if (match < 0) {
                            score1 -= _options.outlier_score;
                        }
                    }
                    score1 += _options.preset_outliers * _options.outlier_score;

                    // top_poses_in_region[region_idx][best_pose_i].score = score1 > 0.0 ? 1.0 / score1 : 0.0;
                    // top_poses_in_region[region_idx][best_pose_i].subscore = score1 + score2 > 0.0 ? 1.0 / (score1 + score2) : 0.0;
                    top_poses_in_region[region_idx][best_pose_i].score = score1 < 0.0 ? score1 : -10000.0;
                    top_poses_in_region[region_idx][best_pose_i].subscore = score1 + _options.subscore_weight * score2;
                    top_poses_in_region[region_idx][best_pose_i].R = R_CtoMAP;
                    top_poses_in_region[region_idx][best_pose_i].p = p_CinMAP;
                    top_poses_in_region[region_idx][best_pose_i].region = region_idx;
                }
            }
        }); // End TBB

    auto comp = [](const PoseScore &a, const PoseScore &b) { return a.score < b.score; };

    std::priority_queue<PoseScore, std::vector<PoseScore>, decltype(comp)> pq(comp);

    for (const auto &poses_in_region : top_poses_in_region) {
        for (const auto &pose : poses_in_region) {
            // if (pose.score > 0) pq.push(pose);
            if (pose.score > -10000.0)
                pq.push(pose);
        }
    }

    std::vector<PoseScore> top_poses;
    for (int i = 0; i < _options.top_poses; ++i) {
        if (pq.empty()) {
            break;
        }
        top_poses.push_back(pq.top());
        pq.pop();
    }

    std::vector<STMatch> top_M;
    Eigen::Matrix3d bestR;
    Eigen::Vector3d bestP;
    double best_subscore = -10000.0;
    for (const auto &pose : top_poses) {
        // cout << "score: " << pose.score << "  subscore: " << pose.subscore << endl;
        // cout << "R: " << pose.R << endl;
        // cout << "p: " << pose.p.transpose() << endl;
        // for (size_t j = 0; j < pose.matches.size(); ++j){
        //     cout << "  " << good_pair.second.centers[j].transpose() << "---->" << pose.matches[j] << " " <<
        //     _pcd->center_points[pose.matches[j]].transpose() << endl;
        // }
        if (best_subscore == -10000.0 || best_subscore < pose.subscore) {
            best_subscore = pose.subscore;

            std::vector<STMatch> M;
            for (size_t j = 0; j < pose.matches.size(); ++j) {
                STMatch m;
                m.st_id = pose.matches[j];
                m.rect = good_pair.second.rects[j];
                m.rect_center = good_pair.second.centers[j];
                m.rect_center_norm = good_pair.second.centers_norm[j];
                m.st_center_map = _pcd->center_points[m.st_id];
                m.st_center_cam = pose.R.transpose() * (_pcd->center_points[m.st_id] - pose.p);
                M.push_back(m);
            }
            top_M = M;
            bestR = pose.R;
            bestP = pose.p;
        }
    }

    cout << "bestR: " << bestR << endl;
    cout << "bestP: " << bestP << endl;

    // 5. Transform the estimated T_CtoMAP to the final T_MAPtoLOC
    Eigen::Matrix4d T_CtoMAP = Eigen::Matrix4d::Identity();
    T_CtoMAP.block(0, 0, 3, 3).noalias() = bestR;
    T_CtoMAP.block(0, 3, 3, 1).noalias() = bestP;

    Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
    T_CtoI.block(0, 0, 3, 3).noalias() = T_ItoC.block(0, 0, 3, 3).transpose();
    T_CtoI.block(0, 3, 3, 1).noalias() = -T_ItoC.block(0, 0, 3, 3).transpose() * T_ItoC.block(0, 3, 3, 1);

    Eigen::Matrix4d T_MAPtoC = Eigen::Matrix4d::Identity();
    T_MAPtoC.block(0, 0, 3, 3).noalias() = bestR.transpose();
    T_MAPtoC.block(0, 3, 3, 1).noalias() = -bestR.transpose() * bestP;

    T_MAPtoLOC = T_ItoG * T_CtoI * T_MAPtoC;

    // Calculate the covariance
    covariance_map = Eigen::MatrixXd::Identity(6, 6);
    covariance_map.block(0, 0, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity();
    covariance_map.block(3, 3, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity();
    initialized_map = true;
}

void InertialInitializer::process_box(const CameraData &img, BoxData &box) {

    // cv::Mat grey_img(img.image.rows, img.image.cols, CV_8UC1);
    // cv::cvtColor(img.image, grey_img, cv::COLOR_BGR2GRAY);
    cv::Mat grey_img = img.image;
    cv::Mat bin_img, low_bin_img;
    cv::threshold(grey_img, bin_img, _options.init_grey_thresh, 255, cv::THRESH_BINARY);
    cv::threshold(grey_img, low_bin_img, _options.init_grey_thresh_low, 255, cv::THRESH_BINARY);

    vector<cv::Rect> box1, box2;
    vector<vector<cv::Point>> contours, contours_low;
    cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(low_bin_img, contours_low, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        // cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;

        int area = rect.width * rect.height;
        if (area < 40) {
            continue;
        }

        if (rect.x != 0 && rect.y != 0 && rect.width != bin_img.cols && rect.height != bin_img.rows) {
            box2.push_back(rect);
        }
    }
    box1 = box.rects;

    for (size_t i = 0; i < contours_low.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours_low[i]);
        int area = rect.width * rect.height;

        if (area < 5) {
            continue;
        }

        if (rect.x != 0 && rect.y != 0 && rect.width != bin_img.cols && rect.height != bin_img.rows) {
            unselected_box_data.rects.push_back(rect);
            cv::Point2f center(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
            unselected_box_data.centers.push_back(Eigen::Vector2f(center.x, center.y));
            cv::Point2f norm_center = camera_calib->undistort_cv(center);
            unselected_box_data.centers_norm.push_back(Eigen::Vector2f(norm_center.x, norm_center.y));
        }
    }

    // Remove false detections and reduce the scale of detection box
    vector<cv::Rect> box12;
    vector<bool> box2uniond(box2.size(), false);
    for (size_t i = 0; i < box1.size(); i++) {
        cv::Rect rect1 = box1[i];
        bool has_union = false;

        cv::Rect rect3;
        for (size_t j = 0; j < box2.size(); j++) {
            cv::Rect rect2 = box2[j];
            int bin_min_u = 10000, bin_min_v = 10000, bin_max_u = -10000, bin_max_v = -10000;

            if ((rect1 & rect2).area()) {
                has_union = true;
                box2uniond[j] = true;
                cv::Rect union_rect = rect1 | rect2;
                for (int u = union_rect.x; u < union_rect.x + union_rect.width; u++) {
                    for (int v = union_rect.y; v < union_rect.y + union_rect.height; v++) {
                        float intensity;
                        intensity = grey_img.ptr<uchar>(v)[u];

                        if (intensity > 252) {
                            int light_nei = 0;
                            for (int du = -3; du < 4; du++)
                                for (int dv = -3; dv < 4; dv++) {
                                    float light_intensity;
                                    light_intensity = grey_img.ptr<uchar>(v + dv)[u + du];
                                    if (light_intensity > 249)
                                        light_nei++;
                                }
                            if (light_nei < 15)
                                continue;
                            bin_min_u = min(bin_min_u, u);
                            bin_min_v = min(bin_min_v, v);
                            bin_max_u = max(bin_max_u, u);
                            bin_max_v = max(bin_max_v, v);
                        }
                    }
                }
            } else
                continue;

            if (bin_max_u < bin_img.cols && bin_max_v < bin_img.rows && bin_min_u >= 0 && bin_min_v >= 0) {
                rect3 = cv::Rect(cv::Point(bin_min_u, bin_min_v), cv::Point(bin_max_u, bin_max_v));
                box12.push_back(rect3);
            }
        }

        // This rarely happens since CNN-detected lights usually can be detected by binary methods
        if (!has_union) {
            int bin_min_u = 10000, bin_min_v = 10000, bin_max_u = -10000, bin_max_v = -10000;
            for (int u = rect1.x; u < rect1.x + rect1.width; u++) {
                for (int v = rect1.y; v < rect1.y + rect1.height; v++) {
                    float intensity;
                    intensity = grey_img.ptr<uchar>(v)[u];

                    if (intensity > 252) {
                        int light_nei = 0;
                        for (int du = -3; du < 4; du++)
                            for (int dv = -3; dv < 4; dv++) {
                                float light_intensity;
                                light_intensity = grey_img.ptr<uchar>(v + dv)[u + du];
                                if (light_intensity > 249)
                                    light_nei++;
                            }
                        if (light_nei < 15)
                            continue;
                        bin_min_u = min(bin_min_u, u);
                        bin_min_v = min(bin_min_v, v);
                        bin_max_u = max(bin_max_u, u);
                        bin_max_v = max(bin_max_v, v);
                    }
                }
            }

            if (bin_min_u < bin_img.cols && bin_min_v < bin_img.rows && bin_max_u >= 0 && bin_max_v >= 0) {
                rect3 = cv::Rect(cv::Point(bin_min_u, bin_min_v), cv::Point(bin_max_u, bin_max_v));
                box12.push_back(rect3);
            }
        }
    }

    std::vector<bool> is_binary(box12.size(), false);
    // Add new detections to provide more lights for initialization
    if (_options.init_use_binary) {
        for (size_t i = 0; i < box2.size(); ++i) {
            if (!box2uniond[i]) {
                box12.push_back(box2[i]);
                is_binary.push_back(true);
            }
        }
    }

    // Ensure no detections overlap
    auto iter_i = box12.begin();
    auto iter_flag_i = is_binary.begin();
    while (iter_i != box12.end()) {
        bool erase_i = false;
        auto iter_j = iter_i + 1;
        auto iter_flag_j = iter_flag_i + 1;
        while (iter_j != box12.end()) {
            if ((*iter_i & *iter_j).area() > 0.25 * (*iter_i).area() || (*iter_i & *iter_j).area() > 0.25 * (*iter_j).area()) {
                if (iter_i->area() > iter_j->area()) {
                    erase_i = true;
                    break;
                } else {
                    iter_j = box12.erase(iter_j);
                    iter_flag_j = is_binary.erase(iter_flag_j);
                    continue;
                }
            }
            ++iter_j;
            ++iter_flag_j;
        }
        if (erase_i) {
            iter_i = box12.erase(iter_i);
            iter_flag_i = is_binary.erase(iter_flag_i);
        } else {
            ++iter_i;
            ++iter_flag_i;
        }
    }

    auto iter = box12.begin();
    auto iter_flag = is_binary.begin();
    while (iter != box12.end()) {

        float avg_intensity = 0.0;
        int area = iter->height * iter->width;
        for (int u = iter->x; u < iter->x + iter->width; u++) {
            for (int v = iter->y; v < iter->y + iter->height; v++) {
                float intensity = grey_img.ptr<uchar>(v)[u];
                avg_intensity += intensity;
            }
        }
        if (avg_intensity / area < 175.0 || area < 40) {
            iter = box12.erase(iter);
            iter_flag = is_binary.erase(iter_flag);
        } else {
            ++iter;
            ++iter_flag;
        }
    }

    vector<cv::Rect> box_bin, box_deep;
    for (size_t i = 0; i < box12.size(); ++i) {
        if (is_binary[i]) {
            box_bin.push_back(box12[i]);
        } else {
            box_deep.push_back(box12[i]);
        }
    }

    auto compare_area = [](const cv::Rect &a, const cv::Rect &b) -> bool { return a.area() > b.area(); };
    box.rects.clear();
    if (box_deep.size() < 6) {
        std::sort(box_bin.begin(), box_bin.end(), compare_area);
        box.rects = box_deep;
        if (box_bin.size() >= 6 - box_deep.size()) {
            box.rects.insert(box.rects.end(), box_bin.begin(), box_bin.begin() + 6 - box_deep.size());
        } else {
            box.rects.insert(box.rects.end(), box_bin.begin(), box_bin.end());
        }
    } else {
        std::sort(box_deep.begin(), box_deep.end(), compare_area);
        box_deep.erase(box_deep.begin() + 6, box_deep.end());
        box.rects = box_deep;
    }

    for (size_t i = 0; i < box.rects.size(); ++i) {
        cv::Point2f center = cv::Point2f(box.rects[i].tl().x + 0.5f * box.rects[i].width, box.rects[i].tl().y + 0.5f * box.rects[i].height);
        box.centers.push_back(Eigen::Vector2f(center.x, center.y));
        cv::Point2f norm_center = camera_calib->undistort_cv(center);
        box.centers_norm.push_back(Eigen::Vector2f(norm_center.x, norm_center.y));

        auto it1 = unselected_box_data.rects.begin();
        auto it2 = unselected_box_data.centers.begin();
        auto it3 = unselected_box_data.centers_norm.begin();
        while (it1 != unselected_box_data.rects.end()) {
            if ((*it1 & box.rects[i]).area() > 0) {
                it1 = unselected_box_data.rects.erase(it1);
                it2 = unselected_box_data.centers.erase(it2);
                it3 = unselected_box_data.centers_norm.erase(it3);
            } else {
                ++it1;
                ++it2;
                ++it3;
            }
        }
    }
}

void InertialInitializer::display_streetlights(cv::Mat &img_out, string overlay) {

    // cv::cvtColor(img_out, img_out, cv::COLOR_GRAY2RGB);
    // If the image is "small" thus we shoudl use smaller display codes
    bool is_small = (std::min(img_out.cols, img_out.rows) < 400);

    if (!initialized_map) {
        // draw, loop through all current streetlight detections
        for (size_t i = 0; i < selected_box_data.rects.size(); ++i) {
            Eigen::Vector2f pt = selected_box_data.centers[i];
            cv::Rect rect = selected_box_data.rects[i];
            cv::circle(img_out, cv::Point2f(pt.x(), pt.y()), (is_small) ? 3 : 6, cv::Scalar(0, 0, 200), -1);
            cv::rectangle(img_out, rect, cv::Scalar(0, 0, 200), 2);
            // cv::putText(img_out, std::to_string(int(pt.x())), rect.tl(),
            // cv::FONT_HERSHEY_COMPLEX_SMALL, (is_small) ? 1.0 : 1.5, cv::Scalar(0, 50, 255), 3);
        }

        for (size_t i = 0; i < unselected_box_data.rects.size(); ++i) {
            Eigen::Vector2f pt = unselected_box_data.centers[i];
            cv::Rect rect = unselected_box_data.rects[i];
            cv::circle(img_out, cv::Point2f(pt.x(), pt.y()), (is_small) ? 3 : 6, cv::Scalar(0, 200, 0), -1);
            cv::rectangle(img_out, rect, cv::Scalar(0, 200, 0), 2);
        }
    }

    // Draw what camera this is
    auto txtpt = (is_small) ? cv::Point(10, 30) : cv::Point(30, 60);
    cv::putText(img_out, overlay, txtpt, cv::FONT_HERSHEY_TRIPLEX,  (is_small) ? 0.75 : 1.5, cv::Scalar(0, 255, 0), 3);
}

} // namespace night_voyager
