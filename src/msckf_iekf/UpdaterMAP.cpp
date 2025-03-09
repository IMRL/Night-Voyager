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
#include "msckf_iekf/UpdaterMAP.h"
#include "core/LandMark.h"
#include "msckf_iekf/State.h"
#include "msckf_iekf/StateHelper.h"
#include "msckf_iekf/UpdaterHelper.h"
#include "streetlight_matcher/PcdManager.h"
#include "streetlight_matcher/StreetlightFeature.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace night_voyager {

BoxData UpdaterMAP::preprocess(const CameraData &img) {

    // We use the binary-based segmented boxes to reduce the area of deep-learning-based boxes
    cv::Mat grey_img = img.image;
    // cv::Mat grey_img(img.image.rows, img.image.cols, CV_8UC1);
    // cv::cvtColor(img.image, grey_img, cv::COLOR_BGR2GRAY);
    cv::Mat bin_img;
    cv::threshold(grey_img, bin_img, 250, 255, cv::THRESH_BINARY);

    vector<cv::Rect> box1, box2;
    vector<vector<cv::Point>> contours;
    cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        // cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;

        int area = rect.width * rect.height;
        if (area < 100 || rect.width < 10 || rect.height / rect.width > 5)
            continue;

        if (rect.x != 0 && rect.y != 0 && rect.width != bin_img.cols && rect.height != bin_img.rows)
            box2.push_back(rect);
    }
    box1 = cur_box.rects;

    // Remove false detections and reduce the scale of detection box
    vector<cv::Rect> box12;
    int edge = 5;
    vector<bool> box2uniond(box2.size(), false);
    for (size_t i = 0; i < box1.size(); i++) {
        cv::Rect rect1 = box1[i];
        bool has_union = false;

        cv::Rect rect3;
        for (size_t j = 0; j < box2.size(); j++) {
            cv::Rect rect2 = box2[j];
            int bin_min_u = 10000, bin_min_v = 10000, bin_max_u = -10000, bin_max_v = -10000;

            if ((rect1 & rect2).area() > 0) {
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
            if (bin_max_u < bin_img.cols - edge && bin_max_v < bin_img.rows - edge && bin_min_u >= edge && bin_min_v >= edge && bin_max_u > 0 && bin_max_v > 0 &&
                bin_min_u < bin_img.cols && bin_min_v < bin_img.rows) {
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

            if (bin_max_u < bin_img.cols - edge && bin_max_v < bin_img.rows - edge && bin_min_u >= edge && bin_min_v >= edge && bin_max_u > 0 && bin_max_v > 0 &&
                bin_min_u < bin_img.cols && bin_min_v < bin_img.rows) {
                rect3 = cv::Rect(cv::Point(bin_min_u, bin_min_v), cv::Point(bin_max_u, bin_max_v));
                box12.push_back(rect3);
            }
        }
    }

    // Ensure no detections overlap
    auto iter_i = box12.begin();
    while (iter_i != box12.end()) {
        bool erase_i = false;
        auto iter_j = iter_i + 1;
        while (iter_j != box12.end()) {
            if ((*iter_i & *iter_j).area() > 0.25 * (*iter_i).area() || (*iter_i & *iter_j).area() > 0.25 * (*iter_j).area()) {
                if (iter_i->area() > iter_j->area()) {
                    erase_i = true;
                    break;
                } else {
                    iter_j = box12.erase(iter_j);
                    continue;
                }
            }
            ++iter_j;
        }
        if (erase_i)
            iter_i = box12.erase(iter_i);
        else
            ++iter_i;
    }

    cur_box.rects.clear();
    cur_box.centers.clear();
    cur_box.centers_norm.clear();
    for (size_t i = 0; i < box12.size(); i++) {

        float avg_intensity = 0.0;
        int area = box12[i].height * box12[i].width;
        for (int u = box12[i].x; u < box12[i].x + box12[i].width; u++) {
            for (int v = box12[i].y; v < box12[i].y + box12[i].height; v++) {
                float intensity = grey_img.ptr<uchar>(v)[u];
                avg_intensity += intensity;
            }
        }
        if (avg_intensity / area < 175.0 || area < 40)
            continue;

        cur_box.rects.push_back(box12[i]);
        cur_box.centers.push_back(Eigen::Vector2f(0.5f * (box12[i].tl().x + box12[i].br().x), 0.5f * (box12[i].tl().y + box12[i].br().y)));
        cv::Point2f norm_center = cam->undistort_cv(cv::Point2f(cur_box.centers.back().x(), cur_box.centers.back().y()));
        cur_box.centers_norm.push_back(Eigen::Vector2f(norm_center.x, norm_center.y));
    }

    return cur_box;
}

void UpdaterMAP::match_and_update_dl_features(std::shared_ptr<State> state, double timestamp) {

    // We first get matches of streetlights in current frame
    boost::posix_time::ptime rT0, rT1;
    rT0 = boost::posix_time::microsec_clock::local_time();

    int Jaco_size = 0;
    // Jaco_size += state->_clones_IMU.at(timestamp)->size();
    Jaco_size += state->_imu->pose()->size();
    Jaco_size += state->_pose_MAPtoLOC->size();

    std::vector<std::shared_ptr<Type>> Hx_order;
    Hx_order.push_back(state->_imu->pose());
    // Hx_order.push_back(state->_clones_IMU.at(timestamp));
    Hx_order.push_back(state->_pose_MAPtoLOC);
    // Calculate R_CtoMAP and p_CinMAP
    // Eigen::Matrix3d R_GtoIi = state->_clones_IMU.at(timestamp)->Rot().transpose();
    // Eigen::Vector3d p_IiinG = state->_clones_IMU.at(timestamp)->pos();
    Eigen::Matrix3d R_IitoG = state->_imu->Rot();
    Eigen::Vector3d p_IiinG = state->_imu->pos();
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM->pos();
    Eigen::Matrix3d R_MAPtoLOC = state->_pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinLOC = state->_pose_MAPtoLOC->pos();

    Eigen::Matrix4d T_ItoC = Eigen::Matrix4d::Identity();
    T_ItoC.block<3, 3>(0, 0) = R_ItoC;
    T_ItoC.block<3, 1>(0, 3) = p_IinC;
    Eigen::Matrix4d T_MAPtoLOC = Eigen::Matrix4d::Identity();
    T_MAPtoLOC.block<3, 3>(0, 0) = R_MAPtoLOC;
    T_MAPtoLOC.block<3, 1>(0, 3) = p_MAPinLOC;
    Eigen::Matrix4d T_GtoIi = Eigen::Matrix4d::Identity();
    T_GtoIi.block<3, 3>(0, 0) = R_IitoG.transpose();
    T_GtoIi.block<3, 1>(0, 3) = -R_IitoG.transpose() * p_IiinG;

    Eigen::Matrix3d R_MAPtoC = R_ItoC * R_IitoG.transpose() * R_MAPtoLOC;
    Eigen::Vector3d p_MAPinC = R_ItoC * R_IitoG.transpose() * (p_MAPinLOC - p_IiinG) + p_IinC;

    // Eigen::Matrix3d R_CtoMAP = R_MAPtoLOC.transpose() * R_IitoG * R_ItoC.transpose();
    // Eigen::Vector3d p_CinMAP = R_MAPtoLOC.transpose() * (-R_IitoG * R_ItoC.transpose() * p_IinC + p_IiinG - p_MAPinLOC);
    // Eigen::Matrix4d T_CtoMAP = Eigen::Matrix4d::Identity();
    // T_CtoMAP.block<3, 3>(0, 0) = R_CtoMAP;
    // T_CtoMAP.block<3, 1>(0, 3) = p_CinMAP;

    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);

    // cout << "P_marg: " << endl << P_marg << endl;
    // [T_IitoG T_MAPtoLOC]
    // T_CtoMAP = T_MAPtoLOC.inverse() * T_IitoG * T_ItoC.inverse(), we use right disturbance model to derive the propagation of covariance
    Eigen::MatrixXd Jaco = Eigen::MatrixXd::Zero(6, Jaco_size);
    Jaco.block(0, 0, 6, 6) = -Adjoint_SEK3(T_ItoC);
    Jaco.block(0, 6, 6, 6) = Adjoint_SEK3(T_ItoC * T_GtoIi);
    // cout << "P_marg: " << endl << P_marg << endl;
    Eigen::MatrixXd P = Jaco * P_marg * Jaco.transpose();

    cur_dl_matches = st_matcher->run_dl_match(cur_box, _pcd, R_MAPtoC, p_MAPinC, P);

    rT1 = boost::posix_time::microsec_clock::local_time();

    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to match deep features\n", (rT1 - rT0).total_microseconds() * 1e-6);
}

void UpdaterMAP::match_and_update_dl_features_box(std::shared_ptr<State> state, const CameraData &message) {

    if (cur_dl_matches.size() + cur_bi_matches.size() >= 2)
        return;
    // We first get matches of streetlights in current frame
    boost::posix_time::ptime rT0, rT1;
    rT0 = boost::posix_time::microsec_clock::local_time();

    // Calculate R_MAPtoC and p_MAPtoC
    Eigen::Matrix3d R_GtoIi = state->_imu->Rot().transpose();
    Eigen::Vector3d p_IiinG = state->_imu->pos();
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM->pos();
    Eigen::Matrix3d R_MAPtoLOC = state->_pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinLOC = state->_pose_MAPtoLOC->pos();

    Eigen::Matrix3d R_MAPtoC = R_ItoC * R_GtoIi * R_MAPtoLOC;
    Eigen::Vector3d p_MAPinC = R_ItoC * R_GtoIi * (p_MAPinLOC - p_IiinG) + p_IinC;

    cur_dl_matches_box = st_matcher->run_dl_match_box(cur_box, _pcd, cur_dl_matches, R_MAPtoC, p_MAPinC);
}

void UpdaterMAP::match_and_update_bi_features(const CameraData &message, std::shared_ptr<State> state) {

    // We first get matches of streetlights in current frame
    boost::posix_time::ptime rT0, rT1;
    rT0 = boost::posix_time::microsec_clock::local_time();

    // Calculate R_MAPtoC and p_MAPtoC
    Eigen::Matrix3d R_GtoIi = state->_imu->Rot().transpose();
    Eigen::Vector3d p_IiinG = state->_imu->pos();
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM->pos();
    Eigen::Matrix3d R_MAPtoLOC = state->_pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinLOC = state->_pose_MAPtoLOC->pos();

    Eigen::Matrix3d R_MAPtoC = R_ItoC * R_GtoIi * R_MAPtoLOC;
    Eigen::Vector3d p_MAPinC = R_ItoC * R_GtoIi * (p_MAPinLOC - p_IiinG) + p_IinC;

    cur_bi_matches = st_matcher->run_bi_match(message, _pcd, cur_dl_matches, R_MAPtoC, p_MAPinC);
    rT1 = boost::posix_time::microsec_clock::local_time();

    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to match binary features\n", (rT1 - rT0).total_microseconds() * 1e-6);

    // Add feature to database
    // for(size_t i = 0; i < cur_bi_matches.size(); i++){
    //     STMatch match = cur_bi_matches[i];
    //     cv::Point2f npt = cam->undistort_cv(cv::Point2f(match.rect_center.x(), match.rect_center.y()));
    //     st_database->update_feature(match.st_id, timestamp, match.rect, match.rect_center, Eigen::Vector2f(npt.x, npt.y));
    // }
}

void UpdaterMAP::update_for_hist_features(std::shared_ptr<State> state, double timestamp) {

    if (st_database->database_empty()) {
        // Return if no features
        return;
    }
    // First update using deep learning-based detections
    boost::posix_time::ptime rT0, rT1, rT3, rT4, rT5;
    rT0 = boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    std::vector<double> clonetimes;
    for (const auto &clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    st_database->clean_old_measurements(clonetimes);
    rT1 = boost::posix_time::microsec_clock::local_time();

    // 2. Calculate the max possible measurement size and max possible state size
    size_t max_meas_size = st_database->count_measurements();
    size_t max_hx_size = state->max_covariance_size();
    for (auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }

    // Large Jacobian and residual of all features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Zero(max_meas_size, max_meas_size);
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
    std::vector<std::shared_ptr<Type>> Hx_order_big;

    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    auto all_idfeats = st_database->get_features_idlookup();
    auto it = all_idfeats.begin();
    while (it != all_idfeats.end()) {

        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        Eigen::MatrixXd R;
        std::vector<std::shared_ptr<Type>> Hx_order;

        if (state->_kf == KFCLASS::MSCKF)
            UpdaterHelper::get_stfeature_jacobian_full_msckf(state, it->second, H_x, res, Hx_order, _pcd, R);
        else {
            UpdaterHelper::get_stfeature_jacobian_full(state, it->second, H_x, res, Hx_order, _pcd, R);
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

        // Append our residual and move forward
        res_big.block(ct_meas, 0, res.rows(), 1) = res;
        ct_meas += res.rows();
        it++;
    }
    rT3 = boost::posix_time::microsec_clock::local_time();

    // Return if we don't have anything and resize our matrices
    if (ct_meas < 1) {
        return;
    }

    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    R_big.conservativeResize(ct_meas, ct_meas);

    // 3. Perform measurement compression
    // our noise is isotropic, so make it here after our compression
    UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    R_big.conservativeResize(res_big.rows(), res_big.rows());
    R_big = _options.map_options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());
    if (Hx_big.rows() < 1) {
        return;
    }
    rT4 = boost::posix_time::microsec_clock::local_time();

    // 4. With all good features update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
    rT5 = boost::posix_time::microsec_clock::local_time();

    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to create hist system (%d features)\n", (rT3 - rT1).total_microseconds() * 1e-6, all_idfeats.size());
    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to compress hist system\n", (rT4 - rT3).total_microseconds() * 1e-6);
    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to update hist features\n", (rT5 - rT4).total_microseconds() * 1e-6);
}

void UpdaterMAP::update_for_hist_features_tracking_recover(std::shared_ptr<State> state, double timestamp, std::shared_ptr<StreetlightFeatureDatabase> st_db) {

    if (st_db->database_empty()) {
        // Return if no features
        return;
    }
    // First update using deep learning-based detections
    boost::posix_time::ptime rT0, rT1, rT3, rT4, rT5;
    rT0 = boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    std::vector<double> clonetimes;
    for (const auto &clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    st_db->clean_old_measurements(clonetimes);
    rT1 = boost::posix_time::microsec_clock::local_time();

    // 2. Calculate the max possible measurement size and max possible state size
    size_t max_meas_size = st_db->count_measurements();
    size_t max_hx_size = state->max_covariance_size();
    for (auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }

    // Large Jacobian and residual of all features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Zero(max_meas_size, max_meas_size);
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
    std::vector<std::shared_ptr<Type>> Hx_order_big;

    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    auto all_idfeats = st_db->get_features_idlookup();
    auto it = all_idfeats.begin();
    while (it != all_idfeats.end()) {

        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        Eigen::MatrixXd R;
        std::vector<std::shared_ptr<Type>> Hx_order;

        if (state->_kf == KFCLASS::MSCKF)
            UpdaterHelper::get_stfeature_jacobian_full_msckf(state, it->second, H_x, res, Hx_order, _pcd, R);
        else {
            UpdaterHelper::get_stfeature_jacobian_full(state, it->second, H_x, res, Hx_order, _pcd, R);
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

        // Append our residual and move forward
        res_big.block(ct_meas, 0, res.rows(), 1) = res;
        ct_meas += res.rows();
        it++;
    }
    rT3 = boost::posix_time::microsec_clock::local_time();

    // Return if we don't have anything and resize our matrices
    if (ct_meas < 1) {
        return;
    }

    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    R_big.conservativeResize(ct_meas, ct_meas);

    // 3. Perform measurement compression
    // our noise is isotropic, so make it here after our compression
    UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    R_big.conservativeResize(res_big.rows(), res_big.rows());
    R_big = _options.map_options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());
    if (Hx_big.rows() < 1) {
        return;
    }
    rT4 = boost::posix_time::microsec_clock::local_time();

    // 4. With all good features update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
    rT5 = boost::posix_time::microsec_clock::local_time();

    // PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to clean [tracking_recovering]\n", (rT1 - rT0).total_microseconds() * 1e-6);
    // PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to create hist system (%d features) [tracking_recovering]\n", (rT3 - rT1).total_microseconds() * 1e-6,
    // all_idfeats.size()); PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to compress hist system [tracking_recovering]\n", (rT4 - rT3).total_microseconds()
    // * 1e-6); PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to update hist features [tracking_recovering]\n", (rT5 - rT4).total_microseconds() * 1e-6);
}

void UpdaterMAP::update_for_dl_features(std::shared_ptr<State> state, double timestamp) {
    // First update using deep learning-based detections
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4;
    rT0 = boost::posix_time::microsec_clock::local_time();

    // 2. Calculate the max possible measurement size and max possible state size
    size_t max_meas_size = 2 * cur_dl_matches.size();
    size_t max_hx_size = state->max_covariance_size();
    for (auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }

    vector<std::shared_ptr<StreetlightFeature>> cur_feats;
    // Large Jacobian and residual of all features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Zero(max_meas_size, max_meas_size);
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
    std::vector<std::shared_ptr<Type>> Hx_order_big;

    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    auto it = cur_dl_matches.begin();
    while (it != cur_dl_matches.end()) {

        // Transform the match to feature
        std::shared_ptr<StreetlightFeature> feat = std::make_shared<StreetlightFeature>();
        feat->featid = it->st_id;
        feat->boxes.push_back(it->rect);
        feat->uvs.push_back(it->rect_center);
        cv::Point2f npt = cam->undistort_cv(cv::Point2f(it->rect_center.x(), it->rect_center.y()));
        feat->uvs_norm.push_back(Eigen::Vector2f(npt.x, npt.y));
        feat->timestamps.push_back(timestamp);

        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        Eigen::MatrixXd R;
        std::vector<std::shared_ptr<Type>> Hx_order;

        if (state->_kf == KFCLASS::MSCKF)
            UpdaterHelper::get_stfeature_jacobian_full_msckf(state, feat, H_x, res, Hx_order, _pcd, R);
        else {
            UpdaterHelper::get_stfeature_jacobian_full(state, feat, H_x, res, Hx_order, _pcd, R);
        }

        Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
        Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
        S.diagonal() += _options.map_options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
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
        if (chi2 > _options.map_options.chi2_multipler_dl * chi2_check) {
            feat->to_delete = true;
            it = cur_dl_matches.erase(it);
            PRINT_DEBUG("featid = %d\n", feat->featid);
            PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options.map_options.chi2_multipler_dl * chi2_check);
            std::stringstream ss;
            ss << "res = " << std::endl << res.transpose() << std::endl;
            PRINT_DEBUG(ss.str().c_str());
            continue;
        }
        // }
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
        if (!feat->to_delete) {
            cur_feats.push_back(feat);
        }
        res_big.block(ct_meas, 0, res.rows(), 1) = res;
        ct_meas += res.rows();
        it++;
    }
    rT1 = boost::posix_time::microsec_clock::local_time();

    // Return if we don't have anything and resize our matrices
    if (ct_meas < 1) {
        return;
    }

    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    R_big.conservativeResize(ct_meas, ct_meas);

    // 3. Perform measurement compression
    // our noise is isotropic, so make it here after our compression
    UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    R_big.conservativeResize(res_big.rows(), res_big.rows());
    R_big = _options.map_options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());
    if (Hx_big.rows() < 1) {
        return;
    }
    rT2 = boost::posix_time::microsec_clock::local_time();

    // 4. With all good features update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);

    rT3 = boost::posix_time::microsec_clock::local_time();

    // 5. Update our StreetlightFeatureDatabase
    for (size_t i = 0; i < cur_feats.size(); i++) {
        st_database->update_feature(cur_feats[i]);
    }
    rT4 = boost::posix_time::microsec_clock::local_time();
    if (!cur_dl_matches.empty())
        no_matches = 0;

    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to create DL system (%d features)\n", (rT1 - rT0).total_microseconds() * 1e-6, cur_dl_matches.size());
    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to compress DL system (%d features)\n", (rT2 - rT1).total_microseconds() * 1e-6, cur_dl_matches.size());
    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to update DL system\n", (rT3 - rT2).total_microseconds() * 1e-6);
    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to update DL features\n", (rT4 - rT3).total_microseconds() * 1e-6);
}

void UpdaterMAP::update_for_dl_features_box(std::shared_ptr<State> state, double timestamp) {
    if (cur_dl_matches.size() + cur_bi_matches.size() >= 2)
        return;
    // First update using deep learning-based detections
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4;
    rT0 = boost::posix_time::microsec_clock::local_time();

    // 2. Calculate the max possible measurement size and max possible state size
    size_t max_meas_size = 2 * cur_dl_matches_box.size();
    size_t max_hx_size = state->max_covariance_size();
    for (auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }

    vector<std::shared_ptr<StreetlightFeature>> cur_feats;
    // Large Jacobian and residual of all features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Zero(max_meas_size, max_meas_size);
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
    std::vector<std::shared_ptr<Type>> Hx_order_big;

    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    auto it = cur_dl_matches_box.begin();
    while (it != cur_dl_matches_box.end()) {

        // Transform the match to feature
        std::shared_ptr<StreetlightFeature> feat = std::make_shared<StreetlightFeature>();
        feat->featid = it->st_id;
        feat->boxes.push_back(it->rect);
        feat->uvs.push_back(it->rect_center);
        cv::Point2f npt = cam->undistort_cv(cv::Point2f(it->rect_center.x(), it->rect_center.y()));
        feat->uvs_norm.push_back(Eigen::Vector2f(npt.x, npt.y));
        feat->timestamps.push_back(timestamp);

        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        Eigen::MatrixXd R;
        std::vector<std::shared_ptr<Type>> Hx_order;

        if (state->_kf == KFCLASS::MSCKF)
            UpdaterHelper::get_stfeature_jacobian_full_msckf(state, feat, H_x, res, Hx_order, _pcd, R);
        else {
            UpdaterHelper::get_stfeature_jacobian_full(state, feat, H_x, res, Hx_order, _pcd, R);
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
        ct_meas += res.rows();
        it++;
    }
    rT1 = boost::posix_time::microsec_clock::local_time();

    // Return if we don't have anything and resize our matrices
    if (ct_meas < 1) {
        return;
    }

    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    R_big.conservativeResize(ct_meas, ct_meas);

    // 3. Perform measurement compression
    // our noise is isotropic, so make it here after our compression
    UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    R_big.conservativeResize(res_big.rows(), res_big.rows());
    R_big = _options.map_options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());
    if (Hx_big.rows() < 1) {
        return;
    }
    rT2 = boost::posix_time::microsec_clock::local_time();

    // 4. With all good features update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);

    rT3 = boost::posix_time::microsec_clock::local_time();

    // 5. Update our StreetlightFeatureDatabase
    for (size_t i = 0; i < cur_feats.size(); i++) {
        st_database->update_feature(cur_feats[i]);
    }
    rT4 = boost::posix_time::microsec_clock::local_time();
    if (!cur_dl_matches_box.empty())
        no_matches = 0;

    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to create DL system (%d features)\n", (rT1 - rT0).total_microseconds() * 1e-6, cur_dl_matches.size());
    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to compress DL system (%d features)\n", (rT2 - rT1).total_microseconds() * 1e-6, cur_dl_matches.size());
    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to update DL system\n", (rT3 - rT2).total_microseconds() * 1e-6);
    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to update DL features\n", (rT4 - rT3).total_microseconds() * 1e-6);
}

vector<STMatch> UpdaterMAP::match_and_update_for_dl_features_tracking_recover(std::shared_ptr<State> state, double timestamp,
                                                                              std::shared_ptr<StreetlightFeatureDatabase> st_db, double reloc_z_th, int reloc_extend) {

    int Jaco_size = 0;
    // Jaco_size += state->_clones_IMU.at(timestamp)->size();
    Jaco_size += state->_imu->pose()->size();
    Jaco_size += state->_pose_MAPtoLOC->size();

    std::vector<std::shared_ptr<Type>> Hx_order;
    Hx_order.push_back(state->_imu->pose());
    // Hx_order.push_back(state->_clones_IMU.at(timestamp));
    Hx_order.push_back(state->_pose_MAPtoLOC);
    // Calculate R_CtoMAP and p_CinMAP
    // Eigen::Matrix3d R_GtoIi = state->_clones_IMU.at(timestamp)->Rot().transpose();
    // Eigen::Vector3d p_IiinG = state->_clones_IMU.at(timestamp)->pos();
    Eigen::Matrix3d R_IitoG = state->_imu->Rot();
    Eigen::Vector3d p_IiinG = state->_imu->pos();
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM->pos();
    Eigen::Matrix3d R_MAPtoLOC = state->_pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinLOC = state->_pose_MAPtoLOC->pos();

    Eigen::Matrix4d T_ItoC = Eigen::Matrix4d::Identity();
    T_ItoC.block<3, 3>(0, 0) = R_ItoC;
    T_ItoC.block<3, 1>(0, 3) = p_IinC;
    Eigen::Matrix4d T_MAPtoLOC = Eigen::Matrix4d::Identity();
    T_MAPtoLOC.block<3, 3>(0, 0) = R_MAPtoLOC;
    T_MAPtoLOC.block<3, 1>(0, 3) = p_MAPinLOC;
    Eigen::Matrix4d T_GtoIi = Eigen::Matrix4d::Identity();
    T_GtoIi.block<3, 3>(0, 0) = R_IitoG.transpose();
    T_GtoIi.block<3, 1>(0, 3) = -R_IitoG.transpose() * p_IiinG;

    Eigen::Matrix3d R_MAPtoC = R_ItoC * R_IitoG.transpose() * R_MAPtoLOC;
    Eigen::Vector3d p_MAPinC = R_ItoC * R_IitoG.transpose() * (p_MAPinLOC - p_IiinG) + p_IinC;

    vector<STMatch> matches = st_matcher->run_dl_match_tracking_recover(cur_box, _pcd, R_MAPtoC, p_MAPinC, reloc_z_th, reloc_extend);

    // 2. Calculate the max possible measurement size and max possible state size
    size_t max_meas_size = 2 * matches.size();
    size_t max_hx_size = state->max_covariance_size();
    for (auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }

    vector<std::shared_ptr<StreetlightFeature>> cur_feats;
    // Large Jacobian and residual of all features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Zero(max_meas_size, max_meas_size);
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
    std::vector<std::shared_ptr<Type>> Hx_order_big;

    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    auto it = matches.begin();
    while (it != matches.end()) {

        // Transform the match to feature
        std::shared_ptr<StreetlightFeature> feat = std::make_shared<StreetlightFeature>();
        feat->featid = it->st_id;
        feat->boxes.push_back(it->rect);
        feat->uvs.push_back(it->rect_center);
        cv::Point2f npt = cam->undistort_cv(cv::Point2f(it->rect_center.x(), it->rect_center.y()));
        feat->uvs_norm.push_back(Eigen::Vector2f(npt.x, npt.y));
        feat->timestamps.push_back(timestamp);

        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        Eigen::MatrixXd R;
        std::vector<std::shared_ptr<Type>> Hx_order;

        if (state->_kf == KFCLASS::MSCKF)
            UpdaterHelper::get_stfeature_jacobian_full_msckf(state, feat, H_x, res, Hx_order, _pcd, R);
        else {
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

        // Append our residual and move forward
        if (!feat->to_delete) {
            cur_feats.push_back(feat);
        }
        res_big.block(ct_meas, 0, res.rows(), 1) = res;

        ct_meas += res.rows();
        it++;
    }

    // Return if we don't have anything and resize our matrices
    if (ct_meas < 1) {
        return std::vector<STMatch>();
    }

    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    R_big.conservativeResize(ct_meas, ct_meas);

    // 3. Perform measurement compression
    // our noise is isotropic, so make it here after our compression
    UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    R_big.conservativeResize(res_big.rows(), res_big.rows());
    R_big = _options.map_options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

    if (Hx_big.rows() < 1) {
        return std::vector<STMatch>();
    }

    // 4. With all good features update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);

    // 5. Update our StreetlightFeatureDatabase
    for (size_t i = 0; i < cur_feats.size(); i++) {
        st_db->update_feature(cur_feats[i]);
    }

    return matches;

    // PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to create DL system (%d features)\n", (rT1 - rT0).total_microseconds() * 1e-6, matches.size());
    // PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to compress DL system (%d features)\n", (rT2 - rT1).total_microseconds() * 1e-6, matches.size());
    // PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to update DL system\n", (rT3 - rT2).total_microseconds() * 1e-6);
    // PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to update DL features\n", (rT4 - rT3).total_microseconds() * 1e-6);
}

void UpdaterMAP::update_for_bi_features(std::shared_ptr<State> state, double timestamp) {

    // First update using deep learning-based detections
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4;
    rT0 = boost::posix_time::microsec_clock::local_time();

    // 2. Calculate the max possible measurement size and max possible state size
    size_t max_meas_size = 2 * cur_bi_matches.size();
    size_t max_hx_size = state->max_covariance_size();
    for (auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }

    vector<std::shared_ptr<StreetlightFeature>> cur_feats;
    // Large Jacobian and residual of all features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Zero(max_meas_size, max_meas_size);
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
    std::vector<std::shared_ptr<Type>> Hx_order_big;

    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    auto it = cur_bi_matches.begin();
    while (it != cur_bi_matches.end()) {

        // Transform the match to feature
        std::shared_ptr<StreetlightFeature> feat = std::make_shared<StreetlightFeature>();
        feat->featid = it->st_id;
        feat->boxes.push_back(it->rect);
        feat->uvs.push_back(it->rect_center);
        cv::Point2f npt = cam->undistort_cv(cv::Point2f(it->rect_center.x(), it->rect_center.y()));
        feat->uvs_norm.push_back(Eigen::Vector2f(npt.x, npt.y));
        feat->timestamps.push_back(timestamp);

        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        Eigen::MatrixXd R;
        std::vector<std::shared_ptr<Type>> Hx_order;

        if (state->_kf == KFCLASS::MSCKF)
            UpdaterHelper::get_stfeature_jacobian_full_msckf(state, feat, H_x, res, Hx_order, _pcd, R);
        else {
            UpdaterHelper::get_stfeature_jacobian_full(state, feat, H_x, res, Hx_order, _pcd, R);
        }

        Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
        Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
        S.diagonal() += _options.map_options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
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
        if (chi2 > _options.map_options.chi2_multipler_bi * chi2_check) {
            feat->to_delete = true;
            it = cur_bi_matches.erase(it);
            PRINT_DEBUG("featid = %d\n", feat->featid);
            PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options.map_options.chi2_multipler_bi * chi2_check);
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

        // Append our residual and move forward
        if (!feat->to_delete) {
            cur_feats.push_back(feat);
        }
        res_big.block(ct_meas, 0, res.rows(), 1) = res;
        ct_meas += res.rows();
        it++;
    }
    rT1 = boost::posix_time::microsec_clock::local_time();

    // Return if we don't have anything and resize our matrices
    if (ct_meas < 1) {
        return;
    }

    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    R_big.conservativeResize(ct_meas, ct_meas);

    // 3. Perform measurement compression
    // our noise is isotropic, so make it here after our compression
    UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    R_big.conservativeResize(res_big.rows(), res_big.rows());
    R_big = _options.map_options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());
    if (Hx_big.rows() < 1) {
        return;
    }
    rT2 = boost::posix_time::microsec_clock::local_time();

    // Our noise is isotropic, so make it here after our compression
    // Eigen::MatrixXd R_big = _options.map_options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

    // 4. With all good features update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);

    rT3 = boost::posix_time::microsec_clock::local_time();

    // 5. Update our StreetlightFeatureDatabase
    for (size_t i = 0; i < cur_feats.size(); i++) {
        st_database->update_feature(cur_feats[i]);
    }
    rT4 = boost::posix_time::microsec_clock::local_time();
    if (!cur_bi_matches.empty())
        no_matches = 0;

    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to create BI system (%d features)\n", (rT1 - rT0).total_microseconds() * 1e-6, cur_bi_matches.size());
    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to compress BI system (%d features)\n", (rT2 - rT1).total_microseconds() * 1e-6, cur_bi_matches.size());
    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to update BI system\n", (rT3 - rT2).total_microseconds() * 1e-6);
    PRINT_ALL("[PRIOR_MAP-UP]: %.4f seconds to update BI features\n", (rT4 - rT3).total_microseconds() * 1e-6);
}

void UpdaterMAP::updatepcd() {

    _pcd->matched_streetlight_cloud->clear();

    for (const auto match : cur_dl_matches) {
        pcl::PointXYZRGB pt;
        pt.x = match.st_center_map.x(), pt.y = match.st_center_map.y(), pt.z = match.st_center_map.z();
        pt.r = 255, pt.g = 0, pt.b = 0;
        _pcd->matched_streetlight_cloud->push_back(pt);
    }

    for (const auto match : cur_bi_matches) {
        pcl::PointXYZRGB pt;
        pt.x = match.st_center_map.x(), pt.y = match.st_center_map.y(), pt.z = match.st_center_map.z();
        pt.r = 255, pt.g = 0, pt.b = 0;
        _pcd->matched_streetlight_cloud->push_back(pt);
    }

    if (cur_dl_matches.empty() && cur_bi_matches.empty() && cur_dl_matches_box.empty()) {
        ++no_matches;
    }
}

void UpdaterMAP::updatepcd_tracking_recover(const std::vector<STMatch> &matches) {

    _pcd->matched_streetlight_cloud->clear();

    for (const auto match : matches) {
        pcl::PointXYZRGB pt;
        pt.x = match.st_center_map.x(), pt.y = match.st_center_map.y(), pt.z = match.st_center_map.z();
        pt.r = 255, pt.g = 0, pt.b = 0;
        _pcd->matched_streetlight_cloud->push_back(pt);
    }
}

void UpdaterMAP::display_streetlights(std::shared_ptr<State> state, cv::Mat &img_out1, cv::Mat &img_out2, string overlay1, string overlay2) {

    // cv::cvtColor(img_out1, img_out1, cv::COLOR_GRAY2RGB);
    // cv::cvtColor(img_out2, img_out2, cv::COLOR_GRAY2RGB);

    dl_boxes = st_matcher->get_dl_boxes();
    bi_boxes = st_matcher->get_bi_boxes();

    // If the image is "small" thus we shoudl use smaller display codes
    bool is_small = (std::min(img_out1.cols, img_out1.rows) < 640);

    Eigen::Matrix3d R_MAPtoC = state->_calib_IMUtoCAM->Rot() * state->_imu->Rot().transpose() * state->_pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinC =
        state->_calib_IMUtoCAM->Rot() * (state->_imu->Rot().transpose() * (state->_pose_MAPtoLOC->pos() - state->_imu->pos())) + state->_calib_IMUtoCAM->pos();

    // draw, loop through all close streetlights
    for (size_t i = 0; i < _pcd->center_points.size(); ++i) {
        Eigen::Vector3d center_point = _pcd->center_points[i];
        Eigen::Vector3d center_Pc = R_MAPtoC * center_point + p_MAPinC;

        if (center_Pc.z() < _options.match_options.update_z_th && center_Pc.z() > 0.1) {
            double tl_x = 5000, tl_y = 5000;
            for (const auto point : _pcd->points[i]) {
                Eigen::Vector3d Pc = R_MAPtoC * point + p_MAPinC;
                double inv_z = 1.0 / Pc.z();
                Eigen::Vector3d pt = inv_z * cam->get_K_eigen() * Pc;
                tl_x = min(tl_x, pt.x());
                tl_y = min(tl_y, pt.y());
            }

            cv::Scalar color = random_colors[i];
            float inv_z = 1.0 / center_Pc.z();
            Eigen::Vector3d center_pt = inv_z * cam->get_K_eigen() * center_Pc;
            cv::circle(img_out2, cv::Point2f(center_pt.x(), center_pt.y()), (is_small) ? 3 : 6, color, -1);

            cv::putText(img_out2, std::to_string(i), cv::Point2i(int(tl_x), int(tl_y)), cv::FONT_HERSHEY_TRIPLEX, (is_small) ? 0.6 : 1.2, color, 3);
        }
    }

    // draw, loop through all current matches
    for (const auto match : cur_dl_matches) {
        Eigen::Vector2f pt = match.rect_center;
        cv::Rect rect = match.rect;
        cv::circle(img_out1, cv::Point2f(pt.x(), pt.y()), (is_small) ? 3 : 6, cv::Scalar(0, 0, 255), -1);
        cv::rectangle(img_out1, rect, cv::Scalar(200, 55, 0), (is_small) ? 2 : 4);
        cv::putText(img_out1, std::to_string(match.st_id), rect.tl(), cv::FONT_HERSHEY_TRIPLEX, (is_small) ? 0.6 : 1.2, cv::Scalar(200, 55, 0), 3); // (0, 230, 255)
    }

    for (const auto match : cur_bi_matches) {
        Eigen::Vector2f pt = match.rect_center;
        cv::Rect rect = match.rect;
        cv::circle(img_out1, cv::Point2f(pt.x(), pt.y()), 3, cv::Scalar(0, 0, 255), -1);
        cv::rectangle(img_out1, rect, cv::Scalar(0, 200, 55), (is_small) ? 2 : 4);
        cv::putText(img_out1, std::to_string(match.st_id), rect.tl(), cv::FONT_HERSHEY_TRIPLEX, (is_small) ? 0.6 : 1.2, cv::Scalar(0, 200, 55), 3); // (255, 230, 0)
    }

    // Draw what camera this is
    auto txtpt = (is_small) ? cv::Point(10, 30) : cv::Point(30, 60);
    cv::putText(img_out1, overlay1, txtpt, cv::FONT_HERSHEY_TRIPLEX, (is_small) ? 0.75 : 1.5, cv::Scalar(0, 255, 0), 3);
    cv::putText(img_out2, overlay2, txtpt, cv::FONT_HERSHEY_TRIPLEX, (is_small) ? 0.75 : 1.5, cv::Scalar(0, 255, 0), 3);
}

void UpdaterMAP::display_streetlights_detection(cv::Mat &img_out, string overlay) {

    // If the image is "small" thus we shoudl use smaller display codes
    bool is_small = (std::min(img_out.cols, img_out.rows) < 640);

    for (const auto &rect : st_matcher->get_dl_boxes()) {
        cv::rectangle(img_out, rect, cv::Scalar(255, 50, 50), (is_small) ? 3 : 6);
    }

    for (const auto &rect : st_matcher->get_bi_boxes()) {
        bool find_overlap = false;
        for (const auto &rect_dl : st_matcher->get_dl_boxes()) {
            cv::Rect intersection = rect_dl & rect;
            if (intersection.area() > 0.6 * rect_dl.area()) {
                find_overlap = true;
                break;
            }
        }
        if (find_overlap)
            continue;
        cv::rectangle(img_out, rect, cv::Scalar(0, 255, 0), (is_small) ? 3 : 6);
    }
    // Draw what camera this is
    auto txtpt = (is_small) ? cv::Point(10, 30) : cv::Point(30, 60);
    cv::putText(img_out, overlay, txtpt, cv::FONT_HERSHEY_TRIPLEX, (is_small) ? 0.75 : 1.5, cv::Scalar(0, 255, 0), 3);
}

void UpdaterMAP::display_streetlights_tracking_recover(std::shared_ptr<State> state, cv::Mat &img_out1, cv::Mat &img_out2, string overlay) {

    // cv::cvtColor(img_out1, img_out1, cv::COLOR_GRAY2RGB);
    // cv::cvtColor(img_out2, img_out2, cv::COLOR_GRAY2RGB);

    // If the image is "small" thus we shoudl use smaller display codes
    bool is_small = (std::min(img_out1.cols, img_out1.rows) < 640);

    Eigen::Matrix3d R_MAPtoC = state->_calib_IMUtoCAM->Rot() * state->_imu->Rot().transpose() * state->_pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinC =
        state->_calib_IMUtoCAM->Rot() * (state->_imu->Rot().transpose() * (state->_pose_MAPtoLOC->pos() - state->_imu->pos())) + state->_calib_IMUtoCAM->pos();

    // draw, loop through all close streetlights
    for (size_t i = 0; i < _pcd->center_points.size(); ++i) {
        Eigen::Vector3d center_point = _pcd->center_points[i];
        Eigen::Vector3d center_Pc = R_MAPtoC * center_point + p_MAPinC;

        if (center_Pc.z() < _options.tracking_recover_options.search_dist_scope && center_Pc.z() > 0.1) {
            double tl_x = 5000, tl_y = 5000;
            for (const auto point : _pcd->points[i]) {
                Eigen::Vector3d Pc = R_MAPtoC * point + p_MAPinC;
                double inv_z = 1.0 / Pc.z();
                Eigen::Vector3d pt = inv_z * cam->get_K_eigen() * Pc;
                // cv::circle(img_out2, cv::Point2f(pt.x(), pt.y()), (is_small) ? 1 : 3, cv::Scalar(0, 0, 255), -1);
                tl_x = min(tl_x, pt.x());
                tl_y = min(tl_y, pt.y());
            }
            cv::Scalar color = random_colors[i];
            float inv_z = 1.0 / center_Pc.z();
            Eigen::Vector3d center_pt = inv_z * cam->get_K_eigen() * center_Pc;
            cv::circle(img_out2, cv::Point2f(center_pt.x(), center_pt.y()), (is_small) ? 3 : 6, color, -1);

            cv::putText(img_out2, std::to_string(i), cv::Point2i(int(tl_x), int(tl_y)), cv::FONT_HERSHEY_TRIPLEX, (is_small) ? 0.6 : 1.2, color, 3);
            // cv::Rect rect(0, int(center_pt.y()) - _options.tracking_recover_options.search_bar_height / 2, cam->w(),
            // _options.tracking_recover_options.search_bar_height); cv::rectangle(img_out1, rect, cv::Scalar(150, 150, 150, 2));
        }
    }

    // draw, loop through all current matches
    for (const auto &match : cur_dl_matches) {
        Eigen::Vector2f pt = match.rect_center;
        cv::Rect rect = match.rect;
        cv::circle(img_out1, cv::Point2f(pt.x(), pt.y()), (is_small) ? 3 : 6, cv::Scalar(0, 0, 255), -1);
        cv::rectangle(img_out1, rect, cv::Scalar(255, 0, 255), (is_small) ? 2 : 4);
        // cv::putText(img_out1, std::to_string(match.st_id), rect.tl(),
        //             cv::FONT_HERSHEY_COMPLEX_SMALL, (is_small) ? 1.0 : 1.5, cv::Scalar(0, 230, 255), 3);
    }

    // for (const auto& box : cur_box.rects){
    //     cv::rectangle(img_out1, box, cv::Scalar(0, 255, 0), 2);
    //     cv::Rect aug_box(box.x - 10, box.y - 10, box.width + 20, box.height + 20);
    //     cv::rectangle(img_out1, box, cv::Scalar(0, 200, 0), 2);
    // }

    // Draw what camera this is
    auto txtpt = (is_small) ? cv::Point(10, 30) : cv::Point(30, 60);
    cv::putText(img_out1, overlay, txtpt, cv::FONT_HERSHEY_TRIPLEX, (is_small) ? 0.75 : 1.5, cv::Scalar(0, 255, 0), 3);
}

void UpdaterMAP::display_other_streetlights_tracking_recover(std::vector<std::shared_ptr<State>> &states, cv::Mat &img_out) {

    // cv::cvtColor(img_out, img_out, cv::COLOR_GRAY2RGB);

    // If the image is "small" thus we shoudl use smaller display codes
    bool is_small = (std::min(img_out.cols, img_out.rows) < 640);

    for (size_t i = 1; i < states.size(); ++i) {
        Eigen::Matrix3d R_MAPtoC = states[i]->_calib_IMUtoCAM->Rot() * states[i]->_imu->Rot().transpose() * states[i]->_pose_MAPtoLOC->Rot();
        Eigen::Vector3d p_MAPinC = states[i]->_calib_IMUtoCAM->Rot() * (states[i]->_imu->Rot().transpose() * (states[i]->_pose_MAPtoLOC->pos() - states[i]->_imu->pos())) +
                                   states[i]->_calib_IMUtoCAM->pos();
        cv::Mat img_clone = img_out.clone();

        for (size_t j = 0; j < _pcd->center_points.size(); ++j) {
            Eigen::Vector3d center_point = _pcd->center_points[j];
            Eigen::Vector3d center_Pc = R_MAPtoC * center_point + p_MAPinC;

            if (center_Pc.z() < _options.tracking_recover_options.search_dist_scope && center_Pc.z() > 0.1) {
                double tl_x = 5000, tl_y = 5000;
                for (const auto point : _pcd->points[j]) {
                    Eigen::Vector3d Pc = R_MAPtoC * point + p_MAPinC;
                    double inv_z = 1.0 / Pc.z();
                    Eigen::Vector3d pt = inv_z * cam->get_K_eigen() * Pc;
                    // cv::circle(img_out2, cv::Point2f(pt.x(), pt.y()), (is_small) ? 1 : 3, cv::Scalar(0, 0, 255), -1);
                    tl_x = min(tl_x, pt.x());
                    tl_y = min(tl_y, pt.y());
                }
                cv::Scalar color = random_colors[j];
                float inv_z = 1.0 / center_Pc.z();
                Eigen::Vector3d center_pt = inv_z * cam->get_K_eigen() * center_Pc;
                cv::circle(img_clone, cv::Point2f(center_pt.x(), center_pt.y()), (is_small) ? 3 : 6, color, -1);

                cv::putText(img_clone, std::to_string(j), cv::Point2i(int(tl_x), int(tl_y)), cv::FONT_HERSHEY_TRIPLEX, (is_small) ? 0.6 : 1.2, color, 3);
                // cv::Rect rect(0, int(center_pt.y()) - _options.tracking_recover_options.search_bar_height / 2, cam->w(),
                // _options.tracking_recover_options.search_bar_height); cv::rectangle(img_out1, rect, cv::Scalar(150, 150, 150, 2));
            }
        }

        if (i > 1) {
            cv::hconcat(img_out, img_clone, img_out);
        } else {
            img_out = img_clone;
        }
    }
}

} // namespace night_voyager
