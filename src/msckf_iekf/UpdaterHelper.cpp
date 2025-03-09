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
#include "msckf_iekf/UpdaterHelper.h"
#include "msckf_iekf/State.h"
#include "streetlight_matcher/PcdManager.h"
#include "streetlight_matcher/StreetlightFeature.h"
#include "utils/Transform.h"

namespace night_voyager {
void UpdaterHelper::get_odom_jacobian_full(std::shared_ptr<State> state, const OdomData &message, Eigen::VectorXd &res, Eigen::MatrixXd &H_x,
                                           std::vector<std::shared_ptr<Type>> &x_order, Eigen::MatrixXd &R) {

    x_order.push_back(state->_imu->v());

    int m_size = 3, h_size = 3;
    H_x = Eigen::MatrixXd::Zero(m_size, h_size);
    res = Eigen::VectorXd::Zero(m_size);

    Eigen::Vector3d measured_vel = message.vm;
    Eigen::Matrix3d R_OtoI = state->_calib_IMUtoOdom->Rot().transpose();
    // Eigen::Matrix3d R_ItoO = state->_calib_IMUtoOdom->Rot();
    Eigen::Matrix3d R_IitoG = state->_imu->Rot();

    R = R_IitoG * R_OtoI;
    measured_vel = R * measured_vel;

    res = measured_vel - state->_imu->vel();
    H_x = Eigen::Matrix3d::Identity();

    // x_order.push_back(state->_imu);

    // int m_size = 3, h_size = state->_imu->size();
    // H_x = Eigen::MatrixXd::Zero(m_size, h_size);
    // res = Eigen::VectorXd::Zero(m_size);

    // Eigen::Vector3d measured_vel = message.vm;
    // Eigen::Matrix3d R_OtoI = state->_calib_IMUtoOdom->Rot().transpose();
    // // Eigen::Matrix3d R_ItoO = state->_calib_IMUtoOdom->Rot();
    // Eigen::Matrix3d R_IitoG = state->_imu->Rot();

    // res = measured_vel - state->_calib_IMUtoOdom->Rot() * R_IitoG.transpose() * state->_imu->vel();
    // H_x.block(0, 0, 3, state->_imu->size()) = Eigen::MatrixXd::Zero(3, state->_imu->size());
    // // H_x.block(0, state->_imu->v()->id() - state->_imu->id(), 3, state->_imu->v()->size()) = state->_calib_IMUtoOdom->Rot() *
    // R_IitoG.transpose(); H_x.block(0, state->_imu->v()->id() - state->_imu->id(), 3, state->_imu->v()->size()) = Eigen::Matrix3d::Identity();
}

void UpdaterHelper::get_odom_jacobian_full_msckf(std::shared_ptr<State> state, const OdomData &message, Eigen::VectorXd &res, Eigen::MatrixXd &H_x,
                                                 std::vector<std::shared_ptr<Type>> &x_order, Eigen::MatrixXd &R) {

    // x_order.push_back(state->_imu->v());

    // int m_size = 3, h_size = 3;
    // H_x = Eigen::MatrixXd::Zero(m_size, h_size);
    // res = Eigen::VectorXd::Zero(m_size);

    // Eigen::Vector3d measured_vel = message.vm;
    // Eigen::Matrix3d R_OtoI = state->_calib_IMUtoOdom->Rot().transpose();
    // Eigen::Matrix3d R_IitoG = state->_imu->Rot();

    // R = R_IitoG * R_OtoI;
    // measured_vel = R * measured_vel;

    // res = measured_vel - state->_imu->vel();
    // H_x = Eigen::Matrix3d::Identity();

    x_order.push_back(state->_imu);

    int m_size = 3, h_size = state->_imu->size();
    H_x = Eigen::MatrixXd::Zero(m_size, h_size);
    res = Eigen::VectorXd::Zero(m_size);

    Eigen::Vector3d measured_vel = message.vm;
    Eigen::Matrix3d R_IitoG = state->_imu->Rot();

    res = measured_vel - state->_calib_IMUtoOdom->Rot() * R_IitoG.transpose() * state->_imu->vel();
    H_x.block(0, 0, 3, state->_imu->R()->size()) = state->_calib_IMUtoOdom->Rot() * skew(R_IitoG.transpose() * state->_imu->vel());
    H_x.block(0, state->_imu->v()->id() - state->_imu->id(), 3, state->_imu->v()->size()) = state->_calib_IMUtoOdom->Rot() * R_IitoG.transpose();
}

void UpdaterHelper::get_feature_jacobian_full(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x,
                                              Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order) {
    // Total number of measurements for this feature
    int total_meas = feature.timestamps.size();

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx; // cloned pose, idx in clones_IMU

    for (size_t m = 0; m < feature.timestamps.size(); m++) {
        // Add this clone if it is not added already
        std::shared_ptr<PoseHamilton> clone_Ci = state->_clones_IMU.at(feature.timestamps.at(m));
        if (map_hx.find(clone_Ci) == map_hx.end()) {
            map_hx.insert({clone_Ci, total_hx});
            x_order.push_back(clone_Ci);
            total_hx += clone_Ci->size();
        }
    }

    if (feature.pseudo_anchor_clone_timestamp == -1.0) {
        feature.pseudo_anchor_clone_timestamp = state->_timestamp;
    }
    std::shared_ptr<PoseHamilton> clone_PSAi = state->_clones_IMU.at(feature.pseudo_anchor_clone_timestamp);
    if (map_hx.find(clone_PSAi) == map_hx.end()) {
        map_hx.insert({clone_PSAi, total_hx});
        x_order.push_back(clone_PSAi);
        total_hx += clone_PSAi->size();
    }

    // Calculate the position of this feature in the global frame
    // If anchored, then we need to calculate the position of the feature in the global
    Eigen::Vector3d p_FinG = feature.p_FinG;

    int c = 0;
    int jacobsize = 3;
    res = Eigen::VectorXd::Zero(2 * total_meas);
    H_f = Eigen::MatrixXd::Zero(2 * total_meas, jacobsize);
    H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);

    //=========================================================================
    //=========================================================================
    std::shared_ptr<PoseHamilton> calibration = state->_calib_IMUtoCAM;
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();

    for (size_t m = 0; m < feature.timestamps.size(); m++) {
        //=========================================================================
        //=========================================================================

        // Get current IMU clone state
        std::shared_ptr<PoseHamilton> clone_Ii = state->_clones_IMU.at(feature.timestamps.at(m));
        Eigen::Matrix3d R_GtoIi = clone_Ii->Rot().transpose();
        Eigen::Vector3d p_IiinG = clone_Ii->pos();

        // Get current feature in the IMU
        Eigen::Vector3d p_FinIi = R_GtoIi * (p_FinG - p_IiinG);

        // Project the current feature into the current frame of reference
        Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
        Eigen::Vector2d uv_norm;
        uv_norm << p_FinCi(0) / p_FinCi(2), p_FinCi(1) / p_FinCi(2);

        // Distort the normalized coordinates (radtan or fisheye)
        Eigen::Vector2d uv_dist;
        uv_dist = state->_cam_intrinsics_camera->distort_d(uv_norm);

        // Our residual
        Eigen::Vector2d uv_m;
        uv_m << (double)feature.uvs.at(m)(0), (double)feature.uvs.at(m)(1);
        res.block(2 * c, 0, 2, 1) = uv_m - uv_dist;

        //=========================================================================
        //=========================================================================

        // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
        Eigen::MatrixXd dz_dzn, dz_dzeta;
        state->_cam_intrinsics_camera->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

        // Normalized coordinates in respect to projection function
        Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
        dzn_dpfc << 1 / p_FinCi(2), 0, -p_FinCi(0) / (p_FinCi(2) * p_FinCi(2)), 0, 1 / p_FinCi(2), -p_FinCi(1) / (p_FinCi(2) * p_FinCi(2));

        if (feature.timestamps.at(m) == feature.pseudo_anchor_clone_timestamp) {

            Eigen::Matrix3d R_AItoG = clone_PSAi->Rot();

            Eigen::Matrix<double, 3, 3> R_GtoAC = R_ItoC * R_AItoG.transpose();

            // compute d(feature_camera)/dx
            // 1. d(feature_camera)/d(pose_AI)
            Eigen::Matrix<double, 3, 6> dpfc_dAI = Eigen::Matrix<double, 3, 6>::Zero();
            dpfc_dAI.block(0, 3, 3, 3) = -R_GtoAC;

            // 2. d(feature_camera)/d(feature_global)
            Eigen::Matrix<double, 3, 3> dpfc_dpfg = Eigen::Matrix<double, 3, 3>::Identity();
            dpfc_dpfg = R_GtoAC;

            // Precompute some matrices
            Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
            Eigen::Matrix<double, 2, 3> dz_dpfg = dz_dpfc * dpfc_dpfg;

            // CHAINRULE: get the total feature Jacobian
            H_f.block(2 * c, 0, 2, H_f.cols()).noalias() = dz_dpfg;
            // CHAINRULE: get state clone Jacobian
            H_x.block(2 * c, map_hx[clone_PSAi], 2, clone_PSAi->size()).noalias() = dz_dpfc * dpfc_dAI;
            c++;
        } else {
            // if not the anchor_clone_pose, then the jacobian would be related with
            // current clone pose and anchor clone pose

            Eigen::Matrix<double, 3, 3> R_IitoG = clone_Ii->Rot();

            Eigen::Matrix<double, 3, 3> R_GtoIC = R_ItoC * R_IitoG.transpose();

            // compute d(feature_camera)/dx

            // 1. d(feature_camera)/d(pose_I)
            Eigen::Matrix<double, 3, 6> dpfc_dI = Eigen::Matrix<double, 3, 6>::Zero();
            dpfc_dI.block(0, 0, 3, 3) = R_GtoIC * skew(p_FinG);
            dpfc_dI.block(0, 3, 3, 3) = -R_GtoIC;

            // 2. d(feature_camera)/d(pose_A)
            Eigen::Matrix<double, 3, 6> dpfc_dAI = Eigen::Matrix<double, 3, 6>::Zero();
            dpfc_dAI.block(0, 0, 3, 3) = -R_GtoIC * skew(p_FinG);

            // 3. d(feature_camera)/d(feature_global)
            Eigen::Matrix<double, 3, 3> dpfc_dpfg = Eigen::Matrix<double, 3, 3>::Identity();
            dpfc_dpfg = R_GtoIC;

            // Precompute some matrices
            Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
            Eigen::Matrix<double, 2, 3> dz_dpfg = dz_dpfc * dpfc_dpfg;

            // CHAINRULE: get the total feature Jacobian
            H_f.block(2 * c, 0, 2, H_f.cols()).noalias() = dz_dpfg;
            // CHAINRULE: get state clone Jacobian
            H_x.block(2 * c, map_hx[clone_PSAi], 2, clone_PSAi->size()).noalias() = dz_dpfc * dpfc_dAI;
            H_x.block(2 * c, map_hx[clone_Ii], 2, clone_Ii->size()).noalias() = dz_dpfc * dpfc_dI;
            c++;
        }
    }
}

void UpdaterHelper::get_feature_jacobian_full_clone_group(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                          Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order) {
    // Total number of measurements for this feature
    int total_meas = feature.timestamps.size();

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx; // cloned pose, idx in clones_IMU

    for (size_t m = 0; m < feature.timestamps.size(); m++) {
        // Add this clone if it is not added already
        std::shared_ptr<PoseHamilton> clone_Ci = state->_clones_IMU.at(feature.timestamps.at(m));
        if (map_hx.find(clone_Ci) == map_hx.end()) {
            map_hx.insert({clone_Ci, total_hx});
            x_order.push_back(clone_Ci);
            total_hx += clone_Ci->size();
        }
    }

    if (feature.pseudo_anchor_clone_timestamp == -1.0) {
        feature.pseudo_anchor_clone_timestamp = state->_timestamp;
    }
    std::shared_ptr<PoseHamilton> clone_PSAi = state->_clones_IMU.at(feature.pseudo_anchor_clone_timestamp);
    if (map_hx.find(clone_PSAi) == map_hx.end()) {
        map_hx.insert({clone_PSAi, total_hx});
        x_order.push_back(clone_PSAi);
        total_hx += clone_PSAi->size();
    }

    // Calculate the position of this feature in the global frame
    // If anchored, then we need to calculate the position of the feature in the global
    Eigen::Vector3d p_FinG = feature.p_FinG;

    int c = 0;
    int jacobsize = 3;
    res = Eigen::VectorXd::Zero(2 * total_meas);
    H_f = Eigen::MatrixXd::Zero(2 * total_meas, jacobsize);
    H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);

    //=========================================================================
    //=========================================================================

    std::shared_ptr<PoseHamilton> calibration = state->_calib_IMUtoCAM;
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();

    for (size_t m = 0; m < feature.timestamps.size(); m++) {
        //=========================================================================
        //=========================================================================

        // Get current IMU clone state
        std::shared_ptr<PoseHamilton> clone_Ii = state->_clones_IMU.at(feature.timestamps.at(m));
        Eigen::Matrix3d R_GtoIi = clone_Ii->Rot().transpose();
        Eigen::Vector3d p_IiinG = clone_Ii->pos();

        // Get current feature in the IMU
        Eigen::Vector3d p_FinIi = R_GtoIi * (p_FinG - p_IiinG);

        // Project the current feature into the current frame of reference
        Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
        Eigen::Vector2d uv_norm;
        uv_norm << p_FinCi(0) / p_FinCi(2), p_FinCi(1) / p_FinCi(2);

        // Distort the normalized coordinates (radtan or fisheye)
        Eigen::Vector2d uv_dist;
        uv_dist = state->_cam_intrinsics_camera->distort_d(uv_norm);

        // Our residual
        Eigen::Vector2d uv_m;
        uv_m << (double)feature.uvs.at(m)(0), (double)feature.uvs.at(m)(1);
        res.block(2 * c, 0, 2, 1) = uv_m - uv_dist;

        //=========================================================================
        //=========================================================================

        // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
        Eigen::MatrixXd dz_dzn, dz_dzeta;
        state->_cam_intrinsics_camera->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

        // Normalized coordinates in respect to projection function
        Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
        dzn_dpfc << 1 / p_FinCi(2), 0, -p_FinCi(0) / (p_FinCi(2) * p_FinCi(2)), 0, 1 / p_FinCi(2), -p_FinCi(1) / (p_FinCi(2) * p_FinCi(2));

        if (feature.timestamps.at(m) == feature.pseudo_anchor_clone_timestamp) {

            Eigen::Matrix3d R_AItoG = clone_PSAi->Rot();

            Eigen::Matrix<double, 3, 3> R_GtoAC = R_ItoC * R_AItoG.transpose();

            // compute d(feature_camera)/dx
            // 1. d(feature_camera)/d(pose_AI)
            Eigen::Matrix<double, 3, 6> dpfc_dAI = Eigen::Matrix<double, 3, 6>::Zero();
            dpfc_dAI.block(0, 3, 3, 3) = -R_GtoAC;

            // 2. d(feature_camera)/d(feature_global)
            Eigen::Matrix<double, 3, 3> dpfc_dpfg = Eigen::Matrix<double, 3, 3>::Identity();
            dpfc_dpfg = R_GtoAC;

            // Precompute some matrices
            Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
            Eigen::Matrix<double, 2, 3> dz_dpfg = dz_dpfc * dpfc_dpfg;

            // CHAINRULE: get the total feature Jacobian
            H_f.block(2 * c, 0, 2, H_f.cols()).noalias() = dz_dpfg;
            // CHAINRULE: get state clone Jacobian
            H_x.block(2 * c, map_hx[clone_PSAi], 2, clone_PSAi->size()).noalias() = dz_dpfc * dpfc_dAI;
            c++;
        } else {
            // if not the anchor_clone_pose, then the jacobian would be related with
            // current clone pose and anchor clone pose

            Eigen::Matrix<double, 3, 3> R_IitoG = clone_Ii->Rot();

            Eigen::Matrix<double, 3, 3> R_GtoIC = R_ItoC * R_IitoG.transpose();

            // compute d(feature_camera)/dx

            // 1. d(feature_camera)/d(pose_I)
            Eigen::Matrix<double, 3, 6> dpfc_dI = Eigen::Matrix<double, 3, 6>::Zero();
            dpfc_dI.block(0, 0, 3, 3) = R_GtoIC * skew(p_FinG);
            dpfc_dI.block(0, 3, 3, 3) = -R_GtoIC;

            // 2. d(feature_camera)/d(pose_A)
            Eigen::Matrix<double, 3, 6> dpfc_dAI = Eigen::Matrix<double, 3, 6>::Zero();
            dpfc_dAI.block(0, 0, 3, 3) = -R_GtoIC * skew(p_FinG);

            // 3. d(feature_camera)/d(feature_global)
            Eigen::Matrix<double, 3, 3> dpfc_dpfg = Eigen::Matrix<double, 3, 3>::Identity();
            dpfc_dpfg = R_GtoIC;

            // Precompute some matrices
            Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
            Eigen::Matrix<double, 2, 3> dz_dpfg = dz_dpfc * dpfc_dpfg;

            // CHAINRULE: get the total feature Jacobian
            H_f.block(2 * c, 0, 2, H_f.cols()).noalias() = dz_dpfg;
            // CHAINRULE: get state clone Jacobian
            H_x.block(2 * c, map_hx[clone_PSAi], 2, clone_PSAi->size()).noalias() = dz_dpfc * dpfc_dAI;
            H_x.block(2 * c, map_hx[clone_Ii], 2, clone_Ii->size()).noalias() = dz_dpfc * dpfc_dI;

            c++;
        }
    }
}

void UpdaterHelper::get_feature_jacobian_full_rel_group(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                        Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order) {
    // Total number of measurements for this feature
    int total_meas = feature.timestamps.size();

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx; // cloned pose, idx in clones_IMU

    for (size_t m = 0; m < feature.timestamps.size(); m++) {
        // Add this clone if it is not added already
        std::shared_ptr<PoseHamilton> clone_Ci = state->_clones_IMU.at(feature.timestamps.at(m));
        if (map_hx.find(clone_Ci) == map_hx.end()) {
            map_hx.insert({clone_Ci, total_hx});
            x_order.push_back(clone_Ci);
            total_hx += clone_Ci->size();
        }
    }

    std::shared_ptr<PoseHamilton> clone_A = state->_pose_MAPtoLOC;
    if (map_hx.find(clone_A) == map_hx.end()) {
        map_hx.insert({clone_A, total_hx});
        x_order.push_back(clone_A);
        total_hx += clone_A->size();
    }

    if (feature.pseudo_anchor_clone_timestamp == -1.0) {
        feature.pseudo_anchor_clone_timestamp = state->_timestamp;
    }

    // Calculate the position of this feature in the global frame
    // If anchored, then we need to calculate the position of the feature in the global
    Eigen::Vector3d p_FinG = feature.p_FinG;

    int c = 0;
    int jacobsize = 3;
    res = Eigen::VectorXd::Zero(2 * total_meas);
    H_f = Eigen::MatrixXd::Zero(2 * total_meas, jacobsize);
    H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);

    //=========================================================================
    //=========================================================================

    std::shared_ptr<PoseHamilton> calibration = state->_calib_IMUtoCAM;
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();

    for (size_t m = 0; m < feature.timestamps.size(); m++) {
        //=========================================================================
        //=========================================================================

        // Get current IMU clone state
        std::shared_ptr<PoseHamilton> clone_Ii = state->_clones_IMU.at(feature.timestamps.at(m));
        Eigen::Matrix3d R_GtoIi = clone_Ii->Rot().transpose();
        Eigen::Vector3d p_IiinG = clone_Ii->pos();

        // Get current feature in the IMU
        Eigen::Vector3d p_FinIi = R_GtoIi * (p_FinG - p_IiinG);

        // Project the current feature into the current frame of reference
        Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
        Eigen::Vector2d uv_norm;
        uv_norm << p_FinCi(0) / p_FinCi(2), p_FinCi(1) / p_FinCi(2);

        // Distort the normalized coordinates (radtan or fisheye)
        Eigen::Vector2d uv_dist;
        uv_dist = state->_cam_intrinsics_camera->distort_d(uv_norm);

        // Our residual
        Eigen::Vector2d uv_m;
        uv_m << (double)feature.uvs.at(m)(0), (double)feature.uvs.at(m)(1);
        res.block(2 * c, 0, 2, 1) = uv_m - uv_dist;

        //=========================================================================
        //=========================================================================

        // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
        Eigen::MatrixXd dz_dzn, dz_dzeta;
        state->_cam_intrinsics_camera->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

        // Normalized coordinates in respect to projection function
        Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
        dzn_dpfc << 1 / p_FinCi(2), 0, -p_FinCi(0) / (p_FinCi(2) * p_FinCi(2)), 0, 1 / p_FinCi(2), -p_FinCi(1) / (p_FinCi(2) * p_FinCi(2));

        // if not the anchor_clone_pose, then the jacobian would be related with
        // current clone pose and anchor clone pose

        Eigen::Matrix<double, 3, 3> R_IitoG = clone_Ii->Rot();

        Eigen::Matrix<double, 3, 3> R_GtoIC = R_ItoC * R_IitoG.transpose();

        // compute d(feature_camera)/dx

        // 1. d(feature_camera)/d(pose_I)
        Eigen::Matrix<double, 3, 6> dpfc_dI = Eigen::Matrix<double, 3, 6>::Zero();
        dpfc_dI.block(0, 0, 3, 3) = R_GtoIC * skew(p_FinG);
        dpfc_dI.block(0, 3, 3, 3) = -R_GtoIC;

        // 2. d(feature_camera)/d(pose_A)
        Eigen::Matrix<double, 3, 6> dpfc_dAI = Eigen::Matrix<double, 3, 6>::Zero();
        dpfc_dAI.block(0, 0, 3, 3) = -R_GtoIC * skew(p_FinG);

        // 3. d(feature_camera)/d(feature_global)
        Eigen::Matrix<double, 3, 3> dpfc_dpfg = Eigen::Matrix<double, 3, 3>::Identity();
        dpfc_dpfg = R_GtoIC;

        // Precompute some matrices
        Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
        Eigen::Matrix<double, 2, 3> dz_dpfg = dz_dpfc * dpfc_dpfg;

        // CHAINRULE: get the total feature Jacobian
        H_f.block(2 * c, 0, 2, H_f.cols()).noalias() = dz_dpfg;
        // CHAINRULE: get state clone Jacobian
        H_x.block(2 * c, map_hx[clone_A], 2, clone_A->size()).noalias() = dz_dpfc * dpfc_dAI;
        H_x.block(2 * c, map_hx[clone_Ii], 2, clone_Ii->size()).noalias() = dz_dpfc * dpfc_dI;

        c++;
    }
}

void UpdaterHelper::get_feature_jacobian_full_no_group(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                       Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order) {
    // Total number of measurements for this feature
    int total_meas = feature.timestamps.size();

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx; // cloned pose, idx in clones_IMU

    for (size_t m = 0; m < feature.timestamps.size(); m++) {
        // Add this clone if it is not added already
        std::shared_ptr<PoseHamilton> clone_Ci = state->_clones_IMU.at(feature.timestamps.at(m));
        if (map_hx.find(clone_Ci) == map_hx.end()) {
            map_hx.insert({clone_Ci, total_hx});
            x_order.push_back(clone_Ci);
            total_hx += clone_Ci->size();
        }
    }

    std::shared_ptr<PoseHamilton> clone_A = state->_pose_MAPtoLOC;
    if (map_hx.find(clone_A) == map_hx.end()) {
        map_hx.insert({clone_A, total_hx});
        x_order.push_back(clone_A);
        total_hx += clone_A->size();
    }

    if (feature.pseudo_anchor_clone_timestamp == -1.0) {
        feature.pseudo_anchor_clone_timestamp = state->_timestamp;
    }

    // Calculate the position of this feature in the global frame
    // If anchored, then we need to calculate the position of the feature in the global
    Eigen::Vector3d p_FinG = feature.p_FinG;

    int c = 0;
    int jacobsize = 3;
    res = Eigen::VectorXd::Zero(2 * total_meas);
    H_f = Eigen::MatrixXd::Zero(2 * total_meas, jacobsize);
    H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);

    //=========================================================================
    //=========================================================================

    std::shared_ptr<PoseHamilton> calibration = state->_calib_IMUtoCAM;
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();

    for (size_t m = 0; m < feature.timestamps.size(); m++) {
        //=========================================================================
        //=========================================================================

        // Get current IMU clone state
        std::shared_ptr<PoseHamilton> clone_Ii = state->_clones_IMU.at(feature.timestamps.at(m));
        Eigen::Matrix3d R_GtoIi = clone_Ii->Rot().transpose();
        Eigen::Vector3d p_IiinG = clone_Ii->pos();

        // Get current feature in the IMU
        Eigen::Vector3d p_FinIi = R_GtoIi * (p_FinG - p_IiinG);

        // Project the current feature into the current frame of reference
        Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
        Eigen::Vector2d uv_norm;
        uv_norm << p_FinCi(0) / p_FinCi(2), p_FinCi(1) / p_FinCi(2);

        // Distort the normalized coordinates (radtan or fisheye)
        Eigen::Vector2d uv_dist;
        uv_dist = state->_cam_intrinsics_camera->distort_d(uv_norm);

        // Our residual
        Eigen::Vector2d uv_m;
        uv_m << (double)feature.uvs.at(m)(0), (double)feature.uvs.at(m)(1);
        res.block(2 * c, 0, 2, 1) = uv_m - uv_dist;

        //=========================================================================
        //=========================================================================

        // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
        Eigen::MatrixXd dz_dzn, dz_dzeta;
        state->_cam_intrinsics_camera->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

        // Normalized coordinates in respect to projection function
        Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
        dzn_dpfc << 1 / p_FinCi(2), 0, -p_FinCi(0) / (p_FinCi(2) * p_FinCi(2)), 0, 1 / p_FinCi(2), -p_FinCi(1) / (p_FinCi(2) * p_FinCi(2));

        // if not the anchor_clone_pose, then the jacobian would be related with
        // current clone pose and anchor clone pose

        Eigen::Matrix<double, 3, 3> R_IitoG = clone_Ii->Rot();

        Eigen::Matrix<double, 3, 3> R_GtoIC = R_ItoC * R_IitoG.transpose();

        // compute d(feature_camera)/dx

        // 1. d(feature_camera)/d(pose_I)
        Eigen::Matrix<double, 3, 6> dpfc_dI = Eigen::Matrix<double, 3, 6>::Zero();
        dpfc_dI.block(0, 0, 3, 3) = R_GtoIC * skew(p_FinG);
        dpfc_dI.block(0, 3, 3, 3) = -R_GtoIC;

        // 2. d(feature_camera)/d(feature_global)
        Eigen::Matrix<double, 3, 3> dpfc_dpfg = Eigen::Matrix<double, 3, 3>::Identity();
        dpfc_dpfg = R_GtoIC;

        // Precompute some matrices
        Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
        Eigen::Matrix<double, 2, 3> dz_dpfg = dz_dpfc * dpfc_dpfg;

        // CHAINRULE: get the total feature Jacobian
        H_f.block(2 * c, 0, 2, H_f.cols()).noalias() = dz_dpfg;
        // CHAINRULE: get state clone Jacobian
        H_x.block(2 * c, map_hx[clone_Ii], 2, clone_Ii->size()).noalias() = dz_dpfc * dpfc_dI;

        c++;
    }
}

void UpdaterHelper::get_feature_jacobian_full_msckf(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                    Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order) {
    // Total number of measurements for this feature
    int total_meas = feature.timestamps.size();

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx; // cloned pose, idx in clones_IMU

    for (size_t m = 0; m < feature.timestamps.size(); m++) {
        // Add this clone if it is not added already
        std::shared_ptr<PoseHamilton> clone_Ci = state->_clones_IMU.at(feature.timestamps.at(m));
        if (map_hx.find(clone_Ci) == map_hx.end()) {
            map_hx.insert({clone_Ci, total_hx});
            x_order.push_back(clone_Ci);
            total_hx += clone_Ci->size();
        }
    }

    if (feature.pseudo_anchor_clone_timestamp == -1.0) {
        feature.pseudo_anchor_clone_timestamp = state->_timestamp;
    }
    std::shared_ptr<PoseHamilton> clone_PSAi = state->_clones_IMU.at(feature.pseudo_anchor_clone_timestamp);
    if (map_hx.find(clone_PSAi) == map_hx.end()) {
        map_hx.insert({clone_PSAi, total_hx});
        x_order.push_back(clone_PSAi);
        total_hx += clone_PSAi->size();
    }

    // Calculate the position of this feature in the global frame
    // If anchored, then we need to calculate the position of the feature in the global
    Eigen::Vector3d p_FinG = feature.p_FinG;

    int c = 0;
    int jacobsize = 3;
    res = Eigen::VectorXd::Zero(2 * total_meas);
    H_f = Eigen::MatrixXd::Zero(2 * total_meas, jacobsize);
    H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);

    //=========================================================================
    //=========================================================================

    std::shared_ptr<PoseHamilton> calibration = state->_calib_IMUtoCAM;
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();

    for (size_t m = 0; m < feature.timestamps.size(); m++) {
        //=========================================================================
        //=========================================================================

        // Get current IMU clone state
        std::shared_ptr<PoseHamilton> clone_Ii = state->_clones_IMU.at(feature.timestamps.at(m));
        Eigen::Matrix3d R_GtoIi = clone_Ii->Rot().transpose();
        Eigen::Vector3d p_IiinG = clone_Ii->pos();

        // Get current feature in the IMU
        Eigen::Vector3d p_FinIi = R_GtoIi * (p_FinG - p_IiinG);

        // Project the current feature into the current frame of reference
        Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
        Eigen::Vector2d uv_norm;
        uv_norm << p_FinCi(0) / p_FinCi(2), p_FinCi(1) / p_FinCi(2);

        // Distort the normalized coordinates (radtan or fisheye)
        Eigen::Vector2d uv_dist;
        uv_dist = state->_cam_intrinsics_camera->distort_d(uv_norm);

        // Our residual
        Eigen::Vector2d uv_m;
        uv_m << (double)feature.uvs.at(m)(0), (double)feature.uvs.at(m)(1);
        res.block(2 * c, 0, 2, 1) = uv_m - uv_dist;

        //=========================================================================
        //=========================================================================

        // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
        Eigen::MatrixXd dz_dzn, dz_dzeta;
        state->_cam_intrinsics_camera->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

        // Normalized coordinates in respect to projection function
        Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
        dzn_dpfc << 1 / p_FinCi(2), 0, -p_FinCi(0) / (p_FinCi(2) * p_FinCi(2)), 0, 1 / p_FinCi(2), -p_FinCi(1) / (p_FinCi(2) * p_FinCi(2));

        // Derivative of p_FinCi in respect to p_FinIi
        Eigen::MatrixXd dpfc_dpfg = R_ItoC * R_GtoIi;

        // Derivative of p_FinCi in respect to camera clone state
        Eigen::MatrixXd dpfc_dclone = Eigen::MatrixXd::Zero(3, 6);
        dpfc_dclone.block(0, 0, 3, 3).noalias() = R_ItoC * skew(p_FinIi);
        dpfc_dclone.block(0, 3, 3, 3) = -dpfc_dpfg;

        //=========================================================================
        //=========================================================================

        // Precompute some matrices
        Eigen::MatrixXd dz_dpfc = dz_dzn * dzn_dpfc;
        Eigen::MatrixXd dz_dpfg = dz_dpfc * dpfc_dpfg;

        // CHAINRULE: get the total feature Jacobian
        H_f.block(2 * c, 0, 2, H_f.cols()).noalias() = dz_dpfg;

        // CHAINRULE: get state clone Jacobian
        H_x.block(2 * c, map_hx[clone_Ii], 2, clone_Ii->size()).noalias() = dz_dpfc * dpfc_dclone;

        c++;
    }
}

void UpdaterHelper::get_stfeature_jacobian_full(std::shared_ptr<State> state, std::shared_ptr<StreetlightFeature> feature, Eigen::MatrixXd &H_x,
                                                Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order, std::shared_ptr<PcdManager> pcd,
                                                Eigen::MatrixXd &R) {
    // Total number of measurement for this feature
    int total_meas = feature->timestamps.size();

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx; // cloned pose, idx in clones_IMU

    for (size_t m = 0; m < feature->timestamps.size(); ++m) {
        // Add this clone if it is not added already
        std::shared_ptr<PoseHamilton> clone_Ci = state->_clones_IMU.at(feature->timestamps.at(m));
        if (map_hx.find(clone_Ci) == map_hx.end()) {
            map_hx.insert({clone_Ci, total_hx});
            x_order.push_back(clone_Ci);
            total_hx += clone_Ci->size();
        }
    }

    std::shared_ptr<PoseHamilton> pose_MAPtoLOC = state->_pose_MAPtoLOC;
    if (map_hx.find(pose_MAPtoLOC) == map_hx.end()) {
        map_hx.insert({pose_MAPtoLOC, total_hx});
        x_order.push_back(pose_MAPtoLOC);
        total_hx += pose_MAPtoLOC->size();
    }

    int c = 0;
    res = Eigen::VectorXd::Zero(2 * total_meas);
    H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);
    R = Eigen::MatrixXd::Zero(res.rows(), res.rows());

    std::shared_ptr<PoseHamilton> calibration = state->_calib_IMUtoCAM;
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();

    Eigen::Matrix3d R_MAPtoLOC = pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinLOC = pose_MAPtoLOC->pos();

    Eigen::Vector3d p_STFinMAP = pcd->center_points[feature->featid];

    // Get current streetlight feature in the LOC (G is LOC)
    Eigen::Vector3d p_STFinG = R_MAPtoLOC * p_STFinMAP + p_MAPinLOC;

    for (size_t m = 0; m < feature->timestamps.size(); m++) {
        //=========================================================================
        //=========================================================================

        // Get current IMU clone state
        std::shared_ptr<PoseHamilton> clone_Ii = state->_clones_IMU.at(feature->timestamps.at(m));
        Eigen::Matrix3d R_GtoIi = clone_Ii->Rot().transpose();
        Eigen::Vector3d p_IiinG = clone_Ii->pos();

        // Project the current feature into the current frame of reference
        Eigen::Matrix3d R_GtoC = R_ItoC * R_GtoIi;
        Eigen::Vector3d p_GinC = -R_ItoC * R_GtoIi * p_IiinG + p_IinC;

        Eigen::Vector3d p_STFinCi = R_GtoC * p_STFinG + p_GinC;
        Eigen::Vector2d uv_norm;
        uv_norm << p_STFinCi(0) / p_STFinCi(2), p_STFinCi(1) / p_STFinCi(2);

        // Distort the normalized coordinates (radtan or fisheye)
        Eigen::Vector2d uv_dist;
        uv_dist = state->_cam_intrinsics_camera->distort_d(uv_norm);

        // Our residual
        Eigen::Vector2d uv_m;
        uv_m << (double)feature->uvs.at(m)(0), (double)feature->uvs.at(m)(1);
        res.block(2 * c, 0, 2, 1) = uv_m - uv_dist;

        // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
        Eigen::MatrixXd dz_dzn, dz_dzeta;
        state->_cam_intrinsics_camera->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

        // Normalized coordinates in respect to projection function
        Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
        dzn_dpfc << 1 / p_STFinCi(2), 0, -p_STFinCi(0) / (p_STFinCi(2) * p_STFinCi(2)), 0, 1 / p_STFinCi(2),
            -p_STFinCi(1) / (p_STFinCi(2) * p_STFinCi(2));

        // compute d(feature_camera)/dx
        // 1. d(feature_camera)/d(pose_I)
        Eigen::Matrix<double, 3, 6> dpfc_dAI = Eigen::Matrix<double, 3, 6>::Zero();
        dpfc_dAI.block(0, 0, 3, 3) = R_GtoC * skew(p_STFinG);
        dpfc_dAI.block(0, 3, 3, 3) = -R_GtoC;

        // 2. d(feature_camera)/d(pose_MAPtoLOC)
        Eigen::Matrix<double, 3, 6> dpfc_dML = Eigen::Matrix<double, 3, 6>::Zero();
        dpfc_dML.block(0, 0, 3, 3) = -R_GtoC * skew(p_STFinG);
        dpfc_dML.block(0, 3, 3, 3) = R_GtoC;

        // CHAINRULE: get the total feature Jacobian
        Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
        H_x.block(2 * c, map_hx[clone_Ii], 2, clone_Ii->size()) = dz_dpfc * dpfc_dAI;
        H_x.block(2 * c, map_hx[pose_MAPtoLOC], 2, pose_MAPtoLOC->size()) = dz_dpfc * dpfc_dML;

        c++;
    }
}

void UpdaterHelper::get_stfeature_jacobian_full_group(std::shared_ptr<State> state, std::shared_ptr<StreetlightFeature> feature, Eigen::MatrixXd &H_x,
                                                      Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order,
                                                      std::shared_ptr<PcdManager> pcd, Eigen::MatrixXd &R) {
    // Total number of measurement for this feature
    int total_meas = feature->timestamps.size();

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx; // cloned pose, idx in clones_IMU

    std::shared_ptr<PoseHamilton> clone_A = state->_clones_IMU.at(state->_pseudo_map_anchor_time);
    for (size_t m = 0; m < feature->timestamps.size(); ++m) {
        // Add this clone if it is not added already
        std::shared_ptr<PoseHamilton> clone_Ci = state->_clones_IMU.at(feature->timestamps.at(m));
        if (map_hx.find(clone_Ci) == map_hx.end()) {
            map_hx.insert({clone_Ci, total_hx});
            x_order.push_back(clone_Ci);
            total_hx += clone_Ci->size();
        }
    }

    std::shared_ptr<PoseHamilton> pose_MAPtoLOC = state->_pose_MAPtoLOC;
    if (map_hx.find(pose_MAPtoLOC) == map_hx.end()) {
        map_hx.insert({pose_MAPtoLOC, total_hx});
        x_order.push_back(pose_MAPtoLOC);
        total_hx += pose_MAPtoLOC->size();
    }

    int c = 0;
    res = Eigen::VectorXd::Zero(2 * total_meas);
    H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);
    R = Eigen::MatrixXd::Zero(res.rows(), res.rows());

    std::shared_ptr<PoseHamilton> calibration = state->_calib_IMUtoCAM;
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();

    Eigen::Matrix3d R_MAPtoLOC = pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinLOC = pose_MAPtoLOC->pos();

    Eigen::Vector3d p_STFinMAP = pcd->center_points[feature->featid];

    // Get current streetlight feature in the LOC (G is LOC)
    Eigen::Vector3d tmp_STF = R_MAPtoLOC * p_STFinMAP;
    Eigen::Vector3d p_STFinG = tmp_STF + p_MAPinLOC;

    for (size_t m = 0; m < feature->timestamps.size(); m++) {
        //=========================================================================
        //=========================================================================

        // Get current IMU clone state
        std::shared_ptr<PoseHamilton> clone_Ii = state->_clones_IMU.at(feature->timestamps.at(m));
        Eigen::Matrix3d R_GtoIi = clone_Ii->Rot().transpose();
        Eigen::Vector3d p_IiinG = clone_Ii->pos();

        // Project the current feature into the current frame of reference
        Eigen::Matrix3d R_GtoC = R_ItoC * R_GtoIi;
        Eigen::Vector3d p_GinC = -R_ItoC * R_GtoIi * p_IiinG + p_IinC;

        Eigen::Vector3d p_STFinCi = R_GtoC * p_STFinG + p_GinC;
        Eigen::Vector2d uv_norm;
        uv_norm << p_STFinCi(0) / p_STFinCi(2), p_STFinCi(1) / p_STFinCi(2);

        // Distort the normalized coordinates (radtan or fisheye)
        Eigen::Vector2d uv_dist;
        uv_dist = state->_cam_intrinsics_camera->distort_d(uv_norm);

        // Our residual
        Eigen::Vector2d uv_m;
        uv_m << (double)feature->uvs.at(m)(0), (double)feature->uvs.at(m)(1);
        res.block(2 * c, 0, 2, 1) = uv_m - uv_dist;

        // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
        Eigen::MatrixXd dz_dzn, dz_dzeta;
        state->_cam_intrinsics_camera->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

        // Normalized coordinates in respect to projection function
        Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
        dzn_dpfc << 1 / p_STFinCi(2), 0, -p_STFinCi(0) / (p_STFinCi(2) * p_STFinCi(2)), 0, 1 / p_STFinCi(2),
            -p_STFinCi(1) / (p_STFinCi(2) * p_STFinCi(2));

        if (std::abs(state->_pseudo_map_anchor_time - feature->timestamps.at(m)) < 1e-7) {
            // compute d(feature_camera)/dx
            // 1. d(feature_camera)/d(pose_I)
            Eigen::Matrix<double, 3, 6> dpfc_dAI = Eigen::Matrix<double, 3, 6>::Zero();
            dpfc_dAI.block(0, 0, 3, 3) = R_GtoC * skew(tmp_STF);
            dpfc_dAI.block(0, 3, 3, 3) = -R_GtoC;

            // 2. d(feature_camera)/d(pose_MAPtoLOC)
            Eigen::Matrix<double, 3, 6> dpfc_dML = Eigen::Matrix<double, 3, 6>::Zero();
            dpfc_dML.block(0, 0, 3, 3) = -R_GtoC * skew(tmp_STF);
            dpfc_dML.block(0, 3, 3, 3) = R_GtoC;

            // CHAINRULE: get the total feature Jacobian
            Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
            H_x.block(2 * c, map_hx[clone_Ii], 2, clone_Ii->size()) = dz_dpfc * dpfc_dAI;
            H_x.block(2 * c, map_hx[pose_MAPtoLOC], 2, pose_MAPtoLOC->size()) = dz_dpfc * dpfc_dML;
        } else {
            // compute d(feature_camera)/dx
            // 1. d(feature_camera)/d(pose_I)
            Eigen::Matrix<double, 3, 6> dpfc_dAI = Eigen::Matrix<double, 3, 6>::Zero();
            dpfc_dAI.block(0, 0, 3, 3) = R_GtoC * skew(p_STFinG);
            dpfc_dAI.block(0, 3, 3, 3) = -R_GtoC;

            // 2. d(feature_camera)/d(pose_MAPtoLOC)
            Eigen::Matrix<double, 3, 6> dpfc_dML = Eigen::Matrix<double, 3, 6>::Zero();
            dpfc_dML.block(0, 0, 3, 3) = -R_GtoC * skew(tmp_STF);
            dpfc_dML.block(0, 3, 3, 3) = R_GtoC;

            // 3. d(feature_camera)/d(pose_A)
            Eigen::Matrix<double, 3, 6> dpfc_dA = Eigen::Matrix<double, 3, 6>::Zero();
            dpfc_dA.block(0, 0, 3, 3) = -R_GtoC * skew(p_MAPinLOC);

            // CHAINRULE: get the total feature Jacobian
            Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
            H_x.block(2 * c, map_hx[clone_Ii], 2, clone_Ii->size()) = dz_dpfc * dpfc_dAI;
            H_x.block(2 * c, map_hx[pose_MAPtoLOC], 2, pose_MAPtoLOC->size()) = dz_dpfc * dpfc_dML;
        }

        c++;
    }
}

void UpdaterHelper::get_stfeature_jacobian_full_msckf(std::shared_ptr<State> state, std::shared_ptr<StreetlightFeature> feature, Eigen::MatrixXd &H_x,
                                                      Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order,
                                                      std::shared_ptr<PcdManager> pcd, Eigen::MatrixXd &R) {
    // Total number of measurement for this feature
    int total_meas = feature->timestamps.size();

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx; // cloned pose, idx in clones_IMU

    for (size_t m = 0; m < feature->timestamps.size(); ++m) {
        // Add this clone if it is not added already
        std::shared_ptr<PoseHamilton> clone_Ci = state->_clones_IMU.at(feature->timestamps.at(m));
        if (map_hx.find(clone_Ci) == map_hx.end()) {
            map_hx.insert({clone_Ci, total_hx});
            x_order.push_back(clone_Ci);
            total_hx += clone_Ci->size();
        }
    }

    std::shared_ptr<PoseHamilton> pose_MAPtoLOC = state->_pose_MAPtoLOC;
    if (map_hx.find(pose_MAPtoLOC) == map_hx.end()) {
        map_hx.insert({pose_MAPtoLOC, total_hx});
        x_order.push_back(pose_MAPtoLOC);
        total_hx += pose_MAPtoLOC->size();
    }

    int c = 0;
    res = Eigen::VectorXd::Zero(2 * total_meas);
    H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);
    R = Eigen::MatrixXd::Zero(res.rows(), res.rows());

    std::shared_ptr<PoseHamilton> calibration = state->_calib_IMUtoCAM;
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();

    Eigen::Matrix3d R_MAPtoLOC = pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinLOC = pose_MAPtoLOC->pos();

    Eigen::Vector3d p_STFinMAP = pcd->center_points[feature->featid];

    // Get current streetlight feature in the LOC (G is LOC)
    Eigen::Vector3d p_STFinG = R_MAPtoLOC * p_STFinMAP + p_MAPinLOC;

    for (size_t m = 0; m < feature->timestamps.size(); m++) {
        //=========================================================================
        //=========================================================================

        // Get current IMU clone state
        std::shared_ptr<PoseHamilton> clone_Ii = state->_clones_IMU.at(feature->timestamps.at(m));
        Eigen::Matrix3d R_GtoIi = clone_Ii->Rot().transpose();
        Eigen::Vector3d p_IiinG = clone_Ii->pos();

        // Project the current feature into the current frame of reference
        Eigen::Matrix3d R_GtoC = R_ItoC * R_GtoIi;
        Eigen::Vector3d p_GinC = -R_ItoC * R_GtoIi * p_IiinG + p_IinC;

        Eigen::Vector3d p_STFinIi = R_GtoIi * (p_STFinG - p_IiinG);
        Eigen::Vector3d p_STFinCi = R_GtoC * p_STFinG + p_GinC;
        Eigen::Vector2d uv_norm;
        uv_norm << p_STFinCi(0) / p_STFinCi(2), p_STFinCi(1) / p_STFinCi(2);

        // Distort the normalized coordinates (radtan or fisheye)
        Eigen::Vector2d uv_dist;
        uv_dist = state->_cam_intrinsics_camera->distort_d(uv_norm);

        // Our residual
        Eigen::Vector2d uv_m;
        uv_m << (double)feature->uvs.at(m)(0), (double)feature->uvs.at(m)(1);
        res.block(2 * c, 0, 2, 1) = uv_m - uv_dist;

        // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
        Eigen::MatrixXd dz_dzn, dz_dzeta;
        state->_cam_intrinsics_camera->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

        // Normalized coordinates in respect to projection function
        Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
        dzn_dpfc << 1 / p_STFinCi(2), 0, -p_STFinCi(0) / (p_STFinCi(2) * p_STFinCi(2)), 0, 1 / p_STFinCi(2),
            -p_STFinCi(1) / (p_STFinCi(2) * p_STFinCi(2));

        // compute d(feature_camera)/dx
        // 1. d(feature_camera)/d(pose_I)
        Eigen::Matrix<double, 3, 6> dpfc_dAI = Eigen::Matrix<double, 3, 6>::Zero();
        dpfc_dAI.block(0, 0, 3, 3) = R_ItoC * skew(p_STFinIi);
        dpfc_dAI.block(0, 3, 3, 3) = -R_GtoC;

        // 2. d(feature_camera)/d(pose_MAPtoLOC)
        Eigen::Matrix<double, 3, 6> dpfc_dML = Eigen::Matrix<double, 3, 6>::Zero();
        dpfc_dML.block(0, 0, 3, 3) = -R_GtoC * R_MAPtoLOC * skew(p_STFinMAP);
        dpfc_dML.block(0, 3, 3, 3) = R_GtoC;

        // CHAINRULE: get the total feature Jacobian
        Eigen::Matrix<double, 2, 3> dz_dpfc = dz_dzn * dzn_dpfc;
        H_x.block(2 * c, map_hx[clone_Ii], 2, clone_Ii->size()) = dz_dpfc * dpfc_dAI;
        H_x.block(2 * c, map_hx[pose_MAPtoLOC], 2, pose_MAPtoLOC->size()) = dz_dpfc * dpfc_dML;

        c++;
    }
}

void UpdaterHelper::get_plane_jacobian_full(std::shared_ptr<PoseHamilton> pose1, std::shared_ptr<PoseHamilton> pose2, const Eigen::Matrix3d &Roi,
                                            Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order) {
    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;

    map_hx.insert({pose1, total_hx});
    x_order.push_back(pose1);
    total_hx += pose1->size();

    map_hx.insert({pose2, total_hx});
    x_order.push_back(pose2);
    total_hx += pose2->size();

    res = Eigen::VectorXd::Zero(2);
    H_x = Eigen::MatrixXd::Zero(2, total_hx);
    Eigen::RowVector3d eR1 = Eigen::Vector3d(0, 0, 1).transpose() * Roi * pose1->Rot().transpose();
    Eigen::Matrix3d R2eR2 = skew(pose2->Rot() * Roi.transpose() * Eigen::Vector3d(0, 0, 1));

    //[R_I0toG, p_I0inG, R_IitoG, p_IiinG]
    res(0) = -(eR1 * (pose2->pos() - pose1->pos()))(0);
    H_x.block(0, map_hx[pose1], 1, pose1->R()->size()) = eR1 * skew(pose2->pos());
    H_x.block(0, map_hx[pose1] + pose1->R()->size(), 1, pose1->p()->size()) = -eR1;
    H_x.block(0, map_hx[pose2], 1, pose2->R()->size()) = -eR1 * skew(pose2->pos());
    H_x.block(0, map_hx[pose2] + pose1->R()->size(), 1, pose1->p()->size()) = eR1;

    res(1) = 1 - (eR1 * pose2->Rot() * Roi.transpose() * Eigen::Vector3d(0, 0, 1))(0);
    H_x.block(1, map_hx[pose1], 1, pose1->R()->size()) = eR1 * R2eR2;
    H_x.block(1, map_hx[pose2], 1, pose1->R()->size()) = -eR1 * R2eR2;
}

void UpdaterHelper::get_plane_jacobian_full_group(std::shared_ptr<PoseHamilton> pose1, std::shared_ptr<PoseHamilton> pose2,
                                                  const Eigen::Matrix3d &Roi, Eigen::MatrixXd &H_x, Eigen::VectorXd &res,
                                                  std::vector<std::shared_ptr<Type>> &x_order) {
    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;

    map_hx.insert({pose1, total_hx});
    x_order.push_back(pose1);
    total_hx += pose1->size();

    map_hx.insert({pose2, total_hx});
    x_order.push_back(pose2);
    total_hx += pose2->size();

    res = Eigen::VectorXd::Zero(2);
    H_x = Eigen::MatrixXd::Zero(2, total_hx);
    Eigen::RowVector3d eR1 = Eigen::Vector3d(0, 0, 1).transpose() * Roi * pose1->Rot().transpose();
    Eigen::Matrix3d R2eR2 = skew(pose2->Rot() * Roi.transpose() * Eigen::Vector3d(0, 0, 1));

    //[R_I0toG, p_I0inG, R_IitoG, p_IiinG]
    res(0) = -(eR1 * (pose2->pos() - pose1->pos()))(0);
    // H_x.block(0, map_hx[pose1], 1,  pose1->R()->size()) = eR1 * skew(pose2->pos());
    H_x.block(0, map_hx[pose1] + pose1->R()->size(), 1, pose1->p()->size()) = -eR1;
    // H_x.block(0, map_hx[pose2], 1, pose2->R()->size()) = -eR1 * skew(pose2->pos());
    H_x.block(0, map_hx[pose2] + pose1->R()->size(), 1, pose1->p()->size()) = eR1;

    res(1) = 1 - (eR1 * pose2->Rot() * Roi.transpose() * Eigen::Vector3d(0, 0, 1))(0);
    H_x.block(1, map_hx[pose1], 1, pose1->R()->size()) = eR1 * R2eR2;
    H_x.block(1, map_hx[pose2], 1, pose1->R()->size()) = -eR1 * R2eR2;
}

void UpdaterHelper::get_plane_jacobian_full_msckf(std::shared_ptr<PoseHamilton> pose1, std::shared_ptr<PoseHamilton> pose2,
                                                  const Eigen::Matrix3d &Roi, Eigen::MatrixXd &H_x, Eigen::VectorXd &res,
                                                  std::vector<std::shared_ptr<Type>> &x_order) {
    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;

    map_hx.insert({pose1, total_hx});
    x_order.push_back(pose1);
    total_hx += pose1->size();

    map_hx.insert({pose2, total_hx});
    x_order.push_back(pose2);
    total_hx += pose2->size();

    res = Eigen::VectorXd::Zero(2);
    H_x = Eigen::MatrixXd::Zero(2, total_hx);
    Eigen::RowVector3d eR1 = Eigen::Vector3d(0, 0, 1).transpose() * Roi * pose1->Rot().transpose();

    //[R_I0toG, p_I0inG, R_IitoG, p_IiinG]
    res(0) = -(eR1 * (pose2->pos() - pose1->pos()))(0);
    H_x.block(0, map_hx[pose1], 1, pose1->R()->size()) =
        Eigen::Vector3d(0, 0, 1).transpose() * Roi * skew(pose1->Rot().transpose() * (pose2->pos() - pose1->pos()));
    H_x.block(0, map_hx[pose1] + pose1->R()->size(), 1, pose1->p()->size()) = -eR1;
    H_x.block(0, map_hx[pose2] + pose1->R()->size(), 1, pose1->p()->size()) = eR1;

    res(1) = 1 - (eR1 * pose2->Rot() * Roi.transpose() * Eigen::Vector3d(0, 0, 1))(0);
    H_x.block(1, map_hx[pose1], 1, pose1->R()->size()) =
        Eigen::Vector3d(0, 0, 1).transpose() * Roi * skew(pose1->Rot().transpose() * pose2->Rot() * Roi.transpose() * Eigen::Vector3d(0, 0, 1));
    H_x.block(1, map_hx[pose2], 1, pose1->R()->size()) = -eR1 * pose2->Rot() * skew(Roi.transpose() * Eigen::Vector3d(0, 0, 1));
}

void UpdaterHelper::get_plane_jacobian_full(std::shared_ptr<PoseHamilton> pose_IitoLOC, std::shared_ptr<PoseHamilton> pose_MAPtoLOC,
                                            const Eigen::Matrix3d &prior_R_ItoMAP, const Eigen::Vector3d &prior_p_IinMAP, const Eigen::Matrix3d &Roi,
                                            Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order) {
    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;

    map_hx.insert({pose_IitoLOC, total_hx});
    x_order.push_back(pose_IitoLOC);
    total_hx += pose_IitoLOC->size();

    map_hx.insert({pose_MAPtoLOC, total_hx});
    x_order.push_back(pose_MAPtoLOC);
    total_hx += pose_MAPtoLOC->size();

    res = Eigen::VectorXd::Zero(2);
    H_x = Eigen::MatrixXd::Zero(2, total_hx);
    Eigen::Matrix3d R_IitoLOC = pose_IitoLOC->Rot();
    Eigen::Vector3d p_IiinLOC = pose_IitoLOC->pos();
    Eigen::Matrix3d R_MAPtoLOC = pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinLOC = pose_MAPtoLOC->pos();

    Eigen::RowVector3d eR1_tmp = Eigen::Vector3d(0, 0, 1).transpose() * Roi * prior_R_ItoMAP.transpose();
    Eigen::RowVector3d eR1 = eR1_tmp * R_MAPtoLOC.transpose();
    Eigen::Matrix3d R2eR2 = skew(R_IitoLOC * Roi.transpose() * Eigen::Vector3d(0, 0, 1));
    Eigen::Vector3d p_IiinMAP = R_MAPtoLOC.transpose() * (p_IiinLOC - p_MAPinLOC);
    res(0) = -(eR1_tmp * (p_IiinMAP - prior_p_IinMAP))(0);
    H_x.block(0, map_hx[pose_IitoLOC], 1, pose_IitoLOC->R()->size()) = -eR1 * skew(p_IiinLOC);
    H_x.block(0, map_hx[pose_IitoLOC] + pose_IitoLOC->R()->size(), 1, pose_IitoLOC->p()->size()) = eR1;
    H_x.block(0, map_hx[pose_MAPtoLOC], 1, pose_MAPtoLOC->R()->size()) = eR1 * skew(p_IiinLOC);
    H_x.block(0, map_hx[pose_MAPtoLOC] + pose_MAPtoLOC->R()->size(), 1, pose_MAPtoLOC->p()->size()) = -eR1;

    res(1) = 1 - (eR1 * R_IitoLOC * Roi.transpose() * Eigen::Vector3d(0, 0, 1))(0);
    H_x.block(1, map_hx[pose_IitoLOC], 1, pose_IitoLOC->R()->size()) = -eR1 * R2eR2;
    H_x.block(1, map_hx[pose_MAPtoLOC], 1, pose_MAPtoLOC->R()->size()) = eR1 * R2eR2;
    // cout << "R_MAPtoLOC: " << R_MAPtoLOC << endl;
    // Eigen::RowVector3d eR1 = Eigen::Vector3d(0, 0, 1).transpose() * Roi * R_IitoLOC.transpose();
    // Eigen::Matrix3d R2eR2 = skew(R_MAPtoLOC * prior_R_ItoMAP * Roi.transpose() * Eigen::Vector3d(0, 0, 1));
    // Eigen::Vector3d prior_p_IinLOC = R_MAPtoLOC * prior_p_IinMAP + p_MAPinLOC;

    // //[R_IitoG, pIiinG, R_MAPtoLOC, p_MAPinLOC]
    // res(0) = -(eR1 * (prior_p_IinLOC - p_IiinLOC))(0);
    // H_x.block(0, map_hx[pose_IitoLOC], 1, pose_IitoLOC->R()->size()) = eR1 * skew(prior_p_IinLOC);
    // H_x.block(0, map_hx[pose_IitoLOC] + pose_IitoLOC->R()->size(), 1, pose_IitoLOC->p()->size()) = -eR1;
    // H_x.block(0, map_hx[pose_MAPtoLOC], 1, pose_MAPtoLOC->R()->size()) = -eR1 * skew(prior_p_IinLOC);
    // H_x.block(0, map_hx[pose_MAPtoLOC] + pose_MAPtoLOC->R()->size(), 1, pose_MAPtoLOC->p()->size()) = eR1;

    // res(1) = 1 - (eR1 * R_MAPtoLOC * prior_R_ItoMAP * Roi.transpose() * Eigen::Vector3d(0, 0, 1))(0);
    // H_x.block(1, map_hx[pose_IitoLOC], 1, pose_IitoLOC->R()->size()) = eR1 * R2eR2;
    // H_x.block(1, map_hx[pose_MAPtoLOC], 1, pose_MAPtoLOC->R()->size()) = -eR1 * R2eR2;
    // // cout << res.transpose() << endl;
    // // cout << H_x << endl;
}

void UpdaterHelper::get_plane_jacobian_full_group(std::shared_ptr<PoseHamilton> pose_IitoLOC, std::shared_ptr<PoseHamilton> pose_MAPtoLOC,
                                                  const Eigen::Matrix3d &prior_R_ItoMAP, const Eigen::Vector3d &prior_p_IinMAP,
                                                  const Eigen::Matrix3d &Roi, Eigen::MatrixXd &H_x, Eigen::VectorXd &res,
                                                  std::vector<std::shared_ptr<Type>> &x_order) {
    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;

    map_hx.insert({pose_IitoLOC, total_hx});
    x_order.push_back(pose_IitoLOC);
    total_hx += pose_IitoLOC->size();

    map_hx.insert({pose_MAPtoLOC, total_hx});
    x_order.push_back(pose_MAPtoLOC);
    total_hx += pose_MAPtoLOC->size();

    res = Eigen::VectorXd::Zero(2);
    H_x = Eigen::MatrixXd::Zero(2, total_hx);
    Eigen::Matrix3d R_IitoLOC = pose_IitoLOC->Rot();
    Eigen::Vector3d p_IiinLOC = pose_IitoLOC->pos();
    Eigen::Matrix3d R_MAPtoLOC = pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinLOC = pose_MAPtoLOC->pos();

    // Eigen::RowVector3d eR1_tmp = Eigen::Vector3d(0, 0, 1).transpose() * Roi * prior_R_ItoMAP.transpose();
    // Eigen::RowVector3d eR1 = eR1_tmp * R_MAPtoLOC.transpose();
    // Eigen::Matrix3d R2eR2 = skew(R_IitoLOC * Roi.transpose() * Eigen::Vector3d(0, 0, 1));
    // Eigen::Vector3d p_IiinMAP = R_MAPtoLOC.transpose() * (p_IiinLOC - p_MAPinLOC);
    // res(0) = -(eR1_tmp * (p_IiinMAP - prior_p_IinMAP))(0);
    // H_x.block(0, map_hx[pose_IitoLOC], 1, pose_IitoLOC->R()->size()) = -eR1 * skew(p_IiinLOC - p_MAPinLOC);
    // H_x.block(0, map_hx[pose_IitoLOC] + pose_IitoLOC->R()->size(), 1, pose_IitoLOC->p()->size()) = eR1;
    // H_x.block(0, map_hx[pose_MAPtoLOC], 1, pose_MAPtoLOC->R()->size()) = eR1 * skew(p_IiinLOC - p_MAPinLOC);
    // H_x.block(0, map_hx[pose_MAPtoLOC] + pose_MAPtoLOC->R()->size(), 1, pose_MAPtoLOC->p()->size()) = -eR1;

    // res(1) = 1 - (eR1 * R_IitoLOC * Roi.transpose() * Eigen::Vector3d(0, 0, 1))(0);
    // H_x.block(1, map_hx[pose_IitoLOC], 1, pose_IitoLOC->R()->size()) = -eR1 * R2eR2;
    // H_x.block(1, map_hx[pose_MAPtoLOC], 1, pose_MAPtoLOC->R()->size()) = eR1 * R2eR2;
    Eigen::RowVector3d eR1 = Eigen::Vector3d(0, 0, 1).transpose() * Roi * R_IitoLOC.transpose();
    Eigen::Matrix3d R2eR2 = skew(R_MAPtoLOC * prior_R_ItoMAP * Roi.transpose() * Eigen::Vector3d(0, 0, 1));
    Eigen::Vector3d tmp_prior = R_MAPtoLOC * prior_p_IinMAP;
    Eigen::Vector3d prior_p_IinLOC = tmp_prior + p_MAPinLOC;

    //[R_IitoG, pIiinG, R_MAPtoLOC, p_MAPinLOC]
    res(0) = -(eR1 * (prior_p_IinLOC - p_IiinLOC))(0);
    H_x.block(0, map_hx[pose_IitoLOC], 1, pose_IitoLOC->R()->size()) = eR1 * skew(tmp_prior);
    H_x.block(0, map_hx[pose_IitoLOC] + pose_IitoLOC->R()->size(), 1, pose_IitoLOC->p()->size()) = -eR1;
    H_x.block(0, map_hx[pose_MAPtoLOC], 1, pose_MAPtoLOC->R()->size()) = -eR1 * skew(tmp_prior);
    H_x.block(0, map_hx[pose_MAPtoLOC] + pose_MAPtoLOC->R()->size(), 1, pose_MAPtoLOC->p()->size()) = eR1;

    res(1) = 1 - (eR1 * R_MAPtoLOC * prior_R_ItoMAP * Roi.transpose() * Eigen::Vector3d(0, 0, 1))(0);
    H_x.block(1, map_hx[pose_IitoLOC], 1, pose_IitoLOC->R()->size()) = eR1 * R2eR2;
    H_x.block(1, map_hx[pose_MAPtoLOC], 1, pose_MAPtoLOC->R()->size()) = -eR1 * R2eR2;
}

void UpdaterHelper::get_plane_jacobian_full_msckf(std::shared_ptr<PoseHamilton> pose_IitoLOC, std::shared_ptr<PoseHamilton> pose_MAPtoLOC,
                                                  const Eigen::Matrix3d &prior_R_ItoMAP, const Eigen::Vector3d &prior_p_IinMAP,
                                                  const Eigen::Matrix3d &Roi, Eigen::MatrixXd &H_x, Eigen::VectorXd &res,
                                                  std::vector<std::shared_ptr<Type>> &x_order) {
    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;

    map_hx.insert({pose_IitoLOC, total_hx});
    x_order.push_back(pose_IitoLOC);
    total_hx += pose_IitoLOC->size();

    map_hx.insert({pose_MAPtoLOC, total_hx});
    x_order.push_back(pose_MAPtoLOC);
    total_hx += pose_MAPtoLOC->size();

    res = Eigen::VectorXd::Zero(2);
    H_x = Eigen::MatrixXd::Zero(2, total_hx);
    Eigen::Matrix3d R_IitoLOC = pose_IitoLOC->Rot();
    Eigen::Vector3d p_IiinLOC = pose_IitoLOC->pos();
    Eigen::Matrix3d R_MAPtoLOC = pose_MAPtoLOC->Rot();
    Eigen::Vector3d p_MAPinLOC = pose_MAPtoLOC->pos();

    Eigen::RowVector3d eR1_tmp = Eigen::Vector3d(0, 0, 1).transpose() * Roi * prior_R_ItoMAP.transpose();
    Eigen::RowVector3d eR1 = eR1_tmp * R_MAPtoLOC.transpose();
    Eigen::Matrix3d R2eR2 = skew(R_IitoLOC * Roi.transpose() * Eigen::Vector3d(0, 0, 1));
    Eigen::Vector3d p_IiinMAP = R_MAPtoLOC.transpose() * (p_IiinLOC - p_MAPinLOC);

    //[R_IitoG, pIiinG, R_MAPtoLOC, p_MAPinLOC]
    res(0) = -(eR1_tmp * (p_IiinMAP - prior_p_IinMAP))(0);
    H_x.block(0, map_hx[pose_IitoLOC], 1, pose_IitoLOC->R()->size()) = Eigen::MatrixXd::Zero(1, 3);
    H_x.block(0, map_hx[pose_IitoLOC] + pose_IitoLOC->R()->size(), 1, pose_IitoLOC->p()->size()) = eR1;
    H_x.block(0, map_hx[pose_MAPtoLOC], 1, pose_MAPtoLOC->R()->size()) = -eR1_tmp * skew(p_IiinMAP);
    H_x.block(0, map_hx[pose_MAPtoLOC] + pose_MAPtoLOC->R()->size(), 1, pose_MAPtoLOC->p()->size()) = -eR1;

    res(1) = 1 - (eR1 * R_IitoLOC * Roi.transpose() * Eigen::Vector3d(0, 0, 1))(0);
    H_x.block(1, map_hx[pose_IitoLOC], 1, pose_IitoLOC->R()->size()) = -eR1 * R2eR2 * R_IitoLOC;
    H_x.block(1, map_hx[pose_MAPtoLOC], 1, pose_MAPtoLOC->R()->size()) = eR1 * R2eR2 * R_MAPtoLOC;
    // cout << "R_MAPtoLOC: " << R_MAPtoLOC << endl;
    // Eigen::RowVector3d eR1 = Eigen::Vector3d(0, 0, 1).transpose() * Roi * R_IitoLOC.transpose();
    // Eigen::Vector3d prior_p_IinLOC = R_MAPtoLOC * prior_p_IinMAP + p_MAPinLOC;

    // //[R_IitoG, pIiinG, R_MAPtoLOC, p_MAPinLOC]
    // res(0) = -(eR1 * (prior_p_IinLOC - p_IiinLOC))(0);
    // H_x.block(0, map_hx[pose_IitoLOC], 1, pose_IitoLOC->R()->size()) = Eigen::Vector3d(0, 0, 1).transpose() * Roi * skew(R_IitoLOC.transpose() *
    // (prior_p_IinLOC - p_IiinLOC)); H_x.block(0, map_hx[pose_IitoLOC] + pose_IitoLOC->R()->size(), 1, pose_IitoLOC->p()->size()) = -eR1;
    // H_x.block(0, map_hx[pose_MAPtoLOC], 1, pose_MAPtoLOC->R()->size()) = -eR1 * R_MAPtoLOC * skew(prior_p_IinMAP);
    // H_x.block(0, map_hx[pose_MAPtoLOC] + pose_MAPtoLOC->R()->size(), 1, pose_MAPtoLOC->p()->size()) = eR1;

    // res(1) = 1 - (eR1 * R_MAPtoLOC * prior_R_ItoMAP * Roi.transpose() * Eigen::Vector3d(0, 0, 1))(0);
    // H_x.block(1, map_hx[pose_IitoLOC], 1, pose_IitoLOC->R()->size()) = Eigen::Vector3d(0, 0, 1).transpose() * Roi * skew(R_IitoLOC.transpose() *
    // R_MAPtoLOC * prior_R_ItoMAP * Roi.transpose() * Eigen::Vector3d(0, 0, 1)); H_x.block(1, map_hx[pose_MAPtoLOC], 1, pose_MAPtoLOC->R()->size()) =
    // -eR1 * R_MAPtoLOC * skew(prior_R_ItoMAP * Roi.transpose() * Eigen::Vector3d(0, 0, 1));
}

void UpdaterHelper::nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

    // Apply the left nullspace of H_f to all variables
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all index
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_f.cols(); ++n) {
        for (int m = (int)H_f.rows() - 1; m > n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (H_x.block(m - 1, 0, 2, H_x.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
        }
    }

    // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
    // NOTE: need to eigen3 eval here since this experiences aliasing!
    // H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
    H_x = H_x.block(H_f.cols(), 0, H_x.rows() - H_f.cols(), H_x.cols()).eval();
    res = res.block(H_f.cols(), 0, res.rows() - H_f.cols(), res.cols()).eval();

    // Sanity check
    assert(H_x.rows() == res.rows());
}

void UpdaterHelper::measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

    // Return if H_x is a fat matrix (there is no need to compress in this case)
    if (H_x.rows() <= H_x.cols())
        return;

    // Do measurement compression through givens rotations
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all index
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_x.cols(); n++) {
        for (int m = (int)H_x.rows() - 1; m > n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_x(m - 1, n), H_x(m, n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_x.block(m - 1, n, 2, H_x.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
        }
    }

    // If H is a fat matrix, then use the rows
    // Else it should be same size as our state
    int r = std::min(H_x.rows(), H_x.cols());

    // Construct the smaller jacobian and residual after measurement compression
    assert(r <= H_x.rows());
    H_x.conservativeResize(r, H_x.cols());
    res.conservativeResize(r, res.cols());
}

void UpdaterHelper::measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res, Eigen::MatrixXd &R) {

    // Return if H_x is a fat matrix (there is no need to compress in this case)
    if (H_x.rows() <= H_x.cols())
        return;

    // Do measurement compression through givens rotations
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all index
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_x.cols(); n++) {
        for (int m = (int)H_x.rows() - 1; m > n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_x(m - 1, n), H_x(m, n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_x.block(m - 1, n, 2, H_x.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (R.block(m - 1, 0, 2, R.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (R.block(0, m - 1, R.rows(), 2)).applyOnTheRight(0, 1, tempHo_GR);
        }
    }

    // If H is a fat matrix, then use the rows
    // Else it should be same size as our state
    int r = std::min(H_x.rows(), H_x.cols());

    // Construct the smaller jacobian and residual after measurement compression
    assert(r <= H_x.rows());
    H_x.conservativeResize(r, H_x.cols());
    res.conservativeResize(r, res.cols());
    R.conservativeResize(r, r);
}

} // namespace night_voyager