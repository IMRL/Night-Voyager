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
#ifndef UPDATER_HELPER_H
#define UPDATER_HELPER_H

#include "core/CommonLib.h"
#include <Eigen/Core>
#include <memory>
#include <vector>

namespace night_voyager {
class State;
class Type;
class PcdManager;
class StreetlightFeature;
class PoseHamilton;

class UpdaterHelper {
  public:
    /**
     * @brief Feature object that our UpdaterHelper leverages, has all measurements and means
     */
    struct UpdaterHelperFeature {

        /// Unique ID of this feature
        size_t featid;

        /// UV coordinates that this feature has been seen from (mapped by camera ID)
        std::vector<Eigen::VectorXf> uvs;

        // UV normalized coordinates that this feature has been seen from (mapped by camera ID)
        std::vector<Eigen::VectorXf> uvs_norm;

        /// Timestamps of each UV measurement (mapped by camera ID)
        std::vector<double> timestamps;

        /// Timestamp of anchor clone
        double anchor_clone_timestamp = -1;

        /// Timestamp of pseudo anchor clone
        double pseudo_anchor_clone_timestamp = -1;

        /// Triangulated position of this feature, in the anchor frame
        Eigen::Vector3d p_FinA;

        /// Triangulated position of this feature, in the global frame
        Eigen::Vector3d p_FinG;
    };

    /**
     * @brief This gets the feature and state Jacobian
     *
     * @param[in] state State of the filter system
     * @param[in] message odom we want to get Jacobians of
     * @param[out] res Measurement residual for this feature
     * @param[out] H_x Extra Jacobians in respect to the state (for example anchored pose)
     * @param[out] x_order Extra variables our extra Jacobian has (for example anchored pose)
     */
    static void get_odom_jacobian_full(std::shared_ptr<State> state, const OdomData &message, Eigen::VectorXd &res, Eigen::MatrixXd &H_x,
                                       std::vector<std::shared_ptr<Type>> &x_order, Eigen::MatrixXd &R);

    static void get_odom_jacobian_full_msckf(std::shared_ptr<State> state, const OdomData &message, Eigen::VectorXd &res, Eigen::MatrixXd &H_x,
                                             std::vector<std::shared_ptr<Type>> &x_order, Eigen::MatrixXd &R);

    /**
     * @brief Will construct the "stacked" Jacobians for a single feature from all its measurements
     *
     * @param[in] state State of the filter system
     * @param[in] feature Feature we want to get Jacobians of (must have feature means)
     * @param[out] H_f Jacobians in respect to the feature error state
     * @param[out] H_x Extra Jacobians in respect to the state (for example anchored pose)
     * @param[out] res Measurement residual for this feature
     * @param[out] x_order Extra variables our extra Jacobian has (for example anchored pose)
     */
    static void get_feature_jacobian_full(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x,
                                          Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order);

    static void get_feature_jacobian_full_clone_group(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                      Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order);

    static void get_feature_jacobian_full_rel_group(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                    Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order);

    static void get_feature_jacobian_full_no_group(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                   Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order);

    static void get_feature_jacobian_full_msckf(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order);

    /**
     * @brief Will construct the "stacked" Jacobians for a single feature from all its measurements
     *
     * @param[in] state State of the filter system
     * @param[in] feature Feature we want to get Jacobians of (must have feature means)
     * @param[out] H_x Extra Jacobians in respect to the state (for example anchored pose)
     * @param[out] res Measurement residual for this feature
     * @param[out] x_order Extra variables our extra Jacobian has (for example anchored pose)
     * @param[in] pcd Provide 3d positions of streetlight clusters
     */
    static void get_stfeature_jacobian_full(std::shared_ptr<State> state, std::shared_ptr<StreetlightFeature> feature, Eigen::MatrixXd &H_x,
                                            Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order, std::shared_ptr<PcdManager> pcd,
                                            Eigen::MatrixXd &R);

    static void get_stfeature_jacobian_full_group(std::shared_ptr<State> state, std::shared_ptr<StreetlightFeature> feature, Eigen::MatrixXd &H_x,
                                                  Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order, std::shared_ptr<PcdManager> pcd,
                                                  Eigen::MatrixXd &R);

    static void get_stfeature_jacobian_full_msckf(std::shared_ptr<State> state, std::shared_ptr<StreetlightFeature> feature, Eigen::MatrixXd &H_x,
                                                  Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order, std::shared_ptr<PcdManager> pcd,
                                                  Eigen::MatrixXd &R);

    /**
     * @brief Will construct the "stacked" Jacobians for a single plane constraint
     *
     * @param[in] pose1 State of current pose
     * @param[in] pose2 State of other pose in sliding window
     * @param[out] H_x Extra Jacobians in respect to the state (for example anchored pose)
     * @param[out] res Measurement residual for this feature
     * @param[out] x_order Extra variables our extra Jacobian has (for example anchored pose)
     */
    static void get_plane_jacobian_full(std::shared_ptr<PoseHamilton> pose1, std::shared_ptr<PoseHamilton> pose2, const Eigen::Matrix3d &Roi,
                                        Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order);

    static void get_plane_jacobian_full_group(std::shared_ptr<PoseHamilton> pose1, std::shared_ptr<PoseHamilton> pose2, const Eigen::Matrix3d &Roi,
                                              Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order);

    static void get_plane_jacobian_full_msckf(std::shared_ptr<PoseHamilton> pose1, std::shared_ptr<PoseHamilton> pose2, const Eigen::Matrix3d &Roi,
                                              Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order);

    /**
     * @brief Will construct the "stacked" Jacobians for a single plane constraint
     *
     * @param[in] pose_IitoLOC State of current pose
     * @param[in] pose_MAPtoLOC State of world to local pose
     * @param[in] R_IitoMAP Stored Rotation matrix in mapping process
     * @param[in] p_IiinMAP Stored tranlation matrix in mapping process
     * @param[out] H_x Extra Jacobians in respect to the state (for example anchored pose)
     * @param[out] res Measurement residual for this feature
     * @param[out] x_order Extra variables our extra Jacobian has (for example anchored pose)
     */
    static void get_plane_jacobian_full(std::shared_ptr<PoseHamilton> pose_IitoLOC, std::shared_ptr<PoseHamilton> pose_MAPtoLOC,
                                        const Eigen::Matrix3d &prior_R_ItoMAP, const Eigen::Vector3d &prior_p_IinMAP, const Eigen::Matrix3d &Roi,
                                        Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order);

    static void get_plane_jacobian_full_group(std::shared_ptr<PoseHamilton> pose_IitoLOC, std::shared_ptr<PoseHamilton> pose_MAPtoLOC,
                                              const Eigen::Matrix3d &prior_R_ItoMAP, const Eigen::Vector3d &prior_p_IinMAP,
                                              const Eigen::Matrix3d &Roi, Eigen::MatrixXd &H_x, Eigen::VectorXd &res,
                                              std::vector<std::shared_ptr<Type>> &x_order);

    static void get_plane_jacobian_full_msckf(std::shared_ptr<PoseHamilton> pose_IitoLOC, std::shared_ptr<PoseHamilton> pose_MAPtoLOC,
                                              const Eigen::Matrix3d &prior_R_ItoMAP, const Eigen::Vector3d &prior_p_IinMAP,
                                              const Eigen::Matrix3d &Roi, Eigen::MatrixXd &H_x, Eigen::VectorXd &res,
                                              std::vector<std::shared_ptr<Type>> &x_order);

    /**
     * @brief This will project the left nullspace of H_f onto the linear system.
     *
     * Please see the @ref update-null for details on how this works.
     * This is the MSCKF nullspace projection which removes the dependency on the feature state.
     * Note that this is done **in place** so all matrices will be different after a function call.
     *
     * @param H_f Jacobian with nullspace we want to project onto the system [res = Hx*(x-xhat)+Hf(f-fhat)+n]
     * @param H_x State jacobian
     * @param res Measurement residual
     */
    static void nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res);

    /**
     * @brief This will perform measurement compression
     *
     * Please see the @ref update-compress for details on how this works.
     * Note that this is done **in place** so all matrices will be different after a function call.
     *
     * @param H_x State jacobian
     * @param res Measurement residual
     */
    static void measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res);

    static void measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res, Eigen::MatrixXd &R);
};

} // namespace night_voyager
#endif