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
#ifndef OPEN_VINS_FEATUREINITIALIZER_H
#define OPEN_VINS_FEATUREINITIALIZER_H

#include <unordered_map>

#include "core/NightVoyagerOptions.h"

namespace night_voyager {
class Feature;

/**
 * @brief Class that triangulates feature
 *
 * This class has the functions needed to triangulate and then refine a given 3D feature.
 * As in the standard MSCKF, we know the clones of the camera from propagation and past updates.
 * Thus, we just need to triangulate a feature in 3D with the known poses and then refine it.
 * One should first call the single_triangulation() function afterwhich single_gaussnewton() allows for refinement.
 * Please see the @ref update-featinit page for detailed derivations.
 */
class FeatureInitializer {

  public:
    /**
     * @brief Structure which stores pose estimates for use in triangulation
     *
     * - R_GtoC - rotation from global to camera
     * - p_CinG - position of camera in global frame
     */
    struct ClonePose {

        /// Rotation
        Eigen::Matrix<double, 3, 3> _Rot;

        /// Position
        Eigen::Matrix<double, 3, 1> _pos;

        /// Constructs pose from rotation and position
        ClonePose(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &p) {
            _Rot = R;
            _pos = p;
        }

        /// Default constructor
        ClonePose() {
            _Rot = Eigen::Matrix<double, 3, 3>::Identity();
            _pos = Eigen::Matrix<double, 3, 1>::Zero();
        }

        /// Accessor for rotation
        const Eigen::Matrix<double, 3, 3> &Rot() { return _Rot; }

        /// Accessor for position
        const Eigen::Matrix<double, 3, 1> &pos() { return _pos; }
    };

    /**
     * @brief Default constructor
     * @param options Options for the initializer
     */
    FeatureInitializer(FeatureInitializerOptions &options) : _options(options) {}

    /**
     * @brief Uses a linear triangulation to get initial estimate for the feature
     *
     * The derivations for this method can be found in the @ref featinit-linear documentation page.
     *
     * @param feat Pointer to feature
     * @param clonesCAM Map of timestamp to camera pose estimate (rotation from global to camera, position of camera
     * in global frame)
     * @return Returns false if it fails to triangulate (based on the thresholds)
     */
    bool single_triangulation(std::shared_ptr<Feature> feat, std::unordered_map<double, ClonePose> &clonesCAM);

    /**
     * @brief Uses a linear triangulation to get initial estimate for the feature, treating the anchor observation as a true bearing.
     *
     * The derivations for this method can be found in the @ref featinit-linear-1d documentation page.
     * This function should be used if you want speed, or know your anchor bearing is reasonably accurate.
     *
     * @param feat Pointer to feature
     * @param clonesCAM Map of timestamp to camera pose estimate (rotation from global to camera, position of camera
     * in global frame)
     * @return Returns false if it fails to triangulate (based on the thresholds)
     */
    bool single_triangulation_1d(std::shared_ptr<Feature> feat, std::unordered_map<double, ClonePose> &clonesCAM);

    /**
     * @brief Uses a nonlinear triangulation to refine initial linear estimate of the feature
     * @param feat Pointer to feature
     * @param clonesCAM Map of timestamp to camera pose estimate (rotation from global to camera, position of camera
     * in global frame)
     * @return Returns false if it fails to be optimize (based on the thresholds)
     */
    bool single_gaussnewton(std::shared_ptr<Feature> feat, std::unordered_map<double, ClonePose> &clonesCAM);

    /**
     * @brief Gets the current configuration of the feature initializer
     * @return Const feature initializer config
     */
    const FeatureInitializerOptions config() { return _options; }

  protected:
    /// Contains options for the initializer process
    FeatureInitializerOptions _options;

    /**
     * @brief Helper function for the gauss newton method that computes error of the given estimate
     * @param clonesCAM Map of timestamp to camera pose estimate
     * @param feat Pointer to the feature
     * @param alpha x/z in anchor
     * @param beta y/z in anchor
     * @param rho 1/z inverse depth
     */
    double compute_error(std::unordered_map<double, ClonePose> &clonesCAM, std::shared_ptr<Feature> feat, double alpha, double beta, double rho);
};
} // namespace night_voyager

#endif