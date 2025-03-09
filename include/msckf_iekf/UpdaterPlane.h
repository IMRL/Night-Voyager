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
#ifndef UPDATER_PLANE_H
#define UPDATER_PLANE_H

#include "core/NightVoyagerOptions.h"

namespace night_voyager {

class State;
class PriorPoseManager;

class UpdaterPlane {

  public:
    /**
     * @brief Default constructor for our Plane updater
     *
     * Our updater has options allow for one to tune the different parameters for update.
     *
     */
    UpdaterPlane(NightVoyagerOptions &options, std::shared_ptr<PriorPoseManager> prpose);

    /**
     * @brief This will try to use the plane constraints to update the state.
     * @param state State of the filter
     */
    void update(std::shared_ptr<State> state);

    /**
     * @brief This will try to use the prior pose to update the state
     * @param state State of the filter
     */
    void update_with_prior(std::shared_ptr<State> state);

    /**
     * @brief This will try to use the prior pose to update the state and return the near prio poses
     * @param state State of the filter
     */
    std::map<int, bool> update_with_prior_tracking_recover(std::shared_ptr<State> state);

    void add_near_pose(const std::map<int, bool> &near_indices_to_store);

  protected:
    /// Options used during update for slam features
    UpdaterPlaneOptions _options_plane;

    /// Chi squared 95th percentile table (lookup would be size of residual)
    std::map<int, double> chi_squared_table;

    /// Prior pose manager
    std::shared_ptr<PriorPoseManager> _prpose;

    /// Select clone poses for reduce computation cost
    int step = 5;

    Eigen::Matrix3d _Roi, _Rio;
};

} // namespace night_voyager

#endif