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
#ifndef UPDATER_SLAM_H
#define UPDATER_SLAM_H

#include "core/NightVoyagerOptions.h"

namespace night_voyager {

class State;
class Feature;
class Landmark;
class FeatureInitializer;

class UpdaterSLAM {

  public:
    /**
     * @brief Default constructor for our SLAM updater
     *
     * Our updater has a feature initializer which we use to initialize features as needed.
     * Also the options allow for one to tune the different parameters for update.
     *
     */
    UpdaterSLAM(NightVoyagerOptions &options);

    /**
     * @brief Given tracked SLAM features, this will try to use them to update the state.
     * @param state State of the filter
     * @param feature_vec Features that can be used for update
     */
    void update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec);

    /**
     * @brief Given max track features, this will try to use them to initialize them in the state.
     * @param state State of the filter
     * @param feature_vec Features that can be used for update
     */
    void delayed_init(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec);

  protected:
    /// Options used during update for slam features
    UpdaterOptions _options_slam;

    /// Feature initializer class object
    std::shared_ptr<FeatureInitializer> initializer_feat;

    /// Chi squared 95th percentile table (lookup would be size of residual)
    std::map<int, double> chi_squared_table;
};

} // namespace night_voyager

#endif