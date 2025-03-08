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
#ifndef UPDATER_MSCKF_H
#define UPDATER_MSCKF_H

#include "core/NightVoyagerOptions.h"
#include <map>
#include <vector>

namespace night_voyager {
class State;
class Feature;
class FeatureInitializer;

class UpdaterMSCKF {

  public:
    /**
     * @brief Default constructor for our MSCKF updater
     *
     * Our updater has a feature initializer which we use to initialize features as needed.
     * Also the options allow for one to tune the different parameters for update.
     */
    UpdaterMSCKF(NightVoyagerOptions &options);

    /**
     * @brief Given tracked features, this will try to use them to update the state.
     *
     * @param state State of the filter
     * @param feature_vec Features that can be used for update
     */
    void update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec);

  protected:
    /// Options used during update
    UpdaterOptions _options;

    /// Feature initializer class object
    std::shared_ptr<FeatureInitializer> initializer_feat;

    /// Chi squared 95th percentile table (lookup would be size of residual)
    std::map<int, double> chi_squared_table;
};
} // namespace night_voyager
#endif