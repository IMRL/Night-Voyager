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
#ifndef UPDATER_ODOM_H
#define UPDATER_ODOM_H

#include "core/CommonLib.h"
#include "core/NightVoyagerOptions.h"

namespace night_voyager {

class State;

class UpdaterOdom {

  public:
    UpdaterOdom(NightVoyagerOptions &options);

    void update(std::shared_ptr<State> state, const OdomData &odom);

  protected:
    /// Options used during update
    UpdaterOdomOptions _options;

    /// Chi squared 95th percentile table (lookup would be size of residual)
    std::map<int, double> chi_squared_table;
};

} // namespace night_voyager

#endif