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
#ifndef FEATURE_H
#define FEATURE_H

#include <Eigen/Eigen>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace night_voyager {
class Feature {

  public:
    /// Unique ID of this feature
    size_t featid;

    /// If this feature should be deleted
    bool to_delete;

    /// UV coordinates that this feature has been seen from (mapped by camera ID)
    std::vector<Eigen::VectorXf> uvs;

    /// UV normalized coordinates that this feature has been seen from (mapped by camera ID)
    std::vector<Eigen::VectorXf> uvs_norm;

    /// Timestamps of each UV measurement (mapped by camera ID)
    std::vector<double> timestamps;

    /// Timestamp of anchor clone
    double anchor_clone_timestamp;

    /// Timestamp of pseudo anchor clone
    double pseudo_anchor_clone_timestamp = -1;

    /// Triangulated position of this feature, in the anchor frame
    Eigen::Vector3d p_FinA;

    /// Triangulated position of this feature, in the global frame
    Eigen::Vector3d p_FinG;

    /**
     * @brief Remove measurements that do not occur at passed timestamps.
     *
     * Given a series of valid timestamps, this will remove all measurements that have not occurred at these times.
     * This would normally be used to ensure that the measurements that we have occur at our clone times.
     *
     * @param valid_times Vector of timestamps that our measurements must occur at
     */
    void clean_old_measurements(const std::vector<double> &valid_times);

    /**
     * @brief Remove measurements that occur at the invalid timestamps
     *
     * Given a series of invalid timestamps, this will remove all measurements that have occurred at these times.
     *
     * @param invalid_times Vector of timestamps that our measurements should not
     */
    void clean_invalid_measurements(const std::vector<double> &invalid_times);

    /**
     * @brief Remove measurements that are older then the specified timestamp.
     *
     * Given a valid timestamp, this will remove all measurements that have occured earlier then this.
     *
     * @param timestamp Timestamps that our measurements must occur after
     */
    void clean_older_measurements(double timestamp);
};
} // namespace night_voyager
#endif