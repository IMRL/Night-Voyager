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
#ifndef STREETLIGHT_FEATUREDATABASE_H
#define STREETLIGHT_FEATUREDATABASE_H

#include <Eigen/Core>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

namespace night_voyager {

class StreetlightFeature;

class StreetlightFeatureDatabase {

  public:
    /**
     * @brief Default constructor
     */
    StreetlightFeatureDatabase(){}

    StreetlightFeatureDatabase(const StreetlightFeatureDatabase &other);

    /**
     * @brief Update a feature object
     * @param id ID of the feature we will update
     * @param timestamp time that this measurement occured at
     * @param box box measurement of streetlight
     * @param u_n undistorted/normalized u coordinate of box center
     * @param v_n undistorted/normalized v coordinate of box center
     *
     * This will update a given feature based on the passed ID it has.
     * It will create a new feature, if it is an ID that we have not seen before.
     */
    void update_feature(size_t id, double timestamp, const cv::Rect &rect, const Eigen::Vector2f &center, const Eigen::Vector2f &center_n,
                        const Eigen::Vector2f &noise = Eigen::Vector2f::Zero());

    void update_feature(std::shared_ptr<StreetlightFeature> feature);

    /**
     * @brief Check if there exists streetlight features
     */
    bool database_empty() { return features_idlookup.empty(); }

    /**
     * @brief Remove measurements that do not occur at passed timestamps.
     *
     * Given a series of valid timestamps, this will remove all measurements that have not occurred at these times.
     * This would normally be used to ensure that the measurements that we have occur at our clone times.
     *
     * @param valid_times Vector of timestamps that our measurements must occur at
     */
    void clean_old_measurements(const std::vector<double> &valid_times);

    /**StreetlightFeatureDatabase
     * @brief Return number of measurements
     */
    size_t count_measurements();

    /**
     * @brief Return the feautures_idlookup
     */
    std::unordered_map<size_t, std::shared_ptr<StreetlightFeature>> &get_features_idlookup() {
        std::lock_guard<std::mutex> lck(mtx);
        return features_idlookup;
    }

  protected:
    /// Mutex lock for our map
    std::mutex mtx;

    /// Our lookup array that allow use to query based on ID
    std::unordered_map<size_t, std::shared_ptr<StreetlightFeature>> features_idlookup;
};

} // namespace night_voyager

#endif