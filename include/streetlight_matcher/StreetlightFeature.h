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
#ifndef STREETLIGHT_FEATURE_H
#define STREETLIGHT_FEATURE_H

#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace night_voyager {

class StreetlightFeature {
  public:
    /// Unique ID of this feature
    size_t featid;

    /// If this feature should be deleted
    bool to_delete;

    /// boxes that this feature has been detected from
    std::vector<cv::Rect> boxes;

    /// UV coordinates of box centers that this feature has been seen from
    std::vector<Eigen::VectorXf> uvs;

    /// UV normalized coordinates of box centers that this feature has been seen from
    std::vector<Eigen::VectorXf> uvs_norm;

    /// Timestamps of each UV measurement
    std::vector<double> timestamps;

    /**
     * @brief Remove measurements that are older then the specified timestamp.
     *
     * Given a valid timestamp, this will remove all measurements that have occured earlier then this.
     *
     * @param timestamp Timestamps that our measurements must occur after
     */
    void clean_older_measurements(double timestamp) {
        // Assert that we have all the parts of a measurement
        assert(timestamps.size() == uvs.size());
        assert(timestamps.size() == uvs_norm.size());
        assert(timestamps.size() == boxes.size());

        // Our iterators
        auto it1 = timestamps.begin();
        auto it2 = uvs.begin();
        auto it3 = uvs_norm.begin();
        auto it4 = boxes.begin();

        // Loop through measurement times, remove ones that are older then the specified one
        while (it1 != timestamps.end()) {
            if (*it1 <= timestamp) {
                it1 = timestamps.erase(it1);
                it2 = uvs.erase(it2);
                it3 = uvs_norm.erase(it3);
                it4 = boxes.erase(it4);
            } else {
                ++it1;
                ++it2;
                ++it3;
                ++it4;
            }
        }
    }

    /**
     * @brief Remove measurements that do not occur at passed timestamps.
     *
     * Given a series of valid timestamps, this will remove all measurements that have not occurred at these times.
     * This would normally be used to ensure that the measurements that we have occur at our clone times.
     *
     * @param valid_times Vector of timestamps that our measurements must occur at
     */
    void clean_old_measurements(const std::vector<double> &valid_times) {

        // Assert that we have all the parts of a measurement
        // std::cout << "timestamps: " << timestamps.size() << " " << "uvs: " << uvs.size() << std::endl;
        assert(timestamps.size() == uvs.size());
        assert(timestamps.size() == uvs_norm.size());
        assert(timestamps.size() == boxes.size());

        // Our iterators
        auto it1 = timestamps.begin();
        auto it2 = uvs.begin();
        auto it3 = uvs_norm.begin();
        auto it4 = boxes.begin();

        // Loop through measurement times, remove ones that are not in our timestamps
        while (it1 != timestamps.end()) {
            if (std::find(valid_times.begin(), valid_times.end(), *it1) == valid_times.end()) {
                it1 = timestamps.erase(it1);
                it2 = uvs.erase(it2);
                it3 = uvs_norm.erase(it3);
                it4 = boxes.erase(it4);
            } else {
                ++it1;
                ++it2;
                ++it3;
                ++it4;
            }
        }
    }
};

} // namespace night_voyager
#endif