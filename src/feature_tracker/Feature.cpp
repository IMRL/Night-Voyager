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
#include "feature_tracker/Feature.h"

namespace night_voyager {
void Feature::clean_old_measurements(const std::vector<double> &valid_times) {

    // Assert that we have all the parts of a measurement
    assert(timestamps.size() == uvs.size());
    assert(timestamps.size() == uvs_norm.size());

    // Our iterators
    auto it1 = timestamps.begin();
    auto it2 = uvs.begin();
    auto it3 = uvs_norm.begin();

    // Loop through measurement times, remove ones that are not in our timestamps
    while (it1 != timestamps.end()) {
        if (std::find(valid_times.begin(), valid_times.end(), *it1) == valid_times.end()) {
            it1 = timestamps.erase(it1);
            it2 = uvs.erase(it2);
            it3 = uvs_norm.erase(it3);
        } else {
            ++it1;
            ++it2;
            ++it3;
        }
    }
}

void Feature::clean_invalid_measurements(const std::vector<double> &invalid_times) {

    // Assert that we have all the parts of a measurement
    assert(timestamps.size() == uvs.size());
    assert(timestamps.size() == uvs_norm.size());

    // Our iterators
    auto it1 = timestamps.begin();
    auto it2 = uvs.begin();
    auto it3 = uvs_norm.begin();

    // Loop through measurement times, remove ones that are in our timestamps
    while (it1 != timestamps.end()) {
        if (std::find(invalid_times.begin(), invalid_times.end(), *it1) != invalid_times.end()) {
            it1 = timestamps.erase(it1);
            it2 = uvs.erase(it2);
            it3 = uvs_norm.erase(it3);
        } else {
            ++it1;
            ++it2;
            ++it3;
        }
    }
}

void Feature::clean_older_measurements(double timestamp) {

    // Assert that we have all the parts of a measurement
    assert(timestamps.size() == uvs.size());
    assert(timestamps.size() == uvs_norm.size());

    // Our iterators
    auto it1 = timestamps.begin();
    auto it2 = uvs.begin();
    auto it3 = uvs_norm.begin();

    // Loop through measurement times, remove ones that are older then the specified one
    while (it1 != timestamps.end()) {
        if (*it1 <= timestamp) {
            it1 = timestamps.erase(it1);
            it2 = uvs.erase(it2);
            it3 = uvs_norm.erase(it3);
        } else {
            ++it1;
            ++it2;
            ++it3;
        }
    }
}
} // namespace night_voyager