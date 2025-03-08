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
#include "feature_tracker/FeatureDatabase.h"
#include "feature_tracker/Feature.h"
#include "utils/Print.h"

namespace night_voyager {
std::shared_ptr<Feature> FeatureDatabase::get_feature(size_t id, bool remove) {
    std::lock_guard<std::mutex> lck(mtx);
    if (features_idlookup.find(id) != features_idlookup.end()) {
        std::shared_ptr<Feature> temp = features_idlookup.at(id);
        if (remove)
            features_idlookup.erase(id);
        return temp;
    } else {
        return nullptr;
    }
}

bool FeatureDatabase::get_feature_clone(size_t id, Feature &feat) {
    std::lock_guard<std::mutex> lck(mtx);
    if (features_idlookup.find(id) == features_idlookup.end())
        return false;
    // TODO: should probably have a copy constructor function in feature class
    std::shared_ptr<Feature> temp = features_idlookup.at(id);
    feat.featid = temp->featid;
    feat.to_delete = temp->to_delete;
    feat.uvs = temp->uvs;
    feat.uvs_norm = temp->uvs_norm;
    feat.timestamps = temp->timestamps;
    feat.anchor_clone_timestamp = temp->anchor_clone_timestamp;
    feat.pseudo_anchor_clone_timestamp = temp->pseudo_anchor_clone_timestamp;
    feat.p_FinA = temp->p_FinA;
    feat.p_FinG = temp->p_FinG;
    return true;
}

void FeatureDatabase::update_feature(size_t id, double timestamp, float u, float v, float u_n, float v_n) {

    // Find this feature using the ID lookup
    std::lock_guard<std::mutex> lck(mtx);
    if (features_idlookup.find(id) != features_idlookup.end()) {
        // Get our feature
        std::shared_ptr<Feature> feat = features_idlookup.at(id);
        // Append this new information to it!
        feat->uvs.push_back(Eigen::Vector2f(u, v));
        feat->uvs_norm.push_back(Eigen::Vector2f(u_n, v_n));
        feat->timestamps.push_back(timestamp);
        return;
    }

    // Debug info
    // PRINT_DEBUG("featdb - adding new feature %d\n",(int)id);

    // Else we have not found the feature, so lets make it be a new one!
    std::shared_ptr<Feature> feat = std::make_shared<Feature>();
    feat->featid = id;
    feat->uvs.push_back(Eigen::Vector2f(u, v));
    feat->uvs_norm.push_back(Eigen::Vector2f(u_n, v_n));
    feat->timestamps.push_back(timestamp);

    // Append this new feature into our database
    features_idlookup[id] = feat;
}

std::vector<std::shared_ptr<Feature>> FeatureDatabase::features_not_containing_newer(double timestamp, bool remove, bool skip_deleted) {

    // Our vector of features that do not have measurements after the specified time
    std::vector<std::shared_ptr<Feature>> feats_old;

    // Now lets loop through all features, and just make sure they are not old
    std::lock_guard<std::mutex> lck(mtx);
    for (auto it = features_idlookup.begin(); it != features_idlookup.end();) {
        // Skip if already deleted
        if (skip_deleted && (*it).second->to_delete) {
            it++;
            continue;
        }
        // Loop through each camera
        // If we have a measurement greater-than or equal to the specified, this measurement is find
        bool has_newer_measurement = false;
        auto const feat = (*it).second->timestamps;
        has_newer_measurement = (!feat.empty() && feat.at(feat.size() - 1) >= timestamp);

        // If it is not being actively tracked, then it is old
        if (!has_newer_measurement) {
            feats_old.push_back((*it).second);
            if (remove)
                features_idlookup.erase(it++);
            else
                it++;
        } else {
            it++;
        }
    }

    // Debugging
    PRINT_DEBUG("feature db size = %u\n", features_idlookup.size())

    // Return the old features
    return feats_old;
}

std::vector<std::shared_ptr<Feature>> FeatureDatabase::features_containing_older(double timestamp, bool remove, bool skip_deleted) {

    // Our vector of old features
    std::vector<std::shared_ptr<Feature>> feats_old;

    // Now lets loop through all features, and just make sure they are not old
    std::lock_guard<std::mutex> lck(mtx);
    for (auto it = features_idlookup.begin(); it != features_idlookup.end();) {
        // Skip if already deleted
        if (skip_deleted && (*it).second->to_delete) {
            it++;
            continue;
        }
        // Loop through each camera
        // Check if we have at least one time older then the requested
        bool found_containing_older = false;
        auto const feat = (*it).second->timestamps;
        found_containing_older = (!feat.empty() && feat.at(0) < timestamp);

        // If it has an older timestamp, then add it
        if (found_containing_older) {
            feats_old.push_back((*it).second);
            if (remove)
                features_idlookup.erase(it++);
            else
                it++;
        } else {
            it++;
        }
    }

    // Debugging
    PRINT_DEBUG("feature db size = %u\n", features_idlookup.size())

    // Return the old features
    return feats_old;
}

std::vector<std::shared_ptr<Feature>> FeatureDatabase::features_containing(double timestamp, bool remove, bool skip_deleted) {

    // Our vector of old features
    std::vector<std::shared_ptr<Feature>> feats_has_timestamp;

    // Now lets loop through all features, and just make sure they are not
    std::lock_guard<std::mutex> lck(mtx);
    for (auto it = features_idlookup.begin(); it != features_idlookup.end();) {
        // Skip if already deleted
        if (skip_deleted && (*it).second->to_delete) {
            it++;
            continue;
        }
        // Boolean if it has the timestamp
        // Break out if we found a single timestamp that is equal to the specified time
        bool has_timestamp = false;
        auto const &feat = (*it).second->timestamps;
        has_timestamp = (std::find(feat.begin(), feat.end(), timestamp) != feat.end());

        // Remove this feature if it contains the specified timestamp
        if (has_timestamp) {
            feats_has_timestamp.push_back((*it).second);
            if (remove)
                features_idlookup.erase(it++);
            else
                it++;
        } else {
            it++;
        }
    }

    // Debugging
    PRINT_DEBUG("feature db size = %u\n", features_idlookup.size())
    PRINT_DEBUG("return vector = %u\n", feats_has_timestamp.size())

    // Return the features
    return feats_has_timestamp;
}

void FeatureDatabase::cleanup() {
    // Loop through all features
    // int sizebefore = (int)features_idlookup.size();
    std::lock_guard<std::mutex> lck(mtx);
    for (auto it = features_idlookup.begin(); it != features_idlookup.end();) {
        // If delete flag is set, then delete it
        if ((*it).second->to_delete) {
            features_idlookup.erase(it++);
        } else {
            it++;
        }
    }
}

void FeatureDatabase::cleanup_measurements(double timestamp) {
    std::lock_guard<std::mutex> lck(mtx);
    for (auto it = features_idlookup.begin(); it != features_idlookup.end();) {
        // Remove the older measurements
        (*it).second->clean_older_measurements(timestamp);
        // Count how many measurements
        int ct_meas = 0;
        ct_meas += (int)((*it).second->timestamps.size());

        // If delete flag is set, then delete it
        if (ct_meas < 1) {
            features_idlookup.erase(it++);
        } else {
            it++;
        }
    }
}

void FeatureDatabase::cleanup_measurements_exact(double timestamp) {
    std::lock_guard<std::mutex> lck(mtx);
    std::vector<double> timestamps = {timestamp};
    for (auto it = features_idlookup.begin(); it != features_idlookup.end();) {
        // Remove the older measurements
        (*it).second->clean_invalid_measurements(timestamps);
        // Count how many measurements
        int ct_meas = 0;
        ct_meas += (int)((*it).second->timestamps.size());

        // If delete flag is set, then delete it
        if (ct_meas < 1) {
            features_idlookup.erase(it++);
        } else {
            it++;
        }
    }
}

double FeatureDatabase::get_oldest_timestamp() {
    std::lock_guard<std::mutex> lck(mtx);
    double oldest_time = -1;
    for (auto const &feat : features_idlookup) {
        auto const time = feat.second->timestamps;
        if (!time.empty() && (oldest_time == -1 || oldest_time > time.at(0))) {
            oldest_time = time.at(0);
        }
    }
    return oldest_time;
}

void FeatureDatabase::append_new_measurements(const std::shared_ptr<FeatureDatabase> &database) {
    std::lock_guard<std::mutex> lck(mtx);

    // Loop through the other database's internal database
    // int sizebefore = (int)features_idlookup.size();
    for (const auto &feat : database->get_internal_data()) {
        if (features_idlookup.find(feat.first) != features_idlookup.end()) {

            // For this feature, now try to append the new measurement data
            std::shared_ptr<Feature> temp = features_idlookup.at(feat.first);
            auto temp_times = temp->timestamps;
            const auto times = feat.second->timestamps;
            for (size_t i = 0; i < feat.second->timestamps.size(); i++) {
                double time_to_find = feat.second->timestamps.at(i);
                if (std::find(temp_times.begin(), temp_times.end(), time_to_find) == temp_times.end()) {
                    temp->timestamps.push_back(feat.second->timestamps.at(i));
                    temp->uvs.push_back(feat.second->uvs.at(i));
                    temp->uvs_norm.push_back(feat.second->uvs_norm.at(i));
                }
            }
        } else {
            // Else we have not found the feature, so lets make it be a new one!
            std::shared_ptr<Feature> temp = std::make_shared<Feature>();
            temp->featid = feat.second->featid;
            temp->timestamps = feat.second->timestamps;
            temp->uvs = feat.second->uvs;
            temp->uvs_norm = feat.second->uvs_norm;
            features_idlookup[feat.first] = temp;
        }
    }
}
} // namespace night_voyager