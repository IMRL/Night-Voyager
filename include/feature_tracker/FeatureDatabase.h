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
#ifndef FEATURE_DATABASE_H
#define FEATURE_DATABASE_H

#include <Eigen/Core>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace night_voyager {
class Feature;

class FeatureDatabase {

  public:
    /**
     * @brief Default constructor
     */
    FeatureDatabase() {}

    /**
     * @brief Get a specified feature
     * @param id What feature we want to get
     * @param remove Set to true if you want to remove the feature from the database (you will need to handle the freeing of memory)
     * @return Either a feature object, or null if it is not in the database.
     */
    std::shared_ptr<Feature> get_feature(size_t id, bool remove = false);

    /**
     * @brief Get a specified feature clone (pointer is thread safe)
     * @param id What feature we want to get
     * @param feat Feature with data in it
     * @return True if the feature was found
     */
    bool get_feature_clone(size_t id, Feature &feat);

    /**
     * @brief Update a feature object
     * @param id ID of the feature we will update
     * @param timestamp time that this measurement occured at
     * @param u raw u coordinate
     * @param v raw v coordinate
     * @param u_n undistorted/normalized u coordinate
     * @param v_n undistorted/normalized v coordinate
     *
     * This will update a given feature based on the passed ID it has.
     * It will create a new feature, if it is an ID that we have not seen before.
     */
    void update_feature(size_t id, double timestamp, float u, float v, float u_n, float v_n);

    /**
     * @brief Get features that do not have newer measurement then the specified time.
     *
     * This function will return all features that do not a measurement at a time greater than the specified time.
     * For example this could be used to get features that have not been successfully tracked into the newest frame.
     * All features returned will not have any measurements occurring at a time greater then the specified.
     */
    std::vector<std::shared_ptr<Feature>> features_not_containing_newer(double timestamp, bool remove = false, bool skip_deleted = false);

    /**
     * @brief Get features that has measurements older then the specified time.
     *
     * This will collect all features that have measurements occurring before the specified timestamp.
     * For example, we would want to remove all features older then the last clone/state in our sliding window.
     */
    std::vector<std::shared_ptr<Feature>> features_containing_older(double timestamp, bool remove = false, bool skip_deleted = false);

    /**
     * @brief Get features that has measurements at the specified time.
     *
     * This function will return all features that have the specified time in them.
     * This would be used to get all features that occurred at a specific clone/state.
     */
    std::vector<std::shared_ptr<Feature>> features_containing(double timestamp, bool remove = false, bool skip_deleted = false);

    /**
     * @brief This function will delete all features that have been used up.
     *
     * If a feature was unable to be used, it will still remain since it will not have a delete flag set
     */
    void cleanup();

    /**
     * @brief This function will delete all feature measurements that are older then the specified timestamp
     */
    void cleanup_measurements(double timestamp);

    /**
     * @brief This function will delete all feature measurements that are at the specified timestamp
     */
    void cleanup_measurements_exact(double timestamp);

    /**
     * @brief Returns the size of the feature database
     */
    size_t size() {
        std::lock_guard<std::mutex> lck(mtx);
        return features_idlookup.size();
    }

    /**
     * @brief Returns the internal data (should not normally be used)
     */
    std::unordered_map<size_t, std::shared_ptr<Feature>> get_internal_data() {
        std::lock_guard<std::mutex> lck(mtx);
        return features_idlookup;
    }

    /**
     * @brief Gets the oldest time in the database
     */
    double get_oldest_timestamp();

    /**
     * @brief Will update the passed database with this database's latest feature information.
     */
    void append_new_measurements(const std::shared_ptr<FeatureDatabase> &database);

  protected:
    /// Mutex lock for our map
    std::mutex mtx;

    /// Our lookup array that allow use to query based on ID
    std::unordered_map<size_t, std::shared_ptr<Feature>> features_idlookup;
};
} // namespace night_voyager
#endif