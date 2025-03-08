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
#ifndef FEATURE_HELPER_H
#define FEATURE_HELPER_H

#include "feature_tracker/Feature.h"
#include "feature_tracker/FeatureDatabase.h"

namespace night_voyager {
class FeatureHelper {

  public:
    /**
     * @brief This functions will compute the disparity between common features in the two frames.
     *
     * First we find all features in the first frame.
     * Then we loop through each and find the uv of it in the next requested frame.
     * Features are skipped if no tracked feature is found (it was lost).
     * NOTE: this is on the RAW coordinates of the feature not the normalized ones.
     * NOTE: This computes the disparity over all cameras!
     *
     * @param db Feature database pointer
     * @param time0 First camera frame timestamp
     * @param time1 Second camera frame timestamp
     * @param disp_mean Average raw disparity
     * @param disp_var Variance of the disparities
     * @param total_feats Total number of common features
     */
    static void compute_disparity(std::shared_ptr<FeatureDatabase> db, double time0, double time1, double &disp_mean, double &disp_var,
                                  int &total_feats) {

        // Get features seen from the first image
        std::vector<std::shared_ptr<Feature>> feats0 = db->features_containing(time0, false, true);

        // Compute the disparity
        std::vector<double> disparities;
        for (auto &feat : feats0) {

            // Get the two uvs for both times
            auto it0 = std::find(feat->timestamps.begin(), feat->timestamps.end(), time0);
            auto it1 = std::find(feat->timestamps.begin(), feat->timestamps.end(), time1);
            if (it0 == feat->timestamps.end() || it1 == feat->timestamps.end())
                continue;
            auto idx0 = std::distance(feat->timestamps.begin(), it0);
            auto idx1 = std::distance(feat->timestamps.begin(), it1);

            // Now lets calculate the disparity
            Eigen::Vector2f uv0 = feat->uvs.at(idx0).block(0, 0, 2, 1);
            Eigen::Vector2f uv1 = feat->uvs.at(idx1).block(0, 0, 2, 1);
            disparities.push_back((uv1 - uv0).norm());
        }

        // If no disparities, just return
        if (disparities.size() < 2) {
            disp_mean = -1;
            disp_var = -1;
            total_feats = 0;
        }

        // Compute mean and standard deviation in respect to it
        disp_mean = 0;
        for (double disp_i : disparities) {
            disp_mean += disp_i;
        }
        disp_mean /= (double)disparities.size();
        disp_var = 0;
        for (double &disp_i : disparities) {
            disp_var += std::pow(disp_i - disp_mean, 2);
        }
        disp_var = std::sqrt(disp_var / (double)(disparities.size() - 1));
        total_feats = (int)disparities.size();
    }

    /**
     * @brief This functions will compute the disparity over all features we have
     *
     * NOTE: this is on the RAW coordinates of the feature not the normalized ones.
     * NOTE: This computes the disparity over all cameras!
     *
     * @param db Feature database pointer
     * @param disp_mean Average raw disparity
     * @param disp_var Variance of the disparities
     * @param total_feats Total number of common features
     * @param newest_time Only compute disparity for ones older (-1 to disable)
     * @param oldest_time Only compute disparity for ones newer (-1 to disable)
     */
    static void compute_disparity(std::shared_ptr<FeatureDatabase> db, double &disp_mean, double &disp_var, int &total_feats, double newest_time = -1,
                                  double oldest_time = -1) {

        // Compute the disparity
        std::vector<double> disparities;
        for (auto &feat : db->get_internal_data()) {

            // Skip if only one observation
            if (feat.second->timestamps.size() < 2)
                continue;

            // Now lets calculate the disparity (assumes time array is monotonic)
            bool found0 = false;
            bool found1 = false;
            Eigen::Vector2f uv0 = Eigen::Vector2f::Zero();
            Eigen::Vector2f uv1 = Eigen::Vector2f::Zero();
            for (size_t idx = 0; idx < feat.second->timestamps.size(); idx++) {
                double time = feat.second->timestamps.at(idx);
                if ((oldest_time == -1 || time > oldest_time) && !found0) {
                    uv0 = feat.second->uvs.at(idx).block(0, 0, 2, 1);
                    found0 = true;
                    continue;
                }
                if ((newest_time == -1 || time < newest_time) && found0) {
                    uv1 = feat.second->uvs.at(idx).block(0, 0, 2, 1);
                    found1 = true;
                    continue;
                }
            }

            // If we found both an old and a new time, then we are good!
            if (!found0 || !found1)
                continue;
            disparities.push_back((uv1 - uv0).norm());
        }

        // If no disparities, just return
        if (disparities.size() < 2) {
            disp_mean = -1;
            disp_var = -1;
            total_feats = 0;
        }

        // Compute mean and standard deviation in respect to it
        disp_mean = 0;
        for (double disp_i : disparities) {
            disp_mean += disp_i;
        }
        disp_mean /= (double)disparities.size();
        disp_var = 0;
        for (double &disp_i : disparities) {
            disp_var += std::pow(disp_i - disp_mean, 2);
        }
        disp_var = std::sqrt(disp_var / (double)(disparities.size() - 1));
        total_feats = (int)disparities.size();
    }

  private:
    // Cannot construct this class
    FeatureHelper() {}
};
} // namespace night_voyager
#endif