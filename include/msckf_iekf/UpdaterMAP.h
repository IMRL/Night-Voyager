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
#ifndef UPDATER_MAP_H
#define UPDATER_MAP_H

#include "core/CommonLib.h"
#include "core/NightVoyagerOptions.h"
#include "prior_pose/PriorPoseManager.h"
#include "streetlight_matcher/StreetlightFeatureDatabase.h"
#include "streetlight_matcher/StreetlightMatcher.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <deque>
#include <memory>
#include <mutex>
#include <random>

namespace night_voyager {
class PcdManager;
class State;

class UpdaterMAP {

  public:
    /**
     * @brief Default constructor for our MAP updater
     *
     */
    UpdaterMAP(NightVoyagerOptions &options, std::shared_ptr<PcdManager> pcd, std::shared_ptr<PriorPoseManager> prpose, std::shared_ptr<CamBase> camera_intrinsic)
        : _options(options), cam(camera_intrinsic), _prpose(prpose), _pcd(pcd), st_database(new StreetlightFeatureDatabase()), no_matches(0) {
        st_matcher = std::make_shared<StreetlightMatcher>(options, camera_intrinsic);

        for (int i = 1; i < 500; i++) {
            boost::math::chi_squared chi_squared_dist(i);
            chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < 300; i++) {
            cv::Scalar color_msg(100, 30, 219);
            // cv::Scalar color_msg(int(255 * dis(gen)), int(255 * dis(gen)), int(255 * dis(gen)));
            random_colors.push_back(color_msg);
        }
    }

    /**
     * @brief Preprocess box data
     */
    BoxData preprocess(const CameraData &img);

    // /**
    //  * @brief This will try to use historical features and newly detected features(DL-based) to update the state.
    //  *
    //  * @param state State of the filter
    //  */
    // void update_for_dl_hist_features(const CameraData& message, std::shared_ptr<State> state);

    /**
     * @brief This will try to use historical features to update the state.
     *
     * @param state State of the filter
     */
    void update_for_hist_features(std::shared_ptr<State> state, double timestamp);

    void update_for_hist_features_tracking_recover(std::shared_ptr<State> state, double timestamp, std::shared_ptr<StreetlightFeatureDatabase> st_db);

    /**
     * @brief This will try to newly detected features(DL-based) to update the state.
     *
     * @param state State of the filter
     */
    void update_for_dl_features(std::shared_ptr<State> state, double timestamp);

    void update_for_dl_features_box(std::shared_ptr<State> state, double timestamp);

    /**
     * @brief This will try to newly detected features(BI-based) to update the state.
     *
     * @param state State of the filter
     */
    void update_for_bi_features(std::shared_ptr<State> state, double timestamp);

    /**
     * @brief Match detections with streetlights and update the streetlight features(DL-based)
     *
     * @param state State of the filter
     *
     * @param timestamp Timestamp of current measurement
     */
    void match_and_update_dl_features(std::shared_ptr<State> state, double timestamp);

    vector<STMatch> match_and_update_for_dl_features_tracking_recover(std::shared_ptr<State> state, double timestamp, std::shared_ptr<StreetlightFeatureDatabase> st_db,
                                                                      double reloc_z_th, int reloc_extend);

    /**
     * @brief Match detections with streetlights and update the streetlight features(BI-based)
     *
     * @param state State of the filter
     *
     * @param timestamp Timestamp of current measurement
     */
    void match_and_update_bi_features(const CameraData &message, std::shared_ptr<State> state);

    void match_and_update_dl_features_box(std::shared_ptr<State> state, const CameraData &message);

    /**
     * @brief Feed function for Box data
     * @param message Contains our timestamp and box information
     * @param oldest_time Time that we can discard measurements before
     */
    void feed_box(const BoxData &message, double oldest_time = -1) {

        // Append it to our vector
        std::lock_guard<std::mutex> lck(box_data_mtx);
        box_data.push_back(message);

        // Clean old measurements
        clean_old_box_measurements(oldest_time - 0.10);
    }

    /**
     * @brief This will remove any IMU measurements that are older then the given measurement time
     * @param oldest_time Time that we can discard measurements before (in IMU clock)
     */
    void clean_old_box_measurements(double oldest_time) {
        if (oldest_time < 0)
            return;
        auto it0 = box_data.begin();
        while (it0 != box_data.end()) {
            if (it0->timestamp < oldest_time) {
                it0 = box_data.erase(it0);
            } else {
                it0++;
            }
        }
    }

    /**
     * @brief This will find if there exists detected box matching current image
     */
    bool findDetection(const double &timestamp) {
        // cout << "state_timestamp: " << timestamp << endl;
        std::lock_guard<std::mutex> lck(box_data_mtx);
        if (!box_data.empty()) {
            assert(box_data.front().timestamp < timestamp);
            if (box_data.back().timestamp < timestamp) {
                PRINT_INFO(WHITE "No streetlight detection results");
                return false;
            } else {
                bool find_box = false;
                auto erase_idx = box_data.begin();

                auto it0 = box_data.begin();
                while (it0 != box_data.end()) {
                    if (abs(it0->timestamp - timestamp) <= 1e-4) {
                        cur_box = *it0;
                        find_box = true;
                        erase_idx = it0;
                        break;
                    }
                    ++it0;
                }

                assert(find_box);
                box_data.erase(box_data.begin(), erase_idx);
                return true;
            }
        } else {
            return false;
        }
    }

    void updatepcd();

    void updatepcd_tracking_recover(const std::vector<STMatch> &matches);

    void display_streetlights(std::shared_ptr<State> state, cv::Mat &img_out1, cv::Mat &img_out2, string overlay1, string overlay2);

    void display_streetlights_detection(cv::Mat &img_out, string overlay);

    void display_streetlights_tracking_recover(std::shared_ptr<State> state, cv::Mat &img_out1, cv::Mat &img_out2, string overlay);

    void display_other_streetlights_tracking_recover(std::vector<std::shared_ptr<State>> &states, cv::Mat &img_out);

    int matches_num() { return cur_dl_matches.size() + cur_bi_matches.size(); }

    void zero_matches_num() {
        cur_dl_matches.clear();
        cur_bi_matches.clear();
    }

    BoxData get_cur_stbox() { return cur_box; }

    std::shared_ptr<StreetlightFeatureDatabase> get_st_database() { return st_database; }

    void set_st_database(std::shared_ptr<StreetlightFeatureDatabase> _st_database) { st_database = _st_database; }

    void set_cur_dl_matches(std::vector<STMatch> _dl_matches) { cur_dl_matches = _dl_matches; }

    void set_chi2_multipler_dl(double chi2_multipler) { _options.map_options.chi2_multipler_dl = chi2_multipler; }

    void set_chi2_multipler_bi(double chi2_multipler) { _options.map_options.chi2_multipler_bi = chi2_multipler; }

    bool exist_matches() { return !cur_bi_matches.empty() || !cur_dl_matches.empty(); }

  protected:
    /// Options used during update
    NightVoyagerOptions _options;

    std::mutex mtx_last_vars;
    cv::Mat img_last;

    std::shared_ptr<StreetlightMatcher> st_matcher;

    std::shared_ptr<CamBase> cam;

    std::shared_ptr<PriorPoseManager> _prpose;

    /// Pointcloud class object
    std::shared_ptr<PcdManager> _pcd;

    /// Stores box data
    std::deque<BoxData> box_data;
    BoxData cur_box;
    std::mutex box_data_mtx;

    /// Stores current matches based on DL-based detections
    std::vector<STMatch> cur_dl_matches, cur_dl_matches_box;
    std::vector<cv::Rect> dl_boxes;
    /// Stores curretn matches based on BI-based detections
    std::vector<STMatch> cur_bi_matches;
    std::vector<cv::Rect> bi_boxes;

    /// Chi squared 95th percentile table (lookup would be size of residual)
    std::map<int, double> chi_squared_table;

    /// Stores Streetlight features
    std::shared_ptr<StreetlightFeatureDatabase> st_database;

    int no_matches;

    std::vector<cv::Scalar> random_colors;
};
} // namespace night_voyager

#endif