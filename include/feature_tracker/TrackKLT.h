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
#ifndef TRACKKLT_H
#define TRACKKLT_H

#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "core/NightVoyagerOptions.h"
#include "feature_tracker/FeatureDatabase.h"

namespace night_voyager {
class CamBase;

class TrackKLT {
  public:
    explicit TrackKLT(const NightVoyagerOptions &options, std::shared_ptr<CamBase> &camera_intrinsic)
        : num_features(options.init_options.init_max_features), camera_calib(camera_intrinsic), histogram_method(options.histogram_method),
          threshold(options.fast_threshold), grid_x(options.grid_x), grid_y(options.grid_y), database(std::make_shared<FeatureDatabase>()),
          min_px_dist(options.min_px_dist) {
        currid = 1;
        cout << "num_features: " << num_features;
    }

    void feed_new_camera(const CameraData &message);

    shared_ptr<FeatureDatabase> get_feature_database() { return database; }

    /// Getter method for number of active features
    int get_num_features() { return num_features; }

    /// Setter method for number of active features
    void set_num_features(int _num_features) { num_features = _num_features; }

    void display_history(cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::vector<size_t> highlighted, std::string overlay);

  protected:
    void feed_monocular(const CameraData &message);

    void perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, std::vector<cv::KeyPoint> &pts0, std::vector<size_t> &ids0);

    void perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                          std::vector<cv::KeyPoint> &pts1, std::vector<uchar> &mask_out);

    // Number of features we should try to track frame to frame
    int num_features;

    std::shared_ptr<CamBase> camera_calib;

    /// What histogram equalization method we should pre-process images with?
    HistogramMethod histogram_method;

    // Parameters for our FAST grid detector
    int threshold;
    int grid_x;
    int grid_y;

    // Database with all our current features
    std::shared_ptr<FeatureDatabase> database;

    // Minimum pixel distance to be "far away enough" to be a different extracted feature
    int min_px_dist;
    int pyr_levels = 5;
    cv::Size win_size = cv::Size(15, 15);

    // Last set of image pyramids
    vector<cv::Mat> img_pyramid_last;
    cv::Mat img_curr;
    vector<cv::Mat> img_pyramid_curr;

    /// Mutexs for our last set of image storage (img_last, pts_last, and ids_last)
    std::mutex mtx_feed;

    /// Mutex for editing the *_last variables
    std::mutex mtx_last_vars;

    /// Last set of images (use map so all trackers render in the same order)
    cv::Mat img_last;

    /// Last set of tracked points
    std::vector<cv::KeyPoint> pts_last;

    /// Set of IDs of each current feature in the database
    std::vector<size_t> ids_last;

    /// Master ID for this tracker (atomic to allow for multi-threading)
    std::atomic<size_t> currid;

    // Timing variables (most children use these...)
    boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;
};
} // namespace night_voyager
#endif