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
#ifndef BIMATCHER_H
#define BIMATCHER_H

#include "core/CommonLib.h"
#include "core/NightVoyagerOptions.h"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <vector>

namespace night_voyager {

class PcdManager;

class BIMatcher {
  public:
    BIMatcher(NightVoyagerOptions &options, std::shared_ptr<CamBase> camera_intrinsic)
        : update_z_th(options.match_options.update_z_th), extend(options.match_options.extend), bi_filter(options.match_options.bi_filter),
          grey_th(options.match_options.grey_th), grey_th_low(options.match_options.grey_th_low), ang_th(options.match_options.ang_th_bi),
          large_off(options.match_options.large_off), large_extend(options.match_options.large_extend),
          remain_match(options.match_options.remain_match), cam(camera_intrinsic) {}

    vector<STMatch> match(const CameraData &img, std::shared_ptr<PcdManager> pcd, const vector<STMatch> &matches, Eigen::Matrix3d &R_MAPtoC,
                          Eigen::Vector3d &p_MAPinC);

    vector<ScoreData> search_streetlight(std::shared_ptr<PcdManager> pcd, const vector<STMatch> &matches, Eigen::Matrix3d &R_MAPtoC,
                                         Eigen::Vector3d &p_MAPinC, double min_dist, double max_dist);

    vector<cv::Rect> binary_segment(const CameraData &img, const vector<STMatch> &matches, bool is_low = false);

    vector<STMatch> search_matches(const vector<cv::Rect> &pre_boxes, const Eigen::Matrix3d &R_MAPtoC, const Eigen::Vector3d &p_MAPinC,
                                   const vector<ScoreData> &STs, std::shared_ptr<PcdManager> pcd, int off, bool use_large_extend = false);

    void match_filter(std::vector<STMatch> &matches, bool remain_match);

    std::vector<cv::Rect> get_boxes() { return bi_boxes; }

  protected:
    /// Threshold for searching streetlights
    double update_z_th;

    /// For expanding the projected center
    int extend;

    bool bi_filter;

    /// For searching more boxes using binary method
    int grey_th;

    int grey_th_low;

    double ang_th;

    int large_off;

    int large_extend;

    bool remain_match;

    std::shared_ptr<CamBase> cam;

    vector<ScoreData> all_STs;

    std::vector<cv::Rect> bi_boxes;
};

} // namespace night_voyager

#endif