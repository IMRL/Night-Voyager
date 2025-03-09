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
#ifndef DLMATCHER_H
#define DLMATCHER_H

#include "core/CommonLib.h"
#include "core/NightVoyagerOptions.h"

namespace night_voyager {

class STMatch;
class PcdManager;

class DLMatcher {
  public:
    DLMatcher(NightVoyagerOptions &options, std::shared_ptr<CamBase> camera_intrinsic)
        : z_th(options.match_options.z_th), dl_filter(options.match_options.dl_filter), alpha(options.match_options.alpha),
          ang_th(options.match_options.ang_th_dl), cam(camera_intrinsic) {}

    vector<STMatch> match(const BoxData &boxes, std::shared_ptr<PcdManager> pcd, Eigen::Matrix3d &R_MAPtoC, Eigen::Vector3d &p_MAPinC,
                          Eigen::MatrixXd &pose_cov);

    vector<STMatch> match_tracking_recover(const BoxData &boxes, std::shared_ptr<PcdManager> pcd, Eigen::Matrix3d &R_MAPtoC,
                                           Eigen::Vector3d &p_MAPinC, double reloc_z_th, int reloc_extend);

    vector<STMatch> match_box(const BoxData &boxes, std::shared_ptr<PcdManager> pcd, vector<STMatch> &matches, Eigen::Matrix3d &R_MAPtoC,
                              Eigen::Vector3d &p_MAPinC);

    double md_distance(const Eigen::Vector2f &box_center, const Eigen::Vector2d &proj_center, const Eigen::Matrix2d &cov_box_center);

    double ang_distance(const Eigen::Vector2f &box_center, const Eigen::Vector3d &streetlight_center, const Eigen::Matrix3d &cov_streetlight_center,
                        const Eigen::Matrix2d &cov_box_center);

    vector<ScoreData> search_streetlight(std::shared_ptr<PcdManager> pcd, const Eigen::Matrix3d &R_MAPtoC, const Eigen::Vector3d &p_MAPinC);

    vector<ScoreData> search_streetlight_tracking_recover(std::shared_ptr<PcdManager> pcd, const Eigen::Matrix3d &R_MAPtoC,
                                                          const Eigen::Vector3d &p_MAPinC, double reloc_z_th);

    void match_filter(std::vector<STMatch> &matches);

    std::vector<cv::Rect> get_boxes() { return dl_boxes; }

  protected:
    /// Threshold for searching streetlights
    double z_th;

    bool dl_filter;

    /// Weight for the matching score
    double alpha;

    double ang_th;

    std::shared_ptr<CamBase> cam;

    std::vector<ScoreData> all_STs;

    std::vector<cv::Rect> dl_boxes;
};
} // namespace night_voyager

#endif