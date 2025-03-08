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
#ifndef STREETLIGHT_MATCHER_H
#define STREETLIGHT_MATCHER_H

#include "core/NightVoyagerOptions.h"
#include "streetlight_matcher/BIMatcher.h"
#include "streetlight_matcher/DLMatcher.h"
#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <memory>

namespace night_voyager {

class PcdManager;

class StreetlightMatcher {
  public:
    StreetlightMatcher(NightVoyagerOptions &options, std::shared_ptr<CamBase> camera_intrinsic) {
        dlmatcher = std::make_shared<DLMatcher>(options, camera_intrinsic);
        bimatcher = std::make_shared<BIMatcher>(options, camera_intrinsic);
    }

    std::vector<STMatch> run_dl_match(const BoxData &boxes, std::shared_ptr<PcdManager> pcd, Eigen::Matrix3d &R_MAPtoC, Eigen::Vector3d &p_MAPinC,
                                      Eigen::MatrixXd &pose_cov) {
        return dlmatcher->match(boxes, pcd, R_MAPtoC, p_MAPinC, pose_cov);
    }

    std::vector<STMatch> run_dl_match_box(const BoxData &boxes, std::shared_ptr<PcdManager> pcd, vector<STMatch> &matches, Eigen::Matrix3d &R_MAPtoC,
                                          Eigen::Vector3d &p_MAPinC) {
        return dlmatcher->match_box(boxes, pcd, matches, R_MAPtoC, p_MAPinC);
    }

    std::vector<STMatch> run_dl_match_tracking_recover(const BoxData &boxes, std::shared_ptr<PcdManager> pcd, Eigen::Matrix3d &R_MAPtoC,
                                                       Eigen::Vector3d &p_MAPinC, double reloc_z_th, int reloc_extend) {
        return dlmatcher->match_tracking_recover(boxes, pcd, R_MAPtoC, p_MAPinC, reloc_z_th, reloc_extend);
    }

    std::vector<STMatch> run_bi_match(const CameraData &img, std::shared_ptr<PcdManager> pcd, vector<STMatch> &matches, Eigen::Matrix3d &R_MAPtoC,
                                      Eigen::Vector3d &p_MAPinC) {
        return bimatcher->match(img, pcd, matches, R_MAPtoC, p_MAPinC);
    }

    std::vector<cv::Rect> get_dl_boxes() { return dlmatcher->get_boxes(); }

    std::vector<cv::Rect> get_bi_boxes() { return bimatcher->get_boxes(); }

  protected:
    std::shared_ptr<DLMatcher> dlmatcher;
    std::shared_ptr<BIMatcher> bimatcher;
};

} // namespace night_voyager

#endif