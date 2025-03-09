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
#ifndef TRACKINGRECOVER_H
#define TRACKINGRECOVER_H

#include "core/NightVoyagerOptions.h"
#include "utils/Print.h"
#include <Eigen/Core>
#include <memory>
#include <vector>

namespace night_voyager {

struct PoseScore {
    float score = -10000;
    Eigen::Matrix3d R;
    Eigen::Vector3d p;
    std::vector<int> matches;
};

class State;
class StreetlightFeatureDatabase;
class PriorPoseManager;
class PcdManager;

class TrackingRecover {
  public:
    TrackingRecover(const NightVoyagerOptions &options, std::shared_ptr<PcdManager> pcd, std::shared_ptr<CamBase> camera_intrinsic,
                    std::shared_ptr<PriorPoseManager> prpose)
        : options(options.tracking_recover_options), _pcd(pcd), _cam(camera_intrinsic), _prpose(prpose), nomatch_dist(-1.0), nomatch_ang(-1.0) {
        tracking_status = TrackingStatus::TRACKINGSUCCESS;
    }

    void trans_after_nomatch(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos);

    bool traverse_all_situations(std::shared_ptr<State> state, const CameraData &img, BoxData &st_boxes,
                                 std::shared_ptr<StreetlightFeatureDatabase> st_database, bool msckf = false);

    void change_tracking_status(TrackingStatus tracking_status_) {
        tracking_status = tracking_status_;
        if (tracking_status == TrackingStatus::TRACKINGSUCCESS) {
            PRINT_INFO(YELLOW "[change-tracking-state]: Tracking state is changed to TRACKINGSUCCESS!\n");
        } else if (tracking_status == TrackingStatus::TRACKINGUNSTABLESUCCESS) {
            PRINT_INFO(YELLOW "[change-tracking-state]: Tracking state is changed to TRACKINGUNSTABLESUCCESS!\n");
        } else if (tracking_status == TrackingStatus::TRACKINGMAYLOST) {
            PRINT_INFO(YELLOW "[change-tracking-state]: Tracking state is changed to TRACKINGMAYLOST\n");
        } else if (tracking_status == TrackingStatus::TRACKINGLOST) {
            PRINT_INFO(YELLOW "[change-tracking-state]: No matches lasts for %f meters, %f radians. Tracking state is changed to TRACKINGLOST\n",
                       nomatch_dist, nomatch_ang);
        }
        nomatch_dist = -1.0;
        nomatch_ang = -1.0;
    }

    bool is_tracking_lost() { return tracking_status == TrackingStatus::TRACKINGLOST; }

    bool check_transform_th() { return (nomatch_dist > options.dist_th && nomatch_ang > options.ang_th); }

    void clear_dist() {
        nomatch_dist = -1.0;
        nomatch_ang = -1.0;
    }

    TrackingStatus check_tracking_status() { return tracking_status; }

    void copy_state_stdatabase(std::shared_ptr<State> state, int states_num, std::shared_ptr<StreetlightFeatureDatabase> st_database);

    bool check_if_select(const BoxData &cur_box);

    std::vector<STMatch> select_best_state(const CameraData &img, const BoxData &cur_box);

    void update_select_state(std::shared_ptr<State> state, std::shared_ptr<StreetlightFeatureDatabase> st_database,
                             const std::vector<STMatch> &matches, bool msckf = false);

    BoxData process_box(const CameraData &img, const BoxData &cur_box);

    std::vector<ScoreData> select_streetlights(const Eigen::Matrix3d &R_MAPtoC, const Eigen::Vector3d &p_MAPinC);

    void generateCombinations(std::vector<int> &combination, int start, int n, int index, int r, std::vector<std::vector<int>> &allCombinations);

    void generatePermutations(std::vector<int> &permutation, std::vector<bool> &chosen, int n, int r, std::vector<std::vector<int>> &allPermutations);

    void store_poses();

    std::vector<std::shared_ptr<State>> states;

    /// Stores Streetlight features
    std::vector<std::shared_ptr<StreetlightFeatureDatabase>> st_databases;

    /// Current matches for each state
    std::vector<vector<STMatch>> cur_stmatches;

    std::vector<std::map<double, Eigen::Matrix3d>> rots_loc;

    std::vector<std::map<double, Eigen::Vector3d>> poss_loc;

    std::vector<std::map<double, Eigen::Matrix3d>> rots_global;

    std::vector<std::map<double, Eigen::Vector3d>> poss_global;

  private:
    TrackingRecoverOptions options;

    /// Pointcloud class object
    std::shared_ptr<PcdManager> _pcd;

    std::shared_ptr<CamBase> _cam;

    std::shared_ptr<PriorPoseManager> _prpose;

    Eigen::Matrix3d nomatch_rot;

    Eigen::Vector3d nomatch_pos;

    float nomatch_dist;

    float nomatch_ang;

    Eigen::Matrix3d last_R_MAPtoC;

    Eigen::Vector3d last_P_MAPinC;

    std::vector<std::pair<BoxData, std::vector<int>>> matches_per_obs;

    TrackingStatus tracking_status;

    int level;
};
} // namespace night_voyager
#endif