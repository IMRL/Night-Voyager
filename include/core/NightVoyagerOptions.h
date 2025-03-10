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
#ifndef NightVoyagerOptions_H
#define NightVoyagerOptions_H

#include "core/CommonLib.h"
#include "utils/Print.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <string>
#include <vector>

using namespace std;

namespace night_voyager {
struct UpdaterOptions {

    /// What chi-squared multipler we should apply
    double chi2_multipler = 5;

    /// Noise sigma for our raw pixel measurements
    double sigma_pix = 1;

    /// Covariance for our raw pixel measurements
    double sigma_pix_sq = 1;
};

struct UpdaterMapOptions {

    /// What chi-squared multipler we should apply
    double chi2_multipler_dl = 5;

    double chi2_multipler_bi = 5;

    /// Noise sigma for our raw pixel measurements
    double sigma_pix = 1;

    /// Covariance for our raw pixel measurements
    double sigma_pix_sq = 1;
};

struct UpdaterOdomOptions {

    /// What chi-squared multipler we should apply
    double chi2_multipler = 5;

    /// Covariance for odometer measurements
    Eigen::Matrix3d odom_cov_eig = 0.01 * Eigen::Matrix3d::Identity();

    vector<double> sigma_odom;
};

struct UpdaterPlaneOptions {

    /// What chi-squared multipler we should apply
    double chi2_multipler_loc = 5;
    double chi2_multipler_prior = 5;

    /// We use the distance between different frames to control the covariance
    double distance_weight_loc = 0.02;
    double distance_weight_loc_sq = 0.02;

    /// We use the distance between frame and prior poses to control the covariance
    double distance_weight_prior = 0.03;
    double distance_weight_prior_sq = 0.03;

    /// Threshold for select pose
    double distance_thresh_loc = 1.5;

    /// Threshold for select prior pose
    double distance_thresh_prior = 2.0;

};

struct StateOptions {

    /// Numerical integration methods
    enum IntegrationMethod { DISCRETE, RK4, ANALYTICAL };

    /// What type of numerical integration is used during propagation
    IntegrationMethod integration_method = IntegrationMethod::RK4;

    /// Delay, in seconds, that we should wait from init before we start estimating SLAM features
    double dt_slam_delay = 2.0;

    /// Max clone size of sliding window
    int max_clone_size = 11;

    /// Max number of estimated SLAM features
    int max_slam_features = 25;

    /// Max number of SLAM features we allow to be included in a single EKF update.
    int max_slam_in_update = 1000;

    /// Max number of MSCKF features we will use at a given image timestep.
    int max_msckf_in_update = 1000;

    /// What type of KF we should use
    KFCLASS kf;
};

struct FeatureInitializerOptions {

    /// If we should perform 1d triangulation instead of 3d
    bool triangulate_1d = false;

    /// If we should perform Levenberg-Marquardt refinment
    bool refine_features = true;

    /// Max runs for Levenberg-Marquardt
    int max_runs = 5;

    /// Init lambda for Levenberg-Marquardt optimization
    double init_lamda = 1e-3;

    /// Max lambda for Levenberg-Marquardt optimization
    double max_lamda = 1e10;

    /// Cutoff for dx increment to consider as converged
    double min_dx = 1e-6;

    /// Cutoff for cost decrement to consider as converged
    double min_dcost = 1e-6;

    /// Multiplier to increase/decrease lambda
    double lam_mult = 10;

    /// Minimum distance to accept triangulated features
    double min_dist = 0.10;

    /// Maximum distance to accept triangulated features
    double max_dist = 60;

    /// Max baseline ratio to accept triangulated features
    double max_baseline = 40;

    /// Max condition number of linear triangulation matrix accept triangulated features
    double max_cond_number = 10000;
};

struct InertialInitializerOptions {
    // Used for VIWO initialization
    // *************************************************************
    /// Amount of time we will initialize over (seconds)
    double init_window_time = 1.0;

    /// Variance threshold on our acceleration to be classified as moving
    double init_imu_thresh = 1.0;

    /// Number of features we should try to track
    int init_max_features = 50;

    double gravity_mag = 9.81;
    // *************************************************************

    // Used for the initialization of relative transformation

    /// If set false, use Identity matrix as T_map_to_loc
    bool need_init_in_map = true;

    /// If set need_init_in_map false, use the prior pose as the initial pose
    Eigen::Matrix3d preset_Rwi;

    Eigen::Vector3d preset_pwi;

    /// Scope for searching streetlights (used for map initialization)
    double search_dist_scope1 = 45;

    double search_dist_scope2 = 75;

    double sine_th = 0.5;

    /// Preset outlier number (use for map initialization)
    int preset_outliers = 0;

    /// If we should use binary method to detect more lights for map initiailization
    bool init_use_binary = true;

    /// The threshold for detecting streetlights
    int init_grey_thresh = 252;

    /// The threshold for detecting small streetlights
    int init_grey_thresh_low = 245;

    /// The weight for balancing different scores
    double subscore_weight = 1;

    /// The score when observation is distributed to non-map point
    double outlier_score;

    double max_st_dist;

    /// Number of top combinations for each region (used for map initialization)
    int top_tuples_in_region;

    /// Number of top poses (used for map initialization)
    int top_poses;

    /// The expansion scope for projected streetlights, thus for determining whether there exist possible observations (used for map initialization)
    int expansion_proj;

    bool prior_init_available = false;

    Eigen::Vector3d prior_init;

    double prior_scale;
};

struct StreetlightMatchOptions {

    /// Search scope for the match of first streetlight detection
    double z_th;

    /// Search scope for the match of second streetlight detection
    double update_z_th;

    /// Extended search scope in image for the second streetlight detection
    int extend;

    /// Threshold for the binary method-based streetlight detection
    int grey_th;

    /// Threshold for more binary method-based streetlight detection
    int grey_th_low;

    /// Weight for the matching score
    double alpha;

    /// If to use filter to remove harmful dl-boxes
    bool dl_filter;

    /// If to use filter to remove harmful bi-boxes
    bool bi_filter;

    /// Threshold for sin theta of binary-based detection in select 3D streetlights (avoid overlapping lights)
    double ang_th_bi;

    /// Threshold for sin theta of learning-based detection in select 3D streetlights (avoid overlapping lights)
    double ang_th_dl;

    /// Safety threshold for determining binary candidate matches (When few streetlights are detected)
    int large_off;

    /// Expanded scope for binary-based streetlight detections (When few streetlights are detected)
    int large_extend;

    /// Whether to remain the match when the streetlight detection boxes are overlapped
    bool remain_match;
};

struct TrackingRecoverOptions {

    /// Traveled distance since no matches
    float dist_th;

    /// Traveled angle since no matches
    float ang_th;

    /// Traveled distance since tracking lost
    float lost_dist_th;

    /// Traveled angle since tracking lost
    float lost_ang_th;

    /// Search scope for match in tracking lost
    double z_th;

    /// Extened search scope in image for match in tracking lost
    int extend;

    /// Threshold for binary-based small streetlight detection
    int grey_thresh_low;

    /// The expansion scope for projected streetlights, thus for determining whether there exist possible observations (used for tracking recover)
    int expansion_proj;

    /// The weight of subscore
    float subscore_weight;

    /// Preset variance for scoring
    float variance;

    /// Scope for searching streetlights (used for tracking recover)
    double search_dist_scope;

    int search_bar_height;

    int area_th;

    /// The weight of balancing subscore
    float expected_weight;

    /// When identifying an outlier, the value that score should minus
    float outlier_cost;

    /// Variance of noise
    float sigma_pix_sq;

    /// Chi2 multipler for dl-based update when tracking may lost lasts for a short time
    double chi2_multipler_dl_maylost;

    /// Chi2 multipler for bi-based update when tracking may lost lasts for a short time
    double chi2_multipler_bi_maylost;

    /// Chi2 multipler for dl-based update when tracking lost lasts for a long time
    double chi2_multipler_dl_lost;

    /// Chi2 multipler for bi-based update when tracking lost lasts for a long time
    double chi2_multipler_bi_lost;

    float prior_z_diff_th;
};

struct NightVoyagerOptions {

    void param_load(ros::NodeHandle &nh) {
        nh.param<bool>("display_ground_truth", display_ground_truth, false);
        nh.param<string>("ground_truth_path", ground_truth_path, "");

        nh.param<string>("pcd_path", pcd_path, "");
        nh.param<string>("virtual_center_path", virtual_center_path, "");
        nh.param<string>("downsampled_pose_path", downsampled_pose_path, "");
        nh.param<string>("prior_pose_path", prior_pose_path, "");
        PRINT_INFO("pcd_path: %s\n", pcd_path.c_str());
        PRINT_INFO("virtual_center_path: %s\n", virtual_center_path.c_str());
        PRINT_INFO("downsampled_pose_path: %s\n", downsampled_pose_path.c_str());
        PRINT_INFO("prior_pose_path: %s\n", prior_pose_path.c_str());
        // pcd_path = root_dir + pcd_path;
        // virtual_center_path = root_dir + virtual_center_path;
        // downsampled_pose_path = root_dir + downsampled_pose_path;
        // prior_pose_path = root_dir + prior_pose_path;

        nh.param<string>("common/img_topic", img_topic, "/camera/color/image_raw/compressed");
        nh.param<string>("common/imu_topic", imu_topic, "/imu_data");
        nh.param<string>("common/odom_topic", odom_topic, "/odom");
        nh.param<string>("common/box_topic", box_topic, "/yolov7_bbox");

        int which_group;
        nh.param<int>("state/which_group", which_group, 0);
        if (which_group == -2)
            state_options.kf = KFCLASS::MSCKF;
        else if (which_group == -1)
            state_options.kf = KFCLASS::IKF_NOGROUP;
        else if (which_group == 0)
            state_options.kf = KFCLASS::IKF_IMUGROUP;
        else if (which_group == 1)
            state_options.kf = KFCLASS::IKF_RELGROUP;
        else if (which_group == 2)
            state_options.kf = KFCLASS::IKF_CHANGEGROUP;
        nh.param<int>("state/max_clone_size", state_options.max_clone_size, 11);
        nh.param<int>("state/max_slam_features", state_options.max_slam_features, 25);
        nh.param<int>("state/max_slam_in_update", state_options.max_slam_in_update, 1000);
        nh.param<int>("state/max_msckf_in_update", state_options.max_msckf_in_update, 1000);

        nh.param<bool>("update/do_update_msckf", do_update_msckf, true);
        nh.param<bool>("update/do_update_slam", do_update_slam, true);
        nh.param<bool>("update/do_update_odom", do_update_odom, true);
        nh.param<bool>("update/do_update_map", do_update_map, true);
        nh.param<bool>("update/do_update_plane", do_update_plane, true);
        nh.param<bool>("update/use_virtual_center", use_virtual_center, true);
        nh.param<bool>("update/use_match_extension", use_match_extension, true);
        nh.param<double>("update/msckf_iekf_sigma_px", msckf_options.sigma_pix, 1);
        nh.param<double>("update/msckf_chi2_multipler", msckf_options.chi2_multipler, 5);
        nh.param<double>("update/slam_sigma_px", slam_options.sigma_pix, 1);
        nh.param<double>("update/slam_chi2_multipler", slam_options.chi2_multipler, 5);
        nh.param<double>("update/zupt_chi2_multipler", zupt_options.chi2_multipler, 5);
        nh.param<double>("update/map_sigma_px_dl", map_options.sigma_pix, 3);
        nh.param<double>("update/map_chi2_multipler_dl", map_options.chi2_multipler_dl, 5);
        nh.param<double>("update/map_chi2_multipler_bi", map_options.chi2_multipler_bi, 5);
        nh.param<double>("update/odom_chi2_multipler", odom_options.chi2_multipler, 1);
        nh.param<double>("update/plane_chi2_multipler_loc", plane_options.chi2_multipler_loc, 1);
        nh.param<double>("update/plane_chi2_multipler_prior", plane_options.chi2_multipler_prior, 1);
        nh.param<double>("update/plane_distance_weight_loc", plane_options.distance_weight_loc, 0.02);
        nh.param<double>("update/plane_distance_weight_prior", plane_options.distance_weight_prior, 0.03);
        nh.param<double>("update/plane_distance_threshold_loc", plane_options.distance_thresh_loc, 1.5);
        nh.param<double>("update/plane_distance_threshold_prior", plane_options.distance_thresh_prior, 2.5);
        msckf_options.sigma_pix_sq = std::pow(msckf_options.sigma_pix, 2);
        slam_options.sigma_pix_sq = std::pow(slam_options.sigma_pix, 2);
        map_options.sigma_pix_sq = std::pow(map_options.sigma_pix, 2);
        plane_options.distance_weight_loc_sq = std::pow(plane_options.distance_weight_loc, 2);
        plane_options.distance_weight_prior_sq = std::pow(plane_options.distance_weight_prior, 2);

        nh.param<double>("match/z_th", match_options.z_th, 55);
        nh.param<double>("match/update_z_th", match_options.update_z_th, 75);
        nh.param<int>("match/extend", match_options.extend, 20);
        nh.param<int>("match/large_extend", match_options.large_extend, 30);
        nh.param<int>("match/grey_th", match_options.grey_th, 248);
        nh.param<int>("match/grey_th_low", match_options.grey_th_low, 240);
        nh.param<double>("match/alpha", match_options.alpha, 0.5);
        nh.param<double>("match/ang_th_dl", match_options.ang_th_dl, 0.9998);
        nh.param<double>("match/ang_th_bi", match_options.ang_th_bi, 0.9998);
        nh.param<int>("match/large_off", match_options.large_off, 30);
        nh.param<bool>("match/dl_filter", match_options.dl_filter, false);
        nh.param<bool>("match/bi_filter", match_options.bi_filter, true);
        nh.param<bool>("match/remain_match", match_options.remain_match, true);

        nh.param<double>("init/viwo_init/init_window_time", init_options.init_window_time, 1.0);
        nh.param<double>("init/viwo_init/init_imu_thresh", init_options.init_imu_thresh, 1.0);
        nh.param<double>("init/viwo_init/gravity_mag", init_options.gravity_mag, 9.81);
        nh.param<int>("init/viwo_init/init_max_features", init_options.init_max_features, 50);
        nh.param<bool>("init/map_init/need_init_in_map", init_options.need_init_in_map, true);
        nh.param<double>("init/map_init/search_dist_scope1", init_options.search_dist_scope1, 45);
        nh.param<double>("init/map_init/search_dist_scope2", init_options.search_dist_scope2, 75);
        nh.param<double>("init/map_init/sine_th", init_options.sine_th, 0.5);
        nh.param<bool>("init/map_init/init_use_binary", init_options.init_use_binary, true);
        nh.param<int>("init/map_init/init_grey_threshold", init_options.init_grey_thresh, 252);
        nh.param<int>("init/map_init/init_grey_threshold_low", init_options.init_grey_thresh_low, 245);
        nh.param<double>("init/map_init/outlier_score", init_options.outlier_score, 5);
        nh.param<double>("init/map_init/max_st_dist", init_options.max_st_dist, 65);
        nh.param<int>("init/map_init/preset_outliers", init_options.preset_outliers, 0);
        nh.param<int>("init/map_init/top_tuples_in_region", init_options.top_tuples_in_region, 150);
        nh.param<int>("init/map_init/top_poses", init_options.top_poses, 50);
        nh.param<int>("init/map_init/expansion_proj", init_options.expansion_proj, 20);
        nh.param<double>("init/map_init/subscore_weight", init_options.subscore_weight, 25);
        nh.param<bool>("init/map_init/prior_init_available", init_options.prior_init_available, false);
        nh.param<double>("init/map_init/prior_scale", init_options.prior_scale, 10.0);
        vector<double> prior_init;
        nh.param<vector<double>>("init/map_init/prior_init", prior_init, vector<double>{0, 0, 0});
        init_options.prior_init = Eigen::Vector3d(prior_init[0], prior_init[1], prior_init[2]);
        if (!init_options.need_init_in_map){
            vector<double> Rwi_vec;
            vector<double> pwi_vec;
            nh.param<vector<double>>("init/map_init/preset_Rwi", Rwi_vec, vector<double>{0, 0, 0, 1});
            nh.param<vector<double>>("init/map_init/preset_pwi", pwi_vec, vector<double>{0, 0, 0});

            assert(Rwi_vec.size() == 9 || Rwi_vec.size() == 4);
            assert(pwi_vec.size() == 3);
            
            if (Rwi_vec.size() == 9)
                init_options.preset_Rwi = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(Rwi_vec.data(), 3, 3).transpose();
            else{
                Eigen::Quaterniond quat(Rwi_vec[3], Rwi_vec[0], Rwi_vec[1], Rwi_vec[2]);
                init_options.preset_Rwi = quat.toRotationMatrix();
            }
            init_options.preset_pwi = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(pwi_vec.data(), 3, 1);
        }

        nh.param<bool>("feat_init/triangulate_1d", feat_init_options.triangulate_1d, false);
        nh.param<bool>("feat_init/refine_features", feat_init_options.refine_features, true);
        nh.param<int>("feat_init/max_runs", feat_init_options.max_runs, 5);
        nh.param<double>("feat_init/init_lamda", feat_init_options.init_lamda, 1e-3);
        nh.param<double>("feat_init/max_lamda", feat_init_options.max_lamda, 1e10);
        nh.param<double>("feat_init/min_dx", feat_init_options.min_dx, 1e-6);
        nh.param<double>("feat_init/min_dcost", feat_init_options.min_dcost, 1e-6);
        nh.param<double>("feat_init/lam_mult", feat_init_options.lam_mult, 10);
        nh.param<double>("feat_init/min_dist", feat_init_options.min_dist, 0.1);
        nh.param<double>("feat_init/max_dist", feat_init_options.max_dist, 60);
        nh.param<double>("feat_init/max_baseline", feat_init_options.max_baseline, 40);
        nh.param<double>("feat_init/max_cond_number", feat_init_options.max_cond_number, 10000);

        nh.param<float>("tracking_recover/dist_th", tracking_recover_options.dist_th, 5);
        nh.param<float>("tracking_recover/ang_th", tracking_recover_options.ang_th, 2);
        nh.param<float>("tracking_recover/lost_dist_th", tracking_recover_options.lost_dist_th, 10);
        nh.param<float>("tracking_recover/lost_ang_th", tracking_recover_options.lost_ang_th, 3);
        nh.param<double>("tracking_recover/z_th", tracking_recover_options.z_th, 35);
        nh.param<int>("tracking_recover/grey_thresh_low", tracking_recover_options.grey_thresh_low, 230);
        nh.param<int>("tracking_recover/extend", tracking_recover_options.extend, 40);
        nh.param<int>("tracking_recover/expansion_proj", tracking_recover_options.expansion_proj, 40);
        nh.param<float>("tracking_recover/subscore_weight", tracking_recover_options.subscore_weight, 1.5);
        nh.param<float>("tracking_recover/variance", tracking_recover_options.variance, 64);
        nh.param<double>("tracking_recover/search_dist_scope", tracking_recover_options.search_dist_scope, 45);
        nh.param<int>("tracking_recover/search_bar_height", tracking_recover_options.search_bar_height, 20);
        nh.param<float>("tracking_recover/expected_weight", tracking_recover_options.expected_weight, 0.2);
        nh.param<int>("tracking_recover/area_th", tracking_recover_options.area_th, 100);
        nh.param<float>("tracking_recover/outlier_cost", tracking_recover_options.outlier_cost, 0.5);
        nh.param<double>("tracking_recover/chi2_multipler_dl_maylost", tracking_recover_options.chi2_multipler_dl_maylost, 16);
        nh.param<double>("tracking_recover/chi2_multipler_bi_maylost", tracking_recover_options.chi2_multipler_bi_maylost, 16);
        nh.param<double>("tracking_recover/chi2_multipler_dl_lost", tracking_recover_options.chi2_multipler_dl_lost, 4);
        nh.param<double>("tracking_recover/chi2_multipler_bi_lost", tracking_recover_options.chi2_multipler_bi_lost, 4);
        nh.param<float>("tracking_recover/sigma_pix", tracking_recover_options.sigma_pix_sq, 1);
        nh.param<float>("tracking_recover/prior_z_diff_th", tracking_recover_options.prior_z_diff_th, 0.2);
        tracking_recover_options.sigma_pix_sq *= tracking_recover_options.sigma_pix_sq;

        nh.param<double>("camera/cam_fx", cam_fx, 910.777);
        nh.param<double>("camera/cam_fy", cam_fy, 910.656);
        nh.param<double>("camera/cam_cx", cam_cx, 639.846);
        nh.param<double>("camera/cam_cy", cam_cy, 355.401);
        nh.param<int>("camera/res_x", res_x, 1280);
        nh.param<int>("camera/res_y", res_y, 720);
        nh.param<vector<double>>("camera/distortion", dist, vector<double>{0, 0, 0, 0});
        Eigen::VectorXd cam_calib(8);
        cam_calib << cam_fx, cam_fy, cam_cx, cam_cy, dist[0], dist[1], dist[2], dist[3];
        nh.param<string>("camera/dist_model", dist_model, "radtan");
        if (dist_model == "equidistant") {
            camera_intrinsic = make_shared<CamEqui>(res_x, res_y);
            camera_intrinsic->set_value(cam_calib);
        } else {
            camera_intrinsic = make_shared<CamRadtan>(res_x, res_y);
            camera_intrinsic->set_value(cam_calib);
        }

        if (!nh.getParam("imu/gyro_std", noises.sigma_w) || !nh.getParam("imu/accel_std", noises.sigma_a) ||
            !nh.getParam("odom/vel_std", odom_options.sigma_odom) || !nh.getParam("imu/gyro_bias_std", noises.sigma_wb) ||
            !nh.getParam("imu/accel_bias_std", noises.sigma_ab)) {
            PRINT_ERROR(RED "[read-params]: Failed to get standard variance of imu and odom!\n" RESET);
            std::exit(EXIT_FAILURE);
        }
        noises.sigma_w.size() == 1 ? noises.sigma_w_eig = noises.sigma_w[0] * Eigen::Matrix3d::Identity()
                                   : noises.sigma_w_eig = Eigen::Vector3d(noises.sigma_w[0], noises.sigma_w[1], noises.sigma_w[2]).asDiagonal();
        noises.sigma_wb.size() == 1 ? noises.sigma_wb_eig = noises.sigma_wb[0] * Eigen::Matrix3d::Identity()
                                    : noises.sigma_wb_eig = Eigen::Vector3d(noises.sigma_wb[0], noises.sigma_wb[1], noises.sigma_wb[2]).asDiagonal();
        noises.sigma_a.size() == 1 ? noises.sigma_a_eig = noises.sigma_a[0] * Eigen::Matrix3d::Identity()
                                   : noises.sigma_a_eig = Eigen::Vector3d(noises.sigma_a[0], noises.sigma_a[1], noises.sigma_a[2]).asDiagonal();
        noises.sigma_ab.size() == 1 ? noises.sigma_ab_eig = noises.sigma_ab[0] * Eigen::Matrix3d::Identity()
                                    : noises.sigma_ab_eig = Eigen::Vector3d(noises.sigma_ab[0], noises.sigma_ab[1], noises.sigma_ab[2]).asDiagonal();
        noises.sigma_w_2_eig = noises.sigma_w_eig * noises.sigma_w_eig;
        noises.sigma_a_2_eig = noises.sigma_a_eig * noises.sigma_a_eig;
        noises.sigma_wb_2_eig = noises.sigma_wb_eig * noises.sigma_wb_eig;
        noises.sigma_ab_2_eig = noises.sigma_ab_eig * noises.sigma_ab_eig;

        odom_options.sigma_odom.size() == 1
            ? odom_options.odom_cov_eig = odom_options.sigma_odom[0] * Eigen::Matrix3d::Identity()
            : odom_options.odom_cov_eig =
                  Eigen::Vector3d(odom_options.sigma_odom[0], odom_options.sigma_odom[1], odom_options.sigma_odom[2]).asDiagonal();
        odom_options.odom_cov_eig = odom_options.odom_cov_eig * odom_options.odom_cov_eig;

        vector<double> Rci_vec;
        vector<double> pci_vec;
        vector<double> Roi_vec;
        vector<double> poi_vec;
        nh.param<vector<double>>("imu/Rci", Rci_vec, vector<double>());
        nh.param<vector<double>>("imu/pci", pci_vec, vector<double>());
        nh.param<vector<double>>("imu/Roi", Roi_vec, vector<double>());
        nh.param<vector<double>>("imu/poi", poi_vec, vector<double>(3, 0));
        assert(Rci_vec.size() == 9);
        assert(pci_vec.size() == 3);
        assert(Roi_vec.size() == 9);
        assert(poi_vec.size() == 3);
        Rci = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(Rci_vec.data(), 3, 3).transpose();
        pci = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(pci_vec.data(), 3, 1);
        Roi = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(Roi_vec.data(), 3, 3).transpose();
        poi = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(poi_vec.data(), 3, 1);
        camera_imu_extrinsic = Eigen::Matrix4d::Identity();
        camera_imu_extrinsic.block(0, 0, 3, 3) = Rci;
        camera_imu_extrinsic.block(0, 3, 3, 1) = pci;
        odom_imu_extrinsic = Eigen::Matrix4d::Identity();
        odom_imu_extrinsic.block(0, 0, 3, 3) = Roi;
        odom_imu_extrinsic.block(0, 3, 3, 1) = poi;

        nh.param<double>("imu/accel_norm", accel_norm, 9.81);
        nh.param<double>("imu/gravity_mag", gravity_mag, 9.81);
        init_options.gravity_mag = gravity_mag;

        int method;
        nh.param<int>("tracker/histogram_method", method, 1);
        histogram_method = HistogramMethod(method);

        nh.param<int>("tracker/num_features", num_features, 150);
        nh.param<int>("tracker/threshold", fast_threshold, 20);
        nh.param<int>("tracker/grid_x", grid_x, 5);
        nh.param<int>("tracker/grid_y", grid_y, 5);
        nh.param<int>("tracker/min_px_dist", min_px_dist, 10);
        nh.param<int>("tracker/pyr_levels", pyr_levels, 5);
        nh.param<int>("tracker/win_size", win_size, 15);

        nh.param<bool>("zupt/try_zupt", try_zupt, true);
        nh.param<double>("zupt/zupt_max_velocity", zupt_max_velocity, 0.5);
        nh.param<double>("zupt/zupt_noise_multiplier", zupt_noise_multiplier, 10);
        nh.param<double>("zupt/zupt_max_disparity", zupt_max_disparity, 0.5);
        nh.param<bool>("zupt/zupt_only_at_beginning", zupt_only_at_beginning, false);
        nh.param<double>("zupt/zupt_max_velocity_odom", zupt_max_velocity_odom, 0.05);

        nh.param<bool>("save/save_total_state", save_total_state, false);
        nh.param<bool>("save/save_time_consume", save_time_consume, false);
        nh.param<string>("save/of_state_est", of_state_est, "");
        nh.param<string>("save/of_state_std", of_state_std, "");
        nh.param<string>("save/of_state_tum_loc", of_state_tum_loc, "");
        nh.param<string>("save/of_state_tum_global", of_state_tum_global, "");
        nh.param<string>("save/of_state_tracking_recover", of_state_tracking_recover, "");
        of_state_est = root_dir + of_state_est;
        of_state_std = root_dir + of_state_std;
        of_state_tum_loc = root_dir + of_state_tum_loc;
        of_state_tum_global = root_dir + of_state_tum_global;
        of_state_tracking_recover = root_dir + of_state_tracking_recover;
    }

    string root_dir;

    string pcd_path, prior_pose_path, downsampled_pose_path, model_path, virtual_center_path, ground_truth_path;
    bool use_virtual_center, display_ground_truth;
    float search_scope;

    string img_topic;
    string imu_topic;
    string odom_topic;
    string box_topic;

    /// Update options for MSCKF features (pixel noise and chi2 multiplier)
    UpdaterOptions msckf_options;

    /// Update options for SLAM features (pixel noise and chi2 multiplier)
    UpdaterOptions slam_options;

    /// Update options for zero velocity (chi2 multiplier)
    UpdaterOptions zupt_options;

    /// Update options for streetlight features (pixel noise and chi2 multiplier)
    UpdaterMapOptions map_options;

    UpdaterOdomOptions odom_options;

    UpdaterPlaneOptions plane_options;

    StateOptions state_options;

    InertialInitializerOptions init_options;

    FeatureInitializerOptions feat_init_options;

    StreetlightMatchOptions match_options;

    TrackingRecoverOptions tracking_recover_options;

    // Parameters of relocalization =======================================
    double alpha1;
    double alpha2;
    double beta2;
    double delta_Rbb_th;
    double delta_pbb_th;
    double delta_Rbb_th2;
    double delta_pbb_th2;

    // Parameters of Intrinsics and Extrinsics ============================
    shared_ptr<CamBase> camera_intrinsic;
    Eigen::Matrix4d camera_imu_extrinsic;
    Eigen::Matrix4d odom_imu_extrinsic;
    double cam_fx;
    double cam_fy;
    double cam_cx;
    double cam_cy;
    vector<double> dist;
    string dist_model;
    int res_x;
    int res_y;

    Eigen::Vector3d gyro_std;
    Eigen::Vector3d accel_std;
    Eigen::Vector3d gyro_bias_std;
    Eigen::Vector3d accel_bias_std;
    NoiseManager noises;

    Eigen::Matrix3d Rci;
    Eigen::Vector3d pci;
    Eigen::Matrix3d Roi;
    Eigen::Vector3d poi;

    double gravity_mag;
    double accel_norm;

    // Parameters of feature tracking =====================================
    HistogramMethod histogram_method;
    int num_features;
    int fast_threshold;
    int grid_x;
    int grid_y;
    int min_px_dist;
    int pyr_levels;
    int win_size;

    // Parameters of Zero Velocity Updater ================================
    bool try_zupt;
    double zupt_max_velocity;
    double zupt_noise_multiplier;
    double zupt_max_disparity;
    double zupt_max_velocity_odom;
    bool zupt_only_at_beginning;

    // Parameters of saving ===============================================
    bool save_total_state, save_center_diff, save_time_consume;
    bool do_update_msckf, do_update_slam, do_update_map, do_update_odom, do_update_plane, use_match_extension;
    string of_state_est;
    string of_state_std;
    string of_state_tum_loc;
    string of_state_tum_global;
    string of_state_tracking_recover;
};
} // namespace night_voyager

#endif