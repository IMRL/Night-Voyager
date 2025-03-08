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
#ifndef PRIOR_POSE_MANAGER_H
#define PRIOR_POSE_MANAGER_H

#include "core/NightVoyagerOptions.h"
#include "utils/Print.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <map>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <random>
#include <string>
#include <vector>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

using namespace std;

namespace night_voyager {

typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> v_vec3d;
typedef vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> v_quatd;

class PriorPoseManager {
  public:
    PriorPoseManager(const NightVoyagerOptions &options) : options_(options) {
        prior_2dpose_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXY>>();
        prior_pose_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        near_prior_pose_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

        downsampled_pose_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXY>>();
        readDownSampledPose(options.downsampled_pose_path);
        readPriorPose(options.prior_pose_path);

        kdtree.setInputCloud(prior_pose_cloud);

        for (size_t i = 0; i < prior_positions.size(); ++i) {
            visualization_msgs::Marker plane_marker;
            plane_marker.header.frame_id = "global";
            plane_marker.id = i;
            plane_marker.type = visualization_msgs::Marker::CUBE;
            plane_marker.action = visualization_msgs::Marker::ADD;
            plane_marker.pose.position.x = prior_positions[i].x();
            plane_marker.pose.position.y = prior_positions[i].y();
            plane_marker.pose.position.z = prior_positions[i].z();
            Eigen::Matrix3d Rio = options.Roi.transpose();
            Eigen::Quaterniond plane_marker_rot = Eigen::Quaterniond(prior_rotations[i].toRotationMatrix() * Rio);
            plane_marker.pose.orientation.w = plane_marker_rot.w();
            plane_marker.pose.orientation.x = plane_marker_rot.x();
            plane_marker.pose.orientation.y = plane_marker_rot.y();
            plane_marker.pose.orientation.z = plane_marker_rot.z();

            plane_marker.scale.x = 0.2;
            plane_marker.scale.y = 0.2;
            plane_marker.scale.z = 0.001;
            plane_marker.color.a = 0.7;
            plane_marker.color.r = 1.0;
            plane_marker.color.g = 1.0;
            plane_marker.color.b = 1.0;
            plane_marker_array.markers.push_back(plane_marker);
        }

        // float radius = 75;
        // for (size_t i = 0; i < downsampled_pose_cloud->points.size(); ++i){
        //     visualization_msgs::Marker circle_marker;
        //     circle_marker.header.frame_id = "global";
        //     circle_marker.id = i;
        //     circle_marker.type = visualization_msgs::Marker::LINE_STRIP;
        //     circle_marker.action = visualization_msgs::Marker::ADD;

        //     circle_marker.scale.x = 1.0;
        //     // circle_marker.scale.y = 1.0;
        //     // circle_marker.scale.z = 1.0;
        //     circle_marker.color.r = 0.612;
        //     circle_marker.color.g = 0.769;
        //     circle_marker.color.b = 0.902;
        //     circle_marker.color.a = 0.5;

        //     circle_marker.pose.orientation.w = 1.0;

        //     for (int j = 0; j <= 360; ++j) {
        //         float angle = j * M_PI / 180.0;
        //         float x = radius * cos(angle) + downsampled_pose_cloud->points[i].x;
        //         float y = radius * sin(angle) + downsampled_pose_cloud->points[i].y;
        //         geometry_msgs::Point p;
        //         p.x = x;
        //         p.y = y;
        //         p.z = 0;
        //         circle_marker.points.push_back(p);
        //     }
        //     circle_marker_array.markers.push_back(circle_marker);
        // }
    }

    void readDownSampledPose(const string &downsampled_pose_path) {
        ifstream downsampledPoseFile;
        downsampledPoseFile.open(downsampled_pose_path.c_str());
        if (!downsampledPoseFile.is_open()) {
            PRINT_ERROR(RED "[read-prior-pose]: Cannot open prior_pose file %s!\n" RESET, downsampled_pose_path.c_str());
            exit(EXIT_FAILURE);
        }

        while (!downsampledPoseFile.eof()) {
            string line;
            getline(downsampledPoseFile, line);
            if (!line.empty()) {
                stringstream ss;
                ss << line;

                pcl::PointXY point;
                ss >> point.x;
                ss >> point.y;
                // ss >> point.z;

                downsampled_pose_cloud->push_back(point);
            }
        }
    }

    void readPriorPose(const string &prior_pose_path) {
        ifstream priorPoseFile;
        priorPoseFile.open(prior_pose_path.c_str());
        if (!priorPoseFile.is_open()) {
            PRINT_ERROR(RED "[read-prior-pose]: Cannot open prior_pose file %s!\n" RESET, prior_pose_path.c_str());
            exit(EXIT_FAILURE);
        }

        std::normal_distribution<double> w(0, 1);
        while (!priorPoseFile.eof()) {
            string line;
            getline(priorPoseFile, line);
            if (!line.empty()) {
                stringstream ss;
                ss << line;

                pcl::PointXYZ point3d;
                ss >> point3d.x;
                ss >> point3d.y;
                ss >> point3d.z;

                pcl::PointXY point2d;
                Eigen::Vector3d prior_pos;
                prior_pos.x() = point3d.x;
                prior_pos.y() = point3d.y;
                prior_pos.z() = point3d.z;
                point2d.x = point3d.x;
                point2d.y = point3d.y;

                Eigen::Quaterniond prior_rot;
                ss >> prior_rot.x();
                ss >> prior_rot.y();
                ss >> prior_rot.z();
                ss >> prior_rot.w();

                prior_pose_cloud->push_back(point3d);
                prior_2dpose_cloud->push_back(point2d);
                prior_positions.push_back(prior_pos);
                prior_rotations.push_back(prior_rot);
            }
            v_vec3d(prior_positions).swap(prior_positions);
            v_quatd(prior_rotations).swap(prior_rotations);
            assert(prior_positions.size() == prior_rotations.size());
            assert(prior_positions.size() == prior_pose_cloud->size());
        }
    }

    void radiusSearch(const Eigen::Vector3d &pos, v_vec3d &prior_poss, v_quatd &prior_quats, std::vector<int> &indices, float search_scope) {
        std::vector<float> distances;

        pcl::PointXYZ point;
        point.x = pos.x();
        point.y = pos.y();
        point.z = pos.z();
        kdtree.radiusSearch(point, search_scope, indices, distances);

        for (const auto &index : indices) {
            prior_poss.push_back(prior_positions[index]);
            prior_quats.push_back(prior_rotations[index]);
        }
    }

    void knnSearch(const Eigen::Vector3d &pos, v_vec3d &prior_poss, v_quatd &prior_quats, std::vector<int> &indices, std::vector<float> &distances,
                   int search_num) {

        pcl::PointXYZ point;
        point.x = pos.x();
        point.y = pos.y();
        point.z = pos.z();
        kdtree.nearestKSearch(point, search_num, indices, distances);

        for (const auto &index : indices) {
            prior_poss.push_back(prior_positions[index]);
            prior_quats.push_back(prior_rotations[index]);
        }
    }

    void addNearPriorPose(const std::map<int, bool> &near_indices_to_store) {

        near_prior_pose_cloud->clear();
        for (const auto &pair : near_indices_to_store) {
            if (!pair.second)
                continue;

            pcl::PointXYZRGB pt;
            pt.x = prior_positions[pair.first].x();
            pt.y = prior_positions[pair.first].y();
            pt.z = prior_positions[pair.first].z();
            pt.r = 255, pt.g = 0, pt.b = 0;
            near_prior_pose_cloud->push_back(pt);
        }
    }

    pcl::PointCloud<pcl::PointXY>::Ptr prior_2dpose_cloud;
    pcl::PointCloud<pcl::PointXY>::Ptr downsampled_pose_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr prior_pose_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr near_prior_pose_cloud;
    v_vec3d prior_positions;
    v_quatd prior_rotations;

    // For ros visualization
    visualization_msgs::MarkerArray plane_marker_array;
    visualization_msgs::MarkerArray circle_marker_array;

  private:
    NightVoyagerOptions options_;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    std::mt19937 gen_prior_pose;
};
} // namespace night_voyager

#endif