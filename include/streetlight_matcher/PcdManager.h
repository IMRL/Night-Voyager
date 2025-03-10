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
#ifndef PCDMANAGER_H
#define PCDMANAGER_H

#include "core/NightVoyagerOptions.h"
#include "utils/Print.h"
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

using namespace std;

namespace night_voyager {

typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> v_vec3d;
typedef vector<v_vec3d, Eigen::aligned_allocator<v_vec3d>> vv_vec3d;

class PcdManager {
  public:
    PcdManager(const NightVoyagerOptions &options) {
        streetlight_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        streetlight2d_center_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXY>>();
        valid_streetlight2d_center_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXY>>();
        matched_streetlight_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

        readPcd(options.pcd_path);
        streetlight2d_center_cloud->resize(cluster_num);
        points.resize(cluster_num);
        center_points.resize(cluster_num);

        for (const auto &point : streetlight_cloud->points) {
            // cout << point.intensity;
            points[int(point.intensity)].push_back(Eigen::Vector3d(point.x, point.y, point.z));
        }

        if (options.use_virtual_center) {
            readVirtualCenters(options.virtual_center_path);

            geometric_center_points.resize(cluster_num);
            for (int i = 0; i < cluster_num; i++) {
                Eigen::Vector3d sum_points = Eigen::Vector3d::Zero();
                for (size_t j = 0; j < points[i].size(); j++) {
                    sum_points += points[i][j];
                }
                geometric_center_points[i] = sum_points / points[i].size();
            }
        } else {
            for (int i = 0; i < cluster_num; i++) {
                Eigen::Vector3d sum_points = Eigen::Vector3d::Zero();
                for (size_t j = 0; j < points[i].size(); j++) {
                    sum_points += points[i][j];
                }
                center_points[i] = sum_points / points[i].size();
            }
        }

        for (int i = 0; i < cluster_num; i++) {
            pcl::PointXY pt_pcl;
            pt_pcl.x = center_points[i].x(), pt_pcl.y = center_points[i].y(), streetlight2d_center_cloud->at(i) = pt_pcl;
            if (!points[i].empty()) {
                valid_streetlight2d_center_cloud->push_back(pt_pcl);
                valid_indices2d.push_back(i);
            }
        }

        kdtree.setInputCloud(valid_streetlight2d_center_cloud);
        pcl::toROSMsg(*streetlight_cloud, streetlight_cloud_ros);
    }

    void readPcd(const string &pcd_path) {
        ifstream pcdFile;
        pcdFile.open(pcd_path.c_str());
        if (!pcdFile.is_open()) {
            PRINT_ERROR(RED "[read-pcd]: Cannot open pcd file %s!\n" RESET, pcd_path.c_str());
            exit(EXIT_FAILURE);
        }

        cluster_num = 0;
        while (!pcdFile.eof()) {
            string line;
            getline(pcdFile, line);
            if (!line.empty()) {
                stringstream ss;
                ss << line;
                
                double index;
                ss >> index;
                cluster_num = max(cluster_num, int(index));
            }
        }
        if (cluster_num == 0){
            PRINT_ERROR(RED "[read-pcd]: Cannot read pcd file %s!\n" RESET, pcd_path.c_str());
            exit(EXIT_FAILURE);
        }
        cluster_num += 1;

        pcdFile.clear();               
        pcdFile.seekg(0, ios::beg); 
        while (!pcdFile.eof()) {
            string line;
            getline(pcdFile, line);
            if (!line.empty()) {
                stringstream ss;
                ss << line;

                pcl::PointXYZI point;

                ss >> point.intensity;

                Eigen::Vector3d pt_imu_world;
                ss >> pt_imu_world.x();
                ss >> pt_imu_world.y();
                ss >> pt_imu_world.z();

                point.x = pt_imu_world.x();
                point.y = pt_imu_world.y();
                point.z = pt_imu_world.z();
                streetlight_cloud->push_back(point);
            }
        }
    }

    void readVirtualCenters(const string &vc_path) {
        ifstream vcFile;
        vcFile.open(vc_path.c_str());
        if (!vcFile.is_open()) {
            PRINT_ERROR(RED "[read-virtual-centers]: Cannot open virtual centers file %s!\n" RESET, vc_path.c_str());
            exit(EXIT_FAILURE);
        }

        while (!vcFile.eof()) {
            string line;
            getline(vcFile, line);
            if (!line.empty()) {
                stringstream ss;
                ss << line;

                int index;
                ss >> index;

                Eigen::Vector3d virtual_center;
                ss >> virtual_center.x();
                ss >> virtual_center.y();
                ss >> virtual_center.z();
                center_points[index] = virtual_center;
            }
        }

        for (size_t i = 0; i < center_points.size(); ++i) {
            if (points[i].empty()) {
                center_points[i].x() = std::numeric_limits<double>::quiet_NaN();
                center_points[i].y() = std::numeric_limits<double>::quiet_NaN();
                center_points[i].z() = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }

    void radiusSearch2d(const Eigen::Vector2d &pos, double search_scope, std::vector<int> &indices, std::vector<float> &distances) {

        std::vector<int> tmp_indices;
        pcl::PointXY point;
        point.x = pos.x();
        point.y = pos.y();
        kdtree.radiusSearch(point, search_scope, tmp_indices, distances);

        indices.resize(tmp_indices.size());
        for (size_t i = 0; i < tmp_indices.size(); ++i) {
            indices[i] = valid_indices2d[tmp_indices[i]];
        }
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr streetlight_cloud;
    pcl::PointCloud<pcl::PointXY>::Ptr streetlight2d_center_cloud;
    pcl::KdTreeFLANN<pcl::PointXY> kdtree;
    sensor_msgs::PointCloud2 streetlight_cloud_ros;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr matched_streetlight_cloud;
    v_vec3d center_points;
    v_vec3d geometric_center_points;
    vv_vec3d points;
    vector<vector<double>> params;

  private:
    int cluster_num;
    vector<int> valid_indices2d;
    pcl::PointCloud<pcl::PointXY>::Ptr valid_streetlight2d_center_cloud;
};
} // namespace night_voyager
#endif