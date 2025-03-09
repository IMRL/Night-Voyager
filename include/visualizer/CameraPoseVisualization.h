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
#ifndef CAMERA_POSE_VISUALIZATION_H
#define CAMERA_POSE_VISUALIZATION_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace night_voyager {

class CameraPoseVisualization {
  public:
    CameraPoseVisualization() : m_scale(2), m_line_width(0.4) {
        m_image_boundary_color.r = 1;
        m_image_boundary_color.g = 1;
        m_image_boundary_color.b = 1;
        m_image_boundary_color.a = 1;
        m_optical_center_connector_color.r = 1;
        m_optical_center_connector_color.g = 1;
        m_optical_center_connector_color.b = 1;
        m_optical_center_connector_color.a = 1;
    }

    CameraPoseVisualization(float r, float g, float b, float a) : m_scale(1), m_line_width(0.2) {
        m_image_boundary_color.r = r;
        m_image_boundary_color.g = g;
        m_image_boundary_color.b = b;
        m_image_boundary_color.a = a;
        m_optical_center_connector_color.r = r;
        m_optical_center_connector_color.g = g;
        m_optical_center_connector_color.b = b;
        m_optical_center_connector_color.a = a;
    }

    void Eigen2Point(const Eigen::Vector3d &v, geometry_msgs::Point &p);

    // void setImageBoundaryColor(float r, float g, float b, float a=1.0);
    // void setOpticalCenterConnectorColor(float r, float g, float b, float a=1.0);
    void setScale(double s) { m_scale = s; }

    void setLineWidth(double width) { m_line_width = width; }

    void setRGBA(float r, float g, float b, float a) {
        m_image_boundary_color.r = r;
        m_image_boundary_color.g = g;
        m_image_boundary_color.b = b;
        m_image_boundary_color.a = a;
        m_optical_center_connector_color.r = r;
        m_optical_center_connector_color.g = g;
        m_optical_center_connector_color.b = b;
        m_optical_center_connector_color.a = a;
    }

    void add_pose(const Eigen::Vector3d &p, const Eigen::Quaterniond &q);

    void reset() { m_markers.clear(); }

    void publish_by(ros::Publisher &pub, const double &stamp);

  private:
    std::vector<visualization_msgs::Marker> m_markers;
    std_msgs::ColorRGBA m_image_boundary_color;
    std_msgs::ColorRGBA m_optical_center_connector_color;
    double m_scale;
    double m_line_width;

    static const Eigen::Vector3d imlt;
    static const Eigen::Vector3d imlb;
    static const Eigen::Vector3d imrt;
    static const Eigen::Vector3d imrb;
    static const Eigen::Vector3d oc;
    static const Eigen::Vector3d lt0;
    static const Eigen::Vector3d lt1;
    static const Eigen::Vector3d lt2;
};
} // namespace night_voyager

#endif