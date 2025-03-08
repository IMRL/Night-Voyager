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
#ifndef COMMONLIB_H
#define COMMONLIB_H

#include "night_voyager/BoundingBoxes.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Imu.h>
#include <vector>

using namespace std;

namespace night_voyager {

struct ImuData {
    double timestamp;

    Eigen::Vector3d wm;

    Eigen::Vector3d am;

    bool operator<(const ImuData &other) const { return timestamp < other.timestamp; }
};

struct NoiseManager {

    /// Gyroscope white noise covariance
    vector<double> sigma_w;
    Eigen::Matrix3d sigma_w_eig;
    Eigen::Matrix3d sigma_w_2_eig;

    /// Gyroscope random walk covariance
    vector<double> sigma_wb;
    Eigen::Matrix3d sigma_wb_eig;
    Eigen::Matrix3d sigma_wb_2_eig;

    /// Accelerometer white noise covariance
    vector<double> sigma_a;
    Eigen::Matrix3d sigma_a_eig;
    Eigen::Matrix3d sigma_a_2_eig;

    /// Accelerometer random walk covariance
    vector<double> sigma_ab;
    Eigen::Matrix3d sigma_ab_eig;
    Eigen::Matrix3d sigma_ab_2_eig;
};

struct OdomNoiseManager {

    Eigen::Vector3d sigma_v;

    Eigen::Vector3d sigma_v_2;
};

struct CameraData {

    double timestamp;

    cv::Mat image;

    cv::Mat color_image;

    bool operator<(const CameraData &other) const { return timestamp < other.timestamp; }
};

struct OdomData {

    double timestamp;

    Eigen::Vector3d vm;

    bool operator<(const OdomData &other) const { return timestamp < other.timestamp; }
};

struct BoxData {

    double timestamp;

    std::vector<cv::Rect> rects;

    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> centers;

    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> centers_norm;

    bool operator<(const BoxData &other) const { return timestamp < other.timestamp; }
};

struct PackData {

    CameraData cam;

    vector<ImuData> imus;

    vector<OdomData> odoms;

    BoxData box;
};

struct STMatch {

    cv::Rect rect;

    Eigen::Vector2f rect_center;

    Eigen::Vector2f rect_center_norm;

    Eigen::Vector3d st_center_map;

    Eigen::Vector3d st_center_cam;

    size_t st_id;
};

struct ScoreData {

    /// index of streetlight cluster
    size_t id;

    /// position of streetlight center under current pose
    Eigen::Vector3d Pc;

    /// projection of streetlight center
    Eigen::Vector2d pt;

    /// covariance for calculating projection error-based score
    Eigen::Matrix2d HPH_R;

    /// covariance for calculating angle error-based score
    Eigen::Matrix3d liftup_HPH_R;
};

enum KFCLASS { MSCKF = -2, IKF_NOGROUP = -1, IKF_IMUGROUP = 0, IKF_RELGROUP = 1, IKF_CHANGEGROUP = 2 };

enum TrackingStatus { TRACKINGSUCCESS = 0, TRACKINGUNSTABLESUCCESS, TRACKINGMAYLOST, TRACKINGLOST };

class CamBase {

  public:
    /**
     * @brief Default constructor
     * @param width Width of the camera (raw pixels)
     * @param height Height of the camera (raw pixels)
     */
    CamBase(int width, int height) : _width(width), _height(height) {}

    virtual ~CamBase() {}

    /**
     * @brief This will set and update the camera calibration values.
     * This should be called on startup for each camera and after update!
     * @param calib Camera calibration information (f_x & f_y & c_x & c_y & k_1 & k_2 & k_3 & k_4)
     */
    virtual void set_value(const Eigen::VectorXd &calib) {

        // Assert we are of size eight
        assert(calib.rows() == 8);
        camera_values = calib;

        // Camera matrix
        cv::Matx33d tempK;
        tempK(0, 0) = calib(0);
        tempK(0, 1) = 0;
        tempK(0, 2) = calib(2);
        tempK(1, 0) = 0;
        tempK(1, 1) = calib(1);
        tempK(1, 2) = calib(3);
        tempK(2, 0) = 0;
        tempK(2, 1) = 0;
        tempK(2, 2) = 1;
        camera_k_OPENCV = tempK;
        camera_k_EIGEN << tempK(0, 0), 0, tempK(0, 2), 0, tempK(1, 1), tempK(1, 2), 0, 0, 1;

        cv::Matx33d tempinvK;
        tempinvK(0, 0) = 1.0 / calib(0);
        tempinvK(0, 1) = 0;
        tempinvK(0, 2) = -calib(2) / calib(0);
        tempinvK(1, 0) = 0;
        tempinvK(1, 1) = 1.0 / calib(1);
        tempinvK(1, 2) = -calib(3) / calib(1);
        tempinvK(2, 0) = 0;
        tempinvK(2, 1) = 0;
        tempinvK(2, 2) = 1;
        camera_inv_k_OPENCV = tempinvK;
        camera_inv_k_EIGEN << tempinvK(0, 0), 0, tempinvK(0, 2), 0, tempinvK(1, 1), tempinvK(1, 2), 0, 0, 1;

        // Distortion parameters
        cv::Vec4d tempD;
        tempD(0) = calib(4);
        tempD(1) = calib(5);
        tempD(2) = calib(6);
        tempD(3) = calib(7);
        camera_d_OPENCV = tempD;
    }

    /**
     * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
     * @param uv_dist Raw uv coordinate we wish to undistort
     * @return 2d vector of normalized coordinates
     */
    virtual Eigen::Vector2f undistort_f(const Eigen::Vector2f &uv_dist) = 0;

    /**
     * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
     * @param uv_dist Raw uv coordinate we wish to undistort
     * @return 2d vector of normalized coordinates
     */
    Eigen::Vector2d undistort_d(const Eigen::Vector2d &uv_dist) {
        Eigen::Vector2f ept1, ept2;
        ept1 = uv_dist.cast<float>();
        ept2 = undistort_f(ept1);
        return ept2.cast<double>();
    }

    /**
     * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
     * @param uv_dist Raw uv coordinate we wish to undistort
     * @return 2d vector of normalized coordinates
     */
    cv::Point2f undistort_cv(const cv::Point2f &uv_dist) {
        Eigen::Vector2f ept1, ept2;
        ept1 << uv_dist.x, uv_dist.y;
        ept2 = undistort_f(ept1);
        cv::Point2f pt_out;
        pt_out.x = ept2(0);
        pt_out.y = ept2(1);
        return pt_out;
    }

    /**
     * @brief Given a normalized uv coordinate this will distort it to the raw image plane
     * @param uv_norm Normalized coordinates we wish to distort
     * @return 2d vector of raw uv coordinate
     */
    virtual Eigen::Vector2f distort_f(const Eigen::Vector2f &uv_norm) = 0;

    /**
     * @brief Given a normalized uv coordinate this will distort it to the raw image plane
     * @param uv_norm Normalized coordinates we wish to distort
     * @return 2d vector of raw uv coordinate
     */
    Eigen::Vector2d distort_d(const Eigen::Vector2d &uv_norm) {
        Eigen::Vector2f ept1, ept2;
        ept1 = uv_norm.cast<float>();
        ept2 = distort_f(ept1);
        return ept2.cast<double>();
    }

    /**
     * @brief Given a normalized uv coordinate this will distort it to the raw image plane
     * @param uv_norm Normalized coordinates we wish to distort
     * @return 2d vector of raw uv coordinate
     */
    cv::Point2f distort_cv(const cv::Point2f &uv_norm) {
        Eigen::Vector2f ept1, ept2;
        ept1 << uv_norm.x, uv_norm.y;
        ept2 = distort_f(ept1);
        cv::Point2f pt_out;
        pt_out.x = ept2(0);
        pt_out.y = ept2(1);
        return pt_out;
    }

    /**
     * @brief Computes the derivative of raw distorted to normalized coordinate.
     * @param uv_norm Normalized coordinates we wish to distort
     * @param H_dz_dzn Derivative of measurement z in respect to normalized
     * @param H_dz_dzeta Derivative of measurement z in respect to intrinic parameters
     */
    virtual void compute_distort_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta) = 0;

    /// Gets the complete intrinsic vector
    Eigen::MatrixXd get_value() { return camera_values; }

    /// Gets the camera matrix
    cv::Matx33d get_K_opencv() { return camera_k_OPENCV; }

    Eigen::Matrix3d get_K_eigen() { return camera_k_EIGEN; }

    /// Gets the camera inverse matrix
    cv::Matx33d get_invK_opencv() { return camera_inv_k_OPENCV; }

    Eigen::Matrix3d get_invK_eigen() { return camera_inv_k_EIGEN; }

    /// Gets the camera distortion
    cv::Vec4d get_D() { return camera_d_OPENCV; }

    /// Gets the width of the camera images
    int w() { return _width; }

    /// Gets the height of the camera images
    int h() { return _height; }

    /// Gets the fx of the camera
    double get_fx() { return camera_k_EIGEN(0, 0); }

    /// Gets the cx of the camera
    double get_cx() { return camera_k_EIGEN(0, 2); }

    /// Gets the fy of the camera
    double get_fy() { return camera_k_EIGEN(1, 1); }

    /// Gets the cy of the camera
    double get_cy() { return camera_k_EIGEN(1, 2); }

  protected:
    // Cannot construct the base camera class, needs a distortion model
    CamBase() = default;

    /// Raw set of camera intrinic values (f_x & f_y & c_x & c_y & k_1 & k_2 & k_3 & k_4)
    Eigen::MatrixXd camera_values;

    /// Camera intrinsics in OpenCV format
    cv::Matx33d camera_k_OPENCV;
    Eigen::Matrix3d camera_k_EIGEN;

    cv::Matx33d camera_inv_k_OPENCV;
    Eigen::Matrix3d camera_inv_k_EIGEN;

    /// Camera distortion in OpenCV format
    cv::Vec4d camera_d_OPENCV;

    /// Width of the camera (raw pixels)
    int _width;

    /// Height of the camera (raw pixels)
    int _height;
};

class CamEqui : public CamBase {

  public:
    /**
     * @brief Default constructor
     * @param width Width of the camera (raw pixels)
     * @param height Height of the camera (raw pixels)
     */
    CamEqui(int width, int height) : CamBase(width, height) {}

    ~CamEqui() {}

    /**
     * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
     * @param uv_dist Raw uv coordinate we wish to undistort
     * @return 2d vector of normalized coordinates
     */
    Eigen::Vector2f undistort_f(const Eigen::Vector2f &uv_dist) override {

        // Determine what camera parameters we should use
        cv::Matx33d camK = camera_k_OPENCV;
        cv::Vec4d camD = camera_d_OPENCV;

        // Convert point to opencv format
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = uv_dist(0);
        mat.at<float>(0, 1) = uv_dist(1);
        mat = mat.reshape(2); // Nx1, 2-channel

        // Undistort it!
        cv::fisheye::undistortPoints(mat, mat, camK, camD);

        // Construct our return vector
        Eigen::Vector2f pt_out;
        mat = mat.reshape(1); // Nx2, 1-channel
        pt_out(0) = mat.at<float>(0, 0);
        pt_out(1) = mat.at<float>(0, 1);
        return pt_out;
    }

    /**
     * @brief Given a normalized uv coordinate this will distort it to the raw image plane
     * @param uv_norm Normalized coordinates we wish to distort
     * @return 2d vector of raw uv coordinate
     */
    Eigen::Vector2f distort_f(const Eigen::Vector2f &uv_norm) override {

        // Get our camera parameters
        Eigen::MatrixXd cam_d = camera_values;

        // Calculate distorted coordinates for fisheye
        double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
        double theta = std::atan(r);
        double theta_d =
            theta + cam_d(4) * std::pow(theta, 3) + cam_d(5) * std::pow(theta, 5) + cam_d(6) * std::pow(theta, 7) + cam_d(7) * std::pow(theta, 9);

        // Handle when r is small (meaning our xy is near the camera center)
        double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
        double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;

        // Calculate distorted coordinates for fisheye
        Eigen::Vector2f uv_dist;
        double x1 = uv_norm(0) * cdist;
        double y1 = uv_norm(1) * cdist;
        uv_dist(0) = (float)(cam_d(0) * x1 + cam_d(2));
        uv_dist(1) = (float)(cam_d(1) * y1 + cam_d(3));
        return uv_dist;
    }

    /**
     * @brief Computes the derivative of raw distorted to normalized coordinate.
     * @param uv_norm Normalized coordinates we wish to distort
     * @param H_dz_dzn Derivative of measurement z in respect to normalized
     * @param H_dz_dzeta Derivative of measurement z in respect to intrinic parameters
     */
    void compute_distort_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta) override {

        // Get our camera parameters
        Eigen::MatrixXd cam_d = camera_values;

        // Calculate distorted coordinates for fisheye
        double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
        double theta = std::atan(r);
        double theta_d =
            theta + cam_d(4) * std::pow(theta, 3) + cam_d(5) * std::pow(theta, 5) + cam_d(6) * std::pow(theta, 7) + cam_d(7) * std::pow(theta, 9);

        // Handle when r is small (meaning our xy is near the camera center)
        double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
        double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;

        // Jacobian of distorted pixel to "normalized" pixel
        Eigen::Matrix<double, 2, 2> duv_dxy = Eigen::Matrix<double, 2, 2>::Zero();
        duv_dxy << cam_d(0), 0, 0, cam_d(1);

        // Jacobian of "normalized" pixel to normalized pixel
        Eigen::Matrix<double, 2, 2> dxy_dxyn = Eigen::Matrix<double, 2, 2>::Zero();
        dxy_dxyn << theta_d * inv_r, 0, 0, theta_d * inv_r;

        // Jacobian of "normalized" pixel to r
        Eigen::Matrix<double, 2, 1> dxy_dr = Eigen::Matrix<double, 2, 1>::Zero();
        dxy_dr << -uv_norm(0) * theta_d * inv_r * inv_r, -uv_norm(1) * theta_d * inv_r * inv_r;

        // Jacobian of r pixel to normalized xy
        Eigen::Matrix<double, 1, 2> dr_dxyn = Eigen::Matrix<double, 1, 2>::Zero();
        dr_dxyn << uv_norm(0) * inv_r, uv_norm(1) * inv_r;

        // Jacobian of "normalized" pixel to theta_d
        Eigen::Matrix<double, 2, 1> dxy_dthd = Eigen::Matrix<double, 2, 1>::Zero();
        dxy_dthd << uv_norm(0) * inv_r, uv_norm(1) * inv_r;

        // Jacobian of theta_d to theta
        double dthd_dth = 1 + 3 * cam_d(4) * std::pow(theta, 2) + 5 * cam_d(5) * std::pow(theta, 4) + 7 * cam_d(6) * std::pow(theta, 6) +
                          9 * cam_d(7) * std::pow(theta, 8);

        // Jacobian of theta to r
        double dth_dr = 1 / (r * r + 1);

        // Total Jacobian wrt normalized pixel coordinates
        H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
        H_dz_dzn = duv_dxy * (dxy_dxyn + (dxy_dr + dxy_dthd * dthd_dth * dth_dr) * dr_dxyn);

        // Calculate distorted coordinates for fisheye
        double x1 = uv_norm(0) * cdist;
        double y1 = uv_norm(1) * cdist;

        // Compute the Jacobian in respect to the intrinsics
        H_dz_dzeta = Eigen::MatrixXd::Zero(2, 8);
        H_dz_dzeta(0, 0) = x1;
        H_dz_dzeta(0, 2) = 1;
        H_dz_dzeta(0, 4) = cam_d(0) * uv_norm(0) * inv_r * std::pow(theta, 3);
        H_dz_dzeta(0, 5) = cam_d(0) * uv_norm(0) * inv_r * std::pow(theta, 5);
        H_dz_dzeta(0, 6) = cam_d(0) * uv_norm(0) * inv_r * std::pow(theta, 7);
        H_dz_dzeta(0, 7) = cam_d(0) * uv_norm(0) * inv_r * std::pow(theta, 9);
        H_dz_dzeta(1, 1) = y1;
        H_dz_dzeta(1, 3) = 1;
        H_dz_dzeta(1, 4) = cam_d(1) * uv_norm(1) * inv_r * std::pow(theta, 3);
        H_dz_dzeta(1, 5) = cam_d(1) * uv_norm(1) * inv_r * std::pow(theta, 5);
        H_dz_dzeta(1, 6) = cam_d(1) * uv_norm(1) * inv_r * std::pow(theta, 7);
        H_dz_dzeta(1, 7) = cam_d(1) * uv_norm(1) * inv_r * std::pow(theta, 9);
    }
};

class CamRadtan : public CamBase {

  public:
    /**
     * @brief Default constructor
     * @param width Width of the camera (raw pixels)
     * @param height Height of the camera (raw pixels)
     */
    CamRadtan(int width, int height) : CamBase(width, height) {}

    ~CamRadtan() {}

    /**
     * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
     * @param uv_dist Raw uv coordinate we wish to undistort
     * @return 2d vector of normalized coordinates
     */
    Eigen::Vector2f undistort_f(const Eigen::Vector2f &uv_dist) override {

        // Determine what camera parameters we should use
        cv::Matx33d camK = camera_k_OPENCV;
        cv::Vec4d camD = camera_d_OPENCV;

        // Convert to opencv format
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = uv_dist(0);
        mat.at<float>(0, 1) = uv_dist(1);
        mat = mat.reshape(2); // Nx1, 2-channel

        // Undistort it!
        cv::undistortPoints(mat, mat, camK, camD);

        // Construct our return vector
        Eigen::Vector2f pt_out;
        mat = mat.reshape(1); // Nx2, 1-channel
        pt_out(0) = mat.at<float>(0, 0);
        pt_out(1) = mat.at<float>(0, 1);
        return pt_out;
    }

    /**
     * @brief Given a normalized uv coordinate this will distort it to the raw image plane
     * @param uv_norm Normalized coordinates we wish to distort
     * @return 2d vector of raw uv coordinate
     */
    Eigen::Vector2f distort_f(const Eigen::Vector2f &uv_norm) override {

        // Get our camera parameters
        Eigen::MatrixXd cam_d = camera_values;

        // Calculate distorted coordinates for radial
        double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
        double r_2 = r * r;
        double r_4 = r_2 * r_2;
        double x1 = uv_norm(0) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + 2 * cam_d(6) * uv_norm(0) * uv_norm(1) +
                    cam_d(7) * (r_2 + 2 * uv_norm(0) * uv_norm(0));
        double y1 = uv_norm(1) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + cam_d(6) * (r_2 + 2 * uv_norm(1) * uv_norm(1)) +
                    2 * cam_d(7) * uv_norm(0) * uv_norm(1);

        // Return the distorted point
        Eigen::Vector2f uv_dist;
        uv_dist(0) = (float)(cam_d(0) * x1 + cam_d(2));
        uv_dist(1) = (float)(cam_d(1) * y1 + cam_d(3));
        return uv_dist;
    }

    /**
     * @brief Computes the derivative of raw distorted to normalized coordinate.
     * @param uv_norm Normalized coordinates we wish to distort
     * @param H_dz_dzn Derivative of measurement z in respect to normalized
     * @param H_dz_dzeta Derivative of measurement z in respect to intrinic parameters
     */
    void compute_distort_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta) override {

        // Get our camera parameters
        Eigen::MatrixXd cam_d = camera_values;

        // Calculate distorted coordinates for radial
        double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
        double r_2 = r * r;
        double r_4 = r_2 * r_2;

        // Jacobian of distorted pixel to normalized pixel
        H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
        double x = uv_norm(0);
        double y = uv_norm(1);
        double x_2 = uv_norm(0) * uv_norm(0);
        double y_2 = uv_norm(1) * uv_norm(1);
        double x_y = uv_norm(0) * uv_norm(1);
        H_dz_dzn(0, 0) = cam_d(0) * ((1 + cam_d(4) * r_2 + cam_d(5) * r_4) + (2 * cam_d(4) * x_2 + 4 * cam_d(5) * x_2 * r_2) + 2 * cam_d(6) * y +
                                     (2 * cam_d(7) * x + 4 * cam_d(7) * x));
        H_dz_dzn(0, 1) = cam_d(0) * (2 * cam_d(4) * x_y + 4 * cam_d(5) * x_y * r_2 + 2 * cam_d(6) * x + 2 * cam_d(7) * y);
        H_dz_dzn(1, 0) = cam_d(1) * (2 * cam_d(4) * x_y + 4 * cam_d(5) * x_y * r_2 + 2 * cam_d(6) * x + 2 * cam_d(7) * y);
        H_dz_dzn(1, 1) = cam_d(1) * ((1 + cam_d(4) * r_2 + cam_d(5) * r_4) + (2 * cam_d(4) * y_2 + 4 * cam_d(5) * y_2 * r_2) + 2 * cam_d(7) * x +
                                     (2 * cam_d(6) * y + 4 * cam_d(6) * y));

        // Calculate distorted coordinates for radtan
        double x1 = uv_norm(0) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + 2 * cam_d(6) * uv_norm(0) * uv_norm(1) +
                    cam_d(7) * (r_2 + 2 * uv_norm(0) * uv_norm(0));
        double y1 = uv_norm(1) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + cam_d(6) * (r_2 + 2 * uv_norm(1) * uv_norm(1)) +
                    2 * cam_d(7) * uv_norm(0) * uv_norm(1);

        // Compute the Jacobian in respect to the intrinsics
        H_dz_dzeta = Eigen::MatrixXd::Zero(2, 8);
        H_dz_dzeta(0, 0) = x1;
        H_dz_dzeta(0, 2) = 1;
        H_dz_dzeta(0, 4) = cam_d(0) * uv_norm(0) * r_2;
        H_dz_dzeta(0, 5) = cam_d(0) * uv_norm(0) * r_4;
        H_dz_dzeta(0, 6) = 2 * cam_d(0) * uv_norm(0) * uv_norm(1);
        H_dz_dzeta(0, 7) = cam_d(0) * (r_2 + 2 * uv_norm(0) * uv_norm(0));
        H_dz_dzeta(1, 1) = y1;
        H_dz_dzeta(1, 3) = 1;
        H_dz_dzeta(1, 4) = cam_d(1) * uv_norm(1) * r_2;
        H_dz_dzeta(1, 5) = cam_d(1) * uv_norm(1) * r_4;
        H_dz_dzeta(1, 6) = cam_d(1) * (r_2 + 2 * uv_norm(1) * uv_norm(1));
        H_dz_dzeta(1, 7) = 2 * cam_d(1) * uv_norm(0) * uv_norm(1);
    }
};

/*
 * @brief Helper class to do OpenCV parallelization
 *
 * This is a utility class required to build with older version of opencv
 * On newer versions this doesn't seem to be needed, but here we just use it to ensure we can work for more opencv version.
 * https://answers.opencv.org/question/65800/how-to-use-lambda-as-a-parameter-to-parallel_for_/?answer=130691#post-id-130691
 */
class LambdaBody : public cv::ParallelLoopBody {
  public:
    explicit LambdaBody(const std::function<void(const cv::Range &)> &body) { _body = body; }
    void operator()(const cv::Range &range) const override { _body(range); }

  private:
    std::function<void(const cv::Range &)> _body;
};

enum HistogramMethod { NONE, HISTOGRAM, CLAHE };

} // namespace night_voyager
#endif