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
#ifndef P3PSOLVER_H
#define P3PSOLVER_H

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace night_voyager {

class P3PSolver {
  public:
    int p3p_ding(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X, std::vector<Eigen::Matrix3d> &output_R,
                 std::vector<Eigen::Vector3d> &output_T);

  private:
    double cubic_cardano_solution(const double beta, const double G, const double k2);

    double cubic_trigonometric_solution(const double alpha, const double beta, const double k2);

    std::array<Eigen::Vector3d, 2> compute_pq(const double s, const double a, const double b, const double m12, const double m13, const double m23);

    std::pair<int, std::array<double, 2>> compute_line_conic_intersection(Eigen::Vector3d &l, const double b, const double m13, const double m23);
    
    void compute_pose(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X, const double a12, const double a13,
                      const double a23, const double m12, const double m13, const double m23, const double x_root, const double y_root,
                      std::vector<Eigen::Matrix3d> &output_R, std::vector<Eigen::Vector3d> &output_T);

    // Performs a few newton steps on the equations
    inline void refine_lambda(double &lambda1, double &lambda2, double &lambda3, const double a12, const double a13, const double a23,
                              const double b12, const double b13, const double b23) {

        for (int iter = 0; iter < 5; ++iter) {
            double r1 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda2 * b12 + lambda2 * lambda2 - a12);
            double r2 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda3 * b13 + lambda3 * lambda3 - a13);
            double r3 = (lambda2 * lambda2 - 2.0 * lambda2 * lambda3 * b23 + lambda3 * lambda3 - a23);
            if (std::abs(r1) + std::abs(r2) + std::abs(r3) < 1e-10)
                return;
            double x11 = lambda1 - lambda2 * b12;
            double x12 = lambda2 - lambda1 * b12;
            double x21 = lambda1 - lambda3 * b13;
            double x23 = lambda3 - lambda1 * b13;
            double x32 = lambda2 - lambda3 * b23;
            double x33 = lambda3 - lambda2 * b23;
            double detJ = 0.5 / (x11 * x23 * x32 + x12 * x21 * x33); // half minus inverse determinant
            // This uses the closed form of the inverse for the jacobean.
            // Due to the zero elements this actually becomes quite nice.
            lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
            lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
            lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
        }
    }
};

} // namespace night_voyager

#endif