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
#include "third_party/p3p_solver.h"

namespace night_voyager {

inline double P3PSolver::cubic_trigonometric_solution(const double alpha, const double beta, const double k2) {
    const double H = std::sqrt(-alpha * alpha * alpha / 27.0);
    const double I = std::sqrt(-alpha / 3.0);
    const double J = std::acos(-beta / (2.0 * H));
    const double K = std::cos(J / 3.0);
    return 2.0 * I * K - k2 / 3.0;
}

inline double P3PSolver::cubic_cardano_solution(const double beta, const double G, const double k2) {
    const double M = std::cbrt(-0.5 * beta + sqrt(G));
    const double N = -std::cbrt(0.5 * beta + sqrt(G));
    return M + N - k2 / 3.0;
}

inline std::array<Eigen::Vector3d, 2> P3PSolver::compute_pq(const double s, const double a, const double b, const double m12, const double m13, const double m23) {
    std::array<Eigen::Vector3d, 2> pq;
    Eigen::Matrix3d C, C_adj;

    C(0, 0) = -a + s * (1 - b);
    C(0, 1) = -m13 * s;
    C(0, 2) = a * m23 + b * m23 * s;
    C(1, 0) = -m13 * s;
    C(1, 1) = s + 1;
    C(1, 2) = -m12;
    C(2, 0) = a * m23 + b * m23 * s;
    C(2, 1) = -m12;
    C(2, 2) = -a - b * s + 1;

    C_adj(0, 0) = C(1, 2) * C(2, 1) - C(1, 1) * C(2, 2);
    C_adj(1, 1) = C(0, 2) * C(2, 0) - C(0, 0) * C(2, 2);
    C_adj(2, 2) = C(0, 1) * C(1, 0) - C(0, 0) * C(1, 1);
    C_adj(0, 1) = C(0, 1) * C(2, 2) - C(0, 2) * C(2, 1);
    C_adj(0, 2) = C(0, 2) * C(1, 1) - C(0, 1) * C(1, 2);
    C_adj(1, 0) = C(1, 0) * C(2, 2) - C(1, 2) * C(2, 0);
    C_adj(1, 2) = C(0, 0) * C(1, 2) - C(0, 2) * C(1, 0);
    C_adj(2, 0) = C(1, 1) * C(2, 0) - C(1, 0) * C(2, 1);
    C_adj(2, 1) = C(0, 0) * C(2, 1) - C(0, 1) * C(2, 0);

    Eigen::Vector3d v;
    if (C_adj(0, 0) > C_adj(1, 1)) {
        if (C_adj(0, 0) > C_adj(2, 2)) {
            v = C_adj.col(0) / std::sqrt(C_adj(0, 0));
        } else {
            v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
        }
    } else if (C_adj(1, 1) > C_adj(2, 2)) {
        v = C_adj.col(1) / std::sqrt(C_adj(1, 1));
    } else {
        v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
    }

    Eigen::Matrix3d D = C;
    D(0, 1) -= v(2);
    D(0, 2) += v(1);
    D(1, 2) -= v(0);
    D(1, 0) += v(2);
    D(2, 0) -= v(1);
    D(2, 1) += v(0);

    pq[0] = D.col(0);
    pq[1] = D.row(0);

    return pq;
}

inline std::pair<int, std::array<double, 2>> P3PSolver::compute_line_conic_intersection(Eigen::Vector3d &l, const double b, const double m13, const double m23) {
    std::pair<int, std::array<double, 2>> result;
    const double cxa = -b * l(1) * l(1) + l(2) * l(2);
    const double cxb = -2 * b * m23 * l(1) * l(2) - 2 * b * l(0) * l(1) - 2 * m13 * l(2) * l(2);
    const double cxc = -2 * b * m23 * l(0) * l(2) - b * l(0) * l(0) - b * l(2) * l(2) + l(2) * l(2);
    const double d = cxb * cxb - 4 * cxa * cxc;

    if (d < 0) {
        result.first = 0;
        return result;
    }

    result.second[0] = (-cxb + std::sqrt(d)) / (2.0 * cxa);
    result.second[1] = (-cxb - std::sqrt(d)) / (2.0 * cxa);
    result.first = d > 0 ? 2 : 1;
    return result;
}

void P3PSolver::compute_pose(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X, const double a12, const double a13, const double a23,
                             const double m12, const double m13, const double m23, const double x_root, const double y_root, std::vector<Eigen::Matrix3d> &output_R,
                             std::vector<Eigen::Vector3d> &output_T) {
    double d3 = std::sqrt(a13) / std::sqrt(x_root * x_root - 2 * m13 * x_root + 1);
    double d2 = y_root * d3;
    double d1 = x_root * d3;

    refine_lambda(d1, d2, d3, a12, a13, a23, m12, m13, m23);

    Eigen::Matrix3d A, B;
    A.col(0) = X[0] - X[1];
    A.col(1) = X[2] - X[0];
    A.col(2) = (X[0] - X[1]).cross(X[2] - X[0]);
    B.col(0) = d1 * x[0] - d2 * x[1];
    B.col(1) = d3 * x[2] - d1 * x[0];
    B.col(2) = B.col(0).cross(B.col(1));

    Eigen::Matrix3d R = B * A.inverse();
    Eigen::Vector3d t = d1 * x[0] - R * X[0];
    output_R.emplace_back(R.transpose());
    output_T.emplace_back(-R.transpose() * t);
}

int P3PSolver::p3p_ding(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X, std::vector<Eigen::Matrix3d> &output_R,
                        std::vector<Eigen::Vector3d> &output_T) {
    output_R.clear();
    output_T.clear();
    output_R.reserve(4);
    output_T.reserve(4);

    const double a12 = (X[0] - X[1]).squaredNorm();
    const double a13 = (X[0] - X[2]).squaredNorm();
    const double a23 = (X[1] - X[2]).squaredNorm();
    const double a = a12 / a23;
    const double b = a13 / a23;

    const double m12 = x[0].dot(x[1]);
    const double m13 = x[0].dot(x[2]);
    const double m23 = x[1].dot(x[2]);

    const double k3_inv = 1.0 / (b * (-b * m23 * m23 + b + m13 * m13 - 1));
    const double k2 = k3_inv * (-2 * a * b * m23 * m23 + 2 * a * b + a * m13 * m13 - a - b * b * m23 * m23 + b * b + 2 * b * m12 * m13 * m23 - 2 * b - m13 * m13 + 1);
    const double k1 = k3_inv * (-a * a * m23 * m23 + a * a - 2 * a * b * m23 * m23 + 2 * a * b + 2 * a * m12 * m13 * m23 - 2 * a + b * m12 * m12 - b - m12 * m12 + 1);
    const double k0 = k3_inv * (a * (-a * m23 * m23 + a + m12 * m12 - 1));
    const double alpha = k1 - 1.0 / 3.0 * k2 * k2;
    const double beta = k0 - 1.0 / 3.0 * k1 * k2 + (2.0 / 27.0) * k2 * k2 * k2;
    const double G = beta * beta / 4.0 + alpha * alpha * alpha / 27.0;

    double s;
    if (G != 0) {
        if (G < 0) {
            s = cubic_trigonometric_solution(alpha, beta, k2);
        } else {
            s = cubic_cardano_solution(beta, G, k2);
        }
    } else {
        s = -k2 / 3.0 + (alpha != 0 ? (3.0 * beta / alpha) : 0);
    }

    std::array<Eigen::Vector3d, 2> pq = compute_pq(s, a, b, m12, m13, m23);

    for (int i = 0; i < 2; i++) {
        auto [n_roots, x_roots] = compute_line_conic_intersection(pq[i], b, m13, m23);
        for (int j = 0; j < n_roots; j++) {
            const double x_root = x_roots[j];
            const double y_root = (-pq[i](0) - pq[i](1) * x_root) / pq[i](2);
            if (x_root <= 0 || y_root <= 0) {
                continue;
            }
            compute_pose(x, X, a12, a13, a23, m12, m13, m23, x_root, y_root, output_R, output_T);
        }

        if ((G == 0) | (G < 0 ? n_roots : !n_roots)) {
            continue;
        }

        break;
    }

    return output_R.size();
}

} // namespace night_voyager