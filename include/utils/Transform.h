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
#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "core/CommonLib.h"
#include <Eigen/Core>
#include <GeographicLib/LocalCartesian.hpp>

namespace night_voyager {

const double TOLERANCE = 1e-7;

inline Eigen::Matrix3d skew(const Eigen::Vector3d &vec) {
    Eigen::Matrix3d mat;
    mat << 0, -vec.z(), vec.y(), vec.z(), 0, -vec.x(), -vec.y(), vec.x(), 0;
    return mat;
}

inline Eigen::Matrix3d Exp_SO3(const Eigen::Vector3d &dx) {

    // get theta
    Eigen::Matrix<double, 3, 3> w_x = skew(dx);
    double theta = dx.norm();
    // Handle small angle values
    double A, B;
    if (theta < TOLERANCE) {
        A = 1;
        B = 0.5;
    } else {
        A = sin(theta) / theta;
        B = (1 - cos(theta)) / (theta * theta);
    }
    // compute so(3) rotation
    Eigen::Matrix<double, 3, 3> R;
    if (theta == 0) {
        R = Eigen::MatrixXd::Identity(3, 3);
    } else {
        R = Eigen::MatrixXd::Identity(3, 3) + A * w_x + B * w_x * w_x;
    }
    return R;
}

inline Eigen::MatrixXd Exp_SEK3(const Eigen::VectorXd &v) {
    int K = (v.size() - 3) / 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Identity(3 + K, 3 + K);
    Eigen::Matrix3d R;
    Eigen::Matrix3d Jl;
    Eigen::Vector3d w = v.head(3);
    double theta = w.norm();
    double A, B, C;
    if (theta < TOLERANCE) {
        A = 1;
        B = 0.5;
        C = 1.0 / 6.0;
    } else {
        double theta2 = theta * theta;
        A = sin(theta) / theta;
        B = (1 - cos(theta)) / theta2;
        C = (1 - A) / theta2;
    }

    Eigen::Matrix3d w_x = skew(w);
    R = Eigen::MatrixXd::Identity(3, 3) + A * w_x + B * w_x * w_x;
    Jl = Eigen::MatrixXd::Identity(3, 3) + B * w_x + C * w_x * w_x;
    X.block<3, 3>(0, 0) = R;
    for (int i = 0; i < K; ++i) {
        X.block<3, 1>(0, 3 + i) = Jl * v.segment<3>(3 + 3 * i);
    }

    return X;
}

/**
 * @brief SO(3) matrix logarithm
 *
 * This definition was taken from "Lie Groups for 2D and 3D Transformations" by Ethan Eade equation 17 & 18.
 * http://ethaneade.com/lie.pdf
 * \f{align*}{
 * \theta &= \textrm{arccos}(0.5(\textrm{trace}(\mathbf{R})-1)) \\
 * \lfloor\mathbf{v}\times\rfloor &= \frac{\theta}{2\sin{\theta}}(\mathbf{R}-\mathbf{R}^\top)
 * @f}
 *
 * @param[in] R 3x3 SO(3) rotation matrix
 * @return 3x1 in the se(3) space [omegax, omegay, omegaz]
 */
inline Eigen::Matrix<double, 3, 1> LOG_SO3(const Eigen::Matrix<double, 3, 3> &R) {
    // magnitude of the skew elements (handle edge case where we sometimes have a>1...)
    double a = 0.5 * (R.trace() - 1);
    double theta = (a > 1) ? acos(1) : ((a < -1) ? acos(-1) : acos(a));
    // Handle small angle values
    double D;
    if (theta < 1e-12) {
        D = 0.5;
    } else {
        D = theta / (2 * sin(theta));
    }
    // calculate the skew symetric matrix
    Eigen::Matrix<double, 3, 3> w_x = D * (R - R.transpose());
    // check if we are near the identity
    if (R != Eigen::MatrixXd::Identity(3, 3)) {
        Eigen::Vector3d vec;
        vec << w_x(2, 1), w_x(0, 2), w_x(1, 0);
        return vec;
    } else {
        return Eigen::Vector3d::Zero();
    }
}

inline Eigen::VectorXd Log_SEK3(const Eigen::MatrixXd &mat) {
    int K = mat.rows() - 3;
    // Get sub-matrices
    Eigen::Matrix3d R = mat.block(0, 0, 3, 3);

    // Get theta (handle edge case where we sometimes have a>1...)
    double a = 0.5 * (R.trace() - 1);
    double theta = (a > 1) ? acos(1) : ((a < -1) ? acos(-1) : acos(a));
    // Handle small angle values
    double A, B, D, E;
    if (theta < TOLERANCE) {
        A = 1;
        B = 0.5;
        D = 0.5;
        E = 1.0 / 12.0;
    } else {
        A = sin(theta) / theta;
        B = (1 - cos(theta)) / (theta * theta);
        D = theta / (2 * sin(theta));
        E = 1 / (theta * theta) * (1 - 0.5 * A / B);
    }

    // Get the skew matrix and V inverse
    Eigen::Matrix3d I_33 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d wskew = D * (R - R.transpose());
    Eigen::Matrix3d Vinv = I_33 - 0.5 * wskew + E * wskew * wskew;

    // Calculate vector
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(3 + K * 3, 1);
    vec.head(3) << wskew(2, 1), wskew(0, 2), wskew(1, 0);
    for (int i = 0; i < K; ++i) {
        vec.segment(3 + i * 3, 3) = Vinv * mat.block(0, 3 + i, 3, 1);
    }
    return vec;
}

/**
 * @brief SE(3) matrix analytical inverse
 *
 * It seems that using the .inverse() function is not a good way.
 * This should be used in all cases we need the inverse instead of numerical inverse.
 * https://github.com/rpng/open_vins/issues/12
 * \f{align*}{
 * \mathbf{T}^{-1} = \begin{bmatrix} \mathbf{R}^\top & -\mathbf{R}^\top\mathbf{p} \\ \mathbf{0} & 1 \end{bmatrix}
 * \f}
 *
 * @param[in] T SE(3) matrix
 * @return inversed SE(3) matrix
 */
inline Eigen::Matrix4d Inv_SE3(const Eigen::Matrix4d &T) {
    Eigen::Matrix4d Tinv = Eigen::Matrix4d::Identity();
    Tinv.block(0, 0, 3, 3) = T.block(0, 0, 3, 3).transpose();
    Tinv.block(0, 3, 3, 1) = -Tinv.block(0, 0, 3, 3) * T.block(0, 3, 3, 1);
    return Tinv;
}

/**
 * @brief Hat operator for R^6 -> Lie Algebra se(3)
 *
 * \f{align*}{
 * \boldsymbol\Omega^{\wedge} = \begin{bmatrix} \lfloor \boldsymbol\omega \times\rfloor & \mathbf u \\ \mathbf 0 & 0 \end{bmatrix}
 * \f}
 *
 * @param vec 6x1 in the se(3) space [omega, u]
 * @return Lie algebra se(3) 4x4 matrix
 */
inline Eigen::Matrix4d Hat_SE3(const Eigen::Matrix<double, 6, 1> &vec) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
    mat.block(0, 0, 3, 3) = skew(vec.head(3));
    mat.block(0, 3, 3, 1) = vec.tail(3);
    return mat;
}

/**
 * @brief Returns vector portion of skew-symmetric
 *
 * See skew() for details.
 *
 * @param[in] w_x skew-symmetric matrix
 * @return 3x1 vector portion of skew
 */
inline Eigen::Matrix<double, 3, 1> Vee(const Eigen::Matrix<double, 3, 3> &w_x) {
    Eigen::Matrix<double, 3, 1> w;
    w << w_x(2, 1), w_x(0, 2), w_x(1, 0);
    return w;
}

/**
 * @brief Computes left Jacobian of SO(3)
 *
 * The left Jacobian of SO(3) is defined equation (7.77b) in [State Estimation for
 * Robotics](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf) by Timothy D. Barfoot @cite Barfoot2017. Specifically it is the
 * following (with \f$\theta=|\boldsymbol\theta|\f$ and \f$\mathbf a=\boldsymbol\theta/|\boldsymbol\theta|\f$): \f{align*}{
 * J_l(\boldsymbol\theta) = \frac{\sin\theta}{\theta}\mathbf I + \Big(1-\frac{\sin\theta}{\theta}\Big)\mathbf a \mathbf a^\top +
 * \frac{1-\cos\theta}{\theta}\lfloor \mathbf a \times\rfloor \f}
 *
 * @param w axis-angle
 * @return The left Jacobian of SO(3)
 */
inline Eigen::Matrix<double, 3, 3> Jl_SO3(const Eigen::Matrix<double, 3, 1> &w) {
    double theta = w.norm();
    if (theta < 1e-6) {
        return Eigen::MatrixXd::Identity(3, 3);
    } else {
        Eigen::Matrix<double, 3, 1> a = w / theta;
        Eigen::Matrix<double, 3, 3> J = sin(theta) / theta * Eigen::MatrixXd::Identity(3, 3) + (1 - sin(theta) / theta) * a * a.transpose() +
                                        ((1 - cos(theta)) / theta) * skew(a);
        return J;
    }
}

/**
 * @brief Computes right Jacobian of SO(3)
 *
 * The right Jacobian of SO(3) is related to the left by Jl(-w)=Jr(w).
 * See equation (7.87) in [State Estimation for Robotics](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf) by Timothy D. Barfoot
 * @cite Barfoot2017. See @ref Jl_so3() for the definition of the left Jacobian of SO(3).
 *
 * @param w axis-angle
 * @return The right Jacobian of SO(3)
 */
inline Eigen::Matrix<double, 3, 3> Jr_SO3(const Eigen::Matrix<double, 3, 1> &w) { return Jl_SO3(-w); }

/**
 * @brief Integrated quaternion from angular velocity
 *
 * See equation (48) of trawny tech report [Indirect Kalman Filter for 3D Attitude
 * Estimation](http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf).
 * This matrix is derived in Section 1.5 of the report by finding the Quaterion Time Derivative.
 *
 * \f{align*}{
 * \boldsymbol{\Omega}(\boldsymbol{\omega}) &=
 * \begin{bmatrix}
 * -\lfloor{\boldsymbol{\omega}}  \rfloor & \boldsymbol{\omega} \\
 * -\boldsymbol{\omega}^\top & 0
 * \end{bmatrix}
 * \f}
 *
 * @param w Angular velocity
 * @return The matrix \f$\boldsymbol{\Omega}\f$
 */
inline Eigen::Matrix<double, 4, 4> Omega(Eigen::Matrix<double, 3, 1> w) {
    Eigen::Matrix<double, 4, 4> mat;
    mat.block(0, 0, 3, 3) = -skew(w);
    mat.block(3, 0, 1, 3) = -w.transpose();
    mat.block(0, 3, 3, 1) = w;
    mat(3, 3) = 0;
    return mat;
}

/**
 * @brief Normalizes a quaternion to make sure it is unit norm
 * @param q_t Quaternion to normalized
 * @return Normalized quaterion
 */
inline Eigen::Matrix<double, 4, 1> quatnorm(Eigen::Matrix<double, 4, 1> q_t) {
    if (q_t(3, 0) < 0) {
        q_t *= -1;
    }
    return q_t / q_t.norm();
}

/**
 * @brief Returns a JPL quaternion from a rotation matrix
 *
 * This is based on the equation 74 in [Indirect Kalman Filter for 3D Attitude Estimation](http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf).
 * In the implementation, we have 4 statements so that we avoid a division by zero and
 * instead always divide by the largest diagonal element. This all comes from the
 * definition of a rotation matrix, using the diagonal elements and an off-diagonal.
 * \f{align*}{
 *  \mathbf{R}(\bar{q})=
 *  \begin{bmatrix}
 *  q_1^2-q_2^2-q_3^2+q_4^2 & 2(q_1q_2+q_3q_4) & 2(q_1q_3-q_2q_4) \\
 *  2(q_1q_2-q_3q_4) & -q_2^2+q_2^2-q_3^2+q_4^2 & 2(q_2q_3+q_1q_4) \\
 *  2(q_1q_3+q_2q_4) & 2(q_2q_3-q_1q_4) & -q_1^2-q_2^2+q_3^2+q_4^2
 *  \end{bmatrix}
 * \f}
 *
 * @param[in] rot 3x3 rotation matrix
 * @return 4x1 quaternion
 */
inline Eigen::Matrix<double, 4, 1> rot_2_quat(const Eigen::Matrix<double, 3, 3> &rot) {
    Eigen::Matrix<double, 4, 1> q;
    double T = rot.trace();
    if ((rot(0, 0) >= T) && (rot(0, 0) >= rot(1, 1)) && (rot(0, 0) >= rot(2, 2))) {
        q(0) = sqrt((1 + (2 * rot(0, 0)) - T) / 4);
        q(1) = (1 / (4 * q(0))) * (rot(0, 1) + rot(1, 0));
        q(2) = (1 / (4 * q(0))) * (rot(0, 2) + rot(2, 0));
        q(3) = (1 / (4 * q(0))) * (rot(1, 2) - rot(2, 1));

    } else if ((rot(1, 1) >= T) && (rot(1, 1) >= rot(0, 0)) && (rot(1, 1) >= rot(2, 2))) {
        q(1) = sqrt((1 + (2 * rot(1, 1)) - T) / 4);
        q(0) = (1 / (4 * q(1))) * (rot(0, 1) + rot(1, 0));
        q(2) = (1 / (4 * q(1))) * (rot(1, 2) + rot(2, 1));
        q(3) = (1 / (4 * q(1))) * (rot(2, 0) - rot(0, 2));
    } else if ((rot(2, 2) >= T) && (rot(2, 2) >= rot(0, 0)) && (rot(2, 2) >= rot(1, 1))) {
        q(2) = sqrt((1 + (2 * rot(2, 2)) - T) / 4);
        q(0) = (1 / (4 * q(2))) * (rot(0, 2) + rot(2, 0));
        q(1) = (1 / (4 * q(2))) * (rot(1, 2) + rot(2, 1));
        q(3) = (1 / (4 * q(2))) * (rot(0, 1) - rot(1, 0));
    } else {
        q(3) = sqrt((1 + T) / 4);
        q(0) = (1 / (4 * q(3))) * (rot(1, 2) - rot(2, 1));
        q(1) = (1 / (4 * q(3))) * (rot(2, 0) - rot(0, 2));
        q(2) = (1 / (4 * q(3))) * (rot(0, 1) - rot(1, 0));
    }
    if (q(3) < 0) {
        q = -q;
    }
    // normalize and return
    q = q / (q.norm());
    return q;
}

/**
 * @brief Converts JPL quaterion to SO(3) rotation matrix
 *
 * This is based on equation 62 in [Indirect Kalman Filter for 3D Attitude Estimation](http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf):
 * \f{align*}{
 *  \mathbf{R} = (2q_4^2-1)\mathbf{I}_3-2q_4\lfloor\mathbf{q}\times\rfloor+2\mathbf{q}\mathbf{q}^\top
 * @f}
 *
 * @param[in] q JPL quaternion
 * @return 3x3 SO(3) rotation matrix
 */
inline Eigen::Matrix<double, 3, 3> quat_2_Rot(const Eigen::Matrix<double, 4, 1> &q) {
    Eigen::Matrix<double, 3, 3> q_x = skew(q.block(0, 0, 3, 1));
    Eigen::MatrixXd Rot = (2 * std::pow(q(3, 0), 2) - 1) * Eigen::MatrixXd::Identity(3, 3) - 2 * q(3, 0) * q_x +
                          2 * q.block(0, 0, 3, 1) * (q.block(0, 0, 3, 1).transpose());
    return Rot;
}

/**
 * @brief Multiply two JPL quaternions
 *
 * This is based on equation 9 in [Indirect Kalman Filter for 3D Attitude Estimation](http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf).
 * We also enforce that the quaternion is unique by having q_4 be greater than zero.
 * \f{align*}{
 *  \bar{q}\otimes\bar{p}=
 *  \mathcal{L}(\bar{q})\bar{p}=
 *  \begin{bmatrix}
 *  q_4\mathbf{I}_3+\lfloor\mathbf{q}\times\rfloor & \mathbf{q} \\
 *  -\mathbf{q}^\top & q_4
 *  \end{bmatrix}
 *  \begin{bmatrix}
 *  \mathbf{p} \\ p_4
 *  \end{bmatrix}
 * @f}
 *
 * @param[in] q First JPL quaternion
 * @param[in] p Second JPL quaternion
 * @return 4x1 resulting q*p quaternion
 */
inline Eigen::Matrix<double, 4, 1> quat_multiply(const Eigen::Matrix<double, 4, 1> &q, const Eigen::Matrix<double, 4, 1> &p) {
    Eigen::Matrix<double, 4, 1> q_t;
    Eigen::Matrix<double, 4, 4> Qm;
    // create big L matrix
    Qm.block(0, 0, 3, 3) = q(3, 0) * Eigen::MatrixXd::Identity(3, 3) - skew(q.block(0, 0, 3, 1));
    Qm.block(0, 3, 3, 1) = q.block(0, 0, 3, 1);
    Qm.block(3, 0, 1, 3) = -q.block(0, 0, 3, 1).transpose();
    Qm(3, 3) = q(3, 0);
    q_t = Qm * p;
    // ensure unique by forcing q_4 to be >0
    if (q_t(3, 0) < 0) {
        q_t *= -1;
    }
    // normalize and return
    return q_t / q_t.norm();
}

inline long int factorial(int n) { return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n; }

inline Eigen::Matrix3d Gamma_SO3(const Eigen::Vector3d &w, int m) {

    assert(m >= 0); //程序中断条件判断
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    double theta = w.norm();
    if (theta < TOLERANCE) {
        return (1.0 / factorial(m)) * I;
    }
    Eigen::Matrix3d A = skew(w);
    double theta2 = theta * theta;

    switch (m) {
    case 0: // so(3)指数映射
        return I + (sin(theta) / theta) * A + ((1 - cos(theta)) / theta2) * A * A;
    case 1: //左雅可比矩阵
        return I + ((1 - cos(theta)) / theta2) * A + ((theta - sin(theta)) / (theta2 * theta)) * A * A;
    case 2: //
        return 0.5 * I + (theta - sin(theta)) / (theta2 * theta) * A + (theta2 + 2 * cos(theta) - 2) / (2 * theta2 * theta2) * A * A;

    default:
        Eigen::Matrix3d R = I + (sin(theta) / theta) * A + ((1 - cos(theta)) / theta2) * A * A;
        Eigen::Matrix3d S = I;
        Eigen::Matrix3d Ak = I;
        long int kfactorial = 1;
        for (int k = 1; k <= m; ++k) {
            kfactorial = kfactorial * k;
            Ak = (Ak * A).eval();
            S = (S + (1.0 / kfactorial) * Ak).eval();
        }
        if (m == 0) {
            return R;
        } else if (m % 2) { // odd
            return (1.0 / kfactorial) * I + (pow(-1, (m + 1) / 2) / pow(theta, m + 1)) * A * (R - S);
        } else { // even
            return (1.0 / kfactorial) * I + (pow(-1, m / 2) / pow(theta, m)) * (R - S);
        }
    }
}

inline Eigen::MatrixXd Adjoint_SEK3(const Eigen::MatrixXd &X) {
    int K = X.cols() - 3;
    Eigen::MatrixXd Adj = Eigen::MatrixXd::Zero(3 + 3 * K, 3 + 3 * K);
    Eigen::Matrix3d R = X.block<3, 3>(0, 0);
    Adj.block<3, 3>(0, 0) = R;
    for (int i = 0; i < K; ++i) {
        Adj.block<3, 3>(3 + 3 * i, 3 + 3 * i) = R;
        Adj.block<3, 3>(3 + 3 * i, 0) = skew(X.block<3, 1>(0, 3 + i)) * R;
    }
    return Adj;
}

} // namespace night_voyager
#endif