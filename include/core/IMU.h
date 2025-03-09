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
#ifndef IMU_H
#define IMU_H

#include "core/Posev.h"
#include "core/Type.h"
#include "core/Vec.h"
#include "utils/Transform.h"

namespace night_voyager {
class IMU : public Type {
  public:
    IMU() : Type(15) {
        _pose = std::shared_ptr<Posev>(new Posev());
        _bg = std::shared_ptr<Vec>(new Vec(3));
        _ba = std::shared_ptr<Vec>(new Vec(3));

        // Set our default state value
        Eigen::MatrixXd imu0 = Eigen::MatrixXd::Identity(5, 7);
        set_value_internal(imu0);
    }

    ~IMU() {}

    /**
     * @brief Sets id used to track location of variable in the filter covariance
     *
     * Note that we update the sub-variables also.
     *
     * @param new_id entry in filter covariance corresponding to this variable
     */
    void set_local_id(int new_id) override {
        _id = new_id;
        _pose->set_local_id(new_id);
        _bg->set_local_id(_pose->id() + ((new_id != -1) ? _pose->size() : 0));
        _ba->set_local_id(_bg->id() + ((new_id != -1) ? _bg->size() : 0));
    }

    /**
     * @brief Performs update operation using Hamilton update for orientation, then vector updates for
     * position, velocity, gyro bias, and accel bias (in that order).
     *
     * @param dx 15 DOF vector encoding update using the following order (q, p, v, bg, ba)
     */
    void update(const Eigen::VectorXd &dx) override {

        assert(dx.rows() == _size);

        Eigen::MatrixXd newX = _value;

        // cout << "oldX: " << endl << newX << endl;
        newX.block(0, 0, 5, 5) = Exp_SEK3(dx.segment(0, 9)) * newX.block(0, 0, 5, 5);
        newX.block(0, 5, 3, 1) += dx.block(9, 0, 3, 1);
        newX.block(0, 6, 3, 1) += dx.block(12, 0, 3, 1);
        // cout << "newX: " << endl << newX << endl;

        set_value(newX);
    }

    /**
     * @brief Performs msckf update operation using Hamilton update for orientation, then vector updates for
     * position, velocity, gyro bias, and accel bias (in that order).
     *
     * @param dx 15 DOF vector encoding update using the following order (q, p, v, bg, ba)
     */
    void msckf_update(const Eigen::VectorXd &dx) override {

        assert(dx.rows() == _size);

        Eigen::MatrixXd newX = _value;

        // cout << "oldX: " << endl << newX.block(0, 0, 3, 3) << endl;
        newX.block(0, 0, 3, 3) = newX.block(0, 0, 3, 3) * Exp_SO3(dx.block(0, 0, 3, 1));
        newX.block(0, 3, 3, 1) += dx.block(3, 0, 3, 1);
        newX.block(0, 4, 3, 1) += dx.block(6, 0, 3, 1);
        newX.block(0, 5, 3, 1) += dx.block(9, 0, 3, 1);
        newX.block(0, 6, 3, 1) += dx.block(12, 0, 3, 1);
        // cout << "newX: " << endl << newX.block(0, 0, 3, 3) << endl;

        set_value(newX);
    }

    /**
     * @brief Sets the value of the estimate
     * @param new_value New value we should set
     */
    void set_value(const Eigen::MatrixXd &new_value) override { set_value_internal(new_value); }

    std::shared_ptr<Type> clone() override {
        auto Clone = std::shared_ptr<Type>(new IMU());
        Clone->set_value(value());
        return Clone;
    }

    std::shared_ptr<Type> check_if_subvariable(const std::shared_ptr<Type> check) override {
        if (check == _pose) {
            return _pose;
        } else if (check == _pose->check_if_subvariable(check)) {
            return _pose->check_if_subvariable(check);
        } else if (check == _bg) {
            return _bg;
        } else if (check == _ba) {
            return _ba;
        }
        return nullptr;
    }

    /// Rotation access
    Eigen::Matrix<double, 3, 3> Rot() const { return _pose->Rot(); }

    /// Quaternion access
    Eigen::Quaterniond quat() const { return _pose->quat(); }

    /// Position access
    Eigen::Matrix<double, 3, 1> pos() const { return _pose->pos(); }

    /// Velocity access
    Eigen::Matrix<double, 3, 1> vel() const { return _pose->vel(); }

    /// Pose + Velocity access
    Eigen::Matrix<double, 5, 5> pose_vel() const { return _pose->value(); }

    /// Gyro bias access
    Eigen::Matrix<double, 3, 1> bias_g() const { return _bg->value(); }

    /// Accel bias access
    Eigen::Matrix<double, 3, 1> bias_a() const { return _ba->value(); }

    /// Posev type access
    std::shared_ptr<Posev> posev() { return _pose; }

    /// Pose type access
    std::shared_ptr<PoseHamilton> pose() { return _pose->pose(); }

    /// Quaternion type access
    std::shared_ptr<Mat> R() { return _pose->R(); }

    /// Position type access
    std::shared_ptr<Vec> p() { return _pose->p(); }

    /// Velocity type access
    std::shared_ptr<Vec> v() { return _pose->v(); }

    /// Gyroscope bias access
    std::shared_ptr<Vec> bg() { return _bg; }

    /// Acceleration bias access
    std::shared_ptr<Vec> ba() { return _ba; }

  protected:
    /// Pose subvariable
    std::shared_ptr<Posev> _pose;

    /// Gyroscope bias subvariable
    std::shared_ptr<Vec> _bg;

    /// Acceleration bias subvariable
    std::shared_ptr<Vec> _ba;

    /**
     * @brief Sets the value of the estimate
     * @param new_value New value we should set
     */
    void set_value_internal(const Eigen::MatrixXd &new_value) {

        assert(new_value.rows() == 5);
        assert(new_value.cols() == 7);

        _pose->set_value(new_value.block(0, 0, 5, 5));
        _bg->set_value(new_value.block(0, 5, 3, 1));
        _ba->set_value(new_value.block(0, 6, 3, 1));

        _value = new_value;
    }
};
} // namespace night_voyager
#endif