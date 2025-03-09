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
#ifndef POSEV_H
#define POSEV_H

#include "core/PoseHamilton.h"

namespace night_voyager {

class Posev : public Type {
  public:
    Posev() : Type(9) {
        // Initialize subvariables
        _pose = std::shared_ptr<PoseHamilton>(new PoseHamilton());
        _v = std::shared_ptr<Vec>(new Vec(3));

        Eigen::Matrix<double, 5, 5> pose0 = Eigen::Matrix<double, 5, 5>::Identity();
        set_value_internal(pose0);
    }

    ~Posev() {}

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
        _v->set_local_id(_pose->id() + ((new_id != -1) ? _pose->size() : 0));
    }

    /**
     * @brief Update q and p using a the HamiltonQuat update for orientation and vector update for position
     *
     * @param dx Correction vector (orientation then position)
     */
    void update(const Eigen::VectorXd &dx) override {

        assert(dx.rows() == _size);
        Eigen::MatrixXd X = _value;
        X = Exp_SEK3(dx) * X;

        set_value(X);
    }

    /**
     * @brief Update q and p using a the HamiltonQuat msckf update for orientation and vector update for position
     *
     * @param dx Correction vector (orientation then position)
     */
    void msckf_update(const Eigen::VectorXd &dx) override {

        assert(dx.rows() == _size);
        Eigen::MatrixXd X = _value;
        X.block(0, 0, 3, 3) = X.block(0, 0, 3, 3) * Exp_SO3(dx.block(0, 0, 3, 1));
        X.block(0, 3, 3, 1) += dx.block(3, 0, 3, 1);
        X.block(0, 4, 3, 1) += dx.block(6, 0, 3, 1);

        set_value(X);
    }

    /**
     * @brief Sets the value of the estimate
     * @param new_value New value we should set
     */
    void set_value(const Eigen::MatrixXd &new_value) override { set_value_internal(new_value); }

    std::shared_ptr<Type> clone() override {
        auto Clone = std::shared_ptr<Type>(new Posev());
        Clone->set_value(value());
        return Clone;
    }

    std::shared_ptr<Type> check_if_subvariable(const std::shared_ptr<Type> check) override {
        if (check == _pose) {
            return _pose;
        } else if (check == _pose->check_if_subvariable(check)) {
            return _pose->check_if_subvariable(check);
        } else if (check == _v) {
            return _v;
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
    Eigen::Matrix<double, 3, 1> vel() const { return _v->value(); }

    /// Pose type access
    std::shared_ptr<PoseHamilton> pose() { return _pose; }

    /// Quaternion type access
    std::shared_ptr<Mat> R() { return _pose->R(); }

    /// Position type access
    std::shared_ptr<Vec> p() { return _pose->p(); }

    /// Velocity type access
    std::shared_ptr<Vec> v() { return _v; }

  protected:
    std::shared_ptr<PoseHamilton> _pose;
    std::shared_ptr<Vec> _v;

    /**
     * @brief Sets the value of the estimate
     * @param new_value New value we should set
     */
    void set_value_internal(const Eigen::MatrixXd &new_value) {

        assert(new_value.rows() == 5);
        assert(new_value.cols() == 5);

        // Set orientation value
        _pose->set_value(new_value.block(0, 0, 4, 4));

        // Set position value
        _v->set_value(new_value.block(0, 4, 3, 1));

        _value = new_value;
    }
};

} // namespace night_voyager

#endif