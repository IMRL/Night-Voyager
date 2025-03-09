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
#ifndef POSEHAMILTON_H
#define POSEHAMILTON_H

#include "core/Mat.h"
#include "core/Type.h"
#include "core/Vec.h"
#include "utils/Transform.h"
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace night_voyager {

class PoseHamilton : public Type {
  public:
    PoseHamilton() : Type(6) {
        // Initialize subvariables
        _R = std::shared_ptr<Mat>(new Mat());
        _p = std::shared_ptr<Vec>(new Vec(3));

        Eigen::Matrix<double, 4, 4> pose0 = Eigen::Matrix<double, 4, 4>::Identity();
        set_value_internal(pose0);
    }

    ~PoseHamilton() {}

    /**
     * @brief Sets id used to track location of variable in the filter covariance
     *
     * Note that we update the sub-variables also.
     *
     * @param new_id entry in filter covariance corresponding to this variable
     */
    void set_local_id(int new_id) override {
        _id = new_id;
        _R->set_local_id(new_id);
        _p->set_local_id(new_id + ((new_id != -1) ? _R->size() : 0));
    }

    /**
     * @brief Update q and p using a the JPLQuat update for orientation and vector update for position
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
     * @brief MSCKF Update q and p using a the HamiltonQuat update for orientation and vector update for position
     *
     * @param dx Correction vector (orientation then position)
     */
    void msckf_update(const Eigen::VectorXd &dx) override {

        assert(dx.rows() == _size);
        Eigen::MatrixXd X = _value;
        X.block(0, 0, 3, 3) = X.block(0, 0, 3, 3) * Exp_SO3(dx.block(0, 0, 3, 1));
        X.block(0, 3, 3, 1) += dx.block(3, 0, 3, 1);

        set_value(X);
    }

    /**
     * @brief Sets the value of the estimate
     * @param new_value New value we should set
     */
    void set_value(const Eigen::MatrixXd &new_value) override { set_value_internal(new_value); }

    std::shared_ptr<Type> clone() override {
        auto Clone = std::shared_ptr<Type>(new PoseHamilton());
        Clone->set_value(value());
        return Clone;
    }

    std::shared_ptr<Type> check_if_subvariable(const std::shared_ptr<Type> check) override {
        if (check == _R) {
            return _R;
        } else if (check == _p) {
            return _p;
        }
        return nullptr;
    }

    /// Rotation access
    Eigen::Matrix<double, 3, 3> Rot() const { return _R->value(); }

    /// Quaternion access
    Eigen::Quaterniond quat() const {
        Eigen::Matrix3d R = _R->value();
        return Eigen::Quaterniond(R);
    }

    /// Position access
    Eigen::Matrix<double, 3, 1> pos() const { return _p->value(); }

    // Quaternion type access
    std::shared_ptr<Mat> R() { return _R; }

    // Position type access
    std::shared_ptr<Vec> p() { return _p; }

  protected:
    std::shared_ptr<Mat> _R;
    std::shared_ptr<Vec> _p;

    /**
     * @brief Sets the value of the estimate
     * @param new_value New value we should set
     */
    void set_value_internal(const Eigen::MatrixXd &new_value) {

        assert(new_value.rows() == 4);
        assert(new_value.cols() == 4);

        // Set orientation value
        _R->set_value(new_value.block(0, 0, 3, 3));

        // Set position value
        _p->set_value(new_value.block(0, 3, 3, 1));

        _value = new_value;
    }
};

} // namespace night_voyager

#endif