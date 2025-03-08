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
#ifndef MAT_H
#define MAT_H

#include "core/Type.h"
#include "utils/Transform.h"

namespace night_voyager {
class Mat : public Type {
  public:
    Mat() : Type(3) { _value = Eigen::Matrix3d::Identity(); }

    ~Mat() {}

    /**
     * @brief Implements the update operation through standard vector addition
     *
     * \f{align*}{
     * \mathbf{v} &= \hat{\mathbf{v}} + \tilde{\mathbf{v}}_{dx}
     * \f}
     *
     * @param dx Additive error state correction
     */
    void update(const Eigen::VectorXd &dx) override {
        assert(dx.rows() == _size);
        set_value(Exp_SO3(dx) * _value);
    }

    /**
     * @brief Implements the update operation through standard vector addition
     *
     * \f{align*}{
     * \mathbf{v} &= \hat{\mathbf{v}} + \tilde{\mathbf{v}}_{dx}
     * \f}
     *
     * @param dx Additive error state correction
     */
    void msckf_update(const Eigen::VectorXd &dx) override {
        assert(dx.rows() == _size);
        set_value(_value * Exp_SO3(dx));
    }

    /**
     * @brief Performs all the cloning
     */
    std::shared_ptr<Type> clone() override {
        auto Clone = std::shared_ptr<Type>(new Mat());
        Clone->set_value(value());
        return Clone;
    }
};
} // namespace night_voyager
#endif