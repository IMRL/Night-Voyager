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
#ifndef TYPE_VEC_H
#define TYPE_VEC_H

#include "core/Type.h"

/**
 * @brief Derived Type class that implements vector variables
 */
namespace night_voyager {
class Vec : public Type {

  public:
    /**
     * @brief Default constructor for Vec
     * @param dim Size of the vector (will be same as error state)
     */
    Vec(int dim) : Type(dim) { _value = Eigen::VectorXd::Zero(dim); }

    ~Vec() {}

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
        set_value(_value + dx);
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
        set_value(_value + dx);
    }

    /**
     * @brief Performs all the cloning
     */
    std::shared_ptr<Type> clone() override {
        auto Clone = std::shared_ptr<Type>(new Vec(_size));
        Clone->set_value(value());
        return Clone;
    }
};
} // namespace night_voyager
#endif