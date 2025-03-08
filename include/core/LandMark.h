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
#ifndef LANDMARK_H
#define LANDMARK_H

#include "core/PoseHamilton.h"
#include "core/Vec.h"
#include "utils/Print.h"
#include <Eigen/Core>

namespace night_voyager {
class Landmark : public Vec {

  public:
    /// Default constructor (feature is a Vec of size 3 or Vec of size 1)
    Landmark(int dim) : Vec(dim) {}

    /// Feature ID of this landmark (corresponds to frontend id)
    size_t _featid;

    /// Timestamp of anchor clone
    double _anchor_clone_timestamp = -1;

    /// Timestamp of pseudo anchor clone
    double pseudo_anchor_clone_timestamp = -1;

    /// Boolean if this landmark has had at least one anchor change
    bool has_had_anchor_change = false;

    /// Boolean if this landmark should be marginalized out
    bool should_marg = false;

    /// Number of times the update has failed for this feature (we should remove if it fails a couple times!)
    int update_fail_count = 0;

    std::shared_ptr<PoseHamilton> anchor_frame;

    /**
     * @brief Overrides the default vector update rule
     * @param dx Additive error state correction
     */
    void update(const Eigen::VectorXd &dx) override {
        // Update estimate
        assert(dx.rows() == _size);
        set_value(_value + dx);
    }

    /**
     * @brief Will return the position of the feature in the global frame of reference.
     * @return Position of feature either in global or anchor frame
     */
    Eigen::Matrix<double, 3, 1> get_xyz() const { return value(); }

    /**
     * @brief Will set the current value.
     * @param p_FinG Position of the feature either in global or anchor frame
     */
    void set_from_xyz(Eigen::Matrix<double, 3, 1> p_FinG) {
        set_value(p_FinG);
        return;
    }

    /**
     * @brief Performs all the cloning
     */
    std::shared_ptr<Type> clone() override {
        auto Clone = std::shared_ptr<Type>(new Landmark(_size));
        Clone->set_value(value());
        return Clone;
    }
};
} // namespace night_voyager
#endif