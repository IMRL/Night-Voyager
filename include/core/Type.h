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
#ifndef TYPE_H
#define TYPE_H

#include <Eigen/Eigen>
#include <memory>

namespace night_voyager {
class Type {
  public:
    /**
     * @brief Default constructor for our Type
     *
     * @param size_ degrees of freedom of variable (i.e., the size of the error state)
     */
    Type(int size_) { _size = size_; }

    virtual ~Type(){};

    /**
     * @brief Sets id used to track location of variable in the filter covariance
     *
     * Note that the minimum ID is -1 which says that the state is not in our covariance.
     * If the ID is larger than -1 then this is the index location in the covariance matrix.
     *
     * @param new_id entry in filter covariance corresponding to this variable
     */
    virtual void set_local_id(int new_id) { _id = new_id; }

    /**
     * @brief Access to variable id (i.e. its location in the covariance)
     */
    int id() { return _id; }

    /**
     * @brief Access to variable size (i.e. its error state size)
     */
    int size() { return _size; }

    /**
     * @brief Update variable due to perturbation of error state
     *
     * @param dx Perturbation used to update the variable through a defined "boxplus" operation
     */
    virtual void update(const Eigen::VectorXd &dx) = 0;

    /**
     * @brief MSCKF Update variable due to perturbation of error state
     *
     * @param dx Perturbation used to update the variable through a defined "boxplus" operation
     */
    virtual void msckf_update(const Eigen::VectorXd &dx) = 0;

    /**
     * @brief Access variable's estimate
     */
    virtual const Eigen::MatrixXd &value() const { return _value; }

    /**
     * @brief Overwrite value of state's estimate
     * @param new_value New value that will overwrite state's value
     */
    virtual void set_value(const Eigen::MatrixXd &new_value) {
        assert(_value.rows() == new_value.rows());
        assert(_value.cols() == new_value.cols());
        _value = new_value;
    }

    /**
     * @brief Create a clone of this variable
     */
    virtual std::shared_ptr<Type> clone() = 0;

    /**
     * @brief Determine if pass variable is a sub-variable
     *
     * If the passed variable is a sub-variable or the current variable this will return it.
     * Otherwise it will return a nullptr, meaning that it was unable to be found.
     *
     * @param check Type pointer to compare our subvariables to
     */
    virtual std::shared_ptr<Type> check_if_subvariable(const std::shared_ptr<Type> check) { return nullptr; }

  protected:
    /// Current best estimate
    Eigen::MatrixXd _value;

    /// Location of error state in covariance
    int _id = -1;

    /// Dimension of error state
    int _size = -1;
};
} // namespace night_voyager
#endif