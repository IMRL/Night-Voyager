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
#ifndef HUNGARY_ESTIMATOR_H
#define HUNGARY_ESTIMATOR_H

#include <Eigen/Core>
#include <vector>

using namespace std;

namespace night_voyager {
class HungaryEstimator {
  public:
    HungaryEstimator(const Eigen::MatrixXd &costMatrix) : n(costMatrix.rows()), mat(costMatrix), matRcd(costMatrix) {
        assign.resize(costMatrix.rows());
        for (int i = 0; i < n; i++)
            assign[i] = -1;
        totalCost = 0;
    }
    void rowSub(); //Row-wise reduction

    void colSub(); //Column-wise reduction

    bool isOptimal(); //Determine whether the optimal allocation solution is found through trial assignment

    void matTrans(); //Matrix transformation to make the cost matrix have enough 0 elements

    vector<int> solve();

  private:
    int n;                  //Number of elements
    vector<int> assign;     //Assignment result
    Eigen::MatrixXd mat;    //Cost Matrix
    Eigen::MatrixXd matRcd; //Cost Matrix
    double totalCost;       
};
} // namespace night_voyager

#endif