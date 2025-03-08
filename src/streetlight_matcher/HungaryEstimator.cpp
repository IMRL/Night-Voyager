#include "streetlight_matcher/HungaryEstimator.h"

namespace night_voyager {
void HungaryEstimator::rowSub() {
    Eigen::VectorXd minEmt;
    minEmt = mat.rowwise().minCoeff();

    for (int j = 0; j < n; j++) {
        mat.col(j) -= minEmt;
    }
}

void HungaryEstimator::colSub() {
    Eigen::VectorXd minEmt;
    minEmt = mat.colwise().minCoeff();

    for (int i = 0; i < n; i++) {
        mat.row(i) -= minEmt;
    }
}

bool HungaryEstimator::isOptimal() {
    vector<int> tAssign(n, -1);
    vector<int> nZero(n, 0);
    vector<bool> rowIsUsed(n, false);
    vector<bool> colIsUsed(n, false);

    int nline = 0;
    while (nline < n) {
        for (int i = 0; i < n; i++)
            nZero[i] = 0;
        for (int i = 0; i < n; i++) {
            if (rowIsUsed[i])
                continue;
            for (int j = 0; j < n; j++)
                if (!colIsUsed[j] && mat(i, j) == 0)
                    ++nZero[i];
        }

        int minZeros = n;
        int rowId = -1;
        for (int i = 0; i < n; i++) {
            if (!rowIsUsed[i] && nZero[i] < minZeros && nZero[i] > 0) {
                minZeros = nZero[i];
                rowId = i;
            }
        }

        if (rowId == -1)
            break;
        for (int j = 0; j < n; j++) {
            if (mat(rowId, j) == 0 && !colIsUsed[j]) {
                rowIsUsed[rowId] = 1;
                colIsUsed[j] = 1;
                tAssign[rowId] = j;
                break;
            }
        }
        ++nline;
    }

    for (int i = 0; i < n; i++) {
        assign[i] = tAssign[i];
    }
    for (int i = 0; i < n; i++) {
        if (assign[i] == -1)
            return false;
    }
    return true;
}

void HungaryEstimator::matTrans() {
    vector<bool> rowTip(n, false);
    vector<bool> colTip(n, false);
    vector<bool> rowLine(n, false);
    vector<bool> colLine(n, false);

    //打勾
    for (int i = 0; i < n; i++)
        if (assign[i] == -1)
            rowTip[i] = 1;

    while (1) {
        int preTip = 0;
        for (int i = 0; i < n; i++)
            preTip += rowTip[i] + colTip[i];
        for (int i = 0; i < n; i++) {
            if (rowTip[i]) {
                for (int j = 0; j < n; j++) {
                    if (mat(i, j) == 0)
                        colTip[j] = 1;
                }
            }
        }
        for (int j = 0; j < n; j++) {
            if (colTip[j]) {
                for (int i = 0; i < n; i++) {
                    if (assign[i] == j)
                        rowTip[i] = 1;
                }
            }
        }
        int curTip = 0;
        for (int i = 0; i < n; i++)
            curTip += rowTip[i] + colTip[i];
        if (preTip == curTip)
            break;
    }

    //画线
    for (int i = 0; i < n; i++) {
        if (!rowTip[i])
            rowLine[i] = 1;
        if (colTip[i])
            colLine[i] = 1;
    }

    //找最小元素
    double minEmt = 1000.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (!rowLine[i] && !colLine[j] && mat(i, j) < minEmt)
                minEmt = mat(i, j);

    //变换
    for (int i = 0; i < n; i++) {
        if (rowTip[i]) {
            for (int j = 0; j < n; j++)
                mat(i, j) -= minEmt;
        }
    }
    for (int j = 0; j < n; j++) {
        if (colTip[j]) {
            for (int i = 0; i < n; i++)
                mat(i, j) += minEmt;
        }
    }
}

vector<int> HungaryEstimator::solve() {
    rowSub();
    colSub();

    // cout << mat << endl;

    while (!isOptimal()) {
        matTrans();
    }

    for (int i = 0; i < n; i++)
        totalCost += matRcd(i, assign[i]);

    // cout << "total_cost: " << totalCost << endl;

    return assign;
}
} // namespace night_voyager