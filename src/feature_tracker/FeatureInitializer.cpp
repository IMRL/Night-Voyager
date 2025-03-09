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
#include "feature_tracker/FeatureInitializer.h"
#include "feature_tracker/Feature.h"
#include "utils/Transform.h"

namespace night_voyager {
bool FeatureInitializer::single_triangulation(std::shared_ptr<Feature> feat, std::unordered_map<double, ClonePose> &clonesCAM) {

    // Total number of measurements
    // Also set the first measurement to be the anchor frame
    feat->anchor_clone_timestamp = feat->timestamps.back();
    feat->pseudo_anchor_clone_timestamp = feat->timestamps.back();

    // Our linear system matrices
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    // Get the position of the anchor pose
    ClonePose anchorclone = clonesCAM.at(feat->anchor_clone_timestamp);
    const Eigen::Matrix<double, 3, 3> &R_GtoA = anchorclone.Rot();
    const Eigen::Matrix<double, 3, 1> &p_AinG = anchorclone.pos();

    // Add CAM_I features
    for (size_t m = 0; m < feat->timestamps.size(); m++) {

        // Get the position of this clone in the global
        const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(feat->timestamps.at(m)).Rot();
        const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(feat->timestamps.at(m)).pos();

        // Convert current position relative to anchor
        Eigen::Matrix<double, 3, 3> R_AtoCi;
        R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
        Eigen::Matrix<double, 3, 1> p_CiinA;
        p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);

        // Get the UV coordinate normal
        Eigen::Matrix<double, 3, 1> b_i;
        // cout << "feat->uvs: " << feat->uvs.at(m).transpose() << endl;
        b_i << feat->uvs_norm.at(m)(0), feat->uvs_norm.at(m)(1), 1;
        b_i = R_AtoCi.transpose() * b_i;
        b_i = b_i / b_i.norm();
        Eigen::Matrix3d Bperp = skew(b_i);

        // Append to our linear system
        Eigen::Matrix3d Ai = Bperp.transpose() * Bperp;
        A += Ai;
        b += Ai * p_CiinA;
    }

    // Solve the linear system
    Eigen::MatrixXd p_f = A.colPivHouseholderQr().solve(b);

    // Check A and p_f
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
    Eigen::MatrixXd singularValues;
    singularValues.resize(svd.singularValues().rows(), 1);
    singularValues = svd.singularValues();
    double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

    // std::stringstream ss;
    // ss << feat->featid << " - cond " << std::abs(condA) << " - z " << p_f(2, 0) << std::endl;
    // PRINT_DEBUG(ss.str().c_str());

    // If we have a bad condition number, or it is too close
    // Then set the flag for bad (i.e. set z-axis to nan)
    // cout << "p_f: " << p_f << endl;
    // cout << "condA: " << condA << endl;
    if (std::abs(condA) > _options.max_cond_number || p_f(2, 0) < _options.min_dist || p_f(2, 0) > _options.max_dist || std::isnan(p_f.norm())) {
        return false;
    }

    // Store it in our feature object
    feat->p_FinA = p_f;
    feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
    return true;
}

bool FeatureInitializer::single_triangulation_1d(std::shared_ptr<Feature> feat, std::unordered_map<double, ClonePose> &clonesCAM) {

    // Total number of measurements
    // Also set the first measurement to be the anchor frame
    feat->anchor_clone_timestamp = feat->timestamps.back();
    feat->pseudo_anchor_clone_timestamp = feat->timestamps.back();

    size_t idx_anchor_bearing = feat->timestamps.size() - 1;

    // Our linear system matrices
    double A = 0.0;
    double b = 0.0;

    // Get the position of the anchor pose
    ClonePose anchorclone = clonesCAM.at(feat->anchor_clone_timestamp);
    const Eigen::Matrix<double, 3, 3> &R_GtoA = anchorclone.Rot();
    const Eigen::Matrix<double, 3, 1> &p_AinG = anchorclone.pos();

    // Get bearing in anchor frame
    Eigen::Matrix<double, 3, 1> bearing_inA;
    bearing_inA << feat->uvs_norm.at(idx_anchor_bearing)(0), feat->uvs_norm.at(idx_anchor_bearing)(1), 1;
    bearing_inA = bearing_inA / bearing_inA.norm();

    // Loop through each camera for this feature

    // Add CAM_I features
    for (size_t m = 0; m < feat->timestamps.size(); m++) {

        if (m == idx_anchor_bearing)
            continue;

        // Get the position of this clone in the global
        const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(feat->timestamps.at(m)).Rot();
        const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(feat->timestamps.at(m)).pos();

        // Convert current position relative to anchor
        Eigen::Matrix<double, 3, 3> R_AtoCi;
        R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
        Eigen::Matrix<double, 3, 1> p_CiinA;
        p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);

        // Get the UV coordinate normal
        Eigen::Matrix<double, 3, 1> b_i;
        b_i << feat->uvs_norm.at(m)(0), feat->uvs_norm.at(m)(1), 1;
        b_i = R_AtoCi.transpose() * b_i;
        b_i = b_i / b_i.norm();
        Eigen::Matrix3d Bperp = skew(b_i);

        // Append to our linear system
        Eigen::Vector3d BperpBanchor = Bperp * bearing_inA;
        A += BperpBanchor.dot(BperpBanchor);
        b += BperpBanchor.dot(Bperp * p_CiinA);
    }

    // Solve the linear system
    double depth = b / A;
    Eigen::MatrixXd p_f = depth * bearing_inA;

    // Then set the flag for bad (i.e. set z-axis to nan)
    if (p_f(2, 0) < _options.min_dist || p_f(2, 0) > _options.max_dist || std::isnan(p_f.norm())) {
        return false;
    }

    // Store it in our feature object
    feat->p_FinA = p_f;
    feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
    return true;
}

bool FeatureInitializer::single_gaussnewton(std::shared_ptr<Feature> feat, std::unordered_map<double, ClonePose> &clonesCAM) {

    // Get into inverse depth
    double rho = 1 / feat->p_FinA(2);
    double alpha = feat->p_FinA(0) / feat->p_FinA(2);
    double beta = feat->p_FinA(1) / feat->p_FinA(2);

    // Optimization parameters
    double lam = _options.init_lamda;
    double eps = 10000;
    int runs = 0;

    // Variables used in the optimization
    bool recompute = true;
    Eigen::Matrix<double, 3, 3> Hess = Eigen::Matrix<double, 3, 3>::Zero();
    Eigen::Matrix<double, 3, 1> grad = Eigen::Matrix<double, 3, 1>::Zero();

    // Cost at the last iteration
    double cost_old = compute_error(clonesCAM, feat, alpha, beta, rho);

    // Get the position of the anchor pose
    const Eigen::Matrix<double, 3, 3> &R_GtoA = clonesCAM.at(feat->anchor_clone_timestamp).Rot();
    const Eigen::Matrix<double, 3, 1> &p_AinG = clonesCAM.at(feat->anchor_clone_timestamp).pos();

    // Loop till we have either
    // 1. Reached our max iteration count
    // 2. System is unstable
    // 3. System has converged
    while (runs < _options.max_runs && lam < _options.max_lamda && eps > _options.min_dx) {

        // Triggers a recomputation of jacobians/information/gradients
        if (recompute) {

            Hess.setZero();
            grad.setZero();

            double err = 0;

            // Loop through each camera for this feature

            // Add CAM_I features
            for (size_t m = 0; m < feat->timestamps.size(); m++) {

                //=====================================================================================
                //=====================================================================================

                // Get the position of this clone in the global
                const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(feat->timestamps.at(m)).Rot();
                const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(feat->timestamps.at(m)).pos();
                // Convert current position relative to anchor
                Eigen::Matrix<double, 3, 3> R_AtoCi;
                R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
                Eigen::Matrix<double, 3, 1> p_CiinA;
                p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);
                Eigen::Matrix<double, 3, 1> p_AinCi;
                p_AinCi.noalias() = -R_AtoCi * p_CiinA;

                //=====================================================================================
                //=====================================================================================

                // Middle variables of the system
                double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
                double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
                double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
                // Calculate jacobian
                double d_z1_d_alpha = (R_AtoCi(0, 0) * hi3 - hi1 * R_AtoCi(2, 0)) / (pow(hi3, 2));
                double d_z1_d_beta = (R_AtoCi(0, 1) * hi3 - hi1 * R_AtoCi(2, 1)) / (pow(hi3, 2));
                double d_z1_d_rho = (p_AinCi(0, 0) * hi3 - hi1 * p_AinCi(2, 0)) / (pow(hi3, 2));
                double d_z2_d_alpha = (R_AtoCi(1, 0) * hi3 - hi2 * R_AtoCi(2, 0)) / (pow(hi3, 2));
                double d_z2_d_beta = (R_AtoCi(1, 1) * hi3 - hi2 * R_AtoCi(2, 1)) / (pow(hi3, 2));
                double d_z2_d_rho = (p_AinCi(1, 0) * hi3 - hi2 * p_AinCi(2, 0)) / (pow(hi3, 2));
                Eigen::Matrix<double, 2, 3> H;
                H << d_z1_d_alpha, d_z1_d_beta, d_z1_d_rho, d_z2_d_alpha, d_z2_d_beta, d_z2_d_rho;
                // Calculate residual
                Eigen::Matrix<float, 2, 1> z;
                z << hi1 / hi3, hi2 / hi3;
                Eigen::Matrix<float, 2, 1> res = feat->uvs_norm.at(m) - z;

                //=====================================================================================
                //=====================================================================================

                // Append to our summation variables
                err += std::pow(res.norm(), 2);
                grad.noalias() += H.transpose() * res.cast<double>();
                Hess.noalias() += H.transpose() * H;
            }
        }

        // Solve Levenberg iteration
        Eigen::Matrix<double, 3, 3> Hess_l = Hess;
        for (size_t r = 0; r < (size_t)Hess.rows(); r++) {
            Hess_l(r, r) *= (1.0 + lam);
        }

        Eigen::Matrix<double, 3, 1> dx = Hess_l.colPivHouseholderQr().solve(grad);
        // Eigen::Matrix<double,3,1> dx = (Hess+lam*Eigen::MatrixXd::Identity(Hess.rows(), Hess.rows())).colPivHouseholderQr().solve(grad);

        // Check if error has gone down
        double cost = compute_error(clonesCAM, feat, alpha + dx(0, 0), beta + dx(1, 0), rho + dx(2, 0));

        // Debug print
        // std::stringstream ss;
        // ss << "run = " << runs << " | cost = " << dx.norm() << " | lamda = " << lam << " | depth = " << 1/rho << endl;
        // PRINT_DEBUG(ss.str().c_str());

        // Check if converged
        if (cost <= cost_old && (cost_old - cost) / cost_old < _options.min_dcost) {
            alpha += dx(0, 0);
            beta += dx(1, 0);
            rho += dx(2, 0);
            eps = 0;
            break;
        }

        // If cost is lowered, accept step
        // Else inflate lambda (try to make more stable)
        if (cost <= cost_old) {
            recompute = true;
            cost_old = cost;
            alpha += dx(0, 0);
            beta += dx(1, 0);
            rho += dx(2, 0);
            runs++;
            lam = lam / _options.lam_mult;
            eps = dx.norm();
        } else {
            recompute = false;
            lam = lam * _options.lam_mult;
            continue;
        }
    }

    // Revert to standard, and set to all
    feat->p_FinA(0) = alpha / rho;
    feat->p_FinA(1) = beta / rho;
    feat->p_FinA(2) = 1 / rho;

    // Get tangent plane to x_hat
    // cout << "feat->p_FinA: " << feat->p_FinA.transpose() << endl;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(feat->p_FinA);
    Eigen::MatrixXd Q = qr.householderQ();

    // Max baseline we have between poses
    double base_line_max = 0.0;

    // Check maximum baseline
    // Loop through each camera for this feature
    for (size_t m = 0; m < feat->timestamps.size(); m++) {
        // Get the position of this clone in the global
        const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(feat->timestamps.at(m)).pos();
        // Convert current position relative to anchor
        Eigen::Matrix<double, 3, 1> p_CiinA = R_GtoA * (p_CiinG - p_AinG);
        // Dot product camera pose and nullspace
        double base_line = ((Q.block(0, 1, 3, 2)).transpose() * p_CiinA).norm();
        if (base_line > base_line_max)
            base_line_max = base_line;
    }
    // std::stringstream ss;
    // ss << feat->featid << " - max base " << (feat->p_FinA.norm() / base_line_max) << " - z " << feat->p_FinA(2) << std::endl;
    // PRINT_DEBUG(ss.str().c_str());

    // Check if this feature is bad or not
    // 1. If the feature is too close
    // 2. If the feature is invalid
    // 3. If the baseline ratio is large
    // cout << "norm_per_base: " << feat->p_FinA.norm() / base_line_max << endl;
    // cout << "base_line_max: " << base_line_max << endl;
    if (feat->p_FinA(2) < _options.min_dist || feat->p_FinA(2) > _options.max_dist || (feat->p_FinA.norm() / base_line_max) > _options.max_baseline ||
        std::isnan(feat->p_FinA.norm())) {
        return false;
    }

    // Finally get position in global frame
    feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
    return true;
}

double FeatureInitializer::compute_error(std::unordered_map<double, ClonePose> &clonesCAM, std::shared_ptr<Feature> feat, double alpha, double beta,
                                         double rho) {

    // Total error
    double err = 0;

    // Get the position of the anchor pose
    const Eigen::Matrix<double, 3, 3> &R_GtoA = clonesCAM.at(feat->anchor_clone_timestamp).Rot();
    const Eigen::Matrix<double, 3, 1> &p_AinG = clonesCAM.at(feat->anchor_clone_timestamp).pos();

    for (size_t m = 0; m < feat->timestamps.size(); m++) {

        //=====================================================================================
        //=====================================================================================

        // Get the position of this clone in the global
        const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(feat->timestamps.at(m)).Rot();
        const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(feat->timestamps.at(m)).pos();
        // Convert current position relative to anchor
        Eigen::Matrix<double, 3, 3> R_AtoCi;
        R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
        Eigen::Matrix<double, 3, 1> p_CiinA;
        p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);
        Eigen::Matrix<double, 3, 1> p_AinCi;
        p_AinCi.noalias() = -R_AtoCi * p_CiinA;

        //=====================================================================================
        //=====================================================================================

        // Middle variables of the system
        double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
        double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
        double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
        // Calculate residual
        Eigen::Matrix<float, 2, 1> z;
        z << hi1 / hi3, hi2 / hi3;
        Eigen::Matrix<float, 2, 1> res = feat->uvs_norm.at(m) - z;
        // Append to our summation variables
        err += pow(res.norm(), 2);
    }

    return err;
}
} // namespace night_voyager