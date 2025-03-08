#include "msckf_iekf/StateHelper.h"
#include "core/LandMark.h"
#include "core/Type.h"
#include "msckf_iekf/State.h"
#include "utils/Transform.h"
#include <boost/math/distributions/chi_squared.hpp>

namespace night_voyager {
void StateHelper::set_initial_covariance(std::shared_ptr<State> state, const Eigen::MatrixXd &covariance,
                                         const std::vector<std::shared_ptr<Type>> &order) {

    // We need to loop through each element and overwrite the current covariance values
    // For example consider the following:
    // x = [ ori pos ] -> insert into -> x = [ ori bias pos ]
    // P = [ P_oo P_op ] -> P = [ P_oo  0   P_op ]
    //     [ P_po P_pp ]        [  0    P*    0  ]
    //                          [ P_po  0   P_pp ]
    // The key assumption here is that the covariance is block diagonal (cross-terms zero with P* can be dense)
    // This is normally the care on startup (for example between calibration and the initial state

    // For each variable, lets copy over all other variable cross terms
    // Note: this copies over itself to when i_index=k_index
    int i_index = 0;
    for (size_t i = 0; i < order.size(); i++) {
        int k_index = 0;
        for (size_t k = 0; k < order.size(); k++) {
            state->_Cov.block(order[i]->id(), order[k]->id(), order[i]->size(), order[k]->size()) =
                covariance.block(i_index, k_index, order[i]->size(), order[k]->size());
            k_index += order[k]->size();
        }
        i_index += order[i]->size();
    }
    state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
}

void StateHelper::EKFPropagation(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &order_NEW,
                                 const std::vector<std::shared_ptr<Type>> &order_OLD, const Eigen::MatrixXd &Phi, const Eigen::MatrixXd &Q) {

    // We need at least one old and new variable
    if (order_NEW.empty() || order_OLD.empty()) {
        PRINT_ERROR(RED "StateHelper::EKFPropagation() - Called with empty variable arrays!\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    // Phi is in order of R_ItoG, p_IinG, v_IinG, bg, ba, features_landmarks, 6DOF features
    // we should collect their covariance from the state->_Cov and compose the covariance in order.

    std::vector<int> Phi_id;   // location of each variable in Phi
    std::vector<int> _Cov_id;  // location of each variable in state->_Cov;
    std::vector<int> var_size; // the size of each variable
    int current_id = 0;
    int size_order_OLD = 0;
    for (const auto var : order_OLD) {
        Phi_id.push_back(current_id);
        _Cov_id.push_back(var->id());
        var_size.push_back(var->size());
        current_id += var->size();
        size_order_OLD += var->size();
    }

    // PRINT_INFO(WHITE "StateHelper::EKFPropagation(): Size of size_order_OLD: %d\n", size_order_OLD);
    // other variables that are not propagated with IMU
    std::vector<std::shared_ptr<Type>> order_notprop;
    std::vector<int> id_notprop;
    std::vector<int> var_size_notprop;
    int notprop_size = 0;

    order_notprop.push_back(state->_pose_MAPtoLOC);
    id_notprop.push_back(notprop_size);
    var_size_notprop.push_back(state->_pose_MAPtoLOC->size());
    notprop_size += state->_pose_MAPtoLOC->size();

    auto iter = state->_clones_IMU.begin();
    while (iter != state->_clones_IMU.end()) {
        order_notprop.push_back(iter->second);
        id_notprop.push_back(notprop_size);
        var_size_notprop.push_back(iter->second->size());
        notprop_size += iter->second->size();
        iter++;
    }

    if (order_OLD.size() == 1 || state->_kf != KFCLASS::IKF_IMUGROUP) {
        auto it = state->_features_SLAM.begin();
        while (it != state->_features_SLAM.end()) {
            order_notprop.push_back(it->second);
            id_notprop.push_back(notprop_size);
            var_size_notprop.push_back(it->second->size());
            notprop_size += it->second->size();
            it++;
        }
    }

    PRINT_INFO(WHITE "StateHelper::EKFPropagation(): Size of notprop_size: %d\n", notprop_size);
    PRINT_INFO(WHITE "StateHelper::EKFPropagation(): Size of State Cov: %d\n", state->_Cov.rows());

    assert(Phi.cols() == size_order_OLD);
    assert(state->_Cov.rows() == (size_order_OLD + notprop_size));

    // collect covariance from state->_Cov
    Eigen::MatrixXd Cov = Eigen::MatrixXd::Zero(size_order_OLD, size_order_OLD);
    for (size_t i = 0; i < _Cov_id.size(); i++) {
        for (size_t j = 0; j < _Cov_id.size(); j++) {
            Cov.block(Phi_id[i], Phi_id[j], var_size[i], var_size[j]) = state->_Cov.block(_Cov_id[i], _Cov_id[j], var_size[i], var_size[j]);
        }
    }

    Eigen::MatrixXd Cross_Cov = Eigen::MatrixXd::Zero(size_order_OLD, notprop_size);
    for (size_t i = 0; i < _Cov_id.size(); i++) {
        for (size_t j = 0; j < order_notprop.size(); j++) {
            Cross_Cov.block(Phi_id[i], id_notprop[j], var_size[i], var_size_notprop[j]) =
                state->_Cov.block(_Cov_id[i], order_notprop[j]->id(), var_size[i], var_size_notprop[j]);
        }
    }

    // propagate the covariance
    Eigen::MatrixXd Cov_new = Phi * Cov * Phi.transpose() + Q;
    Eigen::MatrixXd Cross_Cov_new = Phi * Cross_Cov;

    // reassgin the Cov_new to state->_Cov
    for (size_t i = 0; i < _Cov_id.size(); i++) {
        for (size_t j = 0; j < _Cov_id.size(); j++) {
            state->_Cov.block(_Cov_id[i], _Cov_id[j], var_size[i], var_size[j]) = Cov_new.block(Phi_id[i], Phi_id[j], var_size[i], var_size[j]);
        }

        for (size_t k = 0; k < order_notprop.size(); k++) {
            state->_Cov.block(_Cov_id[i], order_notprop[k]->id(), var_size[i], var_size_notprop[k]) =
                Cross_Cov_new.block(Phi_id[i], id_notprop[k], var_size[i], var_size_notprop[k]);

            state->_Cov.block(order_notprop[k]->id(), _Cov_id[i], var_size_notprop[k], var_size[i]) =
                Cross_Cov_new.block(Phi_id[i], id_notprop[k], var_size[i], var_size_notprop[k]).transpose();
        }
    }

    // check negative
    Eigen::VectorXd diags = state->_Cov.diagonal();
    bool found_neg = false;
    for (int i = 0; i < diags.rows(); i++) {
        if (diags(i) < 0.0) {
            printf(RED "StateHelper::EKFPropagation() - diagonal at %d is %.2f\n" RESET, i, diags(i));
            found_neg = true;
        }
    }
    assert(!found_neg);
}

void StateHelper::augment_clone(std::shared_ptr<State> state) {

    // We can't insert a clone that occured at the same timestamp!
    if (state->_clones_IMU.find(state->_timestamp) != state->_clones_IMU.end()) {
        PRINT_ERROR(RED "TRIED TO INSERT A CLONE AT THE SAME TIME AS AN EXISTING CLONE, EXITING!#!@#!@#\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    // Call on our cloner and add it to our vector of types
    // NOTE: this will clone the clone pose to the END of the covariance...
    std::shared_ptr<Type> posetemp = StateHelper::clone(state, state->_imu->pose());

    // Cast to a JPL pose type, check if valid
    std::shared_ptr<PoseHamilton> pose = std::dynamic_pointer_cast<PoseHamilton>(posetemp);
    if (pose == nullptr) {
        PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    // Append the new clone to our clone vector
    state->_clones_IMU[state->_timestamp] = pose;
}

std::shared_ptr<Type> StateHelper::clone(std::shared_ptr<State> state, std::shared_ptr<Type> variable_to_clone) {

    // Get total size of new cloned variables, and the old covariance size
    int total_size = variable_to_clone->size();
    int old_size = (int)state->_Cov.rows();
    int new_loc = (int)state->_Cov.rows();

    // Resize both our covariance to the new size
    state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(old_size + total_size, old_size + total_size));

    // What is the new state, and variable we inserted
    const std::vector<std::shared_ptr<Type>> new_variables = state->_variables;
    std::shared_ptr<Type> new_clone = nullptr;

    // Loop through all variables, and find the variable that we are going to clone
    for (size_t k = 0; k < state->_variables.size(); k++) {

        // Skip this if it is not the same
        // First check if the top level variable is the same, then check the sub-variables
        std::shared_ptr<Type> type_check = state->_variables.at(k)->check_if_subvariable(variable_to_clone);
        if (state->_variables.at(k) == variable_to_clone) {
            type_check = state->_variables.at(k);
        } else if (type_check != variable_to_clone) {
            continue;
        }

        // So we will clone this one
        int old_loc = type_check->id();

        // Copy the covariance elements
        state->_Cov.block(new_loc, new_loc, total_size, total_size) = state->_Cov.block(old_loc, old_loc, total_size, total_size);
        state->_Cov.block(0, new_loc, old_size, total_size) = state->_Cov.block(0, old_loc, old_size, total_size);
        state->_Cov.block(new_loc, 0, total_size, old_size) = state->_Cov.block(old_loc, 0, total_size, old_size);

        // Create clone from the type being cloned
        new_clone = type_check->clone();
        new_clone->set_local_id(new_loc);
        break;
    }

    // Check if the current state has this variable
    if (new_clone == nullptr) {
        PRINT_ERROR(RED "StateHelper::clone() - Called on variable is not in the state\n" RESET);
        PRINT_ERROR(RED "StateHelper::clone() - Ensure that the variable specified is a variable, or sub-variable..\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    // Add to variable list and return
    state->_variables.push_back(new_clone);
    return new_clone;
}

Eigen::MatrixXd StateHelper::get_marginal_covariance(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &small_variables) {

    // Calculate the marginal covariance size we need to make our matrix
    int cov_size = 0;
    for (size_t i = 0; i < small_variables.size(); i++) {
        cov_size += small_variables[i]->size();
    }

    // Construct our return covariance
    Eigen::MatrixXd Small_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

    // For each variable, lets copy over all other variable cross terms
    // Note: this copies over itself to when i_index=k_index
    int i_index = 0;
    for (size_t i = 0; i < small_variables.size(); i++) {
        int k_index = 0;
        for (size_t k = 0; k < small_variables.size(); k++) {
            Small_cov.block(i_index, k_index, small_variables[i]->size(), small_variables[k]->size()) =
                state->_Cov.block(small_variables[i]->id(), small_variables[k]->id(), small_variables[i]->size(), small_variables[k]->size());
            k_index += small_variables[k]->size();
        }
        i_index += small_variables[i]->size();
    }

    // Return the covariance
    // Small_cov = 0.5*(Small_cov+Small_cov.transpose());
    return Small_cov;
}

void StateHelper::EKFUpdate(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &H_order, const Eigen::MatrixXd &H,
                            const Eigen::VectorXd &res, const Eigen::MatrixXd &R) {
    //==========================================================
    //==========================================================
    // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
    assert(res.rows() == R.rows());
    assert(H.rows() == res.rows());
    Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());
    // Get the location in small jacobian for each measuring variable
    int current_it = 0;
    std::vector<int> H_id;
    for (const auto &meas_var : H_order) {
        H_id.push_back(current_it);
        current_it += meas_var->size();
    }

    //==========================================================
    //==========================================================
    // For each active variable find its M = P*H^T
    for (const auto &var : state->_variables) {
        // Sum up effect of each subjacobian = K_i= \sum_m (P_im Hm^T)
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for (size_t i = 0; i < H_order.size(); i++) {
            std::shared_ptr<Type> meas_var = H_order[i];
            M_i.noalias() += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                             H.block(0, H_id[i], H.rows(), meas_var->size()).transpose();
        }
        M_a.block(var->id(), 0, var->size(), res.rows()) = M_i;
    }

    //==========================================================
    //==========================================================
    // Get covariance of the involved terms
    Eigen::MatrixXd P_small = StateHelper::get_marginal_covariance(state, H_order);

    // // Residual covariance S = H*Cov*H' + R
    Eigen::MatrixXd S(R.rows(), R.rows());
    S.triangularView<Eigen::Upper>() = H * P_small * H.transpose();
    S.triangularView<Eigen::Upper>() += R;

    // Invert our S (should we use a more stable method here??)
    Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
    S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
    Eigen::MatrixXd K = M_a * Sinv.selfadjointView<Eigen::Upper>();

    if (state->_kf == KFCLASS::MSCKF) {
        Eigen::VectorXd dx = K * res;
        for (size_t i = 0; i < state->_variables.size(); i++) {
            state->_variables.at(i)->msckf_update(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
        }
    } else {
        Eigen::VectorXd dx = K * res;
        UpdateVarInvariant(state, dx);
    }

    // Update Covariance
    state->_Cov.triangularView<Eigen::Upper>() -= K * M_a.transpose();
    state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();

    // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
    Eigen::VectorXd diags = state->_Cov.diagonal();
    bool found_neg = false;
    for (int i = 0; i < diags.rows(); i++) {
        if (diags(i) < 0.0) {
            PRINT_WARNING(RED "StateHelper::EKFUpdate() - diagonal at %d is %.2f\n" RESET, i, diags(i));
            found_neg = true;
        }
    }
    if (found_neg) {
        std::exit(EXIT_FAILURE);
    }
}

void StateHelper::UpdateVarInvariant(std::shared_ptr<State> state, const Eigen::VectorXd dx) {

    // first update imu stuff
    Eigen::VectorXd delta_x_imu = dx.block(state->_imu->id(), 0, state->_imu->size(), 1);
    Eigen::Matrix3d dR = Exp_SO3(delta_x_imu.head<3>());

    if (delta_x_imu.segment<3>(6).norm() > 1.0 || delta_x_imu.segment<3>(3).norm() > 2.0) {
        PRINT_INFO(YELLOW "Delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
        PRINT_INFO(YELLOW "Delta position: %f\n", delta_x_imu.segment<3>(3).norm());
        PRINT_INFO(YELLOW "Update is too large\n");
        // return;
    }

    state->_imu->update(delta_x_imu.head<15>());
    auto clone = state->_clones_IMU.begin();
    while (clone != state->_clones_IMU.end()) {
        Eigen::VectorXd delta_clone = dx.segment<6>(clone->second->id());
        clone->second->update(delta_clone);
        clone++;
    }

    // update global to local transform
    Eigen::VectorXd delta_map_to_loc = dx.segment<6>(state->_pose_MAPtoLOC->id());
    state->_pose_MAPtoLOC->update(delta_map_to_loc);

    // update features
    // MSC-InEKF-FDRC
    if (state->_feature_in_clone) {
        auto feat = state->_features_SLAM.begin();
        Eigen::Matrix3d dR_group;
        double last_pseudo_time = -1;
        while (feat != state->_features_SLAM.end()) {
            Eigen::VectorXd d_feat_se = Eigen::VectorXd(6, 1);
            d_feat_se.head<3>() = dx.segment<3>(state->_clones_IMU[feat->second->pseudo_anchor_clone_timestamp]->R()->id());
            d_feat_se.tail<3>() = dx.segment<3>(feat->second->id());
            Eigen::MatrixXd d_feat_SE = Exp_SEK3(d_feat_se);
            if (std::abs(last_pseudo_time - feat->second->pseudo_anchor_clone_timestamp) > 1e-7) {
                dR_group = Exp_SO3(dx.segment<3>(state->_clones_IMU[feat->second->pseudo_anchor_clone_timestamp]->R()->id()));
                last_pseudo_time = feat->second->pseudo_anchor_clone_timestamp;
            }
            assert(last_pseudo_time > 0);
            feat->second->set_value(dR_group * feat->second->get_xyz() + d_feat_SE.block(0, 3, 3, 1));
            feat++;
        }
    }
    // MSC-InEKF-FDRC or MSC-InEKF-FDR
    else if (state->_feature_in_rel_group) {
        auto feat = state->_features_SLAM.begin();
        Eigen::Matrix3d dR_group;
        while (feat != state->_features_SLAM.end()) {
            Eigen::VectorXd d_feat_se = Eigen::VectorXd(6, 1);
            d_feat_se.head<3>() = dx.segment<3>(state->_pose_MAPtoLOC->R()->id());
            d_feat_se.tail<3>() = dx.segment<3>(feat->second->id());
            Eigen::MatrixXd d_feat_SE = Exp_SEK3(d_feat_se);
            dR_group = Exp_SO3(dx.segment<3>(state->_pose_MAPtoLOC->R()->id()));
            feat->second->set_value(dR_group * feat->second->get_xyz() + d_feat_SE.block(0, 3, 3, 1));
            feat++;
        }
    }
    // MSC-InEKF-FDN
    else if (state->_feature_no_group) {
        auto feat = state->_features_SLAM.begin();
        while (feat != state->_features_SLAM.end()) {
            Eigen::Vector3d d_feat = dx.segment<3>(feat->second->id());
            feat->second->set_value(feat->second->get_xyz() + d_feat);
            feat++;
        }
    }
    // MSC-InEKF-FC
    else {
        auto feat = state->_features_SLAM.begin();
        while (feat != state->_features_SLAM.end()) {
            Eigen::VectorXd d_feat_se = Eigen::VectorXd(6, 1);
            d_feat_se.head<3>() = delta_x_imu.head<3>();
            d_feat_se.tail<3>() = dx.segment<3>(feat->second->id());
            Eigen::MatrixXd d_feat_SE = Exp_SEK3(d_feat_se);
            feat->second->set_value(dR * feat->second->value() + d_feat_SE.block(0, 3, 3, 1));
            feat++;
        }
    }
}

bool StateHelper::initialize(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable, const std::vector<std::shared_ptr<Type>> &H_order,
                             Eigen::MatrixXd &H_R, Eigen::MatrixXd &H_L, Eigen::MatrixXd &R, Eigen::VectorXd &res, double chi_2_mult) {

    // cout << "cov_size: " << state->_Cov.size() << endl;

    // Check that this new variable is not already initialized
    if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
        PRINT_ERROR("StateHelper::initialize_invertible() - Called on variable that is already in the state\n");
        PRINT_ERROR("StateHelper::initialize_invertible() - Found this variable at %d in covariance\n", new_variable->id());
        std::exit(EXIT_FAILURE);
    }

    // Check that we have isotropic noise (i.e. is diagonal and all the same value)
    // TODO: can we simplify this so it doesn't take as much time?
    assert(R.rows() == R.cols());
    assert(R.rows() > 0);
    for (int r = 0; r < R.rows(); r++) {
        for (int c = 0; c < R.cols(); c++) {
            if (r == c && R(0, 0) != R(r, c)) {
                PRINT_ERROR(RED "StateHelper::initialize() - Your noise is not isotropic!\n" RESET);
                PRINT_ERROR(RED "StateHelper::initialize() - Found a value of %.2f verses value of %.2f\n" RESET, R(r, c), R(0, 0));
                std::exit(EXIT_FAILURE);
            } else if (r != c && R(r, c) != 0.0) {
                PRINT_ERROR(RED "StateHelper::initialize() - Your noise is not diagonal!\n" RESET);
                PRINT_ERROR(RED "StateHelper::initialize() - Found a value of %.2f at row %d and column %d\n" RESET, R(r, c), r, c);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    //==========================================================
    //==========================================================
    // First we perform QR givens to seperate the system
    // The top will be a system that depends on the new state, while the bottom does not
    size_t new_var_size = new_variable->size();
    assert((int)new_var_size == H_L.cols());

    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_L.cols(); ++n) {
        for (int m = (int)H_L.rows() - 1; m > n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_L(m - 1, n), H_L(m, n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_L.block(m - 1, n, 2, H_L.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (H_R.block(m - 1, 0, 2, H_R.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
        }
    }

    // Separate into initializing and updating portions
    // 1. Invertible initializing system
    Eigen::MatrixXd Hxinit = H_R.block(0, 0, new_var_size, H_R.cols());
    Eigen::MatrixXd H_finit = H_L.block(0, 0, new_var_size, new_var_size);
    Eigen::VectorXd resinit = res.block(0, 0, new_var_size, 1);
    Eigen::MatrixXd Rinit = R.block(0, 0, new_var_size, new_var_size);

    // 2. Nullspace projected updating system
    Eigen::MatrixXd Hup = H_R.block(new_var_size, 0, H_R.rows() - new_var_size, H_R.cols());
    Eigen::VectorXd resup = res.block(new_var_size, 0, res.rows() - new_var_size, 1);
    Eigen::MatrixXd Rup = R.block(new_var_size, new_var_size, R.rows() - new_var_size, R.rows() - new_var_size);

    //==========================================================
    //==========================================================

    // Do mahalanobis distance testing
    Eigen::MatrixXd P_up = get_marginal_covariance(state, H_order);
    assert(Rup.rows() == Hup.rows());
    assert(Hup.cols() == P_up.cols());
    Eigen::MatrixXd S = Hup * P_up * Hup.transpose() + Rup;
    double chi2 = resup.dot(S.llt().solve(resup));

    // Get what our threshold should be
    boost::math::chi_squared chi_squared_dist(res.rows());
    double chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
    if (chi2 > chi_2_mult * chi2_check) {
        return false;
    }

    //==========================================================
    //==========================================================
    // Finally, initialize it in our state
    StateHelper::initialize_invertible(state, new_variable, H_order, Hxinit, H_finit, Rinit, resinit);

    // Update with updating portion
    if (Hup.rows() > 0) {
        StateHelper::EKFUpdate(state, H_order, Hup, resup, Rup);
    }
    return true;
}

void StateHelper::initialize_invertible(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                                        const std::vector<std::shared_ptr<Type>> &H_order, const Eigen::MatrixXd &H_R, const Eigen::MatrixXd &H_L,
                                        const Eigen::MatrixXd &R, const Eigen::VectorXd &res) {

    // Check that this new variable is not already initialized
    if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
        PRINT_ERROR("StateHelper::initialize_invertible() - Called on variable that is already in the state\n");
        PRINT_ERROR("StateHelper::initialize_invertible() - Found this variable at %d in covariance\n", new_variable->id());
        std::exit(EXIT_FAILURE);
    }

    // Check that we have isotropic noise (i.e. is diagonal and all the same value)
    // TODO: can we simplify this so it doesn't take as much time?
    assert(R.rows() == R.cols());
    assert(R.rows() > 0);
    for (int r = 0; r < R.rows(); r++) {
        for (int c = 0; c < R.cols(); c++) {
            if (r == c && R(0, 0) != R(r, c)) {
                PRINT_ERROR(RED "StateHelper::initialize_invertible() - Your noise is not isotropic!\n" RESET);
                PRINT_ERROR(RED "StateHelper::initialize_invertible() - Found a value of %.2f verses value of %.2f\n" RESET, R(r, c), R(0, 0));
                std::exit(EXIT_FAILURE);
            } else if (r != c && R(r, c) != 0.0) {
                PRINT_ERROR(RED "StateHelper::initialize_invertible() - Your noise is not diagonal!\n" RESET);
                PRINT_ERROR(RED "StateHelper::initialize_invertible() - Found a value of %.2f at row %d and column %d\n" RESET, R(r, c), r, c);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    //==========================================================
    //==========================================================
    // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
    assert(res.rows() == R.rows());
    assert(H_L.rows() == res.rows());
    assert(H_L.rows() == H_R.rows());
    Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());

    // Get the location in small jacobian for each measuring variable
    int current_it = 0;
    std::vector<int> H_id;
    for (const auto &meas_var : H_order) {
        H_id.push_back(current_it);
        current_it += meas_var->size();
    }

    //==========================================================
    //==========================================================
    // For each active variable find its M = P*H^T
    for (const auto &var : state->_variables) {
        // Sum up effect of each subjacobian= K_i= \sum_m (P_im Hm^T)
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for (size_t i = 0; i < H_order.size(); i++) {
            std::shared_ptr<Type> meas_var = H_order.at(i);
            M_i += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                   H_R.block(0, H_id[i], H_R.rows(), meas_var->size()).transpose();
        }
        M_a.block(var->id(), 0, var->size(), res.rows()) = M_i;
    }

    //==========================================================
    //==========================================================
    // Get covariance of this small jacobian
    Eigen::MatrixXd P_small = StateHelper::get_marginal_covariance(state, H_order);

    // M = H_R*Cov*H_R' + R
    Eigen::MatrixXd M(H_R.rows(), H_R.rows());
    M.triangularView<Eigen::Upper>() = H_R * P_small * H_R.transpose();
    M.triangularView<Eigen::Upper>() += R;

    // Covariance of the variable/landmark that will be initialized
    assert(H_L.rows() == H_L.cols());
    assert(H_L.rows() == new_variable->size());
    Eigen::MatrixXd H_Linv = H_L.inverse();
    Eigen::MatrixXd P_LL = H_Linv * M.selfadjointView<Eigen::Upper>() * H_Linv.transpose();

    // Augment the covariance matrix
    size_t oldSize = state->_Cov.rows();
    // cout << "oldSize: " << oldSize << endl;
    // cout << "new_variableSize: " << new_variable->size() << endl;
    state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(oldSize + new_variable->size(), oldSize + new_variable->size()));
    state->_Cov.block(0, oldSize, oldSize, new_variable->size()).noalias() = -M_a * H_Linv.transpose();
    state->_Cov.block(oldSize, 0, new_variable->size(), oldSize) = state->_Cov.block(0, oldSize, oldSize, new_variable->size()).transpose();
    state->_Cov.block(oldSize, oldSize, new_variable->size(), new_variable->size()) = P_LL;

    // Update the variable that will be initialized (invertible systems can only update the new variable).
    // However this update should be almost zero if we already used a conditional Gauss-Newton to solve for the initial estimate
    // for invariant error formulate, \hat{f}-\hat{R}R^{-1}f =\tilde{f}  --> f \approx \hat{f} - \tilde{f}
    // cout<<"initialize_invertable, landmark is "<<new_variable->value().transpose()<<endl;
    // cout<<"update value: "<<(-H_Linv * res).transpose()<<endl;
    new_variable->update(H_Linv * res);
    // cout<<"after update, landmark is "<<new_variable->value().transpose()<<endl;
    // Now collect results, and add it to the state variables
    new_variable->set_local_id(oldSize);
    state->_variables.push_back(new_variable);
    // std::cout << new_variable->id() <<  " init dx = " << (H_Linv * res).transpose() << std::endl;
}

void StateHelper::change_state(std::shared_ptr<State> state) {

    double newest_time = -1;
    double oldest_time = state->margtimestep();

    bool exist_prop_feature = false;
    for (auto const feature : state->_features_SLAM) {
        if (std::abs(feature.second->pseudo_anchor_clone_timestamp - oldest_time) < 1e-7)
            exist_prop_feature = true;
    }
    if (!exist_prop_feature)
        return;

    int prop_size = 0;
    if (state->_feature_in_clone)
        prop_size += state->_features_SLAM.size() * 3;
    // if (state->_multistate_as_group)
    //     prop_size += (state->_clones_IMU.size() - 1) * 3;
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(prop_size + 6, prop_size + 6);

    int current_id = 0;
    std::vector<int> Phi_id;
    std::vector<int> Cov_id;
    std::vector<int> var_size;
    int size_order_OLD = 0;

    for (const auto clone : state->_clones_IMU) {
        if (newest_time < clone.first) {
            newest_time = clone.first;
        }
    }

    std::vector<std::shared_ptr<Type>> order_notprop;
    std::vector<int> id_notprop;
    std::vector<int> var_size_notprop;
    int notprop_size = 0;

    {
        Phi_id.push_back(current_id);
        Cov_id.push_back(state->_clones_IMU[newest_time]->R()->id());
        var_size.push_back(state->_clones_IMU[newest_time]->R()->size());
        current_id += state->_clones_IMU[newest_time]->R()->size();
        size_order_OLD += state->_clones_IMU[newest_time]->R()->size();

        order_notprop.push_back(state->_clones_IMU[newest_time]->p());
        id_notprop.push_back(notprop_size);
        var_size_notprop.push_back(state->_clones_IMU[newest_time]->p()->size());
        notprop_size += state->_clones_IMU[newest_time]->p()->size();

        Phi_id.push_back(current_id);
        Cov_id.push_back(state->_clones_IMU[oldest_time]->R()->id());
        var_size.push_back(state->_clones_IMU[oldest_time]->R()->size());
        current_id += state->_clones_IMU[oldest_time]->R()->size();
        size_order_OLD += state->_clones_IMU[oldest_time]->R()->size();

        order_notprop.push_back(state->_clones_IMU[oldest_time]->p());
        id_notprop.push_back(notprop_size);
        var_size_notprop.push_back(state->_clones_IMU[oldest_time]->p()->size());
        notprop_size += state->_clones_IMU[oldest_time]->p()->size();
    }

    order_notprop.push_back(state->_imu);
    id_notprop.push_back(notprop_size);
    var_size_notprop.push_back(state->_imu->size());
    notprop_size += state->_imu->size();

    if (!state->_feature_in_clone) {
        for (const auto feature : state->_features_SLAM) {
            order_notprop.push_back(feature.second);
            id_notprop.push_back(notprop_size);
            var_size_notprop.push_back(feature.second->size());
            notprop_size += feature.second->size();
        }
    } else {
        auto iter = state->_features_SLAM.begin();
        while (iter != state->_features_SLAM.end()) {
            if (std::abs(iter->second->pseudo_anchor_clone_timestamp - oldest_time) < 1e-7) {
                Phi_id.push_back(current_id);
                Cov_id.push_back(iter->second->id());
                var_size.push_back(iter->second->size());
                Phi.block(current_id, 0, iter->second->size(), 3) = skew(iter->second->get_xyz());
                Phi.block(current_id, 3, iter->second->size(), 3) = -skew(iter->second->get_xyz());
                current_id += iter->second->size();
                size_order_OLD += iter->second->size();
                iter->second->pseudo_anchor_clone_timestamp = newest_time;
                ++iter;
            } else {
                order_notprop.push_back(iter->second);
                id_notprop.push_back(notprop_size);
                var_size_notprop.push_back(iter->second->size());
                notprop_size += iter->second->size();
                ++iter;
            }
        }
    }

    order_notprop.push_back(state->_pose_MAPtoLOC);
    id_notprop.push_back(notprop_size);
    var_size_notprop.push_back(state->_pose_MAPtoLOC->size());
    notprop_size += state->_pose_MAPtoLOC->size();

    for (const auto clone : state->_clones_IMU) {
        if (std::abs(clone.first - newest_time) < 1e-7 || std::abs(clone.first - oldest_time) < 1e-7)
            continue;
        order_notprop.push_back(clone.second);
        id_notprop.push_back(notprop_size);
        var_size_notprop.push_back(clone.second->size());
        notprop_size += clone.second->size();
    }

    if (current_id == 6)
        return;
    else {
        Phi.conservativeResize(current_id, current_id);
    }

    Eigen::MatrixXd Cov = Eigen::MatrixXd::Zero(size_order_OLD, size_order_OLD);
    for (size_t i = 0; i < Cov_id.size(); i++) {
        for (size_t j = 0; j < Cov_id.size(); j++) {
            Cov.block(Phi_id[i], Phi_id[j], var_size[i], var_size[j]) = state->_Cov.block(Cov_id[i], Cov_id[j], var_size[i], var_size[j]);
        }
    }

    Eigen::MatrixXd Cross_Cov = Eigen::MatrixXd::Zero(size_order_OLD, notprop_size);
    for (size_t i = 0; i < Cov_id.size(); i++) {
        for (size_t j = 0; j < order_notprop.size(); j++) {
            Cross_Cov.block(Phi_id[i], id_notprop[j], var_size[i], var_size_notprop[j]) =
                state->_Cov.block(Cov_id[i], order_notprop[j]->id(), var_size[i], var_size_notprop[j]);
        }
    }

    // propagate the covariance
    Eigen::MatrixXd Cov_new = Phi * Cov * Phi.transpose();
    Eigen::MatrixXd Cross_Cov_new = Phi * Cross_Cov;

    // reassgin the Cov_new to state->_Cov
    for (size_t i = 0; i < Cov_id.size(); i++) {
        for (size_t j = 0; j < Cov_id.size(); j++) {
            state->_Cov.block(Cov_id[i], Cov_id[j], var_size[i], var_size[j]) = Cov_new.block(Phi_id[i], Phi_id[j], var_size[i], var_size[j]);
        }

        for (size_t k = 0; k < order_notprop.size(); k++) {
            state->_Cov.block(Cov_id[i], order_notprop[k]->id(), var_size[i], var_size_notprop[k]) =
                Cross_Cov_new.block(Phi_id[i], id_notprop[k], var_size[i], var_size_notprop[k]);

            state->_Cov.block(order_notprop[k]->id(), Cov_id[i], var_size_notprop[k], var_size[i]) =
                Cross_Cov_new.block(Phi_id[i], id_notprop[k], var_size[i], var_size_notprop[k]).transpose();
        }
    }

    Eigen::VectorXd diags = state->_Cov.diagonal();
    bool found_neg = false;
    for (int i = 0; i < diags.rows(); i++) {
        if (diags(i) < 0.0) {
            PRINT_WARNING(RED "StateHelper::change_state() - diagonal at %d is %.2f\n" RESET, i, diags(i));
            found_neg = true;
        }
    }
    if (found_neg) {
        std::exit(EXIT_FAILURE);
    }
}

void StateHelper::change_group(std::shared_ptr<State> state, bool rel_to_clone) {

    if (state->_features_SLAM.empty())
        return;

    int prop_size = 0;
    prop_size += state->_features_SLAM.size() * 3;
    Eigen::MatrixXd Phi;

    int current_id = 0;
    std::vector<int> Phi_id;
    std::vector<int> Cov_id;
    std::vector<int> var_size;
    int size_order_OLD = 0;

    std::vector<std::shared_ptr<Type>> order_notprop;
    std::vector<int> id_notprop;
    std::vector<int> var_size_notprop;
    int notprop_size = 0;
    if (rel_to_clone) {
        Phi = Eigen::MatrixXd::Identity(prop_size + 6, prop_size + 6);

        double newest_time = -1;
        for (const auto clone : state->_clones_IMU) {
            if (newest_time < clone.first) {
                newest_time = clone.first;
            }
        }

        Phi_id.push_back(current_id);
        Cov_id.push_back(state->_pose_MAPtoLOC->R()->id());
        var_size.push_back(state->_pose_MAPtoLOC->R()->size());
        current_id += state->_pose_MAPtoLOC->R()->size();
        size_order_OLD += state->_pose_MAPtoLOC->R()->size();

        order_notprop.push_back(state->_pose_MAPtoLOC->p());
        id_notprop.push_back(notprop_size);
        var_size_notprop.push_back(state->_pose_MAPtoLOC->p()->size());
        notprop_size += state->_pose_MAPtoLOC->p()->size();

        Phi_id.push_back(current_id);
        Cov_id.push_back(state->_clones_IMU[newest_time]->R()->id());
        var_size.push_back(state->_clones_IMU[newest_time]->R()->size());
        current_id += state->_clones_IMU[newest_time]->R()->size();
        size_order_OLD += state->_clones_IMU[newest_time]->R()->size();

        order_notprop.push_back(state->_clones_IMU[newest_time]->p());
        id_notprop.push_back(notprop_size);
        var_size_notprop.push_back(state->_clones_IMU[newest_time]->p()->size());
        notprop_size += state->_clones_IMU[newest_time]->p()->size();

        order_notprop.push_back(state->_imu);
        id_notprop.push_back(notprop_size);
        var_size_notprop.push_back(state->_imu->size());
        notprop_size += state->_imu->size();

        auto iter = state->_features_SLAM.begin();
        while (iter != state->_features_SLAM.end()) {
            Phi_id.push_back(current_id);
            Cov_id.push_back(iter->second->id());
            var_size.push_back(iter->second->size());
            Phi.block(current_id, 0, iter->second->size(), 3) = -skew(iter->second->get_xyz());
            Phi.block(current_id, 3, iter->second->size(), 3) = skew(iter->second->get_xyz());
            current_id += iter->second->size();
            size_order_OLD += iter->second->size();
            iter->second->pseudo_anchor_clone_timestamp = newest_time;
            ++iter;
        }

        for (const auto clone : state->_clones_IMU) {
            if (std::abs(clone.first - newest_time) < 1e-7)
                continue;
            order_notprop.push_back(clone.second);
            id_notprop.push_back(notprop_size);
            var_size_notprop.push_back(clone.second->size());
            notprop_size += clone.second->size();
        }
    } else {
        Phi = Eigen::MatrixXd::Identity(prop_size + 3 + 3 * state->_clones_IMU.size(), prop_size + 3 + 3 * state->_clones_IMU.size());

        Phi_id.push_back(current_id);
        Cov_id.push_back(state->_pose_MAPtoLOC->R()->id());
        var_size.push_back(state->_pose_MAPtoLOC->R()->size());
        current_id += state->_pose_MAPtoLOC->R()->size();
        size_order_OLD += state->_pose_MAPtoLOC->R()->size();

        order_notprop.push_back(state->_pose_MAPtoLOC->p());
        id_notprop.push_back(notprop_size);
        var_size_notprop.push_back(state->_pose_MAPtoLOC->p()->size());
        notprop_size += state->_pose_MAPtoLOC->p()->size();

        map<double, int> times_cloneid;
        int start_id = 1;
        for (const auto clone : state->_clones_IMU) {
            times_cloneid.insert({clone.first, start_id});

            Phi_id.push_back(current_id);
            Cov_id.push_back(clone.second->R()->id());
            var_size.push_back(clone.second->R()->size());
            current_id += clone.second->R()->size();
            size_order_OLD += clone.second->R()->size();

            order_notprop.push_back(clone.second->p());
            id_notprop.push_back(notprop_size);
            var_size_notprop.push_back(clone.second->p()->size());
            notprop_size += clone.second->p()->size();

            ++start_id;
        }

        order_notprop.push_back(state->_imu);
        id_notprop.push_back(notprop_size);
        var_size_notprop.push_back(state->_imu->size());
        notprop_size += state->_imu->size();

        auto iter = state->_features_SLAM.begin();
        while (iter != state->_features_SLAM.end()) {
            Phi_id.push_back(current_id);
            Cov_id.push_back(iter->second->id());
            var_size.push_back(iter->second->size());
            Phi.block(current_id, 0, iter->second->size(), 3) = skew(iter->second->get_xyz());
            Phi.block(current_id, 3 * times_cloneid[iter->second->pseudo_anchor_clone_timestamp], iter->second->size(), 3) =
                -skew(iter->second->get_xyz());
            current_id += iter->second->size();
            size_order_OLD += iter->second->size();
            iter->second->pseudo_anchor_clone_timestamp = -1;
            ++iter;
        }
    }

    // Phi.conservativeResize(current_id, current_id);

    Eigen::MatrixXd Cov = Eigen::MatrixXd::Zero(size_order_OLD, size_order_OLD);
    for (size_t i = 0; i < Cov_id.size(); i++) {
        for (size_t j = 0; j < Cov_id.size(); j++) {
            Cov.block(Phi_id[i], Phi_id[j], var_size[i], var_size[j]) = state->_Cov.block(Cov_id[i], Cov_id[j], var_size[i], var_size[j]);
        }
    }

    Eigen::MatrixXd Cross_Cov = Eigen::MatrixXd::Zero(size_order_OLD, notprop_size);
    for (size_t i = 0; i < Cov_id.size(); i++) {
        for (size_t j = 0; j < order_notprop.size(); j++) {
            Cross_Cov.block(Phi_id[i], id_notprop[j], var_size[i], var_size_notprop[j]) =
                state->_Cov.block(Cov_id[i], order_notprop[j]->id(), var_size[i], var_size_notprop[j]);
        }
    }

    // propagate the covariance
    Eigen::MatrixXd Cov_new = Phi * Cov * Phi.transpose();
    Eigen::MatrixXd Cross_Cov_new = Phi * Cross_Cov;

    // reassgin the Cov_new to state->_Cov
    for (size_t i = 0; i < Cov_id.size(); i++) {
        for (size_t j = 0; j < Cov_id.size(); j++) {
            state->_Cov.block(Cov_id[i], Cov_id[j], var_size[i], var_size[j]) = Cov_new.block(Phi_id[i], Phi_id[j], var_size[i], var_size[j]);
        }

        for (size_t k = 0; k < order_notprop.size(); k++) {
            state->_Cov.block(Cov_id[i], order_notprop[k]->id(), var_size[i], var_size_notprop[k]) =
                Cross_Cov_new.block(Phi_id[i], id_notprop[k], var_size[i], var_size_notprop[k]);

            state->_Cov.block(order_notprop[k]->id(), Cov_id[i], var_size_notprop[k], var_size[i]) =
                Cross_Cov_new.block(Phi_id[i], id_notprop[k], var_size[i], var_size_notprop[k]).transpose();
        }
    }

    Eigen::VectorXd diags = state->_Cov.diagonal();
    bool found_neg = false;
    for (int i = 0; i < diags.rows(); i++) {
        if (diags(i) < 0.0) {
            PRINT_WARNING(RED "StateHelper::change_state() - diagonal at %d is %.2f\n" RESET, i, diags(i));
            found_neg = true;
        }
    }
    if (found_neg) {
        std::exit(EXIT_FAILURE);
    }
}

void StateHelper::marginalize_old_clone(std::shared_ptr<State> state) {
    if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
        double marginal_time = state->margtimestep();
        // Lock the mutex to avoid deleting any elements from _clones_IMU while accessing it from other threads
        std::lock_guard<std::mutex> lock(state->_mutex_state);
        assert(marginal_time != INFINITY);
        StateHelper::marginalize(state, state->_clones_IMU.at(marginal_time));
        // Note that the marginalizer should have already deleted the clone
        // Thus we just need to remove the pointer to it from our state
        state->_clones_IMU.erase(marginal_time);
        if (state->_feature_in_clone) {
            for (const auto feature : state->_features_SLAM) {
                assert(abs(feature.second->pseudo_anchor_clone_timestamp - marginal_time) > 1e-3);
            }
        }
    }
}

void StateHelper::marginalize_slam(std::shared_ptr<State> state) {
    // Remove SLAM features that have their marginalization flag set
    // We also check that we do not remove any aruoctag landmarks
    // cout << "feature_size: " << state->_features_SLAM.size() << endl;
    // cout << "cov_size: " << state->_Cov.size() << endl;
    int ct_marginalized = 0;
    auto it0 = state->_features_SLAM.begin();
    while (it0 != state->_features_SLAM.end()) {
        if ((*it0).second->should_marg) {
            StateHelper::marginalize(state, (*it0).second);
            it0 = state->_features_SLAM.erase(it0);
            ct_marginalized++;
        } else {
            it0++;
        }
    }
}

void StateHelper::marginalize(std::shared_ptr<State> state, std::shared_ptr<Type> marg) {

    // Check if the current state has the element we want to marginalize
    if (std::find(state->_variables.begin(), state->_variables.end(), marg) == state->_variables.end()) {
        PRINT_ERROR(RED "StateHelper::marginalize() - Called on variable that is not in the state\n" RESET);
        PRINT_ERROR(RED "StateHelper::marginalize() - Marginalization, does NOT work on sub-variables yet...\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    // Generic covariance has this form for x_1, x_m, x_2. If we want to remove x_m:
    //
    //  P_(x_1,x_1) P(x_1,x_m) P(x_1,x_2)
    //  P_(x_m,x_1) P(x_m,x_m) P(x_m,x_2)
    //  P_(x_2,x_1) P(x_2,x_m) P(x_2,x_2)
    //
    //  to
    //
    //  P_(x_1,x_1) P(x_1,x_2)
    //  P_(x_2,x_1) P(x_2,x_2)
    //
    // i.e. x_1 goes from 0 to marg_id, x_2 goes from marg_id+marg_size to Cov.rows() in the original covariance

    int marg_size = marg->size();
    int marg_id = marg->id();
    int x2_size = (int)state->_Cov.rows() - marg_id - marg_size;

    Eigen::MatrixXd Cov_new(state->_Cov.rows() - marg_size, state->_Cov.rows() - marg_size);

    // P_(x_1,x_1)
    Cov_new.block(0, 0, marg_id, marg_id) = state->_Cov.block(0, 0, marg_id, marg_id);

    // P_(x_1,x_2)
    Cov_new.block(0, marg_id, marg_id, x2_size) = state->_Cov.block(0, marg_id + marg_size, marg_id, x2_size);

    // P_(x_2,x_1)
    Cov_new.block(marg_id, 0, x2_size, marg_id) = Cov_new.block(0, marg_id, marg_id, x2_size).transpose();

    // P(x_2,x_2)
    Cov_new.block(marg_id, marg_id, x2_size, x2_size) = state->_Cov.block(marg_id + marg_size, marg_id + marg_size, x2_size, x2_size);

    // Now set new covariance
    // state->_Cov.resize(Cov_new.rows(),Cov_new.cols());
    state->_Cov = Cov_new;
    // state->Cov() = 0.5*(Cov_new+Cov_new.transpose());
    assert(state->_Cov.rows() == Cov_new.rows());

    // Now we keep the remaining variables and update their ordering
    // Note: DOES NOT SUPPORT MARGINALIZING SUBVARIABLES YET!!!!!!!
    std::vector<std::shared_ptr<Type>> remaining_variables;
    for (size_t i = 0; i < state->_variables.size(); i++) {
        // Only keep non-marginal states
        if (state->_variables.at(i) != marg) {
            if (state->_variables.at(i)->id() > marg_id) {
                // If the variable is "beyond" the marginal one in ordering, need to "move it forward"
                state->_variables.at(i)->set_local_id(state->_variables.at(i)->id() - marg_size);
            }
            remaining_variables.push_back(state->_variables.at(i));
        }
    }

    // Delete the old state variable to free up its memory
    // NOTE: we don't need to do this any more since our variable is a shared ptr
    // NOTE: thus this is automatically managed, but this allows outside references to keep the old variable
    // delete marg;
    marg->set_local_id(-1);

    // Now set variables as the remaining ones
    state->_variables = remaining_variables;
}

Eigen::MatrixXd StateHelper::get_full_covariance(std::shared_ptr<State> state) {

    // Size of the covariance is the active
    int cov_size = (int)state->_Cov.rows();

    // Construct our return covariance
    Eigen::MatrixXd full_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

    // Copy in the active state elements
    full_cov.block(0, 0, state->_Cov.rows(), state->_Cov.rows()) = state->_Cov;

    // Return the covariance
    return full_cov;
}

} // namespace night_voyager
