#include "streetlight_matcher/DLMatcher.h"
#include "streetlight_matcher/HungaryEstimator.h"
#include "streetlight_matcher/PcdManager.h"
#include "utils/Transform.h"

namespace night_voyager {

vector<STMatch> DLMatcher::match(const BoxData &boxes, std::shared_ptr<PcdManager> pcd, Eigen::Matrix3d &R_MAPtoC, Eigen::Vector3d &p_MAPinC, Eigen::MatrixXd &pose_cov) {
    dl_boxes.clear();
    if (boxes.rects.size() == 0) {
        // no boxes detected, directly return
        return vector<STMatch>();
    }

    vector<ScoreData> STs;
    STs = search_streetlight(pcd, R_MAPtoC, p_MAPinC);

    if (STs.empty()) {
        // no streetlights detected, directly return
        return vector<STMatch>();
    }

    /// Begin to calculate the score of each match configuration
    int box_nums = boxes.rects.size();
    /// We expand the number of streetlights with 1 for no match of box
    Eigen::MatrixXd md_matrix = Eigen::MatrixXd::Zero(box_nums, STs.size() + 1);
    Eigen::MatrixXd ad_matrix = Eigen::MatrixXd::Zero(box_nums, STs.size() + 1);

    // cout << "pose_cov: " << endl << pose_cov << endl;

    /// Calculate the covariance
    assert(pose_cov.rows() == 6);
    for (size_t i = 0; i < STs.size(); ++i) {
        double inv_z = 1.0 / STs[i].Pc.z();

        Eigen::Matrix<double, 2, 3> J_pt_Pc;
        J_pt_Pc << cam->get_K_eigen()(0, 0) * inv_z, 0, -cam->get_K_eigen()(0, 0) * STs[i].pt.x() * inv_z * inv_z, 0, cam->get_K_eigen()(1, 1) * inv_z,
            -cam->get_K_eigen()(1, 1) * STs[i].pt.y() * inv_z * inv_z;

        Eigen::Matrix<double, 3, 6> J_Pc_pose;
        J_Pc_pose.block<3, 3>(0, 0) = -skew(STs[i].Pc);
        J_Pc_pose.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 2, 6> J_pt_pose;
        J_pt_pose = J_pt_Pc * J_Pc_pose;

        Eigen::Matrix<double, 2, 3> J_pt_Pw;
        J_pt_Pw = J_pt_Pc * R_MAPtoC;

        Eigen::Matrix2d HPH_R = J_pt_pose * pose_cov * J_pt_pose.transpose();        // + J_pt_Pw * streetlight_cov * J_pt_Pw.transpose();
        Eigen::Matrix3d liftup_HPH_R = J_Pc_pose * pose_cov * J_Pc_pose.transpose(); // + R_MAPtoC * streetlight_cov * R_MAPtoC.transpose();
        STs[i].HPH_R = HPH_R;
        STs[i].liftup_HPH_R = liftup_HPH_R;
    }

    // Calculate the scores
    for (int i = 0; i < box_nums; i++) {

        Eigen::Matrix2d cov_box_center = 9 * Eigen::Matrix2d::Identity();

        for (size_t j = 0; j < STs.size(); j++) {
            md_matrix(i, j) = md_distance(boxes.centers[i], STs[j].pt, STs[j].HPH_R + cov_box_center);
            ad_matrix(i, j) = ang_distance(boxes.centers[i], STs[j].Pc, STs[j].liftup_HPH_R, cov_box_center);
        }
    }

    int max_edge = box_nums + int(STs.size());
    Eigen::MatrixXd cost = Eigen::MatrixXd::Zero(max_edge, max_edge);
    cost.block(0, 0, box_nums, STs.size() + 1) = alpha * md_matrix + (1 - alpha) * ad_matrix;

    // cout << "pose_cov: " << endl << pose_cov << endl;
    // cout << "md_matrix: " << endl << md_matrix << endl;
    // cout << "ad_matrix: " << endl << ad_matrix << endl;

    for (int i = 0; i < box_nums; i++) {
        if (i == 0) {
            cost.block(0, STs.size(), box_nums, 1) = Eigen::VectorXd::Ones(box_nums) - cost.block(0, 0, box_nums, STs.size()).rowwise().sum();
        } else
            cost.block(0, STs.size() + i, box_nums, 1) = cost.block(0, STs.size(), box_nums, 1);
    }

    double max_emt = cost.maxCoeff();
    cost = max_emt * Eigen::MatrixXd::Ones(cost.rows(), cost.cols()) - cost;

    // Begin to search match using Hungary method
    HungaryEstimator estimator(cost);
    std::vector<int> result = estimator.solve();

    // Check the result
    vector<STMatch> M;
    for (int i = 0; i < result.size(); i++) {
        if (i >= box_nums)
            break;
        // invalid match if cost value is too large
        if (result[i] < STs.size() && cost(i, result[i]) >= 0.8) {
            result[i] = STs.size();
            double tmp = cost(i, STs.size());
            for (int j = STs.size(); j < cost.cols(); j++) {
                cost(i, j) = cost(i, result[i]);
            }
            cost(i, result[i]) = tmp;
        }
        if (result[i] < STs.size()) {
            float min_u = -1, max_u = -1, min_v = -1, max_v = -1;
            for (size_t k = 0; k < pcd->points[STs[result[i]].id].size(); k++) {
                Eigen::Vector3d Pci = R_MAPtoC * pcd->points[STs[result[i]].id][k] + p_MAPinC;
                double inv_z = 1.0 / Pci.z();

                Eigen::Vector3d pti = inv_z * cam->get_K_eigen() * Pci;

                if (min_u > pti.x() || min_u < 0)
                    min_u = pti.x();
                if (min_v > pti.y() || min_v < 0)
                    min_v = pti.y();
                if (max_u < pti.x() || max_u < 0)
                    max_u = pti.x();
                if (max_v < pti.y() || max_v < 0)
                    max_v = pti.y();
            }
            cv::Rect proj_rect;
            if (min_u < 0 || min_v < 0 || max_u < 0 || max_v < 0)
                continue;
            else {
                proj_rect.x = int(min_u);
                proj_rect.y = int(min_v);
                proj_rect.width = int(max_u - min_u);
                proj_rect.height = int(max_v - min_v);
            }

            if (proj_rect.area() < 3 * boxes.rects[i].area()) {
                STMatch m;
                m.st_id = STs[result[i]].id;
                m.rect = boxes.rects[i];
                m.rect_center = boxes.centers[i];
                m.st_center_map = pcd->center_points[m.st_id];
                m.st_center_cam = STs[result[i]].Pc;
                M.push_back(m);
            }
        }
    }
    dl_boxes = boxes.rects;

    if (dl_filter)
        match_filter(M);
    // for(size_t i = 0; i < M.size(); ++i){
    //     cout << "center: " << M[i].rect_center.transpose() << "  " << "id: " << M[i].st_id << endl;
    // }

    return M;
}

vector<STMatch> DLMatcher::match_tracking_recover(const BoxData &boxes, std::shared_ptr<PcdManager> pcd, Eigen::Matrix3d &R_MAPtoC, Eigen::Vector3d &p_MAPinC,
                                                  double reloc_z_th, int reloc_extend) {

    vector<ScoreData> STs = search_streetlight_tracking_recover(pcd, R_MAPtoC, p_MAPinC, reloc_z_th);
    if (STs.empty()) {
        // no streetlights detected, directly return
        return vector<STMatch>();
    }

    // Begin to search matches
    vector<bool> has_match(boxes.rects.size(), false);
    vector<STMatch> dl_matches;
    for (const auto &st : STs) {
        cv::Rect st_box(cv::Point2i(st.pt.x() - reloc_extend, st.pt.y() - reloc_extend), cv::Point2i(st.pt.x() + reloc_extend, st.pt.y() + reloc_extend));

        int max_num = -1, sec_max_num = -1;
        int min_box_id = -1, sec_min_box_id = -1;
        size_t i = 0;
        for (const auto &box : boxes.rects) {
            if ((st_box & box).area() > 0) {
                if (has_match[i])
                    continue;
                int num = 0;
                int off = 20;
                for (size_t k = 0; k < pcd->points[st.id].size(); ++k) {
                    Eigen::Vector3d Pci = R_MAPtoC * pcd->points[st.id][k] + p_MAPinC;
                    double inv_z = 1.0 / Pci.z();

                    Eigen::Vector3d pti = inv_z * cam->get_K_eigen() * Pci;
                    if (pti.x() >= box.tl().x - off && pti.y() >= box.tl().y - off && pti.x() <= box.br().x + off && pti.y() <= box.br().y + off) {
                        ++num;
                    }
                }

                if (num != 0) {
                    if (max_num < num) {
                        sec_max_num = max_num;
                        sec_min_box_id = min_box_id;

                        max_num = num;
                        min_box_id = i;
                    } else if (max_num >= num && sec_max_num < num) {
                        sec_max_num = num;
                        sec_min_box_id = i;
                    }
                }
            }
            ++i;
        }

        if (max_num > 0 && max_num >= 2 * sec_max_num) {
            STMatch match;
            match.rect = boxes.rects[min_box_id];
            match.rect_center = boxes.centers[min_box_id];
            match.st_center_map = pcd->center_points[st.id];
            match.st_center_cam = st.Pc;
            match.st_id = st.id;
            dl_matches.push_back(match);
            has_match[min_box_id] = true;
        }

        else if (max_num > 0 && max_num < 2 * sec_max_num) {
            STMatch match;
            if (boxes.rects[min_box_id].area() > boxes.rects[sec_min_box_id].area()) {
                match.rect = boxes.rects[min_box_id];
                match.rect_center = boxes.centers[min_box_id];
                has_match[min_box_id] = true;
            } else {
                match.rect = boxes.rects[sec_min_box_id];
                match.rect_center = boxes.centers[sec_min_box_id];
                has_match[sec_min_box_id] = true;
            }
            match.st_center_map = pcd->center_points[st.id];
            match.st_center_cam = st.Pc;
            match.st_id = st.id;
            dl_matches.push_back(match);
        }
    }

    return dl_matches;
}

vector<STMatch> DLMatcher::match_box(const BoxData &boxes, std::shared_ptr<PcdManager> pcd, vector<STMatch> &matches, Eigen::Matrix3d &R_MAPtoC, Eigen::Vector3d &p_MAPinC) {

    vector<STMatch> matches_box;
    vector<ScoreData> STs;
    STs = search_streetlight(pcd, R_MAPtoC, p_MAPinC);

    auto it_ST = STs.begin();
    while (it_ST != STs.end()) {
        bool has_rep = false;
        for (const auto &match : matches) {
            if (it_ST->id == match.st_id) {
                has_rep = true;
                break;
            }
        }
        if (has_rep)
            it_ST = STs.erase(it_ST);
        else
            ++it_ST;
    }

    vector<vector<cv::Point2i>> proj_STs;
    proj_STs.resize(STs.size());
    for (size_t i = 0; i < STs.size(); ++i) {
        vector<cv::Point2i> proj_points;
        for (const auto &point : pcd->points[STs[i].id]) {
            Eigen::Vector3d Pc = R_MAPtoC * point + p_MAPinC;
            double inv_z = 1 / Pc.z();
            Eigen::Vector3d pt = inv_z * cam->get_K_eigen() * Pc;

            proj_points.push_back(cv::Point2i(int(pt.x()), int(pt.y())));
        }
        proj_STs[i] = proj_points;
    }

    for (const auto &rect : boxes.rects) {

        STMatch match_box;
        bool has_match = false;
        for (const auto &match : matches) {
            if ((rect & match.rect).area() / (rect | match.rect).area() > 0.95)
                has_match = true;
        }
        if (has_match)
            continue;
        float center_x = 0.5 * rect.width + rect.x, center_y = 0.5 * rect.height + rect.y;
        cv::Rect slim_rect(int(center_x - 5), int(center_y - 100), 10, 200);

        float max_ratio = 0.0;
        int max_idx = -1;
        for (size_t i = 0; i < proj_STs.size(); ++i) {
            int num = 0;
            for (const auto &pt : proj_STs[i]) {
                if (slim_rect.contains(pt))
                    ++num;
            }
            if (max_ratio <= 0.8 * float(num) / proj_STs[i].size()) {
                max_ratio = float(num) / proj_STs[i].size();
                max_idx = i;
            } else {
                if ((Eigen::Vector2d(center_x, center_y) - STs[i].pt).norm() < (Eigen::Vector2d(center_x, center_y) - STs[max_idx].pt).norm()) {
                    max_ratio = float(num) / proj_STs[i].size();
                    max_idx = i;
                }
            }
        }

        if (max_ratio > 0 && max_idx >= 0) {
            match_box.rect = rect;
            match_box.rect_center = Eigen::Vector2f(rect.x + 0.5f * rect.width, rect.y + 0.5f * rect.height);
            match_box.st_center_cam = STs[max_idx].Pc;
            match_box.st_center_map = pcd->center_points[STs[max_idx].id];
            match_box.st_id = STs[max_idx].id;
            matches_box.push_back(match_box);
        }
    }
    return matches_box;
}

vector<ScoreData> DLMatcher::search_streetlight(std::shared_ptr<PcdManager> pcd, const Eigen::Matrix3d &R_MAPtoC, const Eigen::Vector3d &p_MAPinC) {

    vector<ScoreData> STs;
    all_STs.clear();
    for (size_t i = 0; i < pcd->center_points.size(); ++i) {
        Eigen::Vector3d cluster_center = pcd->center_points[i];
        if (isnan(cluster_center.norm()))
            continue;

        Eigen::Vector3d Pc;
        Pc = R_MAPtoC * cluster_center + p_MAPinC;
        double inv_z = 1.0 / Pc.z();

        Eigen::Vector3d pt = inv_z * cam->get_K_eigen() * Pc;
        // Ensure the streetlight is in the vision range
        if (Pc.z() > 0.5 && Pc.z() < z_th && pt.x() >= 0 && pt.y() >= 0 && pt.x() < cam->w() && pt.y() < cam->h()) {

            ScoreData data;
            data.id = i;
            data.Pc = Pc;
            data.pt = pt.head<2>();
            all_STs.push_back(data);

            // Remove streetlights that overlap in image
            bool overlap = false;
            Eigen::Vector3d l1 = Pc.normalized();
            double n1 = Pc.norm();
            for (size_t j = 0; j < STs.size(); j++) {
                Eigen::Vector3d l2 = STs[j].Pc.normalized();
                if ((l1.transpose() * l2)(0) > ang_th) {
                    double n2 = STs[j].Pc.norm();
                    overlap = true;
                    // select the closer streetlight
                    if (n2 > n1) {
                        STs[j].id = i;
                        STs[j].Pc = Pc;
                        STs[j].pt = pt.head<2>();
                    }
                    break;
                }
            }
            if (overlap)
                continue;
            STs.push_back(data);
        }
    }
    return STs;
}

vector<ScoreData> DLMatcher::search_streetlight_tracking_recover(std::shared_ptr<PcdManager> pcd, const Eigen::Matrix3d &R_MAPtoC, const Eigen::Vector3d &p_MAPinC,
                                                                 double reloc_z_th) {

    vector<ScoreData> STs;
    for (size_t i = 0; i < pcd->center_points.size(); ++i) {
        Eigen::Vector3d cluster_center = pcd->center_points[i];
        if (isnan(cluster_center.norm()))
            continue;

        Eigen::Vector3d Pc;
        Pc = R_MAPtoC * cluster_center + p_MAPinC;
        double inv_z = 1.0 / Pc.z();

        Eigen::Vector3d pt = inv_z * cam->get_K_eigen() * Pc;
        // Ensure the streetlight is in the vision range
        if (Pc.z() > 0.5 && Pc.z() < z_th && pt.x() >= 0 && pt.y() >= 0 && pt.x() < cam->w() && pt.y() < cam->h()) {

            ScoreData data;
            data.id = i;
            data.Pc = Pc;
            data.pt = pt.head<2>();

            // Remove streetlights that overlap in image
            bool overlap = false;
            Eigen::Vector3d l1 = Pc.normalized();
            double n1 = Pc.norm();
            for (size_t j = 0; j < STs.size(); j++) {
                Eigen::Vector3d l2 = STs[j].Pc.normalized();
                if ((l1.transpose() * l2)(0) > 0.9998) {
                    double n2 = STs[j].Pc.norm();
                    overlap = true;
                    // select the closer streetlight
                    if (n2 > n1) {
                        STs[j].id = i;
                        STs[j].Pc = Pc;
                        STs[j].pt = pt.head<2>();
                    }
                    break;
                }
            }
            if (overlap)
                continue;
            STs.push_back(data);
        }
    }
    return STs;
}

void DLMatcher::match_filter(std::vector<STMatch> &matches) {
    auto it_match = matches.begin();
    int off = 4;
    while (it_match != matches.end()) {
        int contain_points = 0;
        for (size_t i = 0; i < all_STs.size(); ++i) {
            if (all_STs[i].pt.x() < it_match->rect.br().x + off && all_STs[i].pt.x() >= it_match->rect.tl().x - off && all_STs[i].pt.y() < it_match->rect.br().y + off &&
                all_STs[i].pt.y() >= it_match->rect.tl().y - off) {
                ++contain_points;
            }
            Eigen::Vector3d unit_matched_cam = it_match->st_center_cam.normalized();
            Eigen::Vector3d unit_ST_cam = all_STs[i].Pc.normalized();
            if ((unit_matched_cam.transpose() * unit_ST_cam)(0) > 0.99995) {
                --contain_points;
            }
        }
        if (contain_points >= 1) {
            it_match = matches.erase(it_match);
        } else {
            ++it_match;
        }
    }
}

double DLMatcher::md_distance(const Eigen::Vector2f &box_center, const Eigen::Vector2d &proj_center, const Eigen::Matrix2d &cov_box_center) {

    Eigen::Vector2d err(Eigen::Vector2d(box_center.x(), box_center.y()) - proj_center);
    // cout << "err " << err.transpose() << endl;
    Eigen::Vector2d n_err = err.normalized();
    double sigma2 = n_err.transpose() * cov_box_center * n_err;
    // cout << "sigma2: " << endl << sigma2 << endl << endl;
    return exp(-0.5 * err.squaredNorm() / sigma2);
}

double DLMatcher::ang_distance(const Eigen::Vector2f &box_center, const Eigen::Vector3d &streetlight_center, const Eigen::Matrix3d &cov_streetlight_center,
                               const Eigen::Matrix2d &cov_box_center) {

    // Eigen::Vector3d liftup_center;
    // liftup_center << cam->get_invK_eigen() * Eigen::Vector3d(box_center.x(), box_center.y(), 1);

    // double inv_norm_liftup_center = 1.0 / liftup_center.norm();
    // double inv_norm_streetlight_center = 1.0 / streetlight_center.norm();
    // Eigen::Vector3d normd_liftup_center = liftup_center.normalized();
    // Eigen::Vector3d normd_streetlight_center = streetlight_center.normalized();

    // double cos_theta = normd_streetlight_center.transpose() * normd_liftup_center;
    // double err = 1 - cos_theta;
    // cout << err << endl;

    // Eigen::Matrix3d cov_liftup_center = Eigen::Matrix3d::Zero();
    // cov_liftup_center(0, 0) = cov_box_center(0, 0) / (cam->get_K_eigen()(0, 0) * cam->get_K_eigen()(0, 0));
    // cov_liftup_center(1, 1) = cov_box_center(1, 1) / (cam->get_K_eigen()(1, 1) * cam->get_K_eigen()(1, 1));

    // Eigen::Vector3d jaco_cos_liftup_center, jaco_cos_streetlight_center;
    // jaco_cos_liftup_center = inv_norm_liftup_center * (normd_streetlight_center - normd_streetlight_center.transpose() * normd_liftup_center * normd_liftup_center);
    // jaco_cos_streetlight_center = inv_norm_streetlight_center * (normd_liftup_center - normd_liftup_center.transpose() * normd_streetlight_center *
    // normd_streetlight_center);

    // double cov_cos_theta = (jaco_cos_liftup_center.transpose() * cov_liftup_center * jaco_cos_liftup_center)(0) +
    //                        (jaco_cos_streetlight_center.transpose() * cov_streetlight_center * jaco_cos_streetlight_center)(0);
    // // cout << "jaco_cos_streetlight_center" << (jaco_cos_streetlight_center.transpose() * cov_streetlight_center * jaco_cos_streetlight_center) << "
    // jaco_cos_liftup_center" << (jaco_cos_liftup_center.transpose() * cov_liftup_center * jaco_cos_liftup_center) << endl;

    // return exp(-0.5 * err * err / cov_cos_theta);

    Eigen::Vector3d liftup_center;
    liftup_center << cam->get_invK_eigen() * Eigen::Vector3d(box_center.x(), box_center.y(), 1);
    Eigen::Matrix3d cov_liftup_center = Eigen::Matrix3d::Zero();
    cov_liftup_center(0, 0) = cov_box_center(0, 0) / (cam->get_K_eigen()(0, 0) * cam->get_K_eigen()(0, 0));
    cov_liftup_center(1, 1) = cov_box_center(1, 1) / (cam->get_K_eigen()(1, 1) * cam->get_K_eigen()(1, 1));

    double inv_norm_liftup_center = 1.0 / liftup_center.norm();
    double inv_norm_streetlight_center = 1.0 / streetlight_center.norm();
    Eigen::Vector3d normd_liftup_center = liftup_center.normalized();
    Eigen::Vector3d normd_streetlight_center = streetlight_center.normalized();
    Eigen::Matrix3d skew_normd_liftup_center = skew(normd_liftup_center);
    Eigen::Matrix3d skew_normd_streetlight_center = skew(normd_streetlight_center);

    Eigen::Vector3d cross = skew_normd_liftup_center * normd_streetlight_center;
    double sin_theta = cross.norm();

    Eigen::RowVector3d jaco_sin_liftup_center, jaco_sin_streeltight_center;
    jaco_sin_liftup_center = -inv_norm_liftup_center * cross.normalized().transpose() * skew_normd_streetlight_center *
                             (Eigen::Matrix3d::Identity() - normd_liftup_center * normd_liftup_center.transpose());
    jaco_sin_streeltight_center = inv_norm_streetlight_center * cross.normalized().transpose() * skew_normd_liftup_center *
                                  (Eigen::Matrix3d::Identity() - normd_streetlight_center * normd_streetlight_center.transpose());

    double cov_sin_theta = (jaco_sin_liftup_center * cov_liftup_center * jaco_sin_liftup_center.transpose() +
                            jaco_sin_streeltight_center * cov_streetlight_center * jaco_sin_streeltight_center.transpose())(0);
    return exp(-0.5 * sin_theta * sin_theta / cov_sin_theta);
}

} // namespace night_voyager