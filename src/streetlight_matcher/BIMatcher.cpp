#include "streetlight_matcher/BIMatcher.h"
#include "core/CommonLib.h"
#include "streetlight_matcher/PcdManager.h"

namespace night_voyager {

vector<STMatch> BIMatcher::match(const CameraData &img, std::shared_ptr<PcdManager> pcd, const vector<STMatch> &matches, Eigen::Matrix3d &R_MAPtoC,
                                 Eigen::Vector3d &p_MAPinC) {
    bool low_light = false;
    bi_boxes.clear();
    vector<cv::Rect> pre_boxes = binary_segment(img, matches);
    if (pre_boxes.empty() && matches.empty()) {
        pre_boxes = binary_segment(img, matches, true);
        low_light = true;
        // no boxes detected, directly return
        if (pre_boxes.empty())
            return vector<STMatch>();
    }

    // begin to search streetlights
    vector<ScoreData> STs;
    double min_dist = 0;
    double max_dist = update_z_th;
    STs = search_streetlight(pcd, matches, R_MAPtoC, p_MAPinC, min_dist, max_dist);
    // expand search scope
    if (STs.empty())
        STs = search_streetlight(pcd, matches, R_MAPtoC, p_MAPinC, min_dist + 50, max_dist + 50);
    if (STs.empty()) {
        // no streetlights detected, directly return
        bi_boxes = pre_boxes;
        return vector<STMatch>();
    }

    // Begin to search matches
    vector<STMatch> bi_matches = search_matches(pre_boxes, R_MAPtoC, p_MAPinC, STs, pcd, 4);
    if (bi_matches.empty() && matches.empty()) {
        // If we have not done more loose detection
        if (!low_light) {
            // Perform more loose detection and remove repeated detection
            vector<cv::Rect> boxes = binary_segment(img, matches, true);
            auto it_box = boxes.begin();
            while (it_box != boxes.end()) {
                bool find_rep = false;
                for (const auto &box : pre_boxes) {
                    if ((*it_box & box).area() > 0) {
                        find_rep = true;
                        break;
                    }
                }
                if (find_rep) {
                    it_box = boxes.erase(it_box);
                } else {
                    ++it_box;
                }
            }
            boxes.insert(boxes.end(), pre_boxes.begin(), pre_boxes.end());
            bi_matches = search_matches(boxes, R_MAPtoC, p_MAPinC, STs, pcd, large_off, true);
            bi_boxes = boxes;
        }
        // If we have done more loose detection
        else {
            bi_matches = search_matches(pre_boxes, R_MAPtoC, p_MAPinC, STs, pcd, large_off, true);
            bi_boxes = pre_boxes;
        }
    } else {
        bi_boxes = pre_boxes;
    }

    if (bi_filter)
        match_filter(bi_matches, matches.empty() && remain_match);

    return bi_matches;
}

vector<ScoreData> BIMatcher::search_streetlight(std::shared_ptr<PcdManager> pcd, const vector<STMatch> &matches, Eigen::Matrix3d &R_MAPtoC, Eigen::Vector3d &p_MAPinC,
                                                double min_dist, double max_dist) {

    vector<ScoreData> STs;
    all_STs.clear();
    for (size_t i = 0; i < pcd->center_points.size(); ++i) {
        Eigen::Vector3d cluster_center = pcd->center_points[i];
        if (isnan(cluster_center.norm()))
            continue;

        // Ensure the streetlights nor being matched
        bool overlap = false;
        for (const auto match : matches) {
            if (match.st_id == i) {
                overlap = true;
                break;
            }
        }
        if (overlap)
            continue;

        Eigen::Vector3d Pc;
        Pc = R_MAPtoC * cluster_center + p_MAPinC;
        double inv_z = 1.0 / Pc.z();

        Eigen::Vector3d pt = inv_z * cam->get_K_eigen() * Pc;
        // Ensure the streetlight is in the vision range
        if (Pc.z() > min_dist && Pc.z() < max_dist && pt.x() >= 5 && pt.y() >= 5 && pt.x() < cam->w() - 5 && pt.y() < cam->h() - 5) {

            ScoreData data;
            data.id = i;
            data.Pc = Pc;
            data.pt = pt.head<2>();
            all_STs.push_back(data);

            // Remove streetlights that overlap in image
            bool overlap = false;
            Eigen::Vector3d l1 = Pc.normalized();
            double n1 = Pc.norm();
            for (size_t j = 0; j < matches.size(); j++) {
                Eigen::Vector3d l2 = matches[j].st_center_cam.normalized();
                if ((l1.transpose() * l2)(0) > ang_th) {
                    overlap = true;
                    break;
                }
            }
            if (overlap)
                continue;

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

vector<cv::Rect> BIMatcher::binary_segment(const CameraData &img, const vector<STMatch> &matches, bool is_low) {

    cv::Mat grey_img = img.image;
    cv::Mat bin_img;
    if (is_low)
        cv::threshold(grey_img, bin_img, grey_th_low, 255, cv::THRESH_BINARY);
    else
        cv::threshold(grey_img, bin_img, grey_th, 255, cv::THRESH_BINARY);

    vector<cv::Rect> boxes;
    vector<vector<cv::Point>> contours;
    cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    int edge = 10;
    vector<cv::Rect> edge_boxes;
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours[i]);

        int area = rect.width * rect.height;
        if (is_low && (area < 6 || rect.width < 2 || rect.height < 2))
            continue;
        if (!is_low && (area < 20 || rect.width < 4 || rect.height < 4 || rect.height / rect.width > 6))
            continue;

        if (rect.x >= edge && rect.y >= edge && rect.x + rect.width < bin_img.cols - edge && rect.y + rect.height < bin_img.rows - edge) {
            boxes.push_back(rect);
        } else {
            edge_boxes.push_back(rect);
        }
    }

    // We also find that when the large streetlight bulb locate at the edge of image, some small boxes near the bulb will also be detected, this result can influence the match
    // result remove boxes overlapping with already detected ones
    auto iter_box = boxes.begin();
    while (iter_box != boxes.end()) {
        bool find_rep = false;
        for (size_t i = 0; i < matches.size(); ++i) {
            if (matches[i].st_id < 0)
                continue;
            cv::Rect matched_box = matches[i].rect;
            if (((*iter_box) & matched_box).area() > 0.25 * iter_box->area() || ((*iter_box) & matched_box).area() > 0.25 * matched_box.area()) {
                find_rep = true;
                break;
            }
        }
        if (find_rep) {
            iter_box = boxes.erase(iter_box);
        } else {
            int off = 10;
            for (size_t i = 0; i < edge_boxes.size(); ++i) {
                cv::Rect aug_edge_box(edge_boxes[i].x - off, edge_boxes[i].y - off, edge_boxes[i].width + 2 * off, edge_boxes[i].height + 2 * off);
                if ((aug_edge_box & *iter_box).area() > 0) {
                    find_rep = true;
                    break;
                }
            }
            if (find_rep) {
                iter_box = boxes.erase(iter_box);
            } else {
                ++iter_box;
            }
        }
    }

    return boxes;
}

vector<STMatch> BIMatcher::search_matches(const vector<cv::Rect> &pre_boxes, const Eigen::Matrix3d &R_MAPtoC, const Eigen::Vector3d &p_MAPinC, const vector<ScoreData> &STs,
                                          std::shared_ptr<PcdManager> pcd, int off, bool use_large_extend) {
    vector<STMatch> bi_matches;
    vector<bool> has_match(pre_boxes.size(), false);
    for (const auto st : STs) {

        cv::Rect st_box;
        if (use_large_extend) {
            st_box.x = st.pt.x() - large_extend;
            st_box.width = 2 * large_extend;
            st_box.y = st.pt.y() - 0.5 * large_extend;
            st_box.height = large_extend;
        } else {
            st_box.x = st.pt.x() - extend;
            st_box.y = st.pt.y() - extend;
            st_box.width = 2 * extend;
            st_box.height = 2 * extend;
        }

        int max_num = -1, sec_max_num = -1;
        int min_box_id = -1, sec_min_box_id = -1;
        size_t i = 0;
        for (const auto box : pre_boxes) {
            if ((st_box & box).area() > 0) {
                // We use this vector to pass those box having match
                if (has_match[i])
                    continue;
                int num = 0;

                float min_u = -1, max_u = -1, min_v = -1, max_v = -1;
                for (size_t k = 0; k < pcd->points[st.id].size(); k++) {
                    Eigen::Vector3d Pci = R_MAPtoC * pcd->points[st.id][k] + p_MAPinC;
                    double inv_z = 1.0 / Pci.z();

                    Eigen::Vector3d pti = inv_z * cam->get_K_eigen() * Pci;

                    if (pti.x() >= box.tl().x - off && pti.y() >= box.tl().y - off && pti.x() <= box.br().x + off && pti.y() <= box.br().y + off) {
                        ++num;
                    }

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
                if (num != 0 && proj_rect.area() < 3 * box.area()) {
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
            i++;
        }
        // The box with most projected points are determined as the match of streetlight
        if (max_num > 0 && max_num >= 2 * sec_max_num) {
            STMatch match;
            match.rect = pre_boxes[min_box_id];
            match.rect_center = 0.5 * Eigen::Vector2f(match.rect.tl().x + match.rect.br().x, match.rect.tl().y + match.rect.br().y);
            match.st_center_map = pcd->center_points[st.id];
            match.st_center_cam = st.Pc;
            match.st_id = st.id;
            bi_matches.push_back(match);
            has_match[min_box_id] = true;
        }
        // If there exists close boxes possessing similar number of projected points, we choose the large one
        else if (max_num > 0 && max_num < 2 * sec_max_num) {
            STMatch match;
            if (pre_boxes[min_box_id].area() > pre_boxes[sec_min_box_id].area()) {
                match.rect = pre_boxes[min_box_id];
                match.rect_center = 0.5f * Eigen::Vector2f(match.rect.tl().x + match.rect.br().x, match.rect.tl().y + match.rect.br().y);
                has_match[min_box_id] = true;
            } else {
                match.rect = pre_boxes[sec_min_box_id];
                match.rect_center = 0.5f * Eigen::Vector2f(match.rect.tl().x + match.rect.br().x, match.rect.tl().y + match.rect.br().y);
                has_match[sec_min_box_id] = true;
            }
            match.st_center_map = pcd->center_points[st.id];
            match.st_center_cam = st.Pc;
            match.st_id = st.id;
            bi_matches.push_back(match);
        }
    }

    return bi_matches;
}

void BIMatcher::match_filter(std::vector<STMatch> &matches, bool remain_match) {
    if (matches.empty())
        return;
    std::vector<STMatch> matches_copy = matches;
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
    if (remain_match && matches.empty()) {
        matches.push_back(matches_copy[0]);
    }
}

} // namespace night_voyager