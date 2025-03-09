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
#include "feature_tracker/TrackKLT.h"
#include "core/CommonLib.h"
#include "feature_tracker/Feature.h"
#include "feature_tracker/FeatureDatabase.h"
#include "feature_tracker/Grider_GRID.h"
#include "utils/Print.h"

namespace night_voyager {
void TrackKLT::feed_new_camera(const CameraData &message) {

    rT1 = boost::posix_time::microsec_clock::local_time();

    {
        std::lock_guard<std::mutex> lck(mtx_feed);

        // Histogram equalize
        cv::Mat img;
        if (histogram_method == HistogramMethod::HISTOGRAM) {
            cv::equalizeHist(message.image, img);
        } else if (histogram_method == HistogramMethod::CLAHE) {
            double eq_clip_limit = 10.0;
            cv::Size eq_win_size = cv::Size(8, 8);
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
            clahe->apply(message.image, img);
        } else {
            img = message.image;
        }

        // Extract image pyramid
        std::vector<cv::Mat> imgpyr;
        cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);

        // Save!
        img_curr = img;
        img_pyramid_curr = imgpyr;
    }

    feed_monocular(message);
}

void TrackKLT::feed_monocular(const CameraData &message) {

    // Lock this data feed for this camera
    std::lock_guard<std::mutex> lck(mtx_feed);

    // Get our image objects for this image
    cv::Mat img = img_curr;
    std::vector<cv::Mat> imgpyr = img_pyramid_curr;
    // cv::Mat mask = message.masks;
    rT2 = boost::posix_time::microsec_clock::local_time();

    // If we didn't have any successful tracks last time, just extract this time
    // This also handles, the tracking initalization on the first call to this extractor
    if (pts_last.empty()) {
        // Detect new features
        std::vector<cv::KeyPoint> good_left;
        std::vector<size_t> good_ids_left;
        perform_detection_monocular(imgpyr, good_left, good_ids_left);
        // Save the current image and pyramid
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_last = img;
        img_pyramid_last = imgpyr;
        // img_mask_last = mask;
        pts_last = good_left;
        ids_last = good_ids_left;
        return;
    }

    // First we should make that the last images have enough features so we can do KLT
    // This will "top-off" our number of tracks so always have a constant number
    int pts_before_detect = (int)pts_last.size();
    auto pts_left_old = pts_last;
    auto ids_left_old = ids_last;
    perform_detection_monocular(img_pyramid_last, pts_left_old, ids_left_old);
    rT3 = boost::posix_time::microsec_clock::local_time();

    // Our return success masks, and predicted new features
    std::vector<uchar> mask_ll;
    std::vector<cv::KeyPoint> pts_left_new = pts_left_old;

    // Lets track temporally
    perform_matching(img_pyramid_last, imgpyr, pts_left_old, pts_left_new, mask_ll);
    assert(pts_left_new.size() == ids_left_old.size());
    rT4 = boost::posix_time::microsec_clock::local_time();

    // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
    if (mask_ll.empty()) {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_last = img;
        img_pyramid_last = imgpyr;
        // img_mask_last = mask;
        pts_last.clear();
        ids_last.clear();
        PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
        return;
    }

    // Get our "good tracks"
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;

    // Loop through all left points
    for (size_t i = 0; i < pts_left_new.size(); i++) {
        // Ensure we do not have any bad KLT tracks (i.e., points are negative)
        if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= img.cols ||
            (int)pts_left_new.at(i).pt.y >= img.rows)
            continue;
        // Check if it is in the mask
        // NOTE: mask has max value of 255 (white) if it should be
        // if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
        // continue;
        // If it is a good track, and also tracked from left to right
        if (mask_ll[i]) {
            good_left.push_back(pts_left_new[i]);
            good_ids_left.push_back(ids_left_old[i]);
        }
    }

    // Update our feature database, with theses new observations
    for (size_t i = 0; i < good_left.size(); i++) {
        cv::Point2f npt_l = camera_calib->undistort_cv(good_left.at(i).pt);
        database->update_feature(good_ids_left.at(i), message.timestamp, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x, npt_l.y);
    }

    // Move forward in time
    {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_last = img;
        img_pyramid_last = imgpyr;
        // img_mask_last = mask;
        pts_last = good_left;
        ids_last = good_ids_left;
    }
    rT5 = boost::posix_time::microsec_clock::local_time();

    // Timing information
    PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
    PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%d detected)\n", (rT3 - rT2).total_microseconds() * 1e-6,
              (int)pts_last.size() - pts_before_detect);
    PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", (rT4 - rT3).total_microseconds() * 1e-6);
    PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", (rT5 - rT4).total_microseconds() * 1e-6, (int)good_left.size());
    PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}

void TrackKLT::perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, std::vector<cv::KeyPoint> &pts0, std::vector<size_t> &ids0) {

    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less than grid_px_size points away then existing features
    cv::Size size_close((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                        (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
    cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
    float size_x = (float)img0pyr.at(0).cols / (float)grid_x;
    float size_y = (float)img0pyr.at(0).rows / (float)grid_y;
    cv::Size size_grid(grid_x, grid_y); // width x height
    cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
    cv::Mat mask0_updated = cv::Mat::zeros(cv::Size(img0pyr.at(0).cols, img0pyr.at(0).rows), CV_8UC1);
    auto it0 = pts0.begin();
    auto it1 = ids0.begin();
    while (it0 != pts0.end()) {
        // Get current left keypoint, check that it is in bounds
        cv::KeyPoint kpt = *it0;
        int x = (int)kpt.pt.x;
        int y = (int)kpt.pt.y;
        int edge = 10;
        if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate mask coordinates for close points
        int x_close = (int)(kpt.pt.x / (float)min_px_dist);
        int y_close = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate what grid cell this feature is in
        int x_grid = std::floor(kpt.pt.x / size_x);
        int y_grid = std::floor(kpt.pt.y / size_y);
        if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Check if this keypoint is near another point
        if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Now check if it is in a mask area or not
        // NOTE: mask has max value of 255 (white) if it should be
        // if (mask0.at<uint8_t>(y, x) > 127) {
        // it0 = pts0.erase(it0);
        // it1 = ids0.erase(it1);
        // continue;
        // }
        // Else we are good, move forward to the next point
        grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
        if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) {
            grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
        }
        // Append this to the local mask of the image
        if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
            cv::Point pt1(x - min_px_dist, y - min_px_dist);
            cv::Point pt2(x + min_px_dist, y + min_px_dist);
            cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
        }
        it0++;
        it1++;
    }

    // First compute how many more features we need to extract from this image
    // If we don't need any features, just return
    double min_feat_percent = 0.50;
    int num_featsneeded = num_features - (int)pts0.size();
    if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
        return;

    // This is old extraction code that would extract from the whole image
    // This can be slow as this will recompute extractions for grid areas that we have max features already
    // std::vector<cv::KeyPoint> pts0_ext;
    // Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid.cols; x++) {
        for (int y = 0; y < grid_2d_grid.rows; y++) {
            if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req) {
                valid_locs.emplace_back(x, y);
            }
        }
    }
    std::vector<cv::KeyPoint> pts0_ext;
    Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    // Now, reject features that are close a current feature
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext) {
        // Check that it is in bounds
        int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
        int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
            continue;
        // See if there is a point at this location
        if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
            continue;
        // Else lets add it!
        kpts0_new.push_back(kpt);
        pts0_new.push_back(kpt.pt);
        grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
    }

    // Loop through and record only ones that are valid
    // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
    // NOTE: this is due to the fact that we select update features based on feat id
    // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
    // NOTE: not sure how to remove... maybe a better way?
    for (size_t i = 0; i < pts0_new.size(); i++) {
        // update the uv coordinates
        kpts0_new.at(i).pt = pts0_new.at(i);
        // append the new uv coordinate
        pts0.push_back(kpts0_new.at(i));
        // move id foward and append this new point
        size_t temp = ++currid;
        ids0.push_back(temp);
    }
}

void TrackKLT::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, std::vector<uchar> &mask_out) {

    // We must have equal vectors
    assert(kpts0.size() == kpts1.size());

    // Return if we don't have any points
    if (kpts0.empty() || kpts1.empty())
        return;

    // Convert keypoints into points (stupid opencv stuff)
    std::vector<cv::Point2f> pts0, pts1;
    for (size_t i = 0; i < kpts0.size(); i++) {
        pts0.push_back(kpts0.at(i).pt);
        pts1.push_back(kpts1.at(i).pt);
    }

    // If we don't have enough points for ransac just return empty
    // We set the mask to be all zeros since all points failed RANSAC
    if (pts0.size() < 10) {
        for (size_t i = 0; i < pts0.size(); i++)
            mask_out.push_back((uchar)0);
        return;
    }

    // Now do KLT tracking to get the valid new points
    std::vector<uchar> mask_klt;
    std::vector<float> error;
    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);

    // Normalize these points, so we can then do ransac
    // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
    std::vector<cv::Point2f> pts0_n, pts1_n;
    for (size_t i = 0; i < pts0.size(); i++) {
        pts0_n.push_back(camera_calib->undistort_cv(pts0.at(i)));
        pts1_n.push_back(camera_calib->undistort_cv(pts1.at(i)));
    }

    // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
    std::vector<uchar> mask_rsc;
    double max_focallength_img0 = std::max(camera_calib->get_K_opencv()(0, 0), camera_calib->get_K_opencv()(1, 1));
    double max_focallength_img1 = std::max(camera_calib->get_K_opencv()(0, 0), camera_calib->get_K_opencv()(1, 1));
    double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
    cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc);

    // Loop through and record only ones that are valid
    for (size_t i = 0; i < mask_klt.size(); i++) {
        auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
        mask_out.push_back(mask);
    }

    // Copy back the updated positions
    for (size_t i = 0; i < pts0.size(); i++) {
        kpts0.at(i).pt = pts0.at(i);
        kpts1.at(i).pt = pts1.at(i);
    }
}

void TrackKLT::display_history(cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::vector<size_t> highlighted,
                               std::string overlay) {

    // Cache the images to prevent other threads from editing while we viz (which can be slow)
    std::vector<cv::KeyPoint> pts_last_cache;
    std::vector<size_t> ids_last_cache;
    {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        pts_last_cache = pts_last;
        ids_last_cache = ids_last;
    }

    // Get the largest width and height
    int max_width = img_out.cols;
    int max_height = img_out.rows;

    // If the image is "small" thus we shoudl use smaller display codes
    bool is_small = (std::min(max_width, max_height) < 640);

    // Max tracks to show (otherwise it clutters up the screen)
    // size_t maxtracks = 50;

    // cv::cvtColor(img_out, img_out, cv::COLOR_GRAY2RGB);

    // draw, loop through all keypoints
    for (size_t i = 0; i < ids_last_cache.size(); i++) {
        // If a highlighted point, then put a nice box around it
        if (std::find(highlighted.begin(), highlighted.end(), ids_last_cache.at(i)) != highlighted.end()) {
            cv::Point2f pt_c = pts_last_cache.at(i).pt;
            cv::Point2f pt_l_top = cv::Point2f(pt_c.x - ((is_small) ? 5 : 10), pt_c.y - ((is_small) ? 5 : 10));
            cv::Point2f pt_l_bot = cv::Point2f(pt_c.x + ((is_small) ? 5 : 10), pt_c.y + ((is_small) ? 5 : 10));
            cv::rectangle(img_out, pt_l_top, pt_l_bot, cv::Scalar(0, 255, 0), (is_small) ? 2 : 3);
            cv::circle(img_out, pt_c, (is_small) ? 2 : 3, cv::Scalar(0, 255, 0), cv::FILLED);
        }
        // Get the feature from the database
        Feature feat;
        if (!database->get_feature_clone(ids_last_cache.at(i), feat))
            continue;
        if (feat.uvs.empty() || feat.uvs.empty() || feat.to_delete)
            continue;
        // Draw the history of this point (start at the last inserted one)
        // for (size_t z = feat.uvs.size() - 1; z > 0; z--) {
        //     // Check if we have reached the max
        //     if (feat.uvs.size() - z > maxtracks)
        //         break;
        //     // Calculate what color we are drawing in
        //     bool is_stereo = (feat.uvs.size() > 1);
        //     int color_r = (is_stereo ? b2 : r2) - (int)(1.0 * (is_stereo ? b1 : r1) / feat.uvs.size() * z);
        //     int color_g = (is_stereo ? r2 : g2) - (int)(1.0 * (is_stereo ? r1 : g1) / feat.uvs.size() * z);
        //     int color_b = (is_stereo ? g2 : b2) - (int)(1.0 * (is_stereo ? g1 : b1) / feat.uvs.size() * z);
        //     // Draw current point
        //     cv::Point2f pt_c(feat.uvs.at(z)(0), feat.uvs.at(z)(1));
        //     cv::circle(img_out, pt_c, (is_small) ? 1 : 3, cv::Scalar(color_r, color_g, color_b), cv::FILLED);
        //     // If there is a next point, then display the line from this point to the next
        //     if (z + 1 < feat.uvs.size()) {
        //         cv::Point2f pt_n(feat.uvs.at(z + 1)(0), feat.uvs.at(z + 1)(1));
        //         cv::line(img_out, pt_c, pt_n, cv::Scalar(color_r, color_g, color_b));
        //     }
        //     // If the first point, display the ID
        //     if (z == feat.uvs.size() - 1) {
        //         // cv::putText(img_out0, std::to_string(feat->featid), pt_c, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1,
        //         // cv::LINE_AA); cv::circle(img_out0, pt_c, 2, cv::Scalar(color,color,255), CV_FILLED);
        //     }
        // }
    }
    // Draw what camera this is
    auto txtpt = (is_small) ? cv::Point(10, 30) : cv::Point(30, 60);
    cv::putText(img_out, overlay, txtpt, cv::FONT_HERSHEY_TRIPLEX, (is_small) ? 0.75 : 1.5, cv::Scalar(0, 255, 0), 3);
}

} // namespace night_voyager