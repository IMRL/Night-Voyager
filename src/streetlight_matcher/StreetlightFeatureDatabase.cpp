#include "streetlight_matcher/StreetlightFeatureDatabase.h"
#include "streetlight_matcher/StreetlightFeature.h"

namespace night_voyager {

StreetlightFeatureDatabase::StreetlightFeatureDatabase(const StreetlightFeatureDatabase &other) {
    for (const auto &pair : other.features_idlookup) {
        features_idlookup[pair.first] = std::make_shared<StreetlightFeature>(*pair.second);
    }
}

void StreetlightFeatureDatabase::update_feature(size_t id, double timestamp, const cv::Rect &rect, const Eigen::Vector2f &center, const Eigen::Vector2f &center_n,
                                                const Eigen::Vector2f &noise) {
    std::lock_guard<std::mutex> lck(mtx);
    if (features_idlookup.find(id) != features_idlookup.end()) {

        std::shared_ptr<StreetlightFeature> feat = features_idlookup.at(id);
        assert(feat->featid == id);
        feat->boxes.push_back(rect);
        feat->uvs.push_back(center);
        feat->uvs_norm.push_back(center_n);
        feat->timestamps.push_back(timestamp);
        return;
    }

    std::shared_ptr<StreetlightFeature> feat = std::make_shared<StreetlightFeature>();
    feat->featid = id;
    feat->boxes.push_back(rect);
    feat->uvs.push_back(center);
    feat->uvs_norm.push_back(center_n);
    feat->timestamps.push_back(timestamp);

    features_idlookup[id] = feat;
}

void StreetlightFeatureDatabase::update_feature(std::shared_ptr<StreetlightFeature> feature) {

    assert(feature->boxes.size() == feature->timestamps.size());
    assert(feature->boxes.size() == feature->uvs.size());
    assert(feature->boxes.size() == feature->uvs_norm.size());

    std::lock_guard<std::mutex> lck(mtx);
    if (features_idlookup.find(feature->featid) != features_idlookup.end()) {

        std::shared_ptr<StreetlightFeature> feat = features_idlookup.at(feature->featid);
        assert(feat->featid == feature->featid);
        feat->boxes.insert(feat->boxes.end(), feature->boxes.begin(), feature->boxes.end());
        feat->uvs.insert(feat->uvs.end(), feature->uvs.begin(), feature->uvs.end());
        feat->uvs_norm.insert(feat->uvs_norm.end(), feature->uvs_norm.begin(), feature->uvs_norm.end());
        feat->timestamps.insert(feat->timestamps.end(), feature->timestamps.begin(), feature->timestamps.end());
        return;
    }

    std::shared_ptr<StreetlightFeature> feat = std::make_shared<StreetlightFeature>();
    feat->featid = feature->featid;
    feat->boxes = feature->boxes;
    feat->uvs = feature->uvs;
    feat->uvs_norm = feature->uvs_norm;
    feat->timestamps = feature->timestamps;

    features_idlookup[feat->featid] = feat;
}

void StreetlightFeatureDatabase::clean_old_measurements(const std::vector<double> &valid_times) {

    std::lock_guard<std::mutex> lck(mtx);
    auto it0 = features_idlookup.begin();
    while (it0 != features_idlookup.end()) {

        it0->second->clean_old_measurements(valid_times);

        // Count how many measurements
        int ct_meas = it0->second->timestamps.size();

        // Remove if we don't have enough
        if (ct_meas < 1) {
            it0->second->to_delete = true;
            it0 = features_idlookup.erase(it0);
        } else {
            it0++;
        }
    }
}

size_t StreetlightFeatureDatabase::count_measurements() {

    std::lock_guard<std::mutex> lck(mtx);
    size_t max_mea_size = 0;
    auto iter_feature = features_idlookup.begin();
    while (iter_feature != features_idlookup.end()) {
        max_mea_size += 2 * iter_feature->second->timestamps.size();
        iter_feature++;
    }
    return max_mea_size;
}

} // namespace night_voyager