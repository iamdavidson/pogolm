#pragma once

#include "pogolm/pose_graph.hpp"
#include "pogolm/map_module.hpp"
#include "pogolm/loop_detector.hpp"

#include <mutex>
#include <memory>
#include <vector>
#include <thread>
#include <cstdint>
#include <rclcpp/rclcpp.hpp>
#include <gtsam/geometry/Point3.h>

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>


namespace pogolm_np {

struct false_type { static constexpr bool value = false; };
struct true_type { static constexpr bool value = true; };

template <class T> struct key_query : false_type {};
template <class T> struct lm_query : false_type {};
template <class T> struct label_query : false_type {};
template <class T> struct int_prop : false_type {};
template <class T> struct float_prop : false_type {};
template <> struct key_query<uint64_t> : true_type {};
template <> struct label_query<std::string> : true_type {};
template <> struct label_query<const char*> : true_type {};
template <> struct label_query<char*> : true_type {};
template <> struct lm_query<Landmark> : true_type {};
template <> struct int_prop<int> : true_type {};
template <> struct float_prop<float> : true_type {};

enum NeighborSearch { KNN, RANGE };
enum QueryType { KEY, LABEL, LANDMARK };

struct APIParams {
    pose_graph_np::PoseGraphParams pg_params;
    loop_detector_np::LoopDetectorParams ld_params;
};

struct Point3d {
    
    double x,y,z;

    Point3d() : x(0.0), y(0.0), z(0.0) {}
    Point3d(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

    gtsam::Point3 to_gtsam_p() const {
        return gtsam::Point3(x,y,z);
    }
};

struct POGOLMNodes {

    std::shared_ptr<pose_graph_np::PoseGraphComponent> pose_graph_;
    std::shared_ptr<loop_detector_np::LoopDetectorComponent> loop_detector_;
    std::shared_ptr<map_module_np::MapModuleComponent> map_module_;
};

class POGOLM {

private:

    // Nodes
    std::shared_ptr<pose_graph_np::PoseGraphComponent> pose_graph_;
    std::shared_ptr<loop_detector_np::LoopDetectorComponent> loop_detector_;
    std::shared_ptr<map_module_np::MapModuleComponent> map_module_;

    // Executor
    std::shared_ptr<rclcpp::Executor> executor_;
    std::mutex pogolm_mut_;


public:

    APIParams params;

    // Non-standalone only: rclcpp must already be initialized by the application.
    POGOLM(const APIParams& params_) : params(params_)  {

        if (!rclcpp::ok()) {
            throw std::runtime_error("[POGOLM] rclcpp not initialized!");
        }

        init_nodes(params);

        if (!pose_graph_ || !loop_detector_ || !map_module_) {
            throw std::runtime_error("[POGOLM] init_nodes produced null node(s)");
        }
    }

    POGOLM()  {

        params = get_defautl_params();

        if (!rclcpp::ok()) {
            throw std::runtime_error("[POGOLM] rclcpp not initialized!");
        }

        init_nodes(params);

        if (!pose_graph_ || !loop_detector_ || !map_module_) {
            throw std::runtime_error("[POGOLM] init_nodes produced null node(s)");
        }
    }

    ~POGOLM() = default;

    inline void init_nodes(const APIParams& params_) {

        rclcpp::NodeOptions opt;
        opt.use_intra_process_comms(true);

        pose_graph_ = std::make_shared<pose_graph_np::PoseGraphComponent>(params_.pg_params, opt);
        loop_detector_ = std::make_shared<loop_detector_np::LoopDetectorComponent>(params_.ld_params, opt);
        map_module_ = std::make_shared<map_module_np::MapModuleComponent>(opt);
    }

    inline APIParams get_defautl_params() {

        if (!pose_graph_ || !loop_detector_) {
            return APIParams{};
        }

        pose_graph_np::PoseGraphParams pg_params = pose_graph_->get_default_params();
        loop_detector_np::LoopDetectorParams ld_params = loop_detector_->get_default_params();

        return APIParams {pg_params, ld_params};
    }

    inline void attach_to_exec(const std::shared_ptr<rclcpp::Executor>& executor_) {
        
        std::lock_guard<std::mutex> lock(this->pogolm_mut_);

        if (!executor_) {
            throw std::runtime_error("[POGOLM] attach_to_exec got null executor");
        }

        if (!pose_graph_ || !loop_detector_ || !map_module_) {
            throw std::runtime_error("[POGOLM] attach_to_exec: nodes are null (init_nodes failed?)");
        }

        executor_->add_node(pose_graph_);
        executor_->add_node(loop_detector_);
        executor_->add_node(map_module_);
    }

    inline POGOLMNodes get_nodes() {
        
        POGOLMNodes nodes;
        nodes.loop_detector_ = loop_detector_;
        nodes.pose_graph_ = pose_graph_;
        nodes.map_module_ = map_module_;

        return nodes;
    }

    gtsam::Pose3 get_current_pose() const {
        return pose_graph_->get_current_pose();  
    }

    inline std::vector<uint64_t> store_landmark(
        const Point3d &p,
        const std::string label = "LM",
        const std::vector<uint64_t> &association_list = {}
    ) {
        if (!pose_graph_) return {};
        return pose_graph_->add_landmark(p.to_gtsam_p(), label, association_list);
    }

    inline std::vector<std::shared_ptr<Landmark>> get_all_lm() {
        if (!pose_graph_) return {};
        return pose_graph_->get_all_lm();
    }

    template<QueryType type, typename K>
    std::vector<std::shared_ptr<Landmark>>
    get_lm(const K& query_elm) {

        std::vector<std::shared_ptr<Landmark>> res;

        if (!pose_graph_) return res;

        if constexpr (type == KEY) {
            static_assert(key_query<K>::value, "K is not a key (unsigned long int)");
            res = pose_graph_->get_lm_by_key(query_elm); 
        } 
        else if constexpr (type == LABEL) { 
            static_assert(label_query<K>::value, "K is not a string label");
            res = pose_graph_->get_lm_by_label(query_elm); 
        }

        return res;
    }

    // usage get_neighbors<KNN,KEY> (15, 0)
    //       get_neighbors<KNN,LABEL> (15, "ball")
    //       get_neighbors<KNN,KEY> (15, lm)
    //
    //       get_neighbors<RANGE,KEY> (4.2f, 0)
    //       get_neighbors<RANGE,LABEL> (4.2f, "ball")
    //       get_neighbors<RANGE,KEY> (4.2f, lm)
    template <NeighborSearch search, QueryType type, typename V, typename K>
    std::vector<std::shared_ptr<Landmark>>
    get_neighbors (const K& query_elm, const V& search_property) {

        std::vector<std::shared_ptr<Landmark>> res;

        if (!pose_graph_) return res;

        if constexpr (search == KNN) {

            static_assert(int_prop<V>::value, "V is not an int property");

            if constexpr (type == KEY) {
                static_assert(key_query<K>::value, "K is not a key (unsigned long int)");
                res = pose_graph_->get_neighbors_knn_by_key(search_property, query_elm);
            }
            if constexpr (type == LABEL) {
                static_assert(label_query<K>::value, "K is not a string label");
                res = pose_graph_->get_neighbors_knn_by_label(search_property, query_elm);  
            }
            if constexpr (type == LANDMARK) {
                static_assert(lm_query<K>::value, "K is not a Landmark");
                res = pose_graph_->get_neighbors_knn_by_lm(search_property, query_elm);  
            }
        } 

        else if constexpr (search == RANGE) {

            static_assert(float_prop<V>::value, "V is not a float");

            if constexpr (type == KEY) {
                static_assert(key_query<K>::value, "K is not a key (unsigned long int)");
                res = pose_graph_->get_neighbors_range_by_key(search_property, query_elm);
            }
            if constexpr (type == LABEL) {
                static_assert(label_query<K>::value, "K is not a string label");
                res = pose_graph_->get_neighbors_range_by_label(search_property, query_elm);  
            }
            if constexpr (type == LANDMARK) {
                static_assert(lm_query<K>::value, "K is not a Landmark");
                res = pose_graph_->get_neighbors_range_by_lm(search_property, query_elm);  
            }
        } 

        return res;
    }

    inline std::vector<std::shared_ptr<Landmark>>
    has_neighbors(const Point3d& query_elm, const float range, const float eps) {

        if (!pose_graph_) return {};

        const auto T_wb = pose_graph_->get_current_pose();
        const auto q_g  = T_wb.transformFrom(query_elm.to_gtsam_p());

        Landmark lm;
        lm.key = std::numeric_limits<uint64_t>::max();
        lm.point = LMPoint{
            static_cast<float>(q_g.x()),
            static_cast<float>(q_g.y()),
            static_cast<float>(q_g.z()),
            lm.key
        };

        return pose_graph_->get_neighbors_range_by_lm(range + eps, lm);
    }
};

} // namespace pogolm_np