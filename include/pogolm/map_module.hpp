#pragma once

#include <functional>
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <cstdint>
#include <array>
#include <optional>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#define LOG_NODE_TAG NODE_MAP_MODULE

#include "pogolm/utils.hpp"
#include "pogolm/logging.hpp"

#include "factor_graph_interfaces/msg/factor_graph.hpp"

#include "pogolm_interfaces/srv/get_map.hpp"
#include "pogolm_interfaces/msg/key.hpp"


namespace map_module_np {

// current map snapshot (poses + point clouds + landmarks)
struct Map {
    std::map<uint64_t, sensor_msgs::msg::PointCloud2> clouds;
    std::map<uint64_t, geometry_msgs::msg::Pose>  poses;
    std::map<uint64_t, geometry_msgs::msg::Point> landmarks;
};

// map aggregator node, serves latest map over a service
class MapModule : public rclcpp::Node {

private:

    // subscriptions
    rclcpp::Subscription<factor_graph_interfaces::msg::FactorGraph>::SharedPtr
        pose_graph_subscriber;

    // service servers
    rclcpp::Service<pogolm_interfaces::srv::GetMap>::SharedPtr
        get_map_srv;

    // disk stores for point clouds
    std::shared_ptr<disk::DiskCloudStore> pc_storage;
    std::shared_ptr<Map> current_map;

    // mutex
    std::mutex map_mtx;

    // update map cache from incoming factor graph
    void callback_pose_graph(factor_graph_interfaces::msg::FactorGraph::ConstSharedPtr msg);
    
    // service handler for map requests
    void handle_map_request(
        const std::shared_ptr<pogolm_interfaces::srv::GetMap::Request> request,
        std::shared_ptr<pogolm_interfaces::srv::GetMap::Response> response
    );

    // load point cloud from disk for a given key
    std::optional<sensor_msgs::msg::PointCloud2> fetch_pointcloud(uint64_t key);

public:

    MapModule(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~MapModule() = default;

    // get current cached map snapshot
    std::shared_ptr<Map> get_map();

    // get a single point cloud (from cache/storage)
    std::optional<sensor_msgs::msg::PointCloud2> get_pointcloud(uint64_t key);
};

// component wrapper for rclcpp components
class MapModuleComponent : public MapModule {

public:
    MapModuleComponent(const rclcpp::NodeOptions& options)
    : MapModule(options) {}    
};

} // namespace map_module_np
