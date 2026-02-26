#include "pogolm/map_module.hpp"

using namespace std::chrono_literals;

namespace map_module_np
{

void MapModule::callback_pose_graph(factor_graph_interfaces::msg::FactorGraph::ConstSharedPtr msg)
{
  auto new_map = std::make_shared<Map>();

  for (const auto& pose : msg->poses)
  {
    const uint64_t k = key_(pose.key);
    new_map->poses.insert_or_assign(k, pose.pose);
  }

  for (const auto& lm : msg->points)
  {
    const uint64_t k = key_(lm.key);
    new_map->landmarks.insert_or_assign(k, lm.point);
  }

  {
    std::lock_guard<std::mutex> lock(map_mtx);
    current_map = std::move(new_map);
  }
}

void MapModule::handle_map_request(const std::shared_ptr<pogolm_interfaces::srv::GetMap::Request> request,
                                   std::shared_ptr<pogolm_interfaces::srv::GetMap::Response> response)
{
  auto generic_map = get_map();

  const size_t n_poses = generic_map->poses.size();
  const size_t n_landmarks = generic_map->landmarks.size();

  LOGI_LIDAR(this->get_logger(), "map request: %lu poses, %lu landmarks — fetching pointclouds from disk", n_poses,
             n_landmarks);

  response->cloud_with_poses.reserve(n_poses);

  size_t fetched = 0;
  size_t not_found = 0;

  for (const auto& [key, pose] : generic_map->poses)
  {
    pogolm_interfaces::msg::CloudWithPose entry;
    entry.key = key;
    entry.pose = pose;

    sensor_msgs::msg::PointCloud2 pc2;
    bool ok = pc_storage->get(key, pc2, false);

    if (ok)
    {
      entry.cloud = std::move(pc2);
      fetched++;
    }
    else
    {
      not_found++;
    }

    response->cloud_with_poses.emplace_back(std::move(entry));
  }

  response->landmarks.reserve(n_landmarks);

  for (const auto& [id, point] : generic_map->landmarks)
  {
    pogolm_interfaces::msg::LandmarkEntry entry;
    entry.id = id;
    entry.x = point.x;
    entry.y = point.y;
    entry.z = point.z;

    response->landmarks.emplace_back(std::move(entry));
  }

  LOGI_LIDAR(this->get_logger(), "map request done: %lu clouds fetched, %lu not found on disk, %lu landmarks", fetched,
             not_found, n_landmarks);
}

std::optional<sensor_msgs::msg::PointCloud2> MapModule::fetch_pointcloud(uint64_t key)
{
  sensor_msgs::msg::PointCloud2 pc2;
  bool ok = pc_storage->get(key, pc2, false);

  if (!ok)
  {
    LOGI_LIDAR(this->get_logger(), "pointcloud <%lu> not found on disk", key);
    return std::nullopt;
  }

  return pc2;
}

std::shared_ptr<Map> MapModule::get_map()
{
  auto generic_map = std::make_shared<Map>();
  {
    std::lock_guard<std::mutex> lock(map_mtx);

    if (current_map)
    {
      generic_map->poses = current_map->poses;
      generic_map->landmarks = current_map->landmarks;
    }
  }

  for (const auto& [k, v] : generic_map->poses)
  {
    auto ret = get_pointcloud(k);
    if (!ret)
      continue;

    generic_map->clouds[k] = *ret;
  }

  return generic_map;
}

std::optional<sensor_msgs::msg::PointCloud2> MapModule::get_pointcloud(uint64_t key)
{
  return fetch_pointcloud(key);
}

MapModule::MapModule(const rclcpp::NodeOptions& options) : rclcpp::Node("map_module", options)
{
  pc_storage = StoreRegistry::get("/tmp/pc_store.bin", 1, false, false);

  RCLCPP_INFO(get_logger(), "pc_store ptr=%p", (void*)pc_storage.get());
  LOGI_POGOLM(this->get_logger(), "MapModule starting up");

  current_map = std::make_shared<Map>();

  pose_graph_subscriber = this->create_subscription<factor_graph_interfaces::msg::FactorGraph>(
      "pose_graph", 10, std::bind(&MapModule::callback_pose_graph, this, std::placeholders::_1));

  get_map_srv = this->create_service<pogolm_interfaces::srv::GetMap>(
      "get_map", std::bind(&MapModule::handle_map_request, this, std::placeholders::_1, std::placeholders::_2));
}

}  // namespace map_module_np

#ifndef PGO_WITH_VISUAL_NO_MAIN
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<map_module_np::MapModule>();

  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
#endif