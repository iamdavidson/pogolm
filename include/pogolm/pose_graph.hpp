
#pragma once

#include <map>
#include <string>
#include <vector>
#include <mutex>
#include <limits>
#include <thread>
#include <optional>

#include "ikd-Tree/ikd_Tree.h"

#define LOG_NODE_TAG NODE_POSE_GRAPH

#include "pogolm/utils.hpp"
#include "pogolm/logging.hpp"
#include <rclcpp/service.hpp>
#include "factor_graph_interfaces/conversion_gtsam.hpp"
#include "factor_graph_interfaces/msg/factor_graph.hpp"
#include "factor_graph_interfaces/srv/get_point_cloud.hpp"

#include "pogolm_interfaces/msg/factor_graph_debug.hpp"
#include "pogolm_interfaces/msg/loop_debug.hpp"
#include "pogolm_interfaces/msg/loop_edge.hpp"
#include "pogolm_interfaces/msg/merged_info.hpp"
#include "pogolm_interfaces/msg/key.hpp"
#include "pogolm_interfaces/msg/loop_edge_tracker.hpp"
#include "pogolm_interfaces/msg/landmark.hpp"
#include "pogolm_interfaces/srv/store_landmark.hpp"
#include "pogolm_interfaces/srv/query_landmark.hpp"

#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/empty.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BearingRangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

namespace pose_graph_np
{

using Landmark = struct Landmark;
using SearchResult = struct SearchResult;
using MergedInfo = struct MergedInfo;
using LoopEdgeCandidate = struct LoopEdgeCandidate;

struct PoseGraphParams
{
  // input topics
  std::string lidar_in{ "velodyne_points" };
  std::string odom_in{ "odom" };

  // frame id used for published messages
  std::string operating_frame_id{ "odom" };
  double pc_range_thres{ 150. };

  // enable dynamic covariance from twist covariance
  bool dynamic_pose_cov{ false };

  // keyframe selection thresholds
  double lin_cirt{ 1.5 };
  double ang_crit{ 0.3 };
  double to_time_crit{ 2.0 };
  double to_dist_crit{ 0.3 };

  // landmark merge radius
  // specify for automatic landmark merging after PGO
  double lm_range_rad{ 0.4 };

  // noise models (sigmas)
  std::vector<double> bearing_sig{ { 0.1, 0.1, 0.2 } };
  std::vector<double> odometry_sig{ { 0.1, 0.1, 0.1, 0.3, 0.3, 0.05 } };
  std::vector<double> loop_sig{ { 0.2, 0.2, 0.2, 0.3, 0.3, 0.01 } };
  std::vector<double> prior_sig{ { 0.01, 0.01, 0.01, 0.1, 0.1, 0.1 } };

  // output/log paths
  std::string tum_out_path{ "/tmp/pogolm" };
  std::string log_path{ "/tmp/pogolm/pose_graph_logs.txt" };
  bool save_trajectory{ true };

  bool verbose{ false };
};

class PoseGraph : public rclcpp::Node
{
private:
  // callback groups
  rclcpp::CallbackGroup::SharedPtr io_group;
  rclcpp::CallbackGroup::SharedPtr pg_group;

  // subscriptions and publishers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscriber;
  rclcpp::Subscription<pogolm_interfaces::msg::LoopEdge>::SharedPtr le_subscriber;
  rclcpp::Publisher<factor_graph_interfaces::msg::FactorGraph>::SharedPtr fg_publisher;
  rclcpp::Publisher<pogolm_interfaces::msg::Key>::SharedPtr pc_key_publisher;
  rclcpp::TimerBase::SharedPtr pose_add_timer;
  rclcpp::TimerBase::SharedPtr odom_timeout_timer;

  // ROS-DEBUG_Topics
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr trigger_subscriber;
  rclcpp::Publisher<pogolm_interfaces::msg::FactorGraphDebug>::SharedPtr fg_debug_publisher;
  rclcpp::Publisher<pogolm_interfaces::msg::MergedInfo>::SharedPtr merge_info_pub;
  rclcpp::Publisher<pogolm_interfaces::msg::LoopEdgeTracker>::SharedPtr loop_tracker_pub;

  // Services
  rclcpp::Service<pogolm_interfaces::srv::StoreLandmark>::SharedPtr store_landmark_srv;
  rclcpp::Service<pogolm_interfaces::srv::QueryLandmark>::SharedPtr query_landmark_srv;

  // mutex / thread
  std::condition_variable optim_cv;
  std::atomic_bool running{ true }, optim_requested{ false }, optim_in_process{ false }, loop_edge_buffered{ false },
      odom_available{ false }, lidar_available{ false };
  std::thread optim_thread;
  std::mutex pose_graph_mtx, loop_cand_mtx, optim_mtx, lidar_mtx, odom_mtx, pc_file_mtx, od_file_mtx;

  // loop candidates and factors
  std::vector<std::shared_ptr<LoopEdgeCandidate>> loop_candidates;
  std::vector<std::shared_ptr<LoopEdgeCandidate>> loop_edge_tracker;
  std::map<uint64_t, uint64_t> factor_lookup;

  std::shared_ptr<disk::DiskCloudStore> pc_storage;

  // Landmarks
  std::map<uint64_t, std::shared_ptr<Landmark>> landmark_data;
  KD_TREE<LMPoint> lm_ikd_tree;
  double range_radius{ 0.4 };

  // GTSAM
  rclcpp::Time last_odom_receive_time, last_pose_added_time;
  sensor_msgs::msg::PointCloud2::UniquePtr last_pc_msg;
  nav_msgs::msg::Odometry::UniquePtr last_odom_msg;
  gtsam::Pose3 prev_odom_raw, estimated_bias;
  std::array<double, 36> prev_odom_cov, prev_twist_cov;

  double prev_stamp_s = std::numeric_limits<double>::quiet_NaN();

  gtsam::NonlinearFactorGraph pose_graph;
  Eigen::Matrix<double, 6, 6> loop_cov;
  gtsam::noiseModel::Diagonal::shared_ptr odometry_noise, loop_closure_noise, prior_noise, bearing_noise;
  gtsam::Values initial;

  uint64_t current_pose_key{ STARTING_POSE_KEY };
  uint64_t current_landmark_key{ 0 };

  const size_t IKD_REBUILD_THRES = 20;

  std::string operating_frame_id{ "odom" };
  double pc_range_thres{ 150. };

  rclcpp::Time last_optimization;

  double linear_crit_min_dist;
  double angular_crit_min_yaw;
  double timeout_crit_min_time;
  double timeout_crit_min_dist;
  double max_lidar_odom_dt{ 0.10 };

  bool dynamic_pose_cov;

  // trajectory export buffers
  bool save_trajectory{ false };
  std::vector<double> pose_timestamps_;
  std::vector<gtsam::Pose3> raw_odom_poses_;
  std::string tum_out_path_{ "/home/student/d/dsamuelyan/eval/coll_1/quad-easy/pogolm" };
  std::string log_path{ "/home/student/d/dsamuelyan/eval/coll_1/quad-easy/pogolm/pose_graph_logs.txt" };

  void save_trajectory_tum(const std::string& path);
  void save_raw_trajectory_tum(const std::string& path);

  bool verbose{ true };

  std::vector<size_t> lc_num;
  std::vector<double> optim_durs;
  int num_far = 0, num_near = 0;

  int test_count{ 0 };
  size_t dummy_counter{ 0 };

  // odometry callback, stores last odom message
  void callback_odometry(nav_msgs::msg::Odometry::UniquePtr msg);
  // point cloud callback, stores last point cloud
  void callback_pointcloud(sensor_msgs::msg::PointCloud2::UniquePtr msg);
  // timer callback, disables odom if no messages arrive
  void callback_odom_timeout_timer();
  // timer callback, adds keyframes and publishes pose graph
  void callback_pose_add_timer();
  // loop edge callback, buffers candidate edges and triggers optimization
  void callback_loop_edge(const pogolm_interfaces::msg::LoopEdge& msg);
  // debug trigger, injects test landmarks
  void callback_trigger_add_landmark(const std_msgs::msg::Empty& msg);

  // handle the store landmark request
  void handle_store_landmark(const std::shared_ptr<pogolm_interfaces::srv::StoreLandmark::Request> req,
                             std::shared_ptr<pogolm_interfaces::srv::StoreLandmark::Response> res);

  // handle the query landmark request
  void handle_query_landmark(const std::shared_ptr<pogolm_interfaces::srv::QueryLandmark::Request> req,
                             std::shared_ptr<pogolm_interfaces::srv::QueryLandmark::Response> res);

  // helpter function
  pogolm_interfaces::msg::Landmark to_ros_landmark(const Landmark& lm);

  // add a pose node and odometry factor (is_anchor forces insertion)
  std::optional<uint64_t> add_pose(const bool is_anchor = false);
  // optimizer worker thread
  void optimization_worker();

  // optional point cloud filter
  sensor_msgs::msg::PointCloud2::UniquePtr pc_filter(sensor_msgs::msg::PointCloud2::UniquePtr in_pc2);

  // check if a gtsam key exists in Values
  bool node_exists(uint64_t key);
  // check if a gtsam key exists in Values (no lock)
  bool node_exists_unsafe(uint64_t key);
  // remove factor from graph if index is valid
  void fast_remove_factor(uint64_t factor_key);

  // access landmark point by key
  LMPoint get_lm_point(uint64_t key);
  // remove landmark without locking
  bool remove_landmark_unsafe(uint64_t query_landmark_key);
  // check landmark existence without locking
  bool landmark_exists_unsafe(uint64_t key);

  // range query in IKD-tree
  void range_search(const float& radius, const Landmark& lm, SearchResult& result);
  // knn query in IKD-tree
  void knn_search(const int k, const Landmark& lm, SearchResult& result);
  // merge neighboring landmarks (SearchResult)
  void merge_neighbors(Landmark& lm, const SearchResult& result);
  // merge neighboring landmarks (association list)
  void merge_neighbors(Landmark& lm, const std::vector<uint64_t>& associations);

  // update landmark point and trigger background merge (no lock)
  void update_point_unsafe(const uint64_t key, const gtsam::Point3& new_value);
  // rebuild IKD-tree from landmark_data
  void rebuild_ikdtree();

  // keyframe selection criteria
  bool linear_crit(const gtsam::Pose3& relative);
  bool angular_crit(const gtsam::Pose3& relative);
  bool timeout_crit(const gtsam::Pose3& relative);

  // print neighbor keys for a query landmark
  void print_found_neighbors(const uint64_t& query_key, const SearchResult& result);
  // compute and log candidate loop error
  void print_cand_error(const std::shared_ptr<LoopEdgeCandidate>& cand, const gtsam::Values& values, bool verbose);

  // publish merge meta info
  void publish_merge_info(const MergedInfo& info);
  // publish key that has a stored point cloud
  void publish_pc_w_key(const uint64_t& key);
  // publish debug data (landmarks/anchors/factors)
  void publish_debug_info();
  // publish pose graph message
  void publish_pose_graph();
  // publish accepted loop edges
  void publish_loop_edge_tracker(std::vector<std::shared_ptr<LoopEdgeCandidate>>& list);
  // publish loop debug refresh
  void publish_pose_refresh_loop_debug();

  // read params from ROS params
  PoseGraphParams get_yaml_params();
  // init pubs/subs/timers/threads
  void init_node(const PoseGraphParams& params);

public:
  PoseGraph(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

  PoseGraph(const PoseGraphParams& params, const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

  // stops optimizer thread and optionally writes trajectory/logs
  ~PoseGraph();

  // add a landmark in local coordinates (creates an anchor pose internally)
  std::vector<uint64_t> add_landmark(const gtsam::Point3& p, const std::string label = "LM",
                                     const std::vector<uint64_t>& association_list = {});

  // get all landmarks
  std::vector<std::shared_ptr<Landmark>> get_all_lm();
  // get landmark by key
  std::vector<std::shared_ptr<Landmark>> get_lm_by_key(uint64_t query_key);
  // get landmarks by label
  std::vector<std::shared_ptr<Landmark>> get_lm_by_label(std::string query_label);

  // neighbor queries (range)
  std::vector<std::shared_ptr<Landmark>> get_neighbors_range_by_lm(const float radius, const Landmark& query_lm);
  std::vector<std::shared_ptr<Landmark>> get_neighbors_range_by_key(const float radius, const uint64_t query_key);
  std::vector<std::shared_ptr<Landmark>> get_neighbors_range_by_label(const float radius,
                                                                      const std::string query_label);

  // neighbor queries (knn)
  std::vector<std::shared_ptr<Landmark>> get_neighbors_knn_by_lm(const int k, const Landmark& query_lm);
  std::vector<std::shared_ptr<Landmark>> get_neighbors_knn_by_key(const int k, const uint64_t query_key);
  std::vector<std::shared_ptr<Landmark>> get_neighbors_knn_by_label(const int k, const std::string query_label);

  // update a landmark value
  void update_point(const uint64_t key, const gtsam::Point3& new_value);
  // remove a landmark and its factors
  bool remove_landmark(const uint64_t query_landmark_key);
  // check if a landmark exists
  bool landmark_exists(const uint64_t key);

  // current global pose estimate (bias * odom)
  gtsam::Pose3 get_current_pose();

  // default params for this node
  PoseGraphParams get_default_params();
};

// component wrapper for rclcpp components
class PoseGraphComponent : public PoseGraph
{
public:
  PoseGraphComponent(const rclcpp::NodeOptions& options) : PoseGraph(options)
  {
  }

  PoseGraphComponent(const PoseGraphParams& params, const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : PoseGraph(params, options)
  {
  }
};

}  // namespace pose_graph_np
