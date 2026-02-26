#include "pogolm/pose_graph.hpp"

namespace pose_graph_np
{

void PoseGraph::callback_odometry(nav_msgs::msg::Odometry::UniquePtr msg)
{
  {
    std::lock_guard<std::mutex> lock(odom_mtx);
    last_odom_receive_time = this->now();
    last_odom_msg = std::move(msg);
    last_odom_msg->header.frame_id = operating_frame_id;
  }
  odom_available.store(true, std::memory_order_release);
}

void PoseGraph::callback_pointcloud(sensor_msgs::msg::PointCloud2::UniquePtr msg)
{
  // auto msg_filtered = pc_filter(std::move(msg));

  {
    std::lock_guard<std::mutex> lock(lidar_mtx);
    last_pc_msg = std::move(msg);
    last_pc_msg->header.frame_id = operating_frame_id;
  }
  lidar_available.store(true, std::memory_order_release);
}

void PoseGraph::callback_odom_timeout_timer()
{
  if (!odom_available.load(std::memory_order_acquire))
    return;

  rclcpp::Time last_time;
  {
    std::lock_guard<std::mutex> lock(odom_mtx);
    last_time = last_odom_receive_time;
  }

  rclcpp::Duration dur = this->now() - last_time;

  if (dur.seconds() > 5.0)
  {
    odom_available.store(false, std::memory_order_release);
    LOGW_TIMEOUT(this->get_logger(), "No odom for 5s");
  }
}

void PoseGraph::save_raw_trajectory_tum(const std::string& path)
{
  std::vector<gtsam::Pose3> raw_poses;
  std::vector<double> times;
  uint64_t N = 0;

  {
    std::lock_guard<std::mutex> lock(pose_graph_mtx);
    raw_poses = raw_odom_poses_;
    times = pose_timestamps_;
    N = current_pose_key;
  }

  std::ofstream os(path);
  if (!os.is_open())
  {
    throw std::runtime_error("Could not open RAW TUM output: " + path);
  }

  os << std::fixed << std::setprecision(9);

  for (uint64_t k = 0; k < N; ++k)
  {
    if (k >= raw_poses.size())
      continue;

    const gtsam::Pose3 p = raw_poses[k];
    const auto t = p.translation();
    const Eigen::Matrix3d R = p.rotation().matrix();

    Eigen::Quaterniond q(R);
    q.normalize();

    const double ts = (k < times.size() ? times[k] : static_cast<double>(k));

    os << ts << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " "
       << q.w() << "\n";
  }

  RCLCPP_INFO(this->get_logger(), "Saved RAW ODOM TUM trajectory: %s", path.c_str());
}

void PoseGraph::save_trajectory_tum(const std::string& path)
{
  gtsam::Values poses;
  std::vector<double> times;
  uint64_t N = 0;

  {
    std::lock_guard<std::mutex> lock(pose_graph_mtx);
    poses = initial;
    times = pose_timestamps_;
    N = current_pose_key;
  }

  std::ofstream os(path);
  if (!os.is_open())
  {
    throw std::runtime_error("Could not open TUM output: " + path);
  }

  os << std::fixed << std::setprecision(9);

  for (uint64_t k = 0; k < N; ++k)
  {
    if (!poses.exists(x_(k)))
      continue;

    const gtsam::Pose3 p = poses.at<gtsam::Pose3>(x_(k));
    const auto t = p.translation();
    const Eigen::Matrix3d R = p.rotation().matrix();

    Eigen::Quaterniond q(R);
    q.normalize();

    const double ts = (k < times.size() ? times[k] : static_cast<double>(k));

    os << ts << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " "
       << q.w() << "\n";
  }

  RCLCPP_INFO(this->get_logger(), "Saved TUM trajectory: %s", path.c_str());
}

void PoseGraph::callback_pose_add_timer()
{
  publish_pose_graph();

  if (!odom_available.load(std::memory_order_acquire) || !lidar_available.load(std::memory_order_acquire))
    return;

  std::optional<uint64_t> ret = add_pose();
  if (!ret)
    return;

  publish_debug_info();
}

// sensor_msgs::msg::PointCloud2::UniquePtr
// PoseGraph::pc_filter(sensor_msgs::msg::PointCloud2::UniquePtr in_pc2) {

//     const auto norm_sq = [](const PointXYZIRT& p) noexcept -> float {
//         return p.x * p.x + p.y * p.y + p.z * p.z;
//     };

//     const double thres_sq = pc_range_thres * pc_range_thres;

//     in_pcl->points.clear();
//     in_pcl->points.shrink_to_fit();

//     out_pcl->points.clear();
//     out_pcl->points.shrink_to_fit();

//     pcl::fromROSMsg(*in_pc2, *in_pcl);
//     out_pcl->points.reserve(in_pcl->points.size());

//     for (const auto &p : in_pcl->points) {
//         if (norm_sq(p) <= thres_sq) out_pcl->points.push_back(p);
//     }

//     out_pcl->width = static_cast<uint32_t>(out_pcl->points.size());
//     out_pcl->height = 1;
//     out_pcl->is_dense = in_pc2->is_dense;

//     auto out_pc2 = std::make_unique<sensor_msgs::msg::PointCloud2>();
//     pcl::toROSMsg(*out_pcl, *out_pc2);

//     out_pc2->header.stamp = this->now();
//     out_pc2->header.frame_id = operating_frame_id;

//     return out_pc2;
// }

gtsam::Pose3 PoseGraph::get_current_pose()
{
  std::lock_guard<std::mutex> lock(pose_graph_mtx);

  if (!last_odom_msg)
    return gtsam::Pose3::Identity();

  auto pose_odom = to_gtsam(last_odom_msg->pose.pose);
  return estimated_bias * pose_odom;
}

std::optional<uint64_t> PoseGraph::add_pose(const bool is_anchor)
{
  // get the current odometry pose value
  gtsam::Pose3 curr_odom_raw;
  std::array<double, 36> odom_cov;
  std::array<double, 36> twist_cov;
  double stamp_s;
  {
    std::lock_guard<std::mutex> lock(odom_mtx);
    curr_odom_raw = to_gtsam(last_odom_msg->pose.pose);
    odom_cov = last_odom_msg->pose.covariance;
    twist_cov = last_odom_msg->twist.covariance;
    stamp_s = rclcpp::Time(last_odom_msg->header.stamp).seconds();
  }

  double dt = 0.0;
  if (std::isfinite(prev_stamp_s))
    dt = stamp_s - prev_stamp_s;
  if (dt <= 0.0)
    dt = 0.01;
  prev_stamp_s = stamp_s;

  // check point cloud and odometry timing synchronization
  if (!is_anchor)
  {
    double pc_stamp_s = std::numeric_limits<double>::quiet_NaN();
    bool has_pc = false;

    {
      std::lock_guard<std::mutex> lock(lidar_mtx);
      has_pc = (last_pc_msg != nullptr);
      if (has_pc)
        pc_stamp_s = rclcpp::Time(last_pc_msg->header.stamp).seconds();
    }

    if (!has_pc)
    {
      return std::nullopt;
    }

    const double dt_pc = std::abs(pc_stamp_s - stamp_s);
    if (dt_pc > max_lidar_odom_dt)
    {
      return std::nullopt;
    }
  }

  uint64_t new_pose_key;
  gtsam::Pose3 curr_odom_biased;
  gtsam::Pose3 raw_odom;
  {
    std::lock_guard<std::mutex> lock(pose_graph_mtx);

    if (current_pose_key == STARTING_POSE_KEY)
    {
      gtsam::noiseModel::Base::shared_ptr noise;

      if (dynamic_pose_cov)
      {
        Eigen::Matrix<double, 6, 6> Q = ros_twist_to_gtsam_increment_cov(twist_cov, dt);

        noise = gtsam::noiseModel::Gaussian::Covariance(Q);
      }
      else
      {
        noise = prior_noise;
      }

      // Prior Factor
      pose_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(x_(current_pose_key), curr_odom_raw, noise);
    }
    else
    {
      // calculatin the relative pose between (current_pose_key-1) and current_pose_key
      gtsam::Pose3 odom_edge = prev_odom_raw.between(curr_odom_raw);

      // return if either of the pose-adding-crits are satisfied
      if (!is_anchor && !linear_crit(odom_edge) && !angular_crit(odom_edge) && !timeout_crit(odom_edge))
        return std::nullopt;

      gtsam::noiseModel::Base::shared_ptr noise;

      if (dynamic_pose_cov)
      {
        Eigen::Matrix<double, 6, 6> Q = ros_twist_to_gtsam_increment_cov(twist_cov, dt);

        noise = gtsam::noiseModel::Gaussian::Covariance(Q);
      }
      else
      {
        noise = odometry_noise;
      }

      // Adding Odometry Constraint between (current_pose_key-1) and current_pose_key
      pose_graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(x_(current_pose_key - 1), x_(current_pose_key),
                                                                    odom_edge, noise);
    }

    // translating from raw (drifted) odometry pose into loop-closed map frame
    curr_odom_biased = estimated_bias * curr_odom_raw;

    // Adding the pose node into Values
    initial.insert(x_(current_pose_key), curr_odom_biased);

    last_pose_added_time = this->now();
    prev_odom_raw = curr_odom_raw;
    prev_odom_cov = odom_cov;
    new_pose_key = current_pose_key++;

    raw_odom = curr_odom_raw;

    if (pose_timestamps_.size() <= new_pose_key)
      pose_timestamps_.resize(new_pose_key + 1);
    pose_timestamps_[new_pose_key] = stamp_s;

    if (raw_odom_poses_.size() <= new_pose_key)
      raw_odom_poses_.resize(new_pose_key + 1);
    raw_odom_poses_[new_pose_key] = curr_odom_raw;
  }

  // Providing the keyframe point cloud and corresponding key to other nodes
  if (!is_anchor)
  {
    sensor_msgs::msg::PointCloud2::UniquePtr pc_local;
    {
      std::lock_guard<std::mutex> lock(lidar_mtx);
      pc_local = std::move(last_pc_msg);
      last_pc_msg.reset();
    }

    if (!pc_local)
    {
      LOGE_ANY(this->get_logger(), "No pointcloud available for key %lu -> skipping publish/save", new_pose_key);
    }
    else
    {
      pc_storage->put(new_pose_key, *pc_local, false);

      publish_pc_w_key(new_pose_key);
      pc_local.reset();
    }
  }

  return new_pose_key;
}

void PoseGraph::callback_loop_edge(const pogolm_interfaces::msg::LoopEdge& msg)
{
  if (!node_exists(msg.source_key) || !node_exists(msg.target_key))
    return;

  const Eigen::Matrix4d T_mat =
      Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(msg.tranformation.data());
  Eigen::Isometry3d T(T_mat);

  Eigen::Matrix<double, 6, 6> gtsam_loop_hess = ros_to_gtsam_cov(msg.hessian);
  gtsam_loop_hess = 0.5 * (gtsam_loop_hess + gtsam_loop_hess.transpose());

  if (!gtsam_loop_hess.allFinite())
  {
    LOGE_ANY(this->get_logger(), "Loop edge info matrix has NaN/Inf -> rejecting edge");
    return;
  }

  auto information = gtsam::noiseModel::Gaussian::Information(gtsam_loop_hess);
  auto robust = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(1.3), information);

  double loop_dist = T.translation().norm();

  if (verbose)
    LOGI_LOOP(this->get_logger(), "revieced <%lu,%lu> norm: %.4f", key_(msg.source_key), key_(msg.target_key),
              loop_dist);

  auto loop_cand = std::make_shared<LoopEdgeCandidate>();
  loop_cand->transformation = T;
  loop_cand->hessian = gtsam_loop_hess;
  loop_cand->noise = robust;
  loop_cand->target_key = msg.target_key;
  loop_cand->source_key = msg.source_key;
  loop_cand->error = msg.error;
  loop_cand->inliers = msg.inliers;
  loop_cand->kiss_inliers = msg.kiss_inliers;
  loop_cand->sc_dist = msg.sc_dist;

  rclcpp::Duration dur = this->now() - this->now();

  {
    std::lock_guard<std::mutex> lock(loop_cand_mtx);
    loop_candidates.emplace_back(loop_cand);

    lc_num.emplace_back(loop_candidates.size());
  }

  {
    std::lock_guard<std::mutex> lock(optim_mtx);
    optim_requested = true;
  }
  optim_cv.notify_one();
}

void PoseGraph::optimization_worker()
{
  rclcpp::Logger log = rclcpp::get_logger("optim_thread");

  gtsam::LevenbergMarquardtParams params = gtsam::LevenbergMarquardtParams::LegacyDefaults();
  params.linearSolverType = gtsam::NonlinearOptimizerParams::LinearSolverType::MULTIFRONTAL_CHOLESKY;

  while (running)
  {
    {
      std::unique_lock<std::mutex> lock(optim_mtx);
      optim_cv.wait(lock, [&] { return optim_requested.load() || !running.load(); });
      optim_requested = false;
      optim_in_process = true;
    }

    if (!running)
      break;

    auto t1 = std::chrono::steady_clock::now();

    std::vector<std::shared_ptr<LoopEdgeCandidate>> local_loop_edge_buffer;
    {
      std::lock_guard<std::mutex> lock(loop_cand_mtx);
      local_loop_edge_buffer.swap(loop_candidates);
      loop_edge_buffered.store(false, std::memory_order_release);
    }

    gtsam::NonlinearFactorGraph pose_graph_local;
    gtsam::Values initial_local;
    gtsam::Pose3 tail_raw;
    gtsam::Key tail_key;
    {
      std::lock_guard<std::mutex> lock(pose_graph_mtx);

      // If previously no loop edge was buffered, empty for-loop
      pose_graph_local = pose_graph;
      initial_local = initial;
      tail_raw = prev_odom_raw;
      tail_key = current_pose_key - 1;
    }

    for (const auto& cand : local_loop_edge_buffer)
    {
      pose_graph_local.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(cand->source_key, cand->target_key,
                                                                          to_pose3(cand->transformation), cand->noise);
    }

    gtsam::LevenbergMarquardtOptimizer optimizer(pose_graph_local, initial_local, params

    );

    if (verbose)
      LOGI_OPT(log, "================== Optimizing ==================");

    double initial_error = pose_graph_local.error(initial_local);

    gtsam::Values optimized_values = optimizer.optimize();

    double final_error = pose_graph_local.error(optimized_values);

    pose_graph_local.resize(0);
    pose_graph_local = gtsam::NonlinearFactorGraph();

    initial_local.clear();
    initial_local = gtsam::Values();

    double reduction_ratio = 0.0;
    if (initial_error > 1e-12)
    {
      double reduction_frac = final_error / initial_error;
      reduction_ratio = 1.0 - reduction_frac;
    }

    {
      std::lock_guard<std::mutex> lock(pose_graph_mtx);

      // track the optimized loop edges
      for (const auto& cand : local_loop_edge_buffer)
      {
        loop_edge_tracker.emplace_back(cand);
      }

      // update the robot poses
      initial.update(optimized_values);

      // update the landmarks
      for (const auto& v : optimized_values)
      {
        const auto point_v = dynamic_cast<const gtsam::GenericValue<gtsam::Point3>*>(&v.value);
        const uint64_t key = key_(v.key);
        if (!point_v || !landmark_exists_unsafe(key))
          continue;

        gtsam::Point3 new_point = optimized_values.at<gtsam::Point3>(l_(key));
        update_point_unsafe(key, new_point);
      }

      // compute updated bias
      gtsam::Pose3 tail_optimized = optimized_values.at<gtsam::Pose3>(x_(tail_key));
      gtsam::Pose3 updated_bias = tail_optimized * tail_raw.inverse();
      gtsam::Pose3 bias_delta = updated_bias * estimated_bias.inverse();

      // apply bias delta (dt old –> new) on all those poses that were added after storing snapshot (initial_local)
      for (uint64_t k = tail_key + 1; k < current_pose_key; k++)
      {
        if (!node_exists_unsafe(x_(k)))
          continue;

        gtsam::Pose3 pose_k = initial.at<gtsam::Pose3>(x_(k));
        pose_k = bias_delta * pose_k;
        initial.update(x_(k), pose_k);
      }

      // update global bias
      estimated_bias = updated_bias;

      // log the loop edges after optimization step
      for (auto it = loop_edge_tracker.begin(); it != loop_edge_tracker.end(); it++)
      {
        print_cand_error(*it, optimized_values, false);
      }

      publish_loop_edge_tracker(loop_edge_tracker);
    }

    optim_in_process.store(false, std::memory_order_release);

    auto t2 = std::chrono::steady_clock::now();
    double optim_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    if (verbose)
    {
      LOGI_OPT(log, "~~~~~~~~~ reduction ratio %.4f", reduction_ratio);
      LOGI_OPT(log, "––––––––– duration:\t\t\t%.4f ms", optim_duration);
      LOGI_OPT(log, "––––––––– initial error:\t\t\t%.4f", initial_error);
      LOGI_OPT(log, "––––––––– final error:\t\t\t%.4f", final_error);
      LOGI_OPT(log, "================================================");
    }

    {
      std::lock_guard<std::mutex> lock(loop_cand_mtx);
      optim_durs.emplace_back(optim_duration);
    }

    publish_pose_graph();
    publish_debug_info();
  }
  LOGI_OPT(log, "Worker thread terminates");
}

void PoseGraph::print_cand_error(const std::shared_ptr<LoopEdgeCandidate>& cand, const gtsam::Values& values,
                                 bool verbose)
{
  if (!values.exists(cand->source_key) || !values.exists(cand->target_key))
    return;

  auto source_pose = values.at<gtsam::Pose3>(cand->source_key);
  auto target_pose = values.at<gtsam::Pose3>(cand->target_key);

  gtsam::Pose3 T_odom = source_pose.between(target_pose);
  gtsam::Pose3 T_icp = to_pose3(cand->transformation);

  cand->trans_err = static_cast<float>((T_odom.translation() - T_icp.translation()).norm());
  double yaw_err = std::abs((T_odom.rotation().between(T_icp.rotation()).ypr())(0));
  cand->yaw_err_deg = static_cast<float>(yaw_err * 180.0 / M_PI);

  if (verbose)
  {
    LOGI_LOOP_TRACK(this->get_logger(), "––––––––– <%lu,%lu>", key_(cand->source_key), key_(cand->target_key));
    LOGI_LOOP_TRACK(this->get_logger(), "––––––––– translation: %.4f m", cand->trans_err);
    LOGI_LOOP_TRACK(this->get_logger(), "––––––––– rotation: %.4f °", cand->yaw_err_deg);
  }
}

void PoseGraph::publish_pose_graph()
{
  gtsam::NonlinearFactorGraph pose_graph_local;
  gtsam::Values initial_local;
  {
    std::lock_guard<std::mutex> lock(pose_graph_mtx);
    pose_graph_local = pose_graph;
    initial_local = initial;
  }

  auto pg_msg = factor_graph_interfaces::convert_to_msg(pose_graph_local, initial_local);
  pg_msg->header.stamp = this->now();
  pg_msg->header.frame_id = operating_frame_id;

  for (auto& pose : pg_msg->poses)
  {
    pose.type = factor_graph_interfaces::msg::PoseWithID::POINTS;
  }

  for (auto& landmark : pg_msg->points)
  {
    landmark.type = factor_graph_interfaces::msg::PointWithID::LANDMARK;
  }

  auto pg_unique = std::make_unique<factor_graph_interfaces::msg::FactorGraph>(std::move(*pg_msg));

  fg_publisher->publish(std::move(pg_unique));
}

void PoseGraph::publish_pc_w_key(const uint64_t& key)
{
  auto pc_w_key = std::make_unique<pogolm_interfaces::msg::Key>();
  pc_w_key->key = key;
  pc_key_publisher->publish(std::move(pc_w_key));
}

void PoseGraph::fast_remove_factor(uint64_t factor_key)
{
  if (factor_key >= pose_graph.size())
    return;

  pose_graph.remove(factor_key);
}

/**
 * ====================================================================================================
 * **********************************     LANDMARK MANAGEMENT    **************************************
 * ====================================================================================================
 */

void PoseGraph::print_found_neighbors(const uint64_t& query_key, const SearchResult& result)
{
  std::ostringstream oss;
  oss << "Landmark " << query_key << " has neighbors: ";

  for (const auto& [key, dist] : result.dists)
  {
    oss << key << " (d=" << std::fixed << std::setprecision(3) << dist << ") ";
  }
  LOGI_KD(this->get_logger(), "%s", oss.str().c_str());
}

std::vector<uint64_t> PoseGraph::add_landmark(const gtsam::Point3& p, /* in local coord system */
                                              const std::string label, const std::vector<uint64_t>& association_list)
{
  // Check if valid input
  if (std::isnan(p.x()) || std::isnan(p.y()) || std::isnan(p.z()))
    throw std::invalid_argument("Landmark contains NaN values");

  auto new_anchor = add_pose(true);
  if (!new_anchor)
    return {};

  uint64_t new_landmark_key;
  {
    std::lock_guard<std::mutex> lock(pose_graph_mtx);
    new_landmark_key = current_landmark_key++;

    // rebuild if enough landmarks are inserted
    if (landmark_data.size() % IKD_REBUILD_THRES == 0)
      rebuild_ikdtree();

    auto anchor_pose = initial.at<gtsam::Pose3>(x_(*new_anchor));
    auto p_g = anchor_pose.transformFrom(p);

    Landmark new_lm;
    new_lm.key = new_landmark_key;
    new_lm.label = label;
    new_lm.point = LMPoint(p_g.x(), p_g.y(), p_g.z(), new_lm.key);
    new_lm.anchor_list.insert(*new_anchor);

    if (!association_list.empty())
    {
      merge_neighbors(new_lm, association_list);
    }

    // Adding the landmark into the kd-tree
    auto buf = get_buffer(new_lm.point);

    if (lm_ikd_tree.size() == 0)
      lm_ikd_tree.Build(buf);
    else
      lm_ikd_tree.Add_Points(buf, false);

    // Adding the landmark node into pose graph
    initial.insert(l_(new_lm.key), to_gtsam_p(new_lm.point));

    // Storing landmark data
    landmark_data[new_lm.key] = std::make_shared<Landmark>(new_lm);

    for (auto& pose_key : new_lm.anchor_list)
    {
      gtsam::Pose3 pose = initial.at<gtsam::Pose3>(x_(pose_key));
      gtsam::Unit3 bearing = pose.bearing(to_gtsam_p(new_lm.point));
      double range = pose.range(to_gtsam_p(new_lm.point));

      pose_graph.emplace_shared<gtsam::BearingRangeFactor<gtsam::Pose3, gtsam::Point3>>(x_(pose_key), l_(new_lm.key),
                                                                                        bearing, range, bearing_noise);

      // store the factor key into the lookup table
      uint64_t lookup_key = encode_keys(pose_key, new_lm.key);
      factor_lookup[lookup_key] = static_cast<uint64_t>(pose_graph.size()) - 1;
    }
  }

  publish_pose_graph();

  return { new_landmark_key };
}

pogolm_interfaces::msg::Landmark PoseGraph::to_ros_landmark(const Landmark& lm)
{
  pogolm_interfaces::msg::Landmark out;
  out.key = lm.key;
  out.label = lm.label;

  out.position.x = static_cast<double>(lm.point.x);
  out.position.y = static_cast<double>(lm.point.y);
  out.position.z = static_cast<double>(lm.point.z);

  out.anchor_keys.reserve(lm.anchor_list.size());
  for (auto k : lm.anchor_list)
    out.anchor_keys.push_back(k);

  return out;
}

void PoseGraph::handle_store_landmark(const std::shared_ptr<pogolm_interfaces::srv::StoreLandmark::Request> req,
                                      std::shared_ptr<pogolm_interfaces::srv::StoreLandmark::Response> res)
{
  try
  {
    const std::string label = req->label.empty() ? "LM" : req->label;

    // req->position ist im ROBOT-Frame (lokal), so wie add_landmark() es erwartet
    gtsam::Point3 p(req->position.x, req->position.y, req->position.z);

    auto keys = add_landmark(p, label, req->association_list);

    if (keys.empty())
    {
      res->success = false;
      res->message = "add_landmark returned empty";
      res->new_key = 0;
      res->replaced_keys.clear();
      return;
    }

    // Dein add_landmark liefert aktuell {new_key} zurück.
    // Falls du später Merge-Keys zusätzlich zurückgeben willst:
    // keys = {new_key, replaced0, replaced1, ...}
    res->new_key = keys.front();
    if (keys.size() > 1)
    {
      res->replaced_keys.assign(keys.begin() + 1, keys.end());
    }
    else
    {
      res->replaced_keys.clear();
    }

    res->success = true;
    res->message = "ok";
  }
  catch (const std::exception& e)
  {
    res->success = false;
    res->message = std::string("exception: ") + e.what();
    res->new_key = 0;
    res->replaced_keys.clear();
  }
}

void PoseGraph::handle_query_landmark(const std::shared_ptr<pogolm_interfaces::srv::QueryLandmark::Request> req,
                                      std::shared_ptr<pogolm_interfaces::srv::QueryLandmark::Response> res)
{
  try
  {
    std::vector<std::shared_ptr<Landmark>> found;

    switch (req->mode)
    {
      case 0: {  // GET_ALL
        found = get_all_lm();
        break;
      }
      case 1: {  // BY_KEY
        found = get_lm_by_key(req->key);
        break;
      }
      case 2: {  // BY_LABEL
        found = get_lm_by_label(req->label);
        break;
      }

      case 3: {  // KNN_BY_KEY
        found = get_neighbors_knn_by_key(req->k, req->key);
        break;
      }
      case 4: {  // KNN_BY_LABEL
        found = get_neighbors_knn_by_label(req->k, req->label);
        break;
      }
      case 5: {  // KNN_BY_LANDMARK
        Landmark q;
        q.key = req->landmark.key;
        q.label = req->landmark.label;
        q.point = LMPoint(static_cast<float>(req->landmark.position.x), static_cast<float>(req->landmark.position.y),
                          static_cast<float>(req->landmark.position.z), q.key);
        found = get_neighbors_knn_by_lm(req->k, q);
        break;
      }

      case 6: {  // RANGE_BY_KEY
        found = get_neighbors_range_by_key(req->range, req->key);
        break;
      }
      case 7: {  // RANGE_BY_LABEL
        found = get_neighbors_range_by_label(req->range, req->label);
        break;
      }
      case 8: {  // RANGE_BY_LANDMARK
        Landmark q;
        q.key = req->landmark.key;
        q.label = req->landmark.label;
        q.point = LMPoint(static_cast<float>(req->landmark.position.x), static_cast<float>(req->landmark.position.y),
                          static_cast<float>(req->landmark.position.z), q.key);
        found = get_neighbors_range_by_lm(req->range, q);
        break;
      }

      default:
        res->success = false;
        res->message = "unknown query mode";
        res->landmarks.clear();
        return;
    }

    res->landmarks.clear();
    res->landmarks.reserve(found.size());
    for (const auto& lm_ptr : found)
    {
      if (!lm_ptr)
        continue;
      res->landmarks.push_back(to_ros_landmark(*lm_ptr));
    }

    res->success = true;
    res->message = "ok";
  }
  catch (const std::exception& e)
  {
    res->success = false;
    res->message = std::string("exception: ") + e.what();
    res->landmarks.clear();
  }
}

std::vector<std::shared_ptr<Landmark>> PoseGraph::get_lm_by_key(uint64_t query_key)
{
  if (!node_exists(l_(query_key)) || !landmark_exists(query_key))
    throw std::runtime_error("Index out of bounds");

  std::vector<std::shared_ptr<Landmark>> ret;
  {
    std::lock_guard<std::mutex> lock(pose_graph_mtx);
    ret.emplace_back(landmark_data[query_key]);
  }

  return ret;
}

std::vector<std::shared_ptr<Landmark>> PoseGraph::get_lm_by_label(const std::string query_label)
{
  std::vector<std::shared_ptr<Landmark>> ret;
  {
    std::lock_guard<std::mutex> lock(pose_graph_mtx);

    for (const auto& it : landmark_data)
    {
      if (it.second->label == query_label)
        ret.emplace_back(it.second);
    }
  }

  return ret;
}

std::vector<std::shared_ptr<Landmark>> PoseGraph::get_all_lm()
{
  std::lock_guard<std::mutex> lock(pose_graph_mtx);

  std::vector<std::shared_ptr<Landmark>> ret;
  size_t n = static_cast<size_t>(landmark_data.size());
  ret.reserve(n);

  for (const auto& it : landmark_data)
  {
    ret.push_back(it.second);
  }

  return ret;
}

LMPoint PoseGraph::get_lm_point(const uint64_t key)
{
  auto it = landmark_data.find(key);
  if (it == landmark_data.end())
  {
    LOGI_LM(this->get_logger(), "key %lu not found", key);
    return LMPoint();
  }
  return it->second->point;
}

bool PoseGraph::landmark_exists(const uint64_t key)
{
  std::lock_guard<std::mutex> lock(pose_graph_mtx);
  return landmark_data.find(key) != landmark_data.end();
}

bool PoseGraph::landmark_exists_unsafe(uint64_t key)
{
  return landmark_data.find(key) != landmark_data.end();
}

bool PoseGraph::remove_landmark(const uint64_t query_landmark_key)
{
  if (!node_exists(l_(query_landmark_key)) || !landmark_exists(query_landmark_key))
    return false;

  std::lock_guard<std::mutex> lock(pose_graph_mtx);

  const auto& lm_to_remove = landmark_data[query_landmark_key];

  for (uint64_t pose_key : lm_to_remove->anchor_list)
  {
    // mark the factor as nullptr
    uint64_t lookup_key = encode_keys(pose_key, query_landmark_key);
    auto it = factor_lookup.find(lookup_key);
    if (it != factor_lookup.end())
    {
      fast_remove_factor(it->second);
      factor_lookup.erase(it);
    }
  }

  auto buf = get_buffer(lm_to_remove->point);
  lm_ikd_tree.Delete_Points(buf);

  initial.erase(l_(query_landmark_key));
  landmark_data.erase(query_landmark_key);

  return true;
}

bool PoseGraph::remove_landmark_unsafe(uint64_t query_landmark_key)
{
  if (!node_exists_unsafe(l_(query_landmark_key)) || !landmark_exists_unsafe(query_landmark_key))
    return false;

  const auto& lm_to_remove = landmark_data[query_landmark_key];

  for (uint64_t pose_key : lm_to_remove->anchor_list)
  {
    // mark the factor as nullptr
    uint64_t lookup_key = encode_keys(pose_key, query_landmark_key);
    auto it = factor_lookup.find(lookup_key);
    if (it != factor_lookup.end())
    {
      fast_remove_factor(it->second);
      factor_lookup.erase(it);
    }
  }

  auto buf = get_buffer(lm_to_remove->point);
  lm_ikd_tree.Delete_Points(buf);

  initial.erase(l_(query_landmark_key));
  landmark_data.erase(query_landmark_key);

  return true;
}

void PoseGraph::rebuild_ikdtree()
{
  if (landmark_data.size() == 0)
    return;

  auto buf = get_buffer(landmark_data);
  lm_ikd_tree.Build(buf);
}

void PoseGraph::range_search(const float& radius, const Landmark& lm, SearchResult& result)
{
  std::vector<LMPoint, Eigen::aligned_allocator<LMPoint>> result_points;
  auto& query_point = lm.point;

  result.dists.clear();

  if (landmark_data.size() == 0)
  {
    LOGI_KD(this->get_logger(), "Only one landmark — skip range search");
    return;
  }

  lm_ikd_tree.Radius_Search(query_point, radius, result_points);

  if (result_points.empty())
    return;

  for (auto& p : result_points)
  {
    if (p.key == lm.key)
      continue;
    result.dists[p.key] = euclidean_distance_ikd(p, query_point);
  }
}

void PoseGraph::knn_search(const int k, const Landmark& lm, SearchResult& result)
{
  std::vector<LMPoint, Eigen::aligned_allocator<LMPoint>> result_points;
  std::vector<float> sq_dists;
  auto& query_point = lm.point;

  result.dists.clear();

  if (landmark_data.size() == 0)
  {
    LOGI_KD(this->get_logger(), "Only one landmark — skip range search");
    return;
  }

  lm_ikd_tree.Nearest_Search(query_point, k, result_points, sq_dists);

  if (result_points.empty())
    return;

  for (size_t i = 0; i < result_points.size(); i++)
  {
    auto& p = result_points[i];
    if (p.key == lm.key)
      continue;
    result.dists[p.key] = sq_dists[i];
  }
}

void PoseGraph::update_point(const uint64_t key, const gtsam::Point3& new_value)
{
  if (!node_exists(l_(key)) || !landmark_exists(key))
    throw std::runtime_error("Index out of bounds");

  std::lock_guard<std::mutex> lock(pose_graph_mtx);

  update_point_unsafe(key, new_value);
}

void PoseGraph::update_point_unsafe(const uint64_t key, const gtsam::Point3& new_value)
{
  if (!node_exists_unsafe(l_(key)) || !landmark_exists_unsafe(key))
    throw std::runtime_error("Index out of bounds");

  std::shared_ptr<Landmark>& lm = landmark_data[key];
  LMPoint new_point = to_ikd_p(new_value);
  new_point.key = key;

  // update the landmark in the ikd-tree
  auto old_buf = get_buffer(lm->point);
  lm_ikd_tree.Delete_Points(old_buf);
  auto new_buf = get_buffer(new_point);
  lm_ikd_tree.Add_Points(new_buf, false);

  // update the landmark in the data
  lm->point = new_point;

  // update the landmark in the pose graph
  initial.update(l_(lm->key), new_value);

  SearchResult result;
  range_search(range_radius, *lm, result);

  if (!result.dists.empty())
  {
    merge_neighbors(*lm, result);
  }
}

void PoseGraph::merge_neighbors(Landmark& lm, const std::vector<uint64_t>& associations)
{
  SearchResult in;

  for (const auto a : associations)
  {
    in.dists[a] = 0.0;
  }

  merge_neighbors(lm, in);
}

void PoseGraph::merge_neighbors(Landmark& lm, const SearchResult& result)
{
  const double n = static_cast<double>(result.dists.size()) + 1.0;  // +1 because of the query point
  gtsam::Point3 mean = to_gtsam_p(lm.point);
  std::unordered_set<uint64_t> merged_anchors;
  merged_anchors.insert(lm.anchor_list.begin(), lm.anchor_list.end());

  uint64_t min_landmark_key = lm.key;
  std::vector<uint64_t> merged_keys;

  // For each neighboring landmark
  for (const auto& [n_lm_key, _] : result.dists)
  {
    if (!node_exists_unsafe(l_(n_lm_key)) || !landmark_exists_unsafe(n_lm_key))
      continue;

    // Concatinate the anchor lists of the neighbors of query_key
    const auto& it = landmark_data.find(n_lm_key);
    merged_anchors.insert(it->second->anchor_list.begin(), it->second->anchor_list.end());

    // Store the neighbor key for meta information
    merged_keys.push_back(n_lm_key);

    // Calculate the mean of the landmark positions
    mean += to_gtsam_p(get_lm_point(n_lm_key));

    // Select min
    min_landmark_key = std::min(min_landmark_key, n_lm_key);

    // Remove the neighbor landmark
    remove_landmark_unsafe(n_lm_key);
  }

  mean = mean / n;

  lm.anchor_list.swap(merged_anchors);

  MergedInfo merge_info{ lm.key, merged_keys };
  publish_merge_info(merge_info);
}

std::vector<std::shared_ptr<Landmark>> PoseGraph::get_neighbors_range_by_lm(const float radius,
                                                                            const Landmark& query_lm)
{
  std::vector<std::shared_ptr<Landmark>> ret;
  SearchResult res;

  std::lock_guard<std::mutex> lock(pose_graph_mtx);

  range_search(radius, query_lm, res);
  if (res.dists.empty())
    return {};

  ret.reserve(res.dists.size());
  for (const auto& [k, _] : res.dists)
  {
    auto it = landmark_data.find(k);
    if (it == landmark_data.end())
      continue;
    ret.push_back(it->second);
  }
  return ret;
}

std::vector<std::shared_ptr<Landmark>> PoseGraph::get_neighbors_range_by_key(const float radius,
                                                                             const uint64_t query_key)
{
  auto lm = get_lm_by_key(query_key);

  if (lm.empty())
    return {};

  return get_neighbors_range_by_lm(radius, *lm[0]);
}

std::vector<std::shared_ptr<Landmark>> PoseGraph::get_neighbors_range_by_label(const float radius,
                                                                               const std::string query_label)
{
  auto lm = get_lm_by_label(query_label);

  if (lm.empty())
    return {};

  return get_neighbors_range_by_lm(radius, *lm[0]);
}

std::vector<std::shared_ptr<Landmark>> PoseGraph::get_neighbors_knn_by_lm(const int k, const Landmark& query_lm)
{
  std::vector<std::shared_ptr<Landmark>> ret;
  SearchResult res;

  std::lock_guard<std::mutex> lock(pose_graph_mtx);

  knn_search(k, query_lm, res);
  if (res.dists.empty())
    return {};

  ret.reserve(res.dists.size());
  for (const auto& [key, _] : res.dists)
  {
    auto it = landmark_data.find(key);
    if (it == landmark_data.end())
      continue;
    ret.push_back(it->second);
  }
  return ret;
}

std::vector<std::shared_ptr<Landmark>> PoseGraph::get_neighbors_knn_by_key(const int k, const uint64_t query_key)
{
  auto lm = get_lm_by_key(query_key);

  if (lm.empty())
    return {};

  return get_neighbors_knn_by_lm(k, *lm[0]);
}

std::vector<std::shared_ptr<Landmark>> PoseGraph::get_neighbors_knn_by_label(const int k, const std::string query_label)
{
  auto lm = get_lm_by_label(query_label);

  if (lm.empty())
    return {};

  return get_neighbors_knn_by_lm(k, *lm[0]);
}

/**
 * ====================================================================================================
 */

bool PoseGraph::node_exists(uint64_t key)
{
  bool it_does = false;
  {
    std::lock_guard<std::mutex> lock(pose_graph_mtx);
    it_does = initial.exists(key);
  }
  return it_does;
}

bool PoseGraph::node_exists_unsafe(uint64_t key)
{
  return initial.exists(key);
}

bool PoseGraph::linear_crit(const gtsam::Pose3& relative)
{
  return relative.translation().norm() >= linear_crit_min_dist;
}

bool PoseGraph::angular_crit(const gtsam::Pose3& relative)
{
  double yaw = (relative.rotation().ypr())(0);
  return abs(yaw) >= angular_crit_min_yaw;
}

bool PoseGraph::timeout_crit(const gtsam::Pose3& relative)
{
  rclcpp::Duration dur = this->now() - last_pose_added_time;
  double dist = relative.translation().norm();
  return dur.seconds() > timeout_crit_min_time && dist >= timeout_crit_min_dist;
}

void PoseGraph::publish_debug_info()
{
  using pogolm_interfaces::msg::AnchorLookupEntry;
  using pogolm_interfaces::msg::FactorGraphDebug;
  using pogolm_interfaces::msg::FactorLookupEntry;
  using pogolm_interfaces::msg::LandmarkEntry;

  auto msg = FactorGraphDebug();
  msg.header.stamp = this->now();
  msg.header.frame_id = operating_frame_id;

  {
    std::lock_guard<std::mutex> lock(pose_graph_mtx);

    msg.landmarks.reserve(landmark_data.size());
    for (const auto& [id, lm] : landmark_data)
    {
      LandmarkEntry entry;
      entry.id = id;
      entry.x = lm->point.x;
      entry.y = lm->point.y;
      entry.z = lm->point.z;
      msg.landmarks.push_back(std::move(entry));
    }

    msg.anchor_lookup.reserve(landmark_data.size());
    for (const auto& [id, lm] : landmark_data)
    {
      AnchorLookupEntry entry;
      entry.landmark_key = id;

      for (auto pose_key : lm->anchor_list)
      {
        entry.pose_keys.push_back(pose_key);
      }

      msg.anchor_lookup.push_back(std::move(entry));
    }

    msg.factor_lookup.reserve(factor_lookup.size());
    for (const auto& [lookup_key, factor_index] : factor_lookup)
    {
      FactorLookupEntry entry;
      entry.lookup_key = lookup_key;
      entry.factor_index = factor_index;
      msg.factor_lookup.push_back(std::move(entry));
    }
  }

  fg_debug_publisher->publish(msg);
}

void PoseGraph::publish_loop_edge_tracker(std::vector<std::shared_ptr<LoopEdgeCandidate>>& list)
{
  pogolm_interfaces::msg::LoopEdgeTracker msg;
  msg.header.stamp = this->now();
  msg.header.frame_id = operating_frame_id;

  msg.edges.reserve(list.size());

  for (const auto& cand : list)
  {
    pogolm_interfaces::msg::LoopEdgeTrackerEntry e;
    e.source_key = key_(cand->source_key);
    e.target_key = key_(cand->target_key);
    e.icp_error = static_cast<float>(cand->error);
    e.icp_inliers = static_cast<int32_t>(cand->inliers);
    e.trans_err = cand->trans_err;
    e.yaw_err_deg = cand->yaw_err_deg;
    e.kiss_inliers = cand->kiss_inliers;
    e.sc_dist = cand->sc_dist;
    msg.edges.push_back(e);
  }

  std::vector<std::shared_ptr<LoopEdgeCandidate>>().swap(list);

  loop_tracker_pub->publish(msg);
}

void PoseGraph::publish_merge_info(const MergedInfo& info)
{
  pogolm_interfaces::msg::MergedInfo msg;
  msg.header.stamp = this->now();
  msg.header.frame_id = operating_frame_id;

  msg.new_key = info.new_merged_key;

  for (auto key : info.merged_keys)
  {
    msg.merged_keys.push_back(key);
  }

  merge_info_pub->publish(msg);
}

PoseGraphParams PoseGraph::get_yaml_params()
{
  PoseGraphParams params;

  params.lin_cirt = declare_parameter("linear_criteria_min_dist", 1.5);
  params.ang_crit = declare_parameter("angular_criteria_min_yaw", 0.30);
  params.to_time_crit = declare_parameter("timeout_criteria_min_time", 2.0);
  params.to_dist_crit = declare_parameter("timeout_criteria_min_dist", 0.3);

  params.dynamic_pose_cov = declare_parameter("dynamic_pose_cov", false);
  params.lm_range_rad = declare_parameter("range_radius", 0.4);
  params.operating_frame_id = declare_parameter("operating_frame_id", "odom");
  params.pc_range_thres = declare_parameter("pc_range_thres", 150.0);

  params.odometry_sig = declare_parameter("odometry_sigmas", std::vector<double>{ 0.1, 0.1, 0.1, 0.01, 0.01, 0.01 });
  params.loop_sig = declare_parameter("loop_sigmas", std::vector<double>{ 0.5, 0.5, 0.5, 0.05, 0.05, 0.05 });
  params.prior_sig = declare_parameter("prior_sigmas", std::vector<double>{ 0.01, 0.01, 0.01, 0.3, 0.3, 0.3 });
  params.bearing_sig = declare_parameter("bearing_sigmas", std::vector<double>{ 0.1, 0.1, 0.2 });

  params.lidar_in = declare_parameter("lidar_in", "/velodyne_points");
  params.odom_in = declare_parameter("odom_in", "/odom");

  verbose = declare_parameter("verbose", false);
  tum_out_path_ = declare_parameter("trajectory_path", "/tmp/pogolm");
  log_path = declare_parameter("log_path", "/tmp/pose_graph_logs.txt");
  save_trajectory = declare_parameter("save_trajecotry", false);

  return params;
}

PoseGraphParams PoseGraph::get_default_params()
{
  PoseGraphParams params;

  params.lidar_in = "/velodyne_points";
  params.odom_in = "/odom";
  params.operating_frame_id = "map";
  params.pc_range_thres = 150.;
  params.dynamic_pose_cov = false;
  params.lin_cirt = 1.5;
  params.ang_crit = 0.3;
  params.to_time_crit = 2.0;
  params.to_dist_crit = 0.3;
  params.lm_range_rad = 0.4;
  params.bearing_sig = { 0.1, 0.1, 0.2 };
  params.odometry_sig = { 0.1, 0.1, 0.1, 0.3, 0.3, 0.05 };
  params.loop_sig = { 0.2, 0.2, 0.2, 0.3, 0.3, 0.01 };
  params.prior_sig = { 0.01, 0.01, 0.01, 0.1, 0.1, 0.1 };
  params.save_trajectory = true;
  params.tum_out_path = "/tmp/pogolm";
  params.log_path = "/tmp/pose_graph_logs.txt";
  params.verbose = false;

  return params;
}

void PoseGraph::init_node(const PoseGraphParams& params)
{
  pc_storage = StoreRegistry::get("/tmp/pc_store.bin", 1, true, false);

  LOGI_POGOLM(this->get_logger(), "verbose----------------: %s", (verbose) ? "true" : "false");
  LOGI_POGOLM(this->get_logger(), "map frame--------------: /%s", params.operating_frame_id.c_str());
  LOGI_POGOLM(this->get_logger(), "lidar topic------------: %s", params.lidar_in.c_str());
  LOGI_POGOLM(this->get_logger(), "odometry topic---------: %s", params.odom_in.c_str());
  LOGI_POGOLM(this->get_logger(), "covariance form odom---: %s", (params.dynamic_pose_cov) ? "true" : "false");
  LOGI_POGOLM(this->get_logger(), "lin crit---------------: %.3f m", params.lin_cirt);
  LOGI_POGOLM(this->get_logger(), "ang cirt---------------: %.3f °", params.ang_crit);
  LOGI_POGOLM(this->get_logger(), "to time crit-----------: %.3f sec", params.to_time_crit);
  LOGI_POGOLM(this->get_logger(), "to dist crit-----------: %.3f m", params.to_dist_crit);
  LOGI_POGOLM(this->get_logger(), "save trajectory--------: %s", (params.save_trajectory) ? "true" : "false");
  LOGI_POGOLM(this->get_logger(), "trajectory path--------: %s", params.tum_out_path.c_str());
  LOGI_POGOLM(this->get_logger(), "logging path-----------: %s", params.log_path.c_str());

  const std::string lidar_topic_name = params.lidar_in;
  const std::string odom_topic_name = params.odom_in;

  verbose = params.verbose;
  linear_crit_min_dist = params.lin_cirt;
  angular_crit_min_yaw = params.ang_crit;
  timeout_crit_min_time = params.to_time_crit;
  timeout_crit_min_dist = params.to_dist_crit;
  dynamic_pose_cov = params.dynamic_pose_cov;
  range_radius = params.lm_range_rad;
  operating_frame_id = params.operating_frame_id;
  pc_range_thres = params.pc_range_thres;
  save_trajectory = params.save_trajectory;
  tum_out_path_ = params.tum_out_path;
  log_path = params.log_path;

  test_count = 0;

  prev_odom_raw = gtsam::Pose3::Identity();
  estimated_bias = gtsam::Pose3::Identity();

  auto odometry_sigmas_vec = params.odometry_sig;
  auto loop_sigmas_vec = params.loop_sig;
  auto prior_sigmas_vec = params.prior_sig;
  auto bearing_sigmas_vec = params.bearing_sig;
  gtsam::Vector6 odometry_sigmas, loop_sigmas, prior_sigmas;
  gtsam::Vector3 bearing_sigmas;

  for (int i = 0; i < 6; ++i)
  {
    odometry_sigmas(i) = odometry_sigmas_vec[i];
    loop_sigmas(i) = loop_sigmas_vec[i];
    prior_sigmas(i) = prior_sigmas_vec[i];

    if (i < 3)
    {
      bearing_sigmas(i) = bearing_sigmas_vec[i];
    }
  }

  loop_cov = loop_sigmas.array().square().matrix().asDiagonal();

  loop_closure_noise = gtsam::noiseModel::Diagonal::Sigmas(loop_sigmas);
  odometry_noise = gtsam::noiseModel::Diagonal::Sigmas(odometry_sigmas);
  prior_noise = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
  bearing_noise = gtsam::noiseModel::Diagonal::Sigmas(bearing_sigmas);

  current_pose_key = STARTING_POSE_KEY;
  current_landmark_key = 0;

  // Initialize the SUB/PUB/TIMER
  io_group = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  pg_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions io_opt;
  io_opt.callback_group = io_group;

  // INPUT/OUTPUT GROUP
  odom_subscriber = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_name, 10, std::bind(&PoseGraph::callback_odometry, this, std::placeholders::_1), io_opt);

  pc_subscriber = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_name, 10, std::bind(&PoseGraph::callback_pointcloud, this, std::placeholders::_1), io_opt);

  le_subscriber = this->create_subscription<pogolm_interfaces::msg::LoopEdge>(
      "/pogolm/loop_edge", 10, std::bind(&PoseGraph::callback_loop_edge, this, std::placeholders::_1), io_opt);

  // POSE GRAPH GROUP
  pose_add_timer = this->create_wall_timer(50ms, std::bind(&PoseGraph::callback_pose_add_timer, this), pg_group);

  odom_timeout_timer =
      this->create_wall_timer(500ms, std::bind(&PoseGraph::callback_odom_timeout_timer, this), pg_group);

  // PUBLISHER
  fg_publisher = this->create_publisher<factor_graph_interfaces::msg::FactorGraph>("/pogolm/pose_graph", 10);

  pc_key_publisher = this->create_publisher<pogolm_interfaces::msg::Key>("/pogolm/pointcloud_key", 10);

  fg_debug_publisher = this->create_publisher<pogolm_interfaces::msg::FactorGraphDebug>(
      "/pogolm/pose_graph_debug", rclcpp::QoS(1).best_effort().durability_volatile());

  merge_info_pub = this->create_publisher<pogolm_interfaces::msg::MergedInfo>("/pogolm/merge_info", 10);

  loop_tracker_pub = this->create_publisher<pogolm_interfaces::msg::LoopEdgeTracker>("/pogolm/loop_edge_tracker", 10);

  // SERVICES
  store_landmark_srv = this->create_service<pogolm_interfaces::srv::StoreLandmark>(
      "/pogolm/store_landmark",
      std::bind(&PoseGraph::handle_store_landmark, this, std::placeholders::_1, std::placeholders::_2));

  query_landmark_srv = this->create_service<pogolm_interfaces::srv::QueryLandmark>(
      "/pogolm/query_landmark",
      std::bind(&PoseGraph::handle_query_landmark, this, std::placeholders::_1, std::placeholders::_2));

  // start optimizer thread
  optim_thread = std::thread([this] { this->optimization_worker(); });
}

/**
 * Constructor
 */
PoseGraph::PoseGraph(const PoseGraphParams& params, const rclcpp::NodeOptions& options) : Node("pose_graph", options)
{
  RCLCPP_INFO(this->get_logger(), "IPC: %s", options.use_intra_process_comms() ? "ON" : "OFF");

  init_node(params);
}

PoseGraph::PoseGraph(const rclcpp::NodeOptions& options) : Node("pose_graph", options)
{
  RCLCPP_INFO(this->get_logger(), "IPC: %s", options.use_intra_process_comms() ? "ON" : "OFF");

  auto params = get_yaml_params();

  init_node(params);
}

PoseGraph::~PoseGraph()
{
  running = false;
  optim_requested = true;
  optim_cv.notify_all();
  if (optim_thread.joinable())
    optim_thread.join();

  if (save_trajectory)
  {
    try
    {
      namespace fs = std::filesystem;
      fs::path base(tum_out_path_);
      fs::path global = base / "pogolm_global.tum";
      fs::path odom = base / "pogolm_odom.tum";

      save_trajectory_tum(global.string());
      save_raw_trajectory_tum(odom.string());
    }
    catch (const std::exception& e)
    {
      RCLCPP_ERROR(this->get_logger(), "Trajectory export failed: %s", e.what());
    }
  }

  double sum_optim_dur = 0.0;
  size_t optim_num = optim_durs.size();

  for (size_t i = 0; i < optim_num; i++)
  {
    sum_optim_dur += optim_durs[i];
  }

  size_t sum_loop_cand = 0;
  size_t lc_size = lc_num.size();
  for (size_t i = 0; i < lc_size; i++)
  {
    sum_loop_cand += lc_num[i];
  }

  LineWriter w(log_path, LineWriter::Mode::Append);
  w.writeLineParts("times optimized––––––––––––––––––––––––––––: ", optim_num);
  w.writeLineParts("averege optimization duration––––––––––––––: ", sum_optim_dur / (double)optim_num, " ms");
  w.writeLineParts("total loop edge count––––––––––––––––––––––: ", sum_loop_cand);
  w.writeLineParts("averege loop edge number per optimization––: ", (double)sum_loop_cand / (double)lc_size);
  w.writeLineParts("loop edge number valid–––––––––––––––––––––: ", lc_size == optim_num);
  w.flush();
}

}  // namespace pose_graph_np

#ifndef PGO_WITH_VISUAL_NO_MAIN
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto pg_node = std::make_shared<pose_graph_np::PoseGraph>();

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 5);
  executor.add_node(pg_node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
#endif