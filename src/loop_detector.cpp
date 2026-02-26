#include "pogolm/loop_detector.hpp"

namespace loop_detector_np
{

void LoopDetector::callback_pose_graph(factor_graph_interfaces::msg::FactorGraph::ConstSharedPtr msg)
{
  size_t n = msg->poses.size();

  {
    std::lock_guard<std::mutex> lock(pose_mutex);
    pose_graph_size = n;
  }

  update_core_policy(n);
}

void LoopDetector::callback_cleaner()
{
  auto [trimmed, before, after] = clean_rss();
  if (verbose)
    LOGI_CLEANER(this->get_logger(), " %d MB RAM (trimmed<%s> -> omitted %d MB RAM)", after,
                 (trimmed) ? "true" : "false", std::abs(before - after));
}

void LoopDetector::callback_pc_key(pogolm_interfaces::msg::Key::UniquePtr msg)
{
  auto key = msg->key;
  sensor_msgs::msg::PointCloud2 pc2;
  auto ok = pc_storage->get(key, pc2, false);

  if (!ok)
  {
    LOGI_LIDAR(this->get_logger(), "<%ld> konnte nicht abgeholt werden", key);
    return;
  }

  // create and store destrciptor of the point cloud
  pcl::PointCloud<SCPointType>::Ptr sc_pc(new pcl::PointCloud<SCPointType>);
  pcl::fromROSMsg(pc2, *sc_pc);
  {
    std::lock_guard<std::mutex> lock(sc_mutex);
    sc_manager->makeAndSaveScancontextAndKeys(sc_pc, true);
  }

  DetectorJob job = { key };

  for (auto& core : cores)
  {
    {
      std::lock_guard<std::mutex> lock(core.mtx);

      if (core.enabled.load())
      {
        core.jobs.push_back(job);
        free_core_jobs(core.jobs);
      }
    }
    core.cv.notify_one();
  }
}

void LoopDetector::update_core_policy(const size_t n)
{
  bool updated{ false }, enabled{ false };
  auto& near = cores[idx(NEAR)];
  auto& far = cores[idx(FAR)];

  const CorePolicy* policy = &core_policies.back();
  for (size_t i = 0; i < core_policies.size(); i++)
  {
    if (n <= core_policies[i].thres)
    {
      policy = &core_policies[i];
      current_policy = i;
      break;
    }
  }

  {
    std::scoped_lock lock(near.mtx, far.mtx);

    // If of the parameter sets differ from current policy –> update them
    if (!near.params.is_equal(policy->near) || !far.params.is_equal(policy->far))
    {
      near.params = policy->near;
      far.params = policy->far;
      updated = true;
    }

    // If the policy demands enabling and the core was previously disabled –> enabale far
    if (policy->enable_far && !far.enabled.load(std::memory_order_acquire))
    {
      far.enabled.store(true, std::memory_order_release);
      enabled = true;
    }
  }

  if (enabled)
    far.cv.notify_all();
  if (updated)
  {
    LOGI_POLI(this->get_logger(), "Updated core policy (%ld) for |graph|=%lu (LBN=%d)", current_policy, n,
              policy->near.look_back_n);
  }
}

int LoopDetector::idx(CoreType type)
{
  return (int)type;
}

CoreParams LoopDetector::get_starting_core_params(CoreType type)
{
  auto policy = core_policies.at(0);

  return (type == NEAR) ? policy.near : policy.far;
}

void LoopDetector::core_worker(CoreContext& core)
{
  rclcpp::Logger log = rclcpp::get_logger(core.name());

  while (core.running.load())
  {
    CoreType type;
    CoreParams params;
    uint64_t source_key;
    {
      std::unique_lock<std::mutex> lock(core.mtx);

      core.cv.wait(lock, [&] { return (core.enabled.load() && !core.jobs.empty()) || !core.running.load(); });

      if (!core.running.load())
        break;

      params = core.params;
      type = core.type;

      DetectorJob curr_job = std::move(core.jobs.front());
      core.jobs.pop_front();
      source_key = curr_job.key_to_detect;
    }

    std::vector<DescriptorMatchResult> initial_candidates;
    auto t0 = std::chrono::steady_clock::now();

    uint64_t start_key = params.start(source_key, type);
    uint64_t end_key = params.end(source_key, type);

    // LOGI_CORE(log, "Searching Loops für source_key %lu", source_key);
    // LOGI_CORE(log, "––– [%ld ... %ld]", end_key, start_key);

    // Checking for loops of type source_key –> target_key
    for (uint64_t target_key = start_key; target_key >= end_key; target_key -= params.stride)
    {
      if (target_key < end_key || target_key > start_key)
        break;
      // if (target_key>start_key) break;

      if (!pointcloud_exists(target_key))
        continue;

      uint64_t diff = (source_key > target_key) ? source_key - target_key : target_key - source_key;

      if (diff < (uint64_t)min_loop_length)
        continue;

      if (!params.can_visit(source_key, target_key, type))
        continue;

      // match the descriptors
      auto dm_result = descriptor_matching(target_key, source_key);

      if (dm_result.dist <= params.sc_sim_score)
      {
        initial_candidates.emplace_back(dm_result);
      }
    }

    auto dm_pipeline_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::steady_clock::now() - t0)
            .count();

    dm_durs.emplace_back(dm_pipeline_ms);

    // asc
    sort_initial_candidates(initial_candidates);

    std::vector<DescriptorMatchResult> dm_checked;
    std::vector<ScanMatchResult> sc_checked;

    size_t check_max = std::min((size_t)5, initial_candidates.size());

    for (size_t i = 0; i < check_max; i++)
    {
      auto t1 = std::chrono::steady_clock::now();
      double work_time_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

      if (work_time_ms >= RT_LIMIT_MS)
        break;

      auto initial_candidate = initial_candidates[i];
      dm_checked.emplace_back(initial_candidate);

      try
      {
        auto final_candidate = scan_matching(initial_candidate);

        final_candidate.core = (int)core.type;
        final_candidate.sc_sim_score = initial_candidate.dist;
        final_candidate.sc_duration_ms = dm_pipeline_ms;

        bool verified;
        {
          std::lock_guard<std::mutex> lock(core.mtx);
          verified = verify_loop(final_candidate);
        }

        if (verified)
        {
          sc_checked.emplace_back(final_candidate);
          // core_policies[current_policy].near.look_back_n = final_candidate.source_key - final_candidate.target_key;
          // core_policies[current_policy].far.look_back_n  = final_candidate.source_key - final_candidate.target_key;
          LDPDuration d;
          d.fetch_duration_ms = final_candidate.fetch_duration_ms;
          d.icp_duration_ms = final_candidate.icp_duration_ms;
          d.kiss_duration_ms = final_candidate.kiss_duration_ms;
          d.total_duration_ms = final_candidate.total_duration_ms;
          durs.emplace_back(d);
        }
      }
      catch (const std::exception& ex)
      {
        RCLCPP_ERROR(this->get_logger(), "[CALLBACK_LOOP_DETECT_TIMER] Exception %s", ex.what());
      }
    }

    auto t1 = std::chrono::steady_clock::now();
    double total_work_time_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    total_durs.emplace_back(total_work_time_ms);
    if (total_work_time_ms > RT_LIMIT_MS)
      LOGI_CORE(log, "[%ld] overworked for %.5f ms", source_key, total_work_time_ms - RT_LIMIT_MS);

    publish_loop_debug(source_key, dm_checked, sc_checked);

    last_loop_checked_key.store(std::max(source_key, last_loop_checked_key.load(std::memory_order_relaxed)),
                                std::memory_order_relaxed);
  }

  LOGI_CORE(log, "Core %s terminates", core.name().c_str());
}

DescriptorMatchResult LoopDetector::descriptor_matching(const uint64_t target_key, const uint64_t source_key)
{
  auto t0 = std::chrono::steady_clock::now();

  DescriptorMatchResult dm_result;
  dm_result.source_key = source_key;
  dm_result.target_key = target_key;
  dm_result.guess = Eigen::Isometry3d::Identity();

  // SCManager stores the data in std::vector => ids begin at 0, whereas pose-keys at 1
  uint64_t sc_target_key = target_key - 1;
  uint64_t sc_source_key = source_key - 1;

  std::pair<double, int> match_result;
  {
    std::lock_guard<std::mutex> lock(sc_mutex);

    if (!sc_manager->descriptorExists(sc_target_key) || !sc_manager->descriptorExists(sc_source_key))
    {
      dm_result.dist = 100.;
      return dm_result;
    }

    auto target_descrpitor = sc_manager->getDescriptor(sc_target_key);
    auto source_descriptor = sc_manager->getDescriptor(sc_source_key);

    match_result = sc_manager->distanceBtnScanContext(target_descrpitor, source_descriptor);
  }

  int best_shift = match_result.second;

  double sc_yaw = ((double)best_shift) * (2.0 * M_PI / ((double)NUM_SECT));
  dm_result.guess.linear() = Eigen::AngleAxisd(sc_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  dm_result.dist = match_result.first;

  auto t1 = std::chrono::steady_clock::now();

  dm_result.duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

  return dm_result;
}

ScanMatchResult LoopDetector::scan_matching(const DescriptorMatchResult& dm_result)
{
  auto t0 = std::chrono::steady_clock::now();

  Eigen::Isometry3d initial_guess = dm_result.guess;
  small_gicp::RegistrationResult icp_reg_result;
  ScanMatchResult sm_result;
  long kiss_inliers{ -1 };

  auto t_0_f = std::chrono::steady_clock::now();

  std::shared_ptr<disk::VoxelmapCache> target_cache, source_cache;
  auto target_ret = fetch_voxelmap(dm_result.target_key);
  auto source_ret = fetch_voxelmap(dm_result.source_key);

  if (target_ret)
    target_cache = *target_ret;
  else
    target_cache = build_voxelmap(dm_result.target_key);

  if (source_ret)
    source_cache = *source_ret;
  else
    source_cache = build_voxelmap(dm_result.source_key);

  auto t_1_f = std::chrono::steady_clock::now();

  if (source_cache->pts_eigen.empty() || target_cache->pts_eigen.empty())
  {
    RCLCPP_WARN(this->get_logger(), "[ICP] Leere Punktmenge nach Filter: source=%zu target=%zu",
                source_cache->pts_eigen.size(), target_cache->pts_eigen.size());
    sm_result.icp_result.converged = false;
    return sm_result;
  }

  // Compute an initial guess with KISS-Matcher (global registration)
  auto t_0_k = std::chrono::steady_clock::now();
  kiss_initial_guess(initial_guess, kiss_inliers, *target_cache, *source_cache);
  auto t_1_k = std::chrono::steady_clock::now();

  target_cache->pts_eigen.clear();
  target_cache->pts_eigen.shrink_to_fit();
  source_cache->pts_eigen.clear();
  source_cache->pts_eigen.shrink_to_fit();

  sm_result.kiss_converged = kiss_inliers >= KISS_INLIER_TRES;
  // sm_result.kiss_converged = true;

  if (sm_result.kiss_converged)
  {
    // Compute relative transformation with small_gicp (local registration)
    auto t_0_s = std::chrono::steady_clock::now();
    perform_icp_alignment(initial_guess, icp_reg_result, *target_cache, *source_cache);
    auto t_1_s = std::chrono::steady_clock::now();

    sm_result.icp_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_s - t_0_s).count();
  }
  else
  {
    sm_result.icp_result.converged = false;
  }

  target_cache->free();
  source_cache->free();
  target_cache.reset();
  source_cache.reset();

  sm_result.source_key = dm_result.source_key;
  sm_result.target_key = dm_result.target_key;
  sm_result.icp_result = icp_reg_result;
  sm_result.sc_duration_ms = dm_result.duration_ms;
  sm_result.translation_norm = icp_reg_result.T_target_source.translation().norm();
  sm_result.sc_sim_score = dm_result.dist;
  sm_result.kiss_inliers = kiss_inliers;

  auto t_0_r = std::chrono::steady_clock::now();
  // sm_result.rmse = compute_rmse(source_cache, target_cache, icp_reg_result.T_target_source);
  auto t_1_r = std::chrono::steady_clock::now();

  auto t1 = std::chrono::steady_clock::now();

  sm_result.kiss_duration_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_k - t_0_k).count();
  sm_result.rmse_duration_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_r - t_0_r).count();
  sm_result.fetch_duration_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_f - t_0_f).count();
  sm_result.total_duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

  return sm_result;
}

ScanMatchResult LoopDetector::scan_matching(const Eigen::Isometry3d& init_guess, const uint64_t target_key,
                                            const uint64_t source_key)
{
  DescriptorMatchResult temp;
  temp.duration_ms = 0.0;
  temp.dist = 1.0;
  temp.guess = init_guess;
  temp.source_key = source_key;
  temp.target_key = target_key;
  return this->scan_matching(temp);
}

void LoopDetector::perform_icp_alignment(Eigen::Isometry3d& initial_guess, small_gicp::RegistrationResult& icp_reg_res,
                                         const disk::VoxelmapCache& target, const disk::VoxelmapCache& source)
{
  static thread_local std::unique_ptr<
      small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP>>
      reg;

  if (!reg)
  {
    reg = std::make_unique<small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP>>();
    reg->reduction.num_threads = NUM_TH;
    reg->rejector.max_dist_sq = MAX_C_DIST * MAX_C_DIST;
    reg->optimizer.max_iterations = MAX_ITER;
    reg->criteria.rotation_eps = EPS_R;
    reg->criteria.translation_eps = EPS_T;
  }

  const auto icp_result = reg->align(*target.vm_sg, *source.pc_sg, *target.vm_sg, initial_guess);

  icp_reg_res = icp_result;
}

void LoopDetector::kiss_initial_guess(Eigen::Isometry3d& init_guess, long& kiss_inliers,
                                      const disk::VoxelmapCache& target, const disk::VoxelmapCache& source)
{
  static thread_local std::unique_ptr<kiss_matcher::KISSMatcher> matcher;

  if (!matcher)
    matcher = std::make_unique<kiss_matcher::KISSMatcher>(*kiss_config);

  auto source_transformed = apply_transform(source.pts_eigen, init_guess);

  auto kiss_solution = matcher->estimate(source_transformed, target.pts_eigen);

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = kiss_solution.rotation.cast<double>();
  T.translation() = kiss_solution.translation.cast<double>();

  init_guess = T * init_guess;
  kiss_inliers = static_cast<long>(matcher->getNumFinalInliers());
  matcher->clear();
}

void LoopDetector::free_core_jobs(std::deque<DetectorJob>& job_queue)
{
  while (job_queue.size() > JOB_LIMIT)
  {
    job_queue.pop_front();
  }
}

std::optional<std::shared_ptr<disk::VoxelmapCache>> LoopDetector::fetch_voxelmap(uint64_t key)
{
  auto out = std::make_shared<disk::VoxelmapCache>();
  bool ok = vm_storage->get(key, *out, false);

  if (ok)
    return out;

  return std::nullopt;
}

std::shared_ptr<disk::VoxelmapCache> LoopDetector::build_voxelmap(const uint64_t key)
{
  auto t_0 = std::chrono::steady_clock::now();

  auto cache = std::make_shared<disk::VoxelmapCache>();

  sensor_msgs::msg::PointCloud2 pc2;
  auto ok = pc_storage->get(key, pc2, false);

  if (!ok)
  {
    LOGI_KD(this->get_logger(), "<%ld> konnte nicht abgeholt werden", key);
    return cache;
  }

  auto pts = pc2_to_vec3f_fast(pc2);
  auto pts_ds = kiss_matcher::VoxelgridSampling(pts, KISS_RES);
  cache->pts_eigen = std::move(pts_ds);

  // if cloud and tree not cached –> build and cache them
  cache->pc_sg = std::make_shared<small_gicp::PointCloud>(pts);
  cache->pc_sg = small_gicp::voxelgrid_sampling_omp(*cache->pc_sg, DS_RES, NUM_TH / 2);
  auto kdtree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(cache->pc_sg,
                                                                             small_gicp::KdTreeBuilderOMP(NUM_TH / 2));
  small_gicp::estimate_covariances_omp(*cache->pc_sg, *kdtree, NUM_N, NUM_TH / 2);

  cache->vm_sg = std::make_shared<small_gicp::GaussianVoxelMap>(VX_RES);
  cache->vm_sg->insert(*cache->pc_sg);

  vm_storage->put(key, *cache, false);

  auto t_1 = std::chrono::steady_clock::now();
  double caching_duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1 - t_0).count();
  ;

  return cache;
}

bool LoopDetector::verify_loop(ScanMatchResult cand)
{
  int const ICP_INLIER_TRES = 200;
  num_loops++;

  if (cand.icp_result.converged && cand.kiss_converged && cand.icp_result.num_inliers >= ICP_INLIER_TRES)
  {
    num_loops_accepted++;
    if (cand.core == idx(NEAR))
      num_near++;
    else
      num_far++;

    if (verbose)
    {
      LOGI_VERIFY_OK(this->get_logger(), "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––");
      LOGI_VERIFY_OK(this->get_logger(), "___________ :.:.:[%s]:.:.:", (cand.core == idx(NEAR)) ? "NEAR" : "FAR");
      LOGI_VERIFY_OK(this->get_logger(), "___________ VERIFIED LOOP CANDIDATE <%lu>---<%lu>", cand.target_key,
                     cand.source_key);
      LOGI_VERIFY_OK(this->get_logger(), "___________ [kiss]––––––––––inlier-----------: %ld >= %d", cand.kiss_inliers,
                     KISS_INLIER_TRES);
      LOGI_VERIFY_OK(this->get_logger(), "___________ [small_gicp]––––inlier------------: %ld >= %d",
                     cand.icp_result.num_inliers, ICP_INLIER_TRES);
      LOGI_VERIFY_OK(this->get_logger(), "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––");
      LOGI_VERIFY_OK(this->get_logger(), "___________ SC      : %.4f units", cand.sc_sim_score);
      LOGI_VERIFY_OK(this->get_logger(), "___________ RMSE    : %.4f <= %.4f", cand.rmse, valid_max_rmse);
      LOGI_VERIFY_OK(this->get_logger(), "___________ Runtime %.4f ms", cand.total_duration_ms);
      LOGI_VERIFY_OK(this->get_logger(), "___________ ––– fetch: %.4f ms", cand.fetch_duration_ms);
      LOGI_VERIFY_OK(this->get_logger(), "___________ ––– rmse : %.4f ms", cand.rmse_duration_ms);
      LOGI_VERIFY_OK(this->get_logger(), "___________ ––– sc   : %.4f ms", cand.sc_duration_ms);
      LOGI_VERIFY_OK(this->get_logger(), "___________ ––– kiss : %.4f ms", cand.kiss_duration_ms);
      LOGI_VERIFY_OK(this->get_logger(), "___________ ––– icp  : %.4f ms", cand.icp_duration_ms);
      LOGI_VERIFY_OK(this->get_logger(), "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––");
    }

    // publish loop edge
    pogolm_interfaces::msg::LoopEdge loop_edge;
    loop_edge.source_key = x_(cand.source_key);
    loop_edge.target_key = x_(cand.target_key);
    // loop_edge.tranformation = to_array(cand.icp_result.T_target_source.inverse());
    loop_edge.tranformation = to_array(cand.icp_result.T_target_source.inverse());
    loop_edge.hessian = to_array(cand.icp_result.H);
    loop_edge.error = cand.icp_result.error;
    loop_edge.inliers = cand.icp_result.num_inliers;
    loop_edge.core = cand.core;
    loop_edge.kiss_inliers = cand.kiss_inliers;
    loop_edge.sc_dist = cand.sc_sim_score;
    loop_edge_publisher->publish(loop_edge);
    return true;
  }
  else
  {
    if (verbose)
    {
      LOGW_VERIFY_NO(this->get_logger(), "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––");
      LOGW_VERIFY_NO(this->get_logger(), "___________ :.:.:[%s]:.:.:", (cand.core == idx(NEAR)) ? "NEAR" : "FAR");
      LOGW_VERIFY_NO(this->get_logger(), "___________ OMITTED LOOP CANDIDATE <%lu>---<%lu>", cand.target_key,
                     cand.source_key);
      LOGW_VERIFY_NO(this->get_logger(), "___________ [kiss]––––––––––inlier-----------: %ld <= %d", cand.kiss_inliers,
                     KISS_INLIER_TRES);
      LOGW_VERIFY_NO(this->get_logger(), "___________ [small_gicp]––––inlier------------: %ld <= %d",
                     cand.icp_result.num_inliers, ICP_INLIER_TRES);
      LOGW_VERIFY_NO(this->get_logger(), "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––");
      LOGW_VERIFY_NO(this->get_logger(), "___________ SC      : %.4f units", cand.sc_sim_score);
      LOGW_VERIFY_NO(this->get_logger(), "___________ RMSE    : %.4f <= %.4f", cand.rmse, valid_max_rmse);
      LOGW_VERIFY_NO(this->get_logger(), "___________ Runtime %.4f ms", cand.total_duration_ms);
      LOGW_VERIFY_NO(this->get_logger(), "___________ ––– fetch: %.4f ms", cand.fetch_duration_ms);
      LOGW_VERIFY_NO(this->get_logger(), "___________ ––– rmse : %.4f ms", cand.rmse_duration_ms);
      LOGW_VERIFY_NO(this->get_logger(), "___________ ––– sc   : %.4f ms", cand.sc_duration_ms);
      LOGW_VERIFY_NO(this->get_logger(), "___________ ––– kiss : %.4f ms", cand.kiss_duration_ms);
      LOGW_VERIFY_NO(this->get_logger(), "___________ ––– icp  : %.4f ms", cand.icp_duration_ms);
      LOGW_VERIFY_NO(this->get_logger(), "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––");
    }

    return false;
  }
}

void LoopDetector::publish_loop_debug(const uint64_t source_key, const std::vector<DescriptorMatchResult>& desc_matches,
                                      const std::vector<ScanMatchResult>& scan_matches)
{
  if (!loop_debug_publisher)
    return;

  pogolm_interfaces::msg::LoopDebug msg;
  msg.header.stamp = this->now();
  msg.source_key = source_key;
  msg.loop_radius = 0.0f;

  double min_dist = std::numeric_limits<double>::infinity();
  for (const auto& m : desc_matches)
  {
    if (m.source_key != source_key)
      continue;
    min_dist = std::min(min_dist, m.dist);
  }

  if (std::isfinite(min_dist))
  {
    const double eps = 1e-12;
    msg.descriptor_matches.reserve(desc_matches.size());

    for (const auto& m : desc_matches)
    {
      if (m.source_key != source_key)
        continue;

      pogolm_interfaces::msg::LoopDescriptorMatch dm;
      dm.src_key = m.source_key;
      dm.dst_key = m.target_key;
      dm.aligned_dist = static_cast<float>(m.dist);
      dm.is_min = (std::abs(m.dist - min_dist) <= eps);

      msg.descriptor_matches.push_back(dm);
    }
  }

  msg.scan_matches.reserve(scan_matches.size());
  for (const auto& s : scan_matches)
  {
    if (s.source_key != source_key)
      continue;

    pogolm_interfaces::msg::LoopScanMatch sm;
    sm.src_key = s.source_key;
    sm.dst_key = s.target_key;
    sm.translation_norm = static_cast<float>(s.translation_norm);

    // small_gicp::RegistrationResult in ScanMatchResult
    sm.icp_error = static_cast<float>(s.icp_result.error);
    sm.inliers = static_cast<int64_t>(s.icp_result.num_inliers);
    sm.iterations = static_cast<int32_t>(s.icp_result.iterations);
    sm.converged = static_cast<bool>(s.icp_result.converged);

    msg.scan_matches.push_back(sm);
  }

  loop_debug_publisher->publish(msg);
}

void LoopDetector::publish_pose_refresh_loop_debug()
{
  pogolm_interfaces::msg::LoopDebug msg;
  msg.header.stamp = this->now();
  msg.source_key = last_loop_checked_key.load(std::memory_order_acquire);
  msg.loop_radius = static_cast<float>(1000.);

  loop_debug_publisher->publish(msg);
}

bool LoopDetector::pointcloud_exists(uint64_t key)
{
  return pc_storage->contains(key);
}

LoopDetectorParams LoopDetector::get_default_params()
{
  LoopDetectorParams params;

  params.min_loop_length = 20;
  params.sc_dist_th = 0.12;
  params.log_path = "/tmp/pogolm/loop_detector.txt";
  params.verbose = false;
  return params;
}

LoopDetectorParams LoopDetector::get_yaml_params()
{
  LoopDetectorParams params;
  params.sc_dist_th = declare_parameter("sc_dist_th", 0.2);
  params.min_loop_length = declare_parameter("min_loop_length", 10);
  params.log_path = declare_parameter("log_path", "/tmp/pogolm/loop_detector.txt");
  verbose = declare_parameter("verbose", false);
  return params;
}

void LoopDetector::init_node(const LoopDetectorParams& params)
{
  pc_storage = StoreRegistry::get("/tmp/pc_store.bin");
  vm_storage =
      std::make_shared<disk::DiskVoxelmapCacheStore>("/tmp/vm_store.bin", 1, /*truncate*/ true, /*flush*/ false);

  verbose = params.verbose;
  log_path = params.log_path;
  min_loop_length = params.min_loop_length;
  SC_DIST_TRES = params.sc_dist_th;

  LOGI_POGOLM(this->get_logger(), "verbose----------------: %s", (verbose) ? "true" : "false");
  LOGI_POGOLM(this->get_logger(), "min loop length--------: %d", params.min_loop_length);
  LOGI_POGOLM(this->get_logger(), "real time limit--------: %.4f", RT_LIMIT_MS);
  LOGI_POGOLM(this->get_logger(), "logging path-----------: %s", log_path.c_str());
  LOGI_POGOLM(this->get_logger(), "core job limit---------: %ld", JOB_LIMIT);
  LOGI_POGOLM(this->get_logger(), "[small_gicp]");
  LOGI_POGOLM(this->get_logger(), "––– downsample res-----: %.4f", DS_RES);
  LOGI_POGOLM(this->get_logger(), "––– voxel res----------: %.4f", VX_RES);
  LOGI_POGOLM(this->get_logger(), "––– thread number------: %d", NUM_TH);
  LOGI_POGOLM(this->get_logger(), "––– neighbor number----: %d", NUM_N);
  LOGI_POGOLM(this->get_logger(), "––– max corres dist----: %.4f", MAX_C_DIST);
  LOGI_POGOLM(this->get_logger(), "––– max iter-----------: %d", MAX_ITER);
  LOGI_POGOLM(this->get_logger(), "––– eps transl.--------: %.4f", EPS_T);
  LOGI_POGOLM(this->get_logger(), "––– eps rot.-----------: %.4f", EPS_R);
  LOGI_POGOLM(this->get_logger(), "[kiss_matcher]");
  LOGI_POGOLM(this->get_logger(), "––– downsample res-----: %.4f", KISS_RES);
  LOGI_POGOLM(this->get_logger(), "––– inlier threshold---: %d", KISS_INLIER_TRES);
  LOGI_POGOLM(this->get_logger(), "––– use quatro---------: %s", (KISS_USE_QUATRO) ? "true" : "false");
  LOGI_POGOLM(this->get_logger(), "––– use test-----------: %s", (KISS_USE_TEST) ? "true" : "false");
  LOGI_POGOLM(this->get_logger(), "[scan_context]");
  LOGI_POGOLM(this->get_logger(), "–– downsample res------: %.4f", SC_D_RES);
  LOGI_POGOLM(this->get_logger(), "–– dist threshold------: %.4f", params.sc_dist_th);
  LOGI_POGOLM(this->get_logger(), "–– search ratio--------: %.4f", SEARCH_RATIO);
  LOGI_POGOLM(this->get_logger(), "–– lidar height--------: %.4f", LIDAR_H);
  LOGI_POGOLM(this->get_logger(), "–– rings---------------: %d", NUM_RING);
  LOGI_POGOLM(this->get_logger(), "–– sectors-------------: %d", NUM_SECT);
  LOGI_POGOLM(this->get_logger(), "–– exclude recent------: %d", EXCLUDE_RECENT);
  LOGI_POGOLM(this->get_logger(), "–– candidate number----: %d", NUM_CAND);
  LOGI_POGOLM(this->get_logger(), "–– tree making period--: %.4f", TREE_MAKING_P);
  LOGI_POGOLM(this->get_logger(), "[Core Policies]");

  size_t begin = 0;
  for (size_t i = 0; i < core_policies.size(); i++)
  {
    LOGI_POGOLM(this->get_logger(),
                "–– [%ld,%ld] N (lbn:%d, s:%d, sc:%.2f, g:%.1f) F (lbn:%d, s:%d, sc:%.2f, g:%.1f) %s", begin,
                core_policies[i].thres, core_policies[i].near.look_back_n, core_policies[i].near.stride,
                core_policies[i].near.sc_sim_score, core_policies[i].near.gamma, core_policies[i].far.look_back_n,
                core_policies[i].far.stride, core_policies[i].far.sc_sim_score, core_policies[i].far.gamma,
                (core_policies[i].enable_far) ? "true" : "false");

    begin = core_policies[i].thres;
  }

  // detector cores
  auto& near = cores[idx(NEAR)];
  near.type = NEAR;
  near.params = get_starting_core_params(near.type);
  near.enabled.store(true);
  auto& far = cores[idx(FAR)];
  far.type = FAR;
  far.params = get_starting_core_params(far.type);
  far.enabled.store(false);  // disable the far core

  cores[idx(NEAR)].thd = std::thread(&LoopDetector::core_worker, this, std::ref(cores[idx(NEAR)]));
  cores[idx(FAR)].thd = std::thread(&LoopDetector::core_worker, this, std::ref(cores[idx(FAR)]));

  // scancontext++
  sc_manager = std::make_unique<SC2::SCManager>(LIDAR_H, NUM_RING, NUM_SECT, MAX_RAD, EXCLUDE_RECENT, NUM_CAND,
                                                SEARCH_RATIO, SC_DIST_TRES, TREE_MAKING_P, SC_D_RES);

  // KISS-matcher
  kiss_config = std::make_unique<kiss_matcher::KISSMatcherConfig>(static_cast<float>(KISS_RES));
  kiss_config->use_quatro_ = KISS_USE_QUATRO;
  kiss_config->use_ratio_test_ = KISS_USE_TEST;
  kiss_config->use_voxel_sampling_ = false;

  last_key = 0;
  last_loop_checked_key = 0;

  // Initialize the SUB/PUB/TIMER
  io_group = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  rclcpp::SubscriptionOptions io_opt;
  io_opt.callback_group = io_group;

  pose_graph_subscriber = this->create_subscription<factor_graph_interfaces::msg::FactorGraph>(
      "/pogolm/pose_graph", 10, std::bind(&LoopDetector::callback_pose_graph, this, std::placeholders::_1), io_opt);

  pc_key_subscriber = this->create_subscription<pogolm_interfaces::msg::Key>(
      "/pogolm/pointcloud_key", 10, std::bind(&LoopDetector::callback_pc_key, this, std::placeholders::_1), io_opt);

  loop_edge_publisher = this->create_publisher<pogolm_interfaces::msg::LoopEdge>("/pogolm/loop_edge", 10);

  clean_timer = this->create_wall_timer(std::chrono::seconds(10), std::bind(&LoopDetector::callback_cleaner, this));

  loop_debug_publisher = this->create_publisher<pogolm_interfaces::msg::LoopDebug>(
      "loop_debug", rclcpp::QoS(10).best_effort().durability_volatile());
}

LoopDetector::LoopDetector(const LoopDetectorParams& params, const rclcpp::NodeOptions& options)
  : Node("loop_detector", options)
{
  RCLCPP_INFO(this->get_logger(), "IPC: %s", options.use_intra_process_comms() ? "ON" : "OFF");

  init_node(params);
}

LoopDetector::LoopDetector(const rclcpp::NodeOptions& options) : Node("loop_detector", options)
{
  RCLCPP_INFO(this->get_logger(), "IPC: %s", options.use_intra_process_comms() ? "ON" : "OFF");

  auto params = get_yaml_params();
  init_node(params);
}

LoopDetector::~LoopDetector()
{
  for (auto& core : cores)
  {
    core.running.store(false, std::memory_order_release);
    core.cv.notify_all();
  }
  for (auto& core : cores)
  {
    if (core.thd.joinable())
      core.thd.join();
  }

  struct Stats
  {
    std::size_t n{ 0 };
    double min{ std::numeric_limits<double>::quiet_NaN() };
    double max{ std::numeric_limits<double>::quiet_NaN() };
    double mean{ std::numeric_limits<double>::quiet_NaN() };
    double median{ std::numeric_limits<double>::quiet_NaN() };
    double stddev{ std::numeric_limits<double>::quiet_NaN() };
    double p90{ std::numeric_limits<double>::quiet_NaN() };
    double p95{ std::numeric_limits<double>::quiet_NaN() };
    double p99{ std::numeric_limits<double>::quiet_NaN() };
  };

  auto percentile_sorted = [](const std::vector<double>& v_sorted, double p) -> double {
    if (v_sorted.empty())
      return std::numeric_limits<double>::quiet_NaN();

    const double x = p * (static_cast<double>(v_sorted.size() - 1));
    const std::size_t i = static_cast<std::size_t>(std::floor(x));
    const std::size_t j = static_cast<std::size_t>(std::ceil(x));
    if (i == j)
      return v_sorted[i];
    const double t = x - static_cast<double>(i);
    return (1.0 - t) * v_sorted[i] + t * v_sorted[j];
  };

  auto compute_stats = [&](const std::vector<double>& v) -> Stats {
    Stats s;
    s.n = v.size();
    if (v.empty())
      return s;

    // min/max/mean
    auto [mn_it, mx_it] = std::minmax_element(v.begin(), v.end());
    s.min = *mn_it;
    s.max = *mx_it;

    const double sum = std::accumulate(v.begin(), v.end(), 0.0);
    s.mean = sum / static_cast<double>(v.size());

    // median + percentiles
    std::vector<double> tmp = v;
    std::sort(tmp.begin(), tmp.end());

    s.median = percentile_sorted(tmp, 0.50);
    s.p90 = percentile_sorted(tmp, 0.90);
    s.p95 = percentile_sorted(tmp, 0.95);
    s.p99 = percentile_sorted(tmp, 0.99);

    // stddev
    double acc = 0.0;
    for (double x : v)
    {
      const double d = x - s.mean;
      acc += d * d;
    }
    s.stddev = std::sqrt(acc / static_cast<double>(v.size()));
    return s;
  };

  auto fmt_stats = [](const Stats& s) -> std::string {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(4);

    oss << "n=" << s.n;
    if (s.n == 0)
      return oss.str();

    oss << " | min=" << s.min << " max=" << s.max << " mean=" << s.mean << " median=" << s.median << " std=" << s.stddev
        << " p90=" << s.p90 << " p95=" << s.p95 << " p99=" << s.p99;
    return oss.str();
  };

  std::vector<double> kiss_ms, icp_ms, fetch_ms, total_ms, rmse_ms;
  kiss_ms.reserve(durs.size());
  icp_ms.reserve(durs.size());
  fetch_ms.reserve(durs.size());
  total_ms.reserve(durs.size());
  rmse_ms.reserve(durs.size());

  for (const auto& d : durs)
  {
    kiss_ms.push_back(d.kiss_duration_ms);
    icp_ms.push_back(d.icp_duration_ms);
    fetch_ms.push_back(d.fetch_duration_ms);
    total_ms.push_back(d.total_duration_ms);
  }

  const Stats st_kiss = compute_stats(kiss_ms);
  const Stats st_icp = compute_stats(icp_ms);
  const Stats st_fetch = compute_stats(fetch_ms);
  const Stats st_total = compute_stats(total_ms);
  const Stats st_dm_pipeline = compute_stats(dm_durs);
  const Stats st_total_work = compute_stats(total_durs);

  std::vector<double> overhead_ms;
  if (!dm_durs.empty() && !total_durs.empty())
  {
    const std::size_t n = std::min(dm_durs.size(), total_durs.size());
    overhead_ms.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
      overhead_ms.push_back(total_durs[i] - dm_durs[i]);
    }
  }
  const Stats st_overhead = compute_stats(overhead_ms);

  std::vector<double> dm_share_pct;
  if (!dm_durs.empty() && !total_durs.empty())
  {
    const std::size_t n = std::min(dm_durs.size(), total_durs.size());
    dm_share_pct.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
      const double tw = total_durs[i];
      if (std::abs(tw) < 1e-12)
        continue;
      dm_share_pct.push_back(100.0 * dm_durs[i] / tw);
    }
  }
  const Stats st_dm_share = compute_stats(dm_share_pct);

  LineWriter w(log_path, LineWriter::Mode::Append);

  w.writeLine("============================================================");
  w.writeLine("[LoopDetector] Runtime Statistics");
  w.writeLineParts("log_path: ", log_path);
  w.writeLine("------------------------------------------------------------");
  w.writeLineParts("total loops: ", num_loops);
  w.writeLineParts("total verified loops : ", num_loops_accepted, " (N: ", num_near, ", F: ", num_far, ")");
  w.writeLineParts("total omitted loops : ", num_loops - num_loops_accepted);
  w.writeLine("------------------------------------------------------------");
  w.writeLine("[durs] (ScanMatchResult-intern, pro Kandidat)");
  w.writeLineParts("  total_duration_ms : ", fmt_stats(st_total));
  w.writeLineParts("  fetch_duration_ms : ", fmt_stats(st_fetch));
  w.writeLineParts("  kiss_duration_ms  : ", fmt_stats(st_kiss));
  w.writeLineParts("  icp_duration_ms   : ", fmt_stats(st_icp));
  w.writeLine("------------------------------------------------------------");
  w.writeLine("[pipeline]");
  w.writeLineParts("  dm_pipeline_duration_ms : ", fmt_stats(st_dm_pipeline));
  w.writeLineParts("  total_worked_duration_ms: ", fmt_stats(st_total_work));
  w.writeLine("------------------------------------------------------------");
  w.writeLine("[derived]");
  w.writeLineParts("  overhead_ms (= total_worked - dm_pipeline) : ", fmt_stats(st_overhead));
  w.writeLineParts("  dm_share_pct (= 100*dm_pipeline/total_worked): ", fmt_stats(st_dm_share));
  w.writeLine("------------------------------------------------------------");
  w.writeLineParts("sizes: durs=", durs.size(), " dm_durs=", dm_durs.size(), " total_durs=", total_durs.size());
  w.writeLine("============================================================");
  w.flush();
}

}  // namespace loop_detector_np

#ifndef PGO_WITH_VISUAL_NO_MAIN
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<loop_detector_np::LoopDetector>();

  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
#endif