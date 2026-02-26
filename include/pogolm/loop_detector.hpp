// #include <functional>
#include <cctype>
#include <cstdio>
#include <map>
#include <memory>
#include <vector>
#include <mutex>
#include <thread>
#include <queue>
#include <array>
#include <malloc.h>

#include "factor_graph_interfaces/msg/factor_graph.hpp"
#include "factor_graph_interfaces/srv/get_point_cloud.hpp"

#include "pogolm_interfaces/msg/icp_result.hpp"
#include "pogolm_interfaces/msg/loop_edge.hpp"
#include "pogolm_interfaces/msg/key.hpp"
#include "pogolm_interfaces/srv/get_map.hpp"
#include <pogolm_interfaces/msg/loop_debug.hpp>
#include <pogolm_interfaces/msg/loop_pose.hpp>
#include <pogolm_interfaces/msg/loop_descriptor_match.hpp>
#include <pogolm_interfaces/msg/loop_scan_match.hpp>

#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/header.hpp>

#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/benchmark/read_points.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/factors/plane_icp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

#include <pcl/kdtree/kdtree_flann.h>

#include "scancontext/Scancontext.h"

#include <kiss_matcher/KISSMatcher.hpp>
#include <kiss_matcher/points/downsampling.hpp>

#define LOG_NODE_TAG NODE_LOOP_DETECTOR

#include "pogolm/utils.hpp"
#include "pogolm/logging.hpp"


namespace loop_detector_np {

using LoopEdgeCandidate = struct LoopEdgeCandidate;
using DetectorJob = struct DetectorJob;

// duration breakdown for one loop detection run
struct LDPDuration {
    double kiss_duration_ms;
    double icp_duration_ms;
    double fetch_duration_ms;
    double total_duration_ms;
};

struct LoopDetectorParams {
    int min_loop_length{10};
    double sc_dist_th{0.3};
    bool verbose{false};
    std::string log_path{"/tmp/pogolm/loop_detector.txt"};
};


enum CoreType {
    FAR,
    NEAR,
};

// parameters controlling candidate selection for one core
struct CoreParams {

    int look_back_n; 
    uint8_t stride;
    double sc_sim_score;
    double gamma = 0.6;

    CoreParams() = default;
    CoreParams(int lbn, uint8_t s, double sc) : look_back_n(lbn), stride(s), sc_sim_score(sc) {}

    bool is_equal(const CoreParams& other) const {
        double eps = 1e-10;
        return this->look_back_n==other.look_back_n &&
               this->stride==other.stride &&
               std::abs(this->sc_sim_score-other.sc_sim_score) < eps &&
               std::abs(gamma-other.gamma)<eps;
    }

    // first key to consider for this core
    uint64_t start(const uint64_t source_key, const CoreType type) const {

        if(type==NEAR) {
            return std::max((uint64_t) STARTING_POSE_KEY, source_key-1);
        } else {
            return (source_key > look_back_n)? source_key-look_back_n-1 : STARTING_POSE_KEY;
        }
    }

    // last key to consider for this core
    uint64_t end(const uint64_t source_key, const CoreType type) const {
        
        if (look_back_n == -1)
            return STARTING_POSE_KEY;

        if(type==NEAR) {
            return (source_key > (uint64_t)look_back_n)? source_key-look_back_n : STARTING_POSE_KEY;
        } else {
            return STARTING_POSE_KEY;
        }
    }

    // acceptance probability for visiting a target key (used for FAR core)
    bool can_visit(const uint64_t source_key, const uint64_t target_key, const CoreType type) {

        if (type==NEAR) return true;

        double s = (double) start(source_key,type);
        double e = (double) end(source_key,type);
        double x = (double) (target_key-e);

        double delta = std::abs(s - e);

        if (delta <= 1e-9) return true;
        
        // double upper_bound = std::exp((x - delta) / (gamma * std::sqrt(delta/2)));        
        double upper_bound = std::exp(std::log(gamma)*(1-(x/delta)));        
        
        return rand_double() <= upper_bound;
    }

};

// policy for switching between core parameter sets
struct CorePolicy {
    size_t thres;
    CoreParams near;
    CoreParams far;
    bool enable_far;
};

// runtime state for one core thread
struct CoreContext {
    CoreType type;
    CoreParams params;

    std::mutex mtx;
    std::condition_variable cv;
    std::deque<DetectorJob> jobs;
    std::atomic_bool running {true};
    std::atomic_bool enabled {false};
    std::thread thd;

    CoreContext() = default;

    std::string name() {
        return (type==FAR)? "far_core" : "near_core";
    }
    
};

// job passed to a core worker
struct DetectorJob {
    uint64_t key_to_detect;
};


// loop detector node (ScanContext + KISS + ICP)
class LoopDetector : public rclcpp::Node {
    
  private:

    // callback groups
    rclcpp::CallbackGroup::SharedPtr io_group;

	// publishers
	rclcpp::Publisher<pogolm_interfaces::msg::LoopEdge>::SharedPtr
		loop_edge_publisher;
    rclcpp::Publisher<pogolm_interfaces::msg::LoopDebug>::SharedPtr 
        loop_debug_publisher;
	// subscriptions
	rclcpp::Subscription<factor_graph_interfaces::msg::FactorGraph>::SharedPtr
		pose_graph_subscriber;
    rclcpp::Subscription<pogolm_interfaces::msg::Key>::SharedPtr
        pc_key_subscriber;
    // periodic cleanup
    rclcpp::TimerBase::SharedPtr 
        clean_timer;

    // mutex / thread
    std::atomic<uint64_t>
        last_key{0},
        last_pointcloud_inserted{0},
        last_loop_checked_key{0};
    std::mutex
        pose_mutex,
        pc_mutex,
        sc_mutex,
        cache_mutex;

    // max queued jobs per core
    const size_t JOB_LIMIT = 3;

    // disk stores for point clouds and voxelmap caches
    std::shared_ptr<disk::DiskCloudStore> pc_storage;
    std::shared_ptr<disk::DiskVoxelmapCacheStore> vm_storage;
	
    size_t pose_graph_size{0};

    // worker cores (near/far)
    std::array<CoreContext, 2> cores{};

    // policy table for different map sizes (Velodyne setup)
    std::array<CorePolicy,4> core_policies = {{
        {100,    CoreParams{ -1,1,0.4}, CoreParams{-1,1,0.5},   false},
        {1000,  CoreParams{100,1,0.4}, CoreParams{100,2,0.5},     true},
        {3000,  CoreParams{300,3,0.4}, CoreParams{300,5,0.5},     true},
        {10000, CoreParams{300,3,0.4}, CoreParams{300,10,0.5},  true}
    }};

    // // policy table for different map sizes (Ouster setup)
    // std::array<CorePolicy,4> core_policies = {{
    //     {100,    CoreParams{ -1,1,0.1}, CoreParams{-1,1,0.12},   false},
    //     {1000,  CoreParams{100,1,0.1}, CoreParams{100,2,0.12},     true},
    //     {3000,  CoreParams{300,3,0.1}, CoreParams{300,5,0.12},     true},
    //     {10000, CoreParams{300,3,0.1}, CoreParams{300,10,0.12},  true}
    // }};

    std::size_t current_policy;

    // loop closure
    double valid_max_rmse {0.3}; 
	uint16_t min_loop_length;
    std::string log_path{"/tmp/pogolm/loop_detector.txt"};


	// small_gicp parameters
    const int NUM_TH = 2;                       // number of threads in small_gicp
    const int NUM_N = 10;                       // number of neighbors for kNN-search
    const double DS_RES = 0.25;                 // downsample resolution 
    const double VX_RES = 0.1;                  // voxelmap resolution (only for VGICP)
    const double MAX_C_DIST = 5.0 * DS_RES;     // maximum correspodence distance
    const int MAX_ITER = 64;                    // maximum iterations
    const double EPS_T = 1e-3;                  // translational epsilon (convergance criterion)
    const double EPS_R = 0.001745;              // rotational epsilon (convergance criterion)

	// kiss_matcher parameters
	const double KISS_RES = 0.8;                // voxel resolution for KISS-Matcher
	const bool KISS_USE_QUATRO = true;          // yaw-dominant mode when true
	const bool KISS_USE_TEST = false;           // optional prefilter on correspondences
	const int KISS_INLIER_TRES = 30;            // minimum inliers for valid registration

   std::unique_ptr<kiss_matcher::KISSMatcherConfig> kiss_config;

    // scancontext++ parameters
    const double LIDAR_H = 0.0;                 // set if scans are in lidar frame with offset
    const int NUM_RING = 20;                    // number of rings
    const int NUM_SECT = 60;                    // number of sectors
    const double MAX_RAD = 200.0;               // max radius used for descriptor
    const int EXCLUDE_RECENT = 0;               // skip recent frames
    const int NUM_CAND = 10;                    // max candidates from SC
    const double SEARCH_RATIO = 0.8;            // fraction of bins searched
    double SC_DIST_TRES {0.15};                 // SC distance threshold
    const double TREE_MAKING_P = 100.;          // tree rebuild period
    const double SC_D_RES = 0.2;                // SC distance bin size

    std::unique_ptr<SC2::SCManager>
        sc_manager;
          

    bool verbose{false};
    const double RT_LIMIT_MS {150.};             // hard limit per detection cycle           

    // DEBUG-Stuff
    std::vector<LDPDuration> durs;
    std::vector<DescriptorMatchResult> dms;
    std::vector<double> dm_durs;
    std::vector<double> total_durs;
    int num_loops=0, num_loops_accepted=0;
    int num_far=0, num_near=0;
    

    // factor graph callback, updates pose graph state
    void callback_pose_graph(factor_graph_interfaces::msg::FactorGraph::ConstSharedPtr msg);
    // key callback when a new keyframe point cloud is stored
    void callback_pc_key(pogolm_interfaces::msg::Key::UniquePtr msg);
    // periodic dispatch of loop detection jobs
    void callback_loop_detect_timer();
    // periodic memory cleanup
    void callback_cleaner();

    // check if point cloud exists on disk
    bool pointcloud_exists(uint64_t key);

    // load voxelmap cache from disk if available
    std::optional<std::shared_ptr<disk::VoxelmapCache>> fetch_voxelmap(uint64_t key);
    // build voxelmap cache from point cloud and store it
    std::shared_ptr<disk::VoxelmapCache> build_voxelmap(const uint64_t key);

    // initial core params for current map size
    CoreParams get_starting_core_params(CoreType type);
    // map CoreType to core array index
    int idx(CoreType type);
    // core thread body
    void core_worker(CoreContext& core);
    // switch policy based on map size
    void update_core_policy(const size_t n);
    // drop jobs if queue is too large
    void free_core_jobs(std::deque<DetectorJob>& job_queue);

    // scancontext descriptor matching for a given source/target pair
    DescriptorMatchResult
    descriptor_matching(const uint64_t target_key,
                        const uint64_t source_key);

    // scan matching with initial guess
    ScanMatchResult
    scan_matching(const Eigen::Isometry3d& init_guess,
                  const uint64_t target_key,
                  const uint64_t source_key);

    // scan matching using descriptor match result
    ScanMatchResult
    scan_matching(const DescriptorMatchResult& dm_result);

    // run VICP alignment
    void perform_icp_alignment(Eigen::Isometry3d& initial_guess,
                               small_gicp::RegistrationResult& icp_reg_res,
                               const disk::VoxelmapCache& target,
                               const disk::VoxelmapCache& source);

    // compute initial guess using KISS matcher
    void kiss_initial_guess(Eigen::Isometry3d& init_guess,
                            long& kiss_inliers,
                            const disk::VoxelmapCache& target,
                            const disk::VoxelmapCache& source);

    // final gating for loop acceptance
    bool verify_loop(ScanMatchResult cand);

    // publish debug info for one detection step
    void publish_loop_debug(const uint64_t source_key,
                            const std::vector<DescriptorMatchResult>& desc_matches,
                            const std::vector<ScanMatchResult>& scan_matches);

    // publish debug refresh without new loop
    void publish_pose_refresh_loop_debug();

    // read parameters from ROS params
    LoopDetectorParams get_yaml_params();
    // init pubs/subs/threads
    void init_node(const LoopDetectorParams& params);

  public:
 
    LoopDetector(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    
    LoopDetector(const LoopDetectorParams& params,
                 const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
                 
    // stop threads, flush logs
    ~LoopDetector();

    // default settings for standalone
    LoopDetectorParams get_default_params();
};

// component wrapper for rclcpp components
class LoopDetectorComponent : public LoopDetector {

public:

    LoopDetectorComponent(const rclcpp::NodeOptions& options)
    : LoopDetector(options) {}

    LoopDetectorComponent(
        const LoopDetectorParams& params,
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : LoopDetector(params, options) {}

};

} // namespace loop_detector_np