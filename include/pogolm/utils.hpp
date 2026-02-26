#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <malloc.h>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>

#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration_result.hpp>

#include "ikd-Tree/ikd_Tree.h"

#include <geometry_msgs/msg/pose.h>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "pogolm/disk_storage.hpp"

#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#define STARTING_POSE_KEY 1

struct DescriptorMatchResult {
	uint64_t source_key;
	uint64_t target_key;
	Eigen::Isometry3d guess;
	double dist;
	double duration_ms;
};

struct ScanMatchResult {
	small_gicp::RegistrationResult icp_result;
	uint64_t source_key;
	uint64_t target_key;
	bool kiss_converged;
	long kiss_inliers;
	double translation_norm;
	double sc_duration_ms;
	double kiss_duration_ms;
	double icp_duration_ms;
	double total_duration_ms;
	double rmse_duration_ms;
	double fetch_duration_ms;
	double rmse;
	double sc_sim_score;
	int core;
};

struct LoopEdgeCandidate {
	gtsam::noiseModel::Base::shared_ptr noise;
	Eigen::Matrix<double, 6, 6> hessian;
	Eigen::Isometry3d transformation;
	uint64_t source_key;
	uint64_t target_key;
	long inliers;
	long kiss_inliers;
	float sc_dist;
	double error;
	float trans_err = -1.0f;
	float yaw_err_deg = -1.0f;
	double rmse = std::numeric_limits<double>::quiet_NaN();
};

struct Landmark {
	struct LMPoint point;
	std::unordered_set<uint64_t> anchor_list;
	uint64_t key;
	std::string label;

	Landmark() = default;
	Landmark(uint64_t _key, float _x, float _y, float _z, std::string _l)
		: point(_x, _y, _z), key(_key), label(_l) {}
};

struct SearchResult {
	std::map<uint64_t, float> dists;
};

struct MergedInfo {
	uint64_t new_merged_key;
	std::vector<uint64_t> merged_keys;
};

// random number in [0,1]
inline double rand_double() {
	static thread_local std::mt19937_64 rng(std::random_device{}());
	static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);

	double x;
	do {
		x = dist(rng);
	} while (x < 0.0 || x > 1.0);
	return x;
}

// euclidean distance for KD points
inline double euclidean_distance_ikd(const LMPoint &p, const LMPoint &q) {
	LMPoint temp{p.x - q.x, p.y - q.y, p.z - q.z, 0};
	return std::sqrt(temp.x * temp.x + temp.y * temp.y + temp.z * temp.z);
}

// Eigen Isometry to GTSAM Pose3
inline gtsam::Pose3 to_pose3(const Eigen::Isometry3d &iso) {
	const gtsam::Rot3 R(iso.linear());
	const gtsam::Point3 t(iso.translation().x(), iso.translation().y(),
						  iso.translation().z());
	return gtsam::Pose3(R, t);
}

// ROS Pose to GTSAM Pose3
inline gtsam::Pose3 to_gtsam(const geometry_msgs::msg::Pose &pose) {
	return gtsam::Pose3(
		gtsam::Rot3::Quaternion(pose.orientation.w, pose.orientation.x,
								pose.orientation.y, pose.orientation.z),
		gtsam::Point3(pose.position.x, pose.position.y, pose.position.z));
}

// ROS PointCloud2 to Eigen Vec3f vector
inline std::vector<Eigen::Vector3f>
pc2_to_vec3f_fast(const sensor_msgs::msg::PointCloud2 &pc) {
	std::vector<Eigen::Vector3f> out;
	out.reserve(pc.width * pc.height);

	sensor_msgs::PointCloud2ConstIterator<float> ix(pc, "x");
	sensor_msgs::PointCloud2ConstIterator<float> iy(pc, "y");
	sensor_msgs::PointCloud2ConstIterator<float> iz(pc, "z");

	for (; ix != ix.end(); ++ix, ++iy, ++iz) {
		const float x = *ix, y = *iy, z = *iz;
		if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z))
			out.emplace_back(x, y, z);
	}
	return out;
}

// Eigen Isometry to row-major array
inline std::array<double, 16> to_array(const Eigen::Isometry3d &T) {
	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> M = T.matrix();
	std::array<double, 16> out;
	std::memcpy(out.data(), M.data(), 16 * sizeof(double));
	return out;
}

// Eigen Matrix (6x6) to row-major to array
inline std::array<double, 36> to_array(const Eigen::Matrix<double, 6, 6> &mat) {
	std::array<double, 36> arr;
	std::copy(mat.data(), mat.data() + 36, arr.begin());
	return arr;
}

// invert hessian into covariance, fallback if invalid
inline bool
calculate_cov_from_hessian(const Eigen::Matrix<double, 6, 6> &hessian,
						   const Eigen::Matrix<double, 6, 6> &fallback,
						   Eigen::Matrix<double, 6, 6> &cov) {
	cov = fallback;
	Eigen::Matrix<double, 6, 6> hess = hessian;

	// if hessian matrix is invalid or det(hessian)=0 (not invertible): fallback
	if (!hessian.allFinite() || hess.determinant() == 0) {
		return false;
	}

	// ensuring the symetry (A = A^T implies A = 1/2 * (A + A^T))
	hess = 0.5 * (hess + hess.transpose());

	auto hess_inv = hess.inverse();

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(hess_inv);
	auto e_values = solver.eigenvalues();

	for (const auto &e_val : e_values) {
		// for hess_inv to be PSD, its EVs need to be non-negative
		// because the hess_inv is self-adjoint, it will only have real EVs
		if (e_val < 0.0)
			return false;
	}

	cov = hess_inv;

	return cov.allFinite();
}

inline uint64_t x_(uint64_t key) { return gtsam::Symbol('x', key).key(); }
inline uint64_t l_(uint64_t key) { return gtsam::Symbol('l', key).key(); }
inline uint64_t key_(uint64_t symbol_key) {
	return gtsam::Symbol(symbol_key).index();
}

// pack pose+landmark keys into a single 64-bit key
inline uint64_t encode_keys(uint64_t pose_key, uint64_t landmark_key) {
	constexpr uint64_t mask32 = 0xffffffffULL;

	if (pose_key > mask32 || landmark_key > mask32) {
		throw std::out_of_range(
			"[ENCODE_KEYS] keys value exceed 32 bit: encoding does not work!");
	}

	return (pose_key << 32) | landmark_key;
}

// KD buffer for a single point
inline KD_TREE<LMPoint>::PointVector get_buffer(const LMPoint &p) {
	KD_TREE<LMPoint>::PointVector v;
	v.reserve(1);
	v.emplace_back(p);
	return v;
}

// KD buffer from a list of points
inline KD_TREE<LMPoint>::PointVector
get_buffer(const std::map<uint64_t, std::shared_ptr<Landmark>> &p_map) {
	KD_TREE<LMPoint>::PointVector v;
	v.reserve(p_map.size());

	for (const auto &kv : p_map) {
		if (!kv.second)
			continue;
		v.emplace_back(kv.second->point);
	}

	return v;
}

// KD point to GTSAM point
inline auto to_gtsam_p(const LMPoint &p) {
	return gtsam::Point3(p.x, p.y, p.z);
}

// GTSAM point to KD point
inline auto to_ikd_p(const gtsam::Point3 &p) {
	return LMPoint{static_cast<float>(p.x()), static_cast<float>(p.y()),
				   static_cast<float>(p.z()), 0};
}

// apply transform to point set (used for the initial guess of KISS-Matcher)
inline std::vector<Eigen::Vector3f>
apply_transform(const std::vector<Eigen::Vector3f> &pts,
				const Eigen::Isometry3d &T) {
	const Eigen::Isometry3f T_f = T.cast<float>();

	std::vector<Eigen::Vector3f> ret;
	ret.reserve(pts.size());

	for (const auto &p : pts)
		ret.push_back(T_f * p);

	return ret;
}

// Pose3 to 4x4 matrix
static inline Eigen::Matrix4d gtsam_to_mat4(const gtsam::Pose3 &p) {
	Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
	const auto R = p.rotation().matrix();
	const auto t = p.translation();

	T.block<3, 3>(0, 0) = R;

	T(0, 3) = t.x();
	T(1, 3) = t.y();
	T(2, 3) = t.z();

	return T;
}

// append pose as text entry in TUM format
static inline void appendPoseTxtEntry(const std::string &poses_txt_path,
									  uint64_t key, const Eigen::Matrix4d &T) {
	std::ofstream ofs(poses_txt_path, std::ios::out | std::ios::app);
	if (!ofs.is_open()) {
		throw std::runtime_error("Could not open poses_txt_path for append: " +
								 poses_txt_path);
	}
	std::cout << "saving pose " << key << " into " << poses_txt_path << "\n";

	ofs << "Key: " << key << "\n";

	ofs << std::fixed << std::setprecision(6);
	for (int r = 0; r < 4; ++r) {
		ofs << "  " << std::setw(10) << T(r, 0) << "  " << std::setw(10)
			<< T(r, 1) << "  " << std::setw(10) << T(r, 2) << "  "
			<< std::setw(10) << T(r, 3) << "\n";
	}
	ofs << "\n";
}

// comparator for descriptor candidates
inline bool compare(DescriptorMatchResult a, DescriptorMatchResult b) {
	return a.dist < b.dist;
}

// sorter for descriptor candidates
inline void sort_initial_candidates(std::vector<DescriptorMatchResult> &list) {
	std::sort(list.begin(), list.end(), compare);
}

// strip PointCloud2 to xyz only
inline sensor_msgs::msg::PointCloud2
filter_xyz(const sensor_msgs::msg::PointCloud2 &in) {
	sensor_msgs::msg::PointCloud2 out;
	out.header = in.header;
	out.height = in.height;
	out.width = in.width;
	out.is_bigendian = in.is_bigendian;
	out.is_dense = in.is_dense;

	sensor_msgs::PointCloud2Modifier mod(out);
	mod.setPointCloud2Fields(3, "x", 1, sensor_msgs::msg::PointField::FLOAT32,
							 "y", 1, sensor_msgs::msg::PointField::FLOAT32, "z",
							 1, sensor_msgs::msg::PointField::FLOAT32);
	mod.resize(static_cast<size_t>(in.width) * static_cast<size_t>(in.height));

	sensor_msgs::PointCloud2ConstIterator<float> ix(in, "x");
	sensor_msgs::PointCloud2ConstIterator<float> iy(in, "y");
	sensor_msgs::PointCloud2ConstIterator<float> iz(in, "z");

	sensor_msgs::PointCloud2Iterator<float> ox(out, "x");
	sensor_msgs::PointCloud2Iterator<float> oy(out, "y");
	sensor_msgs::PointCloud2Iterator<float> oz(out, "z");

	for (; ix != ix.end(); ++ix, ++iy, ++iz, ++ox, ++oy, ++oz) {
		*ox = *ix;
		*oy = *iy;
		*oz = *iz;
	}

	return out;
}

class StoreRegistry {
  public:
	// returns a shared store instance for (path, level)
	static std::shared_ptr<disk::DiskCloudStore>
	get(const std::string &path, int zstd_level = 1,
		bool truncate_existing = false, bool flush = false) {
		const Key k{path, zstd_level};

		std::lock_guard<std::mutex> lk(mtx_);

		auto it = stores_.find(k);
		if (it != stores_.end()) {
			if (auto sp = it->second.lock()) {
				return sp;
			}
		}

		auto sp = std::make_shared<disk::DiskCloudStore>(
			path, zstd_level, truncate_existing, flush);
		stores_[k] = sp;
		return sp;
	}

	static void clear() {
		std::lock_guard<std::mutex> lk(mtx_);
		stores_.clear();
	}

  private:
	struct Key {
		std::string path;
		int level;

		bool operator==(const Key &o) const {
			return level == o.level && path == o.path;
		}
	};

	struct KeyHash {
		std::size_t operator()(const Key &k) const {
			std::size_t h1 = std::hash<std::string>{}(k.path);
			std::size_t h2 = std::hash<int>{}(k.level);
			// simple combine
			return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
		}
	};

	static inline std::mutex mtx_;
	static inline std::unordered_map<Key, std::weak_ptr<disk::DiskCloudStore>,
									 KeyHash>
		stores_;
};

class LineWriter {
  public:
	enum class Mode { Truncate, Append };

	// simple line writer for logs
	LineWriter(const std::filesystem::path &filePath,
			   Mode mode = Mode::Truncate) {
		if (filePath.has_parent_path()) {
			std::filesystem::create_directories(filePath.parent_path());
		}

		std::ios::openmode openMode = std::ios::out;
		openMode |= (mode == Mode::Append) ? std::ios::app : std::ios::trunc;

		out_.open(filePath, openMode);
		if (!out_.is_open()) {
			throw std::runtime_error("could not open the file: " +
									 filePath.string());
		}
	}

	void writeLine(const std::string &line) {
		out_ << line << '\n';
		if (!out_) {
			throw std::runtime_error("write operation went wrong");
		}
	}

	template <typename... Ts> void writeLineParts(Ts &&...parts) {
		(out_ << ... << std::forward<Ts>(parts)) << '\n';
		if (!out_) {
			throw std::runtime_error("write operation went wrong");
		}
	}

	void flush() { out_.flush(); }

  private:
	std::ofstream out_;
};

// current resident set size in MB
inline size_t get_rss_mb() {
	std::ifstream f("/proc/self/status");
	if (!f.is_open())
		return 0;

	std::string line;
	while (std::getline(f, line)) {
		if (line.rfind("VmRSS:", 0) == 0) {
			std::string num;
			for (char c : line) {
				if (std::isdigit(static_cast<unsigned char>(c)))
					num.push_back(c);
				else if (!num.empty())
					break;
			}
			if (num.empty())
				return 0;
			const size_t kb = static_cast<size_t>(std::stoull(num));
			return kb / 1024;
		}
	}
	return 0;
}

// try malloc_trim and return before/after RSS
inline std::tuple<bool, int, int> clean_rss() {
	const size_t rss_mb = get_rss_mb();
	if (rss_mb == 0)
		return {false, rss_mb, rss_mb};

	const int trimmed = malloc_trim(0);
	bool T = (trimmed == 1) ? true : false;

	const size_t rss_mb_after = get_rss_mb();

	return {T, rss_mb, rss_mb_after};
}

// ROS covariance array to GTSAM order covariance
inline Eigen::Matrix<double, 6, 6>
ros_to_gtsam_cov(const std::array<double, 36> &cov) {
	Eigen::Matrix<double, 6, 6> cov_ros =
		Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(
			cov.data());

	Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();

	A.block<3, 3>(0, 3).setIdentity();
	A.block<3, 3>(3, 0).setIdentity();

	return A * cov_ros * A.transpose();
}

// integrate twist covariance over dt into pose increment covariance
static inline Eigen::Matrix<double, 6, 6>
ros_twist_to_gtsam_increment_cov(const std::array<double, 36> &twist_cov_ros,
								 double dt) {
	Eigen::Matrix<double, 6, 6> C =
		Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(
			twist_cov_ros.data());

	Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Zero();
	P.block<3, 3>(0, 3).setIdentity();
	P.block<3, 3>(3, 0).setIdentity();

	Eigen::Matrix<double, 6, 6> Cg = P * C * P.transpose();

	Cg *= (dt * dt);
	Cg = 0.5 * (Cg + Cg.transpose());

	return Cg;
}
