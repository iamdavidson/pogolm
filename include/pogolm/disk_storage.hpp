#pragma once

#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <memory>
#include <type_traits>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <zstd.h>

#include <Eigen/Dense>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/points/point_cloud.hpp>

namespace disk
{

// timing breakdown for get()
struct GetDurations
{
  size_t points{ 0 };
  double total_duration_ms{ 0.0 };
  double fetch_duratino_ms{ 0.0 };
  double rec_read_duration_ms{ 0.0 };
  double decompress_duration_ms{ 0.0 };
  double deserialize_duration_ms{ 0.0 };
};

// timing breakdown for put()
struct PutDurations
{
  size_t points{ 0 };
  double total_duration_ms{ 0.0 };
  double shrink_duration_ms{ 0.0 };
  double downsample_duration_ms{ 0.0 };
  double serialize_duration_ms{ 0.0 };
  double compress_duration_ms{ 0.0 };
  double file_appedn_duration_ms{ 0.0 };
  double update_index_duration_ms{ 0.0 };
};

#pragma pack(push, 1)
// record header on disk (followed by compressed payload)
struct RecordHeader
{
  uint64_t key;
  uint32_t raw_size;
  uint32_t comp_size;
};
#pragma pack(pop)

// in-memory index entry (file offset + sizes)
struct IndexEntry
{
  uint64_t offset;
  uint32_t raw_size;
  uint32_t comp_size;
};

// append-only store for serialized PointCloud2 messages
class DiskCloudStore
{
public:
  // open/create store file, optionally truncating existing file
  explicit DiskCloudStore(const std::string& path, int zstd_level = 1, bool truncate_existing = false,
                          bool flush_each_put = false)
    : path_(path), zstd_level_(zstd_level), flush_each_put_(flush_each_put)
  {
    namespace fs = std::filesystem;

    if (truncate_existing)
    {
      if (fs::exists(path_))
      {
        std::ofstream trunc(path_, std::ios::binary | std::ios::trunc);
        if (!trunc.is_open())
        {
          throw std::runtime_error("DiskCloudStore: cannot truncate existing file");
        }
      }
    }

    out_.open(path_, std::ios::binary | std::ios::app);
    if (!out_.is_open())
      throw std::runtime_error("DiskCloudStore: cannot open for append");

    in_.open(path_, std::ios::binary);
    if (!in_.is_open())
      throw std::runtime_error("DiskCloudStore: cannot open for read");

    out_.seekp(0, std::ios::end);
    write_offset_ = static_cast<uint64_t>(out_.tellp());
  }

  // check if key exists in the in-memory index
  bool contains(uint64_t key) const
  {
    std::lock_guard<std::mutex> lk(index_mtx_);
    return index_.find(key) != index_.end();
  }

  // number of indexed records
  size_t size() const
  {
    std::lock_guard<std::mutex> lk(index_mtx_);
    return index_.size();
  }

  // serialize, compress, append, and update index
  void put(uint64_t key, const sensor_msgs::msg::PointCloud2& msg, bool verbose = false)
  {
    PutDurations p;
    p.points = static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height);

    auto t_0_s = std::chrono::steady_clock::now();

    rclcpp::Serialization<sensor_msgs::msg::PointCloud2> ser;
    rclcpp::SerializedMessage serialized;
    ser.serialize_message(&msg, &serialized);
    const auto& rcl = serialized.get_rcl_serialized_message();
    const uint8_t* raw = rcl.buffer;
    const size_t raw_size = rcl.buffer_length;

    auto t_1_s = std::chrono::steady_clock::now();

    auto t_0_c = std::chrono::steady_clock::now();
    const size_t bound = ZSTD_compressBound(raw_size);
    std::vector<uint8_t> comp(bound);

    size_t comp_size = ZSTD_compress(comp.data(), comp.size(), raw, raw_size, zstd_level_);
    if (ZSTD_isError(comp_size))
    {
      throw std::runtime_error(std::string("ZSTD_compress: ") + ZSTD_getErrorName(comp_size));
    }
    comp.resize(comp_size);
    auto t_1_c = std::chrono::steady_clock::now();

    auto t_0_a = std::chrono::steady_clock::now();

    RecordHeader h;
    h.key = key;
    h.raw_size = static_cast<uint32_t>(raw_size);
    h.comp_size = static_cast<uint32_t>(comp.size());

    uint64_t offset;
    {
      std::lock_guard<std::mutex> lk(file_mtx_);
      offset = write_offset_;

      out_.write(reinterpret_cast<const char*>(&h), sizeof(h));
      out_.write(reinterpret_cast<const char*>(comp.data()), static_cast<std::streamsize>(comp.size()));

      if (!out_)
        throw std::runtime_error("DiskCloudStore: write failed");

      write_offset_ += sizeof(h) + comp.size();

      if (flush_each_put_)
      {
        out_.flush();
        if (!out_)
          throw std::runtime_error("DiskCloudStore: flush failed");
      }
    }
    auto t_1_a = std::chrono::steady_clock::now();

    auto t_0_u = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lk(index_mtx_);
      index_[key] = IndexEntry{ offset, h.raw_size, h.comp_size };
    }
    auto t_1_u = std::chrono::steady_clock::now();

    p.serialize_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_s - t_0_s).count();
    p.compress_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_c - t_0_c).count();
    p.file_appedn_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_a - t_0_a).count();
    p.update_index_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_u - t_0_u).count();
    p.total_duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_u - t_0_s).count();

    if (verbose)
    {
      RCLCPP_INFO(rclcpp::get_logger("disk"), "|                                  |");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "|                                  |");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "|                                  |");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "====================================");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––– point cloud size: %ld points", (long)p.points);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "====================================");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––– <%llu> total put duration   : %.5f ms", (unsigned long long)key,
                  p.total_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– serialization : %.5f ms", p.serialize_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– compression   : %.5f ms", p.compress_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– append        : %.5f ms", p.file_appedn_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– update index  : %.5f ms", p.update_index_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "====================================");
    }
  }

  // read record by key, decompress, deserialize
  bool get(uint64_t key, sensor_msgs::msg::PointCloud2& out_msg, bool verbose = false) const
  {
    GetDurations d;

    auto t_0_f = std::chrono::steady_clock::now();
    IndexEntry e;
    {
      std::lock_guard<std::mutex> lk(index_mtx_);
      auto it = index_.find(key);
      if (it == index_.end())
        return false;
      e = it->second;
    }
    auto t_1_f = std::chrono::steady_clock::now();
    d.fetch_duratino_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_f - t_0_f).count();

    auto t_0_r = std::chrono::steady_clock::now();
    RecordHeader h{};
    std::vector<uint8_t> comp;
    {
      std::lock_guard<std::mutex> lk(file_mtx_);
      in_.clear();
      in_.seekg(static_cast<std::streamoff>(e.offset), std::ios::beg);

      in_.read(reinterpret_cast<char*>(&h), sizeof(h));
      if (!in_)
        return false;

      if (h.key != key)
      {
        throw std::runtime_error("DiskCloudStore: header key mismatch (corrupt index or file)");
      }

      comp.resize(h.comp_size);
      in_.read(reinterpret_cast<char*>(comp.data()), static_cast<std::streamsize>(comp.size()));
      if (!in_)
        return false;
    }
    auto t_1_r = std::chrono::steady_clock::now();
    d.rec_read_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_r - t_0_r).count();

    auto t_0_d = std::chrono::steady_clock::now();
    std::vector<uint8_t> raw(h.raw_size);
    size_t decomp = ZSTD_decompress(raw.data(), raw.size(), comp.data(), comp.size());
    if (ZSTD_isError(decomp) || decomp != raw.size())
    {
      throw std::runtime_error(std::string("ZSTD_decompress: ") + ZSTD_getErrorName(decomp));
    }
    auto t_1_d = std::chrono::steady_clock::now();
    d.decompress_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_d - t_0_d).count();

    auto t_0_ds = std::chrono::steady_clock::now();

    rclcpp::SerializedMessage serialized(raw.size());
    auto& rcl = serialized.get_rcl_serialized_message();
    std::memcpy(rcl.buffer, raw.data(), raw.size());
    rcl.buffer_length = raw.size();

    rclcpp::Serialization<sensor_msgs::msg::PointCloud2> ser;
    ser.deserialize_message(&serialized, &out_msg);

    auto t_1_ds = std::chrono::steady_clock::now();
    d.deserialize_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_ds - t_0_ds).count();

    d.points = static_cast<size_t>(out_msg.width) * static_cast<size_t>(out_msg.height);
    d.total_duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_1_ds - t_0_f).count();

    if (verbose)
    {
      RCLCPP_INFO(rclcpp::get_logger("disk"), "|                                  |");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "|                                  |");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "|                                  |");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "====================================");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––– point cloud size: %ld points", (long)d.points);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "====================================");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––– <%llu> total get duration   : %.5f ms", (unsigned long long)key,
                  d.total_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– fetch         : %.5f ms", d.fetch_duratino_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– read          : %.5f ms", d.rec_read_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– decompress    : %.5f ms", d.decompress_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– deserialized  : %.5f ms", d.deserialize_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "====================================");
    }

    return true;
  }

private:
  std::string path_;
  int zstd_level_;
  bool flush_each_put_{ false };

  mutable std::mutex index_mtx_;
  mutable std::mutex file_mtx_;

  mutable std::ifstream in_;
  mutable std::ofstream out_;
  mutable uint64_t write_offset_{ 0 };

  std::unordered_map<uint64_t, IndexEntry> index_;
};

// cached objects used by the map module
struct VoxelmapCache
{
  std::shared_ptr<small_gicp::PointCloud> pc_sg;
  std::shared_ptr<small_gicp::GaussianVoxelMap> vm_sg;
  std::vector<Eigen::Vector3f> pts_eigen;

  // release memory owned by the cache
  void free()
  {
    pc_sg.reset();
    vm_sg.reset();
    pts_eigen.clear();
    pts_eigen.shrink_to_fit();
  }
};

struct CachePutDurations
{
  uint64_t pts_eigen_points{ 0 };
  uint64_t pc_points{ 0 };
  uint64_t vm_voxels{ 0 };

  double total_duration_ms{ 0.0 };
  double serialize_duration_ms{ 0.0 };
  double compress_duration_ms{ 0.0 };
  double file_append_duration_ms{ 0.0 };
  double update_index_duration_ms{ 0.0 };
};

struct CacheGetDurations
{
  uint64_t pts_eigen_points{ 0 };
  uint64_t pc_points{ 0 };
  uint64_t vm_voxels{ 0 };

  double total_duration_ms{ 0.0 };
  double fetch_duration_ms{ 0.0 };
  double rec_read_duration_ms{ 0.0 };
  double decompress_duration_ms{ 0.0 };
  double deserialize_duration_ms{ 0.0 };
};

// append raw bytes to buffer
inline void append_bytes(std::vector<uint8_t>& dst, const void* p, size_t n)
{
  const auto* b = reinterpret_cast<const uint8_t*>(p);
  dst.insert(dst.end(), b, b + n);
}

// append POD type to buffer
template <class T>
inline void append_pod(std::vector<uint8_t>& dst, const T& v)
{
  static_assert(std::is_trivially_copyable_v<T>);
  append_bytes(dst, &v, sizeof(T));
}

// read POD type from buffer and advance cursor
template <class T>
inline T read_pod(const uint8_t*& cur, const uint8_t* end)
{
  static_assert(std::is_trivially_copyable_v<T>);
  if (cur + sizeof(T) > end)
    throw std::runtime_error("DiskVoxelmapCacheStore: buffer underrun");
  T out;
  std::memcpy(&out, cur, sizeof(T));
  cur += sizeof(T);
  return out;
}

// cache format constants
static constexpr uint32_t CACHE_MAGIC = 0x31434356;
static constexpr uint32_t CACHE_VERSION = 1;
static constexpr uint32_t SEC_PTS_EIGEN = 1u << 0;
static constexpr uint32_t SEC_PC_SG = 1u << 1;
static constexpr uint32_t SEC_VM_SG = 1u << 2;

// serialize eigen points section
inline void serialize_pts_eigen(std::vector<uint8_t>& raw, const std::vector<Eigen::Vector3f>& pts)
{
  const uint64_t n = static_cast<uint64_t>(pts.size());
  append_pod(raw, n);
  for (const auto& p : pts)
  {
    append_pod(raw, p.x());
    append_pod(raw, p.y());
    append_pod(raw, p.z());
  }
}

// deserialize eigen points section
inline std::vector<Eigen::Vector3f> deserialize_pts_eigen(const uint8_t*& cur, const uint8_t* end)
{
  const uint64_t n = read_pod<uint64_t>(cur, end);
  std::vector<Eigen::Vector3f> pts;
  pts.reserve(static_cast<size_t>(n));
  for (uint64_t i = 0; i < n; ++i)
  {
    const float x = read_pod<float>(cur, end);
    const float y = read_pod<float>(cur, end);
    const float z = read_pod<float>(cur, end);
    pts.emplace_back(x, y, z);
  }
  return pts;
}

// serialize small_gicp point cloud (points + optional covariances)
inline void serialize_pointcloud(std::vector<uint8_t>& raw, const small_gicp::PointCloud& pc)
{
  const uint64_t n = static_cast<uint64_t>(pc.size());
  append_pod(raw, n);

  for (size_t i = 0; i < pc.size(); ++i)
  {
    const auto& pt = pc.points[i];
    const float x = static_cast<float>(pt.x());
    const float y = static_cast<float>(pt.y());
    const float z = static_cast<float>(pt.z());
    append_pod(raw, x);
    append_pod(raw, y);
    append_pod(raw, z);
  }

  const uint8_t has_covs = (!pc.covs.empty()) ? 1 : 0;
  append_pod(raw, has_covs);

  if (has_covs)
  {
    if (pc.covs.size() != pc.size())
    {
      throw std::runtime_error("serialize_pointcloud: covs.size != points.size");
    }
    for (size_t i = 0; i < pc.size(); ++i)
    {
      const auto& C = pc.covs[i];
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
          append_pod(raw, static_cast<double>(C(r, c)));
    }
  }
}

// deserialize small_gicp point cloud
inline std::shared_ptr<small_gicp::PointCloud> deserialize_pointcloud(const uint8_t*& cur, const uint8_t* end)
{
  const uint64_t n = read_pod<uint64_t>(cur, end);

  std::vector<Eigen::Vector3f> pts;
  pts.reserve(static_cast<size_t>(n));
  for (uint64_t i = 0; i < n; ++i)
  {
    const float x = read_pod<float>(cur, end);
    const float y = read_pod<float>(cur, end);
    const float z = read_pod<float>(cur, end);
    pts.emplace_back(x, y, z);
  }

  auto pc = std::make_shared<small_gicp::PointCloud>(pts);

  const uint8_t has_covs = read_pod<uint8_t>(cur, end);
  if (has_covs)
  {
    if (pc->covs.size() != pc->size())
    {
      pc->covs.resize(pc->size(), Eigen::Matrix4d::Zero());
    }
    for (size_t i = 0; i < pc->size(); ++i)
    {
      Eigen::Matrix4d C = Eigen::Matrix4d::Zero();
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
          C(r, c) = read_pod<double>(cur, end);
      pc->covs[i] = C;
    }
  }

  return pc;
}

// serialize gaussian voxel map
inline void serialize_gaussian_voxelmap(std::vector<uint8_t>& raw, const small_gicp::GaussianVoxelMap& vm)
{
  const double inv_leaf = vm.inv_leaf_size;
  const double leaf_size = 1.0 / inv_leaf;

  append_pod(raw, leaf_size);
  append_pod(raw, static_cast<uint64_t>(vm.lru_horizon));
  append_pod(raw, static_cast<uint64_t>(vm.lru_clear_cycle));
  append_pod(raw, static_cast<uint64_t>(vm.lru_counter));

  const uint32_t num_offsets = static_cast<uint32_t>(vm.search_offsets.size());
  append_pod(raw, num_offsets);

  const uint64_t nvox = static_cast<uint64_t>(vm.flat_voxels.size());
  append_pod(raw, nvox);

  for (const auto& sp : vm.flat_voxels)
  {
    const auto& info = sp->first;
    const auto& vox = sp->second;

    append_pod(raw, static_cast<int32_t>(info.coord.x()));
    append_pod(raw, static_cast<int32_t>(info.coord.y()));
    append_pod(raw, static_cast<int32_t>(info.coord.z()));

    append_pod(raw, static_cast<uint64_t>(info.lru));
    append_pod(raw, static_cast<uint64_t>(vox.num_points));

    append_pod(raw, static_cast<double>(vox.mean.x()));
    append_pod(raw, static_cast<double>(vox.mean.y()));
    append_pod(raw, static_cast<double>(vox.mean.z()));

    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        append_pod(raw, static_cast<double>(vox.cov(r, c)));

    const uint8_t fin = static_cast<uint8_t>(vox.finalized ? 1 : 0);
    append_pod(raw, fin);
  }
}

// deserialize gaussian voxel map
inline std::shared_ptr<small_gicp::GaussianVoxelMap> deserialize_gaussian_voxelmap(const uint8_t*& cur,
                                                                                   const uint8_t* end)
{
  const double leaf_size = read_pod<double>(cur, end);
  const uint64_t lru_horizon = read_pod<uint64_t>(cur, end);
  const uint64_t lru_cycle = read_pod<uint64_t>(cur, end);
  const uint64_t lru_counter = read_pod<uint64_t>(cur, end);
  const uint32_t num_offsets = read_pod<uint32_t>(cur, end);
  const uint64_t nvox = read_pod<uint64_t>(cur, end);

  auto vm = std::make_shared<small_gicp::GaussianVoxelMap>(leaf_size);
  vm->set_search_offsets(static_cast<int>(num_offsets));
  vm->lru_horizon = static_cast<size_t>(lru_horizon);
  vm->lru_clear_cycle = static_cast<size_t>(lru_cycle);
  vm->lru_counter = static_cast<size_t>(lru_counter);

  vm->flat_voxels.clear();
  vm->voxels.clear();
  vm->flat_voxels.reserve(static_cast<size_t>(nvox));

  for (uint64_t i = 0; i < nvox; ++i)
  {
    const int32_t cx = read_pod<int32_t>(cur, end);
    const int32_t cy = read_pod<int32_t>(cur, end);
    const int32_t cz = read_pod<int32_t>(cur, end);

    const uint64_t lru = read_pod<uint64_t>(cur, end);
    const uint64_t num_points = read_pod<uint64_t>(cur, end);

    const double mx = read_pod<double>(cur, end);
    const double my = read_pod<double>(cur, end);
    const double mz = read_pod<double>(cur, end);

    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        cov(r, c) = read_pod<double>(cur, end);

    const uint8_t fin = read_pod<uint8_t>(cur, end);

    small_gicp::VoxelInfo info(Eigen::Vector3i(cx, cy, cz), static_cast<size_t>(lru));
    small_gicp::GaussianVoxel gv;
    gv.finalized = (fin != 0);
    gv.num_points = static_cast<size_t>(num_points);
    gv.mean = Eigen::Vector4d(mx, my, mz, 1.0);
    gv.cov = cov;

    auto pair_sp = std::make_shared<std::pair<small_gicp::VoxelInfo, small_gicp::GaussianVoxel>>(info, gv);
    vm->voxels[info.coord] = vm->flat_voxels.size();
    vm->flat_voxels.emplace_back(std::move(pair_sp));
  }

  return vm;
}

// append-only store for voxelmap caches (points/pc/voxelmap sections)
class DiskVoxelmapCacheStore
{
public:
  // open/create cache file, optionally truncating existing file
  explicit DiskVoxelmapCacheStore(const std::string& path, int zstd_level = 1, bool truncate_existing = false,
                                  bool flush_each_put = false)
    : path_(path), zstd_level_(zstd_level), flush_each_put_(flush_each_put)
  {
    namespace fs = std::filesystem;

    if (truncate_existing)
    {
      if (fs::exists(path_))
      {
        std::ofstream trunc(path_, std::ios::binary | std::ios::trunc);
        if (!trunc.is_open())
        {
          throw std::runtime_error("DiskVoxelmapCacheStore: cannot truncate existing file");
        }
      }
    }

    out_.open(path_, std::ios::binary | std::ios::app);
    if (!out_.is_open())
      throw std::runtime_error("DiskVoxelmapCacheStore: cannot open for append");

    in_.open(path_, std::ios::binary);
    if (!in_.is_open())
      throw std::runtime_error("DiskVoxelmapCacheStore: cannot open for read");

    out_.seekp(0, std::ios::end);
    write_offset_ = static_cast<uint64_t>(out_.tellp());
  }

  // check if key exists in the in-memory index
  bool contains(uint64_t key) const
  {
    std::lock_guard<std::mutex> lk(index_mtx_);
    return index_.find(key) != index_.end();
  }

  // number of indexed records
  size_t size() const
  {
    std::lock_guard<std::mutex> lk(index_mtx_);
    return index_.size();
  }

  // serialize selected cache sections, compress, append, and update index
  void put(uint64_t key, const VoxelmapCache& cache, bool verbose = false)
  {
    if (!cache.vm_sg && !cache.pc_sg && cache.pts_eigen.empty())
    {
      throw std::runtime_error("DiskVoxelmapCacheStore::put: empty cache");
    }

    CachePutDurations d;
    const auto t0 = std::chrono::steady_clock::now();

    const auto ts0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> raw;
    raw.reserve(1024);

    append_pod(raw, CACHE_MAGIC);
    append_pod(raw, CACHE_VERSION);

    uint32_t sections = 0;
    if (!cache.pts_eigen.empty())
      sections |= SEC_PTS_EIGEN;
    if (cache.pc_sg)
      sections |= SEC_PC_SG;
    if (cache.vm_sg)
      sections |= SEC_VM_SG;
    append_pod(raw, sections);

    if (sections & SEC_PTS_EIGEN)
    {
      d.pts_eigen_points = static_cast<uint64_t>(cache.pts_eigen.size());
      serialize_pts_eigen(raw, cache.pts_eigen);
    }

    if (sections & SEC_PC_SG)
    {
      d.pc_points = static_cast<uint64_t>(cache.pc_sg->size());
      serialize_pointcloud(raw, *cache.pc_sg);
    }

    if (sections & SEC_VM_SG)
    {
      d.vm_voxels = static_cast<uint64_t>(cache.vm_sg->flat_voxels.size());
      serialize_gaussian_voxelmap(raw, *cache.vm_sg);
    }

    const auto ts1 = std::chrono::steady_clock::now();

    const auto tc0 = std::chrono::steady_clock::now();
    const size_t bound = ZSTD_compressBound(raw.size());
    std::vector<uint8_t> comp(bound);

    size_t comp_size = ZSTD_compress(comp.data(), comp.size(), raw.data(), raw.size(), zstd_level_);
    if (ZSTD_isError(comp_size))
    {
      throw std::runtime_error(std::string("ZSTD_compress: ") + ZSTD_getErrorName(comp_size));
    }
    comp.resize(comp_size);
    const auto tc1 = std::chrono::steady_clock::now();

    const auto ta0 = std::chrono::steady_clock::now();

    RecordHeader h;
    h.key = key;
    h.raw_size = static_cast<uint32_t>(raw.size());
    h.comp_size = static_cast<uint32_t>(comp.size());

    uint64_t offset;
    {
      std::lock_guard<std::mutex> lk(file_mtx_);
      offset = write_offset_;

      out_.write(reinterpret_cast<const char*>(&h), sizeof(h));
      out_.write(reinterpret_cast<const char*>(comp.data()), static_cast<std::streamsize>(comp.size()));
      if (!out_)
        throw std::runtime_error("DiskVoxelmapCacheStore: write failed");

      write_offset_ += sizeof(h) + comp.size();

      if (flush_each_put_)
      {
        out_.flush();
        if (!out_)
          throw std::runtime_error("DiskVoxelmapCacheStore: flush failed");
      }
    }

    const auto ta1 = std::chrono::steady_clock::now();

    const auto tu0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lk(index_mtx_);
      index_[key] = IndexEntry{ offset, h.raw_size, h.comp_size };
    }
    const auto tu1 = std::chrono::steady_clock::now();

    d.serialize_duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(ts1 - ts0).count();
    d.compress_duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tc1 - tc0).count();
    d.file_append_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(ta1 - ta0).count();
    d.update_index_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tu1 - tu0).count();
    d.total_duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tu1 - t0).count();

    if (verbose)
    {
      RCLCPP_INFO(rclcpp::get_logger("disk"), "|                                                 |");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "|                                                 |");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "|                                                 |");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "===================================================");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––– cache sizes: pts=%llu pc=%llu vm=%llu",
                  (unsigned long long)d.pts_eigen_points, (unsigned long long)d.pc_points,
                  (unsigned long long)d.vm_voxels);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "===================================================");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––– <%llu> total CACHE put duration : %.5f ms", (unsigned long long)key,
                  d.total_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "===================================================");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– serialize      : %.5f ms", d.serialize_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– compress       : %.5f ms", d.compress_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– append         : %.5f ms", d.file_append_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– update index   : %.5f ms", d.update_index_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "===================================================");
    }
  }

  // read record by key, decompress, and rebuild cache
  bool get(uint64_t key, VoxelmapCache& out_cache, bool verbose = false) const
  {
    CacheGetDurations d;
    const auto t0 = std::chrono::steady_clock::now();

    const auto tf0 = std::chrono::steady_clock::now();
    IndexEntry e;
    {
      std::lock_guard<std::mutex> lk(index_mtx_);
      auto it = index_.find(key);
      if (it == index_.end())
        return false;
      e = it->second;
    }
    const auto tf1 = std::chrono::steady_clock::now();
    d.fetch_duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tf1 - tf0).count();

    const auto tr0 = std::chrono::steady_clock::now();
    RecordHeader h{};
    std::vector<uint8_t> comp;
    {
      std::lock_guard<std::mutex> lk(file_mtx_);
      in_.clear();
      in_.seekg(static_cast<std::streamoff>(e.offset), std::ios::beg);

      in_.read(reinterpret_cast<char*>(&h), sizeof(h));
      if (!in_)
        return false;

      if (h.key != key)
      {
        throw std::runtime_error("DiskVoxelmapCacheStore: header key mismatch");
      }

      comp.resize(h.comp_size);
      in_.read(reinterpret_cast<char*>(comp.data()), static_cast<std::streamsize>(comp.size()));
      if (!in_)
        return false;
    }
    const auto tr1 = std::chrono::steady_clock::now();
    d.rec_read_duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tr1 - tr0).count();

    const auto td0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> raw(h.raw_size);
    size_t decomp = ZSTD_decompress(raw.data(), raw.size(), comp.data(), comp.size());
    if (ZSTD_isError(decomp) || decomp != raw.size())
    {
      throw std::runtime_error(std::string("ZSTD_decompress: ") + ZSTD_getErrorName(decomp));
    }
    const auto td1 = std::chrono::steady_clock::now();
    d.decompress_duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(td1 - td0).count();

    const auto tds0 = std::chrono::steady_clock::now();

    const uint8_t* cur = raw.data();
    const uint8_t* end = raw.data() + raw.size();

    const uint32_t magic = read_pod<uint32_t>(cur, end);
    const uint32_t ver = read_pod<uint32_t>(cur, end);
    if (magic != CACHE_MAGIC || ver != CACHE_VERSION)
    {
      throw std::runtime_error("DiskVoxelmapCacheStore: bad magic/version");
    }

    const uint32_t sections = read_pod<uint32_t>(cur, end);

    out_cache = VoxelmapCache{};

    if (sections & SEC_PTS_EIGEN)
    {
      out_cache.pts_eigen = deserialize_pts_eigen(cur, end);
      d.pts_eigen_points = static_cast<uint64_t>(out_cache.pts_eigen.size());
    }

    if (sections & SEC_PC_SG)
    {
      out_cache.pc_sg = deserialize_pointcloud(cur, end);
      d.pc_points = static_cast<uint64_t>(out_cache.pc_sg->size());
    }

    if (sections & SEC_VM_SG)
    {
      out_cache.vm_sg = deserialize_gaussian_voxelmap(cur, end);
      d.vm_voxels = static_cast<uint64_t>(out_cache.vm_sg->flat_voxels.size());
    }

    const auto tds1 = std::chrono::steady_clock::now();
    d.deserialize_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tds1 - tds0).count();

    d.total_duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tds1 - t0).count();

    if (verbose)
    {
      RCLCPP_INFO(rclcpp::get_logger("disk"), "|                                                 |");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "|                                                 |");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "|                                                 |");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "===================================================");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––– cache sizes: pts=%llu pc=%llu vm=%llu",
                  (unsigned long long)d.pts_eigen_points, (unsigned long long)d.pc_points,
                  (unsigned long long)d.vm_voxels);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "===================================================");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––– <%llu> total CACHE get duration : %.5f ms", (unsigned long long)key,
                  d.total_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "===================================================");
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– fetch index     : %.5f ms", d.fetch_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– record read     : %.5f ms", d.rec_read_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– decompress      : %.5f ms", d.decompress_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "–––––––– deserialize     : %.5f ms", d.deserialize_duration_ms);
      RCLCPP_INFO(rclcpp::get_logger("disk"), "===================================================");
    }

    return true;
  }

private:
  std::string path_;
  int zstd_level_;
  bool flush_each_put_{ false };

  mutable std::mutex index_mtx_;
  mutable std::mutex file_mtx_;

  mutable std::ifstream in_;
  mutable std::ofstream out_;
  mutable uint64_t write_offset_{ 0 };

  std::unordered_map<uint64_t, IndexEntry> index_;
};

}  // namespace disk