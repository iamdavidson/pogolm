#include <rclcpp/rclcpp.hpp>
#include "pogolm/pogolm_api.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  pogolm_np::APIParams params;
  params.pg_params.operating_frame_id = "lio_init";
  params.pg_params.lidar_in = "/velodyne_points";
  params.pg_params.odom_in = "/pelorus/odom";
  params.pg_params.lin_cirt = 3.;
  params.pg_params.ang_crit = 0.3;
  params.pg_params.to_time_crit = 3.;
  params.pg_params.to_dist_crit = 1.;

  params.ld_params.min_loop_length = 40;

  auto pogolm = std::make_shared<pogolm_np::POGOLM>(params);

  auto exec = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();

  pogolm->attach_to_exec(exec);

  exec->spin();
  rclcpp::shutdown();

  return 0;
}
