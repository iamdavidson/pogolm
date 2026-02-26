cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0144 NEW)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

project(pogolm)

# ===================================================================
# Build settings
# ===================================================================
add_compile_definitions(BOOST_RANGE_ALGORITHM_SORT_HPP_INCLUDED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ===================================================================
# Dependencies
# ===================================================================

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)

find_package(factor_graph_interfaces REQUIRED)
find_package(pogolm_interfaces REQUIRED)

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(PCL REQUIRED COMPONENTS io filters segmentation)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)

find_package(small_gicp REQUIRED)
find_package(robin REQUIRED)
find_package(kiss_matcher REQUIRED)

find_package(yaml-cpp REQUIRED)
find_package(FLANN REQUIRED)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZSTD QUIET)

set(ZSTD_LINK "") 

if(ZSTD_FOUND AND TARGET ZSTD::ZSTD)
  
  set(ZSTD_LINK ZSTD::ZSTD)
else()
 
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(ZSTD_PC REQUIRED libzstd)

 
  find_library(ZSTD_LIB zstd HINTS ${ZSTD_PC_LIBRARY_DIRS})
  if(ZSTD_LIB)
    set(ZSTD_LINK ${ZSTD_LIB})
  else()

    set(ZSTD_LINK zstd)
  endif()
endif()

# ===================================================================
# utils  (header-only helpers)
# ===================================================================

add_library(utils STATIC
  src/utils.cpp
  include/ikd-Tree/ikd_Tree.cpp
)

target_include_directories(utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ikd-Tree>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/ikd-Tree>
  ${PCL_INCLUDE_DIRS}
)

ament_target_dependencies(utils
  nav_msgs
  sensor_msgs
  geometry_msgs
  pcl_conversions
  pcl_msgs
)

target_link_libraries(utils
  Eigen3::Eigen
  gtsam
  small_gicp::small_gicp
  ${PCL_LIBRARIES}
  ${ZSTD_LINK}
)

install(TARGETS utils
  DESTINATION lib
)

# ===================================================================
# Helper macro 
# ===================================================================

# pose_graph sources
set(PG_SOURCES
  src/pose_graph.cpp
)

# loop_detector sources
set(LD_SOURCES
  src/loop_detector.cpp
  include/scancontext/Scancontext.cpp
)

# map_module sources
set(MM_SOURCES
  src/map_module.cpp
)

# ===================================================================
# Executables  
# ===================================================================

# ---- pose_graph ----
add_executable(pose_graph ${PG_SOURCES})

ament_target_dependencies(pose_graph
  rclcpp
  factor_graph_interfaces
  nav_msgs sensor_msgs geometry_msgs visualization_msgs
  pcl_conversions pcl_msgs
  pogolm_interfaces
)

target_include_directories(pose_graph PUBLIC
  ${factor_graph_interfaces_INCLUDE_DIRS}
  include
)

target_link_libraries(pose_graph utils ${PCL_LIBRARIES})


# ---- loop_detector ----
add_executable(loop_detector ${LD_SOURCES})

ament_target_dependencies(loop_detector
  rclcpp
  factor_graph_interfaces small_gicp
  nav_msgs sensor_msgs geometry_msgs visualization_msgs
  pcl_conversions pcl_msgs
  pogolm_interfaces OpenCV
)

target_compile_definitions(loop_detector PRIVATE ${PCL_DEFINITIONS} PCL_NO_PRECOMPILE)

target_include_directories(loop_detector PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  include
  include/scancontext
)

target_link_libraries(loop_detector
  utils
  small_gicp::small_gicp
  kiss_matcher::kiss_matcher_core
  robin::robin
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)


# ---- map_module ----
add_executable(map_module ${MM_SOURCES})

ament_target_dependencies(map_module
  rclcpp
  factor_graph_interfaces
  nav_msgs sensor_msgs geometry_msgs visualization_msgs
  pcl_conversions pcl_msgs
  pogolm_interfaces
)

target_include_directories(map_module PUBLIC include)

target_link_libraries(map_module utils)


# ---- test_node ----
add_executable(test_node src/test_node.cpp)

ament_target_dependencies(test_node
  rclcpp
  factor_graph_interfaces
  nav_msgs sensor_msgs geometry_msgs visualization_msgs
  pcl_conversions pcl_msgs
  pogolm_interfaces small_gicp OpenCV
)

target_include_directories(test_node PUBLIC
  include
  ${factor_graph_interfaces_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(test_node PRIVATE ${PCL_DEFINITIONS})

target_link_libraries(test_node
  utils
  small_gicp::small_gicp
  kiss_matcher::kiss_matcher_core
  robin::robin
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  pose_graph_comp
  loop_detector_comp
  map_module_comp
)

# ===================================================================
# Components 
# ===================================================================



# ---- pose_graph_comp ----
add_library(pose_graph_comp SHARED
  ${PG_SOURCES}
  src/pose_graph_comp.cpp
)

ament_target_dependencies(pose_graph_comp
  rclcpp rclcpp_components
  factor_graph_interfaces
  nav_msgs sensor_msgs geometry_msgs visualization_msgs
  pcl_conversions pcl_msgs
  pogolm_interfaces
)

target_include_directories(pose_graph_comp PUBLIC
  ${factor_graph_interfaces_INCLUDE_DIRS}
  include
)

target_link_libraries(pose_graph_comp utils ${PCL_LIBRARIES})

rclcpp_components_register_node(pose_graph_comp
  PLUGIN     "pose_graph_np::PoseGraphComponent"
  EXECUTABLE pose_graph_node
)

install(TARGETS pose_graph_comp
  EXPORT  export_pose_graph_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)


# ---- loop_detector_comp ----
add_library(loop_detector_comp SHARED
  ${LD_SOURCES}
  src/loop_detector_comp.cpp
)

ament_target_dependencies(loop_detector_comp
  rclcpp rclcpp_components
  factor_graph_interfaces small_gicp
  nav_msgs sensor_msgs geometry_msgs visualization_msgs
  pcl_conversions pcl_msgs
  pogolm_interfaces OpenCV
)

target_compile_definitions(loop_detector_comp PRIVATE ${PCL_DEFINITIONS} PCL_NO_PRECOMPILE)

target_include_directories(loop_detector_comp PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  include
  include/scancontext
)

target_link_libraries(loop_detector_comp
  utils
  small_gicp::small_gicp
  kiss_matcher::kiss_matcher_core
  robin::robin
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

rclcpp_components_register_node(loop_detector_comp
  PLUGIN     "loop_detector_np::LoopDetectorComponent"
  EXECUTABLE loop_detector_node
)

install(TARGETS loop_detector_comp
  EXPORT  export_loop_detector_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)


# ---- map_module_comp ----
add_library(map_module_comp SHARED
  ${MM_SOURCES}
  src/map_module_comp.cpp
)

ament_target_dependencies(map_module_comp
  rclcpp rclcpp_components
  factor_graph_interfaces
  nav_msgs sensor_msgs geometry_msgs visualization_msgs
  pcl_conversions pcl_msgs
  pogolm_interfaces
)

target_include_directories(map_module_comp PUBLIC include)

target_link_libraries(map_module_comp utils)

rclcpp_components_register_node(map_module_comp
  PLUGIN     "map_module_np::MapModuleComponent"
  EXECUTABLE map_module_node
)

install(TARGETS map_module_comp
  EXPORT  export_map_module_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

target_compile_definitions(pose_graph_comp PRIVATE PGO_WITH_VISUAL_NO_MAIN)
target_compile_definitions(loop_detector_comp PRIVATE PGO_WITH_VISUAL_NO_MAIN)
target_compile_definitions(map_module_comp PRIVATE PGO_WITH_VISUAL_NO_MAIN)

# ===================================================================
# pogolm_api 
# ===================================================================

add_library(pogolm_api STATIC
  src/pogolm_api.cpp
  ${PG_SOURCES}
  ${LD_SOURCES}
  ${MM_SOURCES}
  include/ikd-Tree/ikd_Tree.cpp
  include/scancontext/Scancontext.cpp
)

target_include_directories(pogolm_api PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/scancontext>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ikd-Tree>
  $<INSTALL_INTERFACE:include>
  ${factor_graph_interfaces_INCLUDE_DIRS}
  ${pogolm_interfaces_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(pogolm_api
  rclcpp
  factor_graph_interfaces small_gicp
  nav_msgs sensor_msgs geometry_msgs visualization_msgs
  pcl_conversions pcl_msgs
  pogolm_interfaces OpenCV
)

target_compile_definitions(pogolm_api PRIVATE
  ${PCL_DEFINITIONS}
  PCL_NO_PRECOMPILE
  PGO_WITH_VISUAL_NO_MAIN
)

target_link_libraries(pogolm_api
  Eigen3::Eigen
  gtsam
  small_gicp::small_gicp
  kiss_matcher::kiss_matcher_core
  robin::robin
  yaml-cpp
  flann
  Threads::Threads
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${ZSTD_LINK}
)

install(TARGETS pogolm_api
  EXPORT  export_pogolm_api
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# ===================================================================
# Installation
# ===================================================================

install(TARGETS
  pose_graph
  loop_detector
  map_module
  test_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# ===================================================================
# ament exports  (so other packages can find pogolm_api)
# ===================================================================

ament_export_targets(export_pogolm_api HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  Threads
  rosidl_default_runtime
  nav_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  pcl_conversions
  pcl_msgs
  factor_graph_interfaces
  pogolm_interfaces
  Eigen3
  small_gicp
  kiss_matcher
  robin
  GTSAM
  OpenCV
)


ament_package()