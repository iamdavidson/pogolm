#pragma once
#include <rclcpp/rclcpp.hpp>

// ANSI helpers
#define ANSI_ESC        "\033["
#define ANSI_RESET      ANSI_ESC "0m"
#define ANSI_BOLD       ANSI_ESC "1m"
#define ANSI_DIM        ANSI_ESC "2m"

#define ANSI_FG_256(n)  ANSI_ESC "38;5;" #n "m"
#define ANSI_FG_RGB(r,g,b) ANSI_ESC "38;2;" #r ";" #g ";" #b "m"

// Node color tags
#define NODE_POSE_GRAPH    \
  ANSI_BOLD ANSI_FG_RGB(154,199,191) "[po" \
           ANSI_FG_RGB(255,230,247) "se_" \
           ANSI_FG_RGB(255,255,153) "graph]" ANSI_RESET

#define NODE_LOOP_DETECTOR \
  ANSI_BOLD ANSI_FG_RGB(255,192,192) "[loop_detector]" ANSI_RESET

#define NODE_MAP_MODULE    \
  ANSI_BOLD ANSI_FG_RGB(155,230,170) "[map_module]" ANSI_RESET

#define NODE_RANGE_FILTER  \
  ANSI_BOLD ANSI_FG_RGB(255,181,112) "[range_filter]" ANSI_RESET

#define NODE_GENERIC       \
  ANSI_BOLD ANSI_FG_256(244) "[node]" ANSI_RESET

// Tag selection 
#ifndef LOG_NODE_TAG
  #define LOG_NODE_TAG NODE_GENERIC
#endif

// Method
#define TAG_ODOM        ANSI_BOLD ANSI_FG_256(39)  "[ODOM]"        ANSI_RESET 
#define TAG_LIDAR       ANSI_BOLD ANSI_FG_256(81)  "[LIDAR]"       ANSI_RESET   
#define TAG_TIMEOUT     ANSI_BOLD ANSI_FG_256(214) "[ODOM_TO]"     ANSI_RESET
#define TAG_CLEANER     ANSI_BOLD ANSI_FG_256(214) "[CLEANER]"     ANSI_RESET  
#define TAG_ADD_POSE    ANSI_BOLD ANSI_FG_256(46)  "[ADD_POSE]"    ANSI_RESET  
#define TAG_LOOP_EDGE   ANSI_BOLD ANSI_FG_256(199) "[LOOP_EDGE]"   ANSI_RESET  
#define TAG_OPTIM       ANSI_BOLD ANSI_FG_256(226) "[OPTIM]"       ANSI_RESET  
#define TAG_CORE        ANSI_BOLD ANSI_FG_256(208) "[CORE]"         ANSI_RESET 
#define TAG_NEAR        ANSI_BOLD ANSI_FG_256(112) "[NEAR_CORE]"    ANSI_RESET  
#define TAG_POGOLM      ANSI_BOLD ANSI_FG_256(81)  "[POGOLM]"     ANSI_RESET  
#define TAG_LOOP_DET    ANSI_BOLD ANSI_FG_256(33)  "[LOOP_DET]"    ANSI_RESET  
#define TAG_POLI      ANSI_BOLD ANSI_FG_256(93)    "[POLICY]"      ANSI_RESET  
#define TAG_LM          ANSI_BOLD ANSI_FG_256(160) "[LANDMARK]"    ANSI_RESET  
#define TAG_KDTREE      ANSI_BOLD ANSI_FG_256(51)  "[KDTREE]"      ANSI_RESET  
#define TAG_DEBUG       ANSI_BOLD ANSI_FG_256(244) "[DEBUG]"       ANSI_RESET  
#define TAG_LOOP_CAND   ANSI_BOLD ANSI_FG_256(135) "[LOOP_EDGE_TRACKER]" ANSI_RESET  
#define TAG_VERIFY_OK   ANSI_BOLD ANSI_FG_256(46)  "[VERIFY_OK]"   ANSI_RESET 
#define TAG_VERIFY_NO   ANSI_BOLD ANSI_FG_256(196) "[VERIFY_NO]"   ANSI_RESET  

// Logging macros
#define LOGI_ODOM(logger, fmt, ...)        RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_ODOM       " " fmt, ##__VA_ARGS__)
#define LOGI_LIDAR(logger, fmt, ...)       RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_LIDAR      " " fmt, ##__VA_ARGS__)
#define LOGI_CLEANER(logger, fmt, ...)       RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_CLEANER      " " fmt, ##__VA_ARGS__)
#define LOGW_TIMEOUT(logger, fmt, ...)     RCLCPP_WARN( (logger), LOG_NODE_TAG " " TAG_TIMEOUT    " " fmt, ##__VA_ARGS__)
#define LOGI_ADD_POSE(logger, fmt, ...)    RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_ADD_POSE   " " fmt, ##__VA_ARGS__)
#define LOGI_LOOP(logger, fmt, ...)        RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_LOOP_EDGE  " " fmt, ##__VA_ARGS__)
#define LOGI_OPT(logger, fmt, ...)         RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_OPTIM      " " fmt, ##__VA_ARGS__)
#define LOGI_NEAR(logger, fmt, ...)        RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_NEAR       " " fmt, ##__VA_ARGS__)
#define LOGI_CORE(logger, fmt, ...)        RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_CORE        " " fmt, ##__VA_ARGS__)
#define LOGI_LOOP_DET(logger, fmt, ...)    RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_LOOP_DET   " " fmt, ##__VA_ARGS__)
#define LOGI_POLI(logger, fmt, ...)      RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_POLI     " " fmt, ##__VA_ARGS__)
#define LOGI_LM(logger, fmt, ...)          RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_LM         " " fmt, ##__VA_ARGS__)
#define LOGI_KD(logger, fmt, ...)          RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_KDTREE     " " fmt, ##__VA_ARGS__)
#define LOGD_ANY(logger, fmt, ...)         RCLCPP_DEBUG((logger), LOG_NODE_TAG " " TAG_DEBUG      " " fmt, ##__VA_ARGS__)
#define LOGI_LOOP_TRACK(logger, fmt, ...)  RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_LOOP_CAND  " " fmt, ##__VA_ARGS__)
#define LOGI_VERIFY_OK(logger, fmt, ...) RCLCPP_INFO( (logger), LOG_NODE_TAG " " TAG_VERIFY_OK " " fmt, ##__VA_ARGS__)
#define LOGW_VERIFY_NO(logger, fmt, ...) RCLCPP_ERROR( (logger), LOG_NODE_TAG " " TAG_VERIFY_NO " " fmt, ##__VA_ARGS__)
#define LOGI_POGOLM(logger, fmt, ...) RCLCPP_INFO( (logger), LOG_NODE_TAG " " fmt, ##__VA_ARGS__)
#define LOGE_ANY(logger, fmt, ...)         RCLCPP_ERROR((logger), LOG_NODE_TAG " " TAG_DEBUG      " " fmt, ##__VA_ARGS__)
 