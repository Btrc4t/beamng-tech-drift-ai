# Build the BeamNG.tech ROS2 bridge (libros2_beamng.so) from the upstream
# https://github.com/BeamNG/beamng-ros2-integration (the C++ bridge lives on
# the cpp branch) and install it into the game's BinLinux/ directory, where
# the game's ROS2 Editor loads it from.
#
#   source ./env-drift-tools            # sets BNG_HOME
#   cmake -S ros2_bridge_local -B ros2_bridge_local/build
#   cmake --build ros2_bridge_local/build
#
# Self-contained: the repo is downloaded into the build tree and built with
# colcon (the upstream build tool) into the build tree's install space, which
# also gets a local_setup.sh to source for the game's runtime environment;
# only the finished library is copied into the game. No pre-existing ROS
# workspace is needed. If the ROS environment changes, delete the build tree
# (stale CMake caches can otherwise poison the nested configure).
#
# The LD_PRELOAD shim in preload/ keeps working on the rebuilt library: it
# interposes rcl_publish, which the bridge calls for every publish.

cmake_minimum_required(VERSION 3.16)
project(beamng_ros2_bridge NONE)
include(ExternalProject)

set(BNG_ROS2_REPO "https://github.com/BeamNG/beamng-ros2-integration.git"
    CACHE STRING "Upstream repository to build")
set(BNG_ROS2_TAG "cpp" CACHE STRING "Branch, tag, or commit to build")

# --- where the library must land -------------------------------------------
if(NOT DEFINED ENV{BNG_HOME})
  message(FATAL_ERROR
    "BNG_HOME is not set. It must point at the BeamNG.tech installation "
    "(source drift-ai/env-drift-tools), so the bridge can be installed "
    "into $BNG_HOME/BinLinux/.")
endif()
get_filename_component(BNG_HOME "$ENV{BNG_HOME}" ABSOLUTE)
if(NOT IS_DIRECTORY "${BNG_HOME}/BinLinux")
  message(FATAL_ERROR
    "BNG_HOME=${BNG_HOME} does not contain a BinLinux/ directory; "
    "is it a BeamNG.tech installation?")
endif()
set(BNG_HOME_BIN_DIR "${BNG_HOME}/BinLinux")

# --- colcon (the upstream build tool) ---------------------------------------
find_program(COLCON_EXE NAMES colcon)
if(NOT COLCON_EXE)
  message(FATAL_ERROR
    "colcon not found in PATH; install it (python3-colcon-common-extensions "
    "on Ubuntu, or pip install colcon-common-extensions).")
endif()

# --- the ROS 2 prefix (ament_cmake, rclcpp, the rosidl generators) ---------
if(DEFINED ENV{CMAKE_PREFIX_PATH} AND NOT "$ENV{CMAKE_PREFIX_PATH}" STREQUAL "")
  set(ROS_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")
else()
  file(GLOB _ros_prefixes /opt/ros/*)
  if(NOT _ros_prefixes)
    message(FATAL_ERROR
      "No ROS 2 installation found: source your ROS 2 environment (which "
      "sets CMAKE_PREFIX_PATH) or install one under /opt/ros/<distro>.")
  endif()
  list(SORT _ros_prefixes)
  list(POP_BACK _ros_prefixes ROS_PREFIX_PATH)  # newest distro
endif()
list(GET ROS_PREFIX_PATH 0 ROS_PREFIX)

# This ROS install vendors some CMake packages (console_bridge) under
# <prefix>/opt/<name>_vendor; CMake's prefix search does not look there, so
# add them to the prefix path.
file(GLOB _ros_opt_prefixes "${ROS_PREFIX}/opt/*")
foreach(_ros_opt ${_ros_opt_prefixes})
  list(APPEND ROS_PREFIX_PATH "${_ros_opt}")
endforeach()

# --- the ROS 2 environment for the build ------------------------------------
# Sourcing the ROS setup script sets these; the ament/rosidl CMake scripts
# need them (they run the system python3 and query the ament resource index),
# so we reproduce the minimum here and keep anything the user already set.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_filename_component(ROS_DISTRO "${ROS_PREFIX}" NAME)
set(ROS_PY_SITE_DIR
    "${ROS_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
if(NOT IS_DIRECTORY "${ROS_PY_SITE_DIR}")
  file(GLOB _ros_py_dirs "${ROS_PREFIX}/lib/python*/site-packages")
  if(NOT _ros_py_dirs)
    message(FATAL_ERROR
      "No python site directory under ${ROS_PREFIX}/lib; "
      "is this a complete ROS 2 install?")
  endif()
  list(SORT _ros_py_dirs)
  list(GET _ros_py_dirs 0 ROS_PY_SITE_DIR)
endif()
# a colon-separated string for the environment (a CMake list would put ';'
# into the generated Makefile, where it is a command separator)
string(REPLACE ";" ":" ROS_PREFIX_PATH_STR "${ROS_PREFIX_PATH}")
set(ROS_ENV
    "env"
    "ROS_DISTRO=${ROS_DISTRO}"
    "CMAKE_PREFIX_PATH=${ROS_PREFIX_PATH_STR}:$ENV{CMAKE_PREFIX_PATH}"
    "AMENT_PREFIX_PATH=${ROS_PREFIX}:$ENV{AMENT_PREFIX_PATH}"
    "PYTHONPATH=${ROS_PY_SITE_DIR}:$ENV{PYTHONPATH}")

# --- layout inside the build tree -------------------------------------------
set(BNG_ROS2_SRC_DIR      "${CMAKE_BINARY_DIR}/src/beamng-ros2-integration")
set(BNG_ROS2_BUILD_BASE   "${CMAKE_BINARY_DIR}/colcon/build")
set(BNG_ROS2_LOG_BASE     "${CMAKE_BINARY_DIR}/colcon/log")
set(BNG_ROS2_INSTALL_BASE "${CMAKE_BINARY_DIR}/install")
set(BNG_ROS2_LIB          "${BNG_ROS2_INSTALL_BASE}/lib/beamng_ros2/libros2_beamng.so")
set(BNG_ROS2_LIB_DEST     "${BNG_HOME_BIN_DIR}/libros2_beamng.so")

# configure the post-install check with the destination path embedded
# (a -D argument would be mangled by the ExternalProject Makefile generator)
set(CHECK_FILE "${BNG_ROS2_LIB_DEST}")
configure_file("${CMAKE_CURRENT_LIST_DIR}/install_check.cmake"
               "${CMAKE_BINARY_DIR}/install_check.cmake" @ONLY)

message(STATUS "BNG_HOME:    ${BNG_HOME}")
message(STATUS "ROS prefix:  ${ROS_PREFIX_PATH}")
message(STATUS "install to:  ${BNG_HOME_BIN_DIR}/libros2_beamng.so")

# --- download, build, install -----------------------------------------------
# The cpp branch is a two-package ament workspace (beamng_msgs,
# beamng_ros2); colcon orders the build and produces the install space the
# game sources at runtime.
ExternalProject_Add(beamng_ros2_bridge
  GIT_REPOSITORY ${BNG_ROS2_REPO}
  GIT_TAG ${BNG_ROS2_TAG}
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  SOURCE_DIR ${BNG_ROS2_SRC_DIR}
  BINARY_DIR ${CMAKE_BINARY_DIR}/colcon
  DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/download"
  BUILD_ALWAYS TRUE
  # the repo root has no CMakeLists.txt (it is a colcon workspace)
  CONFIGURE_COMMAND ""
  BUILD_COMMAND
    ${ROS_ENV}
    ${COLCON_EXE} --log-base "${BNG_ROS2_LOG_BASE}" build
      --merge-install
      --packages-select beamng_msgs beamng_ros2
      --base-paths "${BNG_ROS2_SRC_DIR}"
      --build-base "${BNG_ROS2_BUILD_BASE}"
      --install-base "${BNG_ROS2_INSTALL_BASE}"
  INSTALL_COMMAND
    cmake -E copy_if_different "${BNG_ROS2_LIB}" "${BNG_HOME_BIN_DIR}/"
  COMMAND
    cmake -P "${CMAKE_BINARY_DIR}/install_check.cmake"
  COMMAND
    cmake -E echo "Installed ${BNG_ROS2_LIB} -> ${BNG_ROS2_LIB_DEST}"
  COMMAND
    cmake -E echo "Runtime env: source the ROS 2 setup plus ${BNG_ROS2_INSTALL_BASE}/local_setup.sh"
)

# The same install as an explicit command (the install step above already
# runs it as part of 'cmake --build'):
#   cmake --install ros2_bridge_local/build
# (a bracket-quoted string would keep ${...} literal in the generated
# cmake_install.cmake, where these variables are out of scope)
install(CODE "
  if(NOT EXISTS \"${BNG_ROS2_LIB}\")
    message(FATAL_ERROR \"${BNG_ROS2_LIB} not found; run 'cmake --build' first.\")
  endif()
  file(COPY \"${BNG_ROS2_LIB}\" DESTINATION \"${BNG_HOME_BIN_DIR}/\")
  message(STATUS \"Installed ${BNG_ROS2_LIB} -> ${BNG_HOME_BIN_DIR}/\")
")
