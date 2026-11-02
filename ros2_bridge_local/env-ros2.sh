#!/bin/sh
# Source this to prepare a shell for running the game with the C++ ROS2
# bridge: the ROS 2 distribution plus the install space produced by
# CMakeLists.txt (the beamng_msgs typesupport libraries the bridge loads at
# runtime).
#
#   source ros2_bridge_local/env-ros2.sh
#
# Then launch the game (or the env) from that shell; it inherits the
# environment. The bridge itself is installed into $BNG_HOME/BinLinux/ by
# the build:
#   cmake -S ros2_bridge_local -B ros2_bridge_local/build
#   cmake --build ros2_bridge_local/build

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [ -n "${CMAKE_PREFIX_PATH:-}" ]; then
  _ros_prefix="${CMAKE_PREFIX_PATH%%:*}"
else
  for _d in /opt/ros/*/; do
    _ros_prefix="${_d%/}"
  done
fi

if [ ! -f "${_ros_prefix}/setup.bash" ]; then
  echo "env-ros2: no ROS 2 setup at ${_ros_prefix}; set CMAKE_PREFIX_PATH or install ROS 2" >&2
  return 1 2>/dev/null || exit 1
fi
if [ ! -f "${_here}/build/install/local_setup.sh" ]; then
  echo "env-ros2: ${_here}/build/install/local_setup.sh is missing; build the bridge first:" >&2
  echo "  cmake -S ${_here} -B ${_here}/build && cmake --build ${_here}/build" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
. "${_ros_prefix}/setup.bash"
# shellcheck disable=SC1091
. "${_here}/build/install/local_setup.sh"
