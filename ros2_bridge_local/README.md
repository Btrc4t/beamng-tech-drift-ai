# LD_PRELOAD shim for the BeamNG ROS2 bridge

The C++ bridge (`libros2_beamng.so`) comes from the upstream
[`BeamNG/beamng-ros2-integration`](https://github.com/BeamNG/beamng-ros2-integration)
repo. We keep the **original, unmodified** `.so` in the game and change its behaviour with a tiny `LD_PRELOAD` shim instead of patching the library.

## The problem the shim solves

The bridge's publish loop spins at 2 kHz (for low-latency control:
`rclcpp::spin_some` every iteration), and in the unpatched code it **publishes
the scalar sensors (State/Electrics/Time/Damage) every iteration** — at 2 kHz.
That has two costs:

1. **In the game:** serialising + DDS-publishing the ~130-field
   `ElectricsSensor` (and the other scalars) at 2 kHz pins a core and drags the
   physics tick rate down (observed: the hybrid env stepped at
   **~2100 ms/step**, the game at a few percent of real time).
2. **In the env:** the rclpy spin thread is flooded with ~6k msg/s and pins a
   second core.

Capping the scalar publish at **120 Hz** (the game's render-loop rate, so the
sensor stream stays as fresh as the main loop allows) fixes both: the hybrid
env steps at **~37 ms/step** and the game runs at or above real time. The 2 kHz
loop still spins for control, so control latency is unchanged. The LiDAR is
unaffected: in the env it is a blob sensor read from shared memory, not the
bridge's topic.

## How the shim works

`preload/bng_ros2_preload.c` interposes `rcl_publish` — the single C entry
point every rclcpp publish goes through (verified: `librcl.so` exports it
unversioned, `librclcpp.so` references it undefined). For each publisher it
tracks the last publish time and drops frames that arrive sooner than the cap
interval (returning `RCL_RET_OK`, so the bridge sees a normal publish).

Gotcha handled: `dlsym(RTLD_NEXT, "rcl_publish")` only sees the *global* symbol
scope, but rclpy (Python extension) and the game's bridge plugin load
`librcl.so` `RTLD_LOCAL`. The shim therefore falls back to finding the loaded
`librcl.so` in `/proc/self/maps` and `dlopen(RTLD_NOLOAD)`-ing it for a direct
handle. (Without this, every publish silently no-ops and the bridge's topics
never deliver.)

## Files

- `preload/bng_ros2_preload.c` — the shim (plain C, no dependencies beyond
  libc/libdl, no constructor: it initialises on the first `rcl_publish` call).
- `preload/libbng_ros2_preload.so` — the built shim (rebuild with `./build.sh`).
- `build.sh` — one-shot build.
- `CMakeLists.txt` — rebuilds the upstream bridge into `$BNG_HOME/BinLinux/`
  (see "Rebuilding the upstream bridge").

## Build + export

```bash
./build.sh
```

The export lives in `drift-ai/env-drift-tools` (sourced before running the
env/benchmarks):

```sh
export LD_PRELOAD=.../drift-ai/ros2_bridge_local/preload/libbng_ros2_preload.so
```

The game is launched by the env (beamngpy), so it inherits `LD_PRELOAD` from
the launching process. The shim is inert in processes that never call
`rcl_publish`.

## Rebuilding the upstream bridge

`CMakeLists.txt` downloads the upstream repo (the `cpp` branch) into the
build tree, builds `libros2_beamng.so` with colcon, and installs it into
`$BNG_HOME/BinLinux/`. It is self-contained (no pre-existing ROS workspace
needed); the build tree's `install/` space also gets a `local_setup.sh` to
source for the game's runtime environment. It fails at configure time if
`BNG_HOME` is not set (or has no `BinLinux/`), and picks the ROS 2 prefix
from `CMAKE_PREFIX_PATH` (sourced ROS) or `/opt/ros/<distro>`.

```bash
source ../env-drift-tools
cmake -S ros2_bridge_local -B ros2_bridge_local/build
cmake --build ros2_bridge_local/build
```

The build's install step copies the finished library into
`$BNG_HOME/BinLinux/` and verifies it landed (so `cmake --build` is
automatic). The same install can be run on its own, e.g. to restore the
library after deleting it:

```bash
cmake --install ros2_bridge_local/build
```

The install is not the whole runtime story: the game also needs the ROS 2
environment plus the install space (the `beamng_msgs` typesupport the bridge
loads at runtime). Source `env-ros2.sh` in the shell that launches the game
(or the env):

```bash
source ros2_bridge_local/env-ros2.sh
```

Use it to restore the original library after patching it, or to update it
(`-DBNG_ROS2_TAG=<branch|tag|commit>` selects the revision). The rebuilt
library is the unmodified upstream one, so the `LD_PRELOAD` shim above is
still what changes its behaviour. If the ROS environment changes, delete the
build tree first (stale CMake caches can poison the nested configure).

## Tuning

- `BNG_ROS2_PRELOAD_HZ` — the cap in Hz (default **120**; `0` disables it).
- `BNG_ROS2_PRELOAD_EXCLUDE` — comma-separated topic substrings exempt from the
  cap (e.g. a high-rate blob topic if you ever publish one >120 Hz).

The shim prints a one-line report on first use and a per-topic
`pub=`/`drop=` count every 5 s to stderr. Note the env launches the game with
stderr on `/dev/null`, so those reports are only visible when you launch the
game from a terminal yourself.
