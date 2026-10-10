import random
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage, Electrics, Lidar

_T0 = time.perf_counter()
def _dbg(msg):
    print(f"[{time.perf_counter() - _T0:8.2f}s] {msg}", flush=True)

# general limits
DAMAGE_LIMIT        = 1000
EPISODE_TIMEOUT     = 64.0
FPS                 = 20   # physics tick rate (fixed by the engine, step(1) = 50 ms of sim time)
RENDER_FPS          = 120  # render loop rate: how often the game's main loop wakes up to process socket requests
FPS_FACTOR          = 1
MIN_SPEED           = 10.0

# drift band, mirrored from the game's driftOptions (drift.lua:86-98)
DRIFT_MIN_ANGLE    = 7.0    # deg, below this is not a drift
DRIFT_MAX_ANGLE    = 165.0  # deg, above this is a spin
DRIFT_MIN_AIRSPEED = 4.0    # kph, minimum speed to count as drifting

# sensor value limits
IMU_ACC_MAX      = 20.0   # m/s^2  (~2 g per axis, covers hard drift loads)
IMU_ANGVEL_MAX   = 10.0   # rad/s
ANGLE_MAX        = 180.0  # drift angle, deg -> [-1, 1]
AIRSPEED_MAX_KPH = 300.0  # kph -> [0, 1]
BEV_N            = 64     # LiDAR BEV grid size
Z_LO, Z_HI       = -1.0, 1.0   # (m) road band: squeeze the scale so the raised divider pops
# LiDAR: full 360-degree scan, attached to the ego car's roof
# We rotate the display / observation to the travel direction.
LIDAR_RES_V      = 64                   # vertical resolution (number of horizontal slices)
LIDAR_V_ANGLE    = 26.9                 # vertical aperture (deg)
LIDAR_HZ         = FPS * FPS_FACTOR     # lidar rotation frequency (Hz)
LIDAR_MAX_DIST   = 40.0                 # max range (m)
LIDAR_DENSITY    = 50                   # point density (1 = dense, 100 = sparse)

DRIFT_CARS_6SPEED = [
    {"model": "bx", "part_config": "pro_drift_M"},
    {"model": "bastion", "part_config": "drift"},
    {"model": "etk800", "part_config": "858_drift_M"},
    {"model": "etkc", "part_config": "kc8_drift_M"},
    {"model": "etki", "part_config": "drift"},
    {"model": "fullsize", "part_config": "drift"},
    {"model": "sbr", "part_config": "drift"},
]

def drift_from_state(state):
    """Per-vehicle drift telemetry from the State sensor, no game round trip.

    Mirrors the game's formulas (drift.lua:230-231): the drift angle is the
    absolute yaw difference between the velocity direction and the heading,
    signed by vel . (up x dir). The game's drift score is player-centric, so
    for a group of vehicles we compute a per-frame score proxy here instead.
    """
    vel = np.asarray(state["vel"], float)
    dirv = np.asarray(state["dir"], float)
    up = np.asarray(state["up"], float)
    speed_kph = float(np.linalg.norm(vel) * 3.6)
    if speed_kph < 1.0:
        return {"angle": 0.0, "airSpeed": speed_kph,
                "isDrifting": False, "scoreAdded": 0.0}

    d = dirv / np.linalg.norm(dirv)
    v = vel / np.linalg.norm(vel)
    # the game measures slip as a horizontal-plane (x, y) yaw difference via
    # atan2, not the full 3D angle (drift.lua:201-230); mirror that exactly so
    # the parity holds on slopes, where the 3D arccos would pick up pitch
    yaw_diff = np.arctan2(d[1], d[0]) - np.arctan2(v[1], v[0])
    yaw_diff = (yaw_diff + np.pi) % (2.0 * np.pi) - np.pi
    angle = float(abs(np.degrees(yaw_diff)))
    sign = float(np.sign(np.dot(vel, np.cross(up, dirv))))
    is_drifting = (DRIFT_MIN_ANGLE < angle < DRIFT_MAX_ANGLE
                   and speed_kph > DRIFT_MIN_AIRSPEED)
    # per-frame score proxy, scaled so the reward stays in a similar range
    score_added = ((angle / DRIFT_MAX_ANGLE)
                   * (speed_kph / AIRSPEED_MAX_KPH) * 100.0
                   if is_drifting else 0.0)
    return {"angle": angle * sign, "airSpeed": speed_kph,
            "isDrifting": is_drifting, "scoreAdded": score_added}

def car_rotation(state):
    """Rotation matrix whose COLUMNS are the car's [forward, left, up] axes in
    world space, built from the (normalised) dir and up vectors."""
    d = np.asarray(state["dir"], float); d /= np.linalg.norm(d)
    u = np.asarray(state["up"], float);  u /= np.linalg.norm(u)
    l = np.cross(u, d)                     # left
    l /= np.linalg.norm(l)
    return np.stack([d, l, u], axis=1)     # columns [fwd, left, up]

def lidar_bev(pc, state, drift, n=BEV_N, range_m=LIDAR_MAX_DIST):
    """Bird's eye view height grid: the car is in the center, the grid spans
    [-range_m, +range_m] around it, values are the highest LiDAR return per
    cell clamped to [0, 1] (0.5 = road level)."""
    out = np.full((n, n), np.nan, np.float32)
    if pc.size == 0:
        return out
    R = car_rotation(state)
    p = (pc - np.asarray(state["pos"], float)) @ R   # [fwd, left, up]
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    slip = float(np.deg2rad(drift.get("angle", 0.0)))
    th = 0.5 * np.pi - slip
    c, s = np.cos(th), np.sin(th)
    dx, dy = x * c - y * s, x * s + y * c
    ix = ((dx + range_m) * n / (2 * range_m)).astype(int)
    iy = ((dy + range_m) * n / (2 * range_m)).astype(int)
    m = (ix >= 0) & (iy >= 0) & (ix < n) & (iy < n)
    cell = (n - 1 - iy[m]) * n + ix[m]
    h = np.clip((z[m] - Z_LO) / (Z_HI - Z_LO), 0.0, 1.0)
    # max height per cell in one O(N) pass via an unbuffered ufunc;
    # ~3-4x faster than argsort + unique + reduceat on a full point cloud
    out_flat = np.full(n * n, -np.inf, np.float32)
    np.maximum.at(out_flat, cell, h)
    out_flat[out_flat == -np.inf] = np.nan
    return out_flat.reshape(n, n)

class BeamNGTechEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS * FPS_FACTOR}

    def __init__(self, render_mode=None):
        self.action_space = spaces.Dict({
            # [steering, throttle/brake, parkingbrake]
            "controls": spaces.Box(
                #              s     t/b   p
                low=np.array([-1.0, -1.0, -1.0]),
                high=np.array([1.0,  1.0,  1.0]),
                dtype=np.float32,
            ),
        })

        # The full observation: one Dict space, one key per sensor channel.
        # Every value is a float already normalized into a bounded range.
        self.observation_space = spaces.Dict({
            "rpm":         spaces.Box(low=0.0,             high=1.0,            shape=(1,),           dtype=np.float32),
            "imu_acc":     spaces.Box(low=-1.0,            high=1.0,            shape=(3,),           dtype=np.float32),
            "imu_angvel":  spaces.Box(low=-1.0,            high=1.0,            shape=(3,),           dtype=np.float32),
            "drift_angle": spaces.Box(low=-1.0,            high=1.0,            shape=(1,),           dtype=np.float32),
            "airspeed":    spaces.Box(low=0.0,             high=1.0,            shape=(1,),           dtype=np.float32),
            "is_shifting": spaces.Box(low=0.0,             high=1.0,            shape=(1,),           dtype=np.float32),
            "bev":         spaces.Box(low=0.0,             high=1.0,            shape=(BEV_N, BEV_N), dtype=np.float32),
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # relevant information for a previous frame, for reward calculation
        self.previous = dict(
            speed = 0.0,
            score = 0.0,
            damage = 0.0
        )

        # relevant information for a current frame, for reward calculation
        self.current = dict(
            speed = 0.0,
            score = 0.0,
            damage = 0.0
        )

        # running total of the self-computed drift score for the episode
        self._score_total = 0.0
        # simulation time at the last reset, for the episode timeout
        self._t0 = 0.0

        # State snapshot from the previous step, for the derived IMU
        self._prev_state = None
        # derived IMU channels for the current step (see _derive_imu)
        self._imu_acc = np.zeros(3, dtype=np.float32)
        self._imu_angvel = np.zeros(3, dtype=np.float32)
        # sim time of the last completed step, for the stale-read check
        self._last_step_time = 0.0
        # in-flight sensor request responses (see _send_sensor_request)
        self._sensor_engine_resp = None
        self._sensor_vehicle_resp = None

        self.window = None

    def _get_obs(self):
        # pure computation from the cached sensor data: no round trips here,
        # step() (or reset()) polls before calling this
        e = self.ego.sensors["electrics"]
        drift = drift_from_state(self.ego.state)

        acc, angvel = self._imu_acc, self._imu_angvel
        if drift["airSpeed"] <= 1.0:
            acc = np.zeros(3, dtype=np.float32)
            angvel = np.zeros(3, dtype=np.float32)

        bev = np.nan_to_num(lidar_bev(self._poll_lidar_points(),
                                      self.ego.state,
                                      drift,
                                      n=BEV_N),
                            nan=0.5)

        return {
            "rpm":         np.clip(np.array([e["rpm"] / e["maxrpm"]], np.float32), 0.0, 1.0),
            "imu_acc":     acc,
            "imu_angvel":  angvel,
            "drift_angle": np.clip(np.array([drift["angle"] / ANGLE_MAX], np.float32), -1.0, 1.0),
            "airspeed":    np.array([min(AIRSPEED_MAX_KPH, drift["airSpeed"]) / AIRSPEED_MAX_KPH], np.float32),
            "is_shifting": np.array([1.0 if e["is_shifting"] else 0.0], np.float32),
            "bev":         bev,
        }

    def _send_sensor_request(self):
        """Send the bundled sensor request without waiting (send half of poll)."""
        engine_reqs, vehicle_reqs = self.ego.sensors._encode_requests(
            tuple(self.ego.sensors._sensors.keys()))
        self._sensor_engine_resp = (self.beamng._send(engine_reqs)
                                    if engine_reqs["sensors"] else None)
        self._sensor_vehicle_resp = (self.ego._send(vehicle_reqs)
                                     if vehicle_reqs["sensors"] else None)

    def _recv_sensor_data(self):
        """Receive and decode a previously sent sensor request (recv half of poll)."""
        sensor_data = dict()
        if self._sensor_engine_resp is not None:
            sensor_data.update(self._sensor_engine_resp.recv("SensorData")["data"])
        if self._sensor_vehicle_resp is not None:
            sensor_data.update(self._sensor_vehicle_resp.recv("SensorData")["data"])
        result = self.ego.sensors._decode_response(sensor_data)
        for sensor, data in result.items():
            self.ego.sensors._sensors[sensor].replace(data)
        self._sensor_engine_resp = None
        self._sensor_vehicle_resp = None

    def _poll_sensors(self):
        """Blocking sensor poll; off the hot path (reset, stale-read fallback)."""
        self._send_sensor_request()
        self._recv_sensor_data()

    def _poll_lidar_points(self):
        """Point cloud from shared memory, skipping the colour buffer (unused).

        Mirrors Lidar._convert_binary_to_array: in streaming mode the game
        writes the point count as the last float of the buffer.
        """
        shmem = self.lidar.point_cloud_shmem
        buf = np.frombuffer(shmem.read(self.lidar.point_cloud_shmem_size),
                            dtype=np.float32)
        n = int(buf[-1])
        if n <= 0 or n * 3 > len(buf) - 1:
            return np.empty((0, 3), dtype=np.float32)
        return buf[: 3 * n].reshape(-1, 3).copy()

    def _derive_imu(self, state):
        """Accel and angular velocity from two consecutive State snapshots,
        replacing the AdvancedIMU poll (one less round trip per step).

        Channels are in the car frame with the [fwd, left, up] axis order:
        acc = [longitudinal, lateral, vertical] (kinematic d(vel)/dt, so no
        gravity offset in the vertical channel, unlike the raw IMU),
        angvel = [roll, pitch, yaw] rates in rad/s.
        """
        prev = self._prev_state
        # copy the fields: ego.state is mutated in place by the next poll, so
        # a reference would make prev identical to state (dt = 0)
        self._prev_state = {
            "vel": np.array(state["vel"], float),
            "dir": np.array(state["dir"], float),
            "up": np.array(state["up"], float),
            "time": float(state["time"]),
        }
        if prev is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        dt = float(state["time"]) - float(prev["time"])
        if dt < 0.04:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        R = car_rotation(state)
        v_car = R.T @ np.asarray(state["vel"], float)
        v_car_prev = car_rotation(prev).T @ np.asarray(prev["vel"], float)
        acc = (v_car - v_car_prev) / dt

        dR = R.T @ car_rotation(prev)
        skew = 0.5 * np.array([
            dR[2, 1] - dR[1, 2],
            dR[0, 2] - dR[2, 0],
            dR[1, 0] - dR[0, 1],
        ])
        theta = float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
        if theta > 1e-4:
            angvel = skew * (theta / np.sin(theta)) / dt
        else:
            angvel = skew / dt

        acc = np.clip(acc / IMU_ACC_MAX, -1.0, 1.0)
        angvel = np.clip(angvel / IMU_ANGVEL_MAX, -1.0, 1.0)
        return acc.astype(np.float32), angvel.astype(np.float32)

    def _update_reward_state(self):
        """Update the reward bookkeeping from fresh sensor data. Called in
        step() before _compute_reward, so the reward reflects the tick that
        just ran (previously _get_obs did this after the reward computation,
        leaving the reward one step stale)."""
        drift = drift_from_state(self.ego.state)
        self.current["speed"] = drift["airSpeed"]
        self._score_total += drift["scoreAdded"]
        self.current["score"] = self._score_total
        dmg = self.ego.sensors["damage"]
        if isinstance(dmg, dict):
            self.current["damage"] = dmg.get("damage", 0.0)
        else:
            self.current["damage"] = 0.0

    def _get_info(self):
        return {}

    def _setup_sensors(self):
        # realistic manual gearbox dropped, arcade set
        self.ego.set_shift_mode("arcade")

        # Sensors
        self.electrics = Electrics()
        self.damage = Damage()

        self.ego.sensors.attach("electrics", self.electrics)
        self.ego.sensors.attach("damage", self.damage)

        # no AdvancedIMU: acc/angvel are derived from the State sensor in
        # _derive_imu, which saves one blocking round trip per step

        self.lidar = Lidar(
            "roof",
            self.beamng,
            self.ego,                              # attached to the car -> rides with it
            requested_update_time= 1.0/LIDAR_HZ,   # 20 Hz
            pos=(0.0, 0.0, 1.5),                   # on the roof, in vehicle space
            vertical_resolution=LIDAR_RES_V,
            vertical_angle=LIDAR_V_ANGLE,
            frequency=LIDAR_HZ,
            max_distance=LIDAR_MAX_DIST,
            density=LIDAR_DENSITY,
            is_360_mode=True,                      # full 360 degrees
            is_rotate_mode=False,
            is_using_shared_memory=True,
            is_visualised=False,
            is_streaming=True,
        )

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        _dbg("reset: start")

        # start BeamNG.drive on the first reset (works with reset-first ordering)
        if self.window is None:
            self._first_time_setup()
        _dbg("reset: first_time_setup done")

        # unpause physics / time
        self.beamng.control.resume()
        time.sleep(1)
        _dbg("reset: resumed")
        # vehicle switch
        self.scenario.remove_vehicle(self.ego)
        self.beamng.scenario.restart()
        _dbg("reset: scenario restarted")
        car = DRIFT_CARS_6SPEED[self.np_random.integers(len(DRIFT_CARS_6SPEED))]
        if options is not None and "car" in options:
            car = options["car"]
        waypoint_keys = list(self.waypoints.keys())
        key = waypoint_keys[self.np_random.integers(len(waypoint_keys))]
        self.ego = Vehicle("ego", license="DRIFT", color="Red", **car)
        self.scenario.add_vehicle(
            self.ego, pos=self.waypoints[key].pos, rot_quat=self.waypoints[key].rot
        )
        _dbg("reset: vehicle added")

        self._setup_sensors()
        _dbg("reset: sensors set up")
        time.sleep(1)

        # pause physics / time
        self.beamng.control.pause()
        _dbg("reset: paused, polling sensors")

        # _get_obs no longer polls; refresh the cached sensor data here
        self._poll_sensors()
        _dbg("reset: sensors polled")
        self._prev_state = None
        self._imu_acc = np.zeros(3, dtype=np.float32)
        self._imu_angvel = np.zeros(3, dtype=np.float32)
        self._last_step_time = self.ego.state["time"]

        obs = self._get_obs()
        # anchor the episode clock and score to the freshly spawned car
        self._score_total = 0.0
        self._t0 = self.ego.state["time"]
        info = self._get_info()
        _dbg("reset: done")

        return obs, info

    def step(self, action):
        steering, combined, parkingbrake = (float(value) for value in action["controls"])
        brake = 0.0
        throttle = 0.0
        if combined > 0.0:
            throttle = combined
        else:
            brake = -combined

        # Store the control input and wait for the game to accept
        # it. This must complete before the step so the tick runs with the
        # new input regardless of the order the game drains its sockets.
        r_control = self.ego._send(dict(
            type="Control",
            steering=steering,
            throttle=throttle,
            brake=brake,
            parkingbrake=0.0 if parkingbrake < 0.0 else parkingbrake,
        ))
        r_control.recv("Controlled")

        # One tick plus the sensor read, sent as one batch. No ack
        # on the Step: the game processes requests in order and the tick runs
        # inside the Step handler, so by the time the sensor data arrives the
        # tick is done. The state["time"] check below catches a stale read
        # (vehicle socket drained before the engine socket in the same frame)
        # and re-polls once.
        self.beamng._send(dict(type="Step", count=1, ack=False))
        self._send_sensor_request()
        self._recv_sensor_data()
        if self.ego.state["time"] - self._last_step_time < 0.04:
            self._poll_sensors()

        # reward and failure from fresh data, then build the observation
        self._update_reward_state()
        self._imu_acc, self._imu_angvel = self._derive_imu(self.ego.state)
        terminated, _ = self._check_failure()
        reward, _ = self._compute_reward()
        observation = self._get_obs()
        self._last_step_time = self.ego.state["time"]
        truncated = (self.ego.state["time"] - self._t0) > EPISODE_TIMEOUT
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        # the game takes care of this without any work.
        pass

    def _first_time_setup(self):
        _dbg("setup: connecting to game")
        self.beamng = BeamNGpy("localhost", 25252)
        self.beamng.open()
        _dbg("setup: connected, loading scenario")

        self.scenario = Scenario("west_coast_usa", "Gymnasium")

        self.scenario.make(self.beamng)
        self.beamng.scenario.load(self.scenario)
        _dbg("setup: scenario loaded")

        self.waypoints = self.scenario.find_waypoints()
        self.waypoints = {w.name: w for w in self.waypoints}
        _dbg(f"setup: {len(self.waypoints)} waypoints")

        # done in reset
        car = random.choice(DRIFT_CARS_6SPEED)
        key = random.choice(list(self.waypoints.keys()))

        self.ego = Vehicle("ego", license="DRIFT", color="Red", **car)
        self.scenario.add_vehicle(
            self.ego, pos=self.waypoints[key].pos, rot_quat=self.waypoints[key].rot
        )
        _dbg("setup: vehicle added")

        self.beamng.scenario.start()
        _dbg("setup: scenario started")

        self.window = [640, 480]

        # Disable driving assists
        self.beamng.queue_lua_command('''
            settings.setValue("steeringAutocenterEnabled", false)
            settings.setValue("steeringSlowdownEnabled", false)
            settings.setValue("steeringStabilizationEnabled", false)
            settings.setValue("steeringUndersteerReductionEnabled", false)
            settings.setValue("steeringLimitEnabled", false)
            settings.setValue("autoClutch", true)
            settings.setValue("autoThrottle", true)
            settings.setValue("absBehavior", "off")
        ''')

        # Fixed windowed resolution
        self.beamng.settings.change("GraphicDisplayMode", "Windowed")
        self.beamng.settings.change("GraphicDisplayResolutions", " ".join(map(str, self.window)))
        self.beamng.settings.apply_graphics()

        # set deterministic
        self.beamng.queue_lua_command('be:setPhysicsDeterministic(true)')

        # run the render loop faster than the physics tick so the game's
        # main loop wakes up more often to process socket requests (each
        # blocking round trip otherwise waits up to a full frame period).
        # The physics stays on its fixed 20 Hz tick - step(1) still advances
        # exactly 50 ms of sim time - but decoupling the render loop from the
        # tick means the simulation can run FASTER than real time when the
        # game is resumed. That is fine here: we step explicitly and measure
        # wall-clock FPS, not sim speed.
        self.beamng.queue_lua_command(f"settings.setValue('fpsLimit', {RENDER_FPS})")
        self.beamng.queue_lua_command('settings.setValue("fpsLimitEnabled", true)')
        self.beamng.queue_lua_command('be:setPhysicsSpeedFactor(1)')

        # settle
        time.sleep(1)

        # pause physics / time
        self.beamng.control.pause()
        _dbg("setup: done (paused)")

    def close(self):
        if self.window is not None:
            self.beamng.close()

    def _check_failure(self):
        """Return (failed: bool, reason: str) from the Damage sensor poll."""

        dmg = self.ego.sensors["damage"]
        if not isinstance(dmg, dict):
            return False, ""
        overall = dmg.get("damage", 0.0)
        tire_flat = dmg.get("lowpressure")

        if overall > DAMAGE_LIMIT:
            return True, f"damage={overall:.3f}"
        if tire_flat:
            return True, f"tire flat (lowpressure={tire_flat})"
        return False, ""

    def _compute_reward(self):
        """Return (total, components_dict). All values are floats."""
        comps = {}

        # drift score (self-computed from State) - a positive component
        drift_delta = max(0.0, self.current["score"] - self.previous["score"])
        # 1 percent of the score will be a reward
        drift_reward = min(0.8, drift_delta * 0.01)

        # acceleration should be rewarded, it increases the chance of drifting
        speed_delta = max(0.0, self.current["speed"] - self.previous["speed"])
        # 1 percent of the speed delta will be a reward
        speed_reward = min(0.2, speed_delta * 0.01)

        # car's damage - a negative component
        dmg_delta = max(0.0, self.current["damage"] - self.previous["damage"])
        # 0.1 percent of the damage will be a punishment
        dmg_reward = max(-1.0, -dmg_delta * 0.001)

        # speed under 10kph is punished, 5% of difference to minimum target
        slow_reward = -0.05 * (MIN_SPEED - self.current["speed"]) if self.current["speed"] < MIN_SPEED else 0;

        # copy, not a reference: _update_reward_state mutates self.current in
        # place, so a reference would make every delta zero after the first step
        self.previous = dict(self.current)
        # save components too - for plotting or debugging - not strictly necessary
        comps["drift_delta"] = drift_delta
        comps["drift_reward"] = drift_reward
        comps["dmg_delta"] = dmg_delta
        comps["dmg_reward"] = dmg_reward
        comps["speed_delta"] = speed_delta
        comps["speed_reward"] = speed_reward
        comps["slow_reward"] = slow_reward

        total = drift_reward + dmg_reward + speed_reward + slow_reward
        return total, comps
