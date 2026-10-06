import random
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import AdvancedIMU, Damage, Electrics, Lidar, Timer

# general limits
DAMAGE_LIMIT        = 1000
EPISODE_TIMEOUT     = 64.0
FPS                 = 20
FPS_FACTOR          = 1
MIN_SPEED           = 10.0

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

        self.window = None

    def _get_obs(self):
        self.ego.sensors.poll()
        e = self.ego.sensors["electrics"]
        imu_data = self.imu.poll()
        drift = self._read_drift()

        # bulk mode returns a list of readings
        if isinstance(imu_data, list):
            imu_data = imu_data[-1]

        acc    = np.clip(np.asarray(imu_data["accRaw"], np.float32) / IMU_ACC_MAX, -1.0, 1.0)    if drift["airSpeed"] > 1.0 else np.zeros(3, dtype=np.float32)
        angvel = np.clip(np.asarray(imu_data["angVel"], np.float32) / IMU_ANGVEL_MAX, -1.0, 1.0) if drift["airSpeed"] > 1.0 else np.zeros(3, dtype=np.float32)

        # BEV from the LiDAR (built in the previous chapter)
        bev = np.nan_to_num(self._lidar_bev(self.lidar.poll()["pointCloud"],
                                      self.ego.state,
                                      drift,
                                      n=BEV_N),
                            nan=0.5)

        # relevant information for a current frame, for reward calculation
        self.current["speed"] = drift["airSpeed"]
        self.current["score"] = drift["potentialScore"]
        dmg = self.ego.sensors["damage"]
        if isinstance(dmg, dict):
            self.current["damage"] = dmg.get("damage", 0.0)
        else:
            self.current["damage"] = 0.0

        return {
            "rpm":         np.clip(np.array([e["rpm"] / e["maxrpm"]], np.float32), 0.0, 1.0),
            "imu_acc":     acc,
            "imu_angvel":  angvel,
            "drift_angle": np.clip(np.array([drift["angle"] / ANGLE_MAX], np.float32), -1.0, 1.0),
            "airspeed":    np.array([min(AIRSPEED_MAX_KPH, drift["airSpeed"]) / AIRSPEED_MAX_KPH], np.float32),
            "is_shifting": np.array([1.0 if e["is_shifting"] else 0.0], np.float32),
            "bev":         bev,
        }

    def _get_info(self):
        return {}

    def _setup_sensors(self):
        # realistic manual gearbox dropped, arcade set
        self.ego.set_shift_mode("arcade")

        # Sensors
        self.electrics = Electrics()
        self.damage = Damage()
        self.timer = Timer()

        self.ego.sensors.attach("electrics", self.electrics)
        self.ego.sensors.attach("damage", self.damage)
        self.ego.sensors.attach("timer", self.timer)

        self.imu = AdvancedIMU("imu",
            self.beamng,
            self.ego,
            gfx_update_time=1.0/LIDAR_HZ,
            is_send_immediately=True,
            is_dir_world_space=False,
            pos=(0, 0, 0.5)
        )
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

    def _action_to_control(self, action):
        steering, combined, parkingbrake = (float(value) for value in action["controls"])
        brake = 0.0
        throttle = 0.0

        if combined > 0.0:
            throttle = combined
        else:
            brake = -combined

        self.ego.control(
            steering=steering,
            throttle=throttle,
            brake=brake,
            parkingbrake=0.0 if parkingbrake < 0.0 else parkingbrake,
        )
        return {
            "steering": steering,
            "throttle": throttle,
            "brake": brake,
            "parkingbrake": parkingbrake,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # start BeamNG.drive on the first reset (works with reset-first ordering)
        if self.window is None:
            self._first_time_setup()

        # unpause physics / time
        self.beamng.control.resume()
        time.sleep(1)
        # vehicle switch
        self.scenario.remove_vehicle(self.ego)
        self.beamng.scenario.restart()
        car = DRIFT_CARS_6SPEED[self.np_random.integers(len(DRIFT_CARS_6SPEED))]
        waypoint_keys = list(self.waypoints.keys())
        key = waypoint_keys[self.np_random.integers(len(waypoint_keys))]
        self.ego = Vehicle("ego", license="DRIFT", color="Red", **car)
        self.scenario.add_vehicle(
            self.ego, pos=self.waypoints[key].pos, rot_quat=self.waypoints[key].rot
        )

        self._setup_sensors()
        time.sleep(1)

        # pause physics / time
        self.beamng.control.pause()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        self._action_to_control(action)

        # advance exactly 1 frame (blocks until done)
        self.beamng.control.step(1)

        # An episode is done if the agent has reached the target
        terminated, _ = self._check_failure()
        reward, _ = self._compute_reward()
        observation = self._get_obs()
        truncated = self.ego.sensors['timer']['time'] > EPISODE_TIMEOUT
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        # the game takes care of this without any work.
        pass

    def _first_time_setup(self):
        self.beamng = BeamNGpy("localhost", 25252)
        self.beamng.open()

        self.scenario = Scenario("west_coast_usa", "Gymnasium")

        self.scenario.make(self.beamng)
        self.beamng.scenario.load(self.scenario)

        self.waypoints = self.scenario.find_waypoints()
        self.waypoints = {w.name: w for w in self.waypoints}

        # done in reset
        car = random.choice(DRIFT_CARS_6SPEED)
        key = random.choice(list(self.waypoints.keys()))

        self.ego = Vehicle("ego", license="DRIFT", color="Red", **car)
        self.scenario.add_vehicle(
            self.ego, pos=self.waypoints[key].pos, rot_quat=self.waypoints[key].rot
        )

        self.beamng.scenario.start()

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

        # set to a value that your PC can handle and also 20 / fpslimit * factor has to be an integer
        fpslimit = int(FPS * FPS_FACTOR)
        self.beamng.queue_lua_command(f"settings.setValue('fpsLimit', {fpslimit})")

        self.beamng.queue_lua_command('settings.setValue("fpsLimitEnabled", true)')
        iterations = FPS / fpslimit * FPS_FACTOR
        self.beamng.queue_lua_command(f'be:setPhysicsSpeedFactor({iterations})')

        # settle
        time.sleep(1)

        # pause physics / time
        self.beamng.control.pause()

    def close(self):
        if self.window is not None:
            self.beamng.close()

    def _read_drift(self):
        """Drift telemetry as a dict. The lua side packs the fields into a
        pipe-delimited string (booleans as 1/0); we split and cast here, so no
        json round-trip is needed."""
        chunk = (
            "local s = gameplay_drift_scoring.getScore() "
            "local b2s = function(x) return x and 1 or 0 end "
            "return string.format('%s|%s|%s|%s|%s|%s', "
            "s.potentialScore, s.score, "
            "b2s(gameplay_drift_drift.getIsDrifting()), "
            "b2s(gameplay_drift_general.getFrozen()), "
            "gameplay_drift_drift.getCurrDegAngleSigned(), "
            "gameplay_drift_drift.getAirSpeed())"
        )
        raw = self.beamng.queue_lua_command(chunk, response=True)
        try:
            p = str(raw).split("|")
            return {"potentialScore": float(p[0]), "score": float(p[1]),
                    "isDrifting": p[2] == "1", "frozen": p[3] == "1",
                    "angle": float(p[4]), "airSpeed": float(p[5])}
        except (TypeError, ValueError, IndexError):
            print("read_drift: unexpected lua response:", repr(raw))
            return {"potentialScore": 0.0, "score": 0.0, "isDrifting": False,
                    "frozen": False, "angle": 0.0, "airSpeed": 0.0}

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

        # drift score (game's potentialScore) - a positive component
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

        self.previous = self.current
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

    def car_rotation(self, state):
        """Rotation matrix whose COLUMNS are the car's [forward, left, up] axes in
        world space, built from the (normalised) dir and up vectors."""
        d = np.asarray(state["dir"], float); d /= np.linalg.norm(d)
        u = np.asarray(state["up"], float);  u /= np.linalg.norm(u)
        l = np.cross(u, d)                     # left
        l /= np.linalg.norm(l)
        return np.stack([d, l, u], axis=1)     # columns [fwd, left, up]

    def _travel_slip(self, drift):
        """Travel-up rotation (rad) from the game's signed drift angle (deg)."""
        return float(np.deg2rad(drift.get("angle", 0.0)))

    def _lidar_bev(self, pc, state, drift, n=BEV_N, range_m=LIDAR_MAX_DIST):
        """
        Prepare a bird's eye view 2D matrix
        The car is in the center and the maximum range displayed in every direction is the lidar range, normalized to the grid size n
        The values (seen as brightness / intensity) represent the height of the highest point in the lidar point cloud, clamped to [0, 1] (0.5 = road level)

        Parameters
        ----------
        pc : np.ndarray, shape (N, 3)
            Lidar point cloud in world coordinates; each row is a point (x, y, z) - (fwd, left, up)       
        state : dict
            The car's current state, must contain "pos" (world position) and the orientation used by car_rotation
        drift : dict
            Drift telemetry from read_drift; must contain "angle", the game's signed drift angle in degrees (nose vs. direction of travel)       
        n : int, optional
            Grid size, the output is (n, n) cells (default is BEV_N = 64) 
        range_m : float, optional
            Half-extent of the grid in meters; it spans [-range_m, +range_m] around the car in both axes (default is LIDAR_MAX_DIST)
        """

        # Prepare a n * n 2D matrix which will represent the BEV, default BEV_N * BEV_N
        out = np.full((n, n), np.nan, np.float32)
        # return early if there are no lidar points
        if pc.size == 0:
            return out
        # get the car's rotation matrix relative to the world (axis)
        R = self.car_rotation(state)
        # subtract the car position from every point cloud dot relative to the car
        # to get the point cloud translated to be relative to the car
        # then rotate to match the car's rotation instead of the world's rotation
        p = (pc - np.asarray(state["pos"], float)) @ R   # [fwd, left, up]
        # split the points by axis
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        # obtain the drift angle and convert deg -> rad
        slip = self._travel_slip(drift)
        # pi/2 is 90 deg - meant to align the grid up axis to the car forward axis
        # the `slip` angle is subtracted to ensure that the direction of travel
        # is what finally represents the grid up axis in the BEV grid
        th = 0.5 * np.pi - slip
        # with the cos and sin of the angle, we can rotate every point on the (x,y) meaning (fwd,left) axis
        c, s = np.cos(th), np.sin(th)
        dx, dy = x * c - y * s, x * s + y * c
        # ix and iy are translations from physical distances in meters into the column and row
        # numbers in the BEV grid, rescaling into the BEV_N * BEV_N space, from lidar range
        ix = ((dx + range_m) * n / (2 * range_m)).astype(int)
        iy = ((dy + range_m) * n / (2 * range_m)).astype(int)
        # m is the mask for points that landed in the BEV_N * BEV_N space
        m = (ix >= 0) & (iy >= 0) & (ix < n) & (iy < n)
        # cell is the index which represents row and column with a single number
        # ex: for a 4x4 grid, index 12 would land in the last row and first column
        # this also flips the grid top to bottom (n-1-iy[m])
        cell = (n - 1 - iy[m]) * n + ix[m]
        # get the value of each dot in the BEV representing the height,
        # clamped to [0, 1] so the channel is bounded (0.5 = road level)
        h = np.clip((z[m] - Z_LO) / (Z_HI - Z_LO), 0.0, 1.0)
        # indices sorted low to high
        order = np.argsort(cell, kind="stable")
        # identify only the points in the grid where lidar points exist
        u, first = np.unique(cell[order], return_index=True)
        # update only those points with height information
        row, col = u // n, u % n
        out[row, col] = np.maximum.reduceat(h[order], first)
        return out
