#!/usr/bin/env python3
import math
import random
import time

import airsim
import numpy as np

from reward_utils import compute_chase_reward
from safety_filter import (
    ACTION_FORWARD_TO_TARGET,
    ACTION_HOVER,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_RIGHT,
    ACTION_MOVE_UP,
    ACTION_NAMES,
    ALTITUDE_MAX_Z,
    OBSTACLE_SLOW_DISTANCE,
    OBSTACLE_STOP_DISTANCE,
    SIDE_DANGER_DISTANCE,
    apply_safety_filter,
)
from target_controller import TargetController

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    gym = None
    spaces = None
    GYM_AVAILABLE = False


VEHICLE_CHASER = "Chaser"
VEHICLE_TARGET = "Target"
VEHICLES = [VEHICLE_CHASER, VEHICLE_TARGET]

SAFE_Z = -5.0
CHASER_SPEED = 2.0
TARGET_SPEED = 1.5
STEP_DURATION = 0.3
MAX_STEPS = 200
CATCH_DISTANCE = 3.0
TOO_FAR_DISTANCE = 200.0
TARGET_ALTITUDE = 8.0
MIN_SAFE_ALTITUDE = 4.0
MAX_SAFE_ALTITUDE = 15.0
HARD_MAX_ALTITUDE = 20.0
MAX_ACTIONS = 6
COLLISION_IGNORE_STEPS = 3
COLLISION_MIN_ALTITUDE_Z = -1.5
BYPASS_HOLD_STEPS = 12
BYPASS_MIN_STEPS_BEFORE_RELEASE = 3
BYPASS_TRIGGER_DISTANCE = 5.0
BYPASS_EMERGENCY_DISTANCE = 3.0
BYPASS_RELEASE_MIN_FRONT_DISTANCE = BYPASS_TRIGGER_DISTANCE
BYPASS_CLEAR_FRONT_DISTANCE = 12.0
DIAGONAL_BYPASS_EMERGENCY_DISTANCE = 3.0
DIAGONAL_BYPASS_MIN_LIDAR = DIAGONAL_BYPASS_EMERGENCY_DISTANCE
DIAGONAL_BYPASS_FRONT_DISTANCE = 5.0
DIAGONAL_BYPASS_RELEASE_FRONT_DISTANCE = 12.0
DIAGONAL_BYPASS_HOLD_STEPS = 12
DIAGONAL_BYPASS_FORWARD_SCALE = 0.3
DIAGONAL_BYPASS_SIDE_SCALE = 0.8
EMERGENCY_FRONT_HARD_DISTANCE = 2.0
EMERGENCY_BACK_SPEED = 2.0
EMERGENCY_SOFT_BACK_SPEED = 0.8
EMERGENCY_SIDE_SPEED = 3.0
EMERGENCY_UP_SPEED = -0.8
CAMERA_CENTERING_SIDE_SAFE_DISTANCE = 8.0
CAMERA_CENTERING_NEAR_DISTANCE = 10.0
CAMERA_CENTERING_FAR_DISTANCE = 20.0
CAMERA_CENTERING_FAR_LATERAL_SCALE = 0.15
CAMERA_CENTERING_MID_LATERAL_SCALE = 0.35
GAP_SIDE_SAFE_DISTANCE = 8.0
GAP_CENTER_FRONT_SAFE_DISTANCE = 8.0
GAP_CENTER_SIDE_MARGIN_DISTANCE = 3.0
GAP_FRONT_SLOW_DISTANCE = 5.0
GAP_FRONT_EMERGENCY_DISTANCE = 3.0
GAP_TARGET_SIDE_DEADBAND = 1.0
FORWARD_DETOUR_SPEED_SCALE = 0.55
NARROW_GAP_FORWARD_SCALE = 0.25
GLOBAL_SEARCH_DISTANCE = 15.0
GLOBAL_SEARCH_FRONT_OPEN_DISTANCE = 12.0
GLOBAL_SEARCH_MIN_SPEED = 3.0
STUCK_RECOVERY_SECONDS = 5.0
STUCK_RECOVERY_MIN_PROGRESS = 1.0
STUCK_RECOVERY_FORWARD_SPEED = 1.5
STUCK_RECOVERY_UP_SPEED = -0.8
OBSTACLE_PREPARE_DISTANCE = 12.0
OBSTACLE_OVERRIDE_DISTANCE = 8.0
OBSTACLE_SOFT_SLOW_DISTANCE = OBSTACLE_PREPARE_DISTANCE
OBSTACLE_SOFT_SPEED_SCALE = 0.6
SMOOTH_VELOCITY_PREVIOUS_WEIGHT = 0.75
SMOOTH_VELOCITY_NEW_WEIGHT = 0.25
HOVER_DRIFT_SPEED = 1.2
HOVER_DRIFT_MIN_DISTANCE = 8.0
CHASER_COMMAND_DURATION_SCALE = 2.0
CHASER_COMMAND_MIN_DURATION = 0.45
RECENT_OVERRIDE_WINDOW = 5
RECENT_OVERRIDE_LIMIT = 3
RECENT_OVERRIDE_CAPTURE_DISTANCE = 6.0
CLOSE_CHASE_DISTANCE = 15.0
CLOSE_CHASE_SLOW_DISTANCE = 5.0
CLOSE_CHASE_LOCK_PREP_DISTANCE = 12.0
CLOSE_CHASE_RECENT_SEEN_SECONDS = 1.0
CAPTURE_DEPTH = 3.5
CAPTURE_WIDTH = 2.5
CAPTURE_HEIGHT = 3.0
CAPTURE_BONUS = 100.0
RESET_ALTITUDE_MIN = 10.0
RESET_ALTITUDE_MAX = 12.5
RESET_ALTITUDE_RETRY_COUNT = 2
ALTITUDE_HOLD_KP = 0.4
ALTITUDE_HOLD_MAX_VZ = 1.0
START_PLACEMENT_TOLERANCE_METERS = 2.0
START_PLACEMENT_MIN_SPEED = 10.0
START_PLACEMENT_MAX_SPEED = 50.0
LEGACY_OBSERVATION_SIZE = 14
EXTENDED_OBSERVATION_SIZE = 26
OBSERVATION_SIZE = EXTENDED_OBSERVATION_SIZE
DEFAULT_OBS_MODE = "extended26"
SUPPORTED_OBS_MODES = ("legacy14", "extended26")
MAX_DIST = 50.0
MAX_SPEED = 5.0
MAX_LIDAR = 20.0
LIDAR_NAME = "Lidar1"
MAX_LIDAR_DISTANCE = 50.0
OBSTACLE_WARN_DISTANCE = 5.0
OBSTACLE_DANGER_DISTANCE = 2.0
LIDAR_SECTOR_NAMES = (
    "front",
    "front_left",
    "front_right",
    "left",
    "right",
    "back",
    "bottom",
)

BaseEnv = gym.Env if GYM_AVAILABLE else object


class AirSimChaseEnv(BaseEnv):
    metadata = {"render_modes": []}

    def __init__(
        self,
        target_mode="simple",
        target_base_speed=1.2,
        target_escape_speed=1.5,
        target_evade_distance=8.0,
        target_danger_distance=4.0,
        chaser_start_x=None,
        chaser_start_y=None,
        chaser_start_z=None,
        target_start_x=None,
        target_start_y=None,
        target_start_z=None,
        target_waypoints=None,
        max_episode_steps=MAX_STEPS,
        use_fast_reset=True,
        step_duration=STEP_DURATION,
        chaser_speed=CHASER_SPEED,
        reward_mode="simple",
        obs_mode=DEFAULT_OBS_MODE,
        use_capture_box=True,
        capture_depth=CAPTURE_DEPTH,
        capture_width=CAPTURE_WIDTH,
        capture_height=CAPTURE_HEIGHT,
        capture_bonus=CAPTURE_BONUS,
        drop_target_on_catch=False,
        catch_radius=CATCH_DISTANCE,
        min_start_distance=None,
        max_start_distance=None,
        random_start_angle=True,
        too_far_distance=TOO_FAR_DISTANCE,
        target_altitude=TARGET_ALTITUDE,
        min_safe_altitude=MIN_SAFE_ALTITUDE,
        max_safe_altitude=MAX_SAFE_ALTITUDE,
        hard_max_altitude=HARD_MAX_ALTITUDE,
        enable_altitude_safety=True,
    ):
        if target_mode not in ("simple", "evasive", "straight", "right_waypoint", "right_escape"):
            raise ValueError(
                f"Unknown target_mode: {target_mode}. "
                "Expected 'simple', 'evasive', 'straight', 'right_waypoint', or 'right_escape'."
            )
        if obs_mode not in SUPPORTED_OBS_MODES:
            raise ValueError(f"Unknown obs_mode: {obs_mode}. Expected one of {SUPPORTED_OBS_MODES}.")

        self.target_mode = target_mode
        self.target_base_speed = float(target_base_speed)
        self.target_escape_speed = float(target_escape_speed)
        self.target_evade_distance = float(target_evade_distance)
        self.target_danger_distance = float(target_danger_distance)
        self.chaser_start_x = None if chaser_start_x is None else float(chaser_start_x)
        self.chaser_start_y = None if chaser_start_y is None else float(chaser_start_y)
        self.target_altitude = max(0.1, float(target_altitude))
        self.chaser_start_z = -self.target_altitude if chaser_start_z is None else float(chaser_start_z)
        self.target_start_x = None if target_start_x is None else float(target_start_x)
        self.target_start_y = None if target_start_y is None else float(target_start_y)
        self.target_start_z = -self.target_altitude if target_start_z is None else float(target_start_z)
        self.target_waypoints = self._normalize_target_waypoints(target_waypoints)
        self.target_waypoint_index = 0
        self.max_episode_steps = max(1, int(max_episode_steps))
        self.use_fast_reset = bool(use_fast_reset)
        self.step_duration = max(0.05, float(step_duration))
        self.chaser_speed = max(0.1, float(chaser_speed))
        self.reward_mode = str(reward_mode)
        self.obs_mode = str(obs_mode)
        self.observation_size = LEGACY_OBSERVATION_SIZE if self.obs_mode == "legacy14" else EXTENDED_OBSERVATION_SIZE
        self.use_capture_box = bool(use_capture_box)
        self.capture_depth = max(0.1, float(capture_depth))
        self.capture_width = max(0.1, float(capture_width))
        self.capture_height = max(0.1, float(capture_height))
        self.capture_bonus = float(capture_bonus)
        self.drop_target_on_catch = bool(drop_target_on_catch)
        self.catch_radius = max(0.1, float(catch_radius))
        self.min_start_distance = None if min_start_distance is None else max(0.0, float(min_start_distance))
        self.max_start_distance = None if max_start_distance is None else max(0.0, float(max_start_distance))
        if self.min_start_distance is not None and self.max_start_distance is not None:
            if self.max_start_distance < self.min_start_distance:
                raise ValueError("max_start_distance must be >= min_start_distance.")
        self.random_start_angle = bool(random_start_angle)
        self.too_far_distance = max(1.0, float(too_far_distance))
        self.min_safe_altitude = max(0.0, float(min_safe_altitude))
        self.max_safe_altitude = max(self.min_safe_altitude + 0.1, float(max_safe_altitude))
        self.hard_max_altitude = max(self.max_safe_altitude + 0.1, float(hard_max_altitude))
        self.enable_altitude_safety = bool(enable_altitude_safety)
        self.client = airsim.MultirotorClient()
        try:
            self.client.confirmConnection()
        except Exception as exc:
            raise RuntimeError(f"AirSim connection failed. Is AirSimNH open? ({exc})") from exc

        self._validate_vehicles()
        self.step_count = 0
        self.previous_distance = None
        self._lidar_empty_warned = False
        self._lidar_error_warned = False
        self._closed = False
        self.last_chaser_velocity = (0.0, 0.0, 0.0)
        self.last_chaser_command_info = self._default_command_info()
        self.last_chaser_future = None
        self.last_target_velocity = (0.0, 0.0, 0.0)
        self.previous_action = ACTION_HOVER
        self.last_safety_overridden = False
        self.bypass_active = False
        self.bypass_action = None
        self.bypass_steps_remaining = 0
        self.bypass_reason = ""
        self.bypass_steps_elapsed = 0
        self.obstacle_bypass_direction = None
        self.obstacle_bypass_steps_remaining = 0
        self.vehicle_home_global = {}
        self.actual_chaser_start_pos = None
        self.actual_target_start_pos = None
        self.fast_reset_chaser_pos = None
        self.fast_reset_target_pos = None
        self._full_reset_done = False
        self._force_full_reset_next = False
        self.episode_safety_override_count = 0
        self.episode_min_lidar_sum = 0.0
        self.episode_min_lidar_count = 0
        self.episode_min_lidar_min = MAX_LIDAR_DISTANCE
        self.recent_safety_overrides = []
        self.previous_min_lidar = None
        self.distance_history = []
        self.last_effective_collision = False
        self.target_dropped_on_catch = False
        self.target_controller = None

        if GYM_AVAILABLE:
            self.action_space = spaces.Discrete(MAX_ACTIONS)
            if self.obs_mode == "legacy14":
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(LEGACY_OBSERVATION_SIZE,),
                    dtype=np.float32,
                )
            else:
                self.observation_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(EXTENDED_OBSERVATION_SIZE,),
                    dtype=np.float32,
                )
        else:
            self.action_space = None
            self.observation_space = None

        if self.target_mode == "evasive":
            self.target_controller = TargetController(
                self.client,
                vehicle_name=VEHICLE_TARGET,
                safe_z=self.target_start_z,
                base_speed=self.target_base_speed,
                escape_speed=self.target_escape_speed,
                evade_distance=self.target_evade_distance,
                danger_distance=self.target_danger_distance,
            )

    def _validate_vehicles(self):
        try:
            vehicles = self.client.listVehicles()
        except Exception as exc:
            raise RuntimeError(f"AirSim vehicles list could not be read: {exc}") from exc

        missing = [name for name in VEHICLES if name not in vehicles]
        if missing:
            raise RuntimeError(
                "Required vehicles not found in AirSim: "
                f"{missing}. Check ~/Documents/AirSim/settings.json and restart AirSimNH."
            )

        return vehicles

    def _normalize_target_waypoints(self, target_waypoints):
        if not target_waypoints:
            return []
        normalized = []
        for waypoint in target_waypoints:
            if len(waypoint) != 3:
                raise ValueError(f"Target waypoint must have 3 values, got: {waypoint}")
            normalized.append((float(waypoint[0]), float(waypoint[1]), float(waypoint[2])))
        return normalized

    def reset(self, seed=None, options=None):
        if GYM_AVAILABLE:
            try:
                super().reset(seed=seed)
            except Exception:
                pass

        self.step_count = 0
        self.previous_distance = None
        self._closed = False
        self.last_chaser_velocity = (0.0, 0.0, 0.0)
        self.last_chaser_command_info = self._default_command_info()
        self.last_chaser_future = None
        self.last_target_velocity = (0.0, 0.0, 0.0)
        self.previous_action = ACTION_HOVER
        self.last_safety_overridden = False
        self._reset_bypass_state()
        self._reset_obstacle_bypass_state()
        self._reset_episode_metrics()
        self.recent_safety_overrides = []
        self.previous_min_lidar = None
        self.distance_history = []
        self.last_effective_collision = False
        self.target_dropped_on_catch = False
        self.target_waypoint_index = 0
        self.actual_chaser_start_pos = None
        self.actual_target_start_pos = None

        reset_ok = False
        if self.use_fast_reset and self._full_reset_done and not self._force_full_reset_next:
            try:
                reset_ok = self._fast_reset_scene()
            except Exception as exc:
                print(f"[WARN] Fast reset failed, falling back to AirSim reset: {exc}", flush=True)
                reset_ok = False

        if not reset_ok:
            self._full_reset_scene()

        collision_info = self._get_collision_info()
        raw_collision = collision_info["has_collided"]
        if raw_collision:
            print(
                "[WARN] Raw collision is true immediately after reset/takeoff; "
                "ignoring during first warmup steps.",
                flush=True,
            )

        obs, info = self._get_obs_info(
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            raw_collision=raw_collision,
            collision=False,
            collision_info=collision_info,
            safety_result=self._default_safety_result(None),
            reward_breakdown=self._default_reward_breakdown(),
            caught=False,
            too_far=False,
            terminated_reason="none",
        )

        self.step_count = 0
        self.previous_distance = info["distance"]
        self.last_reset_time = time.time()

        if not self._any_start_requested() and not 3.0 <= info["distance"] <= 8.0:
            print(
                "[WARN] Reset sonrası Chaser-Target mesafesi beklenen ~5m civarında değil: "
                f"{info['distance']:.2f} m. settings.json spawn offset kontrol edilebilir.",
                flush=True,
            )

        return obs, info

    def step(self, action):
        try:
            action = int(action)
        except Exception:
            action = None

        self.step_count += 1

        chaser_pos = self.get_global_position(VEHICLE_CHASER)
        target_pos = self.get_global_position(VEHICLE_TARGET)
        dx, dy, dz, distance = self._compute_relative(chaser_pos, target_pos)
        pre_capture_state = self._compute_capture_state(chaser_pos, target_pos, distance)
        if pre_capture_state["caught"]:
            collision_info = self._get_collision_info()
            raw_collision = collision_info["has_collided"]
            safety_result = self._default_safety_result(action)
            safety_result["reason"] = "caught before safety filter"
            return self._complete_step(
                action,
                chaser_pos,
                target_pos,
                dx,
                dy,
                dz,
                distance,
                raw_collision,
                False,
                collision_info,
                safety_result,
                pre_capture_state,
                safety_bypassed_for_capture=False,
                override_count_recent=self._recent_safety_override_count(),
            )

        current_lidar_info = self._get_lidar_info()
        safety_bypassed_for_capture = False
        if (
            action == ACTION_FORWARD_TO_TARGET
            and distance <= self.capture_depth + 0.5
            and self._target_in_capture_cone(chaser_pos, target_pos, extra_depth=0.5)
        ):
            safety_bypassed_for_capture = True
            safety_result = self._default_safety_result(action)
            safety_result["reason"] = "target in capture cone; safety bypassed"
        else:
            safety_result = apply_safety_filter(action, current_lidar_info["sectors"], chaser_pos.z_val)
            safety_result = self._apply_bypass_hold(action, safety_result, current_lidar_info["sectors"], chaser_pos.z_val)

        override_count_recent = self._recent_safety_override_count(include_current=safety_result["overridden"])
        if override_count_recent >= RECENT_OVERRIDE_LIMIT and distance < RECENT_OVERRIDE_CAPTURE_DISTANCE:
            recapture_state = self._compute_capture_state(chaser_pos, target_pos, distance)
            if recapture_state["caught"]:
                collision_info = self._get_collision_info()
                raw_collision = collision_info["has_collided"]
                safety_result = self._default_safety_result(action)
                safety_result["reason"] = "caught while breaking safety override loop"
                return self._complete_step(
                    action,
                    chaser_pos,
                    target_pos,
                    dx,
                    dy,
                    dz,
                    distance,
                    raw_collision,
                    False,
                    collision_info,
                    safety_result,
                    recapture_state,
                    safety_bypassed_for_capture=True,
                    override_count_recent=override_count_recent,
                )
            safety_result = dict(safety_result)
            safety_result["safe_action"] = ACTION_FORWARD_TO_TARGET
            safety_result["overridden"] = action != ACTION_FORWARD_TO_TARGET
            safety_result["reason"] = "recent safety override loop near target; forcing FORWARD_TO_TARGET"
            safety_result["risk_level"] = "capture"
            safety_result = self._annotate_bypass_result(safety_result, False)
            safety_bypassed_for_capture = True

        safety_result = self._apply_altitude_safety(safety_result, chaser_pos.z_val)
        safe_action = safety_result["safe_action"]

        command_info = self._apply_chaser_action(
            safe_action,
            chaser_pos,
            target_pos,
            safety_result=safety_result,
            lidar_sectors=current_lidar_info["sectors"],
        )
        safety_result = self._merge_command_altitude_safety(safety_result, command_info)
        self._move_target(chaser_pos, target_pos)

        time.sleep(0.05)

        chaser_pos = self.get_global_position(VEHICLE_CHASER)
        target_pos = self.get_global_position(VEHICLE_TARGET)
        dx, dy, dz, distance = self._compute_relative(chaser_pos, target_pos)
        collision_info = self._get_collision_info()
        raw_collision = collision_info["has_collided"]

        # AirSim may report a stale or early collision flag immediately after
        # reset/takeoff. For environment validation we expose raw_collision in
        # info, but use effective_collision after a short warmup period for
        # reward and termination.
        effective_collision = raw_collision
        if self.step_count <= COLLISION_IGNORE_STEPS:
            effective_collision = False
        if chaser_pos.z_val > COLLISION_MIN_ALTITUDE_Z:
            effective_collision = False

        capture_state = self._compute_capture_state(chaser_pos, target_pos, distance)
        return self._complete_step(
            action,
            chaser_pos,
            target_pos,
            dx,
            dy,
            dz,
            distance,
            raw_collision,
            effective_collision,
            collision_info,
            safety_result,
            capture_state,
            safety_bypassed_for_capture=safety_bypassed_for_capture,
            override_count_recent=override_count_recent,
        )

    def _complete_step(
        self,
        action,
        chaser_pos,
        target_pos,
        dx,
        dy,
        dz,
        distance,
        raw_collision,
        effective_collision,
        collision_info,
        safety_result,
        capture_state,
        safety_bypassed_for_capture=False,
        override_count_recent=None,
    ):
        lidar_info = self._get_lidar_info()
        min_lidar = self._min_lidar_distance(lidar_info["sectors"])
        previous_min_lidar = self.previous_min_lidar
        self._update_episode_metrics(safety_result["overridden"], min_lidar)
        self._record_safety_override(safety_result["overridden"])

        caught = bool(capture_state.get("caught", False))
        altitude = self._altitude_from_z(chaser_pos.z_val)
        too_high = bool(self.enable_altitude_safety and altitude > self.hard_max_altitude)
        too_far = distance > self.too_far_distance
        target_in_front = self._target_in_capture_cone(chaser_pos, target_pos, extra_depth=1.0)
        near_capture_zone = target_in_front and distance <= max(self.capture_depth + 1.0, self.catch_radius + 1.0)
        reward_breakdown = compute_chase_reward(
            distance=distance,
            previous_distance=self.previous_distance,
            collision=effective_collision,
            caught=caught,
            too_far=too_far,
            lidar_sectors=lidar_info["sectors"] if lidar_info["available"] else None,
            safety_overridden=safety_result["overridden"],
            chaser_z=chaser_pos.z_val,
            target_z=target_pos.z_val,
            step_duration=self.step_duration,
            min_lidar=min_lidar,
            previous_min_lidar=previous_min_lidar,
            reward_mode=self.reward_mode,
            near_capture_zone=near_capture_zone,
            target_in_front=target_in_front,
            target_altitude=self.target_altitude,
            min_safe_altitude=self.min_safe_altitude,
            max_safe_altitude=self.max_safe_altitude,
            hard_max_altitude=self.hard_max_altitude,
            too_high=too_high,
            emergency_avoidance=bool(safety_result.get("emergency_avoidance", False)),
        )
        reward_breakdown = dict(reward_breakdown)
        reward_breakdown["capture_bonus_reward"] = self.capture_bonus if caught else 0.0
        if caught:
            reward_breakdown["total"] = float(reward_breakdown["total"] + self.capture_bonus)
            self._drop_target_if_requested()

        reward = reward_breakdown["total"]
        terminated = caught or effective_collision or too_far or too_high
        truncated = self.step_count >= self.max_episode_steps
        if too_high:
            terminated_reason = "too_high"
        elif caught:
            terminated_reason = capture_state.get("done_reason", "caught")
        else:
            terminated_reason = self._termination_reason(caught, effective_collision, too_far, too_high, truncated)
        if effective_collision:
            self._force_full_reset_next = True

        safe_action = safety_result["safe_action"]
        self.previous_action = safe_action
        self.last_safety_overridden = bool(safety_result["overridden"])
        self.last_effective_collision = bool(effective_collision)
        if override_count_recent is None:
            override_count_recent = self._recent_safety_override_count()
        else:
            override_count_recent = max(int(override_count_recent), self._recent_safety_override_count())

        obs = self._make_observation(
            dx,
            dy,
            dz,
            distance,
            chaser_pos,
            target_pos,
            lidar_info["sectors"],
            safety_result["overridden"],
            safe_action,
        )
        info = self._make_info(
            action,
            dx,
            dy,
            dz,
            distance,
            reward,
            terminated,
            truncated,
            raw_collision,
            effective_collision,
            collision_info,
            lidar_info,
            safety_result,
            reward_breakdown,
            caught,
            too_far,
            too_high,
            terminated_reason,
            chaser_pos,
            target_pos,
            capture_state=capture_state,
            safety_bypassed_for_capture=safety_bypassed_for_capture,
            override_count_recent=override_count_recent,
        )
        self.previous_distance = distance
        self.previous_min_lidar = min_lidar
        self._record_distance_history(distance)

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _reset_bypass_state(self):
        self.bypass_active = False
        self.bypass_action = None
        self.bypass_steps_remaining = 0
        self.bypass_reason = ""
        self.bypass_steps_elapsed = 0

    def _reset_obstacle_bypass_state(self):
        self.obstacle_bypass_direction = None
        self.obstacle_bypass_steps_remaining = 0

    def _reset_episode_metrics(self):
        self.episode_safety_override_count = 0
        self.episode_min_lidar_sum = 0.0
        self.episode_min_lidar_count = 0
        self.episode_min_lidar_min = MAX_LIDAR_DISTANCE

    def _update_episode_metrics(self, safety_overridden, min_lidar):
        if safety_overridden:
            self.episode_safety_override_count += 1
        lidar_value = self._safe_distance_value(min_lidar, MAX_LIDAR_DISTANCE)
        self.episode_min_lidar_sum += lidar_value
        self.episode_min_lidar_count += 1
        self.episode_min_lidar_min = min(self.episode_min_lidar_min, lidar_value)

    def _episode_min_lidar_mean(self):
        if self.episode_min_lidar_count <= 0:
            return MAX_LIDAR_DISTANCE
        return self.episode_min_lidar_sum / float(self.episode_min_lidar_count)

    def _default_capture_state(self):
        return {
            "caught": False,
            "distance_caught": False,
            "capture_box": False,
            "done_reason": "none",
            "capture_forward": 0.0,
            "capture_lateral": 0.0,
            "capture_vertical": 0.0,
        }

    def _vehicle_yaw(self, vehicle_name):
        try:
            state = self.client.getMultirotorState(vehicle_name=vehicle_name)
            _, _, yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
            return float(yaw)
        except Exception:
            return None

    def _capture_geometry(self, chaser_pos, target_pos):
        dx = target_pos.x_val - chaser_pos.x_val
        dy = target_pos.y_val - chaser_pos.y_val
        dz = target_pos.z_val - chaser_pos.z_val
        horizontal = math.sqrt(dx * dx + dy * dy)
        yaw = self._vehicle_yaw(VEHICLE_CHASER)
        if yaw is None:
            forward = horizontal
            lateral = 0.0
        else:
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            forward = cos_yaw * dx + sin_yaw * dy
            lateral = -sin_yaw * dx + cos_yaw * dy
        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "horizontal": horizontal,
            "forward": forward,
            "lateral": lateral,
            "yaw_available": yaw is not None,
        }

    def _compute_capture_state(self, chaser_pos, target_pos, distance=None):
        if distance is None:
            _, _, _, distance = self._compute_relative(chaser_pos, target_pos)
        geometry = self._capture_geometry(chaser_pos, target_pos)
        distance_caught = distance <= max(self.catch_radius, 3.0)
        capture_box = False
        if self.use_capture_box:
            capture_box = (
                0.0 <= geometry["forward"] <= self.capture_depth
                and abs(geometry["lateral"]) <= self.capture_width * 0.5
                and abs(geometry["dz"]) <= self.capture_height * 0.5
            )
        caught = bool(distance_caught or capture_box)
        done_reason = "none"
        if distance_caught:
            done_reason = "caught_distance"
        elif capture_box:
            done_reason = "caught_capture_box"
        return {
            "caught": caught,
            "distance_caught": bool(distance_caught),
            "capture_box": bool(capture_box),
            "done_reason": done_reason,
            "capture_forward": geometry["forward"],
            "capture_lateral": geometry["lateral"],
            "capture_vertical": geometry["dz"],
        }

    def _target_in_capture_cone(self, chaser_pos, target_pos, extra_depth=0.5):
        geometry = self._capture_geometry(chaser_pos, target_pos)
        return (
            0.0 <= geometry["forward"] <= self.capture_depth + extra_depth
            and abs(geometry["lateral"]) <= max(self.capture_width, 1.0)
            and abs(geometry["dz"]) <= max(self.capture_height, 1.0)
        )

    def _recent_safety_override_count(self, include_current=None):
        values = list(self.recent_safety_overrides[-RECENT_OVERRIDE_WINDOW:])
        if include_current is not None:
            values.append(bool(include_current))
            values = values[-RECENT_OVERRIDE_WINDOW:]
        return sum(1 for value in values if value)

    def _record_safety_override(self, overridden):
        self.recent_safety_overrides.append(bool(overridden))
        if len(self.recent_safety_overrides) > RECENT_OVERRIDE_WINDOW:
            self.recent_safety_overrides = self.recent_safety_overrides[-RECENT_OVERRIDE_WINDOW:]

    def _drop_target_if_requested(self):
        if not self.drop_target_on_catch or self.target_dropped_on_catch:
            return
        try:
            self.client.armDisarm(False, vehicle_name=VEHICLE_TARGET)
            self.target_dropped_on_catch = True
        except Exception as exc:
            print(f"[WARN] Target drop on catch failed: {exc}", flush=True)

    def _sector_distance(self, lidar_sectors, name):
        if lidar_sectors is None:
            return MAX_LIDAR_DISTANCE
        value = lidar_sectors.get(name, MAX_LIDAR_DISTANCE)
        if value is None:
            return MAX_LIDAR_DISTANCE
        try:
            return float(value)
        except Exception:
            return MAX_LIDAR_DISTANCE

    def _safe_distance_value(self, value, default=MAX_LIDAR_DISTANCE):
        try:
            value = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(value):
            return float(default)
        return value

    def _min_lidar_distance(self, lidar_sectors):
        if lidar_sectors is None:
            return MAX_LIDAR_DISTANCE
        names = ("front", "front_left", "front_right", "left", "right")
        return min(self._sector_distance(lidar_sectors, name) for name in names)

    def _front_obstacle_distance(self, lidar_sectors):
        if lidar_sectors is None:
            return MAX_LIDAR_DISTANCE
        return min(
            self._sector_distance(lidar_sectors, "front"),
            self._sector_distance(lidar_sectors, "front_left"),
            self._sector_distance(lidar_sectors, "front_right"),
        )

    def _altitude_from_z(self, z_val):
        try:
            return -float(z_val)
        except Exception:
            return self.target_altitude

    def _apply_altitude_safety(self, safety_result, chaser_z):
        result = dict(safety_result)
        result.setdefault("altitude_safety_override", False)
        altitude = self._altitude_from_z(chaser_z)
        safe_action = result.get("safe_action")
        altitude_override = False
        reason = result.get("reason", "")

        if not self.enable_altitude_safety:
            result["altitude"] = altitude
            result["altitude_error"] = self.target_altitude - altitude
            return result

        if safe_action == ACTION_MOVE_DOWN and altitude < self.min_safe_altitude:
            safe_action = ACTION_MOVE_UP
            altitude_override = True
            reason = self._append_reason(reason, "altitude below min_safe_altitude; blocking MOVE_DOWN")
        elif safe_action == ACTION_MOVE_UP and altitude > self.max_safe_altitude:
            safe_action = ACTION_HOVER
            altitude_override = True
            reason = self._append_reason(reason, "altitude above max_safe_altitude; blocking MOVE_UP")

        if altitude_override:
            result["safe_action"] = safe_action
            result["overridden"] = safe_action != result.get("original_action")
            result["reason"] = reason
            result["risk_level"] = "altitude"

        result["altitude_safety_override"] = bool(altitude_override)
        result["altitude"] = altitude
        result["altitude_error"] = self.target_altitude - altitude
        return result

    def _merge_command_altitude_safety(self, safety_result, command_info):
        result = dict(safety_result)
        if command_info.get("altitude_safety_override", False):
            result["overridden"] = True
            result["reason"] = self._append_reason(
                result.get("reason", ""),
                command_info.get("reason", "altitude velocity clamp"),
            )
            result["risk_level"] = "altitude"
        result["altitude_safety_override"] = bool(
            result.get("altitude_safety_override", False)
            or command_info.get("altitude_safety_override", False)
        )
        result["altitude"] = command_info.get("altitude", result.get("altitude", self.target_altitude))
        result["altitude_error"] = command_info.get("altitude_error", result.get("altitude_error", 0.0))
        result["too_high"] = bool(command_info.get("too_high", False))
        result["command_vx"] = float(command_info.get("vx", 0.0))
        result["command_vy"] = float(command_info.get("vy", 0.0))
        result["command_vz"] = float(command_info.get("vz", 0.0))
        result["vz_alt_hold"] = float(command_info.get("vz_alt_hold", 0.0))
        result["final_vz"] = float(command_info.get("final_vz", command_info.get("vz", 0.0)))
        result["final_vx"] = float(command_info.get("final_vx", command_info.get("vx", 0.0)))
        result["final_vy"] = float(command_info.get("final_vy", command_info.get("vy", 0.0)))
        result["base_vz"] = float(command_info.get("base_vz", 0.0))
        result["obstacle_bypass"] = bool(command_info.get("obstacle_bypass", False))
        result["diagonal_bypass"] = bool(
            result.get("diagonal_bypass", False)
            or command_info.get("diagonal_bypass", False)
        )
        result["emergency_avoidance"] = bool(
            result.get("emergency_avoidance", False)
            or command_info.get("emergency_avoidance", False)
        )
        result["bypass_direction"] = command_info.get("bypass_direction", "none")
        result["bypass_steps_remaining"] = int(
            command_info.get("bypass_steps_remaining", result.get("bypass_steps_remaining", 0))
        )
        result["forward_scale"] = float(command_info.get("forward_scale", 1.0))
        result["side_scale"] = float(command_info.get("side_scale", 0.0))
        result["speed_scale"] = float(command_info.get("speed_scale", result.get("speed_scale", 1.0)))
        result["camera_centering_limited"] = bool(
            result.get("camera_centering_limited", False)
            or command_info.get("camera_centering_limited", False)
        )
        result["camera_centering_reason"] = command_info.get(
            "camera_centering_reason",
            result.get("camera_centering_reason", ""),
        )
        result["gap_direction"] = command_info.get("gap_direction", result.get("gap_direction", "center"))
        result["gap_safe"] = bool(command_info.get("gap_safe", result.get("gap_safe", True)))
        result["target_side"] = command_info.get("target_side", result.get("target_side", "center"))
        result["front_clear"] = bool(command_info.get("front_clear", result.get("front_clear", True)))
        result["left_blocked"] = bool(command_info.get("left_blocked", result.get("left_blocked", False)))
        result["right_blocked"] = bool(command_info.get("right_blocked", result.get("right_blocked", False)))
        result["close_chase_mode"] = bool(command_info.get("close_chase_mode", result.get("close_chase_mode", False)))
        result["planner_choice"] = command_info.get("planner_choice", result.get("planner_choice", "forward"))
        result["chosen_reason"] = command_info.get("chosen_reason", result.get("chosen_reason", "target_progress"))
        result["obstacle_reaction_zone"] = command_info.get(
            "obstacle_reaction_zone",
            result.get("obstacle_reaction_zone", "none"),
        )
        result["blocked_lateral"] = bool(command_info.get("blocked_lateral", result.get("blocked_lateral", False)))
        result["forward_detour"] = bool(command_info.get("forward_detour", result.get("forward_detour", False)))
        result["climb_avoidance"] = bool(command_info.get("climb_avoidance", result.get("climb_avoidance", False)))
        result["stuck_recovery"] = bool(command_info.get("stuck_recovery", result.get("stuck_recovery", False)))
        result["smooth_velocity"] = bool(command_info.get("smooth_velocity", result.get("smooth_velocity", False)))
        result["hover_drift"] = bool(command_info.get("hover_drift", result.get("hover_drift", False)))
        result["pre_smooth_vx"] = float(command_info.get("pre_smooth_vx", result.get("pre_smooth_vx", 0.0)))
        result["pre_smooth_vy"] = float(command_info.get("pre_smooth_vy", result.get("pre_smooth_vy", 0.0)))
        result["pre_smooth_vz"] = float(command_info.get("pre_smooth_vz", result.get("pre_smooth_vz", 0.0)))
        result["command_duration"] = float(command_info.get("command_duration", result.get("command_duration", 0.0)))
        result["yaw_to_target_deg"] = float(command_info.get("yaw_to_target_deg", result.get("yaw_to_target_deg", 0.0)))
        return result

    def _append_reason(self, current_reason, extra_reason):
        if not current_reason:
            return extra_reason
        return f"{current_reason}; {extra_reason}"

    def _move_up_allowed(self, chaser_z):
        return chaser_z is None or chaser_z > ALTITUDE_MAX_Z

    def _choose_side_bypass_action(self, left_dist, right_dist, reason_prefix):
        if right_dist > left_dist and right_dist > SIDE_DANGER_DISTANCE:
            return ACTION_MOVE_RIGHT, f"{reason_prefix}; bypass hold choosing right"
        if left_dist > SIDE_DANGER_DISTANCE:
            return ACTION_MOVE_LEFT, f"{reason_prefix}; bypass hold choosing left"
        return ACTION_HOVER, f"{reason_prefix}; no safe side available; bypass hold hovering"

    def _choose_bypass_action(self, front_dist, left_dist, right_dist, chaser_z):
        if front_dist < BYPASS_EMERGENCY_DISTANCE:
            reason_prefix = "front obstacle emergency close"
            if self._move_up_allowed(chaser_z):
                return ACTION_MOVE_UP, f"{reason_prefix}; bypass hold moving up"
            return self._choose_side_bypass_action(left_dist, right_dist, reason_prefix)

        reason_prefix = "front obstacle within bypass distance"
        side_action, reason = self._choose_side_bypass_action(left_dist, right_dist, reason_prefix)
        if side_action != ACTION_HOVER:
            return side_action, reason
        if self._move_up_allowed(chaser_z):
            return ACTION_MOVE_UP, f"{reason_prefix}; no safe side available; bypass hold moving up"
        return ACTION_HOVER, reason

    def _held_bypass_action(self, front_dist, left_dist, right_dist, chaser_z):
        if front_dist < BYPASS_EMERGENCY_DISTANCE:
            return self._choose_bypass_action(front_dist, left_dist, right_dist, chaser_z)
        if self.bypass_steps_elapsed < BYPASS_MIN_STEPS_BEFORE_RELEASE:
            if self.bypass_action in ACTION_NAMES:
                return self.bypass_action, self.bypass_reason
        if self.bypass_action == ACTION_MOVE_RIGHT and right_dist <= SIDE_DANGER_DISTANCE:
            return self._choose_bypass_action(front_dist, left_dist, right_dist, chaser_z)
        if self.bypass_action == ACTION_MOVE_LEFT and left_dist <= SIDE_DANGER_DISTANCE:
            return self._choose_bypass_action(front_dist, left_dist, right_dist, chaser_z)
        if self.bypass_action not in ACTION_NAMES:
            return self._choose_bypass_action(front_dist, left_dist, right_dist, chaser_z)
        return self.bypass_action, self.bypass_reason

    def _annotate_bypass_result(self, safety_result, active, action=None, remaining=0, reason="", emergency=False):
        result = dict(safety_result)
        result["bypass_active"] = bool(active)
        result["bypass_action"] = action
        result["bypass_action_name"] = ACTION_NAMES.get(action, "none") if action is not None else "none"
        result["bypass_steps_remaining"] = int(remaining)
        result["bypass_reason"] = reason
        result["bypass_trigger_distance"] = BYPASS_TRIGGER_DISTANCE
        result["emergency_avoid"] = bool(emergency)
        if not active:
            result["diagonal_bypass"] = False
            result["bypass_direction"] = "none"
        return result

    def _bypass_result(self, original_action, safety_result, action, reason, front_dist):
        emergency = front_dist < BYPASS_EMERGENCY_DISTANCE
        self.bypass_action = action
        self.bypass_reason = reason
        self.bypass_active = True
        self.bypass_steps_elapsed += 1
        self.bypass_steps_remaining = max(0, self.bypass_steps_remaining - 1)

        result = dict(safety_result)
        result["safe_action"] = action
        result["overridden"] = action != original_action
        result["reason"] = reason
        result["risk_level"] = "danger" if emergency or front_dist < OBSTACLE_STOP_DISTANCE else "slow"
        result = self._annotate_bypass_result(
            result,
            True,
            action=action,
            remaining=self.bypass_steps_remaining,
            reason=reason,
            emergency=emergency,
        )

        return result

    def _apply_bypass_hold(self, original_action, safety_result, lidar_sectors, chaser_z):
        front_dist = self._sector_distance(lidar_sectors, "front")
        left_dist = min(
            self._sector_distance(lidar_sectors, "left"),
            self._sector_distance(lidar_sectors, "front_left"),
        )
        right_dist = min(
            self._sector_distance(lidar_sectors, "right"),
            self._sector_distance(lidar_sectors, "front_right"),
        )

        if original_action == ACTION_FORWARD_TO_TARGET:
            self._reset_bypass_state()
            safe_action = safety_result.get("safe_action")
            if safe_action not in (ACTION_FORWARD_TO_TARGET, ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT):
                return self._annotate_bypass_result(safety_result, False)

            diagonal_requested = bool(safety_result.get("diagonal_bypass", False))
            if diagonal_requested or front_dist < BYPASS_TRIGGER_DISTANCE:
                result = dict(safety_result)
                result["safe_action"] = ACTION_FORWARD_TO_TARGET
                result["overridden"] = bool(result.get("overridden", False) or front_dist < OBSTACLE_STOP_DISTANCE)
                if not result.get("reason"):
                    result["reason"] = "front obstacle; diagonal bypass"
                if front_dist < OBSTACLE_STOP_DISTANCE:
                    result["risk_level"] = "danger"
                elif result.get("risk_level") == "none":
                    result["risk_level"] = "slow"
                result["diagonal_bypass"] = True
                return self._annotate_bypass_result(
                    result,
                    True,
                    action=ACTION_FORWARD_TO_TARGET,
                    remaining=self.obstacle_bypass_steps_remaining,
                    reason=result["reason"],
                    emergency=front_dist < BYPASS_EMERGENCY_DISTANCE,
                )

            return self._annotate_bypass_result(safety_result, False)

        if self.bypass_active:
            if (
                front_dist > BYPASS_CLEAR_FRONT_DISTANCE
                and self.bypass_steps_elapsed >= BYPASS_MIN_STEPS_BEFORE_RELEASE
            ):
                self._reset_bypass_state()
                return self._annotate_bypass_result(safety_result, False)
            if (
                self.bypass_steps_remaining <= 0
                and front_dist >= BYPASS_RELEASE_MIN_FRONT_DISTANCE
            ):
                self._reset_bypass_state()
                return self._annotate_bypass_result(safety_result, False)

            action, reason = self._held_bypass_action(front_dist, left_dist, right_dist, chaser_z)
            return self._bypass_result(original_action, safety_result, action, reason, front_dist)

        if front_dist < BYPASS_TRIGGER_DISTANCE:
            action, reason = self._choose_bypass_action(front_dist, left_dist, right_dist, chaser_z)
            self.bypass_active = True
            self.bypass_action = action
            self.bypass_steps_remaining = BYPASS_HOLD_STEPS
            self.bypass_steps_elapsed = 0
            self.bypass_reason = reason
            return self._bypass_result(original_action, safety_result, action, reason, front_dist)

        return self._annotate_bypass_result(safety_result, False)

    def close(self):
        if self._closed:
            return True

        cleanup_ok = True
        print("[INFO] AirSimChaseEnv cleanup starting.", flush=True)
        for name in VEHICLES:
            try:
                self.client.hoverAsync(vehicle_name=name).join()
            except Exception as exc:
                cleanup_ok = False
                print(f"[WARN] Hover before landing failed: {name} ({exc})", flush=True)

        for name in VEHICLES:
            try:
                self.client.landAsync(vehicle_name=name).join()
                print(f"[OK] Landing completed: {name}", flush=True)
            except Exception as exc:
                cleanup_ok = False
                print(f"[WARN] Landing failed: {name} ({exc})", flush=True)

        for name in VEHICLES:
            try:
                self.client.armDisarm(False, vehicle_name=name)
                print(f"[OK] Disarmed: {name}", flush=True)
            except Exception as exc:
                cleanup_ok = False
                print(f"[WARN] Disarm failed: {name} ({exc})", flush=True)

            try:
                self.client.enableApiControl(False, vehicle_name=name)
                print(f"[OK] API control disabled: {name}", flush=True)
            except Exception as exc:
                cleanup_ok = False
                print(f"[WARN] Disable API control failed: {name} ({exc})", flush=True)

        if cleanup_ok:
            print("[OK] AirSimChaseEnv cleanup completed.", flush=True)
        else:
            print("[WARN] AirSimChaseEnv cleanup completed with warnings.", flush=True)

        self._closed = True
        return cleanup_ok

    def _zero_vehicle_motion(self, vehicle_name):
        try:
            self.client.moveByVelocityAsync(0.0, 0.0, 0.0, 0.1, vehicle_name=vehicle_name).join()
        except Exception:
            self._safe_hover(vehicle_name)

    def _settle_vehicle_after_reset(self, vehicle_name):
        self._zero_vehicle_motion(vehicle_name)
        self._safe_hover(vehicle_name)

    def _settle_all_after_reset(self):
        for vehicle_name in VEHICLES:
            self._settle_vehicle_after_reset(vehicle_name)

    def _remember_fast_reset_positions(self):
        if self.actual_chaser_start_pos is not None:
            self.fast_reset_chaser_pos = tuple(self.actual_chaser_start_pos)
        if self.actual_target_start_pos is not None:
            self.fast_reset_target_pos = tuple(self.actual_target_start_pos)

    def _reset_altitude_in_range(self, vehicle_name):
        position = self.get_global_position(vehicle_name)
        altitude = self._altitude_from_z(position.z_val)
        return RESET_ALTITUDE_MIN <= altitude <= RESET_ALTITUDE_MAX, position, altitude

    def _desired_reset_z_for_vehicle(self, vehicle_name):
        return self.chaser_start_z if vehicle_name == VEHICLE_CHASER else self.target_start_z

    def _ensure_reset_altitudes(self):
        for attempt in range(1, RESET_ALTITUDE_RETRY_COUNT + 1):
            all_ok = True
            for vehicle_name in VEHICLES:
                ok, position, altitude = self._reset_altitude_in_range(vehicle_name)
                if ok:
                    continue
                all_ok = False
                desired_z = self._desired_reset_z_for_vehicle(vehicle_name)
                print(
                    "[WARN] Reset altitude out of range; reapplying pose "
                    f"vehicle={vehicle_name} attempt={attempt} altitude={altitude:.2f} desired_z={desired_z:.2f}",
                    flush=True,
                )
                self._move_vehicle_to_start_with_fallback(
                    vehicle_name,
                    position.x_val,
                    position.y_val,
                    desired_z,
                )
                self._settle_vehicle_after_reset(vehicle_name)
            if all_ok:
                return True
        return False

    def _place_vehicles_after_full_reset(self):
        self._store_vehicle_home_globals()
        self._enable_api_and_arm()
        self._takeoff_all()
        self._move_all_to_safe_z()
        self._apply_requested_chaser_start()
        self._apply_requested_target_start()
        self._settle_all_after_reset()
        return self._ensure_reset_altitudes()

    def _full_reset_scene(self):
        try:
            self.client.reset()
            time.sleep(1.0)
        except Exception as exc:
            print(f"[WARN] AirSim reset failed, continuing with current scene: {exc}", flush=True)

        placement_ok = self._place_vehicles_after_full_reset()
        if not placement_ok:
            print("[WARN] Reset altitude validation failed; retrying with AirSim reset fallback.", flush=True)
            try:
                self.client.reset()
                time.sleep(1.0)
            except Exception as exc:
                print(f"[WARN] AirSim reset fallback failed, continuing with current scene: {exc}", flush=True)
            placement_ok = self._place_vehicles_after_full_reset()
            if not placement_ok:
                raise RuntimeError("Reset altitude validation failed after fallback reset.")
        self._print_reset_relative()
        time.sleep(1.0)
        self._remember_fast_reset_positions()
        self._full_reset_done = True
        self._force_full_reset_next = False

    def _fast_reset_scene(self):
        if self.fast_reset_chaser_pos is None or self.fast_reset_target_pos is None:
            return False

        self._enable_api_and_arm()
        for vehicle_name in VEHICLES:
            self._zero_vehicle_motion(vehicle_name)

        chaser_x, chaser_y, _ = self.fast_reset_chaser_pos
        target_x, target_y, _ = self.fast_reset_target_pos
        if self._chaser_start_requested():
            chaser_x = self.chaser_start_x if self.chaser_start_x is not None else chaser_x
            chaser_y = self.chaser_start_y if self.chaser_start_y is not None else chaser_y
        if self._target_start_requested() and not self._random_start_requested():
            target_x = self.target_start_x if self.target_start_x is not None else target_x
            target_y = self.target_start_y if self.target_start_y is not None else target_y
        chaser_z = self.chaser_start_z
        target_z = self.target_start_z
        chaser_pos = self._move_vehicle_to_start_with_fallback(VEHICLE_CHASER, chaser_x, chaser_y, chaser_z)
        target_pos = self._move_vehicle_to_start_with_fallback(VEHICLE_TARGET, target_x, target_y, target_z)
        self.actual_chaser_start_pos = (chaser_pos.x_val, chaser_pos.y_val, chaser_pos.z_val)
        self.actual_target_start_pos = (target_pos.x_val, target_pos.y_val, target_pos.z_val)
        if self._random_start_requested():
            self._apply_requested_target_start()
        self._settle_all_after_reset()
        if not self._ensure_reset_altitudes():
            print("[WARN] Fast reset altitude validation failed; falling back to full reset.", flush=True)
            return False
        self._print_reset_relative()
        self._remember_fast_reset_positions()
        self._force_full_reset_next = False
        return True

    def _enable_api_and_arm(self):
        for name in VEHICLES:
            self.client.enableApiControl(True, vehicle_name=name)
        for name in VEHICLES:
            self.client.armDisarm(True, vehicle_name=name)

    def _takeoff_all(self):
        for name in VEHICLES:
            self.client.takeoffAsync(vehicle_name=name).join()

    def _move_all_to_safe_z(self):
        for name in VEHICLES:
            # AirSim NED sisteminde yukari cikmak negatif Z degerine gitmek demektir.
            desired_z = self.chaser_start_z if name == VEHICLE_CHASER else self.target_start_z
            self.client.moveToZAsync(desired_z, self.chaser_speed, vehicle_name=name).join()

    def get_global_position(self, vehicle_name):
        # getMultirotorState position may be local per vehicle; for PPO observations
        # we use simGetObjectPose global/world pose.
        try:
            pose = self.client.simGetObjectPose(vehicle_name)
        except Exception as exc:
            raise RuntimeError(f"simGetObjectPose failed for {vehicle_name}: {exc}") from exc

        position = getattr(pose, "position", None)
        if position is None:
            raise RuntimeError(f"simGetObjectPose returned no position for {vehicle_name}.")

        values = [position.x_val, position.y_val, position.z_val]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Invalid global position for {vehicle_name}: {values}")

        return position

    def _target_start_requested(self):
        return (
            self._random_start_requested()
            or self.target_start_x is not None
            or self.target_start_y is not None
            or not math.isclose(self.target_start_z, SAFE_Z, abs_tol=1e-6)
        )

    def _chaser_start_requested(self):
        return (
            self.chaser_start_x is not None
            or self.chaser_start_y is not None
            or not math.isclose(self.chaser_start_z, SAFE_Z, abs_tol=1e-6)
        )

    def _any_start_requested(self):
        return self._chaser_start_requested() or self._target_start_requested()

    def _random_start_requested(self):
        return self.min_start_distance is not None or self.max_start_distance is not None

    def _chaser_start_tuple(self):
        if not self._chaser_start_requested():
            return None
        return (self.chaser_start_x, self.chaser_start_y, self.chaser_start_z)

    def _target_start_tuple(self):
        if not self._target_start_requested():
            return None
        if self._random_start_requested():
            return (
                "random_relative",
                self.min_start_distance,
                self.max_start_distance,
                self.target_start_z,
            )
        return (self.target_start_x, self.target_start_y, self.target_start_z)

    def _store_vehicle_home_globals(self):
        self.vehicle_home_global = {}
        for vehicle_name in VEHICLES:
            position = self.get_global_position(vehicle_name)
            self.vehicle_home_global[vehicle_name] = airsim.Vector3r(
                position.x_val,
                position.y_val,
                position.z_val,
            )
            print(
                "[PLACEMENT_HOME] "
                f"{vehicle_name} home_global=({position.x_val:.2f}, {position.y_val:.2f}, {position.z_val:.2f})",
                flush=True,
            )

    def _vehicle_home_global(self, vehicle_name):
        home_pos = self.vehicle_home_global.get(vehicle_name)
        if home_pos is not None:
            return home_pos

        position = self.get_global_position(vehicle_name)
        self.vehicle_home_global[vehicle_name] = airsim.Vector3r(
            position.x_val,
            position.y_val,
            position.z_val,
        )
        print(
            "[PLACEMENT_HOME] "
            f"{vehicle_name} home_global=({position.x_val:.2f}, {position.y_val:.2f}, {position.z_val:.2f})",
            flush=True,
        )
        return self.vehicle_home_global[vehicle_name]

    def _pose_for_position(self, position, orientation=None):
        if orientation is None:
            try:
                orientation = airsim.to_quaternion(0.0, 0.0, 0.0)
            except Exception:
                orientation = getattr(airsim, "Quaternionr")(0.0, 0.0, 0.0, 1.0)
        return airsim.Pose(position, orientation)

    def _placement_error(self, position, x_val, y_val, z_val):
        return (
            abs(position.x_val - x_val),
            abs(position.y_val - y_val),
            abs(position.z_val - z_val),
        )

    def _placement_close_enough(self, position, x_val, y_val, z_val):
        return max(self._placement_error(position, x_val, y_val, z_val)) <= START_PLACEMENT_TOLERANCE_METERS

    def _print_placement_result(self, log_prefix, vehicle_label, actual_pos, desired_x, desired_y, desired_z):
        dx_err, dy_err, dz_err = self._placement_error(actual_pos, desired_x, desired_y, desired_z)
        print(
            f"[{log_prefix}] "
            f"actual_{vehicle_label}_pos=({actual_pos.x_val:.2f}, {actual_pos.y_val:.2f}, {actual_pos.z_val:.2f})",
            flush=True,
        )
        print(
            f"[{log_prefix}] "
            f"placement_error=(dx={dx_err:.2f}, dy={dy_err:.2f}, dz={dz_err:.2f})",
            flush=True,
        )

    def _raise_placement_failed(self, vehicle_name, actual_pos, desired_x, desired_y, desired_z):
        message = (
            f"{vehicle_name} start placement failed: "
            f"requested=({desired_x:.2f}, {desired_y:.2f}, {desired_z:.2f}), "
            f"actual=({actual_pos.x_val:.2f}, {actual_pos.y_val:.2f}, {actual_pos.z_val:.2f})"
        )
        print(f"[ERROR] {message}", flush=True)
        raise RuntimeError(message)

    def _safe_hover(self, vehicle_name):
        try:
            self.client.hoverAsync(vehicle_name=vehicle_name).join()
        except Exception as exc:
            print(f"[WARN] Hover after placement command failed: {vehicle_name} ({exc})", flush=True)

    def _placement_velocity(self, reference_pos, desired_x, desired_y, desired_z):
        dx = desired_x - reference_pos.x_val
        dy = desired_y - reference_pos.y_val
        dz = desired_z - reference_pos.z_val
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        return min(START_PLACEMENT_MAX_SPEED, max(START_PLACEMENT_MIN_SPEED, distance / 5.0))

    def _try_pose_setter_calls(
        self,
        method_name,
        method,
        vehicle_name,
        pose,
        command_label,
        command_x,
        command_y,
        command_z,
        desired_x,
        desired_y,
        desired_z,
    ):
        if not callable(method):
            return None

        if method_name == "simSetVehiclePose":
            attempts = (
                lambda: method(pose, ignore_collision=True, vehicle_name=vehicle_name),
                lambda: method(pose, True, vehicle_name=vehicle_name),
                lambda: method(pose, True, vehicle_name),
            )
        else:
            attempts = (
                lambda: method(vehicle_name, pose, teleport=True),
                lambda: method(vehicle_name, pose, True),
                lambda: method(vehicle_name, pose),
            )

        last_exc = None
        for attempt in attempts:
            try:
                print(
                    "[PLACEMENT_POSE_CMD] "
                    f"{vehicle_name} method={method_name} mode={command_label} "
                    f"pose_cmd=({command_x:.2f}, {command_y:.2f}, {command_z:.2f})",
                    flush=True,
                )
                attempt()
            except Exception as exc:
                last_exc = exc
                continue

            time.sleep(0.25)
            self._safe_hover(vehicle_name)
            actual_pos = self.get_global_position(vehicle_name)
            if self._placement_close_enough(actual_pos, desired_x, desired_y, desired_z):
                return actual_pos

            dx_err, dy_err, dz_err = self._placement_error(actual_pos, desired_x, desired_y, desired_z)
            print(
                f"[WARN] {method_name} did not place {vehicle_name} close enough: "
                f"error=(dx={dx_err:.2f}, dy={dy_err:.2f}, dz={dz_err:.2f})",
                flush=True,
            )
            return None

        if last_exc is not None:
            print(f"[WARN] {method_name} failed for {vehicle_name} start: {last_exc}", flush=True)

        return None

    def _pose_command_variants(self, vehicle_name, desired_x, desired_y, desired_z):
        home_pos = self._vehicle_home_global(vehicle_name)
        candidates = [
            ("global", desired_x, desired_y, desired_z),
            ("home_plus", desired_x + home_pos.x_val, desired_y + home_pos.y_val, desired_z + home_pos.z_val),
            ("home_relative", desired_x - home_pos.x_val, desired_y - home_pos.y_val, desired_z - home_pos.z_val),
        ]

        variants = []
        for label, x_val, y_val, z_val in candidates:
            if any(
                abs(x_val - existing[1]) < 1e-6
                and abs(y_val - existing[2]) < 1e-6
                and abs(z_val - existing[3]) < 1e-6
                for existing in variants
            ):
                continue
            variants.append((label, x_val, y_val, z_val))

        return variants

    def _try_global_pose_placement(self, vehicle_name, desired_x, desired_y, desired_z):
        try:
            current_pose = self.client.simGetObjectPose(vehicle_name)
            orientation = getattr(current_pose, "orientation", None)
        except Exception as exc:
            print(f"[WARN] Could not build global {vehicle_name} pose. ({exc})", flush=True)
            return None

        for method_name in ("simSetVehiclePose", "simSetObjectPose"):
            method = getattr(self.client, method_name, None)
            for command_label, command_x, command_y, command_z in self._pose_command_variants(
                vehicle_name,
                desired_x,
                desired_y,
                desired_z,
            ):
                pose = self._pose_for_position(airsim.Vector3r(command_x, command_y, command_z), orientation)
                actual_pos = self._try_pose_setter_calls(
                    method_name,
                    method,
                    vehicle_name,
                    pose,
                    command_label,
                    command_x,
                    command_y,
                    command_z,
                    desired_x,
                    desired_y,
                    desired_z,
                )
                if actual_pos is not None:
                    return actual_pos

        return None

    def _try_move_to_position_placement(self, vehicle_name, desired_x, desired_y, desired_z):
        home_pos = self._vehicle_home_global(vehicle_name)
        velocity = self._placement_velocity(home_pos, desired_x, desired_y, desired_z)
        self._safe_hover(vehicle_name)
        local_x = desired_x - home_pos.x_val
        local_y = desired_y - home_pos.y_val
        local_z = desired_z - home_pos.z_val
        print(
            "[PLACEMENT_LOCAL_CMD] "
            f"{vehicle_name} local_cmd=({local_x:.2f}, {local_y:.2f}, {local_z:.2f})",
            flush=True,
        )
        self.client.moveToPositionAsync(
            local_x,
            local_y,
            local_z,
            velocity,
            vehicle_name=vehicle_name,
        ).join()
        self._safe_hover(vehicle_name)
        return self.get_global_position(vehicle_name)

    def _move_vehicle_to_start_with_fallback(self, vehicle_name, desired_x, desired_y, desired_z):
        self._safe_hover(vehicle_name)
        actual_pos = self._try_global_pose_placement(vehicle_name, desired_x, desired_y, desired_z)
        if actual_pos is None:
            actual_pos = self._try_move_to_position_placement(vehicle_name, desired_x, desired_y, desired_z)
        elif not self._placement_close_enough(actual_pos, desired_x, desired_y, desired_z):
            actual_pos = self._try_move_to_position_placement(vehicle_name, desired_x, desired_y, desired_z)

        return actual_pos

    def _apply_requested_chaser_start(self):
        chaser_pos = self.get_global_position(VEHICLE_CHASER)

        if not self._chaser_start_requested():
            self.actual_chaser_start_pos = (chaser_pos.x_val, chaser_pos.y_val, chaser_pos.z_val)
            return

        desired_x = self.chaser_start_x if self.chaser_start_x is not None else chaser_pos.x_val
        desired_y = self.chaser_start_y if self.chaser_start_y is not None else chaser_pos.y_val
        desired_z = self.chaser_start_z

        print(
            "[RESET_CHASER_START] "
            f"requested=(x={desired_x:.2f}, y={desired_y:.2f}, z={desired_z:.2f})",
            flush=True,
        )

        actual_pos = self._move_vehicle_to_start_with_fallback(
            VEHICLE_CHASER,
            desired_x,
            desired_y,
            desired_z,
        )
        self.actual_chaser_start_pos = (actual_pos.x_val, actual_pos.y_val, actual_pos.z_val)

        self._print_placement_result("RESET_CHASER_START", "chaser", actual_pos, desired_x, desired_y, desired_z)
        if not self._placement_close_enough(actual_pos, desired_x, desired_y, desired_z):
            self._raise_placement_failed(VEHICLE_CHASER, actual_pos, desired_x, desired_y, desired_z)

    def _apply_requested_target_start(self):
        chaser_pos = self.get_global_position(VEHICLE_CHASER)
        target_pos = self.get_global_position(VEHICLE_TARGET)

        if not self._target_start_requested():
            self.actual_target_start_pos = (target_pos.x_val, target_pos.y_val, target_pos.z_val)
            return

        if self._random_start_requested():
            min_distance = self.min_start_distance if self.min_start_distance is not None else self.max_start_distance
            max_distance = self.max_start_distance if self.max_start_distance is not None else self.min_start_distance
            distance = random.uniform(float(min_distance), float(max_distance))
            angle = random.uniform(0.0, 2.0 * math.pi) if self.random_start_angle else 0.0
            desired_x = chaser_pos.x_val + math.cos(angle) * distance
            desired_y = chaser_pos.y_val + math.sin(angle) * distance
            print(
                "[RESET_TARGET_RANDOM] "
                f"distance={distance:.2f} angle={angle:.2f}",
                flush=True,
            )
        elif self._chaser_start_requested():
            desired_x = self.target_start_x if self.target_start_x is not None else target_pos.x_val
            desired_y = self.target_start_y if self.target_start_y is not None else target_pos.y_val
        else:
            # Backward compatibility: when no Chaser start is requested,
            # target_start_x/y keep their previous meaning as Chaser-relative
            # dx/dy offsets.
            current_dx = target_pos.x_val - chaser_pos.x_val
            current_dy = target_pos.y_val - chaser_pos.y_val
            requested_dx = self.target_start_x if self.target_start_x is not None else current_dx
            requested_dy = self.target_start_y if self.target_start_y is not None else current_dy
            desired_x = chaser_pos.x_val + requested_dx
            desired_y = chaser_pos.y_val + requested_dy
        desired_z = self.target_start_z

        print(
            "[RESET_TARGET_START] "
            f"requested=(x={desired_x:.2f}, y={desired_y:.2f}, z={desired_z:.2f})",
            flush=True,
        )

        actual_pos = self._move_vehicle_to_start_with_fallback(
            VEHICLE_TARGET,
            desired_x,
            desired_y,
            desired_z,
        )
        self.actual_target_start_pos = (actual_pos.x_val, actual_pos.y_val, actual_pos.z_val)

        self._print_placement_result("RESET_TARGET_START", "target", actual_pos, desired_x, desired_y, desired_z)
        if not self._placement_close_enough(actual_pos, desired_x, desired_y, desired_z):
            self._raise_placement_failed(VEHICLE_TARGET, actual_pos, desired_x, desired_y, desired_z)

    def _print_reset_relative(self):
        chaser_pos = self.get_global_position(VEHICLE_CHASER)
        target_pos = self.get_global_position(VEHICLE_TARGET)
        dx, dy, dz, distance = self._compute_relative(chaser_pos, target_pos)
        print(
            "[RESET] "
            f"rel_global dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, distance={distance:.2f}",
            flush=True,
        )

    def _move_scripted_target(self):
        phase = self.step_count % 20
        if phase < 5:
            vx, vy = 1.0, 0.0
        elif phase < 10:
            vx, vy = 0.5, 0.8
        elif phase < 15:
            vx, vy = 1.0, 0.0
        else:
            vx, vy = 0.5, -0.8

        self.client.moveByVelocityAsync(
            vx * TARGET_SPEED,
            vy * TARGET_SPEED,
            0.0,
            self.step_duration,
            vehicle_name=VEHICLE_TARGET,
        ).join()

        target_pos = self.get_global_position(VEHICLE_TARGET)
        if abs(target_pos.z_val - self.target_start_z) > 1.5:
            self.client.moveToZAsync(self.target_start_z, 1.0, vehicle_name=VEHICLE_TARGET).join()

        self.last_target_velocity = (vx * TARGET_SPEED, vy * TARGET_SPEED, 0.0)

    def _move_straight_target(self):
        vx = self.target_base_speed
        vy = 0.0
        vz = 0.0
        self.client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            self.step_duration,
            vehicle_name=VEHICLE_TARGET,
        ).join()
        self.last_target_velocity = (vx, vy, vz)

    def _move_right_waypoint_target(self):
        phase = self.step_count % 80
        vx = self.target_base_speed
        if phase < 30:
            vy = 0.0
        elif phase < 60:
            vy = max(0.8, self.target_base_speed)
        else:
            vy = 0.25 * self.target_base_speed
        vz = 0.0
        self.client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            self.step_duration,
            vehicle_name=VEHICLE_TARGET,
        ).join()
        self.last_target_velocity = (vx, vy, vz)

    def _move_target_waypoint_route(self, chaser_pos=None):
        if not self.target_waypoints:
            self._move_right_waypoint_target()
            return

        target_pos = self.get_global_position(VEHICLE_TARGET)
        waypoint = self.target_waypoints[min(self.target_waypoint_index, len(self.target_waypoints) - 1)]
        dx = waypoint[0] - target_pos.x_val
        dy = waypoint[1] - target_pos.y_val
        dz = waypoint[2] - target_pos.z_val
        horizontal_distance = math.sqrt(dx * dx + dy * dy)

        if horizontal_distance < 4.0 and self.target_waypoint_index < len(self.target_waypoints) - 1:
            self.target_waypoint_index += 1
            waypoint = self.target_waypoints[self.target_waypoint_index]
            dx = waypoint[0] - target_pos.x_val
            dy = waypoint[1] - target_pos.y_val
            dz = waypoint[2] - target_pos.z_val
            horizontal_distance = math.sqrt(dx * dx + dy * dy)

        speed = self.target_base_speed
        if chaser_pos is not None:
            chase_dx = target_pos.x_val - chaser_pos.x_val
            chase_dy = target_pos.y_val - chaser_pos.y_val
            chase_distance = math.sqrt(chase_dx * chase_dx + chase_dy * chase_dy)
            if chase_distance < 30.0:
                speed = max(speed, self.target_escape_speed)
        if self.target_waypoint_index >= len(self.target_waypoints) - 1 and horizontal_distance < 5.0:
            speed = max(speed, self.target_base_speed * 0.8)

        if horizontal_distance > 0.001:
            vx = dx / horizontal_distance * speed
            vy = dy / horizontal_distance * speed
        else:
            vx = 0.0
            vy = 0.0
        target_z = waypoint[2]
        z_error = target_z - target_pos.z_val
        vz = max(-0.8, min(0.8, z_error * 0.8))

        self.client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            self.step_duration,
            vehicle_name=VEHICLE_TARGET,
        ).join()
        self.last_target_velocity = (vx, vy, vz)

    def _move_evasive_target(self, chaser_pos, target_pos):
        if self.target_controller is None:
            self.target_controller = TargetController(
                self.client,
                vehicle_name=VEHICLE_TARGET,
                safe_z=self.target_start_z,
                base_speed=self.target_base_speed,
                escape_speed=self.target_escape_speed,
                evade_distance=self.target_evade_distance,
                danger_distance=self.target_danger_distance,
            )

        target_lidar_info = self._get_lidar_info(vehicle_name=VEHICLE_TARGET)
        vx, vy, vz = self.target_controller.compute_target_velocity(
            chaser_pos,
            target_pos,
            lidar_sectors=target_lidar_info["sectors"],
            step_count=self.step_count,
        )
        self.client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            self.step_duration,
            vehicle_name=VEHICLE_TARGET,
        ).join()
        self.last_target_velocity = (vx, vy, vz)

    def _move_target(self, chaser_pos, target_pos):
        if self.target_mode == "evasive":
            self._move_evasive_target(chaser_pos, target_pos)
        elif self.target_mode == "straight":
            self._move_straight_target()
        elif self.target_mode == "right_waypoint":
            self._move_right_waypoint_target()
        elif self.target_mode == "right_escape":
            self._move_target_waypoint_route(chaser_pos=chaser_pos)
        else:
            self._move_scripted_target()

    def _bypass_clearances(self, lidar_sectors):
        left_clear = min(
            self._sector_distance(lidar_sectors, "left"),
            self._sector_distance(lidar_sectors, "front_left"),
        )
        right_clear = min(
            self._sector_distance(lidar_sectors, "right"),
            self._sector_distance(lidar_sectors, "front_right"),
        )
        return left_clear, right_clear

    def _choose_obstacle_bypass_direction(self, lidar_sectors, safe_action=None, requested_direction="none"):
        front_dist = self._front_obstacle_distance(lidar_sectors)
        if front_dist > DIAGONAL_BYPASS_RELEASE_FRONT_DISTANCE:
            self._reset_obstacle_bypass_state()
            return None

        if self.obstacle_bypass_direction and self.obstacle_bypass_steps_remaining > 0:
            self.obstacle_bypass_steps_remaining -= 1
            return self.obstacle_bypass_direction

        left_clear, right_clear = self._bypass_clearances(lidar_sectors)
        if requested_direction == "left" and left_clear > SIDE_DANGER_DISTANCE:
            direction = "left"
        elif requested_direction == "right" and right_clear > SIDE_DANGER_DISTANCE:
            direction = "right"
        elif safe_action == ACTION_MOVE_LEFT and left_clear > SIDE_DANGER_DISTANCE:
            direction = "left"
        elif safe_action == ACTION_MOVE_RIGHT and right_clear > SIDE_DANGER_DISTANCE:
            direction = "right"
        elif left_clear > right_clear + 0.5 and left_clear > SIDE_DANGER_DISTANCE:
            direction = "left"
        elif right_clear > SIDE_DANGER_DISTANCE:
            direction = "right"
        elif left_clear > SIDE_DANGER_DISTANCE:
            direction = "left"
        else:
            self._reset_obstacle_bypass_state()
            return None

        self.obstacle_bypass_direction = direction
        self.obstacle_bypass_steps_remaining = DIAGONAL_BYPASS_HOLD_STEPS
        return direction

    def _chaser_body_axes(self):
        yaw = self._vehicle_yaw(VEHICLE_CHASER)
        if yaw is None:
            yaw = 0.0
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        # AirSim NED/world Y pozitif sag kabul edildigi icin yaw=0 iken sol Y negatiftir.
        left_x = math.sin(yaw)
        left_y = -math.cos(yaw)
        return forward_x, forward_y, left_x, left_y

    def _emergency_bypass_velocity(self, direction, front_dist):
        forward_x, forward_y, left_x, left_y = self._chaser_body_axes()
        back_speed = EMERGENCY_BACK_SPEED if front_dist < EMERGENCY_FRONT_HARD_DISTANCE else EMERGENCY_SOFT_BACK_SPEED
        side_speed = min(max(EMERGENCY_SIDE_SPEED, self.chaser_speed * 0.55), self.chaser_speed)

        vx = -forward_x * back_speed
        vy = -forward_y * back_speed
        side_scale = 0.0
        if direction == "left":
            vx += left_x * side_speed
            vy += left_y * side_speed
            side_scale = side_speed / max(self.chaser_speed, 1e-6)
        elif direction == "right":
            vx -= left_x * side_speed
            vy -= left_y * side_speed
            side_scale = side_speed / max(self.chaser_speed, 1e-6)

        return vx, vy, 0.0, side_scale

    def _normal_bypass_velocity(self, direction, chaser_pos, target_pos):
        target_forward_x, target_forward_y, norm = self._target_direction_xy(chaser_pos, target_pos)
        if target_forward_x is None:
            return None

        _, _, left_x, left_y = self._chaser_body_axes()
        if direction == "left":
            side_x, side_y = left_x, left_y
        elif direction == "right":
            side_x, side_y = -left_x, -left_y
        else:
            side_x, side_y = 0.0, 0.0

        forward_speed = self.chaser_speed * DIAGONAL_BYPASS_FORWARD_SCALE
        side_speed = self.chaser_speed * DIAGONAL_BYPASS_SIDE_SCALE if direction in ("left", "right") else 0.0
        vx = target_forward_x * forward_speed + side_x * side_speed
        vy = target_forward_y * forward_speed + side_y * side_speed
        return vx, vy

    def _diagonal_bypass_velocity(self, safety_result, lidar_sectors, chaser_pos, target_pos):
        if safety_result is None or lidar_sectors is None:
            return None

        original_action = safety_result.get("original_action")
        safe_action = safety_result.get("safe_action")
        diagonal_requested = bool(safety_result.get("diagonal_bypass", False))
        requested_direction = safety_result.get("bypass_direction", "none")

        planner_state = self._gap_planner_state(lidar_sectors, chaser_pos, target_pos)
        front_dist = planner_state["front"]
        closest_front = planner_state["closest_front"]
        center_gap_safe = planner_state["center_gap_safe"]
        force_bypass = front_dist < DIAGONAL_BYPASS_FRONT_DISTANCE or (
            closest_front < DIAGONAL_BYPASS_FRONT_DISTANCE and not center_gap_safe
        )
        emergency_close = front_dist < DIAGONAL_BYPASS_EMERGENCY_DISTANCE or (
            closest_front < DIAGONAL_BYPASS_EMERGENCY_DISTANCE and not center_gap_safe
        )
        if original_action not in (ACTION_FORWARD_TO_TARGET, ACTION_HOVER) and not force_bypass:
            return None
        if front_dist > DIAGONAL_BYPASS_RELEASE_FRONT_DISTANCE:
            self._reset_obstacle_bypass_state()
            return None
        if front_dist >= DIAGONAL_BYPASS_FRONT_DISTANCE and not emergency_close and not diagonal_requested:
            return None

        direction = self._choose_obstacle_bypass_direction(lidar_sectors, safe_action, requested_direction)
        if direction is None:
            if emergency_close or front_dist < DIAGONAL_BYPASS_FRONT_DISTANCE:
                direction = "up" if emergency_close else "slow"
            else:
                return None

        if emergency_close:
            vx, vy, forward_scale, side_scale = self._emergency_bypass_velocity(direction, front_dist)
        else:
            normal_velocity = self._normal_bypass_velocity(direction, chaser_pos, target_pos)
            if normal_velocity is None:
                return None
            vx, vy = normal_velocity
            forward_scale = DIAGONAL_BYPASS_FORWARD_SCALE
            side_scale = DIAGONAL_BYPASS_SIDE_SCALE if direction in ("left", "right") else 0.0
        return vx, vy, direction, forward_scale, side_scale, self.obstacle_bypass_steps_remaining, emergency_close

    def _target_direction_xy(self, chaser_pos, target_pos):
        dx = target_pos.x_val - chaser_pos.x_val
        dy = target_pos.y_val - chaser_pos.y_val
        norm = math.sqrt(dx * dx + dy * dy)
        if norm < 1e-6:
            return None, None, norm
        return dx / norm, dy / norm, norm

    def _yaw_mode_to_target(self, chaser_pos, target_pos):
        dx = target_pos.x_val - chaser_pos.x_val
        dy = target_pos.y_val - chaser_pos.y_val
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return airsim.YawMode(is_rate=True, yaw_or_rate=0.0), 0.0
        desired_yaw_deg = math.degrees(math.atan2(dy, dx))
        return airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_deg), desired_yaw_deg

    def _is_stuck_near_obstacle(self, current_distance):
        if getattr(self, "last_effective_collision", False):
            return False
        if not self.distance_history:
            return False
        now = time.monotonic()
        window_start = now - STUCK_RECOVERY_SECONDS
        recent = [(timestamp, distance) for timestamp, distance in self.distance_history if timestamp >= window_start]
        if len(recent) < max(3, int(STUCK_RECOVERY_SECONDS / max(self.step_duration, 0.05)) // 2):
            return False
        oldest_distance = recent[0][1]
        return oldest_distance - float(current_distance) < STUCK_RECOVERY_MIN_PROGRESS

    def _record_distance_history(self, distance):
        now = time.monotonic()
        try:
            distance = float(distance)
        except Exception:
            return
        self.distance_history.append((now, distance))
        cutoff = now - (STUCK_RECOVERY_SECONDS + 1.0)
        self.distance_history = [item for item in self.distance_history if item[0] >= cutoff]

    def _obstacle_speed_scale(self, lidar_sectors):
        front_center = self._sector_distance(lidar_sectors, "front")
        front_left = self._sector_distance(lidar_sectors, "front_left")
        front_right = self._sector_distance(lidar_sectors, "front_right")
        center_gap_safe = (
            front_center >= GAP_CENTER_FRONT_SAFE_DISTANCE
            and front_left >= GAP_CENTER_SIDE_MARGIN_DISTANCE
            and front_right >= GAP_CENTER_SIDE_MARGIN_DISTANCE
        )
        obstacle_distance = front_center if center_gap_safe else min(front_center, front_left, front_right)
        if front_center > GLOBAL_SEARCH_FRONT_OPEN_DISTANCE:
            return 1.0, front_center
        if obstacle_distance < DIAGONAL_BYPASS_EMERGENCY_DISTANCE:
            return 0.0, obstacle_distance
        if obstacle_distance < DIAGONAL_BYPASS_FRONT_DISTANCE:
            return DIAGONAL_BYPASS_FORWARD_SCALE, obstacle_distance
        if front_center < OBSTACLE_OVERRIDE_DISTANCE:
            return 0.6, front_center
        if front_center < OBSTACLE_PREPARE_DISTANCE:
            return OBSTACLE_SOFT_SPEED_SCALE, front_center
        return 1.0, front_center

    def _apply_soft_obstacle_slowdown(self, vx, vy, command_info, lidar_sectors):
        if command_info.get("obstacle_bypass", False) or lidar_sectors is None:
            command_info["speed_scale"] = float(command_info.get("forward_scale", 1.0))
            return vx, vy, command_info
        speed_scale, _ = self._obstacle_speed_scale(lidar_sectors)
        if 0.0 < speed_scale < 1.0:
            vx *= speed_scale
            vy *= speed_scale
            command_info["speed_scale"] = float(speed_scale)
            command_info["forward_scale"] = float(speed_scale)
        else:
            command_info["speed_scale"] = 1.0
        return vx, vy, command_info

    def _apply_hover_drift(self, action, vx, vy, command_info, chaser_pos, target_pos):
        if action != ACTION_HOVER:
            return vx, vy, command_info
        forward_x, forward_y, distance = self._target_direction_xy(chaser_pos, target_pos)
        if forward_x is None or distance < HOVER_DRIFT_MIN_DISTANCE:
            return vx, vy, command_info
        speed_scale = float(command_info.get("speed_scale", 1.0))
        if speed_scale <= 0.0:
            return vx, vy, command_info
        drift_speed = HOVER_DRIFT_SPEED * min(1.0, speed_scale)
        vx = forward_x * drift_speed
        vy = forward_y * drift_speed
        command_info["hover_drift"] = True
        command_info["speed_scale"] = min(speed_scale, drift_speed / self.chaser_speed)
        return vx, vy, command_info

    def _limit_camera_centering_lateral(self, action, vx, vy, command_info, lidar_sectors, chaser_pos, target_pos):
        if lidar_sectors is None or command_info.get("obstacle_bypass", False):
            return vx, vy, command_info
        if action not in (ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT):
            return vx, vy, command_info

        front_dist = self._front_obstacle_distance(lidar_sectors)
        if front_dist < OBSTACLE_OVERRIDE_DISTANCE:
            return vx, vy, command_info

        forward_x, forward_y, distance = self._target_direction_xy(chaser_pos, target_pos)
        if forward_x is None:
            return vx, vy, command_info

        if distance > CAMERA_CENTERING_FAR_DISTANCE:
            forward_speed = self.chaser_speed
            vx = forward_x * forward_speed + vx * CAMERA_CENTERING_FAR_LATERAL_SCALE
            vy = forward_y * forward_speed + vy * CAMERA_CENTERING_FAR_LATERAL_SCALE
            command_info["camera_centering_limited"] = True
            command_info["camera_centering_reason"] = "distance>20; preferring forward target pursuit"
        elif distance > CAMERA_CENTERING_NEAR_DISTANCE:
            forward_speed = self.chaser_speed * (1.0 - CAMERA_CENTERING_MID_LATERAL_SCALE)
            vx = forward_x * forward_speed + vx * CAMERA_CENTERING_MID_LATERAL_SCALE
            vy = forward_y * forward_speed + vy * CAMERA_CENTERING_MID_LATERAL_SCALE
            command_info["camera_centering_limited"] = True
            command_info["camera_centering_reason"] = "distance>10; damping lateral centering"

        return vx, vy, command_info

    def _target_side_from_pose(self, chaser_pos, target_pos):
        try:
            lateral = float(self._capture_geometry(chaser_pos, target_pos).get("lateral", 0.0))
        except Exception:
            lateral = 0.0
        if lateral > GAP_TARGET_SIDE_DEADBAND:
            return "right", lateral
        if lateral < -GAP_TARGET_SIDE_DEADBAND:
            return "left", lateral
        return "center", lateral

    def _gap_planner_state(self, lidar_sectors, chaser_pos, target_pos):
        front_center = self._sector_distance(lidar_sectors, "front")
        front_left = self._sector_distance(lidar_sectors, "front_left")
        front_right = self._sector_distance(lidar_sectors, "front_right")
        left_clear = min(self._sector_distance(lidar_sectors, "left"), front_left)
        right_clear = min(self._sector_distance(lidar_sectors, "right"), front_right)
        closest_front = min(front_center, front_left, front_right)
        center_gap_safe = (
            front_center >= GAP_CENTER_FRONT_SAFE_DISTANCE
            and front_left >= GAP_CENTER_SIDE_MARGIN_DISTANCE
            and front_right >= GAP_CENTER_SIDE_MARGIN_DISTANCE
        )
        left_safe = left_clear >= GAP_SIDE_SAFE_DISTANCE
        right_safe = right_clear >= GAP_SIDE_SAFE_DISTANCE
        target_side, target_lateral = self._target_side_from_pose(chaser_pos, target_pos)

        if center_gap_safe:
            gap_direction = "center"
            gap_safe = True
        elif target_side == "right" and right_safe:
            gap_direction = "right"
            gap_safe = True
        elif target_side == "left" and left_safe:
            gap_direction = "left"
            gap_safe = True
        elif right_safe and right_clear > left_clear + 0.5:
            gap_direction = "right"
            gap_safe = True
        elif left_safe:
            gap_direction = "left"
            gap_safe = True
        elif right_safe:
            gap_direction = "right"
            gap_safe = True
        else:
            gap_direction = "up"
            gap_safe = False

        return {
            "front": front_center,
            "closest_front": closest_front,
            "front_center": front_center,
            "front_left": front_left,
            "front_right": front_right,
            "left_clear": left_clear,
            "right_clear": right_clear,
            "center_gap_safe": center_gap_safe,
            "left_safe": left_safe,
            "right_safe": right_safe,
            "target_side": target_side,
            "target_lateral": target_lateral,
            "gap_direction": gap_direction,
            "gap_safe": gap_safe,
        }

    def _apply_obstacle_aware_target_following(self, vx, vy, vz, command_info, lidar_sectors, chaser_pos, target_pos):
        if lidar_sectors is None:
            return vx, vy, vz, command_info

        state = self._gap_planner_state(lidar_sectors, chaser_pos, target_pos)
        front_dist = state["front"]
        front_left_dist = state["front_left"]
        front_right_dist = state["front_right"]
        left_dist = state["left_clear"]
        right_dist = state["right_clear"]
        target_side = state["target_side"]
        target_forward_x, target_forward_y, target_distance = self._target_direction_xy(chaser_pos, target_pos)
        body_forward_x, body_forward_y, body_left_x, body_left_y = self._chaser_body_axes()
        body_right_x, body_right_y = -body_left_x, -body_left_y

        if target_forward_x is None:
            target_forward_x, target_forward_y = body_forward_x, body_forward_y

        target_body_forward = target_forward_x * body_forward_x + target_forward_y * body_forward_y
        target_body_side = target_forward_x * body_right_x + target_forward_y * body_right_y
        if target_body_side > 0.2:
            target_side = "right"
        elif target_body_side < -0.2:
            target_side = "left"
        close_chase_mode = bool(target_distance is not None and target_distance < CLOSE_CHASE_DISTANCE)
        front_clear = front_dist >= GAP_CENTER_FRONT_SAFE_DISTANCE
        left_blocked = left_dist < GAP_SIDE_SAFE_DISTANCE
        right_blocked = right_dist < GAP_SIDE_SAFE_DISTANCE
        altitude = self._altitude_from_z(chaser_pos.z_val)
        up_allowed = altitude <= self.max_safe_altitude

        if front_dist < GAP_FRONT_EMERGENCY_DISTANCE:
            obstacle_reaction_zone = "emergency"
        elif front_dist < OBSTACLE_OVERRIDE_DISTANCE:
            obstacle_reaction_zone = "avoid"
        elif front_dist < OBSTACLE_PREPARE_DISTANCE:
            obstacle_reaction_zone = "watch"
        else:
            obstacle_reaction_zone = "none"

        reasons = []
        blocked_lateral = left_blocked or right_blocked
        stuck_recovery = bool(
            target_distance is not None
            and self._is_stuck_near_obstacle(target_distance)
            and (obstacle_reaction_zone != "none" or not state["gap_safe"] or command_info.get("obstacle_bypass", False))
        )

        if obstacle_reaction_zone == "none":
            forward_speed = self.chaser_speed
        elif obstacle_reaction_zone == "watch":
            forward_speed = self.chaser_speed * 0.9
        elif front_dist < GAP_FRONT_SLOW_DISTANCE:
            forward_speed = self.chaser_speed * 0.22
        else:
            forward_speed = self.chaser_speed * 0.6

        if target_distance is not None and target_distance > GLOBAL_SEARCH_DISTANCE and front_clear:
            forward_speed = max(forward_speed, GLOBAL_SEARCH_MIN_SPEED)

        side_speed = min(self.chaser_speed, max(2.0, self.chaser_speed * 0.65))
        diag_side_speed = min(side_speed, self.chaser_speed * 0.55)
        diag_forward_speed = forward_speed * (0.9 if front_dist >= GAP_FRONT_SLOW_DISTANCE else 0.35)
        side_forward_speed = 0.0 if front_dist < GAP_FRONT_SLOW_DISTANCE else self.chaser_speed * 0.25
        close_speed = 0.0
        if close_chase_mode:
            if target_distance < CLOSE_CHASE_SLOW_DISTANCE:
                close_speed = max(0.9, min(2.2, target_distance * 0.45))
            else:
                close_speed = max(3.0, min(self.chaser_speed * 0.9, target_distance * 0.45))

        candidates = []

        def add_candidate(name, forward, side, vertical, clearance, reason):
            candidates.append(
                {
                    "name": name,
                    "forward": float(forward),
                    "side": float(side),
                    "vz": float(vertical),
                    "clearance": float(clearance),
                    "reason": reason,
                }
            )

        add_candidate("forward", forward_speed, 0.0, vz, front_dist, "target_progress")
        add_candidate(
            "forward_left",
            diag_forward_speed,
            -diag_side_speed,
            vz,
            min(front_dist, front_left_dist, left_dist),
            "short_safe_path",
        )
        add_candidate(
            "forward_right",
            diag_forward_speed,
            diag_side_speed,
            vz,
            min(front_dist, front_right_dist, right_dist),
            "short_safe_path",
        )
        add_candidate("left", side_forward_speed, -side_speed, vz, left_dist, "front_blocked_side_escape")
        add_candidate("right", side_forward_speed, side_speed, vz, right_dist, "front_blocked_side_escape")

        if close_chase_mode and obstacle_reaction_zone in ("none", "watch"):
            close_forward = max(0.0, target_body_forward * close_speed)
            close_side = target_body_side * close_speed
            if close_side > 0.25:
                close_name = "forward_right" if close_forward > 0.2 else "right"
                close_clearance = min(front_dist, front_right_dist, right_dist)
            elif close_side < -0.25:
                close_name = "forward_left" if close_forward > 0.2 else "left"
                close_clearance = min(front_dist, front_left_dist, left_dist)
            else:
                close_name = "forward"
                close_clearance = front_dist
            add_candidate(close_name, close_forward, close_side, vz, close_clearance, "target_progress")

        emergency_climb_needed = front_dist < GAP_FRONT_EMERGENCY_DISTANCE and left_dist < GAP_FRONT_SLOW_DISTANCE and right_dist < GAP_FRONT_SLOW_DISTANCE
        if (emergency_climb_needed or stuck_recovery) and up_allowed:
            up_forward = STUCK_RECOVERY_FORWARD_SPEED if (front_dist > GAP_FRONT_SLOW_DISTANCE or stuck_recovery) else 0.0
            add_candidate("up", up_forward, 0.0, -0.8, min(front_dist, left_dist, right_dist), "emergency_climb")

        def candidate_valid(candidate):
            name = candidate["name"]
            side = candidate["side"]
            forward = candidate["forward"]
            if side > 0.0 and right_blocked:
                return False
            if side < 0.0 and left_blocked:
                return False
            if name == "up":
                if not up_allowed:
                    return False
                if not stuck_recovery and not emergency_climb_needed:
                    return False
                return True
            if front_dist < GAP_FRONT_EMERGENCY_DISTANCE and forward > 0.1:
                return False
            if name == "forward" and front_dist < GAP_FRONT_SLOW_DISTANCE:
                return False
            if name in ("forward_left", "left") and left_dist < GAP_FRONT_SLOW_DISTANCE:
                return False
            if name in ("forward_right", "right") and right_dist < GAP_FRONT_SLOW_DISTANCE:
                return False
            return True

        def candidate_score(candidate):
            name = candidate["name"]
            forward = candidate["forward"]
            side = candidate["side"]
            clearance = candidate["clearance"]
            progress = forward * target_body_forward + side * target_body_side
            score = progress * 4.0
            score += min(clearance, OBSTACLE_PREPARE_DISTANCE) * 0.35
            score += max(forward, 0.0) * 0.35
            score -= abs(side) * 0.08

            if name == "forward" and front_dist > OBSTACLE_OVERRIDE_DISTANCE:
                score += 2.0
            elif name.startswith("forward") and front_dist > GAP_FRONT_SLOW_DISTANCE:
                score += 2.0

            if front_clear and name == "forward":
                score += 1.5
                if abs(target_body_side) > 0.35:
                    score -= 1.5
            if front_clear and target_side == "right" and right_blocked and name == "forward":
                score += 3.0
            if front_clear and target_side == "left" and left_blocked and name == "forward":
                score += 3.0

            if side * target_body_side > 0.0:
                score += min(abs(side), self.chaser_speed) * abs(target_body_side) * 0.8

            if target_side == "right" and side > 0.0:
                score += 1.3
            elif target_side == "right" and side < 0.0:
                score += 0.8 if (right_blocked and front_dist < OBSTACLE_OVERRIDE_DISTANCE) else -1.0
            if target_side == "left" and side < 0.0:
                score += 1.3
            elif target_side == "left" and side > 0.0:
                score += 0.8 if (left_blocked and front_dist < OBSTACLE_OVERRIDE_DISTANCE) else -1.0

            if front_dist < OBSTACLE_OVERRIDE_DISTANCE and name == "forward":
                score -= 9.0
            if front_dist < OBSTACLE_OVERRIDE_DISTANCE and target_side == "right" and right_blocked and name in ("forward_left", "left"):
                score += 4.0
            if front_dist < OBSTACLE_OVERRIDE_DISTANCE and target_side == "left" and left_blocked and name in ("forward_right", "right"):
                score += 4.0
            if front_dist < OBSTACLE_OVERRIDE_DISTANCE and name in ("forward_left", "forward_right"):
                score += 1.0
            if front_dist < GAP_FRONT_SLOW_DISTANCE and name.startswith("forward"):
                score -= 4.0
            if clearance < GAP_FRONT_SLOW_DISTANCE and name != "up":
                score -= (GAP_FRONT_SLOW_DISTANCE - clearance) * 2.0
            if name == "up":
                score -= 8.0
                if emergency_climb_needed:
                    score += 6.0
                if stuck_recovery:
                    score += 8.0
            if close_chase_mode:
                score += progress * 2.0
                if name == "forward" and abs(target_body_side) > 0.35:
                    score -= 2.0
                if target_distance is not None and target_distance < CLOSE_CHASE_SLOW_DISTANCE and forward > close_speed:
                    score -= (forward - close_speed) * 2.0
            if progress < 0.0:
                score += progress * 2.0
            if forward < 0.2 and obstacle_reaction_zone not in ("emergency", "avoid"):
                score -= 2.0
            return score

        valid_candidates = [candidate for candidate in candidates if candidate_valid(candidate)]
        if valid_candidates:
            chosen = max(valid_candidates, key=candidate_score)
        elif up_allowed and (emergency_climb_needed or stuck_recovery):
            chosen = {
                "name": "up",
                "forward": STUCK_RECOVERY_FORWARD_SPEED if front_dist > GAP_FRONT_SLOW_DISTANCE else 0.0,
                "side": 0.0,
                "vz": -0.8,
                "clearance": min(front_dist, left_dist, right_dist),
                "reason": "emergency_climb",
            }
        else:
            fallback_name = "forward"
            fallback_side = 0.0
            if front_dist <= GAP_FRONT_SLOW_DISTANCE:
                if left_dist >= right_dist and not left_blocked:
                    fallback_name = "left"
                    fallback_side = -side_speed
                elif not right_blocked:
                    fallback_name = "right"
                    fallback_side = side_speed
            fallback_forward = 0.0 if front_dist < GAP_FRONT_EMERGENCY_DISTANCE else max(0.0, min(forward_speed, self.chaser_speed * 0.25))
            chosen = {
                "name": fallback_name,
                "forward": fallback_forward,
                "side": fallback_side,
                "vz": max(vz, 0.0) if not up_allowed else vz,
                "clearance": max(front_dist, left_dist, right_dist),
                "reason": "short_safe_path",
            }

        planner_choice = chosen["name"]
        chosen_reason = chosen["reason"]
        forward_component = chosen["forward"]
        side_component = chosen["side"]
        selected_vz = chosen["vz"]

        if planner_choice == "up":
            selected_vz = -0.8 if up_allowed else max(vz, 0.0)
        if altitude > self.max_safe_altitude and selected_vz < 0.0:
            selected_vz = max(vz, 0.0)
            if planner_choice == "up":
                planner_choice = "forward" if front_dist > GAP_FRONT_SLOW_DISTANCE else "left" if left_dist >= right_dist and not left_blocked else "right" if not right_blocked else "forward"
                chosen_reason = "short_safe_path"

        forward_detour = bool(
            planner_choice == "forward"
            and front_clear
            and ((target_side == "right" and right_blocked) or (target_side == "left" and left_blocked))
        )
        climb_avoidance = planner_choice == "up"
        gap_direction = {
            "forward": "center",
            "forward_left": "left",
            "forward_right": "right",
            "left": "left",
            "right": "right",
            "up": "up",
        }.get(planner_choice, "center")
        gap_safe = planner_choice != "up" and chosen["clearance"] >= GAP_FRONT_SLOW_DISTANCE

        if stuck_recovery and planner_choice == "up":
            chosen_reason = "emergency_climb"
            reasons.append("distance stalled for 5s; stuck recovery climb")
        if obstacle_reaction_zone in ("avoid", "emergency") and planner_choice in ("forward_left", "forward_right", "left", "right"):
            chosen_reason = "front_blocked_side_escape"
        if obstacle_reaction_zone != "none":
            reasons.append(f"front {front_dist:.2f}; zone={obstacle_reaction_zone}; choice={planner_choice}")
        if forward_detour:
            chosen_reason = "forward_detour"
            reasons.append("target side blocked; using forward_detour")
        if right_blocked and side_component > 0.0:
            side_component = 0.0
            reasons.append(f"right_lidar {right_dist:.2f}<8; blocking right lateral")
        if left_blocked and side_component < 0.0:
            side_component = 0.0
            reasons.append(f"left_lidar {left_dist:.2f}<8; blocking left lateral")
        if (
            close_chase_mode
            and target_distance is not None
            and target_distance < CLOSE_CHASE_SLOW_DISTANCE
            and planner_choice != "up"
            and obstacle_reaction_zone != "emergency"
        ):
            horizontal_component = math.sqrt(forward_component * forward_component + side_component * side_component)
            if horizontal_component > close_speed > 0.0:
                scale = close_speed / horizontal_component
                forward_component *= scale
                side_component *= scale
                chosen_reason = "target_progress"
                reasons.append(f"close chase distance {target_distance:.2f}<5; slowing without dropping target")

        vx = body_forward_x * forward_component + body_right_x * side_component
        vy = body_forward_y * forward_component + body_right_y * side_component
        vz = selected_vz

        if right_dist < GAP_SIDE_SAFE_DISTANCE and vy > 0.0:
            vy = 0.0
            blocked_lateral = True
            reasons.append(f"right_lidar {right_dist:.2f}<8; blocking positive final_vy")
        if left_dist < GAP_SIDE_SAFE_DISTANCE and vy < 0.0:
            vy = 0.0
            blocked_lateral = True
            reasons.append(f"left_lidar {left_dist:.2f}<8; blocking negative final_vy")

        limited = bool(
            obstacle_reaction_zone != "none"
            or blocked_lateral
            or forward_detour
            or climb_avoidance
            or stuck_recovery
            or command_info.get("camera_centering_limited", False)
        )
        if limited:
            command_info["camera_centering_limited"] = True
            previous_reason = command_info.get("camera_centering_reason", "")
            reason = "; ".join(reasons)
            command_info["camera_centering_reason"] = self._append_reason(previous_reason, reason)

        command_info["target_side"] = target_side
        command_info["close_chase_mode"] = bool(close_chase_mode)
        command_info["front_clear"] = bool(front_clear)
        command_info["left_blocked"] = bool(left_blocked)
        command_info["right_blocked"] = bool(right_blocked)
        command_info["planner_choice"] = planner_choice
        command_info["chosen_reason"] = chosen_reason
        command_info["obstacle_reaction_zone"] = obstacle_reaction_zone
        command_info["gap_direction"] = gap_direction
        command_info["gap_safe"] = bool(gap_safe)
        command_info["blocked_lateral"] = bool(blocked_lateral)
        command_info["forward_detour"] = bool(forward_detour)
        command_info["climb_avoidance"] = bool(climb_avoidance)
        command_info["stuck_recovery"] = bool(stuck_recovery)
        command_info["obstacle_bypass"] = bool(
            command_info.get("obstacle_bypass", False)
            or obstacle_reaction_zone in ("avoid", "emergency")
            or planner_choice in ("left", "right", "up", "forward_left", "forward_right")
        )
        command_info["emergency_avoidance"] = bool(
            command_info.get("emergency_avoidance", False)
            or obstacle_reaction_zone == "emergency"
        )
        command_info["forward_scale"] = max(0.0, min(1.0, forward_component / max(self.chaser_speed, 1e-6)))
        command_info["side_scale"] = max(0.0, min(1.0, abs(side_component) / max(self.chaser_speed, 1e-6)))
        command_info["speed_scale"] = command_info["forward_scale"]
        if gap_direction in ("left", "right", "up"):
            command_info["bypass_direction"] = gap_direction
        elif planner_choice == "forward":
            command_info["bypass_direction"] = "none"

        return vx, vy, vz, command_info

    def _smooth_chaser_velocity(self, vx, vy, vz, command_info):
        if command_info.get("emergency_avoidance", False):
            command_info["smooth_velocity"] = False
            command_info["pre_smooth_vx"] = float(vx)
            command_info["pre_smooth_vy"] = float(vy)
            command_info["pre_smooth_vz"] = float(vz)
            return vx, vy, vz, command_info

        last_vx, last_vy, last_vz = self.last_chaser_velocity
        smooth_vx = SMOOTH_VELOCITY_PREVIOUS_WEIGHT * last_vx + SMOOTH_VELOCITY_NEW_WEIGHT * vx
        smooth_vy = SMOOTH_VELOCITY_PREVIOUS_WEIGHT * last_vy + SMOOTH_VELOCITY_NEW_WEIGHT * vy
        smooth_vz = SMOOTH_VELOCITY_PREVIOUS_WEIGHT * last_vz + SMOOTH_VELOCITY_NEW_WEIGHT * vz
        command_info["smooth_velocity"] = True
        command_info["pre_smooth_vx"] = float(vx)
        command_info["pre_smooth_vy"] = float(vy)
        command_info["pre_smooth_vz"] = float(vz)
        return smooth_vx, smooth_vy, smooth_vz, command_info

    def _chaser_command_duration(self):
        return max(
            CHASER_COMMAND_MIN_DURATION,
            self.step_duration * CHASER_COMMAND_DURATION_SCALE,
        )

    def _apply_chaser_action(self, action, chaser_pos, target_pos, safety_result=None, lidar_sectors=None):
        command_info = self._default_command_info()
        diagonal_bypass = self._diagonal_bypass_velocity(safety_result, lidar_sectors, chaser_pos, target_pos)
        if diagonal_bypass is not None:
            vx, vy, direction, forward_scale, side_scale, bypass_steps_remaining, emergency_close = diagonal_bypass
            vz = EMERGENCY_UP_SPEED if emergency_close else 0.0
            command_info["obstacle_bypass"] = True
            command_info["diagonal_bypass"] = not emergency_close
            command_info["emergency_avoidance"] = bool(emergency_close)
            command_info["bypass_direction"] = direction
            command_info["bypass_steps_remaining"] = bypass_steps_remaining
            command_info["forward_scale"] = forward_scale
            command_info["side_scale"] = side_scale
        elif action == 0:
            forward_x, forward_y, norm = self._target_direction_xy(chaser_pos, target_pos)
            if forward_x is None:
                vx, vy = 0.0, 0.0
                command_info["hover_drift"] = True
            else:
                vx = self.chaser_speed * forward_x
                vy = self.chaser_speed * forward_y
            vz = 0.0
        elif action == 1:
            vx, vy, vz = 0.0, -self.chaser_speed, 0.0
        elif action == 2:
            vx, vy, vz = 0.0, self.chaser_speed, 0.0
        elif action == 3:
            vx, vy, vz = 0.0, 0.0, -1.0
        elif action == 4:
            vx, vy, vz = 0.0, 0.0, 1.0
        elif action == ACTION_HOVER:
            vx, vy, vz = 0.0, 0.0, 0.0
        else:
            action = ACTION_HOVER
            vx, vy, vz = 0.0, 0.0, 0.0

        if lidar_sectors is not None:
            front_obstacle_dist = self._front_obstacle_distance(lidar_sectors)
            if front_obstacle_dist < DIAGONAL_BYPASS_FRONT_DISTANCE and action in (
                ACTION_MOVE_LEFT,
                ACTION_MOVE_RIGHT,
                ACTION_MOVE_UP,
            ):
                command_info["obstacle_bypass"] = True
                command_info["emergency_avoidance"] = front_obstacle_dist < DIAGONAL_BYPASS_EMERGENCY_DISTANCE
                command_info["forward_scale"] = 0.0
                if action == ACTION_MOVE_LEFT:
                    command_info["bypass_direction"] = "left"
                    command_info["side_scale"] = 1.0
                elif action == ACTION_MOVE_RIGHT:
                    command_info["bypass_direction"] = "right"
                    command_info["side_scale"] = 1.0
                else:
                    command_info["bypass_direction"] = "up"
                    command_info["side_scale"] = 0.0
                if command_info["emergency_avoidance"]:
                    vz = min(vz, -0.4)

        vx, vy, command_info = self._apply_soft_obstacle_slowdown(vx, vy, command_info, lidar_sectors)
        vx, vy, command_info = self._apply_hover_drift(action, vx, vy, command_info, chaser_pos, target_pos)
        vx, vy, command_info = self._limit_camera_centering_lateral(
            action,
            vx,
            vy,
            command_info,
            lidar_sectors,
            chaser_pos,
            target_pos,
        )
        vx, vy, vz, command_info = self._clamp_chaser_velocity_for_altitude(vx, vy, vz, chaser_pos, command_info)
        vx, vy, vz, command_info = self._smooth_chaser_velocity(vx, vy, vz, command_info)
        vx, vy, vz, command_info = self._apply_obstacle_aware_target_following(
            vx,
            vy,
            vz,
            command_info,
            lidar_sectors,
            chaser_pos,
            target_pos,
        )
        command_info["vx"] = float(vx)
        command_info["vy"] = float(vy)
        command_info["vz"] = float(vz)
        command_info["final_vx"] = float(vx)
        command_info["final_vy"] = float(vy)
        command_info["final_vz"] = float(vz)
        self.last_chaser_velocity = (vx, vy, vz)
        self.last_chaser_command_info = command_info
        command_duration = self._chaser_command_duration()
        command_info["command_duration"] = float(command_duration)
        yaw_mode, desired_yaw_deg = self._yaw_mode_to_target(chaser_pos, target_pos)
        command_info["yaw_to_target_deg"] = float(desired_yaw_deg)
        self.last_chaser_future = self.client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            command_duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode,
            vehicle_name=VEHICLE_CHASER,
        )
        return command_info

    def _default_command_info(self):
        return {
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "altitude": self.target_altitude,
            "altitude_error": 0.0,
            "vz_alt_hold": 0.0,
            "final_vz": 0.0,
            "final_vx": 0.0,
            "final_vy": 0.0,
            "obstacle_bypass": False,
            "diagonal_bypass": False,
            "emergency_avoidance": False,
            "bypass_direction": "none",
            "bypass_steps_remaining": 0,
            "forward_scale": 1.0,
            "side_scale": 0.0,
            "speed_scale": 1.0,
            "camera_centering_limited": False,
            "camera_centering_reason": "",
            "gap_direction": "center",
            "gap_safe": True,
            "target_side": "center",
            "front_clear": True,
            "left_blocked": False,
            "right_blocked": False,
            "close_chase_mode": False,
            "planner_choice": "forward",
            "chosen_reason": "target_progress",
            "obstacle_reaction_zone": "none",
            "blocked_lateral": False,
            "forward_detour": False,
            "climb_avoidance": False,
            "stuck_recovery": False,
            "smooth_velocity": False,
            "hover_drift": False,
            "pre_smooth_vx": 0.0,
            "pre_smooth_vy": 0.0,
            "pre_smooth_vz": 0.0,
            "command_duration": 0.0,
            "yaw_to_target_deg": 0.0,
            "altitude_safety_override": False,
            "too_high": False,
            "reason": "",
        }

    def _altitude_hold_vz(self, altitude):
        altitude_error = self.target_altitude - altitude
        vz_alt_hold = -ALTITUDE_HOLD_KP * altitude_error
        vz_alt_hold = max(-ALTITUDE_HOLD_MAX_VZ, min(ALTITUDE_HOLD_MAX_VZ, vz_alt_hold))
        return altitude_error, vz_alt_hold

    def _clamp_chaser_velocity_for_altitude(self, vx, vy, vz, chaser_pos, command_info=None):
        base_command_info = command_info or {}
        altitude = self._altitude_from_z(chaser_pos.z_val)
        altitude_error, vz_alt_hold = self._altitude_hold_vz(altitude)
        base_vz = vz
        final_vz = vz_alt_hold
        if (
            (base_command_info.get("emergency_avoidance", False) or base_command_info.get("climb_avoidance", False))
            and base_vz < 0.0
        ):
            final_vz = min(final_vz, base_vz)
        command_info = {
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(final_vz),
            "final_vx": float(vx),
            "final_vy": float(vy),
            "altitude": altitude,
            "altitude_error": altitude_error,
            "vz_alt_hold": float(vz_alt_hold),
            "final_vz": float(final_vz),
            "base_vz": float(base_vz),
            "obstacle_bypass": bool(base_command_info.get("obstacle_bypass", False)),
            "diagonal_bypass": bool(base_command_info.get("diagonal_bypass", False)),
            "emergency_avoidance": bool(base_command_info.get("emergency_avoidance", False)),
            "bypass_direction": base_command_info.get("bypass_direction", "none"),
            "bypass_steps_remaining": int(base_command_info.get("bypass_steps_remaining", 0)),
            "forward_scale": float(base_command_info.get("forward_scale", 1.0)),
            "side_scale": float(base_command_info.get("side_scale", 0.0)),
            "speed_scale": float(base_command_info.get("speed_scale", base_command_info.get("forward_scale", 1.0))),
            "camera_centering_limited": bool(base_command_info.get("camera_centering_limited", False)),
            "camera_centering_reason": base_command_info.get("camera_centering_reason", ""),
            "gap_direction": base_command_info.get("gap_direction", "center"),
            "gap_safe": bool(base_command_info.get("gap_safe", True)),
            "target_side": base_command_info.get("target_side", "center"),
            "front_clear": bool(base_command_info.get("front_clear", True)),
            "left_blocked": bool(base_command_info.get("left_blocked", False)),
            "right_blocked": bool(base_command_info.get("right_blocked", False)),
            "close_chase_mode": bool(base_command_info.get("close_chase_mode", False)),
            "planner_choice": base_command_info.get("planner_choice", "forward"),
            "chosen_reason": base_command_info.get("chosen_reason", "target_progress"),
            "obstacle_reaction_zone": base_command_info.get("obstacle_reaction_zone", "none"),
            "blocked_lateral": bool(base_command_info.get("blocked_lateral", False)),
            "forward_detour": bool(base_command_info.get("forward_detour", False)),
            "climb_avoidance": bool(base_command_info.get("climb_avoidance", False)),
            "stuck_recovery": bool(base_command_info.get("stuck_recovery", False)),
            "smooth_velocity": bool(base_command_info.get("smooth_velocity", False)),
            "hover_drift": bool(base_command_info.get("hover_drift", False)),
            "pre_smooth_vx": float(base_command_info.get("pre_smooth_vx", vx)),
            "pre_smooth_vy": float(base_command_info.get("pre_smooth_vy", vy)),
            "pre_smooth_vz": float(base_command_info.get("pre_smooth_vz", base_vz)),
            "command_duration": float(base_command_info.get("command_duration", 0.0)),
            "yaw_to_target_deg": float(base_command_info.get("yaw_to_target_deg", 0.0)),
            "altitude_safety_override": False,
            "too_high": bool(self.enable_altitude_safety and altitude > self.hard_max_altitude),
            "reason": "",
        }
        if not self.enable_altitude_safety:
            command_info["vz"] = float(base_vz)
            command_info["final_vz"] = float(base_vz)
            command_info["too_high"] = False
            return vx, vy, base_vz, command_info

        if altitude > self.max_safe_altitude:
            old_vz = final_vz
            final_vz = max(final_vz, 1.0)
            command_info["altitude_safety_override"] = True
            command_info["reason"] = (
                f"altitude {altitude:.2f}m above max_safe_altitude {self.max_safe_altitude:.2f}m; "
                f"vz {old_vz:.2f}->{final_vz:.2f}"
            )
        elif altitude < self.min_safe_altitude:
            old_vz = final_vz
            final_vz = min(final_vz, -1.0)
            command_info["altitude_safety_override"] = True
            command_info["reason"] = (
                f"altitude {altitude:.2f}m below min_safe_altitude {self.min_safe_altitude:.2f}m; "
                f"vz {old_vz:.2f}->{final_vz:.2f}"
            )

        command_info["vx"] = float(vx)
        command_info["vy"] = float(vy)
        command_info["vz"] = float(final_vz)
        command_info["final_vx"] = float(vx)
        command_info["final_vy"] = float(vy)
        command_info["final_vz"] = float(final_vz)
        return vx, vy, final_vz, command_info

    def _vector_to_tuple(self, value):
        if value is None:
            return None

        try:
            return (float(value.x_val), float(value.y_val), float(value.z_val))
        except Exception:
            return None

    def _get_collision_info(self):
        try:
            collision_info = self.client.simGetCollisionInfo(vehicle_name=VEHICLE_CHASER)
        except Exception:
            return {
                "has_collided": False,
                "object_name": "",
                "position": None,
                "normal": None,
                "impact_point": None,
            }

        return {
            "has_collided": bool(getattr(collision_info, "has_collided", False)),
            "object_name": getattr(collision_info, "object_name", "") or "",
            "position": self._vector_to_tuple(getattr(collision_info, "position", None)),
            "normal": self._vector_to_tuple(getattr(collision_info, "normal", None)),
            "impact_point": self._vector_to_tuple(getattr(collision_info, "impact_point", None)),
        }

    def _get_lidar_points(self, vehicle_name=VEHICLE_CHASER, lidar_name=LIDAR_NAME):
        try:
            lidar_data = self.client.getLidarData(lidar_name=lidar_name, vehicle_name=vehicle_name)
        except Exception as exc:
            if not self._lidar_error_warned:
                print(f"[WARN] LiDAR read failed: {vehicle_name}/{lidar_name} ({exc})", flush=True)
                self._lidar_error_warned = True
            return np.empty((0, 3), dtype=np.float32)

        point_cloud = getattr(lidar_data, "point_cloud", None)
        if point_cloud is None or len(point_cloud) == 0:
            if not self._lidar_empty_warned:
                print(f"[WARN] LiDAR point cloud is empty: {vehicle_name}/{lidar_name}", flush=True)
                self._lidar_empty_warned = True
            return np.empty((0, 3), dtype=np.float32)

        point_array = np.asarray(point_cloud, dtype=np.float32)
        usable_size = (point_array.size // 3) * 3
        if usable_size == 0:
            return np.empty((0, 3), dtype=np.float32)

        return point_array[:usable_size].reshape((-1, 3))

    def _compute_lidar_sectors(self, points):
        sectors = {name: MAX_LIDAR_DISTANCE for name in LIDAR_SECTOR_NAMES}

        if points.size == 0:
            return sectors

        for x_val, y_val, z_val in points:
            if not all(math.isfinite(float(value)) for value in (x_val, y_val, z_val)):
                continue

            x_val = float(x_val)
            y_val = float(y_val)
            z_val = float(z_val)
            distance = math.sqrt(x_val * x_val + y_val * y_val)

            if distance <= 0.2 or distance > MAX_LIDAR_DISTANCE:
                continue
            if abs(z_val) > 5.0:
                continue

            if x_val > 0.0 and abs(y_val) <= 2.5:
                sectors["front"] = min(sectors["front"], distance)
            if x_val > 0.0 and y_val < -2.0:
                sectors["front_left"] = min(sectors["front_left"], distance)
            if x_val > 0.0 and y_val > 2.0:
                sectors["front_right"] = min(sectors["front_right"], distance)
            if abs(x_val) <= 3.0 and y_val < -2.0:
                sectors["left"] = min(sectors["left"], distance)
            if abs(x_val) <= 3.0 and y_val > 2.0:
                sectors["right"] = min(sectors["right"], distance)
            if x_val < -1.0:
                sectors["back"] = min(sectors["back"], distance)
            if z_val < -1.5:
                bottom_dist = math.sqrt(x_val * x_val + y_val * y_val + z_val * z_val)
                sectors["bottom"] = min(sectors["bottom"], bottom_dist)

        return sectors

    def _get_lidar_info(self, vehicle_name=VEHICLE_CHASER, lidar_name=LIDAR_NAME):
        points = self._get_lidar_points(vehicle_name=vehicle_name, lidar_name=lidar_name)
        sectors = self._compute_lidar_sectors(points)
        return {
            "available": points.shape[0] > 0,
            "point_count": int(points.shape[0]),
            "sectors": sectors,
        }

    def _compute_relative(self, chaser_pos, target_pos):
        dx = target_pos.x_val - chaser_pos.x_val
        dy = target_pos.y_val - chaser_pos.y_val
        dz = target_pos.z_val - chaser_pos.z_val
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        return dx, dy, dz, distance

    def _normalize_distance(self, value):
        if value is None or not math.isfinite(float(value)):
            value = 0.0
        return float(value) / MAX_DIST

    def _normalize_velocity(self, value):
        if value is None or not math.isfinite(float(value)):
            value = 0.0
        return float(value) / MAX_SPEED

    def _normalize_lidar_distance(self, distance):
        if distance is None or not math.isfinite(float(distance)):
            distance = MAX_LIDAR_DISTANCE
        return min(float(distance), MAX_LIDAR) / MAX_LIDAR

    def _normalize_legacy_lidar_distance(self, distance):
        if distance is None or not math.isfinite(float(distance)):
            distance = MAX_LIDAR_DISTANCE
        return min(float(distance), MAX_LIDAR_DISTANCE) / MAX_LIDAR_DISTANCE

    def _normalize_action(self, action):
        if action is None:
            return 0.0
        try:
            action = int(action)
        except Exception:
            return 0.0
        if MAX_ACTIONS <= 1:
            return 0.0
        return (float(action) / float(MAX_ACTIONS - 1)) * 2.0 - 1.0

    def _make_observation(
        self,
        dx,
        dy,
        dz,
        distance,
        chaser_pos,
        target_pos,
        lidar_sectors,
        safety_overridden=False,
        previous_action=None,
    ):
        if self.obs_mode == "legacy14":
            return self._make_legacy14_observation(dx, dy, dz, distance, chaser_pos, target_pos, lidar_sectors)
        return self._make_extended26_observation(
            dx,
            dy,
            dz,
            distance,
            chaser_pos,
            target_pos,
            lidar_sectors,
            safety_overridden=safety_overridden,
            previous_action=previous_action,
        )

    def _make_legacy14_observation(self, dx, dy, dz, distance, chaser_pos, target_pos, lidar_sectors):
        return np.array(
            [
                dx / 100.0,
                dy / 100.0,
                dz / 100.0,
                distance / 100.0,
                chaser_pos.x_val / 100.0,
                chaser_pos.y_val / 100.0,
                target_pos.x_val / 100.0,
                target_pos.y_val / 100.0,
                self._normalize_legacy_lidar_distance(lidar_sectors.get("front", MAX_LIDAR_DISTANCE)),
                self._normalize_legacy_lidar_distance(lidar_sectors.get("front_left", MAX_LIDAR_DISTANCE)),
                self._normalize_legacy_lidar_distance(lidar_sectors.get("front_right", MAX_LIDAR_DISTANCE)),
                self._normalize_legacy_lidar_distance(lidar_sectors.get("left", MAX_LIDAR_DISTANCE)),
                self._normalize_legacy_lidar_distance(lidar_sectors.get("right", MAX_LIDAR_DISTANCE)),
                self._normalize_legacy_lidar_distance(lidar_sectors.get("back", MAX_LIDAR_DISTANCE)),
            ],
            dtype=np.float32,
        )

    def _make_extended26_observation(
        self,
        dx,
        dy,
        dz,
        distance,
        chaser_pos,
        target_pos,
        lidar_sectors,
        safety_overridden=False,
        previous_action=None,
    ):
        chaser_vx, chaser_vy, chaser_vz = self.last_chaser_velocity
        target_vx, target_vy, target_vz = self.last_target_velocity
        rel_vx = target_vx - chaser_vx
        rel_vy = target_vy - chaser_vy
        rel_vz = target_vz - chaser_vz
        min_lidar = self._min_lidar_distance(lidar_sectors)

        obs = np.array(
            [
                self._normalize_distance(dx),
                self._normalize_distance(dy),
                self._normalize_distance(dz),
                self._normalize_distance(distance),
                self._normalize_distance(chaser_pos.x_val),
                self._normalize_distance(chaser_pos.y_val),
                self._normalize_distance(target_pos.x_val),
                self._normalize_distance(target_pos.y_val),
                self._normalize_lidar_distance(lidar_sectors.get("front", MAX_LIDAR_DISTANCE)),
                self._normalize_lidar_distance(lidar_sectors.get("front_left", MAX_LIDAR_DISTANCE)),
                self._normalize_lidar_distance(lidar_sectors.get("front_right", MAX_LIDAR_DISTANCE)),
                self._normalize_lidar_distance(lidar_sectors.get("left", MAX_LIDAR_DISTANCE)),
                self._normalize_lidar_distance(lidar_sectors.get("right", MAX_LIDAR_DISTANCE)),
                self._normalize_lidar_distance(lidar_sectors.get("back", MAX_LIDAR_DISTANCE)),
                self._normalize_velocity(chaser_vx),
                self._normalize_velocity(chaser_vy),
                self._normalize_velocity(chaser_vz),
                self._normalize_velocity(target_vx),
                self._normalize_velocity(target_vy),
                self._normalize_velocity(target_vz),
                self._normalize_velocity(rel_vx),
                self._normalize_velocity(rel_vy),
                self._normalize_velocity(rel_vz),
                self._normalize_lidar_distance(min_lidar),
                1.0 if safety_overridden else 0.0,
                self._normalize_action(previous_action),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _make_info(
        self,
        action,
        dx,
        dy,
        dz,
        distance,
        reward,
        terminated,
        truncated,
        raw_collision,
        collision,
        collision_info,
        lidar_info,
        safety_result,
        reward_breakdown,
        caught,
        too_far,
        too_high,
        terminated_reason,
        chaser_pos,
        target_pos,
        capture_state=None,
        safety_bypassed_for_capture=False,
        override_count_recent=None,
    ):
        if capture_state is None:
            capture_state = self._default_capture_state()
        if override_count_recent is None:
            override_count_recent = self._recent_safety_override_count()
        action_name = ACTION_NAMES.get(action, "UNKNOWN") if action is not None else "RESET"
        lidar_sectors = lidar_info["sectors"]
        min_lidar = self._min_lidar_distance(lidar_sectors)
        safe_action = safety_result["safe_action"]
        chaser_vx, chaser_vy, chaser_vz = self.last_chaser_velocity
        target_vx, target_vy, target_vz = self.last_target_velocity
        altitude = self._altitude_from_z(chaser_pos.z_val)
        altitude_error = self.target_altitude - altitude
        return {
            "step": self.step_count,
            "action": action,
            "action_name": action_name,
            "original_action": safety_result["original_action"],
            "final_action": safe_action,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "distance": distance,
            "final_distance": distance,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "caught": caught,
            "capture_box": bool(capture_state.get("capture_box", False)),
            "distance_caught": bool(capture_state.get("distance_caught", False)),
            "done_reason": terminated_reason,
            "capture_forward": float(capture_state.get("capture_forward", 0.0)),
            "capture_lateral": float(capture_state.get("capture_lateral", 0.0)),
            "capture_vertical": float(capture_state.get("capture_vertical", 0.0)),
            "too_far": too_far,
            "too_high": bool(too_high),
            "terminated_reason": terminated_reason,
            "raw_collision": raw_collision,
            "collision": collision,
            "collision_object_name": collision_info.get("object_name", ""),
            "collision_position": collision_info.get("position"),
            "collision_normal": collision_info.get("normal"),
            "collision_impact_point": collision_info.get("impact_point"),
            "chaser_pos": (chaser_pos.x_val, chaser_pos.y_val, chaser_pos.z_val),
            "target_pos": (target_pos.x_val, target_pos.y_val, target_pos.z_val),
            "chaser_vx": chaser_vx,
            "chaser_vy": chaser_vy,
            "chaser_vz": chaser_vz,
            "target_mode": self.target_mode,
            "target_vx": target_vx,
            "target_vy": target_vy,
            "target_vz": target_vz,
            "target_base_speed": self.target_base_speed,
            "target_escape_speed": self.target_escape_speed,
            "target_waypoint_index": self.target_waypoint_index,
            "target_waypoint_total": len(self.target_waypoints),
            "step_duration": self.step_duration,
            "chaser_speed": self.chaser_speed,
            "reward_mode": self.reward_mode,
            "obs_mode": self.obs_mode,
            "use_capture_box": self.use_capture_box,
            "capture_depth": self.capture_depth,
            "capture_width": self.capture_width,
            "capture_height": self.capture_height,
            "catch_radius": self.catch_radius,
            "use_fast_reset": self.use_fast_reset,
            "chaser_start_x": self.chaser_start_x,
            "chaser_start_y": self.chaser_start_y,
            "chaser_start_z": self.chaser_start_z,
            "requested_chaser_start": self._chaser_start_tuple(),
            "actual_chaser_start_pos": self.actual_chaser_start_pos,
            "target_start_x": self.target_start_x,
            "target_start_y": self.target_start_y,
            "target_start_z": self.target_start_z,
            "requested_target_start": self._target_start_tuple(),
            "actual_target_start_pos": self.actual_target_start_pos,
            "min_start_distance": self.min_start_distance,
            "max_start_distance": self.max_start_distance,
            "random_start_angle": self.random_start_angle,
            "max_episode_steps": self.max_episode_steps,
            "too_far_distance": self.too_far_distance,
            "lidar_available": lidar_info["available"],
            "lidar_point_count": lidar_info["point_count"],
            "lidar_front": lidar_sectors["front"],
            "lidar_front_left": lidar_sectors["front_left"],
            "lidar_front_right": lidar_sectors["front_right"],
            "lidar_left": lidar_sectors["left"],
            "lidar_right": lidar_sectors["right"],
            "lidar_back": lidar_sectors["back"],
            "min_lidar": min_lidar,
            "obs_shape": (self.observation_size,),
            "safety_original_action": safety_result["original_action"],
            "safety_original_action_name": action_name,
            "safety_safe_action": safe_action,
            "safety_safe_action_name": ACTION_NAMES.get(safe_action, "UNKNOWN") if safe_action is not None else "RESET",
            "safety_overridden": safety_result["overridden"],
            "safety_override": safety_result["overridden"],
            "safety_reason": safety_result["reason"],
            "safety_risk_level": safety_result["risk_level"],
            "safety_bypassed_for_capture": bool(safety_bypassed_for_capture),
            "altitude": altitude,
            "target_altitude": self.target_altitude,
            "min_safe_altitude": self.min_safe_altitude,
            "max_safe_altitude": self.max_safe_altitude,
            "hard_max_altitude": self.hard_max_altitude,
            "enable_altitude_safety": self.enable_altitude_safety,
            "altitude_error": altitude_error,
            "altitude_safety_override": bool(safety_result.get("altitude_safety_override", False)),
            "chaser_z": chaser_pos.z_val,
            "vx": chaser_vx,
            "vy": chaser_vy,
            "vz": chaser_vz,
            "vz_alt_hold": float(safety_result.get("vz_alt_hold", 0.0)),
            "final_vx": float(safety_result.get("final_vx", chaser_vx)),
            "final_vy": float(safety_result.get("final_vy", chaser_vy)),
            "final_vz": float(safety_result.get("final_vz", chaser_vz)),
            "base_vz": float(safety_result.get("base_vz", 0.0)),
            "obstacle_bypass": bool(safety_result.get("obstacle_bypass", False)),
            "diagonal_bypass": bool(safety_result.get("diagonal_bypass", False)),
            "emergency_avoidance": bool(
                safety_result.get("emergency_avoidance", safety_result.get("emergency_avoid", False))
            ),
            "bypass_direction": safety_result.get("bypass_direction", "none"),
            "forward_scale": float(safety_result.get("forward_scale", 1.0)),
            "side_scale": float(safety_result.get("side_scale", 0.0)),
            "speed_scale": float(safety_result.get("speed_scale", 1.0)),
            "camera_centering_limited": bool(safety_result.get("camera_centering_limited", False)),
            "camera_centering_reason": safety_result.get("camera_centering_reason", ""),
            "gap_direction": safety_result.get("gap_direction", "center"),
            "gap_safe": bool(safety_result.get("gap_safe", True)),
            "target_side": safety_result.get("target_side", "center"),
            "front_clear": bool(safety_result.get("front_clear", True)),
            "left_blocked": bool(safety_result.get("left_blocked", False)),
            "right_blocked": bool(safety_result.get("right_blocked", False)),
            "close_chase_mode": bool(safety_result.get("close_chase_mode", False)),
            "planner_choice": safety_result.get("planner_choice", "forward"),
            "chosen_reason": safety_result.get("chosen_reason", "target_progress"),
            "obstacle_reaction_zone": safety_result.get("obstacle_reaction_zone", "none"),
            "blocked_lateral": bool(safety_result.get("blocked_lateral", False)),
            "forward_detour": bool(safety_result.get("forward_detour", False)),
            "climb_avoidance": bool(safety_result.get("climb_avoidance", False)),
            "stuck_recovery": bool(safety_result.get("stuck_recovery", False)),
            "smooth_velocity": bool(safety_result.get("smooth_velocity", False)),
            "hover_drift": bool(safety_result.get("hover_drift", False)),
            "pre_smooth_vx": float(safety_result.get("pre_smooth_vx", 0.0)),
            "pre_smooth_vy": float(safety_result.get("pre_smooth_vy", 0.0)),
            "pre_smooth_vz": float(safety_result.get("pre_smooth_vz", 0.0)),
            "command_duration": float(safety_result.get("command_duration", 0.0)),
            "yaw_to_target_deg": float(safety_result.get("yaw_to_target_deg", 0.0)),
            "override_count_recent": int(override_count_recent),
            "safety_override_count": self.episode_safety_override_count,
            "min_lidar_mean": self._episode_min_lidar_mean(),
            "min_lidar_min": self.episode_min_lidar_min,
            "bypass_active": safety_result.get("bypass_active", False),
            "bypass_action": safety_result.get("bypass_action"),
            "bypass_action_name": safety_result.get("bypass_action_name", "none"),
            "bypass_steps_remaining": safety_result.get("bypass_steps_remaining", 0),
            "bypass_reason": safety_result.get("bypass_reason", ""),
            "bypass_trigger_distance": safety_result.get("bypass_trigger_distance", BYPASS_TRIGGER_DISTANCE),
            "emergency_avoid": safety_result.get("emergency_avoid", False),
            "obstacle_penalty": float(reward_breakdown.get("obstacle_penalty", 0.0)),
            "reward_breakdown": reward_breakdown,
        }

    def _termination_reason(self, caught, collision, too_far, too_high, truncated):
        if caught:
            return "caught"
        if collision:
            return "collision"
        if too_high:
            return "too_high"
        if too_far:
            return "too_far"
        if truncated:
            return "max_steps"
        return "none"

    def _default_safety_result(self, action):
        return {
            "original_action": action,
            "safe_action": action,
            "overridden": False,
            "reason": "",
            "risk_level": "none",
            "front_dist": MAX_LIDAR_DISTANCE,
            "left_dist": MAX_LIDAR_DISTANCE,
            "right_dist": MAX_LIDAR_DISTANCE,
            "bypass_active": False,
            "bypass_action": None,
            "bypass_action_name": "none",
            "bypass_steps_remaining": 0,
            "bypass_reason": "",
            "bypass_trigger_distance": BYPASS_TRIGGER_DISTANCE,
            "emergency_avoid": False,
            "emergency_avoidance": False,
            "altitude_safety_override": False,
            "altitude": self.target_altitude,
            "altitude_error": 0.0,
            "too_high": False,
            "command_vx": 0.0,
            "command_vy": 0.0,
            "command_vz": 0.0,
            "vz_alt_hold": 0.0,
            "final_vz": 0.0,
            "final_vx": 0.0,
            "final_vy": 0.0,
            "base_vz": 0.0,
            "obstacle_bypass": False,
            "diagonal_bypass": False,
            "bypass_direction": "none",
            "forward_scale": 1.0,
            "side_scale": 0.0,
            "speed_scale": 1.0,
            "camera_centering_limited": False,
            "camera_centering_reason": "",
            "gap_direction": "center",
            "gap_safe": True,
            "target_side": "center",
            "front_clear": True,
            "left_blocked": False,
            "right_blocked": False,
            "close_chase_mode": False,
            "planner_choice": "forward",
            "chosen_reason": "target_progress",
            "obstacle_reaction_zone": "none",
            "blocked_lateral": False,
            "forward_detour": False,
            "climb_avoidance": False,
            "stuck_recovery": False,
            "smooth_velocity": False,
            "hover_drift": False,
            "pre_smooth_vx": 0.0,
            "pre_smooth_vy": 0.0,
            "pre_smooth_vz": 0.0,
            "yaw_to_target_deg": 0.0,
        }

    def _default_reward_breakdown(self):
        return {
            "total": 0.0,
            "reward_mode": self.reward_mode,
            "progress_reward": 0.0,
            "distance_delta_reward": 0.0,
            "approach_speed_reward": 0.0,
            "catch_reward": 0.0,
            "collision_penalty": 0.0,
            "too_far_penalty": 0.0,
            "obstacle_penalty": 0.0,
            "obstacle_progress_reward": 0.0,
            "emergency_escape_reward": 0.0,
            "safety_override_penalty": 0.0,
            "altitude_penalty": 0.0,
            "altitude_safety_penalty": 0.0,
            "too_high_penalty": 0.0,
            "altitude_stability_reward": 0.0,
            "altitude": self.target_altitude,
            "altitude_error": 0.0,
            "too_high": False,
            "unnecessary_climb_penalty": 0.0,
            "altitude_match_reward": 0.0,
            "smart_clearance_reward": 0.0,
            "step_penalty": 0.0,
            "min_lidar": MAX_LIDAR_DISTANCE,
            "near_capture_zone": False,
            "target_in_front": False,
            "capture_bonus_reward": 0.0,
        }

    def _get_obs_info(
        self,
        action,
        reward,
        terminated,
        truncated,
        raw_collision,
        collision,
        collision_info,
        safety_result,
        reward_breakdown,
        caught=False,
        too_far=False,
        too_high=False,
        terminated_reason="none",
    ):
        chaser_pos = self.get_global_position(VEHICLE_CHASER)
        target_pos = self.get_global_position(VEHICLE_TARGET)
        dx, dy, dz, distance = self._compute_relative(chaser_pos, target_pos)
        lidar_info = self._get_lidar_info()
        obs = self._make_observation(
            dx,
            dy,
            dz,
            distance,
            chaser_pos,
            target_pos,
            lidar_info["sectors"],
            safety_result["overridden"],
            safety_result["safe_action"],
        )
        info = self._make_info(
            action,
            dx,
            dy,
            dz,
            distance,
            reward,
            terminated,
            truncated,
            raw_collision,
            collision,
            collision_info,
            lidar_info,
            safety_result,
            reward_breakdown,
            caught,
            too_far,
            too_high,
            terminated_reason,
            chaser_pos,
            target_pos,
        )
        return obs, info
