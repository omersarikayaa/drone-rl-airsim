#!/usr/bin/env python3
import math
import time

import airsim
import numpy as np

from reward_utils import compute_chase_reward
from safety_filter import (
    ACTION_HOVER,
    ACTION_MOVE_LEFT,
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
STEP_DURATION = 0.5
MAX_STEPS = 100
CATCH_DISTANCE = 2.0
TOO_FAR_DISTANCE = 80.0
MAX_ACTIONS = 6
COLLISION_IGNORE_STEPS = 3
COLLISION_MIN_ALTITUDE_Z = -1.5
BYPASS_HOLD_STEPS = 6
BYPASS_MIN_STEPS_BEFORE_RELEASE = 3
BYPASS_TRIGGER_DISTANCE = 3.5
BYPASS_EMERGENCY_DISTANCE = 1.8
BYPASS_RELEASE_MIN_FRONT_DISTANCE = BYPASS_TRIGGER_DISTANCE
BYPASS_CLEAR_FRONT_DISTANCE = OBSTACLE_SLOW_DISTANCE
START_PLACEMENT_TOLERANCE_METERS = 2.0
START_PLACEMENT_MIN_SPEED = 10.0
START_PLACEMENT_MAX_SPEED = 50.0
OBSERVATION_SIZE = 14
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
        chaser_start_z=-5.0,
        target_start_x=None,
        target_start_y=None,
        target_start_z=-5.0,
        max_episode_steps=MAX_STEPS,
    ):
        if target_mode not in ("simple", "evasive"):
            raise ValueError(f"Unknown target_mode: {target_mode}. Expected 'simple' or 'evasive'.")

        self.target_mode = target_mode
        self.target_base_speed = float(target_base_speed)
        self.target_escape_speed = float(target_escape_speed)
        self.target_evade_distance = float(target_evade_distance)
        self.target_danger_distance = float(target_danger_distance)
        self.chaser_start_x = None if chaser_start_x is None else float(chaser_start_x)
        self.chaser_start_y = None if chaser_start_y is None else float(chaser_start_y)
        self.chaser_start_z = float(chaser_start_z)
        self.target_start_x = None if target_start_x is None else float(target_start_x)
        self.target_start_y = None if target_start_y is None else float(target_start_y)
        self.target_start_z = float(target_start_z)
        self.max_episode_steps = max(1, int(max_episode_steps))
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
        self.last_target_velocity = (0.0, 0.0, 0.0)
        self.bypass_active = False
        self.bypass_action = None
        self.bypass_steps_remaining = 0
        self.bypass_reason = ""
        self.bypass_steps_elapsed = 0
        self.vehicle_home_global = {}
        self.actual_chaser_start_pos = None
        self.actual_target_start_pos = None
        self.target_controller = None

        if GYM_AVAILABLE:
            self.action_space = spaces.Discrete(MAX_ACTIONS)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(OBSERVATION_SIZE,),
                dtype=np.float32,
            )
        else:
            self.action_space = None
            self.observation_space = None

        if self.target_mode == "evasive":
            self.target_controller = TargetController(
                self.client,
                vehicle_name=VEHICLE_TARGET,
                safe_z=SAFE_Z,
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

    def reset(self, seed=None, options=None):
        if GYM_AVAILABLE:
            try:
                super().reset(seed=seed)
            except Exception:
                pass

        self.step_count = 0
        self.previous_distance = None
        self._closed = False
        self.last_target_velocity = (0.0, 0.0, 0.0)
        self._reset_bypass_state()
        self.actual_chaser_start_pos = None
        self.actual_target_start_pos = None

        try:
            self.client.reset()
            time.sleep(1.0)
        except Exception as exc:
            print(f"[WARN] AirSim reset failed, continuing with current scene: {exc}", flush=True)

        self._store_vehicle_home_globals()
        self._enable_api_and_arm()
        self._takeoff_all()
        self._move_all_to_safe_z()
        self._apply_requested_chaser_start()
        self._apply_requested_target_start()
        self._print_reset_relative()
        self.client.hoverAsync(vehicle_name=VEHICLE_CHASER).join()
        self.client.hoverAsync(vehicle_name=VEHICLE_TARGET).join()
        time.sleep(1.0)

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
        current_lidar_info = self._get_lidar_info()
        safety_result = apply_safety_filter(action, current_lidar_info["sectors"], chaser_pos.z_val)
        safety_result = self._apply_bypass_hold(action, safety_result, current_lidar_info["sectors"], chaser_pos.z_val)
        safe_action = safety_result["safe_action"]

        self._apply_chaser_action(safe_action, chaser_pos, target_pos)
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

        lidar_info = self._get_lidar_info()
        caught = distance < CATCH_DISTANCE
        too_far = distance > TOO_FAR_DISTANCE
        reward_breakdown = compute_chase_reward(
            distance=distance,
            previous_distance=self.previous_distance,
            collision=effective_collision,
            caught=caught,
            too_far=too_far,
            lidar_sectors=lidar_info["sectors"] if lidar_info["available"] else None,
            safety_overridden=safety_result["overridden"],
        )
        reward = reward_breakdown["total"]
        terminated = caught or effective_collision or too_far
        truncated = self.step_count >= self.max_episode_steps
        terminated_reason = self._termination_reason(caught, effective_collision, too_far, truncated)

        obs = self._make_observation(dx, dy, dz, distance, chaser_pos, target_pos, lidar_info["sectors"])
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
            terminated_reason,
            chaser_pos,
            target_pos,
        )
        self.previous_distance = distance

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _reset_bypass_state(self):
        self.bypass_active = False
        self.bypass_action = None
        self.bypass_steps_remaining = 0
        self.bypass_reason = ""
        self.bypass_steps_elapsed = 0

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
        left_dist = self._sector_distance(lidar_sectors, "left")
        right_dist = self._sector_distance(lidar_sectors, "right")

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
            self.client.moveToZAsync(SAFE_Z, CHASER_SPEED, vehicle_name=name).join()

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
        return self.target_start_x is not None or self.target_start_y is not None

    def _chaser_start_requested(self):
        return self.chaser_start_x is not None or self.chaser_start_y is not None

    def _any_start_requested(self):
        return self._chaser_start_requested() or self._target_start_requested()

    def _chaser_start_tuple(self):
        if not self._chaser_start_requested():
            return None
        return (self.chaser_start_x, self.chaser_start_y, self.chaser_start_z)

    def _target_start_tuple(self):
        if not self._target_start_requested():
            return None
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

        if self._chaser_start_requested():
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
            STEP_DURATION,
            vehicle_name=VEHICLE_TARGET,
        ).join()

        target_pos = self.get_global_position(VEHICLE_TARGET)
        if abs(target_pos.z_val - SAFE_Z) > 1.5:
            self.client.moveToZAsync(SAFE_Z, 1.0, vehicle_name=VEHICLE_TARGET).join()

        self.last_target_velocity = (vx * TARGET_SPEED, vy * TARGET_SPEED, 0.0)

    def _move_evasive_target(self, chaser_pos, target_pos):
        if self.target_controller is None:
            self.target_controller = TargetController(
                self.client,
                vehicle_name=VEHICLE_TARGET,
                safe_z=SAFE_Z,
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
            STEP_DURATION,
            vehicle_name=VEHICLE_TARGET,
        ).join()
        self.last_target_velocity = (vx, vy, vz)

    def _move_target(self, chaser_pos, target_pos):
        if self.target_mode == "evasive":
            self._move_evasive_target(chaser_pos, target_pos)
        else:
            self._move_scripted_target()

    def _apply_chaser_action(self, action, chaser_pos, target_pos):
        if action == 0:
            dx = target_pos.x_val - chaser_pos.x_val
            dy = target_pos.y_val - chaser_pos.y_val
            norm = math.sqrt(dx * dx + dy * dy)
            if norm < 1e-6:
                self.client.hoverAsync(vehicle_name=VEHICLE_CHASER).join()
                return
            vx = CHASER_SPEED * dx / norm
            vy = CHASER_SPEED * dy / norm
            vz = 0.0
        elif action == 1:
            vx, vy, vz = 0.0, -CHASER_SPEED, 0.0
        elif action == 2:
            vx, vy, vz = 0.0, CHASER_SPEED, 0.0
        elif action == 3:
            if chaser_pos.z_val - STEP_DURATION < -15.0:
                self.client.hoverAsync(vehicle_name=VEHICLE_CHASER).join()
                return
            vx, vy, vz = 0.0, 0.0, -1.0
        elif action == 4:
            if chaser_pos.z_val + STEP_DURATION > -2.0:
                self.client.hoverAsync(vehicle_name=VEHICLE_CHASER).join()
                return
            vx, vy, vz = 0.0, 0.0, 1.0
        else:
            self.client.hoverAsync(vehicle_name=VEHICLE_CHASER).join()
            return

        self.client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            STEP_DURATION,
            vehicle_name=VEHICLE_CHASER,
        ).join()

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

    def _normalize_lidar_distance(self, distance):
        if distance is None or not math.isfinite(float(distance)):
            distance = MAX_LIDAR_DISTANCE
        return min(float(distance), MAX_LIDAR_DISTANCE) / MAX_LIDAR_DISTANCE

    def _make_observation(self, dx, dy, dz, distance, chaser_pos, target_pos, lidar_sectors):
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
                self._normalize_lidar_distance(lidar_sectors["front"]),
                self._normalize_lidar_distance(lidar_sectors["front_left"]),
                self._normalize_lidar_distance(lidar_sectors["front_right"]),
                self._normalize_lidar_distance(lidar_sectors["left"]),
                self._normalize_lidar_distance(lidar_sectors["right"]),
                self._normalize_lidar_distance(lidar_sectors["back"]),
            ],
            dtype=np.float32,
        )

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
        terminated_reason,
        chaser_pos,
        target_pos,
    ):
        action_name = ACTION_NAMES.get(action, "UNKNOWN") if action is not None else "RESET"
        lidar_sectors = lidar_info["sectors"]
        safe_action = safety_result["safe_action"]
        target_vx, target_vy, target_vz = self.last_target_velocity
        return {
            "step": self.step_count,
            "action": action,
            "action_name": action_name,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "distance": distance,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "caught": caught,
            "too_far": too_far,
            "terminated_reason": terminated_reason,
            "raw_collision": raw_collision,
            "collision": collision,
            "collision_object_name": collision_info.get("object_name", ""),
            "collision_position": collision_info.get("position"),
            "collision_normal": collision_info.get("normal"),
            "collision_impact_point": collision_info.get("impact_point"),
            "chaser_pos": (chaser_pos.x_val, chaser_pos.y_val, chaser_pos.z_val),
            "target_pos": (target_pos.x_val, target_pos.y_val, target_pos.z_val),
            "target_mode": self.target_mode,
            "target_vx": target_vx,
            "target_vy": target_vy,
            "target_vz": target_vz,
            "target_base_speed": self.target_base_speed,
            "target_escape_speed": self.target_escape_speed,
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
            "max_episode_steps": self.max_episode_steps,
            "lidar_available": lidar_info["available"],
            "lidar_point_count": lidar_info["point_count"],
            "lidar_front": lidar_sectors["front"],
            "lidar_front_left": lidar_sectors["front_left"],
            "lidar_front_right": lidar_sectors["front_right"],
            "lidar_left": lidar_sectors["left"],
            "lidar_right": lidar_sectors["right"],
            "lidar_back": lidar_sectors["back"],
            "obs_shape": (OBSERVATION_SIZE,),
            "safety_original_action": safety_result["original_action"],
            "safety_original_action_name": action_name,
            "safety_safe_action": safe_action,
            "safety_safe_action_name": ACTION_NAMES.get(safe_action, "UNKNOWN") if safe_action is not None else "RESET",
            "safety_overridden": safety_result["overridden"],
            "safety_reason": safety_result["reason"],
            "safety_risk_level": safety_result["risk_level"],
            "bypass_active": safety_result.get("bypass_active", False),
            "bypass_action": safety_result.get("bypass_action"),
            "bypass_action_name": safety_result.get("bypass_action_name", "none"),
            "bypass_steps_remaining": safety_result.get("bypass_steps_remaining", 0),
            "bypass_reason": safety_result.get("bypass_reason", ""),
            "bypass_trigger_distance": safety_result.get("bypass_trigger_distance", BYPASS_TRIGGER_DISTANCE),
            "emergency_avoid": safety_result.get("emergency_avoid", False),
            "reward_breakdown": reward_breakdown,
        }

    def _termination_reason(self, caught, collision, too_far, truncated):
        if caught:
            return "caught"
        if collision:
            return "collision"
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
        }

    def _default_reward_breakdown(self):
        return {
            "total": 0.0,
            "distance_delta_reward": 0.0,
            "catch_reward": 0.0,
            "collision_penalty": 0.0,
            "too_far_penalty": 0.0,
            "obstacle_penalty": 0.0,
            "safety_override_penalty": 0.0,
            "step_penalty": 0.0,
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
        terminated_reason="none",
    ):
        chaser_pos = self.get_global_position(VEHICLE_CHASER)
        target_pos = self.get_global_position(VEHICLE_TARGET)
        dx, dy, dz, distance = self._compute_relative(chaser_pos, target_pos)
        lidar_info = self._get_lidar_info()
        obs = self._make_observation(dx, dy, dz, distance, chaser_pos, target_pos, lidar_info["sectors"])
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
            terminated_reason,
            chaser_pos,
            target_pos,
        )
        return obs, info
