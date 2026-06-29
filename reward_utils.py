#!/usr/bin/env python3


DEFAULT_LIDAR_DISTANCE = 50.0
HIGH_ALTITUDE_Z = -6.0
OBSTACLE_PENALTY_DISTANCE = 3.0
TARGET_ALTITUDE = 8.0
MIN_SAFE_ALTITUDE = 4.0
MAX_SAFE_ALTITUDE = 15.0
HARD_MAX_ALTITUDE = 20.0


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _lidar_min_distance(lidar_sectors, min_lidar=None):
    if min_lidar is not None:
        return _safe_float(min_lidar, DEFAULT_LIDAR_DISTANCE)
    if not lidar_sectors:
        return DEFAULT_LIDAR_DISTANCE

    distances = []
    for name in ("front", "front_left", "front_right", "left", "right"):
        distances.append(_safe_float(lidar_sectors.get(name, DEFAULT_LIDAR_DISTANCE), DEFAULT_LIDAR_DISTANCE))
    return min(distances) if distances else DEFAULT_LIDAR_DISTANCE


def legacy_compute_chase_reward(
    distance,
    previous_distance,
    collision=False,
    caught=False,
    too_far=False,
    lidar_sectors=None,
    safety_overridden=False,
    step_penalty=-0.01,
    chaser_z=None,
    target_z=None,
    step_duration=1.0,
    min_lidar=None,
    reward_mode="legacy",
):
    distance_delta_reward = 0.0
    approach_speed_reward = 0.0
    improvement = 0.0
    if previous_distance is not None:
        improvement = previous_distance - distance
        if improvement > 0.0:
            distance_delta_reward = min(improvement * 2.0, 2.0)
            safe_dt = max(float(step_duration), 1e-3)
            approach_speed = improvement / safe_dt
            approach_speed_reward = min(approach_speed * 0.25, 2.0)
        else:
            distance_delta_reward = max(improvement * 2.0, -2.0)

    catch_reward = 100.0 if caught else 0.0
    collision_penalty = -100.0 if collision else 0.0
    too_far_penalty = -50.0 if too_far else 0.0
    safety_override_penalty = -0.2 if safety_overridden else 0.0

    obstacle_penalty = 0.0
    smart_clearance_reward = 0.0
    if lidar_sectors:
        front = float(lidar_sectors.get("front", DEFAULT_LIDAR_DISTANCE))
        left = float(lidar_sectors.get("left", DEFAULT_LIDAR_DISTANCE))
        right = float(lidar_sectors.get("right", DEFAULT_LIDAR_DISTANCE))
        closest_side = min(left, right)

        if front < 2.0:
            obstacle_penalty -= 5.0
        elif front < 5.0:
            obstacle_penalty -= 1.0

        if closest_side < 1.0:
            obstacle_penalty -= 0.5

        near_obstacle = min(front, closest_side) < 6.0
        if not safety_overridden and improvement > 0.0:
            smart_clearance_reward = 0.15 if near_obstacle else 0.05

    unnecessary_climb_penalty = 0.0
    altitude_match_reward = 0.0
    if chaser_z is not None:
        chaser_z = float(chaser_z)
        if chaser_z < HIGH_ALTITUDE_Z:
            unnecessary_climb_penalty = -min((HIGH_ALTITUDE_Z - chaser_z) * 0.2, 2.0)

        if target_z is not None:
            dz = abs(chaser_z - float(target_z))
            if dz < 0.5:
                altitude_match_reward = 0.5
            elif dz < 1.5:
                altitude_match_reward = 0.2

    total = (
        distance_delta_reward
        + approach_speed_reward
        + catch_reward
        + collision_penalty
        + too_far_penalty
        + obstacle_penalty
        + safety_override_penalty
        + unnecessary_climb_penalty
        + altitude_match_reward
        + smart_clearance_reward
        + step_penalty
    )

    return {
        "total": float(total),
        "reward_mode": reward_mode,
        "progress_reward": float(distance_delta_reward),
        "distance_delta_reward": float(distance_delta_reward),
        "approach_speed_reward": float(approach_speed_reward),
        "catch_reward": float(catch_reward),
        "collision_penalty": float(collision_penalty),
        "too_far_penalty": float(too_far_penalty),
        "obstacle_penalty": float(obstacle_penalty),
        "safety_override_penalty": float(safety_override_penalty),
        "altitude_penalty": float(unnecessary_climb_penalty),
        "unnecessary_climb_penalty": float(unnecessary_climb_penalty),
        "altitude_match_reward": float(altitude_match_reward),
        "smart_clearance_reward": float(smart_clearance_reward),
        "step_penalty": float(step_penalty),
        "min_lidar": float(_lidar_min_distance(lidar_sectors, min_lidar)),
    }


def compute_chase_reward(
    distance,
    previous_distance,
    collision=False,
    caught=False,
    too_far=False,
    lidar_sectors=None,
    safety_overridden=False,
    step_penalty=-0.02,
    chaser_z=None,
    target_z=None,
    step_duration=1.0,
    min_lidar=None,
    previous_min_lidar=None,
    reward_mode="simple",
    near_capture_zone=False,
    target_in_front=False,
    target_altitude=TARGET_ALTITUDE,
    min_safe_altitude=MIN_SAFE_ALTITUDE,
    max_safe_altitude=MAX_SAFE_ALTITUDE,
    hard_max_altitude=HARD_MAX_ALTITUDE,
    too_high=False,
    emergency_avoidance=False,
):
    if reward_mode == "legacy":
        legacy_step_penalty = -0.01 if step_penalty == -0.02 else step_penalty
        return legacy_compute_chase_reward(
            distance=distance,
            previous_distance=previous_distance,
            collision=collision,
            caught=caught,
            too_far=too_far,
            lidar_sectors=lidar_sectors,
            safety_overridden=safety_overridden,
            step_penalty=legacy_step_penalty,
            chaser_z=chaser_z,
            target_z=target_z,
            step_duration=step_duration,
            min_lidar=min_lidar,
        )

    progress_reward = 0.0
    improvement = 0.0
    if previous_distance is not None:
        improvement = float(previous_distance) - float(distance)
        progress_reward = 5.0 * improvement

    catch_reward = 50.0 if caught else 0.0
    collision_penalty = -75.0 if collision else 0.0
    too_far_penalty = -25.0 if too_far else 0.0
    safety_override_penalty = -3.0 if safety_overridden else 0.0

    min_lidar_distance = _lidar_min_distance(lidar_sectors, min_lidar)
    obstacle_penalty = 0.0
    if min_lidar_distance < 2.0:
        obstacle_penalty = -8.0
    elif min_lidar_distance < 3.0:
        obstacle_penalty = -4.0
    elif min_lidar_distance < 5.0:
        obstacle_penalty = -1.0
    if near_capture_zone and target_in_front:
        obstacle_penalty = 0.0

    obstacle_progress_reward = 0.0
    if (
        safety_overridden
        and previous_distance is not None
        and previous_min_lidar is not None
        and improvement > 0.0
    ):
        lidar_improvement = min_lidar_distance - float(previous_min_lidar)
        if lidar_improvement > 0.0:
            obstacle_progress_reward = min(0.5, 0.1 + 0.2 * lidar_improvement)

    emergency_escape_reward = 0.0
    if emergency_avoidance and not collision and previous_min_lidar is not None:
        lidar_improvement = min_lidar_distance - float(previous_min_lidar)
        if lidar_improvement > 0.0:
            emergency_escape_reward = min(0.4, 0.1 + 0.1 * lidar_improvement)

    altitude_penalty = 0.0
    altitude_safety_penalty = 0.0
    too_high_penalty = -100.0 if too_high else 0.0
    altitude_stability_reward = 0.0
    altitude = 0.0
    altitude_error = 0.0
    unnecessary_climb_penalty = 0.0
    if chaser_z is not None:
        altitude = -float(chaser_z)
        altitude_error = float(target_altitude) - altitude
        altitude_penalty = -min(abs(altitude_error) * 0.05, 1.0)
        if altitude < float(min_safe_altitude):
            altitude_safety_penalty = -min((float(min_safe_altitude) - altitude) * 2.0, 8.0)
        elif altitude > float(max_safe_altitude):
            altitude_penalty = -5.0 * (altitude - float(max_safe_altitude))
            if altitude > float(hard_max_altitude):
                too_high_penalty = -100.0
        else:
            altitude_stability_reward = 0.1

    approach_speed_reward = 0.0
    altitude_match_reward = 0.0
    smart_clearance_reward = 0.0

    total = (
        progress_reward
        + catch_reward
        + collision_penalty
        + too_far_penalty
        + obstacle_penalty
        + obstacle_progress_reward
        + emergency_escape_reward
        + safety_override_penalty
        + altitude_penalty
        + altitude_safety_penalty
        + too_high_penalty
        + altitude_stability_reward
        + unnecessary_climb_penalty
        + step_penalty
    )

    return {
        "total": float(total),
        "reward_mode": reward_mode,
        "progress_reward": float(progress_reward),
        "distance_delta_reward": float(progress_reward),
        "approach_speed_reward": float(approach_speed_reward),
        "catch_reward": float(catch_reward),
        "collision_penalty": float(collision_penalty),
        "too_far_penalty": float(too_far_penalty),
        "obstacle_penalty": float(obstacle_penalty),
        "obstacle_progress_reward": float(obstacle_progress_reward),
        "emergency_escape_reward": float(emergency_escape_reward),
        "safety_override_penalty": float(safety_override_penalty),
        "altitude_penalty": float(altitude_penalty),
        "altitude_safety_penalty": float(altitude_safety_penalty),
        "too_high_penalty": float(too_high_penalty),
        "altitude_stability_reward": float(altitude_stability_reward),
        "altitude": float(altitude),
        "altitude_error": float(altitude_error),
        "too_high": bool(too_high or altitude > float(hard_max_altitude)),
        "unnecessary_climb_penalty": float(unnecessary_climb_penalty),
        "altitude_match_reward": float(altitude_match_reward),
        "smart_clearance_reward": float(smart_clearance_reward),
        "step_penalty": float(step_penalty),
        "min_lidar": float(min_lidar_distance),
        "near_capture_zone": bool(near_capture_zone),
        "target_in_front": bool(target_in_front),
    }
