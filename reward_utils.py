#!/usr/bin/env python3


DEFAULT_LIDAR_DISTANCE = 50.0


def compute_chase_reward(
    distance,
    previous_distance,
    collision=False,
    caught=False,
    too_far=False,
    lidar_sectors=None,
    safety_overridden=False,
    step_penalty=-0.01,
):
    distance_delta_reward = 0.0
    if previous_distance is not None:
        improvement = previous_distance - distance
        if improvement > 0.0:
            distance_delta_reward = min(improvement * 2.0, 2.0)
        else:
            distance_delta_reward = max(improvement * 2.0, -2.0)

    catch_reward = 100.0 if caught else 0.0
    collision_penalty = -100.0 if collision else 0.0
    too_far_penalty = -50.0 if too_far else 0.0
    safety_override_penalty = -0.2 if safety_overridden else 0.0

    obstacle_penalty = 0.0
    if lidar_sectors:
        front = float(lidar_sectors.get("front", DEFAULT_LIDAR_DISTANCE))
        left = float(lidar_sectors.get("left", DEFAULT_LIDAR_DISTANCE))
        right = float(lidar_sectors.get("right", DEFAULT_LIDAR_DISTANCE))

        if front < 2.0:
            obstacle_penalty -= 5.0
        elif front < 5.0:
            obstacle_penalty -= 1.0

        if min(left, right) < 1.0:
            obstacle_penalty -= 0.5

    total = (
        distance_delta_reward
        + catch_reward
        + collision_penalty
        + too_far_penalty
        + obstacle_penalty
        + safety_override_penalty
        + step_penalty
    )

    return {
        "total": float(total),
        "distance_delta_reward": float(distance_delta_reward),
        "catch_reward": float(catch_reward),
        "collision_penalty": float(collision_penalty),
        "too_far_penalty": float(too_far_penalty),
        "obstacle_penalty": float(obstacle_penalty),
        "safety_override_penalty": float(safety_override_penalty),
        "step_penalty": float(step_penalty),
    }
