#!/usr/bin/env python3
import math
import random


TARGET_ALTITUDE = 8.0
DEMO_ALTITUDE = 11.0


def _point_from_distance(distance, angle_rad):
    return distance * math.cos(angle_rad), distance * math.sin(angle_rad)


def build_scenario(scenario_id, seed=None):
    rng = random.Random(seed)
    scenario_id = int(scenario_id)

    if scenario_id == 1:
        return {
            "scenario_id": 1,
            "name": "right_escape_long_range_chase",
            "description": "Fixed long range demo: Target goes forward, then escapes right by waypoint route.",
            "chaser_yaw": 0.00,
            "target_yaw": 1.71,
            "env_kwargs": {
                "target_mode": "right_escape",
                "target_base_speed": 3.0,
                "target_escape_speed": 3.0,
                "target_evade_distance": 12.0,
                "target_danger_distance": 5.0,
                "chaser_start_x": 139.36,
                "chaser_start_y": 0.0,
                "chaser_start_z": -DEMO_ALTITUDE,
                "target_start_x": 300.0,
                "target_start_y": 80.0,
                "target_start_z": -DEMO_ALTITUDE,
                "target_waypoints": [
                    (320.0, 80.0, -DEMO_ALTITUDE),
                    (330.0, 130.0, -DEMO_ALTITUDE),
                    (345.0, 160.0, -DEMO_ALTITUDE),
                ],
                "target_altitude": DEMO_ALTITUDE,
                "max_episode_steps": 900,
                "too_far_distance": 320.0,
                "chaser_speed": 6.0,
                "step_duration": 0.25,
            },
        }

    if scenario_id == 2:
        distance = rng.uniform(50.0, 100.0)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        target_x, target_y = _point_from_distance(distance, angle)
        return {
            "scenario_id": 2,
            "name": "evasive_target_chase",
            "description": "Medium range chase with an evasive target.",
            "env_kwargs": {
                "target_mode": "right_waypoint",
                "target_base_speed": 1.5,
                "target_escape_speed": 2.6,
                "target_evade_distance": 18.0,
                "target_danger_distance": 7.0,
                "chaser_start_x": 0.0,
                "chaser_start_y": 0.0,
                "chaser_start_z": -TARGET_ALTITUDE,
                "target_start_x": target_x,
                "target_start_y": target_y,
                "target_start_z": -TARGET_ALTITUDE,
                "max_episode_steps": 900,
                "too_far_distance": 220.0,
                "chaser_speed": 5.0,
                "step_duration": 0.3,
            },
        }

    raise ValueError(f"Unknown scenario: {scenario_id}. Expected 1 or 2.")
