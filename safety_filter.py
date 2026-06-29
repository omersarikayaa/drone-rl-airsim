#!/usr/bin/env python3

ACTION_FORWARD_TO_TARGET = 0
ACTION_MOVE_LEFT = 1
ACTION_MOVE_RIGHT = 2
ACTION_MOVE_UP = 3
ACTION_MOVE_DOWN = 4
ACTION_HOVER = 5

ACTION_NAMES = {
    ACTION_FORWARD_TO_TARGET: "FORWARD_TO_TARGET",
    ACTION_MOVE_LEFT: "MOVE_LEFT",
    ACTION_MOVE_RIGHT: "MOVE_RIGHT",
    ACTION_MOVE_UP: "MOVE_UP",
    ACTION_MOVE_DOWN: "MOVE_DOWN",
    ACTION_HOVER: "HOVER",
}

OBSTACLE_STOP_DISTANCE = 3.0
OBSTACLE_BYPASS_DISTANCE = 5.0
OBSTACLE_OVERRIDE_DISTANCE = 8.0
OBSTACLE_PREPARE_DISTANCE = 12.0
OBSTACLE_SLOW_DISTANCE = OBSTACLE_OVERRIDE_DISTANCE
SIDE_DANGER_DISTANCE = 2.0
ALTITUDE_MIN_Z = -2.0
ALTITUDE_MAX_Z = -20.0
DEFAULT_DISTANCE = 50.0


def _distance(lidar_sectors, name):
    if lidar_sectors is None:
        return DEFAULT_DISTANCE
    value = lidar_sectors.get(name, DEFAULT_DISTANCE)
    if value is None:
        return DEFAULT_DISTANCE
    return float(value)


def _result(
    action,
    safe_action,
    overridden,
    reason,
    risk_level,
    front_dist,
    left_dist,
    right_dist,
    diagonal_bypass=False,
    bypass_direction="none",
):
    return {
        "original_action": action,
        "safe_action": safe_action,
        "overridden": overridden,
        "reason": reason,
        "risk_level": risk_level,
        "front_dist": front_dist,
        "left_dist": left_dist,
        "right_dist": right_dist,
        "diagonal_bypass": bool(diagonal_bypass),
        "bypass_direction": bypass_direction,
    }


def _choose_lateral_or_up(left_clear, right_clear):
    if right_clear > left_clear and right_clear > SIDE_DANGER_DISTANCE:
        return ACTION_MOVE_RIGHT, "choosing right"
    if left_clear > SIDE_DANGER_DISTANCE:
        return ACTION_MOVE_LEFT, "choosing left"
    if right_clear > SIDE_DANGER_DISTANCE:
        return ACTION_MOVE_RIGHT, "choosing right"
    return ACTION_MOVE_UP, "both sides closed; moving up"


def _direction_from_lateral_action(action):
    if action == ACTION_MOVE_LEFT:
        return "left"
    if action == ACTION_MOVE_RIGHT:
        return "right"
    return "none"


def apply_safety_filter(action, lidar_sectors, chaser_z=None):
    try:
        action = int(action)
    except Exception:
        action = None

    front_dist = _distance(lidar_sectors, "front")
    left_dist = _distance(lidar_sectors, "left")
    right_dist = _distance(lidar_sectors, "right")
    front_left_dist = _distance(lidar_sectors, "front_left")
    front_right_dist = _distance(lidar_sectors, "front_right")
    front_obstacle_dist = min(front_dist, front_left_dist, front_right_dist)
    left_clear = min(left_dist, front_left_dist)
    right_clear = min(right_dist, front_right_dist)
    bottom_dist = _distance(lidar_sectors, "bottom")

    if action not in ACTION_NAMES:
        return _result(action, ACTION_HOVER, True, "unknown action", "danger", front_dist, left_dist, right_dist)

    if action == ACTION_MOVE_DOWN and chaser_z is not None and chaser_z > ALTITUDE_MIN_Z:
        return _result(action, ACTION_HOVER, True, "too close to ground; blocking MOVE_DOWN", "danger", front_dist, left_dist, right_dist)

    if action == ACTION_MOVE_UP and chaser_z is not None and chaser_z < ALTITUDE_MAX_Z:
        return _result(action, ACTION_HOVER, True, "too high; blocking MOVE_UP", "danger", front_dist, left_dist, right_dist)

    # Altta engel varsa yukari cik.
    if bottom_dist < 2.5:
        return _result(action, ACTION_MOVE_UP, True, "bottom obstacle; moving up", "danger", front_dist, left_dist, right_dist)

    if action not in (ACTION_FORWARD_TO_TARGET, ACTION_HOVER) and front_obstacle_dist < OBSTACLE_STOP_DISTANCE:
        bypass_action, direction_reason = _choose_lateral_or_up(left_clear, right_clear)
        bypass_direction = _direction_from_lateral_action(bypass_action)
        return _result(
            action,
            action,
            True,
            f"front obstacle emergency; forcing env hard bypass {direction_reason}",
            "danger",
            front_dist,
            left_dist,
            right_dist,
            diagonal_bypass=True,
            bypass_direction=bypass_direction,
        )

    if action not in (ACTION_FORWARD_TO_TARGET, ACTION_HOVER) and front_obstacle_dist < OBSTACLE_BYPASS_DISTANCE:
        bypass_action, direction_reason = _choose_lateral_or_up(left_clear, right_clear)
        bypass_direction = _direction_from_lateral_action(bypass_action)
        return _result(
            action,
            action,
            True,
            f"front obstacle within bypass distance; forcing env diagonal bypass {direction_reason}",
            "slow",
            front_dist,
            left_dist,
            right_dist,
            diagonal_bypass=True,
            bypass_direction=bypass_direction,
        )

    if action in (ACTION_FORWARD_TO_TARGET, ACTION_HOVER) and front_obstacle_dist < OBSTACLE_STOP_DISTANCE:
        bypass_action, direction_reason = _choose_lateral_or_up(left_clear, right_clear)
        bypass_direction = _direction_from_lateral_action(bypass_action)
        if bypass_direction != "none":
            reason = f"front obstacle emergency; hard avoidance {direction_reason}"
            return _result(
                action,
                ACTION_FORWARD_TO_TARGET,
                True,
                reason,
                "danger",
                front_dist,
                left_dist,
                right_dist,
                diagonal_bypass=True,
                bypass_direction=bypass_direction,
            )
        reason = f"front obstacle; {direction_reason}"
        return _result(action, bypass_action, True, reason, "danger", front_dist, left_dist, right_dist)

    if action in (ACTION_FORWARD_TO_TARGET, ACTION_HOVER) and front_obstacle_dist < OBSTACLE_BYPASS_DISTANCE:
        bypass_action, direction_reason = _choose_lateral_or_up(left_clear, right_clear)
        bypass_direction = _direction_from_lateral_action(bypass_action)
        if bypass_direction != "none":
            reason = f"front obstacle within bypass distance; diagonal bypass {direction_reason}"
            return _result(
                action,
                action,
                True,
                reason,
                "slow",
                front_dist,
                left_dist,
                right_dist,
                diagonal_bypass=True,
                bypass_direction=bypass_direction,
            )
        return _result(action, bypass_action, True, "front obstacle within bypass distance; no lateral side", "slow", front_dist, left_dist, right_dist)

    if action in (ACTION_FORWARD_TO_TARGET, ACTION_HOVER) and front_obstacle_dist < OBSTACLE_OVERRIDE_DISTANCE:
        bypass_action, direction_reason = _choose_lateral_or_up(left_clear, right_clear)
        bypass_direction = _direction_from_lateral_action(bypass_action)
        reason = f"front obstacle below override distance; preparing bypass {direction_reason}"
        return _result(
            action,
            action,
            True,
            reason,
            "slow",
            front_dist,
            left_dist,
            right_dist,
            diagonal_bypass=False,
            bypass_direction=bypass_direction,
        )

    if action in (ACTION_FORWARD_TO_TARGET, ACTION_HOVER) and front_obstacle_dist < OBSTACLE_PREPARE_DISTANCE:
        return _result(action, action, False, "front obstacle within prepare distance", "prepare", front_dist, left_dist, right_dist)

    if action == ACTION_MOVE_LEFT and left_clear < SIDE_DANGER_DISTANCE:
        if right_clear > SIDE_DANGER_DISTANCE:
            safe_action = ACTION_MOVE_RIGHT
            reason = "left obstacle; choosing right"
        else:
            safe_action = ACTION_MOVE_UP
            reason = "left obstacle; both sides closed; moving up"
        return _result(action, safe_action, True, reason, "danger", front_dist, left_dist, right_dist)

    if action == ACTION_MOVE_RIGHT and right_clear < SIDE_DANGER_DISTANCE:
        if left_clear > SIDE_DANGER_DISTANCE:
            safe_action = ACTION_MOVE_LEFT
            reason = "right obstacle; choosing left"
        else:
            safe_action = ACTION_MOVE_UP
            reason = "right obstacle; both sides closed; moving up"
        return _result(action, safe_action, True, reason, "danger", front_dist, left_dist, right_dist)

    if front_obstacle_dist < OBSTACLE_OVERRIDE_DISTANCE:
        return _result(
            action,
            action,
            True,
            "front obstacle below override distance; current avoidance action allowed",
            "slow",
            front_dist,
            left_dist,
            right_dist,
            diagonal_bypass=False,
            bypass_direction=_direction_from_lateral_action(action),
        )

    if front_obstacle_dist < OBSTACLE_PREPARE_DISTANCE:
        return _result(action, action, False, "front obstacle within prepare distance", "prepare", front_dist, left_dist, right_dist)

    return _result(action, action, False, "", "none", front_dist, left_dist, right_dist)
