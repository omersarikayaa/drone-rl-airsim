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

OBSTACLE_STOP_DISTANCE = 2.0
OBSTACLE_SLOW_DISTANCE = 5.0
SIDE_DANGER_DISTANCE = 1.5
ALTITUDE_MIN_Z = -2.0
ALTITUDE_MAX_Z = -15.0
DEFAULT_DISTANCE = 50.0


def _distance(lidar_sectors, name):
    if lidar_sectors is None:
        return DEFAULT_DISTANCE
    value = lidar_sectors.get(name, DEFAULT_DISTANCE)
    if value is None:
        return DEFAULT_DISTANCE
    return float(value)


def _result(action, safe_action, overridden, reason, risk_level, front_dist, left_dist, right_dist):
    return {
        "original_action": action,
        "safe_action": safe_action,
        "overridden": overridden,
        "reason": reason,
        "risk_level": risk_level,
        "front_dist": front_dist,
        "left_dist": left_dist,
        "right_dist": right_dist,
    }


def apply_safety_filter(action, lidar_sectors, chaser_z=None):
    try:
        action = int(action)
    except Exception:
        action = None

    front_dist = _distance(lidar_sectors, "front")
    left_dist = _distance(lidar_sectors, "left")
    right_dist = _distance(lidar_sectors, "right")

    if action not in ACTION_NAMES:
        return _result(
            action,
            ACTION_HOVER,
            True,
            "unknown action",
            "danger",
            front_dist,
            left_dist,
            right_dist,
        )

    if action == ACTION_MOVE_DOWN and chaser_z is not None and chaser_z > ALTITUDE_MIN_Z:
        return _result(
            action,
            ACTION_HOVER,
            True,
            "too close to ground; blocking MOVE_DOWN",
            "danger",
            front_dist,
            left_dist,
            right_dist,
        )

    if action == ACTION_MOVE_UP and chaser_z is not None and chaser_z < ALTITUDE_MAX_Z:
        return _result(
            action,
            ACTION_HOVER,
            True,
            "too high; blocking MOVE_UP",
            "danger",
            front_dist,
            left_dist,
            right_dist,
        )

    if action == ACTION_FORWARD_TO_TARGET and front_dist < OBSTACLE_STOP_DISTANCE:
        if right_dist > left_dist and right_dist > SIDE_DANGER_DISTANCE:
            safe_action = ACTION_MOVE_RIGHT
            reason = "front obstacle too close; choosing right side"
        elif left_dist > SIDE_DANGER_DISTANCE:
            safe_action = ACTION_MOVE_LEFT
            reason = "front obstacle too close; choosing left side"
        else:
            safe_action = ACTION_HOVER
            reason = "front obstacle too close; no safe side available"

        return _result(action, safe_action, True, reason, "danger", front_dist, left_dist, right_dist)

    if action == ACTION_FORWARD_TO_TARGET and front_dist < OBSTACLE_SLOW_DISTANCE:
        return _result(
            action,
            action,
            False,
            "front obstacle within slow distance",
            "slow",
            front_dist,
            left_dist,
            right_dist,
        )

    if action == ACTION_MOVE_LEFT and left_dist < SIDE_DANGER_DISTANCE:
        if right_dist > SIDE_DANGER_DISTANCE:
            safe_action = ACTION_MOVE_RIGHT
            reason = "left obstacle too close; choosing right side"
        else:
            safe_action = ACTION_HOVER
            reason = "left obstacle too close; no safe side available"

        return _result(action, safe_action, True, reason, "danger", front_dist, left_dist, right_dist)

    if action == ACTION_MOVE_RIGHT and right_dist < SIDE_DANGER_DISTANCE:
        if left_dist > SIDE_DANGER_DISTANCE:
            safe_action = ACTION_MOVE_LEFT
            reason = "right obstacle too close; choosing left side"
        else:
            safe_action = ACTION_HOVER
            reason = "right obstacle too close; no safe side available"

        return _result(action, safe_action, True, reason, "danger", front_dist, left_dist, right_dist)

    return _result(action, action, False, "", "none", front_dist, left_dist, right_dist)
