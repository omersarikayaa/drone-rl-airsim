#!/usr/bin/env python3
import math
from pathlib import Path
import time
import traceback

import airsim


VEHICLE_CHASER = "Chaser"
VEHICLE_TARGET = "Target"
VEHICLES = [VEHICLE_CHASER, VEHICLE_TARGET]

TARGET_ALTITUDE_Z = -5.0
MOVE_SPEED = 2.0
SEGMENT_PAUSE_SECONDS = 1.0
MOVEMENT_THRESHOLD_METERS = 1.0
DISTANCE_CHANGE_THRESHOLD_METERS = 0.5


def log(level, message):
    print(f"[{level}] {message}", flush=True)


def format_position(position):
    return f"x={position.x_val:.2f}, y={position.y_val:.2f}, z={position.z_val:.2f}"


def validate_position(position, vehicle_name):
    if position is None:
        raise RuntimeError(f"Global position is None: {vehicle_name}")

    values = [position.x_val, position.y_val, position.z_val]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Global position has invalid values: {vehicle_name} ({format_position(position)})")

    return position


def get_global_position(client, vehicle_name):
    try:
        pose = client.simGetObjectPose(vehicle_name)
    except Exception as exc:
        raise RuntimeError(f"simGetObjectPose failed for {vehicle_name}: {exc}") from exc

    position = getattr(pose, "position", None)
    return validate_position(position, vehicle_name)


def compute_relative(chaser_pos, target_pos):
    dx = target_pos.x_val - chaser_pos.x_val
    dy = target_pos.y_val - chaser_pos.y_val
    dz = target_pos.z_val - chaser_pos.z_val
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dx, dy, dz, distance


def print_global_status(client, expected_target_global=None):
    chaser_pos = get_global_position(client, VEHICLE_CHASER)
    target_pos = get_global_position(client, VEHICLE_TARGET)
    dx, dy, dz, distance = compute_relative(chaser_pos, target_pos)

    log("GLOBAL_POS", f"{VEHICLE_CHASER}: {format_position(chaser_pos)}")
    if expected_target_global is not None:
        log("EXPECTED_GLOBAL_APPROX", f"{VEHICLE_TARGET}: {format_position(expected_target_global)}")
    log("GLOBAL_POS", f"{VEHICLE_TARGET}: {format_position(target_pos)}")
    log("REL_GLOBAL", f"dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, distance={distance:.2f}")

    return chaser_pos, target_pos, distance


def list_and_validate_vehicles(client):
    try:
        vehicles = client.listVehicles()
    except Exception as exc:
        raise RuntimeError(f"AirSim vehicles list could not be read: {exc}") from exc

    log("INFO", f"Vehicles: {vehicles}")
    missing = [name for name in VEHICLES if name not in vehicles]
    if missing:
        raise RuntimeError(
            "Required vehicles not found in AirSim: "
            f"{missing}. Check ~/Documents/AirSim/settings.json and restart AirSimNH."
        )

    return vehicles


def enable_api_and_arm(client):
    for name in VEHICLES:
        client.enableApiControl(True, vehicle_name=name)
        log("OK", f"API control enabled: {name}")

    for name in VEHICLES:
        client.armDisarm(True, vehicle_name=name)
        log("OK", f"Armed: {name}")


def takeoff_all(client):
    for name in VEHICLES:
        client.takeoffAsync(vehicle_name=name).join()
        log("OK", f"Takeoff completed: {name}")

    log("OK", "Takeoff completed.")


def move_all_to_safe_altitude(client):
    for name in VEHICLES:
        # AirSim NED sisteminde yukari cikmak negatif Z degerine gitmek demektir.
        client.moveToZAsync(TARGET_ALTITUDE_Z, MOVE_SPEED, vehicle_name=name).join()
        log("OK", f"Safe altitude reached: {name} z={TARGET_ALTITUDE_Z:.1f}")

    log("OK", f"Safe altitude reached: z={TARGET_ALTITUDE_Z:.1f}")


def hover_vehicle(client, vehicle_name):
    client.hoverAsync(vehicle_name=vehicle_name).join()


def hover_all_before_motion(client):
    hover_vehicle(client, VEHICLE_CHASER)
    log("INFO", "Chaser hovering.")

    hover_vehicle(client, VEHICLE_TARGET)
    log("INFO", "Target hovering before scripted motion.")


def hover_vehicle_safe(client, vehicle_name):
    try:
        hover_vehicle(client, vehicle_name)
        log("OK", f"Hover before landing completed: {vehicle_name}")
        return True
    except Exception as exc:
        log("WARN", f"Hover before landing failed: {vehicle_name} ({exc})")
        return False


def land_vehicle_safe(client, vehicle_name):
    try:
        client.landAsync(vehicle_name=vehicle_name).join()
        log("OK", f"Landing completed: {vehicle_name}")
        return True
    except Exception as exc:
        log("WARN", f"Landing failed: {vehicle_name} ({exc})")
        return False


def cleanup_vehicle_safe(client, vehicle_name):
    cleanup_ok = True

    try:
        client.armDisarm(False, vehicle_name=vehicle_name)
        log("OK", f"Disarmed: {vehicle_name}")
    except Exception as exc:
        cleanup_ok = False
        log("WARN", f"Disarm failed: {vehicle_name} ({exc})")

    try:
        client.enableApiControl(False, vehicle_name=vehicle_name)
        log("OK", f"API control disabled: {vehicle_name}")
    except Exception as exc:
        cleanup_ok = False
        log("WARN", f"Disable API control failed: {vehicle_name} ({exc})")

    return cleanup_ok


def distance_between(pos_a, pos_b):
    dx = pos_b.x_val - pos_a.x_val
    dy = pos_b.y_val - pos_a.y_val
    dz = pos_b.z_val - pos_a.z_val
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def count_distinct_positions(positions):
    distinct = []

    for position in positions:
        if all(distance_between(position, known) >= MOVEMENT_THRESHOLD_METERS for known in distinct):
            distinct.append(position)

    return len(distinct)


def target_moved_in_global_coordinates(target_positions):
    if len(target_positions) < 2:
        return False

    start_position = target_positions[0]
    max_displacement = max(distance_between(start_position, position) for position in target_positions[1:])
    return max_displacement >= MOVEMENT_THRESHOLD_METERS


def distance_changed(distances):
    if len(distances) < 2:
        return False

    return max(distances) - min(distances) >= DISTANCE_CHANGE_THRESHOLD_METERS


def build_expected_global_position(target_global_start, local_x, local_y, local_z):
    return airsim.Vector3r(
        target_global_start.x_val + local_x,
        target_global_start.y_val + local_y,
        local_z,
    )


def build_target_local_route(target_global_start):
    x0 = target_global_start.x_val
    y0 = target_global_start.y_val
    z = TARGET_ALTITUDE_Z

    return [
        (x0 + 5.0, y0, z),
        (x0 + 5.0, y0 + 5.0, z),
        (x0 + 10.0, y0 + 5.0, z),
        (x0 + 10.0, y0, z),
    ]


def move_target_to_segment(client, segment_index, local_x, local_y, local_z):
    log(
        f"SEGMENT {segment_index}",
        f"Target local command: x={local_x:.2f}, y={local_y:.2f}, z={local_z:.2f}",
    )
    # AirSim moveToPositionAsync commonly works in the vehicle's local NED frame,
    # while simGetObjectPose returns world/global pose. For PPO observations we
    # will use global poses and relative dx/dy/dz.
    client.moveToPositionAsync(
        local_x,
        local_y,
        local_z,
        MOVE_SPEED,
        vehicle_name=VEHICLE_TARGET,
    ).join()


def run_scripted_motion(client):
    target_positions = []
    distances = []

    _chaser_pos, target_global_start, distance = print_global_status(client)
    log("INFO", f"Target initial GLOBAL_POS stored: {format_position(target_global_start)}")
    target_positions.append(target_global_start)
    distances.append(distance)

    route = build_target_local_route(target_global_start)
    for index, local_command in enumerate(route, start=1):
        expected_global = build_expected_global_position(target_global_start, *local_command)
        move_target_to_segment(client, index, *local_command)
        time.sleep(SEGMENT_PAUSE_SECONDS)

        _chaser_pos, target_pos, distance = print_global_status(client, expected_global)
        target_positions.append(target_pos)
        distances.append(distance)

    log("SEGMENT 5", "Target hover.")
    hover_vehicle(client, VEHICLE_TARGET)
    time.sleep(SEGMENT_PAUSE_SECONDS)

    _chaser_pos, target_pos, distance = print_global_status(client)
    target_positions.append(target_pos)
    distances.append(distance)

    return target_positions, distances


def main():
    client = None
    connected = False
    vehicles_ok = False
    takeoff_ok = False
    hover_ok = False
    motion_samples_collected = False
    motion_verified = False
    target_moved = False
    cleanup_ok = True
    failure_message = ""

    log("STEP2", "Scripted Target Motion Test")
    log("INFO", f"Project folder: {Path.cwd()}")
    log("INFO", "This test does not train PPO. It only moves Target with scripted commands.")
    log("INFO", "Coordinate frames: Target commands are LOCAL NED; simGetObjectPose is GLOBAL/WORLD.")

    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        connected = True
        log("OK", "AirSim connected.")

        list_and_validate_vehicles(client)
        vehicles_ok = True

        enable_api_and_arm(client)
        takeoff_all(client)
        move_all_to_safe_altitude(client)
        takeoff_ok = True

        hover_all_before_motion(client)
        hover_ok = True

        target_positions, distances = run_scripted_motion(client)
        motion_samples_collected = True

        target_moved = target_moved_in_global_coordinates(target_positions)
        distinct_count = count_distinct_positions(target_positions)
        distances_changed = distance_changed(distances)
        motion_verified = target_moved and distinct_count >= 3 and distances_changed

        log("CHECK", f"Target distinct global positions: {distinct_count}")
        log("CHECK", f"Target moved in global coordinates: {target_moved}")
        log("CHECK", f"Chaser-Target distance changed: {distances_changed}")

        if not target_moved:
            failure_message = "Target did not move in global coordinates."
        elif not motion_verified:
            failure_message = "takeoff works but target motion was not verified."
        else:
            log("OK", "Scripted target motion completed.")

    except Exception as exc:
        failure_message = str(exc)
        log("ERROR", failure_message)
        traceback.print_exc()

    finally:
        if client is not None:
            log("INFO", "Safe landing and cleanup starting.")

            for name in VEHICLES:
                cleanup_ok = hover_vehicle_safe(client, name) and cleanup_ok

            for name in VEHICLES:
                cleanup_ok = land_vehicle_safe(client, name) and cleanup_ok

            if cleanup_ok:
                log("OK", "Landing completed.")

            for name in VEHICLES:
                cleanup_ok = cleanup_vehicle_safe(client, name) and cleanup_ok

            if cleanup_ok:
                log("OK", "Cleanup completed.")
            else:
                log("WARN", "At least one landing or cleanup step failed.")

    if connected and vehicles_ok and takeoff_ok and hover_ok and motion_verified and cleanup_ok:
        print("STEP 2 PASSED: Target scripted motion works.", flush=True)
    elif takeoff_ok and motion_samples_collected and not target_moved and cleanup_ok:
        print("STEP 2 FAILED: Target did not move in global coordinates.", flush=True)
    elif takeoff_ok and cleanup_ok:
        print("STEP 2 PARTIAL: takeoff works but target motion was not verified.", flush=True)
    else:
        if not failure_message:
            if not cleanup_ok:
                failure_message = "Landing or cleanup did not complete."
            else:
                failure_message = "Step 2 conditions were not completed."
        print(f"STEP 2 FAILED: {failure_message}", flush=True)


if __name__ == "__main__":
    main()
