#!/usr/bin/env python3
import argparse
import math
import time

import airsim


DEFAULT_VEHICLE = "Chaser"
TARGET_Z = -8.0
MOVE_SPEED = 2.0
VERTICAL_SPEED = 1.0
MOVE_DURATION = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Manual AirSim XYZ drive tool.")
    parser.add_argument("--vehicle", default=DEFAULT_VEHICLE, choices=("Chaser", "Target"))
    return parser.parse_args()


def get_pose(client, vehicle_name):
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    position = state.kinematics_estimated.position
    orientation = state.kinematics_estimated.orientation
    _, _, yaw = airsim.to_eularian_angles(orientation)
    return {
        "x": float(position.x_val),
        "y": float(position.y_val),
        "z": float(position.z_val),
        "yaw_rad": float(yaw),
        "yaw_deg": math.degrees(float(yaw)),
        "altitude": -float(position.z_val),
    }


def print_pose(client, vehicle_name):
    pose = get_pose(client, vehicle_name)
    print(
        f"X={pose['x']:.2f} "
        f"Y={pose['y']:.2f} "
        f"Z={pose['z']:.2f} "
        f"YAW={pose['yaw_deg']:.2f} "
        f"ALTITUDE={pose['altitude']:.2f}",
        flush=True,
    )


def setup_vehicle(client, vehicle_name):
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name=vehicle_name)
    client.armDisarm(True, vehicle_name=vehicle_name)

    try:
        state = client.getMultirotorState(vehicle_name=vehicle_name)
        landed_state = getattr(state, "landed_state", None)
        if landed_state == airsim.LandedState.Landed:
            client.takeoffAsync(vehicle_name=vehicle_name).join()
    except Exception:
        try:
            client.takeoffAsync(vehicle_name=vehicle_name).join()
        except Exception:
            pass

    client.moveToZAsync(TARGET_Z, 2.0, vehicle_name=vehicle_name).join()
    client.hoverAsync(vehicle_name=vehicle_name).join()


def velocity_from_body_command(client, vehicle_name, forward=0.0, right=0.0, vertical=0.0):
    pose = get_pose(client, vehicle_name)
    yaw = pose["yaw_rad"]
    vx = forward * math.cos(yaw) - right * math.sin(yaw)
    vy = forward * math.sin(yaw) + right * math.cos(yaw)
    vz = vertical
    return vx, vy, vz


def move_by_command(client, vehicle_name, command):
    forward = 0.0
    right = 0.0
    vertical = 0.0

    if command == "i":
        forward = MOVE_SPEED
    elif command == "k":
        forward = -MOVE_SPEED
    elif command == "j":
        right = -MOVE_SPEED
    elif command == "l":
        right = MOVE_SPEED
    elif command == "u":
        vertical = -VERTICAL_SPEED
    elif command == "o":
        vertical = VERTICAL_SPEED
    elif command == "h":
        client.hoverAsync(vehicle_name=vehicle_name).join()
        return
    else:
        print("Unknown command.", flush=True)
        return

    vx, vy, vz = velocity_from_body_command(
        client,
        vehicle_name,
        forward=forward,
        right=right,
        vertical=vertical,
    )
    client.moveByVelocityAsync(
        vx,
        vy,
        vz,
        MOVE_DURATION,
        vehicle_name=vehicle_name,
    ).join()


def print_help(vehicle_name):
    print(f"Manual XYZ drive started for vehicle={vehicle_name}", flush=True)
    print("Commands: i=forward, k=back, j=left, l=right, u=up, o=down, h=hover, x=exit", flush=True)


def main():
    args = parse_args()
    vehicle_name = args.vehicle
    client = airsim.MultirotorClient()
    setup_vehicle(client, vehicle_name)
    print_help(vehicle_name)
    print_pose(client, vehicle_name)

    try:
        while True:
            command = input("cmd> ").strip().lower()
            if command == "x":
                break
            if not command:
                print_pose(client, vehicle_name)
                continue
            move_by_command(client, vehicle_name, command)
            time.sleep(0.05)
            print_pose(client, vehicle_name)
    finally:
        try:
            client.hoverAsync(vehicle_name=vehicle_name).join()
        except Exception:
            pass
        print_pose(client, vehicle_name)


if __name__ == "__main__":
    main()
