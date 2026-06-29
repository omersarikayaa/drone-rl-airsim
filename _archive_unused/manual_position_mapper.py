#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import traceback

import airsim


VEHICLES = ("Chaser", "Target")
MOVE_COMMANDS = ("w", "s", "a", "d", "q", "e")


def log(level, message):
    print(f"[{level}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Move one AirSim drone from terminal commands and map positions.")
    parser.add_argument("--vehicle", choices=VEHICLES, default="Chaser", help="Vehicle to control.")
    parser.add_argument("--speed", type=float, default=2.0, help="Movement speed in m/s.")
    parser.add_argument("--z", type=float, default=-5.0, help="Initial NED Z altitude.")
    parser.add_argument("--output", default="mapped_positions.json", help="JSON file for saved points.")
    return parser.parse_args()


def format_position(position):
    return f"x={position.x_val:.2f} y={position.y_val:.2f} z={position.z_val:.2f}"


def validate_position(position, vehicle_name):
    if position is None:
        raise RuntimeError(f"simGetObjectPose returned no position for {vehicle_name}.")

    values = (position.x_val, position.y_val, position.z_val)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Invalid GLOBAL/WORLD position for {vehicle_name}: {values}")

    return position


def get_global_position(client, vehicle_name):
    try:
        pose = client.simGetObjectPose(vehicle_name)
    except Exception as exc:
        raise RuntimeError(f"simGetObjectPose failed for {vehicle_name}: {exc}") from exc

    return validate_position(getattr(pose, "position", None), vehicle_name)


def print_position(client, vehicle_name):
    position = get_global_position(client, vehicle_name)
    print(f"[POS] vehicle={vehicle_name} {format_position(position)}", flush=True)
    return position


def load_positions(output_path):
    if not output_path.exists():
        return {}

    try:
        with output_path.open("r", encoding="utf-8") as output_file:
            data = json.load(output_file)
    except Exception as exc:
        log("WARN", f"Existing output file could not be read; starting empty. ({exc})")
        return {}

    if not isinstance(data, dict):
        log("WARN", "Existing output file is not a JSON object; starting empty.")
        return {}

    return data


def save_positions(output_path, positions):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(positions, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def save_current_point(client, vehicle_name, output_path, point_name):
    point_name = point_name.strip()
    if not point_name:
        log("WARN", "Usage: p point_name")
        return

    position = get_global_position(client, vehicle_name)
    positions = load_positions(output_path)
    positions[point_name] = {
        "vehicle": vehicle_name,
        "x": round(float(position.x_val), 4),
        "y": round(float(position.y_val), 4),
        "z": round(float(position.z_val), 4),
    }
    save_positions(output_path, positions)
    log("SAVE", f"{point_name}: vehicle={vehicle_name} {format_position(position)} -> {output_path}")


def validate_vehicle(client, vehicle_name):
    try:
        vehicles = client.listVehicles()
        log("INFO", f"AirSim vehicles: {vehicles}")
    except Exception as exc:
        log("WARN", f"AirSim vehicles list could not be read: {exc}")
        return

    if vehicle_name not in vehicles:
        raise RuntimeError(f"Vehicle not found in AirSim: {vehicle_name}. Available vehicles: {vehicles}")


def setup_vehicle(client, vehicle_name, z_value, speed):
    client.enableApiControl(True, vehicle_name=vehicle_name)
    log("OK", f"API control enabled: {vehicle_name}")
    client.armDisarm(True, vehicle_name=vehicle_name)
    log("OK", f"Armed: {vehicle_name}")
    client.takeoffAsync(vehicle_name=vehicle_name).join()
    log("OK", f"Takeoff completed: {vehicle_name}")
    client.moveToZAsync(z_value, speed, vehicle_name=vehicle_name).join()
    log("OK", f"Initial altitude reached: {vehicle_name} z={z_value:.2f}")
    client.hoverAsync(vehicle_name=vehicle_name).join()
    log("OK", f"Hovering: {vehicle_name}")


def hover_and_disable_api(client, vehicle_name):
    cleanup_ok = True
    try:
        client.hoverAsync(vehicle_name=vehicle_name).join()
        log("OK", f"Hover completed: {vehicle_name}")
    except Exception as exc:
        cleanup_ok = False
        log("WARN", f"Hover failed: {vehicle_name} ({exc})")

    try:
        client.enableApiControl(False, vehicle_name=vehicle_name)
        log("OK", f"API control disabled: {vehicle_name}")
    except Exception as exc:
        cleanup_ok = False
        log("WARN", f"Disable API control failed: {vehicle_name} ({exc})")

    return cleanup_ok


def land_disarm_and_disable_api(client, vehicle_name):
    cleanup_ok = True
    try:
        client.landAsync(vehicle_name=vehicle_name).join()
        log("OK", f"Landing completed: {vehicle_name}")
    except Exception as exc:
        cleanup_ok = False
        log("WARN", f"Landing failed: {vehicle_name} ({exc})")

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


def command_to_velocity(command, speed):
    if command == "w":
        return speed, 0.0, 0.0
    if command == "s":
        return -speed, 0.0, 0.0
    if command == "a":
        return 0.0, -speed, 0.0
    if command == "d":
        return 0.0, speed, 0.0
    if command == "q":
        return 0.0, 0.0, -speed
    if command == "e":
        return 0.0, 0.0, speed
    return None


def parse_duration(command, value):
    if value is None:
        raise ValueError(f"Usage: {command} seconds")

    duration = float(value)
    if duration <= 0.0:
        raise ValueError("Duration must be positive.")
    return duration


def print_help():
    log("INFO", "Commands:")
    log("INFO", "  w 2    -> +X for 2 seconds")
    log("INFO", "  s 2    -> -X for 2 seconds")
    log("INFO", "  a 2    -> -Y for 2 seconds")
    log("INFO", "  d 2    -> +Y for 2 seconds")
    log("INFO", "  q 1    -> up, z more negative")
    log("INFO", "  e 1    -> down, z more positive")
    log("INFO", "  h      -> hover")
    log("INFO", "  pos    -> print GLOBAL/WORLD position")
    log("INFO", "  p name -> save current GLOBAL/WORLD position")
    log("INFO", "  land   -> land, disarm, disable API, exit")
    log("INFO", "  exit   -> hover, disable API, exit")


def handle_move_command(client, vehicle_name, speed, command, duration_text):
    duration = parse_duration(command, duration_text)
    vx, vy, vz = command_to_velocity(command, speed)
    client.moveByVelocityAsync(
        vx,
        vy,
        vz,
        duration,
        vehicle_name=vehicle_name,
    ).join()
    print_position(client, vehicle_name)


def run_command_loop(client, args, output_path):
    print_help()
    print_position(client, args.vehicle)

    while True:
        raw_command = input("command> ").strip()
        if not raw_command:
            continue

        command_parts = raw_command.split(maxsplit=1)
        command = command_parts[0].lower()
        value = command_parts[1] if len(command_parts) > 1 else None

        try:
            if command in MOVE_COMMANDS:
                handle_move_command(client, args.vehicle, args.speed, command, value)
            elif command == "h":
                client.hoverAsync(vehicle_name=args.vehicle).join()
                log("OK", f"Hover command accepted: {args.vehicle}")
                print_position(client, args.vehicle)
            elif command == "pos":
                print_position(client, args.vehicle)
            elif command == "p":
                save_current_point(client, args.vehicle, output_path, value or "")
                print_position(client, args.vehicle)
            elif command == "land":
                land_disarm_and_disable_api(client, args.vehicle)
                return "land"
            elif command == "exit":
                hover_and_disable_api(client, args.vehicle)
                return "exit"
            elif command in ("help", "?"):
                print_help()
            else:
                log("WARN", f"Unknown command: {raw_command}")
                print_help()
        except ValueError as exc:
            log("WARN", str(exc))


def main():
    args = parse_args()
    output_path = Path(args.output).expanduser()
    client = None
    setup_completed = False
    cleanup_done = False

    try:
        log("INFO", "Manual position mapper starting.")
        log("INFO", f"vehicle={args.vehicle} speed={args.speed:.2f} z={args.z:.2f} output={output_path}")
        log("INFO", "Only the selected vehicle will be controlled.")

        client = airsim.MultirotorClient()
        client.confirmConnection()
        log("OK", "AirSim connected.")
        validate_vehicle(client, args.vehicle)

        setup_vehicle(client, args.vehicle, args.z, args.speed)
        setup_completed = True

        result = run_command_loop(client, args, output_path)
        cleanup_done = result in ("land", "exit")

    except KeyboardInterrupt:
        print("", flush=True)
        log("INFO", "Interrupted by user.")
    except Exception as exc:
        log("ERROR", str(exc))
        traceback.print_exc()
    finally:
        if client is not None and setup_completed and not cleanup_done:
            hover_and_disable_api(client, args.vehicle)


if __name__ == "__main__":
    main()
