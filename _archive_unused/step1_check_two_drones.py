#!/usr/bin/env python3
import json
import math
from pathlib import Path
import time
import traceback

import airsim


VEHICLES = ["Chaser", "Target"]
TAKEOFF_Z = -5
TAKEOFF_VELOCITY = 2
HOVER_SECONDS = 3
NEAR_ZERO_DISTANCE = 0.5


def log(level, message):
    print(f"[{level}] {message}", flush=True)


def list_vehicles_safe(client):
    try:
        vehicles = client.listVehicles()
    except AttributeError:
        log("WARN", "client.listVehicles() bu AirSim API surumunde yok; isim kontrolu API cagrisiyla denenecek.")
        return None
    except Exception as exc:
        log("WARN", f"AirSim vehicle listesi okunamadi: {exc}")
        return None

    log("INFO", f"AirSim vehicles: {vehicles}")
    return vehicles


def validate_vehicle_names(vehicles):
    if vehicles is None:
        return True

    missing = [name for name in VEHICLES if name not in vehicles]
    if not missing:
        return True

    missing_text = ", ".join(missing)
    raise RuntimeError(
        f"AirSim icinde beklenen drone isimleri bulunamadi: {missing_text}. "
        "Lutfen ~/Documents/AirSim/settings.json icinde Chaser ve Target araclarini "
        "kontrol edin ve AirSim'i yeniden baslatin."
    )


def load_expected_spawn_offset():
    settings_paths = [
        Path("~/Documents/AirSim/settings.json").expanduser(),
        Path("step1_settings_example.json"),
    ]

    for settings_path in settings_paths:
        try:
            with settings_path.open("r", encoding="utf-8") as settings_file:
                settings = json.load(settings_file)
        except FileNotFoundError:
            continue
        except Exception as exc:
            log("WARN", f"Settings okunamadi: {settings_path} ({exc})")
            continue

        vehicles = settings.get("Vehicles", {})
        if not all(name in vehicles for name in VEHICLES):
            continue

        try:
            chaser = vehicles["Chaser"]
            target = vehicles["Target"]
            dx = float(target.get("X", 0)) - float(chaser.get("X", 0))
            dy = float(target.get("Y", 0)) - float(chaser.get("Y", 0))
            dz = float(target.get("Z", 0)) - float(chaser.get("Z", 0))
        except Exception as exc:
            log("WARN", f"Settings spawn offset okunamadi: {settings_path} ({exc})")
            continue

        horizontal = math.sqrt(dx * dx + dy * dy)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        return {
            "source": str(settings_path),
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "horizontal": horizontal,
            "distance": distance,
        }

    log("WARN", "Chaser/Target spawn offset settings icinden okunamadi.")
    return None


def enable_api_and_arm(client):
    for name in VEHICLES:
        client.enableApiControl(True, vehicle_name=name)
        log("OK", f"API control enabled: {name}")

    for name in VEHICLES:
        client.armDisarm(True, vehicle_name=name)
        log("OK", f"Armed: {name}")


def takeoff_vehicle(client, name):
    task = client.takeoffAsync(vehicle_name=name)
    task.join()
    log("OK", f"Takeoff completed: {name}")


def move_to_safe_height(client, name):
    # AirSim NED koordinat sisteminde Z negatif deger aldikca drone yukari cikar.
    task = client.moveToZAsync(TAKEOFF_Z, TAKEOFF_VELOCITY, vehicle_name=name)
    task.join()
    log("OK", f"Safe height reached: {name} z={TAKEOFF_Z}")


def hover_vehicle(client, name):
    task = client.hoverAsync(vehicle_name=name)
    task.join()
    log("OK", f"Hover command accepted: {name}")


def get_position(client, vehicle_name):
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    return state.kinematics_estimated.position


def position_values(position):
    return position.x_val, position.y_val, position.z_val


def print_position(label, name, position):
    x_val, y_val, z_val = position_values(position)
    log(label, f"{name}: x={x_val:.2f}, y={y_val:.2f}, z={z_val:.2f}")


def is_valid_position(position):
    if position is None:
        return False

    try:
        values = position_values(position)
    except AttributeError:
        return False

    return all(math.isfinite(value) for value in values)


def get_pose_position(client, method_name, vehicle_name):
    method = getattr(client, method_name, None)
    if method is None:
        raise AttributeError(f"{method_name} bu AirSim Python client icinde yok")

    if method_name == "simGetObjectPose":
        pose = method(vehicle_name)
    else:
        try:
            pose = method(vehicle_name=vehicle_name)
        except TypeError:
            pose = method(vehicle_name)

    position = getattr(pose, "position", None)
    if not is_valid_position(position):
        raise ValueError(f"{method_name} invalid pose dondurdu: {vehicle_name}")

    return position


def get_global_positions_with_method(client, method_name):
    positions = {}

    for name in VEHICLES:
        try:
            positions[name] = get_pose_position(client, method_name, name)
        except Exception as exc:
            log("WARN", f"{method_name} global/world pose okuyamadi: {name} ({exc})")
            return None

    return positions


def get_offset_corrected_positions(state_positions, expected_offset):
    if expected_offset is None:
        return None

    corrected = {}
    for name in VEHICLES:
        x_val, y_val, z_val = position_values(state_positions[name])
        if name == "Target":
            x_val += expected_offset["dx"]
            y_val += expected_offset["dy"]
            z_val += expected_offset["dz"]
        corrected[name] = airsim.Vector3r(x_val, y_val, z_val)

    return corrected


def select_global_positions(client, expected_offset):
    first_valid_positions = None
    first_valid_method = None

    for method_name in ("simGetObjectPose", "simGetVehiclePose"):
        positions = get_global_positions_with_method(client, method_name)
        if positions is None:
            continue

        dx, dy, dz, distance = relative_distance(positions["Chaser"], positions["Target"])
        if first_valid_positions is None:
            first_valid_positions = positions
            first_valid_method = method_name

        if expected_offset and expected_offset["horizontal"] > NEAR_ZERO_DISTANCE and distance < NEAR_ZERO_DISTANCE:
            log(
                "WARN",
                f"{method_name} numeric pose verdi ama relative distance near zero: "
                f"{distance:.2f} m. Settings expected horizontal offset: "
                f"{expected_offset['horizontal']:.2f} m.",
            )
            continue

        log("INFO", f"Global/world pose method selected: {method_name}")
        return positions, method_name

    if first_valid_positions is not None:
        log("WARN", f"Global pose methods near-zero gorunuyor; ilk numeric sonuc kullaniliyor: {first_valid_method}")
        return first_valid_positions, first_valid_method

    log("WARN", "Global/world pose method bulunamadi veya tum pose degerleri invalid.")
    return None, None


def print_positions(label, positions):
    for name in VEHICLES:
        print_position(label, name, positions[name])


def relative_distance(chaser_position, target_position):
    dx = target_position.x_val - chaser_position.x_val
    dy = target_position.y_val - chaser_position.y_val
    dz = target_position.z_val - chaser_position.z_val
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dx, dy, dz, distance


def land_vehicle_safe(client, name):
    try:
        task = client.landAsync(vehicle_name=name)
        task.join()
        log("OK", f"Landing completed: {name}")
        return True
    except Exception as exc:
        log("WARN", f"Landing failed for {name}: {exc}")
        return False


def cleanup_vehicle_safe(client, name):
    cleanup_ok = True

    try:
        client.armDisarm(False, vehicle_name=name)
        log("OK", f"Disarmed: {name}")
    except Exception as exc:
        log("WARN", f"Disarm failed for {name}: {exc}")
        cleanup_ok = False

    try:
        client.enableApiControl(False, vehicle_name=name)
        log("OK", f"API control disabled: {name}")
    except Exception as exc:
        log("WARN", f"Disable API control failed for {name}: {exc}")
        cleanup_ok = False

    return cleanup_ok


def main():
    client = None
    step_passed = False
    step_partial = False
    cleanup_ok = True
    failure_message = ""
    vehicles_listed = False

    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        log("OK", "AirSim baglantisi kuruldu.")

        vehicles = list_vehicles_safe(client)
        validate_vehicle_names(vehicles)
        vehicles_listed = vehicles is not None and all(name in vehicles for name in VEHICLES)
        if not vehicles_listed:
            log("WARN", "Chaser ve Target listVehicles() ile dogrulanamadi; STEP 1 pass sayilmayacak.")

        expected_offset = load_expected_spawn_offset()
        if expected_offset:
            log("INFO", f"Settings offset source: {expected_offset['source']}")

        enable_api_and_arm(client)

        for name in VEHICLES:
            takeoff_vehicle(client, name)

        for name in VEHICLES:
            move_to_safe_height(client, name)

        for name in VEHICLES:
            hover_vehicle(client, name)

        state_positions = {name: get_position(client, name) for name in VEHICLES}
        print_positions("STATE_POS", state_positions)

        global_positions, _global_method = select_global_positions(client, expected_offset)
        if global_positions is not None:
            print_positions("GLOBAL_POS", global_positions)
            rel_source = "GLOBAL"
            rel_positions = global_positions
        else:
            corrected_positions = get_offset_corrected_positions(state_positions, expected_offset)
            if corrected_positions is None:
                rel_source = "STATE_LOCAL"
                rel_positions = state_positions
                log("WARN", "GLOBAL_POS yok; local state pozisyonlari ile gecici hesap yapiliyor.")
            else:
                rel_source = "STATE_PLUS_SETTINGS"
                rel_positions = corrected_positions
                log("WARN", "GLOBAL_POS yok; local state + settings spawn offset ile gecici hesap yapiliyor.")

        dx, dy, dz, distance = relative_distance(rel_positions["Chaser"], rel_positions["Target"])
        log(
            f"REL_{rel_source}",
            "Target relative to Chaser: "
            f"dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, distance={distance:.2f}",
        )

        if expected_offset:
            log(
                "CHECK",
                "Expected initial horizontal offset from settings is about "
                f"{expected_offset['horizontal']:.2f} meters.",
            )

        positions_readable = all(is_valid_position(position) for position in state_positions.values())
        global_readable = global_positions is not None and all(is_valid_position(position) for position in global_positions.values())
        expected_separation = expected_offset and expected_offset["horizontal"] > NEAR_ZERO_DISTANCE
        relative_near_zero = distance < NEAR_ZERO_DISTANCE

        if not vehicles_listed:
            failure_message = "Chaser ve Target listVehicles() ile dogrulanamadi."
        elif not positions_readable:
            failure_message = "STATE/LOCAL pozisyonlari okunamadi."
        elif expected_separation and relative_near_zero:
            step_partial = True
            log("WARN", "Relative distance near zero; global relative coordinate reading is not correct yet.")
        elif not global_readable:
            step_partial = True
            log("WARN", "Global/world coordinate reading dogrulanamadi; fallback ile mesafe hesaplandi.")
        else:
            step_passed = True

        log("INFO", f"Hover/bekleme: {HOVER_SECONDS} saniye")
        time.sleep(HOVER_SECONDS)

    except Exception as exc:
        failure_message = str(exc)
        log("ERROR", failure_message)
        traceback.print_exc()

    finally:
        if client is not None:
            log("INFO", "Guvenli landing ve cleanup basliyor.")
            for name in VEHICLES:
                cleanup_ok = land_vehicle_safe(client, name) and cleanup_ok

            if cleanup_ok:
                log("OK", "Landing completed.")

            for name in VEHICLES:
                cleanup_ok = cleanup_vehicle_safe(client, name) and cleanup_ok

            if cleanup_ok:
                log("OK", "Cleanup completed.")
            else:
                log("WARN", "Landing veya cleanup adimlarindan en az biri basarisiz oldu.")

    if step_passed and cleanup_ok:
        print("STEP 1 PASSED: two-drone connection and global coordinate reading works.", flush=True)
    elif step_partial and cleanup_ok:
        print("STEP 1 PARTIAL: two drones work, but global relative coordinate reading is not correct yet.", flush=True)
    else:
        if not failure_message:
            if not cleanup_ok:
                failure_message = "Landing veya cleanup tamamlanamadi."
            else:
                failure_message = "Step 1 kosullari tamamlanamadi."
        print(f"STEP 1 FAILED: {failure_message}", flush=True)


if __name__ == "__main__":
    main()
