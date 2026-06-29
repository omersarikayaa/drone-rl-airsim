#!/usr/bin/env python3
import math
import sys
import traceback


TEST_CASES = (
    {
        "name": "agac_to_agacSaklanan",
        "chaser": {
            "name": "agac",
            "x": 208.0238,
            "y": 18.1411,
            "z": -5.1752,
        },
        "target": {
            "name": "agacSaklanan",
            "x": 244.0125,
            "y": 15.6701,
            "z": -5.4853,
        },
        "distance_min": 34.0,
        "distance_max": 38.5,
    },
    {
        "name": "agac_to_agacSaklanan2",
        "chaser": {
            "name": "agac",
            "x": 208.0238,
            "y": 18.1411,
            "z": -5.5,
        },
        "target": {
            "name": "agacSaklanan2",
            "x": 210.01,
            "y": 10.60,
            "z": -7.0,
        },
        "distance_min": 6.0,
        "distance_max": 10.5,
    },
)
POSITION_TOLERANCE_METERS = 2.0
DISTANCE_TOLERANCE_METERS = 4.0


def distance_3d(pos_a, pos_b):
    dx = pos_b.x_val - pos_a.x_val
    dy = pos_b.y_val - pos_a.y_val
    dz = pos_b.z_val - pos_a.z_val
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def expected_distance(chaser_start, target_start):
    dx = target_start["x"] - chaser_start["x"]
    dy = target_start["y"] - chaser_start["y"]
    dz = target_start["z"] - chaser_start["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def assert_close_position(label, actual, expected):
    dx_err = abs(actual.x_val - expected["x"])
    dy_err = abs(actual.y_val - expected["y"])
    dz_err = abs(actual.z_val - expected["z"])
    print(
        f"[CHECK] {label} placement_error="
        f"(dx={dx_err:.2f}, dy={dy_err:.2f}, dz={dz_err:.2f})",
        flush=True,
    )

    if max(dx_err, dy_err, dz_err) > POSITION_TOLERANCE_METERS:
        raise RuntimeError(
            f"{label} start position is wrong: "
            f"actual=({actual.x_val:.2f}, {actual.y_val:.2f}, {actual.z_val:.2f}), "
            f"expected=({expected['x']:.2f}, {expected['y']:.2f}, {expected['z']:.2f})"
        )


def assert_not_left_at_origin(chaser_pos):
    horizontal_from_origin = math.sqrt(chaser_pos.x_val * chaser_pos.x_val + chaser_pos.y_val * chaser_pos.y_val)
    if horizontal_from_origin < 10.0:
        raise RuntimeError(
            "Chaser appears to be left near 0,0 instead of agac: "
            f"actual=({chaser_pos.x_val:.2f}, {chaser_pos.y_val:.2f}, {chaser_pos.z_val:.2f})"
        )


def run_case(case):
    from airsim_chase_env import AirSimChaseEnv, OBSERVATION_SIZE, VEHICLE_CHASER, VEHICLE_TARGET

    chaser_start = case["chaser"]
    target_start = case["target"]
    env = None

    try:
        print(f"[CHASER/TARGET START TEST] case={case['name']}", flush=True)
        print(
            "[CHASER/TARGET START TEST] "
            f"Chaser {chaser_start['name']}=({chaser_start['x']:.2f}, {chaser_start['y']:.2f}, {chaser_start['z']:.2f})",
            flush=True,
        )
        print(
            "[CHASER/TARGET START TEST] "
            f"Target {target_start['name']}=({target_start['x']:.2f}, {target_start['y']:.2f}, {target_start['z']:.2f})",
            flush=True,
        )

        env = AirSimChaseEnv(
            target_mode="simple",
            chaser_start_x=chaser_start["x"],
            chaser_start_y=chaser_start["y"],
            chaser_start_z=chaser_start["z"],
            target_start_x=target_start["x"],
            target_start_y=target_start["y"],
            target_start_z=target_start["z"],
            max_episode_steps=100,
        )
        obs, info = env.reset()

        if tuple(obs.shape) != (OBSERVATION_SIZE,):
            raise RuntimeError(f"Unexpected observation shape: {obs.shape}. Expected ({OBSERVATION_SIZE},).")

        chaser_pos = env.get_global_position(VEHICLE_CHASER)
        target_pos = env.get_global_position(VEHICLE_TARGET)
        start_distance = distance_3d(chaser_pos, target_pos)
        expected = expected_distance(chaser_start, target_start)

        print(
            f"[GLOBAL_POS] Chaser: x={chaser_pos.x_val:.2f}, y={chaser_pos.y_val:.2f}, z={chaser_pos.z_val:.2f}",
            flush=True,
        )
        print(
            f"[GLOBAL_POS] Target: x={target_pos.x_val:.2f}, y={target_pos.y_val:.2f}, z={target_pos.z_val:.2f}",
            flush=True,
        )
        print(
            f"[CHECK] start_distance={start_distance:.2f}, expected_about={expected:.2f}",
            flush=True,
        )

        assert_not_left_at_origin(chaser_pos)
        assert_close_position("Chaser", chaser_pos, chaser_start)
        assert_close_position("Target", target_pos, target_start)

        if abs(start_distance - expected) > DISTANCE_TOLERANCE_METERS:
            raise RuntimeError(
                f"Start distance is wrong: actual={start_distance:.2f}, expected about {expected:.2f}"
            )

        if not (case["distance_min"] <= start_distance <= case["distance_max"]):
            raise RuntimeError(
                f"Start distance is outside case bounds: actual={start_distance:.2f}, "
                f"expected range=({case['distance_min']:.2f}, {case['distance_max']:.2f})"
            )

    finally:
        if env is not None:
            env.close()


def main():
    try:
        print("[CHASER/TARGET START TEST] Starting AirSim reset placement tests.", flush=True)
        for case in TEST_CASES:
            run_case(case)
        print("CHASER/TARGET START POSITION TEST PASSED", flush=True)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        traceback.print_exc()
        print("CHASER/TARGET START POSITION TEST FAILED", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
