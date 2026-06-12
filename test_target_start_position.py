#!/usr/bin/env python3
import math
import traceback


TEST_CASES = (
    (15.0, 0.0, -5.0),
    (30.0, 15.0, -5.0),
)


def check_close(label, actual, expected):
    tolerance = max(2.0, abs(expected) * 0.25)
    if abs(actual - expected) > tolerance:
        raise RuntimeError(f"{label} is wrong: actual={actual:.2f}, expected about {expected:.2f}")


def run_case(target_start_x, target_start_y, target_start_z):
    from airsim_chase_env import AirSimChaseEnv, OBSERVATION_SIZE

    env = None
    try:
        print(
            "[TARGET START TEST] "
            f"requested x={target_start_x:.2f} y={target_start_y:.2f}",
            flush=True,
        )
        env = AirSimChaseEnv(
            target_mode="simple",
            target_start_x=target_start_x,
            target_start_y=target_start_y,
            target_start_z=target_start_z,
        )
        obs, info = env.reset()

        if tuple(obs.shape) != (OBSERVATION_SIZE,):
            raise RuntimeError(f"Unexpected observation shape: {obs.shape}. Expected ({OBSERVATION_SIZE},).")

        dx = float(info["dx"])
        dy = float(info["dy"])
        distance = float(info["distance"])
        expected_distance = math.sqrt(target_start_x * target_start_x + target_start_y * target_start_y)

        print(
            "[TARGET START TEST] "
            f"actual dx={dx:.2f} dy={dy:.2f} distance={distance:.2f}",
            flush=True,
        )

        check_close("dx", dx, target_start_x)
        check_close("dy", dy, target_start_y)
        check_close("distance", distance, expected_distance)

    finally:
        if env is not None:
            env.close()


def main():
    try:
        for target_start_x, target_start_y, target_start_z in TEST_CASES:
            run_case(target_start_x, target_start_y, target_start_z)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        traceback.print_exc()
        print("TARGET START POSITION TEST FAILED", flush=True)
        return

    print("TARGET START POSITION TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
