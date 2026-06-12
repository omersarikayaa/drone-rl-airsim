#!/usr/bin/env python3
import traceback


ACTION_SEQUENCE = [
    0,  # FORWARD_TO_TARGET
    0,
    2,  # MOVE_RIGHT
    1,  # MOVE_LEFT
    0,
    5,  # HOVER
    3,  # MOVE_UP
    4,  # MOVE_DOWN
    0,
    0,
    5,
    2,
    1,
    0,
    5,
]
MIN_PASS_STEPS = 10
MIN_NATURAL_TERMINATION_STEPS = 3
REQUIRED_LIDAR_FIELDS = (
    "lidar_front",
    "lidar_front_left",
    "lidar_front_right",
    "lidar_left",
    "lidar_right",
    "lidar_back",
)


def print_lidar(prefix, info):
    print(
        f"[{prefix}] "
        f"front={info['lidar_front']:.2f}, "
        f"front_left={info['lidar_front_left']:.2f}, "
        f"front_right={info['lidar_front_right']:.2f}, "
        f"left={info['lidar_left']:.2f}, "
        f"right={info['lidar_right']:.2f}, "
        f"back={info['lidar_back']:.2f}",
        flush=True,
    )


def validate_lidar_fields(info):
    missing = [name for name in REQUIRED_LIDAR_FIELDS if name not in info]
    if missing:
        raise RuntimeError(f"Missing LiDAR fields in info: {missing}")


def main():
    env = None
    steps_completed = 0
    lidar_seen = False
    step4_passed = False
    step4_partial = False
    cleanup_ok = True
    failure_message = ""

    try:
        from airsim_chase_env import AirSimChaseEnv, ACTION_NAMES, OBSERVATION_SIZE
    except ImportError as exc:
        print(f"STEP 4 FAILED: Import error. numpy and airsim are required. ({exc})", flush=True)
        return

    try:
        print("[STEP4] Observation + LiDAR validation test", flush=True)
        print("[INFO] This script does not train PPO. It only validates observation fields.", flush=True)

        env = AirSimChaseEnv()
        obs, info = env.reset()
        validate_lidar_fields(info)
        lidar_seen = bool(info["lidar_available"])

        print(f"[RESET] obs_shape={obs.shape}", flush=True)
        print(f"[RESET] distance={info['distance']:.2f}", flush=True)
        print(f"[RESET] lidar_available={info['lidar_available']}", flush=True)
        print(f"[RESET] lidar_point_count={info['lidar_point_count']}", flush=True)
        print_lidar("RESET_LIDAR", info)

        if tuple(obs.shape) != (OBSERVATION_SIZE,):
            raise RuntimeError(f"Unexpected observation shape: {obs.shape}. Expected ({OBSERVATION_SIZE},).")

        for action in ACTION_SEQUENCE:
            obs, reward, terminated, truncated, info = env.step(action)
            steps_completed += 1
            validate_lidar_fields(info)
            lidar_seen = lidar_seen or bool(info["lidar_available"])

            print(
                f"[STEP {info['step']:03d}] "
                f"action={action}:{ACTION_NAMES[action]} "
                f"reward={reward:.2f} "
                f"distance={info['distance']:.2f} "
                f"lidar_available={info['lidar_available']} "
                f"points={info['lidar_point_count']} "
                f"front={info['lidar_front']:.2f} "
                f"left={info['lidar_left']:.2f} "
                f"right={info['lidar_right']:.2f} "
                f"back={info['lidar_back']:.2f} "
                f"collision={info['collision']} "
                f"terminated={terminated} "
                f"truncated={truncated}",
                flush=True,
            )

            if terminated or truncated:
                print(f"[INFO] Episode ended at step {steps_completed}.", flush=True)
                break

        if steps_completed >= MIN_PASS_STEPS:
            step4_passed = lidar_seen
            step4_partial = not lidar_seen
        elif (terminated or truncated) and steps_completed >= MIN_NATURAL_TERMINATION_STEPS:
            step4_passed = lidar_seen
            step4_partial = not lidar_seen
        else:
            raise RuntimeError(
                f"Only {steps_completed} env.step(action) calls completed; expected at least "
                f"{MIN_PASS_STEPS}, or natural termination after at least "
                f"{MIN_NATURAL_TERMINATION_STEPS} steps."
            )

        if not lidar_seen:
            print(
                "[WARN] LiDAR point_count stayed 0. Check ~/Documents/AirSim/settings.json "
                "for Lidar1 Enabled=true on Chaser.",
                flush=True,
            )

    except Exception as exc:
        failure_message = str(exc)
        print(f"[ERROR] {failure_message}", flush=True)
        traceback.print_exc()

    finally:
        if env is not None:
            cleanup_ok = env.close()

    if step4_passed and cleanup_ok:
        print("STEP 4 PASSED: observation includes target-relative state and LiDAR sector distances.", flush=True)
    elif step4_partial and cleanup_ok:
        print("STEP 4 PARTIAL: observation shape works but LiDAR data is not available.", flush=True)
    else:
        if not failure_message:
            if not cleanup_ok:
                failure_message = "Environment cleanup did not complete cleanly."
            else:
                failure_message = "Step 4 validation did not complete."
        print(f"STEP 4 FAILED: {failure_message}", flush=True)


if __name__ == "__main__":
    main()
