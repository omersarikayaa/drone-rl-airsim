#!/usr/bin/env python3
import math
import traceback


ACTION_SEQUENCE = [
    5,  # HOVER
    0,  # FORWARD_TO_TARGET
    0,
    2,  # MOVE_RIGHT
    0,
    1,  # MOVE_LEFT
    0,
    5,
    0,
    0,
    2,
    1,
    0,
    5,
    0,
    0,
    2,
    1,
    0,
    5,
    0,
    0,
    2,
    1,
    0,
]
MIN_STEPS = 10
MIN_TARGET_MOVE = 1.0
MIN_DISTANCE_CHANGE = 0.5


def distance_3d(pos_a, pos_b):
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    dz = pos_b[2] - pos_a[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def main():
    env = None
    steps_completed = 0
    failure_message = ""
    test_passed = False
    cleanup_ok = True
    target_positions = []
    distances = []

    try:
        from airsim_chase_env import ACTION_NAMES, AirSimChaseEnv, OBSERVATION_SIZE

        print("[EVASIVE TEST] TargetController validation", flush=True)
        print("[INFO] This test does not train PPO. It only validates evasive Target behavior.", flush=True)

        env = AirSimChaseEnv(target_mode="evasive")
        obs, info = env.reset()
        target_positions.append(info["target_pos"])
        distances.append(info["distance"])

        print(
            f"[RESET] obs_shape={obs.shape} "
            f"distance={info['distance']:.2f} "
            f"target_mode={info['target_mode']} "
            f"target_pos={info['target_pos']}",
            flush=True,
        )

        if tuple(obs.shape) != (OBSERVATION_SIZE,):
            raise RuntimeError(f"Unexpected observation shape: {obs.shape}. Expected ({OBSERVATION_SIZE},).")

        for action in ACTION_SEQUENCE:
            obs, reward, terminated, truncated, info = env.step(action)
            steps_completed += 1
            target_positions.append(info["target_pos"])
            distances.append(info["distance"])

            print(
                f"[EVASIVE TEST STEP {info['step']:03d}] "
                f"distance={info['distance']:.2f} "
                f"target_vx={info['target_vx']:.2f} "
                f"target_vy={info['target_vy']:.2f} "
                f"target_vz={info['target_vz']:.2f} "
                f"target_mode={info['target_mode']} "
                f"chaser_action={action}:{ACTION_NAMES.get(action, 'UNKNOWN')} "
                f"lidar_front={info['lidar_front']:.2f} "
                f"collision={info['collision']}",
                flush=True,
            )

            if terminated or truncated:
                print(f"[INFO] Episode ended at step {steps_completed}: {info['terminated_reason']}", flush=True)
                break

        if steps_completed < MIN_STEPS:
            raise RuntimeError(f"Only {steps_completed} steps completed; expected at least {MIN_STEPS}.")

        target_displacement = distance_3d(target_positions[0], target_positions[-1])
        distance_change = max(distances) - min(distances)
        print(f"[CHECK] target_displacement={target_displacement:.2f}", flush=True)
        print(f"[CHECK] distance_change={distance_change:.2f}", flush=True)

        if target_displacement < MIN_TARGET_MOVE:
            raise RuntimeError("Target did not move enough in global coordinates.")
        if distance_change < MIN_DISTANCE_CHANGE:
            raise RuntimeError("Chaser-Target distance did not change enough.")

        test_passed = True

    except Exception as exc:
        failure_message = str(exc)
        print(f"[ERROR] {failure_message}", flush=True)
        traceback.print_exc()

    finally:
        if env is not None:
            cleanup_ok = env.close()

    if test_passed and cleanup_ok:
        print("EVASIVE TARGET TEST PASSED: TargetController evasive behavior works.", flush=True)
    else:
        if not failure_message:
            if not cleanup_ok:
                failure_message = "Cleanup did not complete cleanly."
            else:
                failure_message = "Evasive Target validation did not complete."
        print(f"EVASIVE TARGET TEST FAILED: {failure_message}", flush=True)


if __name__ == "__main__":
    main()
