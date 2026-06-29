#!/usr/bin/env python3
import random
import traceback


INITIAL_ACTIONS = [
    0,  # FORWARD_TO_TARGET
    0,  # FORWARD_TO_TARGET
    2,  # MOVE_RIGHT
    1,  # MOVE_LEFT
    0,  # FORWARD_TO_TARGET
    5,  # HOVER
    3,  # MOVE_UP
    4,  # MOVE_DOWN
]
MAX_TEST_STEPS = 20
MIN_PASS_STEPS = 8
MIN_NATURAL_TERMINATION_STEPS = 3


def get_action(step_index):
    if step_index < len(INITIAL_ACTIONS):
        return INITIAL_ACTIONS[step_index]
    return random.randint(0, 5)


def termination_reason(info):
    if info["collision"]:
        return "effective_collision"
    if info["distance"] < 2.0:
        return "catch_distance"
    if info["distance"] > 80.0:
        return "too_far_distance"
    return "unknown"


def main():
    env = None
    steps_completed = 0
    step3_passed = False
    cleanup_ok = True
    terminated = False
    truncated = False
    end_reason = ""
    failure_message = ""

    try:
        from airsim_chase_env import AirSimChaseEnv, ACTION_NAMES, OBSERVATION_SIZE
    except ImportError as exc:
        print(f"STEP 3 FAILED: Import error. numpy and airsim are required. ({exc})", flush=True)
        return

    try:
        print("[STEP3] PPO-style AirSimChaseEnv random action test", flush=True)
        print("[INFO] This script does not train PPO. It only validates reset/step.", flush=True)

        env = AirSimChaseEnv()
        obs, info = env.reset()

        print(f"[RESET] obs_shape={obs.shape} obs={obs}", flush=True)
        print(
            "[RESET] "
            f"distance={info['distance']:.2f} "
            f"dx={info['dx']:.2f} dy={info['dy']:.2f} dz={info['dz']:.2f}",
            flush=True,
        )

        if tuple(obs.shape) != (OBSERVATION_SIZE,):
            raise RuntimeError(f"Unexpected observation shape: {obs.shape}. Expected ({OBSERVATION_SIZE},).")

        for step_index in range(MAX_TEST_STEPS):
            action = get_action(step_index)
            obs, reward, terminated, truncated, info = env.step(action)
            steps_completed += 1

            collision_object = info.get("collision_object_name") or ""
            collision_object_text = f" collision_object={collision_object}" if collision_object else ""
            print(
                f"[STEP {info['step']:03d}] "
                f"action={action}:{ACTION_NAMES[action]} "
                f"reward={reward:.2f} "
                f"distance={info['distance']:.2f} "
                f"dx={info['dx']:.2f} "
                f"dy={info['dy']:.2f} "
                f"raw_collision={info['raw_collision']} "
                f"collision={info['collision']} "
                f"terminated={terminated} "
                f"truncated={truncated}"
                f"{collision_object_text}",
                flush=True,
            )

            if terminated or truncated:
                end_reason = termination_reason(info) if terminated else "truncated"
                print(f"[INFO] Episode ended at step {steps_completed}: {end_reason}", flush=True)
                break

        if steps_completed >= MIN_PASS_STEPS:
            step3_passed = True
        elif terminated and steps_completed >= MIN_NATURAL_TERMINATION_STEPS:
            step3_passed = True
        else:
            raise RuntimeError(
                f"Only {steps_completed} env.step(action) calls completed; "
                f"expected at least {MIN_PASS_STEPS}, or natural termination after "
                f"at least {MIN_NATURAL_TERMINATION_STEPS} steps."
            )

    except Exception as exc:
        failure_message = str(exc)
        print(f"[ERROR] {failure_message}", flush=True)
        traceback.print_exc()

    finally:
        if env is not None:
            cleanup_ok = env.close()

    if step3_passed and cleanup_ok:
        print("STEP 3 PASSED: PPO-style AirSimChaseEnv reset/step interface works.", flush=True)
    else:
        if not failure_message:
            if not cleanup_ok:
                failure_message = "Environment cleanup did not complete cleanly."
            else:
                failure_message = "Unknown error."
        print(f"STEP 3 FAILED: {failure_message}", flush=True)


if __name__ == "__main__":
    main()
