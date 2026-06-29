#!/usr/bin/env python3
import traceback

from reward_utils import compute_chase_reward
from safety_filter import (
    ACTION_FORWARD_TO_TARGET,
    ACTION_HOVER,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_NAMES,
    apply_safety_filter,
)


ACTION_SEQUENCE = [
    0,  # FORWARD_TO_TARGET
    0,
    0,
    2,
    1,
    5,
    3,
    4,
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


def action_name(action):
    return ACTION_NAMES.get(action, "UNKNOWN")


def assert_condition(condition, message):
    if not condition:
        raise AssertionError(message)


def run_safety_unit_tests():
    cases = [
        {
            "name": "front_danger",
            "action": ACTION_FORWARD_TO_TARGET,
            "lidar": {"front": 1.0, "left": 10.0, "right": 10.0},
            "check": lambda result: result["overridden"]
            and result["safe_action"] == ACTION_FORWARD_TO_TARGET
            and result.get("diagonal_bypass", False)
            and result.get("bypass_direction") in ("left", "right")
            and result["risk_level"] == "danger",
        },
        {
            "name": "front_slow",
            "action": ACTION_FORWARD_TO_TARGET,
            "lidar": {"front": 4.0, "left": 10.0, "right": 10.0},
            "check": lambda result: not result["overridden"] and result["risk_level"] == "slow",
        },
        {
            "name": "left_danger",
            "action": ACTION_MOVE_LEFT,
            "lidar": {"front": 10.0, "left": 0.8, "right": 10.0},
            "check": lambda result: result["overridden"] and result["safe_action"] == ACTION_MOVE_RIGHT,
        },
        {
            "name": "right_danger",
            "action": ACTION_MOVE_RIGHT,
            "lidar": {"front": 10.0, "left": 10.0, "right": 0.8},
            "check": lambda result: result["overridden"] and result["safe_action"] == ACTION_MOVE_LEFT,
        },
        {
            "name": "down_near_ground",
            "action": ACTION_MOVE_DOWN,
            "lidar": {"front": 10.0, "left": 10.0, "right": 10.0},
            "chaser_z": -1.0,
            "check": lambda result: result["overridden"] and result["safe_action"] == ACTION_HOVER,
        },
    ]

    for case in cases:
        result = apply_safety_filter(case["action"], case["lidar"], case.get("chaser_z"))
        assert_condition(case["check"](result), f"Safety unit failed: {case['name']} -> {result}")
        print(
            f"[SAFETY_UNIT] case={case['name']} "
            f"original={action_name(result['original_action'])} "
            f"safe={action_name(result['safe_action'])} "
            f"overridden={result['overridden']} "
            f"reason={result['reason']}",
            flush=True,
        )


def run_reward_unit_tests():
    cases = [
        {
            "name": "distance_improved",
            "kwargs": {"previous_distance": 10.0, "distance": 8.0},
            "check": lambda result: result["distance_delta_reward"] > 0.0,
        },
        {
            "name": "distance_worse",
            "kwargs": {"previous_distance": 8.0, "distance": 10.0},
            "check": lambda result: result["distance_delta_reward"] < 0.0,
        },
        {
            "name": "caught",
            "kwargs": {"previous_distance": 3.0, "distance": 1.5, "caught": True},
            "check": lambda result: result["catch_reward"] == 50.0,
        },
        {
            "name": "collision",
            "kwargs": {"previous_distance": 5.0, "distance": 5.0, "collision": True},
            "check": lambda result: result["collision_penalty"] == -50.0,
        },
        {
            "name": "safety_override",
            "kwargs": {"previous_distance": 5.0, "distance": 5.0, "safety_overridden": True},
            "check": lambda result: result["safety_override_penalty"] == -2.0,
        },
    ]

    for case in cases:
        result = compute_chase_reward(**case["kwargs"])
        assert_condition(case["check"](result), f"Reward unit failed: {case['name']} -> {result}")
        print(f"[REWARD_UNIT] case={case['name']} total={result['total']:.2f} breakdown={result}", flush=True)


def format_reward_parts(reward_breakdown):
    return (
        "{"
        f"distance={reward_breakdown['distance_delta_reward']:.2f}, "
        f"obstacle={reward_breakdown['obstacle_penalty']:.2f}, "
        f"safety={reward_breakdown['safety_override_penalty']:.2f}, "
        f"collision={reward_breakdown['collision_penalty']:.2f}, "
        f"catch={reward_breakdown['catch_reward']:.2f}, "
        f"step={reward_breakdown['step_penalty']:.2f}"
        "}"
    )


def run_airsim_integration_test():
    from airsim_chase_env import AirSimChaseEnv, OBSERVATION_SIZE

    env = None
    steps_completed = 0
    cleanup_ok = True
    integration_ok = False
    integration_message = ""

    try:
        env = AirSimChaseEnv()
        obs, info = env.reset()

        print(f"[RESET] obs_shape={obs.shape}", flush=True)
        print(
            f"[RESET_LIDAR] front={info['lidar_front']:.2f} "
            f"left={info['lidar_left']:.2f} right={info['lidar_right']:.2f}",
            flush=True,
        )

        assert_condition(tuple(obs.shape) == (OBSERVATION_SIZE,), f"Unexpected obs shape: {obs.shape}")

        for action in ACTION_SEQUENCE:
            obs, reward, terminated, truncated, info = env.step(action)
            steps_completed += 1

            assert_condition("safety_overridden" in info, "Missing safety_overridden in info.")
            assert_condition("safety_reason" in info, "Missing safety_reason in info.")
            assert_condition("reward_breakdown" in info, "Missing reward_breakdown in info.")

            original = info["safety_original_action"]
            safe = info["safety_safe_action"]
            reward_parts = format_reward_parts(info["reward_breakdown"])
            print(
                f"[STEP {info['step']:03d}] "
                f"original={original}:{info['safety_original_action_name']} "
                f"safe={safe}:{info['safety_safe_action_name']} "
                f"overridden={info['safety_overridden']} "
                f"risk={info['safety_risk_level']} "
                f"reward={reward:.2f} "
                f"distance={info['distance']:.2f} "
                f"front={info['lidar_front']:.2f} "
                f"left={info['lidar_left']:.2f} "
                f"right={info['lidar_right']:.2f} "
                f"reason=\"{info['safety_reason']}\" "
                f"reward_parts={reward_parts}",
                flush=True,
            )

            if terminated or truncated:
                print(f"[INFO] Episode ended at step {steps_completed}.", flush=True)
                break

        if steps_completed >= MIN_PASS_STEPS:
            integration_ok = True
        elif (terminated or truncated) and steps_completed >= MIN_NATURAL_TERMINATION_STEPS:
            integration_ok = True
        else:
            integration_message = (
                f"Only {steps_completed} env.step(action) calls completed; expected at least "
                f"{MIN_PASS_STEPS}, or natural termination after at least "
                f"{MIN_NATURAL_TERMINATION_STEPS} steps."
            )

    finally:
        if env is not None:
            cleanup_ok = env.close()
            if not cleanup_ok:
                print("[WARN] AirSim integration cleanup had warnings.", flush=True)

    return integration_ok, cleanup_ok, integration_message


def main():
    pure_tests_passed = False
    integration_passed = False
    integration_attempted = False
    failure_message = ""

    try:
        print("[STEP5] Reward breakdown + safety filter validation test", flush=True)
        print("[INFO] This script does not train PPO. It only validates safety/reward integration.", flush=True)

        run_safety_unit_tests()
        run_reward_unit_tests()
        pure_tests_passed = True

        integration_attempted = True
        integration_passed, cleanup_ok, integration_message = run_airsim_integration_test()
        if not cleanup_ok:
            integration_passed = False
            integration_message = "AirSim cleanup did not complete cleanly."
        if integration_message:
            failure_message = integration_message

    except Exception as exc:
        failure_message = str(exc)
        print(f"[ERROR] {failure_message}", flush=True)
        traceback.print_exc()

    if pure_tests_passed and integration_passed:
        print("STEP 5 PASSED: reward breakdown and safety filter integration works.", flush=True)
    elif pure_tests_passed and integration_attempted:
        if failure_message:
            print(f"[WARN] AirSim integration issue: {failure_message}", flush=True)
        print("STEP 5 PARTIAL: pure safety/reward tests passed but AirSim integration failed.", flush=True)
    else:
        if not failure_message:
            failure_message = "Step 5 validation did not complete."
        print(f"STEP 5 FAILED: {failure_message}", flush=True)


if __name__ == "__main__":
    main()
