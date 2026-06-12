#!/usr/bin/env python3
import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "ppo_chaser_step6.zip"

REQUIRED_PACKAGES = (
    ("stable_baselines3", "stable-baselines3"),
    ("gymnasium", "gymnasium"),
    ("numpy", "numpy"),
    ("torch", "torch"),
)


def check_required_packages():
    missing = [pip_name for module_name, pip_name in REQUIRED_PACKAGES if importlib.util.find_spec(module_name) is None]
    if not missing:
        return True

    for package in missing:
        if package == "stable-baselines3":
            print("[ERROR] stable-baselines3 is not installed.", flush=True)
        elif package == "gymnasium":
            print("[ERROR] gymnasium is not installed.", flush=True)
        else:
            print(f"[ERROR] {package} is not installed.", flush=True)

    print("Install with:", flush=True)
    print("python3 -m pip install stable-baselines3 gymnasium", flush=True)
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Run trained PPO Chaser agent in AirSim demo mode.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to trained PPO .zip model.")
    parser.add_argument("--steps", type=int, default=100, help="Maximum demo steps.")
    parser.add_argument("--episode-max-steps", type=int, default=100, help="Environment episode max steps before truncation.")
    parser.add_argument("--target-mode", choices=("simple", "evasive"), default="simple", help="Target behavior mode.")
    parser.add_argument("--target-base-speed", type=float, default=1.2, help="Evasive Target base speed.")
    parser.add_argument("--target-escape-speed", type=float, default=1.5, help="Evasive Target maximum escape speed.")
    parser.add_argument("--target-evade-distance", type=float, default=8.0, help="Distance where Target starts evading.")
    parser.add_argument("--target-danger-distance", type=float, default=4.0, help="Distance where stronger lateral evasion is added.")
    parser.add_argument("--chaser-start-x", type=float, default=None, help="Requested Chaser reset start global X.")
    parser.add_argument("--chaser-start-y", type=float, default=None, help="Requested Chaser reset start global Y.")
    parser.add_argument("--chaser-start-z", type=float, default=-5.0, help="Requested Chaser reset start global Z.")
    parser.add_argument("--target-start-x", type=float, default=None, help="Requested Target reset start X.")
    parser.add_argument("--target-start-y", type=float, default=None, help="Requested Target reset start Y.")
    parser.add_argument("--target-start-z", type=float, default=-5.0, help="Requested Target reset start Z.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True, help="Use deterministic PPO actions.")
    parser.add_argument("--delay", type=float, default=0.0, help="Optional sleep after each step.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Print per-step logs.")
    return parser.parse_args()


def action_to_int(action):
    try:
        return int(action.item())
    except AttributeError:
        return int(action)


def resolve_model_path(model_arg):
    model_path = Path(model_arg)
    if not model_path.is_absolute():
        model_path = PROJECT_DIR / model_path
    return model_path


def print_reset_info(obs, info):
    print(
        f"[RESET] obs_shape={obs.shape} "
        f"distance={info['distance']:.2f} "
        f"lidar_available={info['lidar_available']} "
        f"lidar_front={info['lidar_front']:.2f}",
        flush=True,
    )
    print(f"[RESET] chaser_pos={info['chaser_pos']}", flush=True)
    print(f"[RESET] target_pos={info['target_pos']}", flush=True)
    print(f"[RESET] requested_chaser_start={info.get('requested_chaser_start')}", flush=True)
    print(f"[RESET] actual_chaser_start_pos={info.get('actual_chaser_start_pos')}", flush=True)
    print(f"[RESET] requested_target_start={info.get('requested_target_start')}", flush=True)
    print(f"[RESET] actual_target_start_pos={info.get('actual_target_start_pos')}", flush=True)


def print_step(step_index, reward, info):
    safety_reason = info.get("safety_reason", "")
    if not safety_reason:
        safety_reason = "none"
    print(
        f"[RUN STEP {step_index:03d}] "
        f"action={info['action']}:{info['action_name']} "
        f"safe={info['safety_safe_action']}:{info['safety_safe_action_name']} "
        f"reward={reward:.2f} "
        f"distance={info['distance']:.2f} "
        f"dx={info['dx']:.2f} "
        f"dy={info['dy']:.2f} "
        f"dz={info['dz']:.2f} "
        f"front={info['lidar_front']:.2f} "
        f"left={info['lidar_left']:.2f} "
        f"right={info['lidar_right']:.2f} "
        f"target_mode={info.get('target_mode', 'simple')} "
        f"target_vx={float(info.get('target_vx', 0.0)):.2f} "
        f"target_vy={float(info.get('target_vy', 0.0)):.2f} "
        f"bypass_active={info.get('bypass_active', False)} "
        f"bypass_action={info.get('bypass_action_name', 'none')} "
        f"bypass_steps_remaining={info.get('bypass_steps_remaining', 0)} "
        f"bypass_trigger_distance={float(info.get('bypass_trigger_distance', 0.0)):.2f} "
        f"emergency_avoid={info.get('emergency_avoid', False)} "
        f"overridden={info['safety_overridden']} "
        f"collision={info['collision']} "
        f"caught={info['caught']} "
        f"reason={info['terminated_reason']} "
        f"bypass_reason=\"{info.get('bypass_reason', '')}\" "
        f"safety_reason=\"{safety_reason}\"",
        flush=True,
    )


def print_result(step_count, total_reward, final_info, max_steps_reached=False):
    if final_info is None:
        print("[RESULT] No environment step completed.", flush=True)
        return

    if max_steps_reached:
        print("[RESULT] Max run steps reached.", flush=True)
    else:
        print(f"[RESULT] Episode ended at step {step_count}.", flush=True)

    print(f"[RESULT] reason={final_info['terminated_reason']}", flush=True)
    print(f"[RESULT] final_distance={final_info['distance']:.2f}", flush=True)
    print(f"[RESULT] total_reward={total_reward:.2f}", flush=True)
    print(f"[RESULT] caught={final_info['caught']}", flush=True)
    print(f"[RESULT] collision={final_info['collision']}", flush=True)


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)

    print("[STEP7] Run trained PPO Chaser agent", flush=True)
    print("[INFO] This script does not train PPO. It only loads and runs a trained model.", flush=True)
    print(f"[INFO] model={model_path}", flush=True)
    print(f"[INFO] max_steps={args.steps}", flush=True)
    print(f"[INFO] episode_max_steps={args.episode_max_steps}", flush=True)
    print(f"[INFO] target_mode={args.target_mode}", flush=True)
    print(
        f"[INFO] target_base_speed={args.target_base_speed:.2f} "
        f"target_escape_speed={args.target_escape_speed:.2f}",
        flush=True,
    )
    if args.chaser_start_x is None:
        print("[INFO] chaser_start_x=None, using settings/default spawn", flush=True)
    else:
        print(f"[INFO] chaser_start_x={args.chaser_start_x:.2f}", flush=True)
    if args.chaser_start_y is None:
        print("[INFO] chaser_start_y=None, using settings/default spawn", flush=True)
    else:
        print(f"[INFO] chaser_start_y={args.chaser_start_y:.2f}", flush=True)
    print(f"[INFO] chaser_start_z={args.chaser_start_z:.2f}", flush=True)
    if args.target_start_x is None:
        print("[INFO] target_start_x=None, using settings/default spawn", flush=True)
    else:
        print(f"[INFO] target_start_x={args.target_start_x:.2f}", flush=True)
    if args.target_start_y is None:
        print("[INFO] target_start_y=None, using settings/default spawn", flush=True)
    else:
        print(f"[INFO] target_start_y={args.target_start_y:.2f}", flush=True)
    print(f"[INFO] target_start_z={args.target_start_z:.2f}", flush=True)
    print(f"[INFO] deterministic={args.deterministic}", flush=True)

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", flush=True)
        print("Run training first:", flush=True)
        print("python train_ppo_step6.py --timesteps 1000", flush=True)
        sys.exit(1)

    if not check_required_packages():
        sys.exit(1)

    from stable_baselines3 import PPO

    from airsim_chase_env import AirSimChaseEnv

    env = None
    total_reward = 0.0
    steps_completed = 0
    final_info = None
    caught = False

    try:
        model = PPO.load(str(model_path), device="cpu")
        env = AirSimChaseEnv(
            target_mode=args.target_mode,
            target_base_speed=args.target_base_speed,
            target_escape_speed=args.target_escape_speed,
            target_evade_distance=args.target_evade_distance,
            target_danger_distance=args.target_danger_distance,
            chaser_start_x=args.chaser_start_x,
            chaser_start_y=args.chaser_start_y,
            chaser_start_z=args.chaser_start_z,
            target_start_x=args.target_start_x,
            target_start_y=args.target_start_y,
            target_start_z=args.target_start_z,
            max_episode_steps=args.episode_max_steps,
        )
        obs, info = env.reset()
        print_reset_info(obs, info)

        for step_index in range(1, args.steps + 1):
            action, _states = model.predict(obs, deterministic=args.deterministic)
            action_int = action_to_int(action)
            obs, reward, terminated, truncated, info = env.step(action_int)
            total_reward += float(reward)
            steps_completed += 1
            final_info = info
            caught = bool(info.get("caught", False))

            if args.verbose:
                print_step(step_index, reward, info)

            if terminated or truncated:
                print_result(steps_completed, total_reward, final_info)
                break

            if args.delay > 0.0:
                time.sleep(args.delay)
        else:
            if final_info is not None:
                final_info["terminated_reason"] = "max_steps"
            print_result(steps_completed, total_reward, final_info, max_steps_reached=True)

        if steps_completed <= 0:
            raise RuntimeError("No demo steps completed.")

        if caught:
            print("STEP 7 SUCCESS: Chaser caught Target.", flush=True)
        else:
            print("STEP 7 PASSED: trained PPO agent ran, but target was not caught in this run.", flush=True)
        print("STEP 7 PASSED: trained PPO agent ran in AirSim demo mode.", flush=True)

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.", flush=True)
        print("[INFO] Cleaning up AirSim vehicles...", flush=True)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        traceback.print_exc()
        print(f"STEP 7 FAILED: {exc}", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
