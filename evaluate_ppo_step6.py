#!/usr/bin/env python3
import argparse
import importlib.util
import sys
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
    parser = argparse.ArgumentParser(description="Step 6 PPO model evaluation.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to a saved PPO .zip model.")
    parser.add_argument("--steps", type=int, default=100, help="Maximum evaluation steps.")
    parser.add_argument("--episode-max-steps", type=int, default=100, help="Environment episode max steps before truncation.")
    parser.add_argument("--chaser-start-x", type=float, default=None, help="Requested Chaser reset start global X.")
    parser.add_argument("--chaser-start-y", type=float, default=None, help="Requested Chaser reset start global Y.")
    parser.add_argument("--chaser-start-z", type=float, default=-5.0, help="Requested Chaser reset start global Z.")
    parser.add_argument("--target-start-x", type=float, default=None, help="Requested Target reset start X.")
    parser.add_argument("--target-start-y", type=float, default=None, help="Requested Target reset start Y.")
    parser.add_argument("--target-start-z", type=float, default=-5.0, help="Requested Target reset start Z.")
    return parser.parse_args()


def action_to_int(action):
    try:
        return int(action.item())
    except AttributeError:
        return int(action)


def main():
    if not check_required_packages():
        sys.exit(1)

    from stable_baselines3 import PPO

    from airsim_chase_env import ACTION_NAMES, AirSimChaseEnv

    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_DIR / model_path

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", flush=True)
        print("Run training first:", flush=True)
        print("python3 train_ppo_step6.py --timesteps 1000", flush=True)
        sys.exit(1)

    env = None
    steps_completed = 0

    try:
        print("[STEP6] PPO evaluation", flush=True)
        print(f"[INFO] model={model_path}", flush=True)
        print(f"[INFO] max_steps={args.steps}", flush=True)
        print(f"[INFO] episode_max_steps={args.episode_max_steps}", flush=True)
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

        model = PPO.load(str(model_path), device="cpu")
        env = AirSimChaseEnv(
            chaser_start_x=args.chaser_start_x,
            chaser_start_y=args.chaser_start_y,
            chaser_start_z=args.chaser_start_z,
            target_start_x=args.target_start_x,
            target_start_y=args.target_start_y,
            target_start_z=args.target_start_z,
            max_episode_steps=args.episode_max_steps,
        )
        obs, info = env.reset()

        for step_index in range(1, args.steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            action_int = action_to_int(action)
            obs, reward, terminated, truncated, info = env.step(action_int)
            steps_completed += 1

            print(
                f"[EVAL STEP {step_index:03d}] "
                f"action={info['action']}:{info['action_name']} "
                f"safe={info['safety_safe_action']}:{info['safety_safe_action_name']} "
                f"reward={reward:.2f} "
                f"distance={info['distance']:.2f} "
                f"front={info['lidar_front']:.2f} "
                f"overridden={info['safety_overridden']} "
                f"collision={info['collision']} "
                f"caught={info.get('caught', False)} "
                f"reason={info.get('terminated_reason', 'none')}",
                flush=True,
            )

            if terminated or truncated:
                print(f"[INFO] Episode ended at eval step {step_index}.", flush=True)
                break

        if steps_completed <= 0:
            raise RuntimeError("No evaluation steps completed.")

        print("STEP 6 EVAL PASSED: trained PPO model loaded and ran in AirSim.", flush=True)

    except KeyboardInterrupt:
        print("[WARN] Evaluation interrupted by user. Cleanup will run now.", flush=True)
        print("STEP 6 EVAL FAILED: interrupted before completion.", flush=True)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        traceback.print_exc()
        print(f"STEP 6 EVAL FAILED: {exc}", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
