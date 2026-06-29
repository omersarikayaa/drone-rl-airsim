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
        print(f"[ERROR] {package} is not installed.", flush=True)
    print("Install with:", flush=True)
    print("python3 -m pip install stable-baselines3 gymnasium", flush=True)
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Step 6 PPO model evaluation.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to a saved PPO .zip model.")
    parser.add_argument("--steps", type=int, default=100, help="Maximum evaluation steps.")
    parser.add_argument("--episode-max-steps", type=int, default=100, help="Environment episode max steps before truncation.")
    parser.add_argument("--obs-mode", choices=("legacy14", "extended26"), default="legacy14", help="Observation mode. Use legacy14 for old ppo_chaser_step6.zip models.")
    parser.add_argument("--use-capture-box", action=argparse.BooleanOptionalAction, default=True, help="Enable capture box termination around Target.")
    parser.add_argument("--capture-depth", type=float, default=3.5, help="Forward capture box depth in meters.")
    parser.add_argument("--capture-width", type=float, default=2.5, help="Capture box width in meters.")
    parser.add_argument("--capture-height", type=float, default=3.0, help="Capture box height in meters.")
    parser.add_argument("--catch-radius", type=float, default=3.0, help="Distance-based catch radius in meters.")
    parser.add_argument("--drop-target-on-catch", action="store_true", help="Disarm/drop Target when caught.")
    parser.add_argument("--disable-altitude-safety", action="store_true", help="Disable altitude clamp/too_high termination during evaluation.")
    parser.add_argument("--target-mode", choices=("simple", "evasive"), default="simple", help="Target behavior mode.")
    parser.add_argument("--target-base-speed", type=float, default=1.2)
    parser.add_argument("--target-escape-speed", type=float, default=1.5)
    parser.add_argument("--target-evade-distance", type=float, default=8.0)
    parser.add_argument("--target-danger-distance", type=float, default=4.0)
    parser.add_argument("--chaser-start-x", type=float, default=None)
    parser.add_argument("--chaser-start-y", type=float, default=None)
    parser.add_argument("--chaser-start-z", type=float, default=None)
    parser.add_argument("--target-start-x", type=float, default=None)
    parser.add_argument("--target-start-y", type=float, default=None)
    parser.add_argument("--target-start-z", type=float, default=None)
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
        sys.exit(1)

    env = None
    steps_completed = 0

    try:
        print("[STEP6] PPO evaluation", flush=True)
        print(f"[INFO] model={model_path}", flush=True)
        print(f"[INFO] max_steps={args.steps}", flush=True)
        print(f"[INFO] episode_max_steps={args.episode_max_steps}", flush=True)
        print(f"[INFO] obs_mode={args.obs_mode}", flush=True)
        print(
            f"[INFO] capture_box={args.use_capture_box} "
            f"depth={args.capture_depth:.2f} width={args.capture_width:.2f} "
            f"height={args.capture_height:.2f} catch_radius={args.catch_radius:.2f} "
            f"drop_target_on_catch={args.drop_target_on_catch}",
            flush=True,
        )
        print(f"[INFO] target_mode={args.target_mode}", flush=True)
        print(f"[INFO] altitude_safety={not args.disable_altitude_safety}", flush=True)

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
            obs_mode=args.obs_mode,
            use_capture_box=args.use_capture_box,
            capture_depth=args.capture_depth,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            catch_radius=args.catch_radius,
            drop_target_on_catch=args.drop_target_on_catch,
            enable_altitude_safety=not args.disable_altitude_safety,
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
                f"left={info['lidar_left']:.2f} "
                f"right={info['lidar_right']:.2f} "
                f"min_lidar={float(info.get('min_lidar', 0.0)):.2f} "
                f"obstacle_bypass={info.get('obstacle_bypass', False)} "
                f"emergency_avoidance={info.get('emergency_avoidance', False)} "
                f"bypass_direction={info.get('bypass_direction', 'none')} "
                f"camera_centering_limited={info.get('camera_centering_limited', False)} "
                f"close_chase_mode={info.get('close_chase_mode', False)} "
                f"target_side={info.get('target_side', 'center')} "
                f"front_clear={info.get('front_clear', True)} "
                f"left_blocked={info.get('left_blocked', False)} "
                f"right_blocked={info.get('right_blocked', False)} "
                f"planner_choice={info.get('planner_choice', 'forward')} "
                f"chosen_reason={info.get('chosen_reason', 'target_progress')} "
                f"obstacle_reaction_zone={info.get('obstacle_reaction_zone', 'none')} "
                f"gap_direction={info.get('gap_direction', 'center')} "
                f"gap_safe={info.get('gap_safe', True)} "
                f"blocked_lateral={info.get('blocked_lateral', False)} "
                f"forward_detour={info.get('forward_detour', False)} "
                f"climb_avoidance={info.get('climb_avoidance', False)} "
                f"stuck_recovery={info.get('stuck_recovery', False)} "
                f"forward_scale={float(info.get('forward_scale', 1.0)):.2f} "
                f"side_scale={float(info.get('side_scale', 0.0)):.2f} "
                f"final_vx={float(info.get('final_vx', info.get('vx', 0.0))):.2f} "
                f"final_vy={float(info.get('final_vy', info.get('vy', 0.0))):.2f} "
                f"chaser_z={float(info.get('chaser_z', 0.0)):.2f} "
                f"target_altitude={float(info.get('target_altitude', 0.0)):.2f} "
                f"altitude={float(info.get('altitude', 0.0)):.2f} "
                f"altitude_error={float(info.get('altitude_error', 0.0)):.2f} "
                f"vz_alt_hold={float(info.get('vz_alt_hold', 0.0)):.2f} "
                f"final_vz={float(info.get('final_vz', info.get('vz', 0.0))):.2f} "
                f"altitude_override={info.get('altitude_safety_override', False)} "
                f"too_high={info.get('too_high', False)} "
                f"overridden={info['safety_overridden']} "
                f"capture_box={info.get('capture_box', False)} "
                f"distance_caught={info.get('distance_caught', False)} "
                f"collision={info['collision']} "
                f"caught={info.get('caught', False)} "
                f"done_reason={info.get('done_reason', info.get('terminated_reason', 'none'))} "
                f"safety_bypassed_for_capture={info.get('safety_bypassed_for_capture', False)} "
                f"override_count_recent={info.get('override_count_recent', 0)} "
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
        print("[WARN] Evaluation interrupted by user.", flush=True)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
