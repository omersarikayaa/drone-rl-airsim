#!/usr/bin/env python3
import argparse
import importlib.util
import sys
import traceback
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "logs"
DEFAULT_LOAD_MODEL = MODELS_DIR / "ppo_chaser_step6.zip"
DEFAULT_SAVE_MODEL = MODELS_DIR / "ppo_chaser_step6_continued_longrange_obstacle_altitude.zip"

INFO_KEYWORDS = (
    "distance",
    "caught",
    "collision",
    "too_far",
    "done_reason",
    "min_lidar",
    "obstacle_penalty",
    "safety_override",
    "safety_override_count",
    "altitude",
    "altitude_error",
    "altitude_safety_override",
    "target_altitude",
    "vz_alt_hold",
    "final_vz",
    "too_high",
    "chaser_z",
    "vx",
    "vy",
    "vz",
    "obstacle_bypass",
    "emergency_avoidance",
    "bypass_direction",
    "forward_scale",
    "side_scale",
    "final_vx",
    "final_vy",
    "final_vz",
    "min_lidar_mean",
    "min_lidar_min",
)

CURRICULUM_LEVELS = {
    1: {
        "min_start_distance": 20.0,
        "max_start_distance": 50.0,
        "chaser_speed": 3.0,
        "target_base_speed": 1.2,
        "target_escape_speed": 2.0,
        "episode_max_steps": 300,
        "too_far_distance": 80.0,
    },
    2: {
        "min_start_distance": 50.0,
        "max_start_distance": 100.0,
        "chaser_speed": 4.0,
        "target_base_speed": 1.8,
        "target_escape_speed": 2.7,
        "episode_max_steps": 500,
        "too_far_distance": 140.0,
    },
    3: {
        "min_start_distance": 100.0,
        "max_start_distance": 200.0,
        "chaser_speed": 5.0,
        "target_base_speed": 2.2,
        "target_escape_speed": 3.5,
        "episode_max_steps": 800,
        "too_far_distance": 260.0,
    },
}

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


def resolve_project_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def stable_baselines_save_path(zip_path):
    return zip_path.with_suffix("") if zip_path.suffix == ".zip" else zip_path


def parse_args():
    parser = argparse.ArgumentParser(description="Continue PPO step6 training with legacy14 observations.")
    parser.add_argument("--load-model", default=str(DEFAULT_LOAD_MODEL), help="Existing PPO .zip model to continue from.")
    parser.add_argument("--save-model", default=str(DEFAULT_SAVE_MODEL), help="New output PPO .zip model path.")
    parser.add_argument("--total-timesteps", type=int, default=30000, help="Additional timesteps to train.")
    parser.add_argument("--curriculum-level", type=int, choices=sorted(CURRICULUM_LEVELS), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", choices=("simple", "evasive"), default="evasive")
    parser.add_argument("--step-duration", type=float, default=0.3)
    parser.add_argument("--use-fast-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-start-angle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-altitude", type=float, default=8.0)
    parser.add_argument("--min-safe-altitude", type=float, default=4.0)
    parser.add_argument("--max-safe-altitude", type=float, default=15.0)
    parser.add_argument("--hard-max-altitude", type=float, default=20.0)
    parser.add_argument("--capture-depth", type=float, default=3.5)
    parser.add_argument("--capture-width", type=float, default=2.5)
    parser.add_argument("--capture-height", type=float, default=3.0)
    parser.add_argument("--catch-radius", type=float, default=3.0)
    parser.add_argument("--drop-target-on-catch", action="store_true")
    parser.add_argument("--check-env", action="store_true", help="Optional SB3 environment check.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not check_required_packages():
        sys.exit(1)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    from airsim_chase_env import AirSimChaseEnv

    class ContinueProgressCallback(BaseCallback):
        def __init__(self, print_freq=50, verbose=0):
            super().__init__(verbose=verbose)
            self.print_freq = print_freq

        def _on_step(self):
            infos = self.locals.get("infos", [])
            info = infos[0] if infos else {}
            dones = self.locals.get("dones", [])
            if len(dones) > 0 and bool(dones[0]) and "episode" in info:
                episode_info = info.get("episode", {})
                print(
                    "[EPISODE] "
                    f"timestep={self.num_timesteps} "
                    f"episode_reward={float(episode_info.get('r', 0.0)):.2f} "
                    f"episode_length={int(episode_info.get('l', 0))} "
                    f"final_distance={float(episode_info.get('distance', info.get('distance', 0.0))):.2f} "
                    f"caught={episode_info.get('caught', info.get('caught', False))} "
                    f"collision={episode_info.get('collision', info.get('collision', False))} "
                    f"too_far={episode_info.get('too_far', info.get('too_far', False))} "
                    f"too_high={episode_info.get('too_high', info.get('too_high', False))} "
                    f"safety_override_count={int(episode_info.get('safety_override_count', info.get('safety_override_count', 0)))} "
                    f"min_lidar_min={float(episode_info.get('min_lidar_min', info.get('min_lidar_min', 0.0))):.2f} "
                    f"altitude={float(episode_info.get('altitude', info.get('altitude', 0.0))):.2f} "
                    f"reason={episode_info.get('done_reason', info.get('done_reason', 'none'))}",
                    flush=True,
                )

            if self.num_timesteps % self.print_freq != 0:
                return True

            print(
                "[TRAIN_CONTINUE] "
                f"timestep={self.num_timesteps} "
                f"action={info.get('action')}:{info.get('action_name', 'UNKNOWN')} "
                f"safe_action={info.get('safety_safe_action')}:{info.get('safety_safe_action_name', 'UNKNOWN')} "
                f"reward={float(info.get('reward', 0.0)):.2f} "
                f"distance={float(info.get('distance', 0.0)):.2f} "
                f"min_lidar={float(info.get('min_lidar', 0.0)):.2f} "
                f"front={float(info.get('lidar_front', 0.0)):.2f} "
                f"left={float(info.get('lidar_left', 0.0)):.2f} "
                f"right={float(info.get('lidar_right', 0.0)):.2f} "
                f"obstacle_penalty={float(info.get('obstacle_penalty', 0.0)):.2f} "
                f"safety_override={info.get('safety_override', False)} "
                f"obstacle_bypass={info.get('obstacle_bypass', False)} "
                f"emergency_avoidance={info.get('emergency_avoidance', False)} "
                f"bypass_direction={info.get('bypass_direction', 'none')} "
                f"forward_scale={float(info.get('forward_scale', 1.0)):.2f} "
                f"side_scale={float(info.get('side_scale', 0.0)):.2f} "
                f"vx={float(info.get('vx', 0.0)):.2f} "
                f"vy={float(info.get('vy', 0.0)):.2f} "
                f"vz={float(info.get('vz', 0.0)):.2f} "
                f"final_vx={float(info.get('final_vx', info.get('vx', 0.0))):.2f} "
                f"final_vy={float(info.get('final_vy', info.get('vy', 0.0))):.2f} "
                f"chaser_z={float(info.get('chaser_z', 0.0)):.2f} "
                f"target_altitude={float(info.get('target_altitude', 0.0)):.2f} "
                f"altitude={float(info.get('altitude', 0.0)):.2f} "
                f"altitude_error={float(info.get('altitude_error', 0.0)):.2f} "
                f"vz_alt_hold={float(info.get('vz_alt_hold', 0.0)):.2f} "
                f"final_vz={float(info.get('final_vz', info.get('vz', 0.0))):.2f} "
                f"altitude_override={info.get('altitude_safety_override', False)} "
                f"done_reason={info.get('done_reason', 'none')}",
                flush=True,
            )
            return True

    cfg = CURRICULUM_LEVELS[args.curriculum_level]
    load_model = resolve_project_path(args.load_model)
    save_model = resolve_project_path(args.save_model)
    if save_model.suffix != ".zip":
        save_model = save_model.with_suffix(".zip")
    save_path = stable_baselines_save_path(save_model)

    if not load_model.exists():
        print(f"[ERROR] Load model not found: {load_model}", flush=True)
        sys.exit(1)
    if load_model.resolve() == save_model.resolve():
        raise RuntimeError("Refusing to overwrite the loaded model. Choose a different --save-model path.")

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    checkpoints_dir = MODELS_DIR / "checkpoints_step6_continue"
    checkpoints_dir.mkdir(exist_ok=True)
    monitor_path = LOGS_DIR / f"step6_continue_level{args.curriculum_level}_monitor.csv"
    tensorboard_dir = LOGS_DIR / "tensorboard_step6_continue"

    env = None
    try:
        print("[STEP6_CONTINUE] PPO Chaser legacy14 continuation training", flush=True)
        print(f"[INFO] load_model={load_model}", flush=True)
        print(f"[INFO] save_model={save_model}", flush=True)
        print(f"[INFO] total_timesteps={args.total_timesteps} curriculum_level={args.curriculum_level}", flush=True)
        print(f"[INFO] curriculum={cfg}", flush=True)
        print("[INFO] compatibility obs_mode=legacy14 action_space=Discrete(6)", flush=True)

        env = AirSimChaseEnv(
            target_mode=args.target_mode,
            target_base_speed=cfg["target_base_speed"],
            target_escape_speed=cfg["target_escape_speed"],
            target_evade_distance=12.0,
            target_danger_distance=5.0,
            chaser_start_z=-args.target_altitude,
            target_start_z=-args.target_altitude,
            min_start_distance=cfg["min_start_distance"],
            max_start_distance=cfg["max_start_distance"],
            random_start_angle=args.random_start_angle,
            max_episode_steps=cfg["episode_max_steps"],
            too_far_distance=cfg["too_far_distance"],
            use_fast_reset=args.use_fast_reset,
            step_duration=args.step_duration,
            chaser_speed=cfg["chaser_speed"],
            reward_mode="simple",
            obs_mode="legacy14",
            use_capture_box=True,
            capture_depth=args.capture_depth,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            catch_radius=args.catch_radius,
            capture_bonus=100.0,
            drop_target_on_catch=args.drop_target_on_catch,
            target_altitude=args.target_altitude,
            min_safe_altitude=args.min_safe_altitude,
            max_safe_altitude=args.max_safe_altitude,
            hard_max_altitude=args.hard_max_altitude,
            enable_altitude_safety=True,
        )
        if getattr(env.observation_space, "shape", None) != (14,):
            raise RuntimeError(f"Expected legacy14 observation shape (14,), got {env.observation_space}")
        if getattr(env.action_space, "n", None) != 6:
            raise RuntimeError(f"Expected Discrete(6) action space, got {env.action_space}")

        if args.check_env:
            from stable_baselines3.common.env_checker import check_env

            print("[INFO] Running stable_baselines3 check_env...", flush=True)
            check_env(env, warn=True)

        env = Monitor(env, filename=str(monitor_path), info_keywords=INFO_KEYWORDS)

        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=str(checkpoints_dir),
            name_prefix=f"step6_continue_level{args.curriculum_level}_checkpoint",
        )
        progress_callback = ContinueProgressCallback(print_freq=50)

        model = PPO.load(str(load_model), env=env, device="cpu")
        model.verbose = 1
        model.tensorboard_log = str(tensorboard_dir)

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, progress_callback],
            reset_num_timesteps=False,
            tb_log_name=f"step6_continue_level{args.curriculum_level}",
        )
        model.save(str(save_path))

        if not save_model.exists():
            raise RuntimeError(f"Model zip was not found after save: {save_model}")

        print(f"[OK] Continued model saved: {save_model}", flush=True)
        print(f"[OK] Monitor log: {monitor_path}", flush=True)
        print(f"[OK] TensorBoard log dir: {tensorboard_dir}", flush=True)
        print("[OK] Original step6 model was not overwritten.", flush=True)

    except KeyboardInterrupt:
        print("[WARN] Continuation training interrupted by user.", flush=True)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        traceback.print_exc()
        print(f"STEP 6 CONTINUE FAILED: {exc}", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
