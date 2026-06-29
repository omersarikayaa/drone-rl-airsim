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
DEFAULT_SAVE_MODEL = MODELS_DIR / "ppo_chaser_step6_today_demo.zip"

INFO_KEYWORDS = (
    "distance",
    "caught",
    "collision",
    "too_far",
    "done_reason",
    "min_lidar",
    "lidar_front",
    "lidar_left",
    "lidar_right",
    "obstacle_penalty",
    "safety_override",
    "safety_override_count",
    "obstacle_bypass",
    "emergency_avoidance",
    "bypass_direction",
    "forward_scale",
    "side_scale",
    "final_vx",
    "final_vy",
    "final_vz",
    "altitude",
    "altitude_error",
    "altitude_safety_override",
    "min_lidar_min",
)

CURRICULUM_LEVELS = {
    1: {
        "label": "A",
        "default_timesteps": 10000,
        "min_start_distance": 30.0,
        "max_start_distance": 60.0,
        "chaser_speed": 3.5,
        "target_base_speed": 1.0,
        "target_escape_speed": 1.6,
        "target_evade_distance": 12.0,
        "target_danger_distance": 5.0,
        "episode_max_steps": 350,
        "too_far_distance": 100.0,
    },
    2: {
        "label": "B",
        "default_timesteps": 15000,
        "min_start_distance": 70.0,
        "max_start_distance": 120.0,
        "chaser_speed": 4.5,
        "target_base_speed": 1.4,
        "target_escape_speed": 2.3,
        "target_evade_distance": 16.0,
        "target_danger_distance": 6.0,
        "episode_max_steps": 550,
        "too_far_distance": 180.0,
    },
    3: {
        "label": "C",
        "default_timesteps": 15000,
        "min_start_distance": 100.0,
        "max_start_distance": 150.0,
        "chaser_speed": 5.0,
        "target_base_speed": 1.6,
        "target_escape_speed": 2.7,
        "target_evade_distance": 18.0,
        "target_danger_distance": 7.0,
        "episode_max_steps": 700,
        "too_far_distance": 240.0,
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
    print("Install with the project venv you already use, for example:", flush=True)
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
    parser = argparse.ArgumentParser(description="Today-demo PPO fine-tune from the legacy step6 model.")
    parser.add_argument("--load-model", default=str(DEFAULT_LOAD_MODEL))
    parser.add_argument("--save-model", default=str(DEFAULT_SAVE_MODEL))
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--curriculum-level", type=int, choices=sorted(CURRICULUM_LEVELS), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", choices=("simple", "evasive"), default="evasive")
    parser.add_argument("--step-duration", type=float, default=0.3)
    parser.add_argument("--drop-target-on-catch", action="store_true")
    parser.add_argument("--use-fast-reset", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not check_required_packages():
        sys.exit(1)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    from airsim_chase_env import AirSimChaseEnv

    class DemoProgressCallback(BaseCallback):
        def __init__(self, print_freq=50, verbose=0):
            super().__init__(verbose=verbose)
            self.print_freq = int(print_freq)

        def _on_step(self):
            infos = self.locals.get("infos", [])
            info = infos[0] if infos else {}
            dones = self.locals.get("dones", [])
            if len(dones) > 0 and bool(dones[0]) and "episode" in info:
                ep = info.get("episode", {})
                print(
                    "[TODAY_EPISODE] "
                    f"timestep={self.num_timesteps} "
                    f"episode_reward={float(ep.get('r', 0.0)):.2f} "
                    f"episode_length={int(ep.get('l', 0))} "
                    f"final_distance={float(ep.get('distance', info.get('distance', 0.0))):.2f} "
                    f"caught={ep.get('caught', info.get('caught', False))} "
                    f"collision={ep.get('collision', info.get('collision', False))} "
                    f"done_reason={ep.get('done_reason', info.get('done_reason', 'none'))} "
                    f"safety_override_count={int(ep.get('safety_override_count', info.get('safety_override_count', 0)))}",
                    flush=True,
                )

            if self.num_timesteps % self.print_freq != 0:
                return True

            print(
                "[TODAY_TRAIN] "
                f"timestep={self.num_timesteps} "
                f"distance={float(info.get('distance', 0.0)):.2f} "
                f"action={info.get('action')}:{info.get('action_name', 'UNKNOWN')} "
                f"safe_action={info.get('safety_safe_action')}:{info.get('safety_safe_action_name', 'UNKNOWN')} "
                f"min_lidar={float(info.get('min_lidar', 0.0)):.2f} "
                f"front={float(info.get('lidar_front', 0.0)):.2f} "
                f"left={float(info.get('lidar_left', 0.0)):.2f} "
                f"right={float(info.get('lidar_right', 0.0)):.2f} "
                f"safety_override={info.get('safety_override', False)} "
                f"obstacle_bypass={info.get('obstacle_bypass', False)} "
                f"emergency_avoidance={info.get('emergency_avoidance', False)} "
                f"bypass_direction={info.get('bypass_direction', 'none')} "
                f"forward_scale={float(info.get('forward_scale', 1.0)):.2f} "
                f"side_scale={float(info.get('side_scale', 0.0)):.2f} "
                f"final_vx={float(info.get('final_vx', 0.0)):.2f} "
                f"final_vy={float(info.get('final_vy', 0.0)):.2f} "
                f"final_vz={float(info.get('final_vz', 0.0)):.2f} "
                f"altitude={float(info.get('altitude', 0.0)):.2f} "
                f"collision={info.get('collision', False)} "
                f"caught={info.get('caught', False)} "
                f"done_reason={info.get('done_reason', 'none')}",
                flush=True,
            )
            return True

    cfg = CURRICULUM_LEVELS[args.curriculum_level]
    total_timesteps = int(args.total_timesteps or cfg["default_timesteps"])
    load_model = resolve_project_path(args.load_model)
    save_model = resolve_project_path(args.save_model)
    if save_model.suffix != ".zip":
        save_model = save_model.with_suffix(".zip")

    if not load_model.exists():
        print(f"[ERROR] Load model not found: {load_model}", flush=True)
        sys.exit(1)
    if save_model.resolve() == DEFAULT_LOAD_MODEL.resolve():
        raise RuntimeError("Refusing to overwrite models/ppo_chaser_step6.zip.")

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    checkpoints_dir = MODELS_DIR / "checkpoints_today_demo"
    checkpoints_dir.mkdir(exist_ok=True)
    monitor_path = LOGS_DIR / f"today_demo_level{cfg['label']}_monitor.csv"
    tensorboard_dir = LOGS_DIR / "tensorboard_today_demo"

    env = None
    try:
        print("[TODAY_DEMO_TRAIN] Fine-tuning PPO Chaser for demo", flush=True)
        print(f"[INFO] load_model={load_model}", flush=True)
        print(f"[INFO] save_model={save_model}", flush=True)
        print(f"[INFO] curriculum_level={args.curriculum_level} label={cfg['label']} cfg={cfg}", flush=True)
        print(f"[INFO] total_timesteps={total_timesteps}", flush=True)
        print("[INFO] compatibility obs_mode=legacy14 action_space=Discrete(6)", flush=True)

        env = AirSimChaseEnv(
            target_mode=args.target_mode,
            target_base_speed=cfg["target_base_speed"],
            target_escape_speed=cfg["target_escape_speed"],
            target_evade_distance=cfg["target_evade_distance"],
            target_danger_distance=cfg["target_danger_distance"],
            chaser_start_z=-8.0,
            target_start_z=-8.0,
            min_start_distance=cfg["min_start_distance"],
            max_start_distance=cfg["max_start_distance"],
            random_start_angle=True,
            max_episode_steps=cfg["episode_max_steps"],
            too_far_distance=cfg["too_far_distance"],
            use_fast_reset=args.use_fast_reset,
            step_duration=args.step_duration,
            chaser_speed=cfg["chaser_speed"],
            reward_mode="simple",
            obs_mode="legacy14",
            use_capture_box=True,
            capture_depth=2.0,
            capture_width=2.8,
            capture_height=3.0,
            catch_radius=2.0,
            capture_bonus=100.0,
            drop_target_on_catch=args.drop_target_on_catch,
            target_altitude=8.0,
            min_safe_altitude=4.0,
            max_safe_altitude=15.0,
            hard_max_altitude=20.0,
            enable_altitude_safety=True,
        )
        if getattr(env.observation_space, "shape", None) != (14,):
            raise RuntimeError(f"Expected legacy14 observation shape (14,), got {env.observation_space}")
        if getattr(env.action_space, "n", None) != 6:
            raise RuntimeError(f"Expected Discrete(6), got {env.action_space}")

        env = Monitor(env, filename=str(monitor_path), info_keywords=INFO_KEYWORDS)
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=str(checkpoints_dir),
            name_prefix=f"today_demo_level{cfg['label']}",
        )
        progress_callback = DemoProgressCallback(print_freq=50)

        model = PPO.load(str(load_model), env=env, device="cpu")
        model.verbose = 1
        model.tensorboard_log = str(tensorboard_dir)
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, progress_callback],
            reset_num_timesteps=False,
            tb_log_name=f"today_demo_level{cfg['label']}",
        )
        model.save(str(stable_baselines_save_path(save_model)))

        if not save_model.exists():
            raise RuntimeError(f"Model zip was not found after save: {save_model}")
        print(f"[OK] Today demo model saved: {save_model}", flush=True)
        print(f"[OK] Monitor log: {monitor_path}", flush=True)
        print(f"[OK] TensorBoard log dir: {tensorboard_dir}", flush=True)
        print("[OK] Original models/ppo_chaser_step6.zip was not overwritten.", flush=True)

    except KeyboardInterrupt:
        print("[WARN] Today demo training interrupted by user.", flush=True)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
