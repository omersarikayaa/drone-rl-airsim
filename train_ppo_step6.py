#!/usr/bin/env python3
import argparse
import importlib.util
import sys
import traceback
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "logs"

INFO_KEYWORDS = (
    "distance",
    "caught",
    "collision",
    "too_far",
    "min_lidar",
    "safety_override",
    "safety_override_count",
    "min_lidar_mean",
    "min_lidar_min",
    "terminated_reason",
)

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
    parser = argparse.ArgumentParser(description="PPO Chaser training for AirSimChaseEnv.")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total PPO timesteps.")
    parser.add_argument("--model-name", default="ppo_chaser_step7_ext26", help="Model name saved under models/.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for PPO.")
    parser.add_argument("--episode-max-steps", type=int, default=200, help="Environment episode max steps before truncation.")
    parser.add_argument("--step-duration", type=float, default=0.3, help="AirSim moveByVelocityAsync duration per env step.")
    parser.add_argument("--chaser-speed", type=float, default=2.0, help="Chaser speed in m/s for discrete actions.")
    parser.add_argument("--use-fast-reset", action=argparse.BooleanOptionalAction, default=True, help="Use pose-based fast reset after the first full reset.")
    parser.add_argument("--reward-mode", choices=("simple", "legacy"), default="simple", help="Reward function mode.")
    parser.add_argument("--obs-mode", choices=("legacy14", "extended26"), default="extended26", help="Observation mode for training.")
    parser.add_argument("--check-env", action="store_true", help="Run Stable-Baselines3 check_env before training.")
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
    parser.add_argument("--resume-from", default=None, help="Optional PPO .zip model to resume from.")
    return parser.parse_args()


def model_paths(model_name):
    model_name = Path(model_name).name
    if model_name.endswith(".zip"):
        zip_path = MODELS_DIR / model_name
        save_path = zip_path.with_suffix("")
    else:
        save_path = MODELS_DIR / model_name
        zip_path = MODELS_DIR / f"{model_name}.zip"
    return save_path, zip_path


def main():
    if not check_required_packages():
        sys.exit(1)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    from airsim_chase_env import AirSimChaseEnv

    class TrainingProgressCallback(BaseCallback):
        def __init__(self, print_freq=100, verbose=0):
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
                    f"safety_override_count={int(episode_info.get('safety_override_count', info.get('safety_override_count', 0)))} "
                    f"min_lidar_min={float(episode_info.get('min_lidar_min', info.get('min_lidar_min', 0.0))):.2f} "
                    f"min_lidar_mean={float(episode_info.get('min_lidar_mean', info.get('min_lidar_mean', 0.0))):.2f} "
                    f"reason={episode_info.get('terminated_reason', info.get('terminated_reason', 'none'))}",
                    flush=True,
                )

            if self.num_timesteps % self.print_freq != 0:
                return True

            rewards = self.locals.get("rewards", [])
            try:
                reward = float(rewards[0])
            except Exception:
                reward = float(info.get("reward", 0.0))
            print(
                "[TRAIN] "
                f"timestep={self.num_timesteps} "
                f"reward={reward:.2f} "
                f"distance={float(info.get('distance', 0.0)):.2f} "
                f"min_lidar={float(info.get('min_lidar', 0.0)):.2f} "
                f"safety_override={info.get('safety_override', False)} "
                f"lidar_front={float(info.get('lidar_front', 0.0)):.2f} "
                f"collision={info.get('collision', False)} "
                f"terminated={info.get('terminated', False)}",
                flush=True,
            )
            return True

    args = parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    checkpoints_dir = MODELS_DIR / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    save_path, zip_path = model_paths(args.model_name)
    monitor_path = LOGS_DIR / "step6_monitor.csv"

    env = None
    model_saved = False

    try:
        print("[STEP6] PPO Chaser training", flush=True)
        print("[INFO] PPO Chaser training with scripted/evasive Target.", flush=True)
        print("[INFO] AirSim training can be slow because the simulator runs in real time.", flush=True)
        print(f"[INFO] timesteps={args.timesteps} seed={args.seed}", flush=True)
        print(f"[INFO] episode_max_steps={args.episode_max_steps}", flush=True)
        print(f"[INFO] step_duration={args.step_duration:.2f} chaser_speed={args.chaser_speed:.2f}", flush=True)
        print(f"[INFO] fast_reset={args.use_fast_reset} reward_mode={args.reward_mode} obs_mode={args.obs_mode}", flush=True)
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
        if args.resume_from:
            print(f"[INFO] resume_from={args.resume_from}", flush=True)

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
            use_fast_reset=args.use_fast_reset,
            step_duration=args.step_duration,
            chaser_speed=args.chaser_speed,
            reward_mode=args.reward_mode,
            obs_mode=args.obs_mode,
        )
        if args.check_env:
            from stable_baselines3.common.env_checker import check_env

            print("[INFO] Running stable_baselines3 check_env...", flush=True)
            check_env(env, warn=True)

        env = Monitor(env, filename=str(monitor_path), info_keywords=INFO_KEYWORDS)

        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=str(checkpoints_dir),
            name_prefix=f"{Path(args.model_name).stem}_checkpoint",
        )
        progress_callback = TrainingProgressCallback(print_freq=100)
        callbacks = CallbackList([checkpoint_callback, progress_callback])

        if args.resume_from:
            resume_path = Path(args.resume_from)
            if not resume_path.is_absolute():
                resume_path = PROJECT_DIR / resume_path
            if not resume_path.exists():
                raise RuntimeError(f"Resume model not found: {resume_path}")
            model = PPO.load(str(resume_path), env=env, device="cpu")
            model.verbose = 1
        else:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=str(LOGS_DIR / "tensorboard"),
                seed=args.seed,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                clip_range=0.2,
                device="cpu",
            )

        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=not bool(args.resume_from),
        )
        model.save(str(save_path))
        model_saved = zip_path.exists()

        print(f"[OK] Model saved: {zip_path}", flush=True)
        print(f"[OK] Monitor log: {monitor_path}", flush=True)
        print(f"[OK] TensorBoard log dir: {LOGS_DIR / 'tensorboard'}", flush=True)
        print(f"[CHECK] Model exists: {model_saved}", flush=True)
        print(f"[CHECK] Logs dir exists: {LOGS_DIR.exists()}", flush=True)
        monitor_files = list(LOGS_DIR.glob("*monitor*")) + list(LOGS_DIR.glob("*.csv"))
        print(f"[CHECK] Monitor/log files found: {len(monitor_files) > 0}", flush=True)

        if not model_saved:
            raise RuntimeError(f"Model zip was not found after save: {zip_path}")

        print("STEP 6 PASSED: PPO training ran and model was saved.", flush=True)

    except KeyboardInterrupt:
        print("[WARN] Training interrupted by user. Cleanup will run now.", flush=True)
        print("STEP 6 FAILED: training interrupted before completion.", flush=True)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        traceback.print_exc()
        print(f"STEP 6 FAILED: {exc}", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
