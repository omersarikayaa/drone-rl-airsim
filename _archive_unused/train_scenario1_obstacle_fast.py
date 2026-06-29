#!/usr/bin/env python3
import argparse
import importlib.util
import math
import random
import sys
import traceback
from pathlib import Path

try:
    import gymnasium as gym
except Exception:
    gym = None


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "logs"
DEFAULT_LOAD_MODEL = MODELS_DIR / "ppo_chaser_step6.zip"
DEFAULT_SAVE_MODEL = MODELS_DIR / "ppo_chaser_step6_scenario1_obstacle_fast.zip"

CHASER_BASE_START = (139.36, 0.0, -11.0)
TARGET_BASE_START = (300.0, 80.0, -11.0)
TARGET_BASE_WAYPOINTS = (
    (320.0, 80.0, -11.0),
    (330.0, 130.0, -11.0),
    (345.0, 160.0, -11.0),
)

CURRICULUM_STAGES = (
    {"stage": 1, "chaser_speed": 3.5, "timesteps": 3000},
    {"stage": 2, "chaser_speed": 4.5, "timesteps": 5000},
    {"stage": 3, "chaser_speed": 5.5, "timesteps": 7000},
    {"stage": 4, "chaser_speed": 6.0, "timesteps": 5000},
)

INFO_KEYWORDS = (
    "distance",
    "caught",
    "collision",
    "too_far",
    "done_reason",
    "min_lidar",
    "lidar_front",
    "lidar_front_left",
    "lidar_front_right",
    "lidar_left",
    "lidar_right",
    "obstacle_penalty",
    "safety_override",
    "safety_override_count",
    "override_count_recent",
    "obstacle_bypass",
    "emergency_avoidance",
    "bypass_direction",
    "forward_scale",
    "side_scale",
    "speed_scale",
    "camera_centering_limited",
    "gap_direction",
    "gap_safe",
    "blocked_lateral",
    "forward_detour",
    "climb_avoidance",
    "final_vx",
    "final_vy",
    "final_vz",
    "altitude",
    "altitude_error",
    "altitude_safety_override",
    "min_lidar_min",
    "capture_box",
    "distance_caught",
    "target_waypoint_index",
    "target_waypoint_total",
    "scenario_stage",
    "scenario_reward_bonus",
    "scenario_progress_reward",
    "scenario_camera_center_reward",
    "scenario_catch_reward",
    "scenario_collision_penalty",
    "scenario_lidar_penalty",
    "scenario_front_penalty",
    "scenario_side_penalty",
    "scenario_safety_penalty",
    "scenario_jerk_penalty",
    "scenario_target_like_capture",
    "scenario_jerk",
    "scenario_chaser_start_x",
    "scenario_chaser_start_y",
    "scenario_target_start_x",
    "scenario_target_start_y",
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
        print(f"[ERROR] {package} is not installed.", flush=True)
    print("Use the project venv that already has training dependencies installed.", flush=True)
    return False


def resolve_project_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def stable_baselines_save_path(zip_path):
    return zip_path.with_suffix("") if zip_path.suffix == ".zip" else zip_path


def clamp(value, low, high):
    return max(low, min(high, value))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        if default is None:
            return None
        return float(default)


def parse_args():
    parser = argparse.ArgumentParser(description="Scenario 1 obstacle-fast PPO fine-tune from legacy step6.")
    parser.add_argument("--load-model", default=str(DEFAULT_LOAD_MODEL), help="Existing PPO .zip model to fine-tune.")
    parser.add_argument("--save-model", default=str(DEFAULT_SAVE_MODEL), help="New output PPO .zip model path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-stage", type=int, choices=[stage["stage"] for stage in CURRICULUM_STAGES], default=1)
    parser.add_argument("--end-stage", type=int, choices=[stage["stage"] for stage in CURRICULUM_STAGES], default=4)
    parser.add_argument("--timesteps-scale", type=float, default=1.0, help="Multiplier for each stage timesteps.")
    parser.add_argument("--step-duration", type=float, default=0.25)
    parser.add_argument("--target-speed", type=float, default=2.0)
    parser.add_argument("--max-episode-steps", type=int, default=900)
    parser.add_argument("--too-far-distance", type=float, default=320.0)
    parser.add_argument("--use-fast-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-target-on-catch", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args()


ScenarioWrapperBase = gym.Wrapper if gym is not None else object


class Scenario1RandomizedRewardWrapper(ScenarioWrapperBase):
    def __init__(
        self,
        env,
        rng,
        stage_id,
        chaser_xy_jitter=5.0,
        target_xy_jitter=10.0,
        waypoint_y_jitter=15.0,
    ):
        if gym is not None:
            super().__init__(env)
        else:
            self.env = env
        self.rng = rng
        self.stage_id = int(stage_id)
        self.chaser_xy_jitter = float(chaser_xy_jitter)
        self.target_xy_jitter = float(target_xy_jitter)
        self.waypoint_y_jitter = float(waypoint_y_jitter)
        self.previous_distance = None
        self.previous_velocity = None
        self.last_randomization = {}
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _jitter(self, amount):
        return self.rng.uniform(-amount, amount)

    def _randomize_episode(self):
        chaser_x = CHASER_BASE_START[0] + self._jitter(self.chaser_xy_jitter)
        chaser_y = CHASER_BASE_START[1] + self._jitter(self.chaser_xy_jitter)
        target_x = TARGET_BASE_START[0] + self._jitter(self.target_xy_jitter)
        target_y = TARGET_BASE_START[1] + self._jitter(self.target_xy_jitter)
        waypoints = [
            (x, y + self._jitter(self.waypoint_y_jitter), z)
            for x, y, z in TARGET_BASE_WAYPOINTS
        ]

        self.env.chaser_start_x = float(chaser_x)
        self.env.chaser_start_y = float(chaser_y)
        self.env.chaser_start_z = CHASER_BASE_START[2]
        self.env.target_start_x = float(target_x)
        self.env.target_start_y = float(target_y)
        self.env.target_start_z = TARGET_BASE_START[2]
        self.env.target_altitude = 11.0
        self.env.target_waypoints = [(float(x), float(y), float(z)) for x, y, z in waypoints]
        self.env.target_waypoint_index = 0

        self.last_randomization = {
            "scenario_stage": self.stage_id,
            "scenario_chaser_start_x": float(chaser_x),
            "scenario_chaser_start_y": float(chaser_y),
            "scenario_target_start_x": float(target_x),
            "scenario_target_start_y": float(target_y),
            "scenario_waypoint0_y": float(waypoints[0][1]),
            "scenario_waypoint1_y": float(waypoints[1][1]),
            "scenario_waypoint2_y": float(waypoints[2][1]),
        }

    def reset(self, **kwargs):
        self._randomize_episode()
        obs, info = self.env.reset(**kwargs)
        self.previous_distance = None if info.get("distance") is None else safe_float(info.get("distance"))
        self.previous_velocity = None
        info.update(self._zero_reward_info())
        info.update(self.last_randomization)
        return obs, info

    def close(self):
        return self.env.close()

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()
        return None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        scenario_reward, reward_info = self._scenario_reward(info)
        reward = float(reward) + scenario_reward
        info.update(reward_info)
        info.update(self.last_randomization)
        info["reward"] = float(reward)
        self.previous_distance = safe_float(info.get("distance"), self.previous_distance)
        self.previous_velocity = (
            safe_float(info.get("final_vx")),
            safe_float(info.get("final_vy")),
            safe_float(info.get("final_vz")),
        )
        return obs, reward, terminated, truncated, info

    def _zero_reward_info(self):
        return {
            "scenario_stage": self.stage_id,
            "scenario_reward_bonus": 0.0,
            "scenario_progress_reward": 0.0,
            "scenario_camera_center_reward": 0.0,
            "scenario_catch_reward": 0.0,
            "scenario_collision_penalty": 0.0,
            "scenario_lidar_penalty": 0.0,
            "scenario_front_penalty": 0.0,
            "scenario_side_penalty": 0.0,
            "scenario_safety_penalty": 0.0,
            "scenario_jerk_penalty": 0.0,
            "scenario_target_like_capture": False,
            "scenario_jerk": 0.0,
        }

    def _scenario_reward(self, info):
        distance = safe_float(info.get("distance"), 0.0)
        previous_distance = self.previous_distance
        improvement = 0.0 if previous_distance is None else float(previous_distance) - distance
        progress_reward = clamp(1.5 * improvement, -3.0, 4.0)

        forward = safe_float(info.get("capture_forward"), 0.0)
        lateral = safe_float(info.get("capture_lateral"), 0.0)
        vertical = safe_float(info.get("capture_vertical"), 0.0)
        lateral_window = max(3.0, distance * 0.20)
        vertical_window = max(2.0, distance * 0.12)
        lateral_score = clamp(1.0 - abs(lateral) / lateral_window, 0.0, 1.0)
        vertical_score = clamp(1.0 - abs(vertical) / vertical_window, 0.0, 1.0)
        forward_score = 1.0 if forward > 0.0 else 0.0
        camera_center_reward = 0.45 * forward_score * lateral_score * vertical_score

        caught = bool(info.get("caught", False))
        distance_caught = distance <= 3.0
        catch_reward = 75.0 if caught or distance_caught else 0.0
        collision_penalty = -75.0 if bool(info.get("collision", False)) else 0.0

        min_lidar = safe_float(info.get("min_lidar"), 50.0)
        front = min(
            safe_float(info.get("lidar_front"), 50.0),
            safe_float(info.get("lidar_front_left"), 50.0),
            safe_float(info.get("lidar_front_right"), 50.0),
        )
        left = safe_float(info.get("lidar_left"), 50.0)
        right = safe_float(info.get("lidar_right"), 50.0)

        target_like_capture = (
            distance <= 4.5
            and forward >= -0.2
            and forward <= 4.5
            and abs(lateral) <= 3.5
            and abs(vertical) <= 3.5
        )
        lidar_penalty = -0.8 * max(0.0, 6.0 - min_lidar)
        front_penalty = -0.6 * max(0.0, 8.0 - front)
        side_penalty = -0.3 * (max(0.0, 6.0 - left) + max(0.0, 6.0 - right))
        if target_like_capture:
            lidar_penalty *= 0.1
            front_penalty *= 0.1
            side_penalty *= 0.1

        override_recent = int(info.get("override_count_recent", 0) or 0)
        safety_penalty = -0.05 * max(0, override_recent - 2)
        if bool(info.get("safety_override", False)):
            safety_penalty -= 0.05

        current_velocity = (
            safe_float(info.get("final_vx")),
            safe_float(info.get("final_vy")),
            safe_float(info.get("final_vz")),
        )
        jerk = 0.0
        if self.previous_velocity is not None:
            jerk = math.sqrt(
                (current_velocity[0] - self.previous_velocity[0]) ** 2
                + (current_velocity[1] - self.previous_velocity[1]) ** 2
                + (current_velocity[2] - self.previous_velocity[2]) ** 2
            )
        jerk_penalty = -min(1.5, 0.12 * max(0.0, jerk - 1.5))

        total = (
            progress_reward
            + camera_center_reward
            + catch_reward
            + collision_penalty
            + lidar_penalty
            + front_penalty
            + side_penalty
            + safety_penalty
            + jerk_penalty
        )

        return float(total), {
            "scenario_stage": self.stage_id,
            "scenario_reward_bonus": float(total),
            "scenario_progress_reward": float(progress_reward),
            "scenario_camera_center_reward": float(camera_center_reward),
            "scenario_catch_reward": float(catch_reward),
            "scenario_collision_penalty": float(collision_penalty),
            "scenario_lidar_penalty": float(lidar_penalty),
            "scenario_front_penalty": float(front_penalty),
            "scenario_side_penalty": float(side_penalty),
            "scenario_safety_penalty": float(safety_penalty),
            "scenario_jerk_penalty": float(jerk_penalty),
            "scenario_target_like_capture": bool(target_like_capture),
            "scenario_jerk": float(jerk),
        }


def make_env(args, stage, rng):
    from stable_baselines3.common.monitor import Monitor

    from airsim_chase_env import AirSimChaseEnv

    env = AirSimChaseEnv(
        target_mode="right_escape",
        target_base_speed=args.target_speed,
        target_escape_speed=args.target_speed,
        target_evade_distance=12.0,
        target_danger_distance=5.0,
        chaser_start_x=CHASER_BASE_START[0],
        chaser_start_y=CHASER_BASE_START[1],
        chaser_start_z=CHASER_BASE_START[2],
        target_start_x=TARGET_BASE_START[0],
        target_start_y=TARGET_BASE_START[1],
        target_start_z=TARGET_BASE_START[2],
        target_waypoints=TARGET_BASE_WAYPOINTS,
        target_altitude=11.0,
        max_episode_steps=args.max_episode_steps,
        too_far_distance=args.too_far_distance,
        use_fast_reset=args.use_fast_reset,
        step_duration=args.step_duration,
        chaser_speed=stage["chaser_speed"],
        reward_mode="simple",
        obs_mode="legacy14",
        use_capture_box=True,
        capture_depth=3.0,
        capture_width=3.0,
        capture_height=3.0,
        catch_radius=3.0,
        capture_bonus=100.0,
        drop_target_on_catch=args.drop_target_on_catch,
        min_safe_altitude=4.0,
        max_safe_altitude=15.0,
        hard_max_altitude=20.0,
        enable_altitude_safety=True,
    )

    if getattr(env.observation_space, "shape", None) != (14,):
        raise RuntimeError(f"Expected legacy14 observation shape (14,), got {env.observation_space}")
    if getattr(env.action_space, "n", None) != 6:
        raise RuntimeError(f"Expected Discrete(6) action space, got {env.action_space}")

    env = Scenario1RandomizedRewardWrapper(env, rng, stage_id=stage["stage"])
    monitor_path = LOGS_DIR / f"scenario1_obstacle_fast_stage{stage['stage']}_monitor.csv"
    return Monitor(env, filename=str(monitor_path), info_keywords=INFO_KEYWORDS), monitor_path


def selected_stages(start_stage, end_stage):
    if end_stage < start_stage:
        raise ValueError("--end-stage must be >= --start-stage")
    return [stage for stage in CURRICULUM_STAGES if start_stage <= stage["stage"] <= end_stage]


def main():
    args = parse_args()
    if not check_required_packages():
        sys.exit(1)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    class ScenarioProgressCallback(BaseCallback):
        def __init__(self, stage_id, print_freq=50, verbose=0):
            super().__init__(verbose=verbose)
            self.stage_id = int(stage_id)
            self.print_freq = int(print_freq)

        def _on_step(self):
            infos = self.locals.get("infos", [])
            info = infos[0] if infos else {}
            dones = self.locals.get("dones", [])
            if len(dones) > 0 and bool(dones[0]) and "episode" in info:
                ep = info.get("episode", {})
                print(
                    "[SCENARIO1_EPISODE] "
                    f"stage={self.stage_id} "
                    f"timestep={self.num_timesteps} "
                    f"episode_reward={float(ep.get('r', 0.0)):.2f} "
                    f"episode_length={int(ep.get('l', 0))} "
                    f"final_distance={float(ep.get('distance', info.get('distance', 0.0))):.2f} "
                    f"caught={ep.get('caught', info.get('caught', False))} "
                    f"collision={ep.get('collision', info.get('collision', False))} "
                    f"reason={ep.get('done_reason', info.get('done_reason', 'none'))} "
                    f"safety_override_count={int(ep.get('safety_override_count', info.get('safety_override_count', 0)))} "
                    f"min_lidar_min={float(ep.get('min_lidar_min', info.get('min_lidar_min', 0.0))):.2f}",
                    flush=True,
                )

            if self.num_timesteps % self.print_freq != 0:
                return True

            print(
                "[SCENARIO1_TRAIN] "
                f"stage={self.stage_id} "
                f"timestep={self.num_timesteps} "
                f"distance={float(info.get('distance', 0.0)):.2f} "
                f"action={info.get('action')}:{info.get('action_name', 'UNKNOWN')} "
                f"safe_action={info.get('safety_safe_action')}:{info.get('safety_safe_action_name', 'UNKNOWN')} "
                f"front={float(info.get('lidar_front', 0.0)):.2f} "
                f"min_lidar={float(info.get('min_lidar', 0.0)):.2f} "
                f"left={float(info.get('lidar_left', 0.0)):.2f} "
                f"right={float(info.get('lidar_right', 0.0)):.2f} "
                f"safety_override={info.get('safety_override', False)} "
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
                f"override_recent={int(info.get('override_count_recent', 0) or 0)} "
                f"obstacle_bypass={info.get('obstacle_bypass', False)} "
                f"emergency_avoidance={info.get('emergency_avoidance', False)} "
                f"bypass_direction={info.get('bypass_direction', 'none')} "
                f"forward_scale={float(info.get('forward_scale', 1.0)):.2f} "
                f"side_scale={float(info.get('side_scale', 0.0)):.2f} "
                f"speed_scale={float(info.get('speed_scale', 1.0)):.2f} "
                f"final_vx={float(info.get('final_vx', 0.0)):.2f} "
                f"final_vy={float(info.get('final_vy', 0.0)):.2f} "
                f"final_vz={float(info.get('final_vz', 0.0)):.2f} "
                f"scenario_reward={float(info.get('scenario_reward_bonus', 0.0)):.2f} "
                f"camera_center={float(info.get('scenario_camera_center_reward', 0.0)):.2f} "
                f"jerk={float(info.get('scenario_jerk', 0.0)):.2f} "
                f"caught={info.get('caught', False)} "
                f"done_reason={info.get('done_reason', 'none')}",
                flush=True,
            )
            return True

    load_model = resolve_project_path(args.load_model)
    save_model = resolve_project_path(args.save_model)
    if save_model.suffix != ".zip":
        save_model = save_model.with_suffix(".zip")
    if not load_model.exists():
        print(f"[ERROR] Load model not found: {load_model}", flush=True)
        sys.exit(1)
    if save_model.resolve() == DEFAULT_LOAD_MODEL.resolve() or save_model.resolve() == load_model.resolve():
        raise RuntimeError("Refusing to overwrite the source step6 model. Use a different --save-model path.")

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    tensorboard_dir = LOGS_DIR / "tensorboard_scenario1_obstacle_fast"
    checkpoint_dir = MODELS_DIR / "checkpoints_scenario1_obstacle_fast"
    checkpoint_dir.mkdir(exist_ok=True)

    stages = selected_stages(args.start_stage, args.end_stage)
    current_load_model = load_model
    rng = random.Random(args.seed)

    print("[SCENARIO1_FINE_TUNE] PPO Chaser scenario-specific obstacle-fast training", flush=True)
    print(f"[INFO] source_model={load_model}", flush=True)
    print(f"[INFO] output_model={save_model}", flush=True)
    print("[INFO] compatibility preserved: obs_mode=legacy14, observation shape=14, action_space=Discrete(6)", flush=True)
    print(f"[INFO] scenario starts: chaser={CHASER_BASE_START}, target={TARGET_BASE_START}", flush=True)
    print(f"[INFO] target_waypoints={TARGET_BASE_WAYPOINTS}", flush=True)

    for stage in stages:
        timesteps = max(1, int(round(stage["timesteps"] * args.timesteps_scale)))
        env = None
        try:
            print(
                "[SCENARIO1_STAGE] "
                f"stage={stage['stage']} "
                f"chaser_speed={stage['chaser_speed']:.1f} "
                f"timesteps={timesteps} "
                f"load={current_load_model}",
                flush=True,
            )
            env, monitor_path = make_env(args, stage, rng)

            if args.check_env:
                from stable_baselines3.common.env_checker import check_env

                print("[INFO] Running stable_baselines3 check_env...", flush=True)
                check_env(env, warn=True)

            checkpoint_callback = CheckpointCallback(
                save_freq=2500,
                save_path=str(checkpoint_dir),
                name_prefix=f"scenario1_stage{stage['stage']}",
            )
            progress_callback = ScenarioProgressCallback(stage_id=stage["stage"], print_freq=50)

            model = PPO.load(str(current_load_model), env=env, device="cpu")
            model.verbose = 1
            model.tensorboard_log = str(tensorboard_dir)
            model.learn(
                total_timesteps=timesteps,
                callback=[checkpoint_callback, progress_callback],
                reset_num_timesteps=False,
                tb_log_name=f"scenario1_stage{stage['stage']}",
            )
            model.save(str(stable_baselines_save_path(save_model)))
            if not save_model.exists():
                raise RuntimeError(f"Model zip was not found after save: {save_model}")
            current_load_model = save_model
            print(f"[OK] Stage {stage['stage']} saved model: {save_model}", flush=True)
            print(f"[OK] Stage {stage['stage']} Monitor log: {monitor_path}", flush=True)
        finally:
            if env is not None:
                env.close()

    print(f"[OK] Scenario 1 fine-tune complete: {save_model}", flush=True)
    print(f"[OK] TensorBoard log dir: {tensorboard_dir}", flush=True)
    print("[OK] Original models/ppo_chaser_step6.zip was not overwritten.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[WARN] Scenario 1 fine-tune interrupted by user.", flush=True)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
