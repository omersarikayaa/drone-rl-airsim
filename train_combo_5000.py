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
DEFAULT_SAVE_MODEL = MODELS_DIR / "ppo_chaser_step6_combo_5000.zip"

CHASER_BASE_START = (139.36, 0.0, -11.0)
TARGET_BASE_START = (300.0, 80.0, -11.0)
TARGET_BASE_WAYPOINTS = (
    (320.0, 80.0, -11.0),
    (330.0, 130.0, -11.0),
    (345.0, 160.0, -11.0),
)

COMBO_STAGES = (
    {"stage": 1, "chaser_speed": 4.5, "timesteps": 2000},
    {"stage": 2, "chaser_speed": 5.5, "timesteps": 3000},
)

REQUIRED_PACKAGES = (
    ("stable_baselines3", "stable-baselines3"),
    ("gymnasium", "gymnasium"),
    ("numpy", "numpy"),
    ("torch", "torch"),
)

INFO_KEYWORDS = (
    "distance",
    "caught",
    "capture_box",
    "distance_caught",
    "collision",
    "too_far",
    "done_reason",
    "lidar_front",
    "lidar_left",
    "lidar_right",
    "min_lidar",
    "safety_override",
    "obstacle_bypass",
    "emergency_avoidance",
    "gap_direction",
    "gap_safe",
    "blocked_lateral",
    "forward_detour",
    "climb_avoidance",
    "camera_centering_limited",
    "final_vx",
    "final_vy",
    "final_vz",
    "speed_scale",
    "combo_reward_bonus",
    "combo_progress_reward",
    "combo_view_reward",
    "combo_center_reward",
    "combo_capture_reward",
    "combo_collision_penalty",
    "combo_obstacle_penalty",
    "combo_lateral_penalty",
    "combo_slow_penalty",
    "combo_search_reward",
    "combo_jerk_penalty",
    "combo_smooth_reward",
    "combo_gap_reward",
    "combo_target_in_view",
    "combo_bbox_center_x",
    "combo_bbox_center_y",
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
    parser = argparse.ArgumentParser(description="Scenario 1 combo 5000-step PPO fine-tune.")
    parser.add_argument("--load-model", default=str(DEFAULT_LOAD_MODEL))
    parser.add_argument("--save-model", default=str(DEFAULT_SAVE_MODEL))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-duration", type=float, default=0.25)
    parser.add_argument("--target-speed", type=float, default=2.0)
    parser.add_argument("--max-episode-steps", type=int, default=900)
    parser.add_argument("--too-far-distance", type=float, default=320.0)
    parser.add_argument("--use-fast-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-target-on-catch", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args()


WrapperBase = gym.Wrapper if gym is not None else object


class ComboScenarioRewardWrapper(WrapperBase):
    def __init__(self, env, rng, stage_id, chaser_xy_jitter=5.0, target_xy_jitter=10.0, waypoint_y_jitter=15.0):
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
        waypoints = [(x, y + self._jitter(self.waypoint_y_jitter), z) for x, y, z in TARGET_BASE_WAYPOINTS]

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
            "combo_stage": self.stage_id,
            "combo_chaser_start_x": float(chaser_x),
            "combo_chaser_start_y": float(chaser_y),
            "combo_target_start_x": float(target_x),
            "combo_target_start_y": float(target_y),
        }

    def reset(self, **kwargs):
        self._randomize_episode()
        obs, info = self.env.reset(**kwargs)
        self.previous_distance = safe_float(info.get("distance"), None)
        self.previous_velocity = None
        info.update(self._zero_combo_info())
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
        bonus, combo_info = self._combo_reward(info)
        reward = float(reward) + bonus
        info.update(combo_info)
        info.update(self.last_randomization)
        info["reward"] = float(reward)
        self.previous_distance = safe_float(info.get("distance"), self.previous_distance)
        self.previous_velocity = (
            safe_float(info.get("final_vx")),
            safe_float(info.get("final_vy")),
            safe_float(info.get("final_vz")),
        )
        return obs, reward, terminated, truncated, info

    def _zero_combo_info(self):
        return {
            "combo_reward_bonus": 0.0,
            "combo_progress_reward": 0.0,
            "combo_view_reward": 0.0,
            "combo_center_reward": 0.0,
            "combo_capture_reward": 0.0,
            "combo_collision_penalty": 0.0,
            "combo_obstacle_penalty": 0.0,
            "combo_lateral_penalty": 0.0,
            "combo_slow_penalty": 0.0,
            "combo_search_reward": 0.0,
            "combo_jerk_penalty": 0.0,
            "combo_smooth_reward": 0.0,
            "combo_gap_reward": 0.0,
            "combo_target_in_view": False,
            "combo_bbox_center_x": -1.0,
            "combo_bbox_center_y": -1.0,
        }

    def _camera_projection(self, info, width=640.0, height=360.0, fov_degrees=90.0):
        forward = safe_float(info.get("capture_forward"), 0.0)
        lateral = safe_float(info.get("capture_lateral"), 0.0)
        vertical = safe_float(info.get("capture_vertical"), 0.0)
        if forward <= 0.25:
            return False, None, 0.0
        focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
        px = width / 2.0 + (lateral / forward) * focal
        py = height / 2.0 + (vertical / forward) * focal
        target_in_view = 0.0 <= px < width and 0.0 <= py < height
        dx = abs(px - width / 2.0) / (width / 2.0)
        dy = abs(py - height / 2.0) / (height / 2.0)
        center_score = clamp(1.0 - 0.65 * dx - 0.35 * dy, 0.0, 1.0)
        return target_in_view, (px, py), center_score

    def _target_velocity_alignment(self, info):
        dx = safe_float(info.get("dx"), 0.0)
        dy = safe_float(info.get("dy"), 0.0)
        norm = math.sqrt(dx * dx + dy * dy)
        if norm < 1e-6:
            return 0.0
        vx = safe_float(info.get("final_vx"), 0.0)
        vy = safe_float(info.get("final_vy"), 0.0)
        return (vx * dx + vy * dy) / norm

    def _combo_reward(self, info):
        distance = safe_float(info.get("distance"), 0.0)
        previous_distance = self.previous_distance
        improvement = 0.0 if previous_distance is None else float(previous_distance) - distance
        progress_reward = clamp(2.5 * improvement, -4.0, 5.0)
        speed_progress_reward = clamp(0.6 * max(0.0, improvement / max(safe_float(info.get("step_duration"), 0.25), 1e-3)), 0.0, 2.0)

        target_in_view, bbox_center, center_score = self._camera_projection(info)
        view_reward = 0.35 if target_in_view else 0.0
        center_reward = 0.9 * center_score if target_in_view else 0.0
        if target_in_view and center_score > 0.75 and distance < 10.0:
            center_reward += 4.0

        caught = bool(info.get("caught", False))
        capture_reward = 100.0 if caught or bool(info.get("capture_box", False)) or distance <= 3.0 else 0.0
        collision_penalty = -120.0 if bool(info.get("collision", False)) else 0.0

        front = safe_float(info.get("lidar_front"), 50.0)
        front_left = safe_float(info.get("lidar_front_left"), 50.0)
        front_right = safe_float(info.get("lidar_front_right"), 50.0)
        left = min(safe_float(info.get("lidar_left"), 50.0), front_left)
        right = min(safe_float(info.get("lidar_right"), 50.0), front_right)
        obstacle_penalty = -0.6 * max(0.0, 8.0 - front)
        obstacle_penalty += -1.4 * max(0.0, 5.0 - front)
        obstacle_penalty += -0.25 * (max(0.0, 6.0 - left) + max(0.0, 6.0 - right))

        final_vy = safe_float(info.get("final_vy"), 0.0)
        lateral_penalty = 0.0
        if right < 8.0 and final_vy > 0.1:
            lateral_penalty -= 2.0 + 0.5 * final_vy
        if left < 8.0 and final_vy < -0.1:
            lateral_penalty -= 2.0 + 0.5 * abs(final_vy)

        vx = safe_float(info.get("final_vx"), 0.0)
        vy = safe_float(info.get("final_vy"), 0.0)
        vz = safe_float(info.get("final_vz"), 0.0)
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        alignment = self._target_velocity_alignment(info)

        slow_penalty = 0.0
        if front > 12.0 and distance > 15.0:
            if speed < 3.0:
                slow_penalty -= 1.2 * (3.0 - speed)
            if safe_float(info.get("speed_scale"), 1.0) < 0.95:
                slow_penalty -= 0.8

        search_reward = 0.0
        if not target_in_view and distance > 15.0:
            if alignment > 3.0:
                search_reward += 1.0
            elif alignment > 2.0:
                search_reward += 0.4
            else:
                search_reward -= 1.0

        jerk = 0.0
        if self.previous_velocity is not None:
            jerk = math.sqrt(
                (vx - self.previous_velocity[0]) ** 2
                + (vy - self.previous_velocity[1]) ** 2
                + (vz - self.previous_velocity[2]) ** 2
            )
        jerk_penalty = -min(2.0, 0.2 * max(0.0, jerk - 2.0))
        smooth_reward = 0.25 if speed > 2.0 and jerk < 1.5 and distance > 10.0 else 0.0
        if speed < 0.8 and distance > 10.0:
            jerk_penalty -= 1.0

        gap_reward = 0.0
        if bool(info.get("gap_safe", True)) and front < 12.0 and not bool(info.get("collision", False)):
            gap_reward += 0.35
        if bool(info.get("forward_detour", False)):
            gap_reward += 0.25
        if bool(info.get("climb_avoidance", False)) and front < 3.0:
            gap_reward += 0.2

        total = (
            progress_reward
            + speed_progress_reward
            + view_reward
            + center_reward
            + capture_reward
            + collision_penalty
            + obstacle_penalty
            + lateral_penalty
            + slow_penalty
            + search_reward
            + jerk_penalty
            + smooth_reward
            + gap_reward
        )
        bbox_x = -1.0 if bbox_center is None else float(bbox_center[0])
        bbox_y = -1.0 if bbox_center is None else float(bbox_center[1])
        return float(total), {
            "combo_reward_bonus": float(total),
            "combo_progress_reward": float(progress_reward + speed_progress_reward),
            "combo_view_reward": float(view_reward),
            "combo_center_reward": float(center_reward),
            "combo_capture_reward": float(capture_reward),
            "combo_collision_penalty": float(collision_penalty),
            "combo_obstacle_penalty": float(obstacle_penalty),
            "combo_lateral_penalty": float(lateral_penalty),
            "combo_slow_penalty": float(slow_penalty),
            "combo_search_reward": float(search_reward),
            "combo_jerk_penalty": float(jerk_penalty),
            "combo_smooth_reward": float(smooth_reward),
            "combo_gap_reward": float(gap_reward),
            "combo_target_in_view": bool(target_in_view),
            "combo_bbox_center_x": bbox_x,
            "combo_bbox_center_y": bbox_y,
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

    wrapped = ComboScenarioRewardWrapper(env, rng, stage_id=stage["stage"])
    monitor_path = LOGS_DIR / f"combo_5000_stage{stage['stage']}_monitor.csv"
    return Monitor(wrapped, filename=str(monitor_path), info_keywords=INFO_KEYWORDS), monitor_path


def main():
    args = parse_args()
    if not check_required_packages():
        sys.exit(1)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    class ComboProgressCallback(BaseCallback):
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
                    "[COMBO_EPISODE] "
                    f"stage={self.stage_id} "
                    f"timestep={self.num_timesteps} "
                    f"episode_reward={float(ep.get('r', 0.0)):.2f} "
                    f"episode_length={int(ep.get('l', 0))} "
                    f"final_distance={float(ep.get('distance', info.get('distance', 0.0))):.2f} "
                    f"caught={ep.get('caught', info.get('caught', False))} "
                    f"collision={ep.get('collision', info.get('collision', False))} "
                    f"reason={ep.get('done_reason', info.get('done_reason', 'none'))}",
                    flush=True,
                )

            if self.num_timesteps % self.print_freq != 0:
                return True
            print(
                "[COMBO_TRAIN] "
                f"stage={self.stage_id} "
                f"timestep={self.num_timesteps} "
                f"distance={float(info.get('distance', 0.0)):.2f} "
                f"front={float(info.get('lidar_front', 0.0)):.2f} "
                f"left={float(info.get('lidar_left', 0.0)):.2f} "
                f"right={float(info.get('lidar_right', 0.0)):.2f} "
                f"target_in_view={info.get('combo_target_in_view', False)} "
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
                f"camera_centering_limited={info.get('camera_centering_limited', False)} "
                f"final_vx={float(info.get('final_vx', 0.0)):.2f} "
                f"final_vy={float(info.get('final_vy', 0.0)):.2f} "
                f"final_vz={float(info.get('final_vz', 0.0)):.2f} "
                f"combo_reward={float(info.get('combo_reward_bonus', 0.0)):.2f} "
                f"caught={info.get('caught', False)} "
                f"collision={info.get('collision', False)}",
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
        raise RuntimeError("Refusing to overwrite the source PPO model.")

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    tensorboard_dir = LOGS_DIR / "tensorboard_combo_5000"
    checkpoint_dir = MODELS_DIR / "checkpoints_combo_5000"
    checkpoint_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    current_load_model = load_model
    print("[COMBO_5000] Scenario 1 combo fine-tune", flush=True)
    print(f"[INFO] source_model={load_model}", flush=True)
    print(f"[INFO] output_model={save_model}", flush=True)
    print("[INFO] compatibility preserved: obs_mode=legacy14 action_space=Discrete(6)", flush=True)
    print("[INFO] stages: 2000 steps @ speed 4.5, then 3000 steps @ speed 5.5", flush=True)

    for stage in COMBO_STAGES:
        env = None
        try:
            print(
                f"[COMBO_STAGE] stage={stage['stage']} speed={stage['chaser_speed']:.1f} "
                f"timesteps={stage['timesteps']} load={current_load_model}",
                flush=True,
            )
            env, monitor_path = make_env(args, stage, rng)
            if args.check_env:
                from stable_baselines3.common.env_checker import check_env

                print("[INFO] Running stable_baselines3 check_env...", flush=True)
                check_env(env, warn=True)

            model = PPO.load(str(current_load_model), env=env, device="cpu")
            model.verbose = 1
            model.tensorboard_log = str(tensorboard_dir)
            callbacks = [
                CheckpointCallback(
                    save_freq=2500,
                    save_path=str(checkpoint_dir),
                    name_prefix=f"combo_5000_stage{stage['stage']}",
                ),
                ComboProgressCallback(stage_id=stage["stage"], print_freq=50),
            ]
            model.learn(
                total_timesteps=stage["timesteps"],
                callback=callbacks,
                reset_num_timesteps=False,
                tb_log_name=f"combo_5000_stage{stage['stage']}",
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

    print(f"[OK] Combo model saved: {save_model}", flush=True)
    print(f"[OK] TensorBoard log dir: {tensorboard_dir}", flush=True)
    print("[OK] Original models/ppo_chaser_step6.zip was not overwritten.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[WARN] Combo fine-tune interrupted by user.", flush=True)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
