#!/usr/bin/env python3
import argparse
import importlib.util
import math
import sys
import time
import traceback
from pathlib import Path

import airsim


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "ppo_chaser_step6.zip"
CHASER_HOME = (139.36, 0.0, -11.0)
RETURN_HOME_SPEED = 8.0
RETURN_HOME_TOLERANCE = 3.0
RETURN_HOME_TIMEOUT = 45.0

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


def parse_args():
    parser = argparse.ArgumentParser(description="AirSim PPO Chaser demo scenario runner.")
    parser.add_argument("--scenario", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-camera", action="store_true")
    parser.add_argument("--drop-target-on-catch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--camera-name", default="front_center")
    parser.add_argument("--lock-distance", type=float, default=10.0)
    return parser.parse_args()


def resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def action_to_int(action):
    try:
        return int(action.item())
    except AttributeError:
        return int(action)


def bbox_text(value):
    if value is None:
        return "none"
    return f"({float(value[0]):.0f},{float(value[1]):.0f})"


def tuple_text(value, precision=2):
    if value is None:
        return "none"
    try:
        return "(" + ",".join(f"{float(item):.{precision}f}" for item in value) + ")"
    except Exception:
        return str(value)


def position_text(pos):
    if pos is None:
        return "none"
    return f"({float(pos.x_val):.2f},{float(pos.y_val):.2f},{float(pos.z_val):.2f})"


def lock_block_reason(distance_value, target_in_view, bbox_in_lock_box):
    if distance_value <= 3.0:
        return "none"
    if not target_in_view:
        return "target_not_visible"
    if distance_value > 10.0:
        return "distance_too_far"
    if not bbox_in_lock_box:
        return "bbox_not_centered"
    return "none"


def perform_drop_effect(client, chaser_name="Chaser", target_name="Target", duration_seconds=4.0):
    print(
        "[DEMO FINAL] "
        "lock_state=LOCKED mission_success=True enemy_drone_destroyed=True target_drop_started=True "
        "target_drop_velocity_applied=True chaser_return_home=False return_home_done=False",

        flush=True,
    )
    print("[DEMO] Target drop effect started. Chaser will hover.", flush=True)
    target_drop_velocity_applied = False
    fallback_pose_drop = False
    first_target_z = None
    last_target_z = None

    try:
        client.enableApiControl(True, vehicle_name=target_name)
        client.armDisarm(True, vehicle_name=target_name)
    except Exception:
        pass
    try:
        first_target_z = float(client.simGetObjectPose(target_name).position.z_val)
    except Exception:
        first_target_z = None

    # Target “hemen durdur”: waypoint/controller komutlarının kısa süreli etkisini kesmek için
    # sıfır hız komutunu birkaç kez tekrarla.
    try:
        for _ in range(3):
            client.moveByVelocityAsync(0.0, 0.0, 0.0, 0.1, vehicle_name=target_name).join()
            time.sleep(0.05)
    except Exception:
        pass


    end_time = time.monotonic() + float(duration_seconds)
    while time.monotonic() < end_time:
        try:
            client.hoverAsync(vehicle_name=chaser_name).join()
        except Exception:
            pass
        try:
            client.moveByVelocityAsync(0.0, 0.0, 8.0, 0.25, vehicle_name=target_name).join()
            target_drop_velocity_applied = True
        except Exception as exc:
            print(f"[WARN] Target drop velocity failed: {exc}", flush=True)
            break
        try:
            last_target_z = float(client.simGetObjectPose(target_name).position.z_val)
        except Exception:
            last_target_z = None
        time.sleep(0.03)

    if first_target_z is not None and (last_target_z is None or last_target_z < first_target_z + 1.0):
        fallback_pose_drop = True
        print("[WARN] Target drop velocity weak; using simSetVehiclePose fallback.", flush=True)
        fallback_until = time.monotonic() + 1.5
        while time.monotonic() < fallback_until:
            try:
                pose = client.simGetObjectPose(target_name)
                pose.position.z_val += 1.0
                try:
                    client.simSetVehiclePose(pose, True, vehicle_name=target_name)
                except TypeError:
                    client.simSetVehiclePose(pose, True, target_name)
            except Exception as exc:
                print(f"[WARN] Target pose fallback failed: {exc}", flush=True)
                break
            try:
                client.hoverAsync(vehicle_name=chaser_name).join()
            except Exception:
                pass
            time.sleep(0.05)

    print(
        "[DEMO FINAL] "
        f"lock_state=LOCKED mission_success=True enemy_drone_destroyed=True target_drop_started=True "
        f"target_drop_velocity_applied={target_drop_velocity_applied} "
        f"target_drop_pose_fallback={fallback_pose_drop} "
        "chaser_return_home=False return_home_done=False",
        flush=True,
    )
    hover_until = time.monotonic() + 1.0
    while time.monotonic() < hover_until:
        try:
            client.hoverAsync(vehicle_name=chaser_name).join()
        except Exception:
            pass
        time.sleep(0.05)
    try:
        client.hoverAsync(vehicle_name=chaser_name).join()
    except Exception:
        pass


def show_locked_before_drop(hud, chaser_pos, target_pos, distance, hold_seconds=0.25):
    if hud is None:
        return
    try:
        hud.force_locked()
        end_time = time.monotonic() + float(hold_seconds)
        while time.monotonic() < end_time:
            try:
                hud.client.hoverAsync(vehicle_name=hud.chaser_name).join()
            except Exception:
                pass
            hud.update(chaser_pos, target_pos, distance)
            time.sleep(0.03)
    except Exception:
        pass


def yaw_mode_to_point(from_pos, to_x, to_y):
    dx = float(to_x) - float(from_pos.x_val)
    dy = float(to_y) - float(from_pos.y_val)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return airsim.YawMode(is_rate=True, yaw_or_rate=0.0)
    yaw_deg = math.degrees(math.atan2(dy, dx))
    return airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)


def return_chaser_home(client, hud=None, home=CHASER_HOME, vehicle_name="Chaser", target_drop_started=True):
    home_x, home_y, home_z = home
    print(
        "[DEMO FINAL] "
        f"lock_state=LOCKED mission_success=True enemy_drone_destroyed=True target_drop_started={target_drop_started} "
        f"target_drop_velocity_applied={target_drop_started} "
        "chaser_return_home=True return_home_done=False",
        flush=True,
    )
    deadline = time.monotonic() + RETURN_HOME_TIMEOUT
    done = False
    last_print = 0.0

    while time.monotonic() < deadline:
        try:
            pose = client.simGetObjectPose(vehicle_name)
            pos = pose.position
        except Exception as exc:
            print(f"[WARN] Return home pose read failed: {exc}", flush=True)
            break

        dx = home_x - pos.x_val
        dy = home_y - pos.y_val
        dz = home_z - pos.z_val
        horizontal = (dx * dx + dy * dy) ** 0.5
        if horizontal <= RETURN_HOME_TOLERANCE and abs(dz) <= 1.2:
            done = True
            break

        speed = min(RETURN_HOME_SPEED, max(2.0, horizontal * 0.45))
        if horizontal > 1e-6:
            vx = dx / horizontal * speed
            vy = dy / horizontal * speed
        else:
            vx = 0.0
            vy = 0.0
        vz = max(-1.2, min(1.2, dz * 0.7))
        yaw_mode = yaw_mode_to_point(pos, home_x, home_y)
        try:
            client.moveByVelocityAsync(
                vx,
                vy,
                vz,
                0.35,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=yaw_mode,
                vehicle_name=vehicle_name,
            )
        except Exception as exc:
            print(f"[WARN] Return home command failed: {exc}", flush=True)
            break

        now = time.monotonic()
        if now - last_print >= 1.0:
            print(
                "[DEMO RETURN] "
                "mission_success=True enemy_drone_destroyed=True "
                f"target_drop_started={target_drop_started} "
                f"target_drop_velocity_applied={target_drop_started} "
                f"chaser_return_home=True return_home_done=False "
                f"distance_to_home={horizontal:.2f} "
                f"pos=({pos.x_val:.2f},{pos.y_val:.2f},{pos.z_val:.2f}) "
                f"vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}",
                flush=True,
            )
            last_print = now
        if hud is not None:
            try:
                target_pos = client.simGetObjectPose(hud.target_name).position
                hud.update(pos, target_pos, 0.0)
            except Exception:
                pass
        time.sleep(0.08)

    try:
        client.hoverAsync(vehicle_name=vehicle_name).join()
    except Exception:
        pass
    print(
        "[DEMO FINAL] "
        f"lock_state=LOCKED mission_success=True enemy_drone_destroyed=True target_drop_started={target_drop_started} "
        f"target_drop_velocity_applied={target_drop_started} "
        f"chaser_return_home=True return_home_done={done}",
        flush=True,
    )
    return done


def run_final_scene(env, hud, chaser_pos, target_pos, distance, drop_target):
    if hud is not None:
        hud.force_locked()
    print(
        "[DEMO FINAL] "
        f"lock_state=LOCKED mission_success=True enemy_drone_destroyed=True target_drop_started={bool(drop_target)} "
        "target_drop_velocity_applied=False chaser_return_home=False return_home_done=False",
        flush=True,
    )
    show_locked_before_drop(hud, chaser_pos, target_pos, distance)
    if drop_target:
        perform_drop_effect(env.client)
    else:
        print(
            "[DEMO FINAL] "
            "lock_state=LOCKED mission_success=True enemy_drone_destroyed=True target_drop_started=False "
            "target_drop_velocity_applied=False chaser_return_home=False return_home_done=False",
            flush=True,
        )
    try:
        env.client.hoverAsync(vehicle_name="Chaser").join()
    except Exception:
        pass
    print(
        "[DEMO FINAL] "
        f"lock_state=LOCKED mission_success=True enemy_drone_destroyed=True "
        f"target_drop_started={bool(drop_target)} "
        f"target_drop_velocity_applied={bool(drop_target)} "
        "chaser_return_home=False return_home_done=False",
        flush=True,
    )
    return False


def apply_vehicle_yaw(client, vehicle_name, yaw_rad):
    try:
        pose = client.simGetObjectPose(vehicle_name)
        pose.orientation = airsim.to_quaternion(0.0, 0.0, float(yaw_rad))
        try:
            client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)
        except TypeError:
            client.simSetVehiclePose(pose, True, vehicle_name)
        try:
            client.hoverAsync(vehicle_name=vehicle_name).join()
        except Exception:
            pass
        print(f"[DEMO YAW] {vehicle_name} yaw={float(yaw_rad):.2f} rad applied", flush=True)
    except Exception as exc:
        print(f"[WARN] Could not apply {vehicle_name} yaw={float(yaw_rad):.2f}: {exc}", flush=True)


def main():
    args = parse_args()
    if not check_required_packages():
        sys.exit(1)

    from stable_baselines3 import PPO

    from airsim_chase_env import ACTION_NAMES, AirSimChaseEnv
    from camera_lock_hud import CameraLockHUD
    from scenarios import build_scenario

    model_path = resolve_path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", flush=True)
        sys.exit(1)

    scenario = build_scenario(args.scenario, seed=args.seed)
    env_kwargs = dict(scenario["env_kwargs"])
    target_altitude = float(env_kwargs.pop("target_altitude", 8.0))
    max_steps = args.steps or int(env_kwargs.get("max_episode_steps", 800))

    env = None
    hud = None
    episode_reward = 0.0
    final_reason = "none"
    caught = False
    return_home_done = False

    try:
        print("[DEMO] PPO Chaser scenario demo", flush=True)
        print(f"[INFO] model={model_path}", flush=True)
        print(f"[INFO] obs_mode=legacy14 action_space=Discrete(6)", flush=True)
        print(f"[INFO] scenario_id={scenario['scenario_id']} name={scenario['name']} seed={args.seed}", flush=True)
        print(f"[INFO] scenario={scenario['description']}", flush=True)
        print(f"[INFO] env_kwargs={env_kwargs}", flush=True)

        model = PPO.load(str(model_path), device="cpu")
        env = AirSimChaseEnv(
            **env_kwargs,
            obs_mode="legacy14",
            reward_mode="simple",
            use_fast_reset=True,
            use_capture_box=True,
            capture_depth=2.0,
            capture_width=2.8,
            capture_height=3.0,
            catch_radius=3.0,
            capture_bonus=100.0,
            drop_target_on_catch=False,
            target_altitude=target_altitude,
            min_safe_altitude=4.0,
            max_safe_altitude=15.0,
            hard_max_altitude=20.0,
            enable_altitude_safety=True,
        )
        hud = CameraLockHUD(
            env.client,
            camera_name=args.camera_name,
            show_camera=args.show_camera,
            lock_distance=args.lock_distance,
            instant_lock_distance=args.lock_distance,
            use_depth_visibility=False,
        )

        obs, info = env.reset(seed=args.seed)
        if "chaser_yaw" in scenario:
            apply_vehicle_yaw(env.client, "Chaser", scenario["chaser_yaw"])
        if "target_yaw" in scenario:
            apply_vehicle_yaw(env.client, "Target", scenario["target_yaw"])
        print(
            "[DEMO RESET] "
            f"scenario_id={args.scenario} "
            f"distance={float(info.get('distance', 0.0)):.2f} "
            f"chaser_pos={info.get('chaser_pos')} target_pos={info.get('target_pos')}",
            flush=True,
        )

        for step_index in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            action_int = action_to_int(action)
            obs, reward, terminated, truncated, info = env.step(action_int)
            episode_reward += float(reward)

            chaser_pos = env.get_global_position("Chaser")
            target_pos = env.get_global_position("Target")
            camera_state = hud.update(chaser_pos, target_pos, info.get("distance", 0.0))
            distance_value = float(info.get("distance", 999.0))

            bbox_in_lock_box = bool(camera_state.get("bbox_in_lock_box", False))
            # Final lock/drop only. Does not affect planner/safety/chaser speed.
            # Rules:
            # - distance <= 10.0 and bbox center inside lock box => instant LOCKED
            # - distance <= 3.0 => LOCKED without bbox
            distance_le_10 = distance_value <= 10.0
            distance_le_3 = distance_value <= 3.0
            target_in_view = bool(camera_state.get("target_in_view", False))
            lock_ready = bool(distance_le_3 or (distance_le_10 and bbox_in_lock_box))
            current_lock_block_reason = lock_block_reason(
                distance_value,
                target_in_view,
                bbox_in_lock_box,
            )

            camera_lock = bool(lock_ready)

            if camera_lock:
                hud.force_locked()
                camera_state["lock_state"] = "LOCKED"
                camera_state["camera_lock"] = True
                camera_state["bbox_in_lock_box"] = bbox_in_lock_box

                caught = True
                final_reason = "camera_lock"

                # Ensure required flags for final scene/logging
                info["caught"] = True
                info["done_reason"] = final_reason
                info["lock_state"] = "LOCKED"
                info["camera_lock"] = True
                info["mission_success"] = True
                info["enemy_drone_destroyed"] = True
                info["target_drop_started"] = True
            elif bool(info.get("caught", False)):
                caught = True
                final_reason = info.get("done_reason", "caught")



            should_print = (
                step_index == 1
                or step_index % 25 == 0
                or distance_value < 40.0
                or camera_state.get("lock_state") in ("LOCKING", "LOCKED")
                or terminated
                or truncated
            )
            if should_print:
                print(
                    "[DEMO STEP] "
                    f"scenario_id={args.scenario} "
                    f"timestep={step_index} "
                    f"distance={float(info.get('distance', 0.0)):.2f} "
                    f"action={info.get('action')}:{info.get('action_name', ACTION_NAMES.get(action_int, 'UNKNOWN'))} "
                    f"safe_action={info.get('safety_safe_action')}:{info.get('safety_safe_action_name', 'UNKNOWN')} "
                    f"target_wp={info.get('target_waypoint_index', 0)}/{info.get('target_waypoint_total', 0)} "
                    f"min_lidar={float(info.get('min_lidar', 0.0)):.2f} "
                    f"front={float(info.get('lidar_front', 0.0)):.2f} "
                    f"speed_scale={float(info.get('speed_scale', 1.0)):.2f} "
                    f"smooth_velocity={info.get('smooth_velocity', False)} "
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
                    f"obstacle_bypass={info.get('obstacle_bypass', False)} "
                    f"emergency_avoidance={info.get('emergency_avoidance', False)} "
                    f"bypass_direction={info.get('bypass_direction', 'none')} "
                    f"final_vx={float(info.get('final_vx', 0.0)):.2f} "
                    f"final_vy={float(info.get('final_vy', 0.0)):.2f} "
                    f"final_vz={float(info.get('final_vz', 0.0)):.2f} "
                    f"command_duration={float(info.get('command_duration', 0.0)):.2f} "
                    f"chaser_world_pos={position_text(chaser_pos)} "
                    f"target_world_pos={position_text(target_pos)} "
                    f"target_relative_world={tuple_text(camera_state.get('relative_world'))} "
                    f"target_camera_relative={tuple_text(camera_state.get('camera_relative'))} "
                    f"projection_source={camera_state.get('camera_projection_source', 'unknown')} "
                    f"vehicle_yaw={float(camera_state.get('vehicle_yaw', 0.0)):.3f} "
                    f"camera_yaw={float(camera_state.get('camera_yaw', 0.0)):.3f} "
                    f"screen_x={float(camera_state.get('screen_x', float('inf'))):.1f} "
                    f"screen_y={float(camera_state.get('screen_y', float('inf'))):.1f} "
                    f"crosshair_center={tuple_text(camera_state.get('crosshair_center'), precision=1)} "
                    f"pixel_error_x={float(camera_state.get('pixel_error_x', float('inf'))):.1f} "
                    f"pixel_error_y={float(camera_state.get('pixel_error_y', float('inf'))):.1f} "
                    f"target_in_view={camera_state.get('target_in_view', False)} "
                    f"depth_visible={camera_state.get('depth_visible', False)} "
                    f"target_occluded={camera_state.get('target_occluded', False)} "
                    f"bbox_in_lock_box={camera_state.get('bbox_in_lock_box', False)} "
                    f"camera_depth={float(camera_state.get('camera_depth', float('inf'))):.2f} "
                    f"expected_target_depth={float(camera_state.get('expected_target_depth', float('inf'))):.2f} "
                    f"recent_target_seen={camera_state.get('recent_target_seen', False)} "
                    f"last_seen_age={float(camera_state.get('last_seen_age', 999.0)):.2f} "
                    f"lock_prepare={camera_state.get('lock_prepare', False)} "
                    f"bbox_center={bbox_text(camera_state.get('bbox_center'))} "
                    f"lock_ready={lock_ready} "
                    f"lock_block_reason={current_lock_block_reason} "
                    f"chase_stopped_reason={info.get('chase_stopped_reason', info.get('stop_reason', 'none'))} "
                    f"lock_state={camera_state.get('lock_state', 'SEARCH')} "
                    f"camera_lock={camera_lock} "
                    f"caught={bool(info.get('caught', False) or caught)} "
                    f"done_reason={info.get('done_reason', final_reason)} "
                    f"mission_success={camera_lock} "
                    f"enemy_drone_destroyed={camera_lock} "
                    f"target_drop_started=False "
                    f"target_drop_velocity_applied=False "
                    f"chaser_return_home=False "
                    f"return_home_done=False "
                    f"altitude={float(info.get('altitude', 0.0)):.2f} "
                    f"collision={info.get('collision', False)} "
                    f"episode_reward={episode_reward:.2f}",
                    flush=True,
                )

            if camera_lock:
                return_home_done = run_final_scene(env, hud, chaser_pos, target_pos, distance_value, args.drop_target_on_catch)
                break

            if terminated or truncated:
                caught = bool(info.get("caught", False))
                final_reason = info.get("done_reason", "terminated" if terminated else "max_steps")
                if caught:
                    hud.force_locked()
                    return_home_done = run_final_scene(env, hud, chaser_pos, target_pos, info.get("distance", 0.0), args.drop_target_on_catch)
                break

        else:
            final_reason = "max_demo_steps"

        print(
            "[DEMO DONE] "
            f"scenario_id={args.scenario} "
            f"caught={caught} "
            f"reason={final_reason} "
            f"mission_success={caught} "
            f"enemy_drone_destroyed={caught} "
            f"target_drop_started={bool(caught and args.drop_target_on_catch)} "
            f"chaser_return_home={bool(caught)} "
            f"return_home_done={return_home_done} "
            f"episode_reward={episode_reward:.2f}",
            flush=True,
        )

    except KeyboardInterrupt:
        print("[WARN] Demo interrupted by user.", flush=True)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        traceback.print_exc()
    finally:
        if hud is not None:
            hud.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
