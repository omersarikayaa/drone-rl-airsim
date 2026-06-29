from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from airsim_chase_env import AirSimChaseEnv


LOAD_MODEL = "models/ppo_chaser_step6.zip"
SAVE_MODEL = "models/ppo_chaser_step6_seed42_crash_spot.zip"


def make_env(chaser_speed=3.5):
    env = AirSimChaseEnv(
        obs_mode="legacy14",
        target_mode="simple",
        target_base_speed=0.8,
        target_escape_speed=1.2,
        target_evade_distance=12.0,
        target_danger_distance=5.0,

        chaser_start_x=0.0,
        chaser_start_y=0.0,
        chaser_start_z=-8.0,

        target_start_x=110.3751026229528,
        target_start_y=78.730420780914,
        target_start_z=-8.0,

        max_episode_steps=800,
        too_far_distance=260.0,
        chaser_speed=chaser_speed,
        step_duration=0.3,

        use_capture_box=True,
        capture_depth=3.5,
        capture_width=2.5,
        capture_height=2.0,
        catch_radius=2.5,
        drop_target_on_catch=True,
    )
    return Monitor(env)


def train_stage(load_model, save_model, timesteps, speed):
    print("=" * 60)
    print(f"[TRAIN_FIXED_SEED42] speed={speed} timesteps={timesteps}")
    print(f"[LOAD] {load_model}")
    print(f"[SAVE] {save_model}")
    print("=" * 60)

    env = make_env(chaser_speed=speed)

    model = PPO.load(load_model, env=env, device="cpu")
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
    model.save(save_model)

    env.close()
    print(f"[SAVED] {save_model}")


def main():
    # Aynı çarpılan konumda kolaydan zora eğitim.
    train_stage(
        load_model=LOAD_MODEL,
        save_model=SAVE_MODEL,
        timesteps=2000,
        speed=3.5,
    )

    train_stage(
        load_model=SAVE_MODEL,
        save_model=SAVE_MODEL,
        timesteps=3000,
        speed=4.2,
    )

    train_stage(
        load_model=SAVE_MODEL,
        save_model=SAVE_MODEL,
        timesteps=3000,
        speed=5.0,
    )

    print("[DONE] Fixed seed42 crash spot training completed.")


if __name__ == "__main__":
    main()
