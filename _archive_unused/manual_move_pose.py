import math
import time
from pathlib import Path

import airsim


VEHICLE_NAME = "Chaser"   # İstersen "Target" yaparsın
SPEED = 3.0               # m/s
MOVE_DURATION = 0.5       # saniye
YAW_RATE = 35             # derece/saniye
POSE_FILE = Path("current_pose.py")


def get_pose(client, vehicle_name):
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    pos = state.kinematics_estimated.position
    orientation = state.kinematics_estimated.orientation

    pitch, roll, yaw = airsim.to_eularian_angles(orientation)
    yaw_deg = math.degrees(yaw)

    return {
        "x": pos.x_val,
        "y": pos.y_val,
        "z": pos.z_val,
        "yaw_rad": yaw,
        "yaw_deg": yaw_deg,
        "altitude": -pos.z_val,
    }


def write_pose_file(pose, vehicle_name):
    text = f'''# AUTO-GENERATED FILE
# Bu dosya manual_move_pose.py tarafından sürekli güncellenir.
# AirSim NED: z negatifse drone yukarıdadır.

VEHICLE_NAME = "{vehicle_name}"

X = {pose["x"]:.3f}
Y = {pose["y"]:.3f}
Z = {pose["z"]:.3f}

YAW_DEG = {pose["yaw_deg"]:.3f}
ALTITUDE = {pose["altitude"]:.3f}

# Kopyalamak için:
POSITION = ({pose["x"]:.3f}, {pose["y"]:.3f}, {pose["z"]:.3f})
'''
    POSE_FILE.write_text(text, encoding="utf-8")


def print_pose(pose):
    print(
        f'X={pose["x"]:.2f}  '
        f'Y={pose["y"]:.2f}  '
        f'Z={pose["z"]:.2f}  '
        f'YAW={pose["yaw_deg"]:.1f}°  '
        f'ALT={pose["altitude"]:.2f}m'
    )


def move_body_direction(client, vehicle_name, forward=0.0, right=0.0, up=0.0):
    pose = get_pose(client, vehicle_name)
    yaw = pose["yaw_rad"]

    # Drone gövdesine göre ileri/sağ hareketi dünya koordinatına çeviriyoruz.
    vx = forward * math.cos(yaw) + right * math.cos(yaw + math.pi / 2)
    vy = forward * math.sin(yaw) + right * math.sin(yaw + math.pi / 2)

    # AirSim NED: yukarı çıkmak için z hızı negatif olmalı.
    vz = -up

    client.moveByVelocityAsync(
        vx,
        vy,
        vz,
        MOVE_DURATION,
        vehicle_name=vehicle_name
    ).join()


def yaw_turn(client, vehicle_name, direction):
    # direction: -1 sola, +1 sağa
    client.moveByVelocityAsync(
        0,
        0,
        0,
        MOVE_DURATION,
        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=direction * YAW_RATE),
        vehicle_name=vehicle_name
    ).join()


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    client.armDisarm(True, vehicle_name=VEHICLE_NAME)

    print("\nMANUAL MOVE MODE")
    print("----------------")
    print("w: ileri")
    print("s: geri")
    print("a: sol")
    print("d: sağ")
    print("r: yukarı")
    print("f: aşağı")
    print("q: sola dön")
    print("e: sağa dön")
    print("p: sadece koordinatı yaz")
    print("x: dur")
    print("exit: çık")
    print("----------------\n")

    while True:
        pose = get_pose(client, VEHICLE_NAME)
        write_pose_file(pose, VEHICLE_NAME)
        print_pose(pose)

        cmd = input("Komut gir: ").strip().lower()

        if cmd in ["exit", "quit", "çık"]:
            break

        if cmd == "w":
            move_body_direction(client, VEHICLE_NAME, forward=SPEED)
        elif cmd == "s":
            move_body_direction(client, VEHICLE_NAME, forward=-SPEED)
        elif cmd == "a":
            move_body_direction(client, VEHICLE_NAME, right=-SPEED)
        elif cmd == "d":
            move_body_direction(client, VEHICLE_NAME, right=SPEED)
        elif cmd == "r":
            move_body_direction(client, VEHICLE_NAME, up=SPEED)
        elif cmd == "f":
            move_body_direction(client, VEHICLE_NAME, up=-SPEED)
        elif cmd == "q":
            yaw_turn(client, VEHICLE_NAME, direction=-1)
        elif cmd == "e":
            yaw_turn(client, VEHICLE_NAME, direction=1)
        elif cmd == "x":
            client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
        elif cmd == "p":
            pass
        else:
            print("Bilinmeyen komut.")

        time.sleep(0.1)

    client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
    print("Çıkıldı.")


if __name__ == "__main__":
    main()