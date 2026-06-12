#!/usr/bin/env python3
import math


class TargetController:
    def __init__(
        self,
        client=None,
        vehicle_name="Target",
        safe_z=-5.0,
        base_speed=1.2,
        escape_speed=1.5,
        evade_distance=8.0,
        danger_distance=4.0,
        random_maneuver_interval=5,
        lateral_strength=0.35,
    ):
        self.client = client
        self.vehicle_name = vehicle_name
        self.safe_z = safe_z
        self.base_speed = base_speed
        self.escape_speed = escape_speed
        self.evade_distance = evade_distance
        self.danger_distance = danger_distance
        self.random_maneuver_interval = max(1, int(random_maneuver_interval))
        self.lateral_strength = lateral_strength
        self._side_direction = 1.0

    def compute_target_velocity(self, chaser_pos, target_pos, lidar_sectors=None, step_count=0):
        dx = target_pos.x_val - chaser_pos.x_val
        dy = target_pos.y_val - chaser_pos.y_val
        distance = math.sqrt(dx * dx + dy * dy)

        if step_count % self.random_maneuver_interval == 0:
            self._side_direction *= -1.0

        if distance < self.evade_distance:
            vx, vy = self._escape_velocity(dx, dy, distance)
            if distance < self.danger_distance:
                side_x, side_y = self._perpendicular(dx, dy, distance)
                vx += side_x * self.escape_speed * self.lateral_strength * self._side_direction
                vy += side_y * self.escape_speed * self.lateral_strength * self._side_direction
        else:
            vx = self.base_speed
            vy = self.base_speed * self.lateral_strength * self._side_direction

        vx, vy = self._apply_obstacle_avoidance(vx, vy, lidar_sectors)
        vz = self._altitude_correction(target_pos.z_val)
        vx, vy, vz = self._limit_velocity(vx, vy, vz, self.escape_speed)
        return vx, vy, vz

    def _escape_velocity(self, dx, dy, distance):
        if distance < 1e-6:
            return self.escape_speed, 0.0
        return self.escape_speed * dx / distance, self.escape_speed * dy / distance

    def _perpendicular(self, dx, dy, distance):
        if distance < 1e-6:
            return 0.0, 1.0
        return -dy / distance, dx / distance

    def _apply_obstacle_avoidance(self, vx, vy, lidar_sectors):
        if not lidar_sectors:
            return vx, vy

        front = float(lidar_sectors.get("front", 50.0))
        left = float(lidar_sectors.get("left", 50.0))
        right = float(lidar_sectors.get("right", 50.0))

        if front >= 3.0:
            return vx, vy

        vx = min(vx, 0.0)
        if right > left and right > 2.0:
            vy = max(abs(vy), self.base_speed)
        elif left > 2.0:
            vy = -max(abs(vy), self.base_speed)
        else:
            vx *= 0.25
            vy *= 0.25

        return vx, vy

    def _limit_velocity(self, vx, vy, vz, max_speed):
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        if speed <= max_speed or speed < 1e-6:
            return vx, vy, vz
        scale = max_speed / speed
        return vx * scale, vy * scale, vz * scale

    def _altitude_correction(self, target_z):
        error = self.safe_z - target_z
        if abs(error) < 0.4:
            return 0.0
        return max(min(error, 0.6), -0.6)
