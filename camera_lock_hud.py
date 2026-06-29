#!/usr/bin/env python3
import math
import time

import airsim
import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


WINDOW_NAME = "Chaser Camera Lock View"


class CameraLockHUD:

    def __init__(
        self,
        client,
        chaser_name="Chaser",
        target_name="Target",
        camera_name="front_center",
        width=640,
        height=360,
        fov_degrees=90.0,
        show_camera=False,
        lock_distance=10.0,
        lock_hold_seconds=0.5,
        center_fraction=0.16,
        depth_sample_radius=8,
        instant_lock_distance=10.0,
        use_depth_visibility=False,
    ):
        self.client = client
        self.chaser_name = chaser_name
        self.target_name = target_name
        self.camera_name = camera_name
        self.width = int(width)
        self.height = int(height)
        self.fov_degrees = float(fov_degrees)
        self.show_camera = bool(show_camera and CV2_AVAILABLE)
        self.lock_distance = float(lock_distance)
        self.lock_hold_seconds = float(lock_hold_seconds)
        self.center_fraction = float(center_fraction)
        self.depth_sample_radius = int(depth_sample_radius)
        self.instant_lock_distance = float(instant_lock_distance)
        self.use_depth_visibility = bool(use_depth_visibility)
        self.lock_started_at = None
        self.locked = False
        self.last_seen_at = None
        self.last_error = ""
        self.last_depth_error = ""

        if show_camera and not CV2_AVAILABLE:
            print("[WARN] OpenCV is not available; camera HUD window disabled.", flush=True)

    def close(self):
        if self.show_camera:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass

    def update(self, chaser_pos, target_pos, distance):
        projection = self._project_target(chaser_pos, target_pos)
        projected_in_view = bool(projection.get("target_in_view", False))
        projected_bbox_center = projection.get("bbox_center")
        distance = float(distance)
        distance_lock = distance <= 3.0
        if self.use_depth_visibility and distance <= max(self.lock_distance, self.instant_lock_distance):
            depth_state = self._depth_visibility(projection, projected_bbox_center, distance)
            depth_visible = bool(depth_state.get("depth_visible", False))
            target_occluded = bool(projected_in_view and not depth_visible)
        else:
            depth_state = {
                "depth_visible": projected_in_view,
                "camera_depth": float("inf"),
                "expected_target_depth": float(projection.get("forward", distance) or distance),
            }
            depth_visible = projected_in_view
            target_occluded = False
        depth_valid = bool(math.isfinite(float(depth_state.get("camera_depth", float("inf")))))
        target_in_view = bool(projected_in_view and depth_visible)
        bbox_center = projected_bbox_center if target_in_view else None
        bbox_in_lock_box = self._bbox_in_lock_box(target_in_view, bbox_center)
        in_lock_zone = bool(bbox_in_lock_box and distance <= self.lock_distance)
        instant_camera_lock = bool(bbox_in_lock_box and distance <= self.instant_lock_distance)
        now = time.monotonic()
        if target_in_view:
            self.last_seen_at = now
        last_seen_age = float("inf") if self.last_seen_at is None else now - self.last_seen_at
        recent_target_seen = last_seen_age <= 1.0
        lock_prepare = bool(in_lock_zone)
        crosshair_center = (self.width / 2.0, self.height / 2.0)
        if bbox_center is None:
            pixel_error_x = float("inf")
            pixel_error_y = float("inf")
        else:
            pixel_error_x = float(bbox_center[0] - crosshair_center[0])
            pixel_error_y = float(bbox_center[1] - crosshair_center[1])

        lock_candidate = bool(distance_lock or instant_camera_lock or in_lock_zone)

        if self.locked:
            lock_state = "LOCKED"
        elif distance_lock or instant_camera_lock:
            self.locked = True
            if self.lock_started_at is None:
                self.lock_started_at = now
            lock_state = "LOCKED"
        elif lock_candidate:
            if self.lock_started_at is None:
                self.lock_started_at = now
            if now - self.lock_started_at >= self.lock_hold_seconds:
                self.locked = True
                lock_state = "LOCKED"
            else:
                lock_state = "LOCKING"
        else:
            self.lock_started_at = None
            lock_state = "VISIBLE" if target_in_view else "SEARCH"

        state = {
            "projected_in_view": projected_in_view,
            "target_in_view": target_in_view,
            "bbox_center": bbox_center,
            "projected_bbox_center": projected_bbox_center,
            "lock_state": lock_state,
            "camera_lock": self.locked,
            "distance_lock": distance_lock,
            "in_lock_zone": in_lock_zone,
            "bbox_in_lock_box": bbox_in_lock_box,
            "instant_camera_lock": instant_camera_lock,
            "depth_visible": depth_visible,
            "depth_valid": depth_valid,
            "target_occluded": target_occluded,
            "camera_depth": depth_state.get("camera_depth", float("inf")),
            "expected_target_depth": depth_state.get("expected_target_depth", float("inf")),
            "depth_error": self.last_depth_error,
            "screen_x": projection.get("screen_x", float("inf")),
            "screen_y": projection.get("screen_y", float("inf")),
            "crosshair_center": crosshair_center,
            "pixel_error_x": pixel_error_x,
            "pixel_error_y": pixel_error_y,
            "relative_world": projection.get("relative_world"),
            "camera_relative": projection.get("camera_relative"),
            "camera_world_pos": projection.get("camera_world_pos"),
            "camera_projection_source": projection.get("projection_source", "vehicle_yaw"),
            "camera_yaw": projection.get("camera_yaw", float("nan")),
            "vehicle_yaw": projection.get("vehicle_yaw", float("nan")),
            "lock_prepare": lock_prepare,
            "recent_target_seen": recent_target_seen,
            "last_seen_age": last_seen_age,
            "camera_error": self.last_error,
            "projection": projection,
        }
        self._draw(state, distance)
        return state

    def force_locked(self):
        self.locked = True
        self.lock_started_at = time.monotonic()

    def _in_lock_zone(self, target_in_view, bbox_center, distance):
        if not target_in_view or bbox_center is None:
            return False
        cx, cy = bbox_center
        max_dx = self.width * self.center_fraction
        max_dy = self.height * self.center_fraction
        return (
            abs(cx - self.width / 2.0) <= max_dx
            and abs(cy - self.height / 2.0) <= max_dy
            and float(distance) <= self.lock_distance
        )

    def _bbox_in_lock_box(self, target_in_view, bbox_center):
        if not target_in_view or bbox_center is None:
            return False
        cx, cy = bbox_center
        max_dx = self.width * self.center_fraction
        max_dy = self.height * self.center_fraction
        return (
            abs(cx - self.width / 2.0) <= max_dx
            and abs(cy - self.height / 2.0) <= max_dy
        )

    def _depth_visibility(self, projection, bbox_center, distance):
        projected_in_view = bool(projection.get("target_in_view", False))
        if not projected_in_view or bbox_center is None:
            self.last_depth_error = ""
            return {
                "depth_visible": False,
                "camera_depth": float("inf"),
                "expected_target_depth": float("inf"),
            }

        expected_depth = float(projection.get("forward", distance) or distance)
        if not math.isfinite(expected_depth) or expected_depth <= 0.0:
            expected_depth = float(distance)

        camera_depth = self._sample_depth_at(bbox_center)
        if camera_depth is None or not math.isfinite(camera_depth):
            return {
                "depth_visible": False,
                "camera_depth": float("inf"),
                "expected_target_depth": expected_depth,
            }

        tolerance = max(2.0, min(4.0, expected_depth * 0.35))
        depth_visible = abs(camera_depth - expected_depth) <= tolerance
        return {
            "depth_visible": bool(depth_visible),
            "camera_depth": float(camera_depth),
            "expected_target_depth": float(expected_depth),
        }

    def _sample_depth_at(self, bbox_center):
        try:
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest(
                        self.camera_name,
                        airsim.ImageType.DepthPerspective,
                        True,
                        False,
                    )
                ],
                vehicle_name=self.chaser_name,
            )
        except Exception as exc:
            self.last_depth_error = str(exc)
            return None

        if not responses:
            self.last_depth_error = "depth response missing"
            return None

        response = responses[0]
        width = int(getattr(response, "width", 0) or 0)
        height = int(getattr(response, "height", 0) or 0)
        data = getattr(response, "image_data_float", None)
        data_len = 0 if data is None else len(data)
        if width <= 0 or height <= 0 or data_len <= 0:
            self.last_depth_error = "depth image empty"
            return None

        depth = np.asarray(data, dtype=np.float32)
        if depth.size != width * height:
            self.last_depth_error = f"depth size mismatch {depth.size}!={width * height}"
            return None

        depth = depth.reshape((height, width))
        cx, cy = bbox_center
        px = int(round(float(cx) * width / max(1.0, float(self.width))))
        py = int(round(float(cy) * height / max(1.0, float(self.height))))
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))

        radius_x = max(2, int(round(self.depth_sample_radius * width / max(1.0, float(self.width)))))
        radius_y = max(2, int(round(self.depth_sample_radius * height / max(1.0, float(self.height)))))
        x0 = max(0, px - radius_x)
        x1 = min(width, px + radius_x + 1)
        y0 = max(0, py - radius_y)
        y1 = min(height, py + radius_y + 1)
        region = depth[y0:y1, x0:x1]
        valid = region[np.isfinite(region) & (region > 0.1) & (region < 1000.0)]
        if valid.size == 0:
            self.last_depth_error = "depth region has no valid values"
            return None

        self.last_depth_error = ""
        return float(np.percentile(valid, 10.0))

    def _project_target(self, chaser_pos, target_pos):
        try:
            camera_pos, camera_rotation, source, vehicle_yaw, camera_yaw = self._camera_world_pose(chaser_pos)
            dx = float(target_pos.x_val - camera_pos[0])
            dy = float(target_pos.y_val - camera_pos[1])
            dz = float(target_pos.z_val - camera_pos[2])
            delta_world = np.array([dx, dy, dz], dtype=np.float64)
            camera_relative = camera_rotation.T.dot(delta_world)
            forward = float(camera_relative[0])
            right = float(camera_relative[1])
            down = float(camera_relative[2])

            if forward <= 0.25:
                return {
                    "target_in_view": False,
                    "bbox_center": None,
                    "forward": forward,
                    "right": right,
                    "down": down,
                    "relative_world": (dx, dy, dz),
                    "camera_relative": (forward, right, down),
                    "camera_world_pos": camera_pos,
                    "projection_source": source,
                    "camera_yaw": camera_yaw,
                    "vehicle_yaw": vehicle_yaw,
                }

            focal = self.width / (2.0 * math.tan(math.radians(self.fov_degrees) / 2.0))
            px = self.width / 2.0 + (right / forward) * focal
            py = self.height / 2.0 + (down / forward) * focal
            target_in_view = 0.0 <= px < self.width and 0.0 <= py < self.height
            bbox_size = max(24.0, min(160.0, 260.0 / max(forward, 0.5)))
            return {
                "target_in_view": target_in_view,
                "bbox_center": (px, py) if target_in_view else None,
                "screen_x": px,
                "screen_y": py,
                "bbox_size": bbox_size,
                "forward": forward,
                "right": right,
                "down": down,
                "relative_world": (dx, dy, dz),
                "camera_relative": (forward, right, down),
                "camera_world_pos": camera_pos,
                "projection_source": source,
                "camera_yaw": camera_yaw,
                "vehicle_yaw": vehicle_yaw,
            }
        except Exception as exc:
            self.last_error = str(exc)
            return {"target_in_view": False, "bbox_center": None, "error": str(exc)}

    def _camera_world_pose(self, chaser_pos):
        vehicle_pose = self.client.simGetObjectPose(self.chaser_name)
        vehicle_pos = self._vector_tuple(vehicle_pose.position)
        vehicle_rotation = self._rotation_matrix(vehicle_pose.orientation)
        vehicle_yaw = self._yaw_from_quaternion(vehicle_pose.orientation)

        try:
            camera_info = self.client.simGetCameraInfo(
                self.camera_name,
                vehicle_name=self.chaser_name,
            )
        except Exception:
            return vehicle_pos, vehicle_rotation, "vehicle_pose_fallback", vehicle_yaw, vehicle_yaw

        camera_pose = camera_info.pose
        camera_pos = self._vector_tuple(camera_pose.position)
        camera_rotation = self._rotation_matrix(camera_pose.orientation)
        camera_yaw = self._yaw_from_quaternion(camera_pose.orientation)

        distance_to_vehicle = self._tuple_distance(camera_pos, vehicle_pos)
        camera_norm = math.sqrt(camera_pos[0] * camera_pos[0] + camera_pos[1] * camera_pos[1] + camera_pos[2] * camera_pos[2])
        vehicle_norm = math.sqrt(vehicle_pos[0] * vehicle_pos[0] + vehicle_pos[1] * vehicle_pos[1] + vehicle_pos[2] * vehicle_pos[2])
        camera_pose_is_relative = distance_to_vehicle > 20.0 and camera_norm < max(20.0, vehicle_norm * 0.25)

        if camera_pose_is_relative:
            camera_offset_world = vehicle_rotation.dot(np.array(camera_pos, dtype=np.float64))
            camera_world_pos = (
                float(vehicle_pos[0] + camera_offset_world[0]),
                float(vehicle_pos[1] + camera_offset_world[1]),
                float(vehicle_pos[2] + camera_offset_world[2]),
            )
            camera_world_quat = self._quat_multiply(vehicle_pose.orientation, camera_pose.orientation)
            return (
                camera_world_pos,
                self._rotation_matrix(camera_world_quat),
                "camera_info_relative",
                vehicle_yaw,
                self._yaw_from_quaternion(camera_world_quat),
            )

        return camera_pos, camera_rotation, "camera_info_world", vehicle_yaw, camera_yaw

    def _vector_tuple(self, vector):
        return (float(vector.x_val), float(vector.y_val), float(vector.z_val))

    def _tuple_distance(self, first, second):
        dx = float(first[0]) - float(second[0])
        dy = float(first[1]) - float(second[1])
        dz = float(first[2]) - float(second[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _quat_values(self, quat):
        return (
            float(quat.w_val),
            float(quat.x_val),
            float(quat.y_val),
            float(quat.z_val),
        )

    def _rotation_matrix(self, quat):
        w, x, y, z = self._quat_values(quat)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm <= 1e-9:
            return np.eye(3, dtype=np.float64)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _quat_multiply(self, first, second):
        aw, ax, ay, az = self._quat_values(first)
        bw, bx, by, bz = self._quat_values(second)
        return airsim.Quaternionr(
            ax * bw + aw * bx + ay * bz - az * by,
            ay * bw + aw * by + az * bx - ax * bz,
            az * bw + aw * bz + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    def _yaw_from_quaternion(self, quat):
        _, _, yaw = airsim.to_eularian_angles(quat)
        return float(yaw)

    def _camera_yaw(self):
        vehicle_yaw = self._chaser_yaw()
        try:
            camera_info = self.client.simGetCameraInfo(
                self.camera_name,
                vehicle_name=self.chaser_name,
            )
            _, _, camera_yaw = airsim.to_eularian_angles(camera_info.pose.orientation)
            camera_yaw = float(camera_yaw)
            # Some AirSim builds report camera yaw as a vehicle-relative zero.
            # If so, vehicle yaw is the correct world camera yaw for a front camera.
            if abs(camera_yaw) < 1e-4 and abs(vehicle_yaw) > 1e-4:
                return vehicle_yaw, "vehicle_yaw_camera_zero", vehicle_yaw
            return camera_yaw, "camera_info", vehicle_yaw
        except Exception:
            return vehicle_yaw, "vehicle_yaw_fallback", vehicle_yaw

    def _chaser_yaw(self):
        state = self.client.getMultirotorState(vehicle_name=self.chaser_name)
        _, _, yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
        return float(yaw)

    def _camera_frame(self):
        if not self.show_camera:
            return None
        try:
            image_response = self.client.simGetImage(
                self.camera_name,
                airsim.ImageType.Scene,
                vehicle_name=self.chaser_name,
            )
            if image_response is None:
                self.last_error = "simGetImage returned None"
                return None
            image_array = np.frombuffer(image_response, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is None:
                self.last_error = "cv2.imdecode failed"
                return None
            self.last_error = ""
            return frame
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def _draw(self, state, distance):
        if not self.show_camera:
            return

        frame = self._camera_frame()
        if frame is None:
            if getattr(self, "_last_frame", None) is not None:
                frame = self._last_frame.copy()
            else:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (self.width, self.height))
            self._last_frame = frame.copy()


        center = (self.width // 2, self.height // 2)
        cv2.drawMarker(frame, center, (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
        lock_radius_x = int(self.width * self.center_fraction)
        lock_radius_y = int(self.height * self.center_fraction)
        cv2.rectangle(
            frame,
            (center[0] - lock_radius_x, center[1] - lock_radius_y),
            (center[0] + lock_radius_x, center[1] + lock_radius_y),
            (255, 255, 255),
            1,
        )

        projection = state.get("projection", {})
        bbox_center = state.get("bbox_center")
        lock_state = state.get("lock_state", "SEARCH")

        if lock_state == "LOCKED":
            color = (0, 0, 255)
        elif lock_state == "LOCKING":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        if bbox_center is not None:
            cx, cy = int(bbox_center[0]), int(bbox_center[1])
            size = int(projection.get("bbox_size", 48))
            half = size // 2
            cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)

        cv2.putText(frame, f"{lock_state}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.putText(frame, f"distance={float(distance):.2f}m", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        if lock_state == "LOCKED":
            font = cv2.FONT_HERSHEY_SIMPLEX
            lines = ("LOCKED", "DUSMANIN IMHASI BASARILI", "TARGET DUSURULDU")

            scale = 1.25
            thickness = 4
            y = max(110, self.height // 2 - 28)
            for line in lines:
                (text_w, text_h), _ = cv2.getTextSize(line, font, scale, thickness)
                origin = ((self.width - text_w) // 2, y)
                cv2.putText(frame, line, (origin[0] + 4, origin[1] + 4), font, scale, (0, 0, 0), thickness + 4)
                cv2.putText(frame, line, (origin[0] + 2, origin[1] + 2), font, scale, (0, 255, 255), thickness + 2)
                cv2.putText(frame, line, origin, font, scale, (0, 0, 255), thickness)
                y += text_h + 22

        if self.last_error:
            cv2.putText(frame, f"camera: {self.last_error[:60]}", (18, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)

        try:
            # Bloklamasın: event pump için kısa waitKey
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)
        except Exception as exc:
            self.last_error = str(exc)
            # show_camera'ı kapatmayalım; bir kerelik display hatasında donmayı önlemek için skip
            print(f"[WARN] Camera HUD frame display failed (skip): {exc}", flush=True)
