"""Sequence viewer for Hand3R scene, camera trajectory, hands, and contact."""
import threading
import time

import numpy as np
import trimesh
import viser
import viser.transforms as vtf


VIEW_MODES = ("Hybrid", "3D", "4D", "Accumulate")


def visible_frame_indices(mode, current, count, accumulate_start=0):
    if mode == "4D":
        return {current}, {current}
    if mode == "Hybrid":
        return set(range(count)), {current}
    if mode == "3D":
        return set(range(count)), set(range(count))
    if mode == "Accumulate":
        start = min(max(0, int(accumulate_start)), current)
        indices = set(range(start, current + 1))
        return indices, indices
    raise ValueError(f"Unknown view mode: {mode}")


def filtered_cloud(frame, threshold, stride):
    points = frame["pts3d_world"][::stride, ::stride]
    colors = frame["rgb"][::stride, ::stride]
    confidence = frame["confidence"][::stride, ::stride]
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(confidence) & (confidence >= threshold)
    return np.ascontiguousarray(points[valid], dtype=np.float32), np.ascontiguousarray(colors[valid])


def contact_mesh_glb(vertices, faces, colors, opacity=1.0):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)

    def material(tree):
        # Viser's GLB handle has no opacity property. Keep vertex colors and
        # encode transparency in the material when the requested style changes.
        index = len(tree.setdefault("materials", []))
        tree["materials"].append({
            "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, float(opacity)],
                                    "metallicFactor": 0, "roughnessFactor": 0.8},
            "alphaMode": "BLEND" if opacity < 1 else "OPAQUE",
            "doubleSided": True,
        })
        for item in tree["meshes"]:
            for primitive in item["primitives"]:
                primitive["material"] = index

    return trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh), tree_postprocessor=material)


class Hand3RSequenceViewer:
    def __init__(self, frames, faces, args, fps, color_fn, server=None):
        if not frames:
            raise ValueError("Cannot display an empty sequence")
        self.frames, self.faces, self.args = frames, faces, args
        self.color_fn = color_fn
        self.server = server if server is not None else viser.ViserServer(
            host=args.host, port=args.port, label="Hand3R")
        self.server.scene.set_up_direction("-y")
        self.lock = threading.RLock()
        self.clouds, self.meshes, self.mesh_styles, self.cameras = {}, {}, {}, {}
        self.current_camera = self.trajectory = None
        self._trajectory_range = None
        self._cloud_signature = None
        self.positions = np.stack([f["camera_to_world"][:3, 3] for f in frames])
        self._build_controls(fps)
        self.server.on_client_connect(self.connect_client)
        self.rebuild_clouds()
        self.rebuild_cameras()
        self.update()

    @property
    def current(self):
        return min(max(0, int(self.frame.value)), len(self.frames) - 1)

    def _build_controls(self, fps):
        gui = self.server.gui
        self.mode = gui.add_dropdown("View mode", options=VIEW_MODES, initial_value=self.args.viewer_mode)
        with gui.add_folder("Playback"):
            self.frame = gui.add_slider("Frame", min=0, max=max(1, len(self.frames) - 1), step=1,
                                       initial_value=0, disabled=len(self.frames) == 1)
            self.accumulate_start = gui.add_slider(
                "Accumulate start",
                min=0,
                max=max(1, len(self.frames) - 1),
                step=1,
                initial_value=0,
                disabled=len(self.frames) == 1,
            )
            self.previous = gui.add_button("Previous", icon="chevron-left")
            self.next = gui.add_button("Next", icon="chevron-right")
            self.playing = gui.add_checkbox("Play", initial_value=False)
            self.fps = gui.add_number("FPS", initial_value=min(120.0, max(0.1, fps)), min=0.1, max=120, step=1)
        with gui.add_folder("Point cloud"):
            self.show_scene = gui.add_checkbox("Show points", initial_value=True)
            self.confidence = gui.add_slider("Visibility threshold", min=0,
                                            max=max(50.0, self.args.vis_threshold), step=0.1,
                                            initial_value=self.args.vis_threshold)
            self.point_size = gui.add_slider("Point size", min=0.0001, max=0.1, step=0.0001,
                                            initial_value=self.args.point_size)
            self.point_stride = gui.add_number("Pixel stride", initial_value=self.args.downsample_factor,
                                              min=1, max=1000, step=1)
            self.scene_stride = gui.add_number("Scene frame stride", initial_value=1,
                                              min=1, max=max(1, len(self.frames)), step=1)
        with gui.add_folder("Cameras"):
            self.show_cameras = gui.add_checkbox("Show cameras", initial_value=True)
            self.show_trajectory = gui.add_checkbox("Show trajectory", initial_value=True)
            self.camera_stride = gui.add_number("Camera stride", initial_value=self.args.camera_downsample,
                                               min=1, max=max(200, len(self.frames)), step=1)
            self.camera_size = gui.add_slider("Camera size", min=0.005, max=0.5, step=0.005,
                                             initial_value=self.args.camera_size)
            self.camera_image = gui.add_checkbox("Camera image", initial_value=False)
            self.follow = gui.add_checkbox("Follow camera", initial_value=self.args.follow_camera)
            self.fit_scene_button = gui.add_button("Fit scene", icon="focus-centered")
            self.fit_hands_button = gui.add_button("Fit hands", icon="focus-centered")
            self.reset_up_button = gui.add_button("Reset up direction", icon="arrow-back-up")
        with gui.add_folder("Hands / Contact"):
            self.show_hands = gui.add_checkbox("Show hands", initial_value=True)
            self.hand_stride = gui.add_number("Hand frame stride", initial_value=self.args.smpl_downsample,
                                             min=1, max=max(200, len(self.frames)), step=1)
            self.contact_mode = gui.add_dropdown("Contact", options=("probability", "binary", "off"),
                                                initial_value=self.args.contact_mode)
            self.contact_threshold = gui.add_slider("Contact threshold", min=0, max=1, step=0.01,
                                                    initial_value=self.args.contact_threshold)
            self.opacity = gui.add_slider("Mesh opacity", min=0, max=1, step=0.05, initial_value=1.0)
            self.fade = gui.add_dropdown("History opacity", options=("None", "By time", "Age"), initial_value="None")
            self.min_opacity = gui.add_slider("Minimum opacity", min=0, max=1, step=0.05, initial_value=0.1)
            self.decay = gui.add_number("Decay frames", initial_value=10, min=1, step=1)
        for control in (self.frame, self.show_scene, self.scene_stride, self.show_cameras,
                        self.show_trajectory, self.camera_image, self.show_hands, self.hand_stride,
                        self.contact_mode, self.contact_threshold, self.opacity, self.fade,
                        self.min_opacity, self.decay, self.follow):
            control.on_update(self.update)
        # 切换模式时，相机采样的起点可能发生变化，因此重新创建相机。
        self.mode.on_update(self.rebuild_cameras)
        # 修改累计起始帧时，也要从新的起点重新创建相机。
        self.accumulate_start.on_update(self.rebuild_cameras)
        self.confidence.on_update(self.rebuild_clouds)
        self.point_stride.on_update(self.rebuild_clouds)
        self.point_size.on_update(self.update_point_size)
        self.camera_stride.on_update(self.rebuild_cameras)
        self.camera_size.on_update(self.update_camera_size)
        self.previous.on_click(lambda _: self.step(-1))
        self.next.on_click(lambda _: self.step(1))
        self.fit_scene_button.on_click(lambda event: self.fit_clients(event, hands=False))
        self.fit_hands_button.on_click(lambda event: self.fit_clients(event, hands=True))
        self.reset_up_button.on_click(self.reset_up)

    def step(self, delta):
        start = (
            int(self.accumulate_start.value)
            if self.mode.value == "Accumulate"
            else 0
        )
        start = min(max(0, start), len(self.frames) - 1)
        span = len(self.frames) - start
        current = max(self.current, start)
        self.frame.value = start + (current - start + delta) % span

    def rebuild_clouds(self, _=None):
        with self.lock, self.server.atomic():
            signature = (float(self.confidence.value), max(1, int(self.point_stride.value)))
            if signature == self._cloud_signature:
                return
            for handle in self.clouds.values():
                handle.remove()
            self.clouds.clear()
            for i, frame in enumerate(self.frames):
                points, colors = filtered_cloud(frame, *signature)
                if len(points):
                    self.clouds[i] = self.server.scene.add_point_cloud(
                        f"/scene/{i}", points=points, colors=colors, point_size=self.point_size.value,
                        visible=False)
            self._cloud_signature = signature
            self.update()

    def update_point_size(self, _=None):
        with self.lock, self.server.atomic():
            for handle in self.clouds.values():
                handle.point_size = self.point_size.value

    def camera_parameters(self, index):
        frame = self.frames[index]
        height, width = frame["rgb"].shape[:2]
        pose, k = frame["camera_to_world"], frame["intrinsics"]
        return dict(fov=float(2 * np.arctan(height / (2 * k[1, 1]))), aspect=width / height,
                    position=pose[:3, 3], wxyz=vtf.SO3.from_matrix(pose[:3, :3]).wxyz)

    def rebuild_cameras(self, _=None):
        with self.lock, self.server.atomic():
            for handle in self.cameras.values():
                handle.remove()
            self.cameras.clear()
            start = (
                int(self.accumulate_start.value)
                if self.mode.value == "Accumulate"
                else 0
            )
            start = min(max(0, start), len(self.frames) - 1)
            indices = set(
                range(start, len(self.frames), max(1, int(self.camera_stride.value)))
            )
            indices.add(len(self.frames) - 1)
            for i in sorted(indices):
                ratio = i / max(1, len(self.frames) - 1)
                color = tuple((np.array([45, 135, 210]) * (1 - ratio)
                               + np.array([220, 155, 50]) * ratio).astype(np.uint8))
                self.cameras[i] = self.server.scene.add_camera_frustum(
                    f"/cameras/history/{i}", **self.camera_parameters(i), scale=self.camera_size.value,
                    color=color, thickness=1.2, thickness_units="screen", visible=False)
            if self.current_camera is None:
                self.current_camera = self.server.scene.add_camera_frustum(
                    "/cameras/current", **self.camera_parameters(self.current), scale=self.camera_size.value,
                    color=(245, 75, 60), thickness=2.5, thickness_units="screen")
            self.update()

    def update_camera_size(self, _=None):
        with self.lock, self.server.atomic():
            for handle in self.cameras.values():
                handle.scale = self.camera_size.value
            if self.current_camera is not None:
                self.current_camera.scale = self.camera_size.value

    def mesh_opacity(self, index):
        factor = 1.0
        if self.fade.value == "By time":
            factor = self.min_opacity.value + (1 - self.min_opacity.value) * index / max(1, len(self.frames) - 1)
        elif self.fade.value == "Age":
            age = max(0, self.current - index)
            factor = 1 - min(1, age / max(1, self.decay.value)) * (1 - self.min_opacity.value)
        return round(float(self.opacity.value * factor), 3)

    def update_meshes(self, indices, anchor=0):
        desired = set()
        if self.show_hands.value:
            stride = max(1, int(self.hand_stride.value))
            for i in sorted(indices):
                if (i - anchor) % stride != 0 and i != self.current:
                    continue
                frame = self.frames[i]
                probability = frame["contact_probability"]
                style = (self.contact_mode.value, self.contact_threshold.value, self.mesh_opacity(i))
                for side in np.flatnonzero(frame["hand_mask"]):
                    key = (i, int(side))
                    desired.add(key)
                    if self.mesh_styles.get(key) != style:
                        colors = self.color_fn(None if probability is None else probability[side],
                                               side, style[0], style[1])
                        data = contact_mesh_glb(frame["vertices_world"][side], self.faces[side], colors, style[2])
                        if key in self.meshes:
                            self.meshes[key].remove()
                        self.meshes[key] = self.server.scene.add_glb(f"/hands/{i}/{side}", glb_data=data)
                        self.mesh_styles[key] = style
        for key, handle in self.meshes.items():
            handle.visible = key in desired

    def update(self, _=None):
        with self.lock, self.server.atomic():
            anchor = (
                int(self.accumulate_start.value)
                if self.mode.value == "Accumulate"
                else 0
            )
            anchor = min(max(0, anchor), self.current)

            scene_indices, hand_indices = visible_frame_indices(
                self.mode.value,
                self.current,
                len(self.frames),
                anchor,
            )
            stride = max(1, int(self.scene_stride.value))
            for i, handle in self.clouds.items():
                handle.visible = (self.show_scene.value and i in scene_indices
                                  and ((i - anchor) % stride == 0 or i == self.current))
            self.update_meshes(hand_indices, anchor)
            for i, handle in self.cameras.items():
                handle.visible = (self.show_cameras.value and not self.follow.value and self.mode.value != "4D"
                                  and i in scene_indices and i != self.current)
            if self.current_camera is not None:
                for name, value in self.camera_parameters(self.current).items():
                    setattr(self.current_camera, name, value)
                self.current_camera.image = self.frames[self.current]["rgb"] if self.camera_image.value else None
                self.current_camera.visible = self.show_cameras.value and not self.follow.value

            start = anchor if self.mode.value == "Accumulate" else 0
            end = self.current + 1 if self.mode.value == "Accumulate" else len(self.frames)
            trajectory_range = (start, end)

            if trajectory_range != self._trajectory_range:
                if self.trajectory is not None:
                    self.trajectory.remove()
                    self.trajectory = None

                if end - start > 1:
                    segments = np.stack(
                        [
                            self.positions[start:end - 1],
                            self.positions[start + 1:end],
                        ],
                        axis=1,
                    ).astype(np.float32)
                    self.trajectory = self.server.scene.add_line_segments(
                        "/cameras/trajectory",
                        points=segments,
                        colors=(50, 160, 175),
                        thickness=2,
                        thickness_units="screen",
                    )

                self._trajectory_range = trajectory_range
            if self.trajectory is not None:
                self.trajectory.visible = self.show_trajectory.value
            if self.follow.value:
                for client in self.server.get_clients().values():
                    self.follow_client(client)

    def follow_client(self, client):
        frame = self.frames[self.current]
        pose = frame["camera_to_world"]
        with client.atomic():
            client.camera.position = pose[:3, 3]
            client.camera.up_direction = -pose[:3, 1]
            client.camera.look_at = pose[:3, 3] + pose[:3, 2]
            client.camera.fov = self.camera_parameters(self.current)["fov"]

    def fit_client(self, client, hands=False):
        points = []
        if hands:
            frame = self.frames[self.current]
            points = [frame["vertices_world"][frame["hand_mask"]].reshape(-1, 3)]
        else:
            # Bound the fitting sample, not the displayed scene.
            for i in range(0, len(self.frames), max(1, len(self.frames) // 32)):
                sampled, _ = filtered_cloud(self.frames[i], self.confidence.value, max(1, int(self.point_stride.value)))
                points.append(sampled[::max(1, len(sampled) // 2000)])
        points = np.concatenate(points, axis=0) if points else self.positions
        points = points[np.isfinite(points).all(axis=-1)]
        if not len(points):
            points = self.positions[np.isfinite(self.positions).all(axis=-1)]
        if not len(points):
            points = np.zeros((1, 3))
        lower, upper = np.percentile(points, [5, 95], axis=0)
        center = (lower + upper) / 2
        distance = max(0.3, float(np.linalg.norm(upper - lower)) * 1.1)
        with client.atomic():
            client.camera.position = center + np.array([0, -0.2, -1]) * distance
            client.camera.up_direction = (0, -1, 0)
            client.camera.look_at = center
            client.camera.fov = np.deg2rad(60)

    def fit_clients(self, event=None, hands=False):
        self.follow.value = False
        clients = [event.client] if event is not None and event.client is not None else self.server.get_clients().values()
        for client in clients:
            self.fit_client(client, hands)

    def reset_up(self, event):
        if event.client is not None:
            event.client.camera.up_direction = (0, -1, 0)

    def connect_client(self, client):
        with self.lock:
            if self.follow.value:
                self.follow_client(client)
            else:
                self.fit_client(client)

    def run(self):
        print(f"Viewer: http://{self.args.host}:{self.server.get_port()} (Ctrl+C to stop)", flush=True)
        try:
            while True:
                time.sleep(1 / max(0.1, float(self.fps.value)))
                if self.playing.value and len(self.frames) > 1:
                    self.step(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()
