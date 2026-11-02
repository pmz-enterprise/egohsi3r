#!/usr/bin/env python3
"""EgoHSI3R sequence inference, contact visualization, and NPZ export."""
import argparse
import ast
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/egohsi3r-matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/egohsi3r-ultralytics")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
from natsort import natsorted
import numpy as np
from PIL import Image, ImageOps
import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", help="Checkpoint path; default: weights/egohsi3r.pth or EGOHSI3R_MODEL_PATH.")
    parser.add_argument("--seq_path", help="Video, image directory, or exported sequence.")
    parser.add_argument("--results_dir", help="Replay an earlier --save export without loading a model.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512, help="Long edge, divisible by 16.")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Limit selected frames; default: 0 (all frames).")
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--reset_interval", type=int, default=20,
                        help="Reset every N selected frames (default: 20); 0 disables resets.")
    ttt = parser.add_mutually_exclusive_group()
    ttt.add_argument("--use_ttt3r", action="store_true", default=True,
                     help="Enable TTT3R state updates (default: enabled).")
    ttt.add_argument("--no_ttt3r", action="store_false", dest="use_ttt3r",
                     help="Disable TTT3R state updates.")
    parser.add_argument("--output_dir", default=str(ROOT / "outputs/demo"))
    parser.add_argument("--output_target", help="Parent for temporary images, removed after inference.")
    parser.add_argument("--save", action="store_true", help="Export one NPZ per frame.")
    parser.add_argument("--render", action="store_true", help="Export mesh/contact overlay PNGs.")
    parser.add_argument("--render_video", action="store_true", help="Export overlays and an MP4.")
    parser.add_argument("--fps", type=float, help="Playback FPS; otherwise video FPS / subsample.")
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vis_threshold", type=float, default=2.0)
    parser.add_argument("--downsample_factor", type=int, default=6)
    parser.add_argument("--viewer_mode", choices=("Hybrid", "3D", "4D", "Accumulate"), default="3D")
    parser.add_argument("--point_size", type=float, default=0.0001)
    parser.add_argument("--camera_size", type=float, default=0.05)
    parser.add_argument("--camera_downsample", type=int, default=20)
    parser.add_argument("--smpl_downsample", "--hand_downsample", type=int, default=20,
                        help="Historical hand mesh frame stride in the viewer.")
    parser.add_argument("--follow_camera", action="store_true",
                        help="Follow the reconstructed camera with the browser viewpoint.")
    parser.add_argument("--contact_mode", choices=("probability", "binary", "off"), default="binary")
    parser.add_argument("--contact_threshold", type=float, default=0.5)
    parser.add_argument("--hand_source", choices=("detector", "annotations"), default="detector",
                        help="annotations uses exported boxes/masks/intrinsics, never GT hand poses.")
    parser.add_argument("--detector_path", default=os.environ.get(
        "HAND3R_DETECTOR_PATH", str(ROOT / "weights/detector.pt")))
    parser.add_argument("--detector_conf", type=float, default=0.25)
    parser.add_argument("--bbox_padding", type=float, default=0.25,
                        help="Padding per side relative to detector box size (default: 0.25).")
    parser.add_argument("--track", action="store_true", help="Enable ByteTrack (requires lap).")
    args = parser.parse_args(argv)
    if args.results_dir:
        if args.model_path or args.seq_path:
            parser.error("Use --results_dir alone, without --model_path or --seq_path")
    else:
        args.model_path = args.model_path or os.environ.get(
            "EGOHSI3R_MODEL_PATH", str(ROOT / "weights/egohsi3r.pth"))
        if not args.seq_path:
            parser.error("Inference requires --seq_path")
    if args.size <= 0 or args.size % 16:
        parser.error("--size must be a positive multiple of 16")
    if args.subsample < 1 or args.downsample_factor < 1:
        parser.error("--subsample and --downsample_factor must be positive")
    if args.camera_downsample < 1 or args.smpl_downsample < 1:
        parser.error("Camera and hand downsampling factors must be positive")
    if not 0.0001 <= args.point_size <= 0.1 or not 0.005 <= args.camera_size <= 0.5:
        parser.error("Point size must be in [0.0001, 0.1]; camera size in [0.005, 0.5]")
    if not np.isfinite(args.vis_threshold) or args.vis_threshold < 0:
        parser.error("--vis_threshold must be finite and nonnegative")
    if args.max_frames < 0 or args.reset_interval < 0:
        parser.error("--max_frames and --reset_interval must be nonnegative")
    if args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and positive")
    if not 0 <= args.contact_threshold <= 1 or not 0 <= args.detector_conf <= 1:
        parser.error("Contact and detector thresholds must be in [0, 1]")
    if not np.isfinite(args.bbox_padding) or args.bbox_padding < 0:
        parser.error("--bbox_padding must be finite and nonnegative")
    if not 1 <= args.port <= 65535:
        parser.error("Invalid viewer port")
    return args


def parse_seq_path(path, temporary_dir, subsample=1, max_frames=0):
    """Select frames before decoding/resizing; return (source index, path) pairs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        folder = path / "rgb" if (path / "rgb").is_dir() else path
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = natsorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions])
        selected = list(enumerate(files))[::subsample]
        if max_frames:
            selected = selected[:max_frames]
        fps = 30.0 / subsample
    else:
        cap = cv2.VideoCapture(str(path))
        selected = []
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = (fps if np.isfinite(fps) and fps > 0 else 30.0) / subsample
            index = 0
            while cap.grab():
                if index % subsample == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        raise ValueError(f"Cannot decode video frame {index}")
                    frame_path = Path(temporary_dir) / f"source_{index:08d}.png"
                    if not cv2.imwrite(str(frame_path), frame):
                        raise OSError(f"Cannot write {frame_path}")
                    selected.append((index, frame_path))
                    if max_frames and len(selected) >= max_frames:
                        break
                index += 1
        finally:
            cap.release()
    if not selected:
        raise ValueError(f"No supported images found: {path}")
    return selected, fps


def checkpoint_model_options(expression):
    """Read the saved constructor without evaluating arbitrary Python code."""
    tree = ast.parse(expression, mode="eval").body
    if not (isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name)
            and tree.func.id == "ARCroco3DStereo" and len(tree.args) == 1 and not tree.keywords):
        raise ValueError("Expected a Hand3R ARCroco3DStereo training checkpoint")
    config = tree.args[0]
    if not (isinstance(config, ast.Call) and isinstance(config.func, ast.Name)
            and config.func.id == "ARCroco3DStereoConfig" and not config.args):
        raise ValueError("Unsupported saved model configuration")

    class Infinity(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "inf":
                return ast.copy_location(ast.Constant(float("inf")), node)
            return node

    options = {}
    for keyword in config.keywords:
        if keyword.arg is None:
            raise ValueError("Expanded model kwargs are not supported")
        options[keyword.arg] = ast.literal_eval(Infinity().visit(keyword.value))
    if "+mano" not in options.get("output_mode", ""):
        raise ValueError("This demo requires a Hand3R checkpoint with MANO outputs")
    return options


def load_demo_model(path, device):
    from dust3r.hand3r import ARCroco3DStereo, ARCroco3DStereoConfig

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    expression = saved_args["model"] if hasattr(saved_args, "__getitem__") else saved_args.model
    options = checkpoint_model_options(expression)
    model = ARCroco3DStereo(ARCroco3DStereoConfig(**options))
    state = checkpoint["model"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    # Inference must never silently leave contact weights randomly initialized.
    model.load_state_dict(state, strict=True)
    info = dict(checkpoint=str(Path(path).resolve()), epoch=checkpoint.get("epoch"),
                best_so_far=checkpoint.get("best_so_far"),
                contact_head=options.get("contact_head", False),
                contact_feedback=options.get("contact_feedback", False),
                model=expression)
    del state, checkpoint
    model.gradient_checkpointing = False
    if hasattr(model, "hand_backbone"):
        model.hand_backbone.use_checkpoint = False
    return model.to(device).eval(), info


def prepare_input(selected, args, temporary_dir):
    from preprocessing import _long_edge_resolution, resize_and_pad_frame
    from dust3r.mano_model import MANOModel

    images, boxes, masks, intrinsics, detector_files = [], [], [], [], []
    for ordinal, (source_index, path) in enumerate(selected):
        with Image.open(path) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        height, width = rgb.shape[:2]
        box = np.zeros((2, 4), np.float32)
        mask = np.zeros(2, bool)
        camera = np.eye(3, dtype=np.float32)
        if args.hand_source == "annotations":
            sequence = path.parent.parent
            with (sequence / "meta.json").open() as stream:
                meta = json.load(stream)
            order = meta.get("side_order", ["right", "left"])
            indices = [order.index(side) for side in ("right", "left")]
            with (sequence / "mano" / f"{path.stem}.pkl").open("rb") as stream:
                annotation = pickle.load(stream)
                box = annotation["bbox"][indices].astype(np.float32)
                mask = annotation["mano_mask"][indices].astype(bool)
            with np.load(sequence / "cam" / f"{path.stem}.npz") as calibration:
                camera = calibration["intrinsics"].astype(np.float32)
        resolution = _long_edge_resolution((width, height), args.size, 16)
        rgb, _, camera, box, _, _ = resize_and_pad_frame(
            rgb, np.zeros((height, width), np.float32), camera, box, resolution)
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
        if images and image.shape != images[0].shape:
            raise ValueError("All selected frames must have the same padded resolution")
        images.append(image)
        boxes.append(box)
        masks.append(mask)
        intrinsics.append(camera)
        if args.hand_source == "detector":
            target = Path(temporary_dir) / f"prepared_{ordinal:08d}.png"
            if not cv2.imwrite(str(target), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Cannot write {target}")
            detector_files.append(str(target))
    if args.hand_source == "detector":
        from hand_detector import detect_track, parse_tracks
        _, tracks = detect_track(detector_files, thresh=args.detector_conf,
                                 detector_path=args.detector_path, device=args.device,
                                 tracker="bytetrack.yaml" if args.track else None)
        masks, boxes = parse_tracks(tracks, num_frames=len(images),
                                   img_h=images[0].shape[-2], img_w=images[0].shape[-1],
                                   padding=args.bbox_padding, max_gap=5, smooth_window=3)
    views, hand_count = [], 0
    for i, (image, box, mask) in enumerate(zip(images, boxes, masks)):
        box = np.asarray(box, np.float32)
        mask = np.asarray(mask, bool) & np.isfinite(box).all(axis=1)
        mask &= ((box[:, 2:] - box[:, :2]) > 1).all(axis=1)
        box[~mask] = 0
        bbox, mano_mask = torch.from_numpy(box).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)
        height, width = image.shape[-2:]
        reset = bool(args.reset_interval and (i + 1) % args.reset_interval == 0 and i + 1 < len(images))
        view = dict(img=image, ray_map=torch.full((1, 6, height, width), torch.nan),
                    true_shape=torch.tensor([[height, width]]), idx=i, instance=str(selected[i][1]),
                    camera_pose=torch.eye(4).unsqueeze(0), img_mask=torch.tensor([True]),
                    ray_mask=torch.tensor([False]), update=torch.tensor([True]),
                    reset=torch.tensor([reset]), mano_mask=mano_mask, bbox=bbox,
                    source_frame_index=torch.tensor([selected[i][0]]),
                    resized_crop=MANOModel.process_hand_crops(None, image, bbox, mano_mask))
        if args.hand_source == "annotations":
            view["camera_intrinsics"] = torch.from_numpy(intrinsics[i]).unsqueeze(0)
        hand_count += int(mano_mask.sum())
        views.append(view)
        if reset:
            views.append(dict(view, reset=torch.tensor([False])))
    print(f"Prepared {len(images)} frames; detected/annotated hand slots: {hand_count}", flush=True)
    return views


def contact_colors(probabilities, side, mode="probability", threshold=0.5):
    # Saturated side colors keep the mesh distinct from skin and scene points.
    base = np.array((0, 255, 255) if side == 0 else (255, 40, 230), dtype=np.uint8)
    if probabilities is None or mode == "off":
        return np.tile(base, (778, 1))
    probability = np.clip(np.nan_to_num(np.asarray(probabilities), nan=0.0), 0, 1)
    if mode == "binary":
        probability = (probability >= threshold).astype(np.float32)
    cold, hot = base.astype(np.float32), np.array([100, 255, 0])
    return np.round(cold + probability[:, None] * (hot - cold)).astype(np.uint8)


@torch.no_grad()
def prepare_output(outputs):
    from dust3r.mano_model import build_native_mano_layers, forward_native_mano
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.post_process import estimate_focal_knowing_depth

    layers = build_native_mano_layers("cpu")
    faces = [np.asarray(layer.faces, np.int32) for layer in layers]
    frames, base, skip_overlap = [], np.eye(4, dtype=np.float32), False
    if len(outputs["pred"]) != len(outputs["views"]):
        raise ValueError("Prediction/view count mismatch")
    for prediction, view in zip(outputs["pred"], outputs["views"]):
        if skip_overlap:
            skip_overlap = False
            continue
        c2w = base @ pose_encoding_to_camera(prediction["camera_pose"].cpu())[0].numpy()
        skip_overlap = bool(view["reset"].item())
        if skip_overlap:
            base = c2w
        points = prediction["pts3d_in_self_view"][0].cpu().float().numpy()
        height, width = points.shape[:2]
        if "camera_intrinsics" in view:
            camera = view["camera_intrinsics"][0].cpu().numpy()
        else:
            pp = torch.tensor([[width // 2, height // 2]], dtype=torch.float32)
            focal = float(estimate_focal_knowing_depth(torch.from_numpy(points)[None], pp,
                                                      focal_mode="weiszfeld")[0])
            camera = np.array([[focal, 0, pp[0, 0]], [0, focal, pp[0, 1]], [0, 0, 1]], np.float32)
        mask = view["mano_mask"][0].cpu().bool()
        sides = torch.where(mask)[0]
        pose, shape, translation = [prediction[key][0].cpu().float()
                                   for key in ("mano_rotmat", "mano_shape", "mano_transl")]
        vertices, joints = torch.zeros(2, 778, 3), torch.zeros(2, 21, 3)
        if len(sides):
            decoded = forward_native_mano(layers, pose[sides], shape[sides], translation[sides], sides)
            vertices[sides], joints[sides] = decoded["vertices"], decoded["joints"]
        vertices = vertices.numpy()
        logits = prediction.get("contact_logits")
        frame = dict(rgb=np.clip((view["img"][0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5,
                                0, 255).round().astype(np.uint8),
                     pts3d_camera=points, pts3d_world=points @ c2w[:3, :3].T + c2w[:3, 3],
                     confidence=prediction["conf_self"][0].cpu().float().numpy(),
                     intrinsics=camera, camera_to_world=c2w, hand_mask=mask.numpy(),
                     vertices_camera=vertices, vertices_world=vertices @ c2w[:3, :3].T + c2w[:3, 3],
                     joints_camera=joints.numpy(), mano_rotmat=pose.numpy(),
                     mano_shape=shape.numpy(), mano_transl=translation.numpy(),
                     bbox=view["bbox"][0].cpu().numpy(),
                     source_frame_index=np.asarray(view["source_frame_index"][0].item()),
                     contact_probability=None if logits is None else logits[0].cpu().float().sigmoid().numpy())
        frames.append(frame)
    if not frames:
        raise ValueError("Inference returned no frames")
    return frames, faces


def render_overlay(frame, faces, mode, threshold):
    import pyrender
    import trimesh

    rgb = frame["rgb"]
    if not frame["hand_mask"].any():
        return rgb.copy()
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.8, 0.8, 0.8])
    probability = frame["contact_probability"]
    for side in np.flatnonzero(frame["hand_mask"]):
        mesh = trimesh.Trimesh(vertices=frame["vertices_camera"][side], faces=faces[side],
                               vertex_colors=contact_colors(None if probability is None else probability[side],
                                                            side, mode, threshold), process=False)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    k = frame["intrinsics"]
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    scene.add(pyrender.IntrinsicsCamera(k[0, 0], k[1, 1], k[0, 2], k[1, 2]), pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=pose)
    renderer = pyrender.OffscreenRenderer(rgb.shape[1], rgb.shape[0])
    try:
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    output = rgb.copy()
    visible = depth > 0
    output[visible] = (0.85 * color[visible, :3] + 0.15 * rgb[visible]).astype(np.uint8)
    return output


def save_results(frames, faces, args, info, fps):
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(info, arguments=vars(args), num_frames=len(frames), fps=fps,
                    hand_order=["right", "left"],
                    contact_colors="right=bright cyan, left=bright magenta, contact=neon green",
                    source_frame_indices=[int(f["source_frame_index"]) for f in frames],
                    intrinsics_source=(info["intrinsics_source"] if args.results_dir else
                                       "annotation" if args.hand_source == "annotations" else "estimated"))
    with (folder / "run.json").open("w") as stream:
        json.dump(metadata, stream, indent=2, default=str)
    if args.save:
        for i, frame in enumerate(frames):
            arrays = {k: v for k, v in frame.items() if v is not None}
            if frame["contact_probability"] is not None:
                arrays["contact_prediction"] = ((frame["contact_probability"] >= args.contact_threshold)
                                                & frame["hand_mask"][:, None])
            np.savez_compressed(folder / f"frame_{i:06d}.npz", **arrays,
                                faces_right=faces[0], faces_left=faces[1])
    writer = None
    try:
        if args.render_video:
            height, width = frames[0]["rgb"].shape[:2]
            writer = cv2.VideoWriter(str(folder / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Cannot create overlay.mp4; check OpenCV video codec support")
        if args.render or args.render_video:
            for i, frame in enumerate(frames):
                overlay = render_overlay(frame, faces, args.contact_mode, args.contact_threshold)
                bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(folder / f"overlay_{i:06d}.png"), bgr):
                    raise OSError("Cannot write overlay PNG")
                if writer is not None:
                    writer.write(bgr)
    finally:
        if writer is not None:
            writer.release()
    print(f"Output: {folder.resolve()}", flush=True)


def load_results(folder, subsample=1, max_frames=0):
    folder = Path(folder)
    with (folder / "run.json").open() as stream:
        info = json.load(stream)
    if not info.get("arguments", {}).get("save"):
        raise ValueError("This run has no NPZ export; run inference with --save first")
    indices = list(range(info["num_frames"]))[::subsample]
    if max_frames:
        indices = indices[:max_frames]
    frames, faces = [], None
    for i in indices:
        with np.load(folder / f"frame_{i:06d}.npz", allow_pickle=False) as data:
            frame = dict(data)
        current_faces = [frame.pop("faces_right"), frame.pop("faces_left")]
        if faces is not None and any(not np.array_equal(a, b) for a, b in zip(faces, current_faces)):
            raise ValueError("Inconsistent MANO topology in exported frames")
        faces = current_faces
        frame.setdefault("contact_probability", None)
        frames.append(frame)
    if not frames:
        raise ValueError("No exported frames to replay")
    return frames, faces, info, info["fps"] / subsample


def show_viewer(frames, faces, args, fps):
    from hand3r_viewer import Hand3RSequenceViewer

    Hand3RSequenceViewer(frames, faces, args, fps, contact_colors).run()


def run_inference(args):
    if args.results_dir:
        frames, faces, info, fps = load_results(args.results_dir, args.subsample, args.max_frames)
        fps = args.fps or fps
        if args.save or args.render or args.render_video:
            if Path(args.output_dir).resolve() == Path(args.results_dir).resolve():
                raise ValueError("Replay exports must use a different --output_dir")
            save_results(frames, faces, args, info, fps)
        if not args.no_viewer:
            show_viewer(frames, faces, args, fps)
        return frames
    from dust3r.inference import inference_recurrent_lighter

    if not Path(args.model_path).is_file():
        raise FileNotFoundError(args.model_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu explicitly for a CPU test.")
    if args.hand_source == "detector" and not Path(args.detector_path).is_file():
        raise FileNotFoundError(f"Set --detector_path: {args.detector_path}")
    if args.hand_source == "annotations" and not Path(args.seq_path).is_dir():
        raise ValueError("--hand_source annotations requires an exported sequence directory")
    torch.manual_seed(42)
    np.random.seed(42)
    if args.output_target:
        Path(args.output_target).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="egohsi3r-inference-", dir=args.output_target) as temporary_dir:
        selected, fps = parse_seq_path(args.seq_path, temporary_dir, args.subsample, args.max_frames)
        fps = args.fps or fps
        views = prepare_input(selected, args, temporary_dir)
        print(f"Loading checkpoint: {args.model_path}", flush=True)
        model, info = load_demo_model(args.model_path, args.device)
        print(f"Contact head={info['contact_head']}, feedback={info['contact_feedback']}", flush=True)
        print(f"TTT3R={args.use_ttt3r}, reset_interval={args.reset_interval}", flush=True)
        start = time.monotonic()
        outputs, _ = inference_recurrent_lighter(views, model, args.device, use_ttt3r=args.use_ttt3r)
        frames, faces = prepare_output(outputs)
        print(f"Inference/decode: {time.monotonic() - start:.1f}s, {len(frames)} frames", flush=True)
        del model, outputs, views
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not info["contact_head"]:
        print("Checkpoint has no contact head; using hand-side colors.", flush=True)
        args.contact_mode = "off"
    save_results(frames, faces, args, info, fps)
    if not args.no_viewer:
        show_viewer(frames, faces, args, fps)
    return frames


if __name__ == "__main__":
    run_inference(parse_args())
