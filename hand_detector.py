"""Hand detection and native right/left box preparation."""
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

def _to_scalar(value, default=0):
    try:
        return np.array(value).reshape(-1)[0].item()
    except Exception:
        return default



def _clip_xyxy(box, img_w, img_h):
    box = np.asarray(box, dtype=np.float32).copy()
    box[0::2] = np.clip(box[0::2], 0, img_w - 1)
    box[1::2] = np.clip(box[1::2], 0, img_h - 1)
    if box[2] <= box[0]:
        box[2] = min(img_w - 1, box[0] + 1)
    if box[3] <= box[1]:
        box[3] = min(img_h - 1, box[1] + 1)
    return box



def _xyxy_iou(box, boxes):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-6)



def _select_target_box_dual_conf(clses, confs, bboxes, target_cls, iou_thr=0.7):
    target_mask = clses == target_cls
    if not target_mask.any():
        return None

    selected = int(np.argmax(np.where(target_mask, confs, -1.0)))
    other_mask = clses == (1 - target_cls)
    if other_mask.any():
        other_idx = np.where(other_mask)[0]
        overlaps = _xyxy_iou(bboxes[selected], bboxes[other_idx])
        if len(overlaps) > 0:
            best = int(np.argmax(overlaps))
            best_other_conf = float(confs[other_idx[best]])
            if overlaps[best] > iou_thr and best_other_conf > float(confs[selected]):
                return None

    return selected



def _select_frame_hand_slots(clses, confs, bboxes, iou_thr=0.7):
    selected = []
    for target_cls in (1, 0):  # Hand3R slot order: right first, left second.
        idx = _select_target_box_dual_conf(clses, confs, bboxes, target_cls, iou_thr=iou_thr)
        if idx is not None and idx not in selected:
            selected.append(idx)
    return selected



def detect_track(imgfiles, thresh=0.5, iou=0.55, tracker="bytetrack.yaml",
                 detector_path=None, device=None):
    detector_path = detector_path or os.environ.get(
        "HAND3R_DETECTOR_PATH",
        os.path.join(os.path.dirname(__file__), "weights", "detector.pt"),
    )
    if not os.path.isfile(detector_path):
        raise FileNotFoundError(f"Missing hand detector: {detector_path}")
    original_torch_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = torch_load_compat
        hand_det_model = YOLO(detector_path)
    finally:
        torch.load = original_torch_load

    boxes_ = []
    tracks = {}
    for t, imgpath in enumerate(tqdm(imgfiles)):
        img_cv2 = cv2.imread(imgpath)
        if img_cv2 is None:
            boxes_.append(np.zeros((0, 6), dtype=np.float32))
            continue

        with torch.no_grad():
            with torch.autocast("cuda", enabled=torch.cuda.is_available() and str(device) != "cpu"):
                track_kwargs = dict(
                    source=img_cv2,
                    conf=thresh,
                    iou=iou,
                    verbose=False,
                )
                if device is not None:
                    track_kwargs["device"] = device
                if tracker is not None:
                    track_kwargs["tracker"] = tracker
                    track_kwargs["persist"] = True
                    results = hand_det_model.track(**track_kwargs)
                else:
                    results = hand_det_model.predict(**track_kwargs)
                
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                handedness = results[0].boxes.cls.cpu().numpy()
                if not results[0].boxes.id is None:
                    track_id = results[0].boxes.id.cpu().numpy()
                else:
                    track_id = [-1] * len(boxes)

                if len(boxes) == 0:
                    boxes_.append(np.zeros((0, 6), dtype=np.float32))
                    continue

                boxes = np.hstack([boxes, confs[:, None], handedness[:, None]])
                boxes_.append(boxes.astype(np.float32))

                selected = _select_frame_hand_slots(
                    handedness.astype(int),
                    confs,
                    boxes[:, :4],
                    iou_thr=0.7,
                )
                for idx in selected:
                    if track_id[idx] == -1:
                        if handedness[idx] > 0:
                            id = int(10000 + t)
                        else:
                            id = int(5000 + t)
                    else:
                        id = int(track_id[idx])
                    subj = dict()
                    subj['frame'] = t 
                    subj['det'] = True
                    subj['det_box'] = boxes[[idx]]
                    subj['det_handedness'] = handedness[[idx]]
                    
                    if id in tracks:
                        tracks[id].append(subj)
                    else:
                        tracks[id] = [subj]
    tracks = np.array(tracks, dtype=object)
    boxes_ = np.array(boxes_, dtype=object)

    return boxes_, tracks



def _as_tracks_dict(tracks):
    if isinstance(tracks, np.ndarray):
        if tracks.dtype == object and tracks.size == 1:
            tracks = tracks.item()
        elif tracks.size == 0:
            tracks = {}

    if tracks is None:
        return {}
    if isinstance(tracks, dict):
        return tracks

    try:
        return dict(tracks)
    except Exception as exc:
        raise ValueError('Unsupported tracks type') from exc



def _fill_short_bbox_gaps(hand_masks, bboxes, max_gap=5):
    if max_gap <= 0:
        return hand_masks, bboxes

    for hand_idx in range(hand_masks.shape[1]):
        valid = np.where(hand_masks[:, hand_idx])[0]
        if len(valid) < 2:
            continue

        for left, right in zip(valid[:-1], valid[1:]):
            gap = right - left - 1
            if gap <= 0 or gap > max_gap:
                continue

            start_box = bboxes[left, hand_idx].copy()
            end_box = bboxes[right, hand_idx].copy()
            for step in range(1, gap + 1):
                alpha = step / (gap + 1)
                frame = left + step
                bboxes[frame, hand_idx] = (1.0 - alpha) * start_box + alpha * end_box
                hand_masks[frame, hand_idx] = True

    return hand_masks, bboxes



def _smooth_bbox_tracks(hand_masks, bboxes, img_h, img_w, aspect_ratio, smooth_window=3):
    if smooth_window <= 1:
        return bboxes

    half = smooth_window // 2
    smoothed = bboxes.copy()
    for hand_idx in range(hand_masks.shape[1]):
        valid = np.where(hand_masks[:, hand_idx])[0]
        if len(valid) < 3:
            continue

        breaks = np.where(np.diff(valid) > 1)[0] + 1
        chunks = np.split(valid, breaks)
        for chunk in chunks:
            if len(chunk) < 3:
                continue
            src = smoothed[:, hand_idx].copy()
            for frame in chunk:
                lo = max(chunk[0], frame - half)
                hi = min(chunk[-1], frame + half)
                win = chunk[(chunk >= lo) & (chunk <= hi)]
                if len(win) < 2:
                    continue
                box = np.median(src[win], axis=0)
                smoothed[frame, hand_idx] = get_bbox_from_kps(
                    box,
                    img_h=img_h,
                    img_w=img_w,
                    aspect_ratio=aspect_ratio,
                    padding=0.0,
                )

    return smoothed.astype(np.float32)



def parse_tracks(
    tracks,
    num_frames=None,
    img_h=384,
    img_w=512,
    aspect_ratio=3/4,
    padding=0.25,
    max_gap=5,
    smooth_window=3,
):
    """
    Convert YOLO hand tracks into per-frame Hand3R masks and boxes.

    Output convention:
    - hand_masks: [num_frames, 2], index 0=right hand, index 1=left hand
    - bboxes: [num_frames, 2, 4], xyxy in resized-image coordinates

    The parser keeps the highest-confidence detection for each frame/side, expands
    boxes to the crop aspect ratio, clamps to image bounds, fills short missed
    detections, and smooths small frame-to-frame jitter.
    """
    tracks = _as_tracks_dict(tracks)

    if num_frames is None:
        max_frame = -1
        for subjects in tracks.values():
            for subj in subjects:
                max_frame = max(max_frame, int(subj.get('frame', -1)))
        num_frames = max_frame + 1 if max_frame >= 0 else 0

    hand_masks = np.zeros((num_frames, 2), dtype=bool)
    bboxes = np.zeros((num_frames, 2, 4), dtype=np.float32)
    bboxes_conf = np.full((num_frames, 2), -np.inf, dtype=np.float32)

    for subjects in tracks.values():
        for subj in subjects:
            if not subj.get('det', False):
                continue

            frame = int(subj.get('frame', -1))
            if frame < 0 or frame >= num_frames:
                continue

            det_box = subj.get('det_box')
            if det_box is None:
                continue
            arr = np.asarray(det_box).reshape(-1)
            if arr.size < 4:
                continue

            handedness = subj.get('det_handedness')
            if handedness is None and arr.size >= 6:
                handedness = arr[5]
            hand_idx = 0 if float(_to_scalar(handedness, 0)) > 0 else 1

            conf = float(arr[4]) if arr.size >= 5 else 1.0
            if conf < bboxes_conf[frame, hand_idx]:
                continue

            raw_box = _clip_xyxy(arr[:4], img_w=img_w, img_h=img_h)
            crop_box = get_bbox_from_kps(
                raw_box,
                img_h=img_h,
                img_w=img_w,
                aspect_ratio=aspect_ratio,
                padding=padding,
            )
            bboxes[frame, hand_idx] = np.asarray(crop_box, dtype=np.float32)
            bboxes_conf[frame, hand_idx] = conf
            hand_masks[frame, hand_idx] = True

    hand_masks, bboxes = _fill_short_bbox_gaps(hand_masks, bboxes, max_gap=max_gap)
    bboxes = _smooth_bbox_tracks(
        hand_masks,
        bboxes,
        img_h=img_h,
        img_w=img_w,
        aspect_ratio=aspect_ratio,
        smooth_window=smooth_window,
    )

    return hand_masks, bboxes



def get_bbox_from_kps(input_bbox, img_h=384, img_w=512, aspect_ratio=3/4, padding=0.25):
    """
    Get bounding box from 2D keypoints with fixed aspect ratio w:h = 3:4
    Maintains aspect ratio even when clamped to image boundaries.

    Args:
        kps2d: torch.Tensor [1, 21, 2] or [21, 2] - 2D keypoints
        img_h: image height (default 384)
        img_w: image width (default 512)
        aspect_ratio: width/height ratio (default 3/4)
        padding: padding around keypoints as fraction of box size

    Returns:
        bbox: [x_min, y_min, x_max, y_max] - bounding box coordinates
    """
    # Get min/max coordinates
    x_min, y_min, x_max, y_max = input_bbox

    # Center of keypoints
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    # Current size with padding
    kps_w = x_max - x_min
    kps_h = y_max - y_min
    pad_w = kps_w * padding
    pad_h = kps_h * padding

    # Size needed to contain keypoints with padding
    size_w = kps_w + 2 * pad_w
    size_h = kps_h + 2 * pad_h

    # Adjust to maintain aspect ratio w:h = 3:4
    if size_w / size_h < aspect_ratio:
        size_w = size_h * aspect_ratio
    else:
        size_h = size_w / aspect_ratio

    # Calculate bbox with center at (cx, cy)
    x_min = cx - size_w / 2
    x_max = cx + size_w / 2
    y_min = cy - size_h / 2
    y_max = cy + size_h / 2

    # Check if box goes out of bounds
    out_of_bounds = x_min < 0 or x_max > img_w or y_min < 0 or y_max > img_h
    
    if out_of_bounds:
        # Find the maximum possible box size that fits in the image
        # while maintaining aspect ratio and staying within bounds
        
        # Try to fit by adjusting position (not size)
        # First, try to shift the box to fit
        shift_x = 0
        shift_y = 0
        
        if x_min < 0:
            shift_x = -x_min  # Move right
        elif x_max > img_w:
            shift_x = img_w - x_max  # Move left
            
        if y_min < 0:
            shift_y = -y_min  # Move down
        elif y_max > img_h:
            shift_y = img_h - y_max  # Move up
        
        # Apply shifts
        x_min += shift_x
        x_max += shift_x
        y_min += shift_y
        y_max += shift_y
        
        # If still out of bounds after shifting, we need to shrink
        if x_min < 0 or x_max > img_w or y_min < 0 or y_max > img_h:
            # Calculate maximum possible dimensions
            max_width = img_w
            max_height = img_h
            
            # Find the limiting dimension
            if max_width / max_height < aspect_ratio:
                # Width is limiting factor
                final_width = max_width
                final_height = final_width / aspect_ratio
            else:
                # Height is limiting factor
                final_height = max_height
                final_width = final_height * aspect_ratio
            
            # Recalculate box with new size, centered as much as possible
            # but constrained to image bounds
            x_min = max(0, min(img_w - final_width, cx - final_width / 2))
            x_max = x_min + final_width
            y_min = max(0, min(img_h - final_height, cy - final_height / 2))
            y_max = y_min + final_height
    else:
        # Box is within bounds, no adjustment needed
        pass

    # Final rounding to integers
    x_min = int(np.floor(x_min))
    y_min = int(np.floor(y_min))
    x_max = int(np.ceil(x_max))
    y_max = int(np.ceil(y_max))

    return [x_min, y_min, x_max, y_max]
