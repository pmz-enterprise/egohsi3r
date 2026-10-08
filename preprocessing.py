"""Full-frame preprocessing shared with the original training pipeline."""
import cv2
import numpy as np

def _long_edge_resolution(image_size, long_edge, pad_to_multiple):
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image_size}")
    if long_edge <= 0 or pad_to_multiple <= 0:
        raise ValueError("long_edge and pad_to_multiple must be positive")
    if long_edge % pad_to_multiple != 0:
        raise ValueError(
            f"long_edge={long_edge} must be divisible by "
            f"pad_to_multiple={pad_to_multiple}"
        )

    scale = float(long_edge) / max(width, height)
    content_width = max(1, int(round(width * scale)))
    content_height = max(1, int(round(height * scale)))
    if pad_to_multiple > 1:
        if width >= height:
            content_width = long_edge
            content_height = (
                (content_height + pad_to_multiple - 1) // pad_to_multiple
            ) * pad_to_multiple
        else:
            content_width = (
                (content_width + pad_to_multiple - 1) // pad_to_multiple
            ) * pad_to_multiple
            content_height = long_edge
    return content_width, content_height



def _resize_map(array, size, interpolation):
    width, height = size
    return cv2.resize(array, (width, height), interpolation=interpolation)



def resize_and_pad_frame(image, depthmap, intrinsics, bbox, resolution, maps=None):
    """Resize a full frame without cropping and transform all pixel annotations."""
    target_width, target_height = map(int, resolution)
    source_height, source_width = depthmap.shape
    if image.shape[:2] != (source_height, source_width):
        raise ValueError(
            f"RGB/depth shape mismatch: {image.shape[:2]} vs {depthmap.shape}"
        )

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, min(target_width, int(round(source_width * scale))))
    resized_height = max(1, min(target_height, int(round(source_height * scale))))
    scale_x = resized_width / source_width
    scale_y = resized_height / source_height
    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2

    rgb_interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized_image = _resize_map(image, (resized_width, resized_height), rgb_interp)
    image_out = np.zeros((target_height, target_width, 3), dtype=image.dtype)
    image_out[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized_image

    resized_depth = _resize_map(
        depthmap, (resized_width, resized_height), cv2.INTER_NEAREST
    )
    depth_out = np.zeros((target_height, target_width), dtype=np.float32)
    depth_out[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized_depth.astype(np.float32)

    intrinsics_out = np.asarray(intrinsics, dtype=np.float32).copy()
    intrinsics_out[0] *= scale_x
    intrinsics_out[1] *= scale_y
    intrinsics_out[0, 2] += pad_left
    intrinsics_out[1, 2] += pad_top

    bbox_out = np.asarray(bbox, dtype=np.float32).copy()
    bbox_out[:, [0, 2]] = bbox_out[:, [0, 2]] * scale_x + pad_left
    bbox_out[:, [1, 3]] = bbox_out[:, [1, 3]] * scale_y + pad_top
    bbox_out[:, [0, 2]] = np.clip(bbox_out[:, [0, 2]], 0, target_width)
    bbox_out[:, [1, 3]] = np.clip(bbox_out[:, [1, 3]], 0, target_height)

    content_mask = np.zeros((target_height, target_width), dtype=bool)
    content_mask[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = True

    map_outputs = {}
    for name, (array, interpolation, fill_value) in (maps or {}).items():
        resized = _resize_map(array, (resized_width, resized_height), interpolation)
        output = np.full(
            (target_height, target_width), fill_value, dtype=array.dtype
        )
        output[
            pad_top : pad_top + resized_height,
            pad_left : pad_left + resized_width,
        ] = resized
        map_outputs[name] = output

    return (
        image_out,
        depth_out,
        intrinsics_out,
        bbox_out,
        content_mask,
        map_outputs,
    )

