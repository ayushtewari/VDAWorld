"""VGGT depth and point-map adapter."""

import logging
import os

import cv2
import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if _DEVICE.type == "cuda":
    _DTYPE = (
        torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    )
else:
    _DTYPE = torch.float32

_MODEL_CACHE = None


def get_model():
    """Load and cache the VGGT checkpoint."""

    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        logger.info("Loading VGGT on %s", _DEVICE)
        _MODEL_CACHE = VGGT.from_pretrained("facebook/VGGT-1B").to(_DEVICE)
        _MODEL_CACHE.eval()
    return _MODEL_CACHE


def estimate(temp_dir: str) -> dict[str, np.ndarray]:
    """Estimate a point map and intrinsics for images in `temp_dir`."""

    image_names = sorted(
        os.path.join(temp_dir, name)
        for name in os.listdir(temp_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_names:
        raise ValueError(f"No input images found in {temp_dir}")

    images = load_and_preprocess_images(image_names).to(_DEVICE)
    original = cv2.imread(image_names[0])
    if original is None:
        raise ValueError(f"Could not read input image: {image_names[0]}")
    orig_h, orig_w = original.shape[:2]

    autocast_enabled = _DEVICE.type == "cuda"
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=_DEVICE.type,
            dtype=_DTYPE,
            enabled=autocast_enabled,
        ),
    ):
        predictions = get_model()(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"],
        images.shape[-2:],
    )
    extrinsic[0, :] = torch.eye(4, device=_DEVICE)[None, :3]

    point_map = unproject_depth_map_to_point_map(
        predictions["depth"].squeeze(0),
        extrinsic.squeeze(0),
        intrinsic.squeeze(0),
    )
    point_map[:, :, :, 1:] *= -1

    if isinstance(point_map, np.ndarray):
        mean_dist = np.mean(np.linalg.norm(point_map.reshape((-1, 3)), axis=-1))
        points = point_map / mean_dist
    else:
        mean_dist = torch.mean(torch.linalg.norm(point_map.reshape((-1, 3)), dim=-1))
        points = (point_map / mean_dist).detach().cpu().numpy()

    intrinsics = intrinsic[0, 0].detach().cpu().numpy()
    if points.ndim == 4 and points.shape[0] == 1:
        points = points[0]

    curr_h, curr_w = points.shape[:2]
    scale_x = orig_w / curr_w
    scale_y = orig_h / curr_h
    intrinsics[0, (0, 2)] *= scale_x
    intrinsics[1, (1, 2)] *= scale_y

    points = cv2.resize(
        points.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return {
        "pts3d": points,
        "intrinsics": intrinsics.astype(np.float32),
    }
