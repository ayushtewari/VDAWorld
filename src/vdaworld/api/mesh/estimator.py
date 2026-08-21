"""SAM 3D Objects mesh adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from trimesh import Trimesh

_MODEL_CACHE = None


def _configure_upstream_path() -> Path:
    raw_path = os.environ.get("SAM3D_PATH")
    if not raw_path:
        raise RuntimeError("SAM3D_PATH must point to a SAM 3D Objects checkout.")
    root = Path(raw_path).expanduser().resolve()
    notebook = root / "notebook"
    if not (root / "checkpoints" / "hf" / "pipeline.yaml").is_file():
        raise RuntimeError(
            f"SAM3D_PATH does not contain checkpoints/hf/pipeline.yaml: {root}"
        )
    for path in (root, notebook):
        value = str(path)
        if value not in sys.path:
            sys.path.append(value)
    return root


def get_model():
    """Load and cache the upstream inference pipeline."""

    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        import logging
        from inference import Inference

        root = _configure_upstream_path()
        logging.info("Loading SAM 3D Objects")
        _MODEL_CACHE = Inference(
            str(root / "checkpoints" / "hf" / "pipeline.yaml"),
            compile=False,
        )
    return _MODEL_CACHE


def make_manifold(mesh: Trimesh, refine_steps: int = 0) -> Trimesh:
    """Return a manifold version of `mesh` when conversion succeeds."""

    import manifold3d

    if not isinstance(mesh, Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    manifold_mesh = manifold3d.Mesh(
        vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32),
    )
    manifold = manifold3d.Manifold(manifold_mesh)
    if refine_steps > 0:
        manifold = manifold.refine(refine_steps)
    converted = manifold.to_mesh()
    if len(converted.vert_properties) == 0:
        return mesh
    return Trimesh(
        vertices=converted.vert_properties,
        faces=converted.tri_verts,
        process=True,
    )


def remove_interior_surfaces(mesh: Trimesh) -> Trimesh:
    """Keep the largest connected mesh component."""

    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh
    return max(components, key=lambda component: component.volume)


def estimate(
    image: np.ndarray,
    mask: np.ndarray,
    max_vertices: int = 10000,
) -> Trimesh:
    """Reconstruct a masked object and return a simplified mesh."""

    _configure_upstream_path()
    from sam3d_objects.utils.visualization import SceneVisualizer

    output = get_model()(image, mask, seed=42)
    mesh = output["glb"]
    mesh_to_splat = np.array(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        dtype=np.float32,
    )
    mesh_vertices = mesh.vertices @ mesh_to_splat
    vertices = (
        SceneVisualizer.object_pointcloud(
            points_local=torch.tensor(mesh_vertices)
            .float()
            .unsqueeze(0)
            .to(output["rotation"].device),
            quat_l2c=output["rotation"],
            trans_l2c=output["translation"],
            scale_l2c=output["scale"],
        )
        .points_list()[0]
        .detach()
        .cpu()
        .numpy()
    )
    mesh.vertices = vertices @ np.array(
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        dtype=np.float32,
    )

    mesh = remove_interior_surfaces(mesh)
    mesh = mesh.simplify_quadric_decimation(
        face_count=min(max_vertices, len(mesh.faces))
    )
    return make_manifold(mesh)
