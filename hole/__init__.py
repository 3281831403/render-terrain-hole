"""Terrain-hole masking, stitching, and LOD package."""

from .punch import (
    EPS,
    boundary_cells,
    clip_triangles,
    depth_bias,
    hole_batch,
    hole_mask,
    lod_consistent,
    stitch_ring,
    validate_hole,
)

__all__ = [
    "EPS",
    "boundary_cells",
    "clip_triangles",
    "depth_bias",
    "hole_batch",
    "hole_mask",
    "lod_consistent",
    "stitch_ring",
    "validate_hole",
]
