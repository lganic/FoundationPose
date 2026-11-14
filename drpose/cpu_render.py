"""A plain CPU silhouette, for checks that must not compete with a GPU job for its memory.

It shares no code with nvdiffrast, which also makes it an independent check on what that drew.
"""
import cv2
import numpy as np


def silhouette(vertices, faces, pose, K, H, W):
  """Union of the camera-facing triangles of a closed mesh. Vertices in meters, pose object-to-camera."""
  cam = vertices @ pose[:3, :3].T + pose[:3, 3]
  uv = cam[:, :2] / cam[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
  tri = uv[faces]
  area = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
  fixed = np.round(tri[area > 0] * 16).astype(np.int32)   # 4 fractional bits
  out = np.zeros((H, W), np.uint8)
  for t in fixed:
    cv2.fillConvexPoly(out, t, 1, lineType=cv2.LINE_8, shift=4)
  return out.astype(bool)
