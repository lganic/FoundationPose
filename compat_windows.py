'''
Replacement logic for two dependenceies from pytorch3d that are hard to build on windows.

They should be nearly identical to the originals. 
'''

import math
import numpy as np
import torch
import torch.nn.functional as F

def so3_exp_map(log_rot):
  '''
  Axis-angle (N,3) to rotation matrices (N,3,3) through a unit quaternion.

  Same route as pytorch3d.transforms.so3_exp_map. torch.sinc keeps it exact near zero, which
  matters here because the refiner's last iterations produce very small rotation updates.
  '''

  angles = torch.linalg.vector_norm(log_rot, dim=-1, keepdim=True)
  q = torch.cat([torch.cos(angles * 0.5), log_rot * 0.5 * torch.sinc(angles * 0.5 / math.pi)], dim=-1)

  r, i, j, k = q.unbind(-1)
  two_s = 2.0 / (q * q).sum(-1)

  return torch.stack((
    1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
    two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
    two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
  ), -1).reshape(-1, 3, 3)


def rotation_6d_to_matrix(d6):
  '''
  Equivalent of pytorch3d.transforms.rotation_6d_to_matrix

  Just a Gram-Schmidt on two 3-vectors
  '''
  a1, a2 = d6[..., :3], d6[..., 3:]
  b1 = F.normalize(a1, dim=-1)
  b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
  b2 = F.normalize(b2, dim=-1)
  b3 = torch.cross(b1, b2, dim=-1)
  return torch.stack((b1, b2, b3), dim=-2)

class mycpp:
  '''
  Replacement for the compiled `mycpp` module
  '''

  @staticmethod
  def cluster_poses(angle_diff, dist_diff, poses_in, symmetry_tfs):
    '''
    Greedy pose de-duplication, same visiting order and float32 math as mycpp/src/app/pybind_api.cpp.

    A pose starts a new cluster unless some existing cluster has a translation closer than
    `dist_diff` (meters) and a rotation within `angle_diff` (degrees) under any symmetry.
    '''
    poses_in = np.asarray(poses_in, dtype=np.float32)
    symmetry_tfs = np.asarray(symmetry_tfs, dtype=np.float32)
    radian_thres = np.float32(angle_diff / 180.0 * math.pi)
    kept = [poses_in[0]]
    for cur in poses_in[1:]:
      clusters = np.stack(kept)
      near = np.linalg.norm(clusters[:, :3, 3] - cur[:3, 3], axis=1) < dist_diff
      is_new = True
      if near.any():
        rots = (cur[None] @ symmetry_tfs)[:, :3, :3]                  # (S,3,3)
        rel = np.einsum('sij,ckj->scik', rots, clusters[near][:, :3, :3])
        cos = (np.trace(rel, axis1=2, axis2=3) - 1) / 2.0
        diff = np.arccos(np.clip(cos, -1.0, 1.0))
        is_new = not (diff < radian_thres).any()
      if is_new:
        kept.append(cur)
    return [p.copy() for p in kept]
