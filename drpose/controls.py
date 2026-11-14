'''
The controls that differentiable refinement has to beat.

A gain from the refiner only means something if a cheaper, simpler step would not get the same gain:
  
  icp is the classical answer to aligning a model with a depth map, with no rendering and no Sgradients. It answers "is a differentiable renderer needed at all?".

It sees exactly the same frame and mask as the refiner.
'''

import numpy as np
import open3d as o3d

def icp(mesh_points, mesh_normals, depth, mask, K, pose, max_dist=0.02, iters=30):
  '''
  Point-to-plane ICP from the model to the masked depth points. Returns the refined pose.

  With too few observed points to constrain a pose, the input pose comes back unchanged.
  '''
  
  vs, us = np.nonzero(mask & (depth > 1e-3))

  if len(us) < 50:
    return pose

  z = depth[vs, us]

  scene = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], -1)
  target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene))
  source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh_points))
  source.normals = o3d.utility.Vector3dVector(mesh_normals)
  # The model has reliable normals and the scene does not, so align scene -> model and invert.
  result = o3d.pipelines.registration.registration_icp(
    target, source, max_dist, np.linalg.inv(pose),
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
  if len(result.correspondence_set) < 50:
    return pose
  return np.linalg.inv(result.transformation)
