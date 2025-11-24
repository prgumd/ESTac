# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
code_dir = 'third_party/FoundationPose'
sys.path.insert(0, code_dir)

from estimater import *
from datareader import *
import argparse

if __name__=='__main__':
  lock_type = "tubular_hard"
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/{lock_type}/mesh/mesh_reconstructed.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/{lock_type}')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--track', action="store_true", help="Use tracking mode")
  parser.add_argument('--wait', action="store_true", help="Wait for user input after each frame")
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file,process=False)
  # scale_dict = {"pin_tumbler": 0.19, "disc_detainer": 1.0, "dimpled": 1.0, "tubular":1.0}
  # mesh.apply_scale(scale_dict[lock_type])
  # Create coordinate frame (X=red, Y=green, Z=blue)
  frame = trimesh.creation.axis(origin_size=0.001, axis_length=0.0172/2)
  # Combine mesh and frame in a scene
  scene = trimesh.Scene([mesh, frame])
  # Show the scene
  scene.show()
  # exit()

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
  to_origin_old, _ = trimesh.bounds.oriented_bounds(mesh)
  print(to_origin_old)
  # Apply transformation to mesh
  mesh.apply_transform(to_origin_old)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  print(to_origin)

  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  bbox[0,0] = 0.0
  bbox[1,0] = extents[0]
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  pred_poses = []
  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if not args.track or i==0:
      mask = reader.get_mask(i).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      # center_pose = pose
      center_pose = pose @ np.linalg.inv(to_origin)
      center_pose[:3,:3] = pose[:3,:3]
      center_pose = center_pose @ to_origin_old
      # center_pose[:3,3] -= to_origin[:3,3] #@np.linalg.inv(to_origin)
      # center_pose[:3,:3] = center_pose[:3,:3] @ np.array([[-1,0,0],
      #                                [0,0,1],
      #                                [0,1,0]])
      pred_poses.append(center_pose)
      # vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      if args.wait:
        cv2.waitKey(0)
      else:
        cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
    
    print(f"len of pred {len(pred_poses)}")
  pred_poses = np.array(pred_poses)
  np.save(args.test_scene_dir+"/predicted_hole_poses.npy",pred_poses)


