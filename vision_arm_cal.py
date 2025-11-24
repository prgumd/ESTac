import argparse
import glob
import os
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def generate_checkerboard_corners(rows, cols, square_size):
    p_corners_t_x = square_size * np.arange(0, rows-1, dtype=float)
    p_corners_t_y = square_size * np.arange(0, cols-1, dtype=float)
    p_corners_t_xx, p_corners_t_yy = np.meshgrid(p_corners_t_x, p_corners_t_y)
    p_corners_t_zz = np.zeros_like(p_corners_t_xx)
    p_corners_t = np.stack((p_corners_t_xx.flatten(),
                            p_corners_t_yy.flatten(),
                            p_corners_t_zz.flatten()), axis=1)
    return p_corners_t

# Copied from calibrate.py
def _plot_reprojection(p_c_i, p_c_i_detected, camera_res, title_suffix, alpha=0.0025):
    p_c_i_detected = p_c_i_detected.squeeze()

    plt.figure()
    plt.subplot(2, 2, 4)
    # plot the l2 reprojection error distribution
    diff_p_c_i = (p_c_i - p_c_i_detected).reshape((-1, 2))
    norm_diff_p_c_i = np.linalg.norm(diff_p_c_i, axis=1)
    plt.hist(norm_diff_p_c_i, bins=100)
    # plt.title('Reprojection error distribution for ' + title_suffix)
    plt.xlabel('l2 error (pixels)')
    plt.ylabel('count')

    # Visualize reprojection for all p_c_i_detected
    plt.subplot(2, 2, (1, 2))
    msize = 5.0
    plt.quiver(p_c_i_detected[:, :, 0],
               p_c_i_detected[:, :, 1],
               (p_c_i - p_c_i_detected)[:, :, 0],
               (p_c_i - p_c_i_detected)[:, :, 1],
               angles='xy', scale_units='xy', scale=1)
    plt.scatter(p_c_i_detected[:, :, 0], p_c_i_detected[:, :, 1], marker='o', color='orange', label='detected', s=msize)
    plt.scatter(p_c_i[:, :, 0], p_c_i[:, :, 1], marker='x', color='blue', label='reprojected', s=msize, alpha=0.5)
    plt.grid()
    plt.xlim([0, camera_res[0]])
    plt.ylim([0, camera_res[1]])
    plt.legend()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncols=2)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')

    plt.subplot(2, 2, 3)
    plt.scatter(p_c_i[:, :, 0].flatten() - p_c_i_detected[:, :, 0].flatten(),
                p_c_i[:, :, 1].flatten() - p_c_i_detected[:, :, 1].flatten(),
                marker='o', color='orange', label='reproj error', s=msize, alpha=alpha)
    # plt.xlim([-5, 5])
    # plt.ylim([-5, 5])
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.grid()

    plt.suptitle(title_suffix)
    plt.tight_layout()

def load_data(sequence, checkerboard_size, vis_detections=False):
  # Load intrinsics from camera
  K = np.load(os.path.join(sequence, 'rgb_intrinsics.npy'))

  # Get images
  images = sorted(list(glob.glob(os.path.join(sequence, 'img_*.png'))))

  # Get recorded poses, assume that are from flange to base
  T_bf_array = np.load(os.path.join(sequence, 'flange_poses.npy'))

  # Discard the first two (hack)
  discard_indices = []
  print('Discarding samples:', discard_indices)
  images_filt = []
  T_bf_array_filt = []
  for i in range(len(images)):
    if i not in discard_indices:
      images_filt.append(images[i])
      T_bf_array_filt.append(T_bf_array[i])
  images = images_filt
  T_bf_array = np.array(T_bf_array_filt)

  # Process images
  p_i_list = []
  T_bf_list = []
  for i in range(len(images)):
    image = cv2.imread(images[i])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, 
      (checkerboard_size[0]-1, checkerboard_size[1]-1), None)
 
    # Hack to try and keep an order
    if corners[0][0][0] < corners[-1][0][0]:
      corners = corners[::-1]
      corners = np.ascontiguousarray(corners)

    # If found, add object points, image points (after refining them)
    if ret == True: 
        p_i = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        p_i_list.append(p_i.squeeze(1))
        T_bf_list.append(T_bf_array[i, :, :])
 
    # Draw and display the corners
    if vis_detections:
      if ret: cv2.drawChessboardCorners(image,
        (checkerboard_size[0]-1, checkerboard_size[1]-1), p_i, ret)
      cv2.imshow('image', image)
      cv2.waitKey(0)
  T_bf_array = np.array(T_bf_list)

  res_shape = image.shape
  return res_shape, K, p_i_list, T_bf_array

def calibrate_arm(sequence,
                  p_i_list,
                  T_bf_array,
                  checkerboard_size,
                  square_size,
                  K0,
                  vis_reprojection_closed_form=False,
                  vis_reprojections_joint=False,
                  res_shape=None,
                  refine_joint=False):
  # Construct points on checkerboard
  p_corners_t = generate_checkerboard_corners(checkerboard_size[0], checkerboard_size[1], square_size)

  t_ct_list = []
  R_ct_list = []
  T_bf_list = []
  p_i_list_pnp = []
  for i in range(len(p_i_list)):
    ret, rotvec_ct, t_ct = cv2.solvePnP(p_corners_t, p_i_list[i], K0, None)
    if ret:
      R_ct = R.from_rotvec(rotvec_ct.flatten()).as_matrix()
      t_ct_list.append(t_ct)
      R_ct_list.append(R_ct)
      T_bf_list.append(T_bf_array[i, :, :])
      p_i_list_pnp.append(p_i_list[i])
  R_ct_array = np.array(R_ct_list)
  t_ct_array = np.array(t_ct_list)
  T_bf_array = np.array(T_bf_list)
  p_i_array_pnp = np.array(p_i_list_pnp)

  # The calibration routine needs fb instead of bf
  R_fb_list = []
  t_fb_list = []
  for T_bf in T_bf_array:
    R_fb_list.append(T_bf[:3, :3].T)
    t_fb_list.append(-T_bf[:3, :3].T @ T_bf[:3, 3])
  R_fb_array = np.array(R_fb_list)
  t_fb_array = np.array(t_fb_list)

  # Solve the AX = ZB equation to
  # get the transform from base to target (base to world)
  # and the transform from flange to camera
  R_tb, t_tb, R_cf, t_cf = cv2.calibrateRobotWorldHandEye(R_ct_array, t_ct_array,
                                                          R_fb_array, t_fb_array)
  t_tb = t_tb.flatten()
  t_cf = t_cf.flatten()

  # At this point we have
  # T_bf (robot)
  # T_fc (calibrated)
  # T_ct (PnP) (we will not use this during refinement)
  # T_tb (calibrated)

  # Reproject the target points into the camera frame
  def reproject(R_tb, t_tb, R_cf, t_cf, K):
    p_i_list_reprojected = []
    R_bt = R_tb.T
    t_bt = -R_tb.T @ t_tb
    for i in range(R_fb_array.shape[0]):
      # T_cf @ T_fb @ T_bt @ p_corners_t  
      p_corners_b = (R_bt          @ p_corners_t.T).T + t_bt
      p_corners_f = (R_fb_array[i] @ p_corners_b.T).T + t_fb_array[i]
      p_corners_c = (R_cf          @ p_corners_f.T).T + t_cf
      p_corners_i = (K @ p_corners_c.T).T
      p_corners_i = p_corners_i[:, 0:2] / p_corners_i[:, 2][:, None]
      p_i_list_reprojected.append(p_corners_i)
    p_i_array_reprojected = np.array(p_i_list_reprojected)
    return p_i_array_reprojected

  p_i_array_reprojected = reproject(R_tb, t_tb, R_cf, t_cf, K0)

  reproj_rms = np.sqrt(np.mean(np.square(p_i_array_reprojected - p_i_array_pnp)))
  # print(f'Reprojection RMS closed form {reproj_rms:0.3f}')

  if vis_reprojection_closed_form:
    _plot_reprojection(p_i_array_reprojected, p_i_array_pnp, res_shape[0:2][::-1], '', alpha=0.01)
    plt.savefig(os.path.join(sequence, 'reprojection_closed_form.png'))

  def reprojection_rms_optimizer(x):
    w_bb_p = x[0:3]
    t_tb   = x[3:3+3]
    w_ff_p = x[6:6+3]
    t_cf   = x[9:9+3]
    if refine_joint:
      K_coef = x[12:12+4]
      K = np.array(((K_coef[0], 0, K_coef[2]), 
                    (0, K_coef[1], K_coef[3]),
                    (0, 0, 1)))
    else:
       K = K0

    R_tb_p = R_tb @ R.from_rotvec(w_bb_p).as_matrix()
    R_cf_p = R_cf @ R.from_rotvec(w_ff_p).as_matrix()

    # print('hi')
    # print(R_tb_p)
    # print(t_tb)
    # print(R_cf_p)
    # print(t_cf)

    p_i_array_reprojected = reproject(R_tb_p, t_tb, R_cf_p, t_cf, K)

    return (p_i_array_reprojected - p_i_array_pnp).flatten() / np.sqrt(p_i_array_pnp.size)

  K_coef_0 = np.array((K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]))
  if refine_joint:
    x0 = np.concatenate((np.zeros((3,)), t_tb.flatten(), np.zeros((3,)), t_cf.flatten(), K_coef_0.flatten()))
  else:
    x0 = np.concatenate((np.zeros((3,)), t_tb.flatten(), np.zeros((3,)), t_cf.flatten()))
  # res = reprojection_rms_optimizer(x0)
  # print('pre', np.sqrt(np.sum(np.square(res))))
  sol = least_squares(reprojection_rms_optimizer,
                      x0=x0,
                      verbose=0)
  # print(f'Reprojection RMS joint refinement {np.sqrt(2*sol.cost):0.3f}')

  R_tb_star = R_tb @ R.from_rotvec(sol.x[0:3]).as_matrix()
  t_tb_star = sol.x[3:3+3]
  R_cf_star = R_cf @ R.from_rotvec(sol.x[6:6+3]).as_matrix()
  t_cf_star = sol.x[9:9+3]
  if refine_joint:
    K_star    = sol.x[12:12+4]
    K_star = np.array(((K_star[0], 0, K_star[2]),
                      (0, K_star[1], K_star[3]),
                      (0, 0, 1)))
  else:
    K_star = K0

  p_i_array_reprojected_star = reproject(R_tb_star, t_tb_star, R_cf_star, t_cf_star, K_star)

  reproj_rms = np.sqrt(np.mean(np.square(p_i_array_reprojected_star - p_i_array_pnp)))
  # print(f'Reprojection RMS joint refinement {reproj_rms:0.3f}')

  # Calculate error per checkerboard
  # print(f'Reprojection RMS per board')
  # reproj_rms_per_board = np.sqrt(np.mean(np.square(p_i_array_reprojected_star - p_i_array_pnp), axis=(1, 2)))
  # for i in range(reproj_rms_per_board.shape[0]):
  #   print(f'{i:3d}: {reproj_rms_per_board[i]:5.2f}')

  if vis_reprojections_joint:
    _plot_reprojection(p_i_array_reprojected_star, p_i_array_pnp, res_shape[0:2][::-1], '', alpha=0.1)
    plt.savefig(os.path.join(args.sequence, 'reprojection_joint.png'))

  return reproj_rms, R_tb_star, t_tb_star, R_cf_star, t_cf_star, K_star

def run_calibration(sequence, checkerboard_size, refine_joint=False):
  res_shape, K0, p_i_list, T_bf_array = load_data(sequence, checkerboard_size, vis_detections=True)

  reproj_rms_list = []
  best_reproj_rms = None
  square_size_nom = 0.025
  square_size_array = np.linspace(0.9*square_size_nom, 1.1*square_size_nom, num=50)
  for square_size in square_size_array:
    (
      reproj_rms,
      R_tb, # base to target
      t_tb,
      R_cf, # flange to camera
      t_cf,
      K,
    ) = calibrate_arm(sequence,
                      p_i_list, T_bf_array,
                      checkerboard_size=checkerboard_size,
                      square_size=square_size,
                      K0=K0,
                      vis_reprojection_closed_form=False,
                      vis_reprojections_joint=False,
                      res_shape=res_shape,
                      refine_joint=refine_joint)
    reproj_rms_list.append(reproj_rms)
    print(f'square_size {square_size:6.4f} reproj_rms {reproj_rms:0.3f}')

    if best_reproj_rms is None or best_reproj_rms > reproj_rms:
      best_reproj_rms = reproj_rms
      best_square_size = square_size
      best_R_tb = R_tb
      best_t_tb = t_tb
      best_R_cf = R_cf
      best_t_cf = t_cf

  # Run one more time for visualizations
  print('\nbest_square_size visualization')
  calibrate_arm(sequence,
                p_i_list, T_bf_array,
                checkerboard_size=checkerboard_size,
                square_size=best_square_size,
                K0=K,
                vis_reprojection_closed_form=False,
                vis_reprojections_joint=True,
                res_shape=res_shape,
                refine_joint=refine_joint)

  print(f'\nprevious K')
  print(K0)
  print(f'new K')
  print(K)

  print(f'\nbest_reproj_rms {best_reproj_rms:0.3f}')
  print(f'best best_square_size {best_square_size:0.3f}')
  print('T_tb')
  print(best_R_tb)
  print(best_t_tb)
  print('T_cf')
  print(best_R_cf)
  print(best_t_cf)


  # Save the transforms
  T_f2c = np.vstack((np.hstack((best_R_cf, best_t_cf.reshape(3, 1))), [0, 0, 0, 1]))
  T_b2w = np.vstack((np.hstack((best_R_tb, best_t_tb.reshape(3, 1))), [0, 0, 0, 1]))
  np.save(os.path.join(sequence, 'world_to_base.npy'), np.linalg.inv(T_b2w))
  np.save(os.path.join(sequence, 'camera_to_flange.npy'), np.linalg.inv(T_f2c))
  np.save(os.path.join(sequence, 'square_size.npy'), best_square_size)

  # Visualize the camera and flange frames
  # import open3d as o3d
  # camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
  # flange_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
  # flange_frame.transform(T_f2c)
  # o3d.visualization.draw_geometries([camera_frame, flange_frame])

  np.savez(os.path.join(sequence, 'arm_extrinsics.npz'),
           R_tb = best_R_tb,
           t_tb = best_t_tb,
           R_cf = best_R_cf,
           t_cf = best_t_cf,
           K=K)

  plot_square_size_to_rms = True
  if plot_square_size_to_rms:
    plt.figure()
    plt.plot(square_size_array, reproj_rms_list)
    plt.grid()
    plt.ylabel('Reprojection RMS (pixels)')
    plt.xlabel('square size (meters)')
    plt.title('Square size to final reprojection error')
    plt.tight_layout()

  plt.show()

def run_collection(sequence):
  from realsense import RealSense
  import rtde_control
  import rtde_receive

  rtde_c = rtde_control.RTDEControlInterface("10.0.0.78")
  rtde_r = rtde_receive.RTDEReceiveInterface("10.0.0.78")

  tool_offset_pose = np.zeros((6))
  rtde_c.setTcp(tool_offset_pose)

  # Initialize camera 
  camera = RealSense(align_color=True, structured_light=1)
  print("Camera initialized")

  # Discarding first frames from realsense 
  for _ in range(50):
      depth, rgb = camera.get_aligned_rgbd()

  os.makedirs(sequence, exist_ok=True)

  K = camera.get_rgb_intrinsics()
  np.save(os.path.join(sequence, "rgb_intrinsics.npy"), K)

  flange_poses = []
  for i in range(16):
      while True:
          _, rgb = camera.get_aligned_rgbd()
          cv2.imshow(f"Image {i+1}/16", rgb)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              break
      # now capture the image and display
      _, rgb = camera.get_aligned_rgbd()
      cv2.imwrite(os.path.join(sequence, f"img_{i+1:03d}.png"), rgb)
      # cv2.imshow("RGB", rgb)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
      
      # Capture flange pose
      T_f2b = np.eye(4)
      curr_pose = rtde_r.getActualTCPPose()
      T_f2b[:3,:3] = R.from_rotvec(curr_pose[3:]).as_matrix()
      T_f2b[:3,3] = curr_pose[:3]
      flange_poses.append(T_f2b)
  flange_poses = np.array(flange_poses)
  np.save(os.path.join(sequence, "flange_poses.npy"), flange_poses)

def run_test(sequence, checkerboard_size):
  from realsense import RealSense
  import rtde_control
  import rtde_receive

  K = np.load(os.path.join(sequence, "rgb_intrinsics.npy"))
  T_w2b = np.load(os.path.join(sequence, "world_to_base.npy"))
  T_c2f = np.load(os.path.join(sequence, "camera_to_flange.npy"))
  square_size = np.load(os.path.join(sequence, "square_size.npy"))

  # Generate checkerboard corners
  corners_3d = generate_checkerboard_corners(checkerboard_size[0], checkerboard_size[1], square_size)

  rtde_c = rtde_control.RTDEControlInterface("10.0.0.78")
  rtde_r = rtde_receive.RTDEReceiveInterface("10.0.0.78")

  # Set cameras as the tool center point
  tool_offset_pose = np.zeros((6))
  tool_offset_pose[:3] += T_c2f[:3,3]
  tool_offset_pose[3:] = R.from_matrix(T_c2f[:3,:3]).as_rotvec()
  rtde_c.setTcp(tool_offset_pose)

  # Plot the reprojected corners onto a real image
  camera = RealSense(align_color=True,structured_light=1)
  while True:
      # Project corners into image
      actual_pose = rtde_r.getActualTCPPose()
      T_c2b = np.eye(4)
      T_c2b[:3,:3] = R.from_rotvec(actual_pose[3:]).as_matrix()
      T_c2b[:3, 3] = actual_pose[:3]
      T_w2c_measured = np.linalg.inv(T_c2b) @ T_w2b
      corners_2d, _ = cv2.projectPoints(
          corners_3d, R.from_matrix(T_w2c_measured[:3,:3]).as_rotvec(), T_w2c_measured[:3, 3], K, None)
      corners_2d = corners_2d.reshape(-1, 2)

      _, rgb = camera.get_aligned_rgbd()
      for pt in corners_2d:
          x, y = int(pt[0]), int(pt[1])
          if 0 <= x < 848 and 0 <= y < 480:
              cv2.circle(rgb, (x, y), 3, (0, 0, 255), -1)
      cv2.imshow("Reprojection", rgb)
      if cv2.waitKey(1) == ord('q'):
          break
      time.sleep(0.03)

def run_test_move(sequence, checkerboard_size):
  import rtde_control
  import rtde_receive

  input("Press Enter to move the stylus to the origin of checkerboard...")

  T_w2b = np.load(os.path.join(sequence, "world_to_base.npy"))

  rtde_c = rtde_control.RTDEControlInterface("10.0.0.78")
  rtde_r = rtde_receive.RTDEReceiveInterface("10.0.0.78")

  # Set cameras as the tool center point
  tool_offset_pose = np.zeros((6))
  tool_offset_pose[2] += 0.120
  rtde_c.setTcp(tool_offset_pose)

  target_pose = rtde_r.getActualTCPPose()
  target_pose[:3] = T_w2b[:3, 3]
  rtde_c.moveL(target_pose, 0.1, 0.01)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--sequence',     default='keyinsertion/data/calibration')
  parser.add_argument('--collect', action='store_true', help='Collect data for calibration')
  parser.add_argument('--calibrate', action='store_true', help='Run calibration on collected data')
  parser.add_argument('--test', action='store_true', help='Test calibration by reprojecting points online')
  parser.add_argument('--test_move', action='store_true', help='Test calibration by moving stylus to origin')
  parser.add_argument('--joint', action='store_true', help='Joint intrinsics/extrinsics refinement')
  args = parser.parse_args()
  if not args.collect and not args.calibrate and not args.test:
    print('Please specify --collect, --calibrate or --test or --test_move')
  checkerboard_size = (5, 7) # rows, cols

  if args.collect:
    run_collection(args.sequence)
  if args.calibrate:
    run_calibration(args.sequence, checkerboard_size, args.joint)
  if args.test:
    run_test(args.sequence, checkerboard_size)
  if args.test_move:
    run_test_move(args.sequence, checkerboard_size)