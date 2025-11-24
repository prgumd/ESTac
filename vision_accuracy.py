import argparse
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import os
from sam2.build_sam import build_sam2_video_predictor
import torch
from PIL import Image
import glob
from pprint import pprint

def T2P(T):
    P = np.zeros(6)
    P[:3] = T[:3, 3]
    P[3:] = R.from_matrix(T[:3, :3]).as_rotvec()
    return P

def P2T(P):
    T = np.eye(4)
    T[:3, 3] = P[:3]
    T[:3, :3] = R.from_rotvec(P[3:]).as_matrix()
    return T

# This code assumes the Z axis of the lock points into the lock
def sample_poses(N, xlim, ylim, zlim, K, res, 
    mount_center_in_lock_frame,
    lock_diameter, roll_limit_deg, seed=None, plot=False):
    theta = np.linspace(0, 2*np.pi, num=100)
    p_l = (lock_diameter / 2) * np.stack((mount_center_in_lock_frame[0] + np.cos(theta),
                                          mount_center_in_lock_frame[1] + np.sin(theta),
                                          mount_center_in_lock_frame[2] + np.zeros_like(theta)),
                                         axis=1)

    if seed is not None:
        np.random.seed(seed)

    num_good_samples = 0
    bad_orientations = 0
    t_lc_list = []
    R_lc_list = []
    p_i_list = []
    for i in range(N):
        # Random position
        x = np.random.uniform(xlim[0], xlim[1])
        y = np.random.uniform(ylim[0], ylim[1])
        z = np.random.uniform(zlim[0], zlim[1])
        t_lc = np.array((x, y, z))

        # Sample orientations uniformly until
        # We find one where all points on the locks perimeter are in the field of view
        while True:
            q = np.random.normal(size=(4,))
            q /= np.linalg.norm(q)
            R_lc = R.from_quat(q).as_matrix()

            # Project all points into image
            R_cl = R_lc.T
            t_cl = -R_lc.T @ t_lc

            p_c = (R_cl @ p_l.T).T + t_cl

            p_i = (K @ p_c.T).T
            p_i = p_i[:, 0:2] / p_i[:, 2][:, None]

            all_points_in_fov = (np.all(p_c[:, 2] > 0)
                                 and np.all(p_i[:, 0] >= 0)
                                 and np.all(p_i[:, 0] < res[0] - 1)
                                 and np.all(p_i[:, 1] >= 0)
                                 and np.all(p_i[:, 1] <= res[1] - 1))

            ypr = R.from_matrix(R_lc).as_euler('YXZ', degrees=True)
            roll_limit_respected = (np.all(ypr[2] > roll_limit_deg[0])
                                    and np.all(ypr[2] < roll_limit_deg[1]))

            if all_points_in_fov and roll_limit_respected:
                break
            bad_orientations += 1

        t_lc_list.append(t_lc)
        R_lc_list.append(R_lc)
        p_i_list.append(p_i)

    t_lc_array = np.array(t_lc_list)
    R_lc_array = np.array(R_lc_list)

    if plot:
        print('N poses', N, 'N discarded orientations', bad_orientations)
        plt.figure(figsize=(8, 10))
        plt.subplot(3,1,1)
        for p_i in p_i_list:
            plt.scatter(p_i[:, 0], p_i[:, 1])
            plt.xlim((0, res[0]-1))
            plt.ylim((0, res[1]-1))
            plt.axis('equal')
            plt.gca().invert_yaxis()
            plt.title('Lock perimeters in image')

        plt.subplot(3,1,2)
        plt.scatter(t_lc_array[:, 0], 0*np.ones_like(t_lc_array[:, 0]), label='x distribution')
        plt.scatter(t_lc_array[:, 1], 1*np.ones_like(t_lc_array[:, 1]), label='y distribution')
        plt.scatter(t_lc_array[:, 2], 2*np.ones_like(t_lc_array[:, 2]), label='z distribution')
        plt.gca().invert_yaxis()
        plt.title('Translation distribution (meters)')
        plt.legend()

        ypr = R.from_matrix(R_lc_array).as_euler('YXZ', degrees=True)
        plt.subplot(3,1,3)
        plt.scatter(ypr[:, 0], 0*np.ones_like(ypr[:, 0]), label='yaw distribution')
        plt.scatter(ypr[:, 1], 1*np.ones_like(ypr[:, 1]), label='pitch distribution')
        plt.scatter(ypr[:, 2], 2*np.ones_like(ypr[:, 2]), label='roll distribution')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title('Rotation distribution (degrees, intrinsic euler YXZ)')
        plt.tight_layout()

    return t_lc_array, R_lc_array, p_i_list

def test_pose_sampler(calibration):
    K = np.load(os.path.join(calibration, 'rgb_intrinsics.npy'))
    t_lc, R_lc = sample_poses(200,
                             [-0.1,  0.1],
                             [-0.1,  0.1],
                             [-0.35, -0.5],
                             K,
                             res=(848, 480),
                             mount_center_in_lock_frame=(0.0, 0.15 / 2, 0.0),
                             lock_diameter=0.15,
                             roll_limit_deg=[-45, 45],
                             seed=None,
                             plot=True)
    
    T_lc = np.zeros((R_lc.shape[0], 4, 4))
    T_lc[:, :3, :3] = R_lc
    T_lc[:, :3, 3] = t_lc
    T_lc[:, 3, 3] = 1

    # Viz pose distribution
    poses = []
    hole_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    hole_pose.transform(np.eye(4))
    for pose in T_lc:
        pose_transform = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        pose_transform.transform(pose)
        poses.append(pose_transform)
    o3d.visualization.draw_geometries([hole_pose] + poses)
    plt.show()

def run_collection(save_dir, calibration, t_hc_bounds, N_sample, lock_diameter, T_k2f, seed, i_sample):
    from realsense import RealSense
    import rtde_control
    import rtde_receive

    K = np.load(os.path.join(calibration, 'rgb_intrinsics.npy'))
    T_c2f = np.load(os.path.join(calibration, 'camera_to_flange.npy'))

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'rgb_overlay'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'depth'), exist_ok=True)
    np.savetxt(os.path.join(save_dir, 'cam_K.txt'), K)

    # Start robot interfaces
    rtde_c = rtde_control.RTDEControlInterface("10.0.0.78")
    rtde_r = rtde_receive.RTDEReceiveInterface("10.0.0.78")

    # Setting camera frame as the tool frame
    rtde_c.setTcp(T2P(T_c2f))

    # Assume the key is currently at the lock and aligned with the hole
    # Retrieve the corresponding pose of the camera (T_ca2w)
    # Use T_ca2w to compute the corresponding pose of the key (T_kh2w)
    # Use T_ca2w and T_kh2w to construct the pose of the camera if it were
    # at the lock and aligned with the hole (T_ch2w)
    user_choice = input("Attention User!!! Using previously saved camera origin pose for GT computation. Press 'l' to load or 'm' to measure...")
    if user_choice == 'l':
        if os.path.exists(os.path.join(save_dir, 'camera_align_pose.npy')):
            print("Loading camera origin pose from file...")
            P_ca2w = np.load(os.path.join(save_dir, 'camera_align_pose.npy'))
        else:
            user_choice = input("Attention User!!! Using current camera origin pose for GT computation. Press 'c' to continue... or 'exit' to exit.")
            if user_choice == 'c':
                P_ca2w = rtde_r.getActualTCPPose()
                np.save(os.path.join(save_dir, 'camera_align_pose.npy'), P_ca2w)
            elif user_choice == 'e':
                exit(0)
            else:
                print("Invalid choice. Exiting...")
                exit(0)
    elif user_choice == 'm':
        print("Measuring camera origin pose...")
        P_ca2w = rtde_r.getActualTCPPose()
        np.save(os.path.join(save_dir, 'camera_align_pose.npy'), P_ca2w)
    else:
        print("Invalid choice. Exiting...")
        exit(0)

    T_ca2w = P2T(P_ca2w)
    T_kh2w = T_ca2w @ np.linalg.inv(T_c2f) @ T_k2f
    T_ch2w = np.eye(4)
    T_ch2w[:3,:3] = T_ca2w[:3,:3]
    T_ch2w[:3,3] = T_kh2w[:3,3]

    # Sample camera poses in the lock (hole) frame
    # These poses assume that the hole coordinate frames orientation
    # is aligned with the camera's axis
    t_c2ch_array, R_c2ch_array, p_i_list = sample_poses(N_sample,
        t_hc_bounds[:,0], t_hc_bounds[:,1], t_hc_bounds[:,2],
        K, res=(848, 480), mount_center_in_lock_frame=(0.0, 0.0, 0.0),
        lock_diameter=lock_diameter, roll_limit_deg=[-45, 45],
        seed=seed, plot=False)

    if i_sample is not None:
        t_c2ch_array = t_c2ch_array[i_sample][None, :]
        R_c2ch_array = R_c2ch_array[i_sample][None, :]
        p_i_list = [p_i_list[i_sample]]

    # Discarding first frames from realsense 
    camera = RealSense(align_color=True, structured_light=1)
    for _ in range(50):
        _ = camera.get_aligned_rgbd()
    cv2.namedWindow("image")
    cv2.imshow("image", camera.get_aligned_rgbd()[1])
    cv2.waitKey(500)

    all_T_ch2c = []
    for i, (t_c2ch, R_c2ch, p_i) in enumerate(zip(t_c2ch_array, R_c2ch_array, p_i_list)):
        T_c2ch = P2T(np.concatenate((t_c2ch, R.from_matrix(R_c2ch).as_rotvec())))
        T_c2w_desired = T_ch2w @ T_c2ch
        rtde_c.moveL(T2P(T_c2w_desired), 0.15)
        time.sleep(0.2)

        T_c2w = np.eye(4)
        curr_pose = rtde_r.getActualTCPPose()
        T_c2w[:3,:3] = R.from_rotvec(curr_pose[3:]).as_matrix()
        T_c2w[:3,3] = curr_pose[:3]
        T_ch2c = np.linalg.inv(T_c2w) @ T_ch2w
        all_T_ch2c.append(T_ch2c)

        depth, rgb = camera.get_aligned_rgbd()
        color_filename = os.path.join(save_dir, f"rgb/{i:06d}.png")
        depth_filename = os.path.join(save_dir, f"depth/{i:06d}.png")
        cv2.imwrite(depth_filename, depth)
        cv2.imwrite(color_filename, rgb)

        for p in p_i:
            cv2.circle(rgb, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)

        T_ch2c = np.linalg.inv(T_c2ch)
        o = (K @ T_ch2c[:3,3].reshape(3,1)).flatten()
        o /= o[2]
        px = (K @ (T_ch2c[:3,3].reshape(3,1) + 0.1*T_ch2c[:3,0].reshape(3,1))).flatten()
        px /= px[2]
        py = (K @ (T_ch2c[:3,3].reshape(3,1) + 0.1*T_ch2c[:3,1].reshape(3,1))).flatten()
        py /= py[2]
        pz = (K @ (T_ch2c[:3,3].reshape(3,1) + 0.1*T_ch2c[:3,2].reshape(3,1))).flatten()
        pz /= pz[2]

        cv2.line(rgb, (int(o[0]), int(o[1])), (int(px[0]), int(px[1])), (0, 0, 255), 2)
        cv2.line(rgb, (int(o[0]), int(o[1])), (int(py[0]), int(py[1])), (0, 255, 0), 2)
        cv2.line(rgb, (int(o[0]), int(o[1])), (int(pz[0]), int(pz[1])), (255, 0, 0), 2)

        color_overlay_filename = os.path.join(save_dir, f"rgb_overlay/{i:06d}.png")
        cv2.imwrite(color_overlay_filename, rgb)

        cv2.imshow("image", rgb)
        cv2.waitKey(0)

    all_T_ch2c = np.array(all_T_ch2c)
    np.save(os.path.join(save_dir, "hole_poses.npy"), all_T_ch2c)


def convert_png_to_jpg(png_path, quality=95):
    # Ensure file has .png extension
    if not png_path.lower().endswith('.png'):
        raise ValueError("Input file must be a .png")

    # Load the image
    img = Image.open(png_path).convert("RGB")  # Ensure no alpha channel for JPG

    # Create .jpg path
    jpg_path = os.path.splitext(png_path)[0] + '.jpg'

    # Save as JPG
    img.save(jpg_path, 'JPEG', quality=quality)
    return jpg_path

def convert_jpg_to_png(jpg_path):
    # Ensure file has .jpg extension
    if not jpg_path.lower().endswith('.jpg'):
        raise ValueError("Input file must be a .jpg")

    # Load the image
    img = Image.open(jpg_path).convert("RGB")

    # Create .png path
    png_path = os.path.splitext(jpg_path)[0] + '.png'

    # Save as PNG
    img.save(png_path, 'PNG')
    return png_path

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def get_click_prompts(video_dir, frame_names, frame_idx):
    # Initialize lists to store points and labels
    input_points = []
    input_labels = []
    image = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))
    # Click event callback
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            input_points.append([x, y])
            input_labels.append(1)  # Foreground object

            # Draw the point on the image for visualization
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Click to Segment", image)
        elif event == cv2.EVENT_RBUTTONDOWN:
            input_points.append([x, y])
            input_labels.append(0)  # Background object

            # Draw the point on the image for visualization
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Click to Segment", image)
    
    # Display image and set callback
    cv2.namedWindow("Click to Segment", cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Click to Segment", image)
    cv2.setMouseCallback("Click to Segment", click_event)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press Enter to segment
            # Convert lists to numpy arrays
            points = np.array(input_points, np.int16)
            labels = np.array(input_labels, np.int16)
            break

    return points, labels

def estimate_mask(sequence):
    masks_dir = os.path.join(sequence, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    config_name = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    video_dir = f"{sequence}/rgb"
    save_folder = f"{sequence}/masks"

    device = torch.device("cuda")

    predictor = build_sam2_video_predictor(config_name, sam2_checkpoint, device=device)

    video_files_png = glob.glob(os.path.join(video_dir, "*.png"))
    video_files = []
    for video_file in video_files_png:
        jpg_file = convert_png_to_jpg(video_file)
        video_files.append(jpg_file)

    # Depending on your file naming convention, you may need to sort the files differently
    # frame_names = sorted(video_files, key=lambda f: int(re.search(r'(\d+)\.jpg', f).group(1)))
    frame_names = sorted([os.path.basename(f) for f in video_files])
    frame_names_pngs = sorted([os.path.basename(f) for f in video_files_png])

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    points, labels = get_click_prompts(video_dir, frame_names, 0)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )

    # Show the segmentation results
    # plt.figure(figsize=(9, 6))
    # plt.title(f"frame {ann_frame_idx}")
    # plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    # show_points(points, labels, plt.gca())
    # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    # plt.show()

    video_segments = {}  
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Render the segmentation results every few frames
    # You can adjust the vis_frame_stride to control how many frames to skip
    vis_frame_stride = 1
    plt.close("all")

    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        new_prompt = [False]
        def on_press(event):
            if event.key == 'r':
                new_prompt[0] = True
                print('Resetting the figure')
                plt.close(event.canvas.figure)

        while True:
            fig = plt.figure(figsize=(6, 4))
            fig.canvas.mpl_connect('key_press_event', on_press)

            plt.title(f"frame {out_frame_idx}")
            
            img = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
            plt.imshow(img)
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                plt.show()
                image_np = np.array(img)

                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() 
                white_background = torch.ones_like(image_tensor) * 255
                black_background = torch.zeros_like(image_tensor)
                segmented = torch.where(torch.from_numpy(out_mask), white_background, black_background)
                segmented_np = segmented.permute(1, 2, 0).byte().numpy()

                # Display the segmented image
                # plt.imshow(segmented_np)
                # plt.axis("off")
                # plt.title("Segmented Image (Background White)")
                # plt.show()

                # Save the segmented image
                Image.fromarray(segmented_np).save(os.path.join(save_folder, frame_names_pngs[out_frame_idx]))

            if new_prompt[0]:
                new_prompt[0] = False
                predictor.clear_all_prompts_in_frame(inference_state, out_frame_idx, 1, need_output=False)
                inference_state = predictor.init_state(video_path=video_dir)
                predictor.reset_state(inference_state)
                points, labels = get_click_prompts(video_dir, frame_names, out_frame_idx)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=out_frame_idx,
                    obj_id=1,
                    points=points,
                    labels=labels,
                )
                video_segments = {}  
                for out_frame_idx_, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx_] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
            else:
                break
    
    for file_path in video_files:
        if os.path.isfile(file_path):
            os.remove(file_path)

def compute_orientation_errors(pred_poses, gt_poses):
    """
    Compute the mean and standard deviation of orientation errors in roll, pitch, yaw.
    
    Args:
        pred_poses (np.ndarray): Predicted poses of shape (N, 4, 4).
        gt_poses (np.ndarray): Ground truth poses of shape (N, 4, 4).
        
    Returns:
        dict: Mean and standard deviation of roll, pitch, and yaw errors.
    """
    assert pred_poses.shape == gt_poses.shape, "Predicted and GT poses must have the same shape"
    assert pred_poses.shape[1:] == (4, 4), "Poses must be (N, 4, 4) transformation matrices"
        
    pred_orientations = []
    gt_orientations = []
    errors = []
    for i in range(pred_poses.shape[0]):
        pred_rot = R.from_matrix(pred_poses[i, :3, :3])  # Extract rotation matrix
        gt_rot = R.from_matrix(gt_poses[i, :3, :3])
        
        error_rot = gt_rot.inv() * pred_rot  # Compute relative rotation
        err_ypr = error_rot.as_euler('YXZ', degrees=True)  # Convert to roll, pitch, yaw errors

        pred_ypr = pred_rot.as_euler('YXZ', degrees=True)
        pred_orientations.append(pred_ypr)
        gt_orientations.append(gt_rot.as_euler('YXZ', degrees=True))
        errors.append(err_ypr)

    pred_orientations = np.array(pred_orientations)
    gt_orientations = np.array(gt_orientations)
    errors = np.array(errors)

    mean_errors = np.mean(np.abs(errors), axis=0)
    std_errors = np.std(np.abs(errors), axis=0)
    
    return {
        "mean_abs_yaw_error": mean_errors[0].item(),
        "std_abs_yaw_error": std_errors[0].item(),
        "mean_abs_pitch_error": mean_errors[1].item(),
        "std_abs_pitch_error": std_errors[1].item(),
        "mean_abs_roll_error": mean_errors[2].item(),
        "std_abs_roll_error": std_errors[2].item(),
    }, pred_orientations, gt_orientations, errors

def compute_position_errors(pred_poses, gt_poses):
    """
    Compute the mean and standard deviation of position errors in x, y, z coordinates.
    
    Args:
        pred_poses (np.ndarray): Predicted poses of shape (N, 4, 4).
        gt_poses (np.ndarray): Ground truth poses of shape (N, 4, 4).
        
    Returns:
        dict: Mean and standard deviation of position errors in x, y, z coordinates.
    """
    assert pred_poses.shape == gt_poses.shape, "Predicted and GT poses must have the same shape"
    assert pred_poses.shape[1:] == (4, 4), "Poses must be (N, 4, 4) transformation matrices"
    
    pred_positions = []
    gt_positions = []
    errors = []
    for i in range(pred_poses.shape[0]):
        pred_pos = pred_poses[i, :3, 3]  # Extract translation vector
        gt_pos = gt_poses[i, :3, 3]    
        error_pos = gt_pos - pred_pos  # Compute position error

        pred_positions.append(pred_pos)
        gt_positions.append(gt_pos)
        errors.append(error_pos)
  
    pred_positions = np.array(pred_positions)
    gt_positions = np.array(gt_positions)
    errors = np.array(errors)

    mean_errors = np.mean(np.abs(errors), axis=0)
    std_errors = np.std(np.abs(errors), axis=0)
    
    return {
        "mean_abs_x_error": mean_errors[0].item(),
        "std_abs_x_error": std_errors[0].item(),
        "mean_abs_y_error": mean_errors[1].item(),
        "std_abs_y_error": std_errors[1].item(),
        "mean_abs_z_error": mean_errors[2].item(),
        "std_abs_z_error": std_errors[2].item(),
    }, pred_positions, gt_positions, errors

def esimate_accuracy(sequence):
    gt_poses = np.load(os.path.join(sequence, "hole_poses.npy"))
    pred_poses = np.load(os.path.join(sequence, "predicted_hole_poses.npy"))

    orientation_stats, pred_orientations, gt_orientations, err_orientations = compute_orientation_errors(pred_poses, gt_poses)
    position_stats, pred_positions, gt_positions, err_positions = compute_position_errors(pred_poses, gt_poses)

    pprint(orientation_stats)
    pprint(position_stats)

    plt.subplots(2, 3, figsize=(15, 10))
    names = ['yaw', 'pitch', 'roll']
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.title(names[i])
        gt = gt_orientations[:, i]
        plt.plot((np.min(gt), np.max(gt)), (np.min(gt), np.max(gt)), label='ideal', color='r')
        plt.scatter(gt_orientations[:, i], pred_orientations[:, i], label='est')
        plt.grid()
        plt.xlabel('gt')
        plt.ylabel('pred')
        plt.legend()
    names = ['x', 'y', 'z']
    for i in range(3):
        plt.subplot(2, 3, i+4)
        plt.title(names[i])
        gt = gt_positions[:, i]
        plt.plot((np.min(gt), np.max(gt)), (np.min(gt), np.max(gt)), label='ideal', color='r')
        plt.scatter(gt_positions[:, i], pred_positions[:, i], label='est')
        plt.grid()
        plt.xlabel('gt')
        plt.ylabel('pred')
        plt.legend()
    plt.show()

def move_to_lock(output_dir, calibration, T_k2f):
    import rtde_control
    import rtde_receive

    rtde_c = rtde_control.RTDEControlInterface("10.0.0.78")
    rtde_r = rtde_receive.RTDEReceiveInterface("10.0.0.78")

    T_c2f = np.load(os.path.join(calibration, "camera_to_flange.npy"))
    T_h2c = np.load(os.path.join(output_dir, "predicted_hole_poses.npy"))
    assert len(T_h2c.shape) == 3
    assert T_h2c.shape[0] == 1
    T_h2c = T_h2c[0]

    rtde_c.setTcp(np.zeros((6,)))

    T_f2w = np.eye(4)
    curr_pose = rtde_r.getActualTCPPose()
    T_f2w[:3,:3] = R.from_rotvec(curr_pose[3:]).as_matrix()
    T_f2w[:3,3] = curr_pose[:3]

    T_h2w = T_f2w @ T_c2f @ T_h2c

    P_k2f = np.zeros((6))
    P_k2f[:3] = T_k2f[:3,3]
    P_k2f[3:] = R.from_matrix(T_k2f[:3,:3]).as_rotvec()
    rtde_c.setTcp(P_k2f)

    # Move key to hole pose
    P_h2w = np.zeros(6)
    P_h2w[:3] = T_h2w[:3,3]
    P_h2w[3:] = R.from_matrix(T_h2w[:3,:3]).as_rotvec()
    rtde_c.moveL(P_h2w, 0.1,0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_sampler', action='store_true', help='Test the pose sampler and display visualizations')
    parser.add_argument('--collect', action='store_true', help='Collect data using the pose sampler and camera')
    parser.add_argument('--mask', action='store_true', help='Compute masks using SAM2')
    parser.add_argument('--accuracy', action='store_true', help='Calculate the accuracy scores and produce plots')
    parser.add_argument('--moveto', action='store_true', help='Place a key in the lock opening using visual pose estimate')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--N_sample', type=int, default=200)
    parser.add_argument('--i_sample', type=int, default=None, help='Move to particular sample out of the N_samples')
    parser.add_argument('--key_offset_x', type=float, default=0.0)
    parser.add_argument('--key_offset_y', type=float, default=0.005)
    parser.add_argument('--key_offset_z', type=float, default=0.135)
    parser.add_argument('--lock_type', type=str, default='pin_tumbler')
    parser.add_argument('--lock_diameter', type=float, default=0.055)
    parser.add_argument('--calibration', type=str, default='keyinsertion/data/calibration')
    parser.add_argument('--sequence_base', type=str, default='keyinsertion/data')
    args = parser.parse_args()
    if (not args.test_sampler
        and not args.collect
        and not args.mask
        and not args.accuracy
        and not args.moveto):
        print("Please specify an action: --test_sampler, --collect, --mask, --accuracy, or --moveto")
        exit(1)
    
    if args.test_sampler:
        test_pose_sampler(args.calibration)

    sequence_dir = os.path.join(args.sequence_base, args.lock_type)

    # Construct the transform from key to flange from arguments
    # The key frame is defined have the axis of a camera
    T_k2f = np.eye(4)
    T_k2f[:3,:3] = np.array([[0, -1, 0],
                            [1,  0, 0],
                            [0,  0, 1]])
    T_k2f[:3, 3] = [args.key_offset_x, args.key_offset_y, args.key_offset_z]

    if args.collect:
        # bounds on the position of the camera center in the hole frame
        t_hc_bounds = np.array([[-0.1, -0.1, -0.4],  # lower bounds
                                [ 0.1,  0.1, -0.2]]) # upper bounds
        run_collection(sequence_dir, args.calibration, t_hc_bounds, args.N_sample,
                       args.lock_diameter, T_k2f, args.seed, args.i_sample)

    if args.mask:
        estimate_mask(sequence_dir)

    if args.accuracy:
        esimate_accuracy(sequence_dir)

    if args.moveto:
        move_to_lock(sequence_dir, args.calibration, T_k2f)
