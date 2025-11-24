if __name__ == '__main__':
    # When importing numpy, by default try to limit the number of threads
    import os; os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import numpy as np
    # When importing OpenCV, by default try to limit the number of threads
    import cv2; cv2.setNumThreads(1)
    import multiprocessing; multiprocessing.set_start_method('spawn')
else:
    import numpy as np
    import cv2

from vme_research.hardware.v4l2_camera import V4L2Camera
from vme_research.messaging.shared_ndarray import SharedNDArrayPubSub, SharedNDArrayPool
from vme_research.hardware.record import Record, Load, make_sequence_directory, get_latest_sequence_directory
from vme_research.algorithms.patch_track import (JAffineTrackRotInvariant, affine_I_W_p_all_jit,
                                                 JHomographyTrackRotInvariant, homography_I_W_p_all_jit,
                                                 JHom4pTrackRotInvariant, hom_4p_I_W_p_all_jit)


import rtde_control
import rtde_receive
import time
import sys
from multiprocessing import Value, Queue
from queue import Empty
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import cma
import copy 

AXIS_LABELS = {0:"X",1:"Y",2:"Z"}

def adjust_data(data):
    array_data = {}
    for key,value in data.items():
        if isinstance(value, np.ndarray):
            array_data[key] = value
    adjusted_data = {}
    end = np.min([value.shape[0] for value in array_data.values()])
    for key,value in array_data.items():
        adjusted_data[key] = value[:end]

    return adjusted_data

class TimeZeroSource:
    def __init__(self, t0=None):
        self.t0 = t0
        if self.t0 is None: self.t0 = time.time()

    def time(self):
        return time.time() - self.t0

class PressureSensor:
    def __init__(self, name="Image", loading=False, sequence=None, N_frameskip=2):
        stop = Value('i', 0)
        time_source = TimeZeroSource()
        ndarray_pool = None
        self.camera_pub_sub = SharedNDArrayPubSub(max_q_size=500,zero_copy=False)

        if loading:
            if sequence is None:
                sequence = get_latest_sequence_directory(name)
            loader = Load(sequence)
            self.camera = V4L2Camera(stop, time_source, device='/dev/video0', loader=loader, pub_sub=self.camera_pub_sub, ndarray_pool=ndarray_pool)
        else:
            # sequence_directory = make_sequence_directory(name)
            # recorder = Record(save_directory=sequence_directory, time_source=time_source)
            self.camera = V4L2Camera(stop, time_source, device='/dev/video0', recorder=None, pub_sub=self.camera_pub_sub, ndarray_pool=ndarray_pool, fps=9.0)
        self.last_t_frame = None
        self.tracker = None
        self.first_frame = None
        self.tracker_t_list = []
        self.tracker_p_list = []
        self.error = 0.0
        self.p0 = None
        self.camera.start()
        self.N_frameskip = N_frameskip
        self.N_skipped = 0
        self.first_frame_float = None
        self.last_frame_float = None
        self.flag = False
        
    def pressure_error(self):
        t_frame = None
        frame = None
        while self.camera_pub_sub.size() > 0:
            try:
                new_t_frame, new_frame = self.camera_pub_sub.get(timeout=0.001)
            except Empty:
                break
                # continue

            if t_frame is not None:
                print('Warning pressure_sensor.py frame skip')
            t_frame = new_t_frame
            frame = new_frame
            # break

        if t_frame is None:
            time.sleep(0.001)
            # print("t_frame is None")
            return None, None

        if self.N_skipped < self.N_frameskip:
            self.N_skipped += 1
            # print("N skip is low")
            return None, None
        # print(self.N_skipped, self.N_frameskip)

        frame = cv2.resize(frame, (320, 240))
        
        if self.last_t_frame is None:
            self.last_t_frame = t_frame
        
        error = None
        points = None
        if self.last_t_frame != t_frame:
            s = 1 / 10.0
            rect = [int(frame.shape[1]*s), int(frame.shape[0]*s), frame.shape[1]-int(frame.shape[1]*s), frame.shape[0]-int(frame.shape[0]*s)]
            self.p0 = np.array([[rect[0],rect[1],1],[rect[0],rect[3],1],[rect[2],rect[3],1],[rect[2],rect[1],1]]).astype(np.float32).T
            frame_float = cv2.cvtColor(frame.astype(np.float32) / 255.0, cv2.COLOR_BGR2GRAY)
            if self.tracker is None:
                try:
                    self.tracker = JHom4pTrackRotInvariant(
                    # self.tracker = JAffineTrackRotInvariant(
                        rect=rect,
                        template_image=frame_float,
                        R_c_fc=np.eye(3),
                        K=np.eye(3),
                        delta_p_stop=0.0001,
                        stride=1.0,
                        max_steps=250,
                        blur_new_frame=True,
                        )
                    self.first_frame = np.copy(frame)
                    self.first_frame_float = frame_float

                    # self.tracker = AffineTrackRotInvariant(
                    #             patch_coordinates=rect,
                    #             template_image=frame_float,
                    #             template_q_c_to_fc=template_q_c_to_fc,
                    #             K=K,
                    #             delta_p_stop=0.1,
                    #             delta_p_mult=1.0,
                    #             visualize=False,
                    #             visualize_verbose=False,
                    #             wait_key=0,
                    #             stride=3.0,
                    #             inverse=True,
                    #             max_update_time=0.02
                    #             )
                    print("Tracker created")

                except np.linalg.LinAlgError:
                    print('Could not create tracker')
                    self.tracker = None

            if self.tracker is not None:
                tracker_p = self.tracker.update(frame_gray=frame_float, R_c_fc=np.eye(3))
                # tracker_p = tracker_p.reshape((2,-1)) @ self.p0
                # tracker_p = tracker_p.reshape(-1)
                self.tracker_t_list.append(t_frame)
                self.tracker_p_list.append(tracker_p)
                self.last_t_frame = t_frame

                # if len(self.tracker_p_list) > 1:
                # error = np.linalg.norm(self.tracker_p_list[-1].reshape((2, -1))-self.tracker.p0.reshape((2,-1)), axis=0)
                # error = (self.tracker_p_list[-1].reshape((2, -1))-self.tracker.p0.reshape((2,-1)))
                error = np.linalg.norm(self.tracker_p_list[-1] - np.array(self.tracker.p0))
                if error < 3.0:
                    error = 0.0
                elif error > 15.0:
                    self.flag = True
                points = tracker_p.reshape((2, -1))
                if tracker_p is not None: # TODO is this check necessary
                    # print(tracker_p - self.tracker.p0)
                    frame_warped_back = np.array(hom_4p_I_W_p_all_jit(frame, tracker_p, np.eye(3), np.eye(3), 1, True, self.tracker.p0))

                    points = tracker_p.reshape((2, -1))
                    diff_warped = np.abs(frame_warped_back.astype(np.float32) - self.first_frame.astype(np.float32)).astype(np.uint8)
                    diff = np.abs(frame.astype(np.float32) - self.first_frame.astype(np.float32)).astype(np.uint8)
                    cv2.line(frame, (int(points[0, 0]), int(points[1, 0])), (int(points[0, 1]), int(points[1, 1])), thickness=2, color=(255, 255, 255))
                    cv2.line(frame, (int(points[0, 1]), int(points[1, 1])), (int(points[0, 2]), int(points[1, 2])), thickness=2, color=(255, 255, 255))
                    cv2.line(frame, (int(points[0, 2]), int(points[1, 2])), (int(points[0, 3]), int(points[1, 3])), thickness=2, color=(255, 255, 255))
                    cv2.line(frame, (int(points[0, 3]), int(points[1, 3])), (int(points[0, 0]), int(points[1, 0])), thickness=2, color=(255, 255, 255))

                    for i in range(points.shape[1]):
                        cv2.circle(frame, (int(points[0, i]), int(points[1, i])), radius=4, color=(255, 0, 0), thickness=-1)
                    
                    p0_points = np.array(self.tracker.p0).reshape((2,4))
                    for i in range(p0_points.shape[1]):
                        cv2.circle(frame, (int(p0_points[0, i]), int(p0_points[1, i])), radius=4, color=(0, 0, 255), thickness=-1)

                    full_frame = np.hstack((self.first_frame, frame, 20*diff,4*diff_warped))
                    cv2.imshow('gelsense', full_frame)

                    # Diff images to see biases in incoming frames
                    if self.last_frame_float is None:
                        self.last_frame_float = np.copy(frame_float)
                    cv2.imshow('float diff',
                               np.hstack((10*(frame_float - self.first_frame_float) + 0.5,
                               10*(frame_float - self.last_frame_float) + 0.5)))
                    self.last_frame_float = np.copy(frame_float)

                    cv2.waitKey(1)

        return error, points
            
# Error functions
def error_y():
    y = rtde_r.getActualTCPPose()[1]
    return abs(y - y_h)

# High-pass filter
def high_pass_filter(value, hp_prev_value, prev_value, tau, dt):
    alpha = tau / (tau + dt)
    return alpha * (hp_prev_value + value - prev_value)

# Low-pass filter
def low_pass_filter(value, prev_value, tau, dt):
    alpha = dt / (tau + dt)
    return prev_value + alpha * (value - prev_value)

def potential_well(x_a,y_a,z_a,x_init,y_init,z_init):
    x = x_a-x_init
    y = y_a-y_init

    a = 0.01
    k = 1000.0

    distance = np.sqrt(x**2 + y**2)
    if abs(x) <= a/2 and abs(y) <= a/2:
        return 0.0
    else:
        return min(np.exp(k * (distance - a/2)),20.0)


def detect_hole(trajectory,window_size = 100,dx=1.0):
    data = trajectory[:,1].reshape(-1)
    
    if window_size % 2 == 0:
        window_size += 1
    
    # Create the moving average filter
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='same')
    derivative = np.gradient(data, dx)

    return derivative[-1] > 0.00015

def objective_function(lambda_p):
    """Objective function for CMA-ES, combines pressure and position errors."""

    # Get the pressure and position error
    p_error, points = sensor.pressure_error()
    e_y = error_y()

    if p_error is None:
        return None, None, None, None
    # Composite error combining pressure, position, and rotation
    composite_error = lambda_p * p_error +  e_y

    return composite_error, p_error, e_y, points

def distance_SE3(r1,r2):
    r1 = Rotation.from_rotvec(r1)
    r2 = Rotation.from_rotvec(r2)
    
    q1 = r1.as_quat()  # Quaternion (x, y, z, w)
    q2 = r2.as_quat()
    
    distance = 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))
    return distance 

def interpolate_poses(T1, T2, rotvec1, rotvec2, translation_step=0.0002):
    
    # Compute translation interpolation steps
    total_translation_distance = np.linalg.norm(T2 - T1)
    num_translation_steps = max(int(total_translation_distance / translation_step), 1)
    translation_interp = [T1 + (T2 - T1) * (i / num_translation_steps) for i in range(num_translation_steps + 1)]

    rs = Rotation.from_rotvec([rotvec1,rotvec2])

    # Matching steps for smooth transition for now
    num_rotation_steps = num_translation_steps  

    slerp = Slerp([0,1],rs)
    rotation_interp = slerp(np.linspace(0,1,num_rotation_steps+1)).as_rotvec()

    # Combine translations and rotations
    interpolated_poses = [np.concatenate([translation_interp[i], rotation_interp[i]]) for i in range(1, num_translation_steps + 1)]

    return interpolated_poses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--key_offset_x', type=float, default=0.250)
    parser.add_argument('--key_offset_y', type=float, default=0.0)
    parser.add_argument('--key_offset_z', type=float, default=0.0)
    parser.add_argument('--penetration_depth', type=float, default=0.015)
    parser.add_argument('--output_name', type=str, default=f'pin_tumbler')
    parser.add_argument('--sequence_base', type=str, default='keyinsertion/data/cma/')
    args = parser.parse_args()


    # sensor = PressureSensor(name="./outputs/gelsight/"+"test")
    # p_list = []
    # for _ in range(5000):
    #     sensor.pressure_error()
    # try:
    #     while True:
    #         p,_ = sensor.pressure_error()
    #         if p is None:
    #             continue
    #         print(f'{p}')  # Use \r to return to the start of the line
    #         # sys.stdout.flush()
    # except KeyboardInterrupt:
    #     plt.plot(p_list)
    #     plt.show()

    rtde_c = rtde_control.RTDEControlInterface("10.0.0.78")
    rtde_r = rtde_receive.RTDEReceiveInterface("10.0.0.78")
    # Parameters
    velocity = 0.01
    acceleration = 0.5
    dt_rtde = 1/125.0
    lookahead_time = 0.06
    gain = 100
    tool_offset_z = args.key_offset_z
    tool_offset_y = args.key_offset_y
    tool_offset_x = args.key_offset_x
    
    tool_offset_pose = np.zeros((6))
    tool_offset_pose[2] = args.key_offset_z
    tool_offset_pose[1] = args.key_offset_y
    tool_offset_pose[0] = args.key_offset_x
    rtde_c.setTcp(tool_offset_pose)

    output_name = args.output_name
    shift = args.penetration_depth
    # Hole position. I imagine the position fixed in the center of a 10x10 grid with a -2 depth
    y_h = rtde_r.getActualTCPPose()[1]-shift

    # Max interations parameter
    max_iterations = 101

    # Multiplier for pressure error  
    lambda_p = 0.0004
    # Multiplier for arena error  
    lambda_a = 0.0 # 0.001

    print(f"yh :: {y_h}")
    init_pose = rtde_r.getActualTCPPose()
    print("Initial Pose :: ",init_pose)
    x,y,z = init_pose[0], init_pose[1], init_pose[2]
    x0,y0,z0 = init_pose[0], init_pose[1], init_pose[2]
    rx0 = init_pose[3]
    ry0 = init_pose[4]
    rz0 = init_pose[5]
    init_orientation = Rotation.from_rotvec(init_pose[3:])
    eulers = init_orientation.as_euler('XYZ')  # Extract initial Euler angles (rx, ry, rz)
    eulers_init = np.copy(eulers)
    x_init, y_init, z_init = init_pose[0], init_pose[1], init_pose[2]

    # Initialize CMA-ES with 6 parameters (x, y, z, rx, ry, rz)
    initial_pose = [x, y, z, eulers[0], eulers[1], eulers[2]]  # Initial guess for x, y, z and rotations
    sigma = 0.0003  # Exploration step size
    max_iterations = 101

    # **New part: initialize CMA-ES for translation and rotation**
    es = cma.CMAEvolutionStrategy(initial_pose, sigma, {
        'maxiter': max_iterations,
        'bounds': [[x_init - 0.05, y_init - 0.05, z_init - 0.05, eulers_init[0] - 15*np.pi/180, eulers_init[1] - 15*np.pi/180, eulers_init[2] - 15*np.pi/180],
                   [x_init + 0.05, y_init + 0.05, z_init + 0.05, eulers_init[0] + 15*np.pi/180, eulers_init[1] + 15*np.pi/180, eulers_init[2] + 15*np.pi/180]],
        'CMA_diagonal': 2,
        'AdaptSigma' : False,
        'popsize': 12
    })

    rscale_cma = 15.0
    tscale_cma = 1.0

    # Extremum seeking controller loop
    t = 0
    iteration = 0
    trajectory = []
    angle_trajectory = []
    actual_trajectory = []
    control_input = []
    error_history = []
    pressure_history = []
    composite_error_history = []
    composite_error_history_hp = []
    demodulated_signal = []
    demodulated_signal_lp = []
    times = []
    points_array = []
    composite_error_prev = shift
    composite_error_hp_prev = shift
    demodulated_x_prev = 0.0
    demodulated_z_prev = 0.0
    demodulated_y_prev = 0.0
    demodulated_xrot_prev = 0.0
    demodulated_zrot_prev = 0.0
    demodulated_yrot_prev = 0.0
    p_error_prev = 0.0
    p_error_hp_prev = 0.0
    hole_found = False
    phase_shift = 0
    counter = 0
    bool_press = False
    errors_list =[]
    points_list = []
    p_error_list = []
    e_y_list=[]

    sensor = PressureSensor(name="./keyinsertion/outputs/gelsight/"+output_name)
    for _ in range(5000):
        sensor.pressure_error()
    p_error = 0.0

    no_pressure_frame = False
    try:
        tstart = time.time()
        t_last = None
        while iteration < max_iterations:

            candidate_solutions = es.ask(12)
            # print(len(candidate_solutions))
            errors_list =[]
            points_list = []
            p_error_list = []
            e_y_list=[]
            loss_landscape = []
            for pose in candidate_solutions:
                t_start = rtde_c.initPeriod()
                x, y, z, rx, ry, rz = pose
                # Apply the candidate solution pose to the robot
                target_pose = rtde_r.getActualTCPPose()
                curr_pose = copy.deepcopy(target_pose)

                # Convert Euler angles to rotation vector and apply rotation
                target_orientation = Rotation.from_euler('XYZ', [rx, ry, rz])
                target_rotvec = target_orientation.as_rotvec()
                x = tscale_cma*(x-x0)+x0
                # y = tscale_cma*(y-y0)+y0
                z = tscale_cma*(z-z0)+z0
                
                x = max(-0.01, min(0.01, x-x0)) + x0
                # y = max(-shift, min(0.01, y-y0)) + y0
                z = max(-0.01, min(0.01, z-z0)) + z0
                
                target_rotvec[0] = rscale_cma*(target_rotvec[0]-rx0)+rx0
                target_rotvec[1] = rscale_cma*(target_rotvec[1]-ry0)+ry0
                target_rotvec[2] = rscale_cma*(target_rotvec[2]-rz0)+rz0

                target_poses = interpolate_poses(np.array(curr_pose[:3]), np.array([x,y,z]), np.array(curr_pose[3:]), target_rotvec)
                for target_pose in target_poses:
                    # Send the new pose to the robot
                    trans_dist = np.linalg.norm(np.array(target_pose[:3])-np.array(curr_pose[:3]))
                    rot_dist = distance_SE3(target_pose[3:],curr_pose[3:])
                    # thresholds as 0.2 mm and ~1 deg
                    while trans_dist > 3e-4 or rot_dist > 0.01:
                        rtde_c.servoL(target_pose, velocity, acceleration, dt_rtde, lookahead_time, gain)
                        rtde_c.waitPeriod(t_start)
                        p_error,_ = sensor.pressure_error()
                        while p_error is None:
                            p_error,_ = sensor.pressure_error()                        
                            # print(p_error)
                        
                        if p_error > 9.0:
                            break
                        curr_pose = rtde_r.getActualTCPPose()
                        trans_dist = np.linalg.norm(np.array(target_pose[:3])-np.array(curr_pose[:3]))
                        rot_dist = distance_SE3(target_pose[3:],curr_pose[3:])
                        # print(f"{trans_dist:.5f} {rot_dist:.3f}")
                    if p_error > 9.0:
                        print("pressure warning!")
                        break
                rtde_c.servoStop()
                # print("out")
                # **Get the tactile sensor error and store it**
                error, p_error, e_y, points = objective_function(lambda_p)
                while error is None: 
                    error, p_error, e_y, points = objective_function(lambda_p)
                
                # debugging log
                loss_landscape.append([x-x0,y-y0,z-z0])

                if sensor.flag:
                    print('pressure high!!!')
                    break

                # print(p_error)
                errors_list.append(error)
                p_error_list.append(p_error)
                e_y_list.append(e_y)
                points_list.append(points)
            print("------")
            print(np.array(loss_landscape))
            print("------")
            if sensor.flag:
                print('pressure high!!!')
                break
            # print("candidate solutions \n-----------------------")
            # print(np.array(candidate_solutions))
            # **Update CMA-ES based on the evaluated errors**
            es.tell(candidate_solutions, errors_list)
            print("error list \n----------------------")
            print(p_error_list)
            print("---------------")
            # Get the best solution found so far
            best_solution = es.result.xbest
            # print(best_solution)
            best_error = es.result.fbest
            # print(best_error)
            # index = errors_list.index(best_error)
            # points = points_list[index]
            # p_error = p_error_list[index]
            # e_y = e_y_list[index]

            # **Extract x, y, z, rx, ry, rz from the best solution**
            x, y, z, rx, ry, rz = best_solution

            # Apply the best solution pose to the robot
            curr_pose = rtde_r.getActualTCPPose()
            target_pose = copy.deepcopy(curr_pose)            

            # Convert Euler angles to rotation vector and apply rotation
            target_orientation = Rotation.from_euler('XYZ', [rx, ry, rz]) #TODO need to be verified
            target_rotvec = target_orientation.as_rotvec()
            x = tscale_cma*(x-x0)+x0
            # y = tscale_cma*(y-y0)+y0
            z = tscale_cma*(z-z0)+z0
            
            x = max(-0.01, min(0.01, x-x0)) + x0
            # y = max(-shift, min(0.01, y-y0)) + y0
            z = max(-0.01, min(0.01, z-z0)) + z0
            
            print("-------")
            print(x-x0,y-y0,z-z0)
            print("-------")
            target_rotvec[0] = rscale_cma*(target_rotvec[0]-rx0)+rx0
            target_rotvec[1] = rscale_cma*(target_rotvec[1]-ry0)+ry0
            target_rotvec[2] = rscale_cma*(target_rotvec[2]-rz0)+rz0

            target_poses = interpolate_poses(np.array(curr_pose[:3]), np.array([x,y,z]), np.array(curr_pose[3:]), target_rotvec)
            for target_pose in target_poses:
                # Send the new pose to the robot
                trans_dist = np.linalg.norm(np.array(target_pose[:3])-np.array(curr_pose[:3]))
                rot_dist = distance_SE3(target_pose[3:],curr_pose[3:])
                # thresholds as 0.1 mm and ~1 deg
                while trans_dist > 1e-4 or rot_dist > 0.01:
                    t_start = rtde_c.initPeriod()
                    rtde_c.servoL(target_pose, velocity, acceleration, dt_rtde, lookahead_time, gain)
                    rtde_c.waitPeriod(t_start)
                    p_error,_ = sensor.pressure_error()
                    while p_error is None:
                        p_error,_ = sensor.pressure_error()                        
                        # print(p_error)
                    
                    if p_error > 9.0:
                        break
                    curr_pose = rtde_r.getActualTCPPose()
                    trans_dist = np.linalg.norm(np.array(target_pose[:3])-np.array(curr_pose[:3]))
                    rot_dist = distance_SE3(target_pose[3:],curr_pose[3:])
                if p_error > 9.0:
                    print("pressure warning!")
                    break
            rtde_c.servoStop()

            objective_function(lambda_p)
            if sensor.flag:
                print("failed while execution")
                print('pressure high!!!')
                break

            current_pose = rtde_r.getActualTCPPose()

            # Log the results
            trajectory.append(best_solution)
            error_history.append(es.result.fbest)
            pressure_history.append(p_error)
            times.append(time.time() - tstart)

             # **Print progress and check for termination**
            best_error = es.result.fbest
            print(f"Iteration {iteration}: Best Error = {best_error}")
            
            # Logging History
            # times.append(t)
            # points_array.append(points)
            # control_input.append(target_pose)
            # error_history.append(e_y)
            # pressure_history.append(p_error)
            # composite_error_history.append(best_error)
            # trajectory.append([x, y, z])
            # angle_trajectory.append(eulers.tolist())
            # actual_trajectory.append(current_pose)
            iteration += 1

            if best_error <= 0.003:
                print("Key inserted!")
                break

    except KeyboardInterrupt:
        pass

    rtde_c.stopScript()

    # loss_landscape = np.array(loss_landscape)
    # np.save("./outputs/cma/"+output_name+"_loss_landscape",loss_landscape)
    trajectory = np.array(trajectory)
    # angle_trajectory = np.array(angle_trajectory)
    times = np.array(times)
    # derrivative = detect_hole(trajectory)
    # control_input = np.array(control_input)
    # composite_error_history = np.array(composite_error_history)
    error_history = np.array(error_history)
    pressure_history = np.array(pressure_history)
    # actual_trajectory = np.array(actual_trajectory)
    # points_array = np.array(points_array)

    data_dict = {"Time": times,
                "Estimated Positions": trajectory,
                # "Estimated Angles": angle_trajectory,
                # "Control Input": control_input,
                "Total Error": error_history,
                # "Position Error": error_history,
                "Pressure Error": pressure_history,
                # "Actual Trajectory" : actual_trajectory,
                # "Tracker": points_array,
                "Tool Offset": tool_offset_pose[2],
                "Shift": shift}

    sequence_dir = os.path.join(args.sequence_base, args.output_name)
    if not os.path.exists(sequence_dir):
        os.makedirs(sequence_dir)
    np.save(os.path.join(sequence_dir, "insertion_data.npy"), data_dict)

    data = adjust_data(data_dict)

    sensor.camera.stop.value = 1
    time.sleep(2.0)
    sensor.camera.join()
    time.sleep(2.0)
