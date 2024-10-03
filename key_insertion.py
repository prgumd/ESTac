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
                continue

            if t_frame is not None:
                print('Warning pressure_sensor.py frame skip')
            t_frame = new_t_frame
            frame = new_frame
            break

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
                # if tracker_p is not None: # TODO is this check necessary
                #     # print(tracker_p - self.tracker.p0)
                #     frame_warped_back = np.array(hom_4p_I_W_p_all_jit(frame, tracker_p, np.eye(3), np.eye(3), 1, True, self.tracker.p0))

                #     points = tracker_p.reshape((2, -1))
                #     diff_warped = np.abs(frame_warped_back.astype(np.float32) - self.first_frame.astype(np.float32)).astype(np.uint8)
                #     diff = np.abs(frame.astype(np.float32) - self.first_frame.astype(np.float32)).astype(np.uint8)
                #     cv2.line(frame, (int(points[0, 0]), int(points[1, 0])), (int(points[0, 1]), int(points[1, 1])), thickness=2, color=(255, 255, 255))
                #     cv2.line(frame, (int(points[0, 1]), int(points[1, 1])), (int(points[0, 2]), int(points[1, 2])), thickness=2, color=(255, 255, 255))
                #     cv2.line(frame, (int(points[0, 2]), int(points[1, 2])), (int(points[0, 3]), int(points[1, 3])), thickness=2, color=(255, 255, 255))
                #     cv2.line(frame, (int(points[0, 3]), int(points[1, 3])), (int(points[0, 0]), int(points[1, 0])), thickness=2, color=(255, 255, 255))

                #     for i in range(points.shape[1]):
                #         cv2.circle(frame, (int(points[0, i]), int(points[1, i])), radius=4, color=(255, 0, 0), thickness=-1)
                    
                #     p0_points = np.array(self.tracker.p0).reshape((2,4))
                #     for i in range(p0_points.shape[1]):
                #         cv2.circle(frame, (int(p0_points[0, i]), int(p0_points[1, i])), radius=4, color=(0, 0, 255), thickness=-1)

                #     full_frame = np.hstack((self.first_frame, frame, 20*diff,4*diff_warped))
                #     cv2.imshow('gelsense', full_frame)

                #     # Diff images to see biases in incoming frames
                #     if self.last_frame_float is None:
                #         self.last_frame_float = np.copy(frame_float)
                #     cv2.imshow('float diff',
                #                np.hstack((10*(frame_float - self.first_frame_float) + 0.5,
                #                10*(frame_float - self.last_frame_float) + 0.5)))
                #     self.last_frame_float = np.copy(frame_float)

                #     cv2.waitKey(1)

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




if __name__ == "__main__":

    rtde_c = rtde_control.RTDEControlInterface("10.0.0.78")
    rtde_r = rtde_receive.RTDEReceiveInterface("10.0.0.78")
    # Parameters
    velocity = 0.01
    acceleration = 0.5
    dt_rtde = 1/125.0
    lookahead_time = 0.06
    gain = 100
    
    tool_offset_pose = np.zeros((6))
    tool_offset_pose[2] += 0.112
    rtde_c.setTcp(tool_offset_pose)

    output_name = "knob_0_-1.9__0_0_0_trail1"
    shift = 0.018
    # Hole position. I imagine the position fixed in the center of a 10x10 grid with a -2 depth
    y_h = rtde_r.getActualTCPPose()[1]+shift
    # Amplitudes of perturbations
    A_x, A_z, A_y = 0.00, 0.00, 0.0005 # 0.00, 0.00, 0.0005  
    Ap_x, Ap_z, Ap_y = 0.0002, 0.0002, 0.0005 # 0.0002, 0.0002, 0.0005
    A_xrot, A_zrot, A_yrot = 0.0, 0.0, 0.0
    Ap_xrot, Ap_zrot, Ap_yrot = 3*np.pi/800, 3*np.pi/800, 3*np.pi/800 
    # Frequencies of perturbations
    omega_x, omega_z, omega_y = 0.9*2*np.pi, 0.83*2*np.pi, 0.7*2*np.pi
    omega_xrot, omega_zrot, omega_yrot = 1.05*2*np.pi, 1.0*2*np.pi, 0.95*2*np.pi

    # Time constant for high-pass filter
    hp_tau = 1.0/(2*np.pi*0.7)
    lp_tau = 1.0/(2*np.pi*1.59)  
    # Max interations parameter
    max_iterations = 25000
    # Rate of phase shift for composite sinusoidal movement  
    phase_shift_rate = 0.01
    phi1_x = 0.0
    phi1_y = 0.0 #-np.pi/2.0
    phi1_z = 0.0
    # Multiplier for pressure error  
    lambda_p = 0.0005
    # Multiplier for arena error  
    lambda_a = 0.0 # 0.001

    print(f"yh :: {y_h}")
    init_pose = rtde_r.getActualTCPPose()
    print("Initial Pose :: ",init_pose)
    x,y,z = init_pose[0], init_pose[1], init_pose[2]
    init_orientation = Rotation.from_rotvec(init_pose[3:])
    eulers = init_orientation.as_euler('XYZ')
    eulers_init = np.copy(eulers)
    x_init,y_init,z_init = init_pose[0], init_pose[1], init_pose[2]
    
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

    sensor = PressureSensor(name="./outputs/gelsight/"+output_name)
    for _ in range(5000):
        sensor.pressure_error()
    p_error = 0.0

    no_pressure_frame = False
    try:
        tstart = time.time()
        t_last = None
        while iteration < max_iterations:
            if not no_pressure_frame:
                t = time.time()-tstart
                if t_last is not None:
                    dt = t-t_last
                else:
                    dt = 1/9.0
                t_last = t

                # Ensuring the inital HP filter response 
                # doesn't affect the estimated parameters
                # by setting a zero gain on all parameters
                # until the filter settles down                
                if t < 2.0:
                    k_xz,k_y = 0.0, 0.0
                    kp_xz,kp_y = 0.0, 0.0
                    k_xrot,kp_xrot = 0.0, 0.0
                    k_zrot,kp_zrot = 0.0, 0.0
                    k_yrot,kp_yrot = 0.0, 0.0
                else:
                    k_xz = 0.0  #0.0 Gain for updating position in the XY plane
                    k_y = 1.1  #1.1 Gain for updating position in the Z direction
                    # Gain for updating position while pressure is experienced
                    kp_xz = 0.7 #0.7  # Gain for updating position in the XY plane
                    kp_y = 1.1  #1.1 Gain for updating position in the 
                    k_xrot,kp_xrot = 0.00, 10.0
                    k_zrot,kp_zrot = 0.00, 10.0
                    k_yrot,kp_yrot = 0.000, 10.0

                # Modulating Signal
                # if p_error > 0.0 or bool_press:
                bool_press = True
                perturbation_x = Ap_x * np.sin(omega_x * t)
                perturbation_y = Ap_y * np.sin(omega_y * t)
                perturbation_z = Ap_z * np.sin(omega_z * t)
                perturbation_xrot = Ap_xrot * np.sin(omega_xrot * t)
                perturbation_yrot = Ap_yrot * np.sin(omega_yrot * t)
                perturbation_zrot = Ap_zrot * np.sin(omega_zrot * t)
                # else:
                #     perturbation_x = A_x * np.sin(omega_x * t)
                #     perturbation_y = A_y * np.sin(omega_y * t)
                #     perturbation_z = A_z * np.sin(omega_z * t)
                #     perturbation_xrot = A_xrot * np.sin(omega_xrot * t)
                #     perturbation_yrot = A_yrot * np.sin(omega_yrot * t)
                #     perturbation_zrot = A_zrot * np.sin(omega_zrot * t)

                # Apply perturbations to the current position and rotation
                t_start = rtde_c.initPeriod()
                target_pose = rtde_r.getActualTCPPose()
                
                # ROBOT SAFETY!!!
                # limiting actions so that arm won't do extreme movements
                if abs(target_pose[0]- x - perturbation_x) < 0.005: 
                    target_pose[0] = x + perturbation_x
                else:
                    print("Movement commanded :: ",target_pose[0]- x - perturbation_x)
                    print("Peturbation part of it :: ",perturbation_x)
                    print("Shift in estimated parameter :: ",target_pose[0]- x)
                    print("Current value :: ",target_pose[0])
                    print("Estimated value :: ",x)
                    print("X movement limit exceeded!!!!!!!")
                    break
                if abs(target_pose[1]- y - perturbation_y) < 0.005: 
                    target_pose[1] = y + perturbation_y
                else:
                    print("Movement commanded :: ",target_pose[1]- y - perturbation_y)
                    print("Peturbation part of it :: ",perturbation_y)
                    print("Shift in estimated parameter :: ",target_pose[1]- y)
                    print("Current value :: ",target_pose[1])
                    print("Estimated value :: ",y)
                    print("Y movement limit exceeded!!!!!!!")
                    break
                if abs(target_pose[2]- z - perturbation_z) < 0.005: 
                    target_pose[2] = z + perturbation_z
                else:
                    print("Movement commanded :: ",target_pose[2]- z - perturbation_z)
                    print("Peturbation part of it :: ",perturbation_z)
                    print("Shift in estimated parameter :: ",target_pose[2]- z)
                    print("Current value :: ",target_pose[2])
                    print("Estimated value :: ",z)
                    print("Z movement limit exceeded!!!!!!!")
                    break
                
                delta_angles = np.array([perturbation_xrot,perturbation_zrot,perturbation_yrot])
                new_angles = eulers+delta_angles
                
                # Clamping the actions to search arena size
                target_pose[0] = min(max(target_pose[0],x_init-0.01),x_init+0.01)
                target_pose[1] = min(max(target_pose[1],y_init-0.01),y_init+0.02)
                target_pose[2] = min(max(target_pose[2],z_init-0.01),z_init+0.01)
                new_angles[0] = min(max(new_angles[0],eulers_init[0]-0.5),eulers_init[0]+0.5)
                new_angles[1] = min(max(new_angles[1],eulers_init[1]-0.5),eulers_init[1]+0.5)
                new_angles[2] = min(max(new_angles[2],eulers_init[2]-0.5),eulers_init[2]+0.5)
                
                target_orientation = Rotation.from_euler('XYZ',new_angles)
                target_rotvec = target_orientation.as_rotvec()

                target_pose[3] = target_rotvec[0]
                target_pose[4] = target_rotvec[1]
                target_pose[5] = target_rotvec[2]

                # init_orientation = Rotation.from_rotvec(init_pose[3:])
                # eulers = init_orientation.as_euler('XYZ')

                # Sending commands to the robot
                rtde_c.servoL(target_pose, velocity, acceleration, dt_rtde, lookahead_time, gain)
                rtde_c.waitPeriod(t_start)
                # time.sleep(0.3)
                # Measure position errors
                e_y = error_y()
                # Computing barrier function 
                current_pose = rtde_r.getActualTCPPose()
                x_a,y_a,z_a = current_pose[3], current_pose[4], current_pose[5]
                a_error = potential_well(x_a,z_a,y_a,x_init,z_init,y_init)
            
            p_error, points = sensor.pressure_error()
            if p_error is None:
                # print("frame missed")
                no_pressure_frame = True
                continue
            else:
                no_pressure_frame = False
            
            # Computing Composite Error
            composite_error = lambda_p*p_error + lambda_a*a_error + e_y
            if e_y <=0.0005:
                print("Key inserted!")
                break
            
            # Logging
            print(f" Iteration {iteration}: fps: {(1.0/dt):.4f} Error = {composite_error:.4f}, PError = {p_error:.4f}, AError = {a_error:.4f}, ZError = {e_y:.4f}")

            # Termination for too much pressure
            if sensor.flag:
                print("don't hurt me")
                break
            
            # Apply high-pass filter
            composite_error_hp = high_pass_filter(composite_error, composite_error_hp_prev,composite_error_prev, hp_tau, dt)
            composite_error_prev = composite_error
            composite_error_hp_prev = composite_error_hp
            
            p_error_hp = high_pass_filter(lambda_p*p_error, p_error_hp_prev,p_error_prev, hp_tau, dt)
            p_error_prev = lambda_p*p_error
            p_error_hp_prev = p_error_hp

            # Demodulation: Multiply the error difference by sin(omega * t)
            demodulated_x = composite_error_hp * np.sin(omega_x * t + phi1_x)
            demodulated_y = composite_error_hp * np.sin(omega_y * t + phi1_y)
            demodulated_z = composite_error_hp * np.sin(omega_z * t + phi1_z)
            demodulated_xrot = composite_error_hp * np.sin(omega_xrot * t)
            demodulated_yrot = composite_error_hp * np.sin(omega_yrot * t)
            demodulated_zrot = composite_error_hp * np.sin(omega_zrot * t)

            demodulated_x_lpf = low_pass_filter(demodulated_x, demodulated_x_prev, lp_tau, dt)
            demodulated_z_lpf = low_pass_filter(demodulated_z, demodulated_z_prev, lp_tau, dt)
            demodulated_y_lpf = low_pass_filter(demodulated_y, demodulated_y_prev, lp_tau, dt)
            demodulated_xrot_lpf = low_pass_filter(demodulated_xrot, demodulated_xrot_prev, lp_tau, dt)
            demodulated_zrot_lpf = low_pass_filter(demodulated_zrot, demodulated_zrot_prev, lp_tau, dt)
            demodulated_yrot_lpf = low_pass_filter(demodulated_yrot, demodulated_yrot_prev, lp_tau, dt)
            
            demodulated_x_prev = demodulated_x_lpf
            demodulated_z_prev = demodulated_z_lpf
            demodulated_y_prev = demodulated_y_lpf
            demodulated_xrot_prev = demodulated_xrot_lpf
            demodulated_zrot_prev = demodulated_zrot_lpf
            demodulated_yrot_prev = demodulated_yrot_lpf

            # Parameter Update: Update the position using the demodulated signals
            # if p_error > 0.0 or bool_press:
            bool_press = True
            x -= kp_xz * demodulated_x_lpf * dt
            z -= kp_xz * demodulated_z_lpf * dt
            y -= kp_y * demodulated_y_lpf * dt
            eulers[0] -= kp_xrot * demodulated_xrot_lpf * dt
            eulers[2] -= kp_yrot * demodulated_yrot_lpf * dt
            eulers[1] -= kp_zrot * demodulated_zrot_lpf * dt 
            # else:
            #     x -= k_xz * demodulated_x_lpf * dt
            #     z -= k_xz * demodulated_z_lpf * dt
            #     y -= k_y * demodulated_y_lpf * dt
            #     eulers[0] -= k_xzrot * demodulated_xrot_lpf * dt
            #     eulers[2] -= k_yrot * demodulated_yrot_lpf * dt
            #     eulers[1] -= k_xzrot * demodulated_zrot_lpf * dt

            # Logging History
            times.append(t)
            points_array.append(points)
            control_input.append(np.concatenate((target_pose[0:3],new_angles)).tolist())
            error_history.append(e_y)
            pressure_history.append(p_error)
            composite_error_history.append(composite_error)
            composite_error_history_hp.append(composite_error_hp)
            demodulated_signal.append([demodulated_x,demodulated_y,demodulated_z,
                           demodulated_xrot,demodulated_yrot,demodulated_zrot])
            demodulated_signal_lp.append([demodulated_x_lpf,demodulated_y_lpf,demodulated_z_lpf,
                           demodulated_xrot_lpf,demodulated_yrot_lpf,demodulated_zrot_lpf])
            trajectory.append([x, y, z])
            angle_trajectory.append(eulers.tolist())
            actual_trajectory.append(current_pose)
            iteration += 1

    except KeyboardInterrupt:
        pass

    rtde_c.stopScript()

    trajectory = np.array(trajectory)
    angle_trajectory = np.array(angle_trajectory)
    times = np.array(times)
    derrivative = detect_hole(trajectory)
    control_input = np.array(control_input)
    composite_error_history = np.array(composite_error_history)
    composite_error_history_hp = np.array(composite_error_history_hp)
    error_history = np.array(error_history)
    pressure_history = np.array(pressure_history)
    demodulated_signal = np.array(demodulated_signal)
    demodulated_signal_lp = np.array(demodulated_signal_lp)
    actual_trajectory = np.array(actual_trajectory)
    points_array = np.array(points_array)

    data_dict = {"Time": times,
                "Estimated Positions": trajectory,
                "Estimated Angles": angle_trajectory,
                "Control Input": control_input,
                "Total Error HP": composite_error_history_hp,
                "Total Error": composite_error_history,
                "Position Error": error_history,
                "Pressure Error": pressure_history,
                "Demodulated Signal": demodulated_signal,
                "Demodulated Signal LP": demodulated_signal_lp,
                "Actual Trajectory" : actual_trajectory,
                "Tracker": points_array,
                "Tool Offset": tool_offset_pose[2]}
    
    np.save("./outputs/final_exp/"+output_name,data_dict)

    data = adjust_data(data_dict)

    sensor.camera.stop.value = 1
    time.sleep(2.0)
    sensor.camera.join()
    time.sleep(2.0)
