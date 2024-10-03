from multiprocessing import Process
import time

import cv2
import mujoco # TODO lazy imports...
import numpy as np

V4L2CameraFieldsOptions = {
    'name': 'V4L2Camera',
    'version': '0.0.1',
    'fields': [{'name': 'frame', 'type': str(np.ndarray), 'split': True}],
    'append_fields': [{'name': 'res', 'type': str(list), 'split': False},
                      {'name': 'K', 'type': str(list), 'split': False},
                      {'name': 'dist', 'type': str(list), 'split': False}]
}

class V4L2Camera(Process):
    def __init__(self,
                 stop,
                 time_source,
                 device='/dev/video0',
                 res=(640,480), fps=60,
                 recorder=None, loader=None,
                 pub_sub=None, pub_sub_sim=None,
                 ndarray_pool = None,
                 K=None, dist=None, K_new=None, shape_new=None, model='planar'):
        super(V4L2Camera, self).__init__(daemon=True)

        self.stop = stop
        self.time_source = time_source
        self.device = device
        self.fps = fps
        self.recorder = recorder
        if self.recorder: self.recorder.set_fields_options(V4L2CameraFieldsOptions)
        self.loader = loader
        self.pub_sub = pub_sub
        self.pub_sub_sim = pub_sub_sim
        self.ndarray_pool = ndarray_pool

        # If loading from data and the user has not specified their own calibration
        if self.loader and K is None:
            self.res  = np.array(self.loader.get_appended()['res'])
            self.K    = np.array(self.loader.get_appended()['K'])
            self.dist = np.array(self.loader.get_appended()['dist'])
        else:
            self.res  = res
            self.K    = K
            self.dist = dist

        self.K_new = K_new
        self.shape_new = shape_new
        self.model = model
        if self.K_new is not None:
            if self.model == 'planar':
                self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.dist, np.eye(3), self.K_new, (self.shape_new[1], self.shape_new[0]), cv2.CV_32FC1)
            elif self.model == 'fisheye':
                self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.dist, np.eye(3), self.K_new, (self.shape_new[1], self.shape_new[0]), cv2.CV_32FC1)
            else:
                raise Exception('Unknown camera model')

        self.last_frame_t = None

    def post_process(self, t, data):
        frame_t = t
        frame = data[0]

        if self.K_new is not None:
            if self.ndarray_pool is None:
                frame_out = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
            else:
                frame_out = self.ndarray_pool.get()
                cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR, dst=frame_out.x)
        else:
            frame_out = frame

        if self.pub_sub is not None:
            self.pub_sub.pub((frame_t, frame_out), use_shm=[False, True])

    def run(self):
        cv2.setNumThreads(1)

        try:
            if self.loader:
                self.run_loader()
            else:
                self.run_live()
        except KeyboardInterrupt: pass
        finally:
            if self.recorder:
                if self.K is not None:
                    append_values = [list(self.res), self.K.tolist(), self.dist.tolist()]
                else:
                    append_values = [list(self.res), None, None]
                self.recorder.close(append_values)

            if self.pub_sub:
                self.pub_sub.cleanup()

    def run_live(self):
        if not self.pub_sub_sim:
            self.cam = cv2.VideoCapture(self.device, cv2.CAP_V4L)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.res[0])
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res[1])
            self.cam.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cam.set(cv2.CAP_PROP_FPS, self.fps)

        while self.stop.value == 0:
            if not self.pub_sub_sim:
                ret = self.cam.grab()
            else:
                ret = self.pub_sub_sim.size() > 0

            if ret:
                if self.ndarray_pool is not None and self.K_new is None:
                    if not self.pub_sub_sim:
                        frame = self.ndarray_pool.get()
                        ret, _ = self.cam.retrieve(frame.x)
                    else:
                        frame_t, frame_cam = self.pub_sub_sim.get()
                        frame = self.ndarray_pool.get()
                        frame.x[:] = frame_cam.x
                        ret = True
                else:
                    if not self.pub_sub_sim:
                        ret, frame = self.cam.retrieve()
                    else:
                        frame_t, frame = self.pub_sub_sim.get()
                        ret = True

                if ret:
                    if not self.pub_sub_sim:
                        frame_t = self.cam.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        
                        # Check sample duration
                        if self.last_frame_t is None:
                            self.last_frame_t = frame_t - 1.0 / self.fps
                        
                        # Check within 20%
                        if 1.2 / self.fps < frame_t - self.last_frame_t:
                            print('Warning v4l2_camera not acheiving target fps', self.fps, 1.0 / (frame_t - self.last_frame_t))
                        self.last_frame_t = frame_t

                    if self.recorder is not None:
                        self.recorder.pub(frame_t, (frame,))

                    self.post_process(frame_t, (frame,))
            else:
                time.sleep(0.001)

        if not self.pub_sub_sim:
            self.cam.release()

    def run_loader(self):
        while self.stop.value == 0:
            t = self.time_source.time()
            ret, frame_t, (frame,) = self.loader.get(t)

            if ret:
                if self.ndarray_pool and self.K_new is None:
                    frame_shm = self.ndarray_pool.get()
                    frame_shm.x[:] = frame
                    frame = frame_shm

                self.post_process(frame_t, (frame,))
            else:
                time.sleep(0.001)

class V4L2CameraSimulate():
    def __init__(self, v4l2_camera, mujoco_name):
        self.mujoco_name = mujoco_name
        self.pub_sub = v4l2_camera.pub_sub_sim
        self.cam_res = v4l2_camera.res
        self.fps = v4l2_camera.fps
        self.last_render_t = None

    def init(self, m, d):
        # Make all the things needed to render a simulated camera
        self.gl_ctx = mujoco.GLContext(*self.cam_res)
        self.gl_ctx.make_current()

        self.scn = mujoco.MjvScene(m, maxgeom=100)

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.mujoco_name)

        self.vopt = mujoco.MjvOption()
        self.vopt.geomgroup[1] = 0 # Group 1 is the mocap markers for visualization # TODO no
        self.pert = mujoco.MjvPerturb()

        self.ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)

        self.viewport = mujoco.MjrRect(0, 0, *self.cam_res)

    def callback(self, m, d):
        if self.last_render_t is None or d.time - self.last_render_t > 1.0 / self.fps:
            # Render the simulated camera
            mujoco.mjv_updateScene(m, d, self.vopt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
            mujoco.mjr_render(self.viewport, self.scn, self.ctx)
            frame = np.empty((self.cam_res[1], self.cam_res[0], 3), dtype=np.uint8) # TODO shm?
            mujoco.mjr_readPixels(frame, None, self.viewport, self.ctx)
            frame = cv2.flip(frame, 0) # OpenGL renders with inverted y axis
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            self.pub_sub.pub((d.time, frame), use_shm=[False, False, True])
            self.last_render_t = d.time

    def exit(self):
        pass
