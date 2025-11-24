import pyrealsense2 as rs
import numpy as np

class RealSense:
    def __init__(self,align_color=False,structured_light=0):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Left = IR1, Right = IR2
        self.config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)  # Left IR Camera
        self.config.enable_stream(rs.stream.infrared, 2, 848, 480, rs.format.y8, 30)  # Right IR Camera
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30) # depth
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30) # color
        pipeline_profile = self.pipeline.start(self.config)
        
        # To switch structured light on and off
        device = pipeline_profile.get_device()
        depth_sensor = device.query_sensors()[0]
        depth_sensor.set_option(rs.option.emitter_enabled, structured_light) # 0 - off, 1 - on 

        # Create an align object to align depth to color frame
        if align_color:
            self.align = rs.align(rs.stream.color)

        # Get intrinsic parameters
        profile = self.pipeline.get_active_profile()
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
    
    def get_ir_images(self):
        frames = self.pipeline.wait_for_frames()

        # Get infrared frames
        ir1_frame = frames.get_infrared_frame(1)
        ir2_frame = frames.get_infrared_frame(2)  

        if not ir1_frame or not ir2_frame:
            return None, None

        ir1_image = np.asanyarray(ir1_frame.get_data())
        ir2_image = np.asanyarray(ir2_frame.get_data())

        return ir1_image, ir2_image

    def get_aligned_rgbd(self):
        frames = self.pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            return None, None

        # Convert frames to numpy arrays
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return aligned_depth_image, color_image
    
    def get_rgb_intrinsics(self):
        # Get frames
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Get intrinsics
        intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                      [0, intrinsics.fy, intrinsics.ppy],
                      [0, 0, 1]])
        # Print intrinsics
        print("RGB Camera Intrinsics:")
        print(f"Width: {intrinsics.width}")
        print(f"Height: {intrinsics.height}")
        print(f"PPX: {intrinsics.ppx}")
        print(f"PPY: {intrinsics.ppy}")
        print(f"FX: {intrinsics.fx}")
        print(f"FY: {intrinsics.fy}")
        print(f"Distortion Model: {intrinsics.model}")
        print(f"Coefficients: {intrinsics.coeffs}")
        print(K)
        return K
    
    def get_ir1_intrinsics(self):
        # Get frames
        frames = self.pipeline.wait_for_frames()
        ir1_frame = frames.get_infrared_frame(1)

        # Get intrinsics
        intrinsics = ir1_frame.get_profile().as_video_stream_profile().get_intrinsics()
        K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                      [0, intrinsics.fy, intrinsics.ppy],
                      [0, 0, 1]])
        # Print intrinsics
        print("RGB Camera Intrinsics:")
        print(f"Width: {intrinsics.width}")
        print(f"Height: {intrinsics.height}")
        print(f"PPX: {intrinsics.ppx}")
        print(f"PPY: {intrinsics.ppy}")
        print(f"FX: {intrinsics.fx}")
        print(f"FY: {intrinsics.fy}")
        print(f"Distortion Model: {intrinsics.model}")
        print(f"Coefficients: {intrinsics.coeffs}")
        print(K)
        return K

    def release(self):
        self.pipeline.stop()
