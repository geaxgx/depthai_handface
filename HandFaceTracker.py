import numpy as np
from collections import namedtuple

from numpy.lib.arraysetops import isin
import mediapipe_utils as mpu
import depthai as dai
import cv2
from pathlib import Path
from FPS import FPS, now
import time
import sys
from string import Template
import marshal
from HostSpatialCalc import HostSpatialCalc
from face_geometry import ( 
                PCF,
                get_metric_landmarks,
                procrustes_landmark_basis,
                canonical_metric_landmarks
            )


SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_pp_top2_th50_sh4.blob")

HAND_LANDMARK_MODEL = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
HAND_TEMPLATE_MANAGER_SCRIPT_SOLO = str(SCRIPT_DIR / "hand_template_manager_script_solo.py")
HAND_TEMPLATE_MANAGER_SCRIPT_DUO = str(SCRIPT_DIR / "hand_template_manager_script_duo.py")

FACE_DETECTION_MODEL = str(SCRIPT_DIR / "models/face_detection_short_range_pp_top1_th50_sh1.blob")
FACE_LANDMARK_MODEL = str(SCRIPT_DIR / "models/face_landmark_pp_sh4.blob")
FACE_LANDMARK_WITH_ATTENTION_MODEL = str(SCRIPT_DIR / "models/face_landmark_with_attention_pp_sh4.blob")
FACE_TEMPLATE_MANAGER_SCRIPT = str(SCRIPT_DIR / "face_template_manager_script.py")



class DepthSync:
    """
    Store depth frames history (if 'xyz' is True) to assure that alignment of rgb frame and depth frame is 
    made on synchronized frames
    """
    def __init__(self):
        self.msg_lst = []
    def add(self, msg):
        if isinstance(msg, list):
            for m in msg:
                self.msg_lst.append(m)
        else:
            self.msg_lst.append(msg)
    def get(self, msg):
        """
        Return message from the list that have the closest timestamp to 'msg' timestamp
        and clean the list from old messages
        """
        if len(self.msg_lst) == 0:
            return None
        ts = msg.getTimestamp()
        delta_min = abs(ts - self.msg_lst[0].getTimestamp())
        msg_id = 0
        for i in range(1, len(self.msg_lst)):
            delta =  abs(ts - self.msg_lst[i].getTimestamp())
            if delta < delta_min:
                delta_min = delta
                msg_id = i
            else:
                break
        msg = self.msg_lst[msg_id]
        del self.msg_lst[:msg_id+1]
        return msg        


class HandFaceTracker:
    """
    Mediapipe Hand and Face Tracker for depthai (= Mediapipe Hand tracker + Mediapipe Facemesh)
    Arguments:
    - input_src: frame source, 
            - "rgb" or None: OAK* internal color camera,
            - a file path of an image or a video,
            - an integer (eg 0) for a webcam id,
    - nb_hands: 0, 1 or 2. Number of hands max tracked. If 0, then hand tracking is not used. 1 is faster than 2.
    - use_face_pose: boolean. If yes, compute the face pose transformation matrix and the metric landmarks.
            The face pose tranformation matrix provides mapping from the static canonical face model to the runtime face.
            The metric landmarks are the 3D runtime metric landmarks aligned with the canonical metric face landmarks (unit: cm).
    - xyz : boolean, when True calculate the (x, y, z) coords of face (measure on the forehead) and hands.
    - crop : boolean which indicates if square cropping on source images is applied or not
    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - resolution : sensor resolution "full" (1920x1080) or "ultra" (3840x2160),
    - internal_frame_height : when using the internal color camera, set the frame height (calling setIspScale()).
            The width is calculated accordingly to height and depends on value of 'crop'
    - use_gesture : boolean, when True, recognize hand poses froma predefined set of poses
                    (ONE, TWO, THREE, FOUR, FIVE, OK, PEACE, FIST)
    - single_hand_tolerance_thresh (when nb_hands=2 only) : if there is only one hand in a frame, 
            in order to know when a second hand will appear you need to run the palm detection 
            in the following frames. Because palm detection is slow, you may want to delay 
            the next time you will run it. 'single_hand_tolerance_thresh' is the number of 
            frames during only one hand is detected before palm detection is run again.  
    - focus: None or int between 0 and 255. Color camera focus. 
            If None, auto-focus is active. Otherwise, the focus is set to 'focus' 
    - trace : int, 0 = no trace, otherwise print some debug messages or show output of ImageManip nodes
            if trace & 1, print application level info like number of palm detections,
            if trace & 2, print lower level info like when a message is sent or received by the manager script node,
            if trace & 4, show in cv2 windows outputs of ImageManip node,
            if trace & 8, save in file tmp_code.py the python code of the manager script node
            Ex: if trace==3, both application and low level info are displayed.
                      
    """
    def __init__(self, input_src=None,
                with_attention=True,
                double_face=False,
                nb_hands=2,
                use_face_pose=False,
                xyz=False,
                crop=False,
                internal_fps=None,
                resolution="full",
                internal_frame_height=640,
                use_gesture=False,
                hlm_score_thresh=0.8,
                single_hand_tolerance_thresh=3,
                focus=None,
                trace=0
                ):

        self.pd_model = PALM_DETECTION_MODEL
        print(f"Palm detection blob     : {self.pd_model}")

        self.hlm_model = HAND_LANDMARK_MODEL
        self.hlm_score_thresh = hlm_score_thresh
        print(f"Landmark blob           : {self.hlm_model}")

        self.fd_model = FACE_DETECTION_MODEL
        print(f"Face detection blob     : {self.fd_model}")

        self.with_attention = with_attention
        if self.with_attention:
            self.flm_model = FACE_LANDMARK_WITH_ATTENTION_MODEL
        else:
            self.flm_model = FACE_LANDMARK_MODEL
        self.flm_score_thresh = 0.5
        print(f"Face landmark blob      : {self.flm_model}")

        self.nb_hands = nb_hands
        
        self.xyz = False
        self.crop = crop 
        self.use_world_landmarks = True
        if focus is None:
            self.focus = None
        else:
            self.focus = max(min(255, int(focus)), 0)
           
        self.trace = trace
        self.use_gesture = use_gesture
        self.single_hand_tolerance_thresh = single_hand_tolerance_thresh
        self.double_face = double_face
        if self.double_face:
            print("This is an experimental feature that should help to improve the FPS")
            if self.nb_hands > 0:
                print("With double_face flag, the hand tracking is disabled !")
                self.nb_hands = 0

        self.device = dai.Device()

        if input_src == None or input_src == "rgb":
            self.input_type = "rgb" # OAK* internal color camera
            if resolution == "full":
                self.resolution = (1920, 1080)
            elif resolution == "ultra":
                self.resolution = (3840, 2160)
            else:
                print(f"Error: {resolution} is not a valid resolution !")
                sys.exit()
            print("Sensor resolution:", self.resolution)

            if xyz:
                # Check if the device supports stereo
                cameras = self.device.getConnectedCameras()
                if dai.CameraBoardSocket.LEFT in cameras and dai.CameraBoardSocket.RIGHT in cameras:
                    self.xyz = True
                else:
                    print("Warning: depth unavailable on this device, 'xyz' argument is ignored")

            if internal_fps is None:
                if self.double_face:
                    if self.with_attention:
                        self.internal_fps = 14
                    else:
                        self.internal_fps = 41
                else:
                    if self.with_attention:
                        self.internal_fps = 11
                    else:
                        if self.nb_hands == 0:
                            self.internal_fps = 27
                        elif self.nb_hands == 1:
                            self.internal_fps = 24
                        else: # nb_hands = 2
                            self.internal_fps = 19


            else:
                self.internal_fps = internal_fps 
            
            
                if self.input_type == "rgb" and internal_fps is None:
                    if self.with_attention:
                        self.internal_fps = 14
                    else:
                        self.internal_fps = 41
                

            print(f"Internal camera FPS set to: {self.internal_fps}") 

            self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps

            if self.crop:
                self.frame_size, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height, self.resolution)
                self.img_h = self.img_w = self.frame_size
                self.pad_w = self.pad_h = 0
                self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2
            else:
                width, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height * self.resolution[0] / self.resolution[1], self.resolution, is_height=False)
                self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
                self.pad_h = (self.img_w - self.img_h) // 2
                self.pad_w = 0
                self.frame_size = self.img_w
                self.crop_w = 0
        
            print(f"Internal camera image size: {self.img_w} x {self.img_h} - pad_h: {self.pad_h}")

        elif input_src.endswith('.jpg') or input_src.endswith('.png') :
            self.input_type= "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            self.img_h, self.img_w = self.img.shape[:2]
        else:
            self.input_type = "video"
            if input_src.isdigit():
                input_src = int(input_src)
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("Video FPS:", self.video_fps)
        
        if self.input_type != "rgb":
            self.xyz = False
            print(f"Original frame size: {self.img_w}x{self.img_h}")
            if self.crop:
                self.frame_size = min(self.img_w, self.img_h)
            else:
                self.frame_size = max(self.img_w, self.img_h)
            self.crop_w = max((self.img_w - self.frame_size) // 2, 0)
            if self.crop_w: print("Cropping on width :", self.crop_w)
            self.crop_h = max((self.img_h - self.frame_size) // 2, 0)
            if self.crop_h: print("Cropping on height :", self.crop_h)

            self.pad_w = max((self.frame_size - self.img_w) // 2, 0)
            if self.pad_w: print("Padding on width :", self.pad_w)
            self.pad_h = max((self.frame_size - self.img_h) // 2, 0)
            if self.pad_h: print("Padding on height :", self.pad_h)
                     
            if self.crop: self.img_h = self.img_w = self.frame_size
            print(f"Frame working size: {self.img_w}x{self.img_h}")

        # Define and start pipeline
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        # Define data queues 
        if self.input_type == "rgb":
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=2, blocking=True)
        else:
            self.q_face_manager_in = self.device.getInputQueue(name="face_manager_in")
        if self.nb_hands > 0:
            self.q_hand_manager_out = self.device.getOutputQueue(name="hand_manager_out", maxSize=2, blocking=True)
        self.q_face_manager_out = self.device.getOutputQueue(name="face_manager_out", maxSize=2, blocking=True)
        self.q_flm_nn_out = self.device.getOutputQueue(name="flm_nn_out", maxSize=2, blocking=True)
        # For showing outputs of ImageManip nodes (debugging)
        if self.trace & 4:
            if self.nb_hands > 0:
                self.q_pre_pd_manip_out = self.device.getOutputQueue(name="pre_pd_manip_out", maxSize=1, blocking=False)
                self.q_pre_hlm_manip_out = self.device.getOutputQueue(name="pre_hlm_manip_out", maxSize=1, blocking=False)    
            self.q_pre_fd_manip_out = self.device.getOutputQueue(name="pre_fd_manip_out", maxSize=1, blocking=False)
            self.q_pre_flm_manip_out = self.device.getOutputQueue(name="pre_flm_manip_out", maxSize=1, blocking=False)    
        if self.xyz:
            self.q_depth_out = self.device.getOutputQueue(name="depth_out", maxSize=5, blocking=True)
            self.depth_sync = DepthSync()
            self.spatial_calc = HostSpatialCalc(self.device, delta=int(self.img_w/100), thresh_high=3000)
       
        self.fps = FPS()
        self.seq_num = 0

        self.use_face_pose = use_face_pose
        if self.use_face_pose:
            calib_data = self.device.readCalibration()
            self.rgb_matrix= np.array(calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RGB, resizeWidth=self.img_w, resizeHeight=self.img_h))
            self.rgb_dist_coef = np.array(calib_data.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
            self.pcf = PCF(
                near=1,
                far=10000,
                frame_height=self.img_h,
                frame_width=self.img_w,
                fy=self.rgb_matrix[1][1],
            )

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        # pipeline.setXLinkChunkSize(0)  # << important to increase throughtput!!! ?
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)
        self.pd_input_length = 128

        if self.input_type == "rgb":
            # ColorCamera
            print("Creating Color Camera")
            # _pgraph_ name 
            cam = pipeline.createColorCamera()
            if self.resolution[0] == 1920:
                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            else:
                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam.setInterleaved(False)
            cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
            cam.setFps(self.internal_fps)
            if self.focus is not None:
                cam.initialControl.setManualFocus(self.focus)

            if self.crop:
                cam.setVideoSize(self.frame_size, self.frame_size)
                cam.setPreviewSize(self.frame_size, self.frame_size)
            else: 
                cam.setVideoSize(self.img_w, self.img_h)
                cam.setPreviewSize(self.img_w, self.img_h)

        face_manager_script = pipeline.create(dai.node.Script)
        face_manager_script.setScript(self.build_face_manager_script())
        face_manager_script.setProcessor(dai.ProcessorType.LEON_CSS)
        if self.input_type == "rgb":
            cam.preview.link(face_manager_script.inputs["cam_in"])
            face_manager_script.inputs["cam_in"].setQueueSize(1)
            face_manager_script.inputs["cam_in"].setBlocking(False)
        else:
            host_to_face_manager_in = pipeline.createXLinkIn()
            host_to_face_manager_in.setStreamName("face_manager_in")
            host_to_face_manager_in.out.link(face_manager_script.inputs["cam_in"])
            

        if self.input_type == "rgb":
            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")
            # cam_out.input.setQueueSize(1)
            # cam_out.input.setBlocking(False)
            face_manager_script.outputs["cam_out"].link(cam_out.input)

        if self.nb_hands > 0:
            # Define hand manager script node
            hand_manager_script = pipeline.create(dai.node.Script)
            hand_manager_script.setScript(self.build_hand_manager_script())
            hand_manager_script.setProcessor(dai.ProcessorType.LEON_CSS)
            face_manager_script.outputs["hand_manager"].link(hand_manager_script.inputs["face_manager"])

            # Define palm detection pre processing: resize preview to (self.pd_input_length, self.pd_input_length)
            print("Creating Palm Detection pre processing image manip")
            pre_pd_manip = pipeline.create(dai.node.ImageManip)
            pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
            pre_pd_manip.initialConfig.setResizeThumbnail(self.pd_input_length, self.pd_input_length, 0, 0, 0)

            # pre_pd_manip.setWaitForConfigInput(True)
            pre_pd_manip.inputImage.setQueueSize(1)
            hand_manager_script.outputs['pre_pd_manip_frame'].link(pre_pd_manip.inputImage)
            # hand_manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig)

            # For debugging
            if self.trace & 4:
                pre_pd_manip_out = pipeline.createXLinkOut()
                pre_pd_manip_out.setStreamName("pre_pd_manip_out")
                pre_pd_manip.out.link(pre_pd_manip_out.input)

            # Define palm detection model
            print("Creating Palm Detection Neural Network...")
            pd_nn = pipeline.create(dai.node.NeuralNetwork)
            pd_nn.setBlobPath(self.pd_model)
            pre_pd_manip.out.link(pd_nn.input)
            pd_nn.out.link(hand_manager_script.inputs['from_post_pd_nn'])

            # Define link to send result to host 
            hand_manager_out = pipeline.create(dai.node.XLinkOut)
            hand_manager_out.setStreamName("hand_manager_out")
            hand_manager_script.outputs['host'].link(hand_manager_out.input)
            hand_manager_script.setProcessor(dai.ProcessorType.LEON_CSS)


            # Define hand landmark pre processing image manip
            print("Creating Hand Landmark pre processing image manip") 
            self.hlm_input_length = 224
            pre_hlm_manip = pipeline.create(dai.node.ImageManip)
            pre_hlm_manip.setMaxOutputFrameSize(self.hlm_input_length*self.hlm_input_length*3)
            pre_hlm_manip.setWaitForConfigInput(True)
            pre_hlm_manip.inputImage.setQueueSize(2)

            # For debugging
            if self.trace & 4:
                pre_hlm_manip_out = pipeline.createXLinkOut()
                pre_hlm_manip_out.setStreamName("pre_hlm_manip_out")
                pre_hlm_manip.out.link(pre_hlm_manip_out.input)

            hand_manager_script.outputs['pre_lm_manip_frame'].link(pre_hlm_manip.inputImage)
            hand_manager_script.outputs['pre_lm_manip_cfg'].link(pre_hlm_manip.inputConfig)

            # Define hand landmark model
            print(f"Creating Hand Landmark Neural Network")          
            hlm_nn = pipeline.create(dai.node.NeuralNetwork)
            hlm_nn.setBlobPath(self.hlm_model)
            pre_hlm_manip.out.link(hlm_nn.input)
            hlm_nn.out.link(hand_manager_script.inputs['from_lm_nn'])

        ### Face
        self.fd_input_length = 128

        print("Creating Face Detection pre processing image manip")
        pre_fd_manip = pipeline.create(dai.node.ImageManip)
        pre_fd_manip.setMaxOutputFrameSize(self.fd_input_length*self.fd_input_length*3)
        # pre_fd_manip.setWaitForConfigInput(True)
        pre_fd_manip.initialConfig.setResizeThumbnail(self.fd_input_length, self.fd_input_length, 0, 0, 0)
        pre_fd_manip.inputImage.setQueueSize(1)
        face_manager_script.outputs['pre_fd_manip_frame'].link(pre_fd_manip.inputImage)
        # face_manager_script.outputs['pre_fd_manip_cfg'].link(pre_fd_manip.inputConfig)

        # For debugging
        if self.trace & 4:
            pre_fd_manip_out = pipeline.createXLinkOut()
            pre_fd_manip_out.setStreamName("pre_fd_manip_out")
            pre_fd_manip.out.link(pre_fd_manip_out.input)

        # Define face detection model
        print("Creating Face Detection Neural Network")
        fd_nn = pipeline.create(dai.node.NeuralNetwork)
        fd_nn.setBlobPath(self.fd_model)
        pre_fd_manip.out.link(fd_nn.input)
        fd_nn.out.link(face_manager_script.inputs['from_post_fd_nn'])

        # Define link to send result to host 
        face_manager_out = pipeline.create(dai.node.XLinkOut)
        face_manager_out.setStreamName("face_manager_out")
        face_manager_script.outputs['host'].link(face_manager_out.input)

        # Define face landmark pre processing image manip
        print("Creating Face Landmark pre processing image manip") 
        self.flm_input_length = 192
        pre_flm_manip = pipeline.create(dai.node.ImageManip)
        pre_flm_manip.setMaxOutputFrameSize(self.flm_input_length*self.flm_input_length*3)
        pre_flm_manip.setWaitForConfigInput(True)
        pre_flm_manip.inputImage.setQueueSize(2)

        # For debugging
        if self.trace & 4:
            pre_flm_manip_out = pipeline.createXLinkOut()
            pre_flm_manip_out.setStreamName("pre_flm_manip_out")
            pre_flm_manip.out.link(pre_flm_manip_out.input)

        face_manager_script.outputs['pre_lm_manip_frame'].link(pre_flm_manip.inputImage)
        face_manager_script.outputs['pre_lm_manip_cfg'].link(pre_flm_manip.inputConfig)

        # Define face landmark model
        print(f"Creating Face Landmark Neural Network")          
        flm_nn = pipeline.create(dai.node.NeuralNetwork)
        flm_nn.setBlobPath(self.flm_model)
        pre_flm_manip.out.link(flm_nn.inputs["lm_input_1"])
        face_manager_script.outputs['sqn_rr'].link(flm_nn.inputs['pp_sqn_rr'])
        face_manager_script.outputs['rot'].link(flm_nn.inputs['pp_rot'])
        flm_nn.out.link(face_manager_script.inputs['from_lm_nn'])

        flm_nn_out = pipeline.create(dai.node.XLinkOut)
        flm_nn_out.setStreamName("flm_nn_out")
        flm_nn.out.link(flm_nn_out.input)

        if self.double_face:
            print("Creating Face Landmark pre processing image manip 2") 
            pre_flm_manip2 = pipeline.create(dai.node.ImageManip)
            pre_flm_manip2.setMaxOutputFrameSize(self.flm_input_length*self.flm_input_length*3)
            pre_flm_manip2.setWaitForConfigInput(True)
            pre_flm_manip2.inputImage.setQueueSize(2)

            face_manager_script.outputs['pre_lm_manip_frame2'].link(pre_flm_manip2.inputImage)
            face_manager_script.outputs['pre_lm_manip_cfg2'].link(pre_flm_manip2.inputConfig)
            
            print(f"Creating Face Landmark Neural Network 2")          
            flm_nn2 = pipeline.create(dai.node.NeuralNetwork)
            flm_nn2.setBlobPath(self.flm_model)
            pre_flm_manip2.out.link(flm_nn2.inputs["lm_input_1"])
            face_manager_script.outputs['sqn_rr2'].link(flm_nn2.inputs['pp_sqn_rr'])
            face_manager_script.outputs['rot2'].link(flm_nn2.inputs['pp_rot'])
            flm_nn2.out.link(face_manager_script.inputs['from_lm_nn2'])

            flm_nn2.out.link(flm_nn_out.input)


        if self.xyz:
            print("Creating MonoCameras, Stereo and SpatialLocationCalculator nodes")
            # For now, RGB needs fixed focus to properly align with depth.
            # The value used during calibration should be used here
            calib_data = self.device.readCalibration()
            calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.RGB)
            print(f"RGB calibration lens position: {calib_lens_pos}")
            cam.initialControl.setManualFocus(calib_lens_pos)

            mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
            left = pipeline.createMonoCamera()
            left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            left.setResolution(mono_resolution)
            left.setFps(self.internal_fps)

            right = pipeline.createMonoCamera()
            right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            right.setResolution(mono_resolution)
            right.setFps(self.internal_fps)

            stereo = pipeline.createStereoDepth()
            stereo.setConfidenceThreshold(150)
            # LR-check is required for depth alignment
            stereo.setLeftRightCheck(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            stereo.setSubpixel(False)  # subpixel True brings latency
            # MEDIAN_OFF necessary in depthai 2.7.2. 
            # Otherwise : [critical] Fatal error. Please report to developers. Log: 'StereoSipp' '533'
            # stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)

            left.out.link(stereo.left)
            right.out.link(stereo.right)    

            depth_out = pipeline.create(dai.node.XLinkOut)
            depth_out.setStreamName("depth_out")
            stereo.depth.link(depth_out.input)

        print("Pipeline created.")
        return pipeline        
    
    def build_hand_manager_script(self):
        '''
        The code of the scripting node 'manager_script' depends on :
            - the score threshold,
            - the video frame shape
        So we build this code from the content of the file template_manager_script_*.py which is a python template
        '''
        # Read the template
        with open(HAND_TEMPLATE_MANAGER_SCRIPT_SOLO if self.nb_hands == 1 else HAND_TEMPLATE_MANAGER_SCRIPT_DUO, 'r') as file:
            template = Template(file.read())
        
        # Perform the substitution
        code = template.substitute(
                    _TRACE1 = "node.warn" if self.trace & 1 else "#",
                    _TRACE2 = "node.warn" if self.trace & 2 else "#",
                    _lm_score_thresh = self.hlm_score_thresh,
                    _pad_h = self.pad_h,
                    _img_h = self.img_h,
                    _img_w = self.img_w,
                    _frame_size = self.frame_size,
                    _crop_w = self.crop_w,
                    _IF_USE_HANDEDNESS_AVERAGE = "",
                    _single_hand_tolerance_thresh= self.single_hand_tolerance_thresh,
                    _IF_USE_WORLD_LANDMARKS = "" if self.use_world_landmarks else '"""',
        )
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub('\n\s*\n', '\n', code)
        # For debugging
        if self.trace & 8:
            with open("hand_tmp_code.py", "w") as file:
                file.write(code)

        return code

    def build_face_manager_script(self):
        '''
        The code of the scripting node 'manager_script' depends on :
            - the score threshold,
            - the video frame shape
        So we build this code from the content of the file template_manager_script_*.py which is a python template
        '''
        # Read the template
        with open(FACE_TEMPLATE_MANAGER_SCRIPT, 'r') as file:
            template = Template(file.read())
        
        # Perform the substitution
        code = template.substitute(
                    _TRACE1 = "node.warn" if self.trace & 1 else "#",
                    _TRACE2 = "node.warn" if self.trace & 2 else "#",
                    _with_attention = self.with_attention,
                    _lm_score_thresh = self.flm_score_thresh,
                    _pad_h = self.pad_h,
                    _img_h = self.img_h,
                    _img_w = self.img_w,
                    _frame_size = self.frame_size,
                    _crop_w = self.crop_w,
                    _IF_SEND_RGB_TO_HOST = "" if self.input_type == "rgb" else '"""',
                    _track_hands = self.nb_hands > 0,
                    _double_face = self.double_face
        )
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub('\n\s*\n', '\n', code)
        # For debugging
        if self.trace & 8:
            with open("face_tmp_code.py", "w") as file:
                file.write(code)

        return code

    def extract_hand_data(self, res, hand_idx):
        hand = mpu.HandRegion()
        hand.rect_x_center_a = res["rect_center_x"][hand_idx] * self.frame_size
        hand.rect_y_center_a = res["rect_center_y"][hand_idx] * self.frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][hand_idx] * self.frame_size
        hand.rotation = res["rotation"][hand_idx] 
        hand.rect_points = mpu.rotated_rect_to_points(hand.rect_x_center_a, hand.rect_y_center_a, hand.rect_w_a, hand.rect_h_a, hand.rotation)
        hand.lm_score = res["lm_score"][hand_idx]
        hand.handedness = res["handedness"][hand_idx]
        hand.label = "right" if hand.handedness > 0.5 else "left"
        hand.norm_landmarks = np.array(res['rrn_lms'][hand_idx]).reshape(-1,3)
        hand.landmarks = (np.array(res["sqn_lms"][hand_idx]) * self.frame_size).reshape(-1,2).astype(np.int)

        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
        if self.pad_h > 0:
            hand.landmarks[:,1] -= self.pad_h
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][1] -= self.pad_h
        if self.pad_w > 0:
            hand.landmarks[:,0] -= self.pad_w
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][0] -= self.pad_w

        # World landmarks
        if self.use_world_landmarks:
            hand.world_landmarks = np.array(res["world_lms"][hand_idx]).reshape(-1, 3)

        if self.use_gesture: mpu.recognize_gesture(hand)

        return hand

    def extract_face_data(self, res_lm_script, res_lm_nn):
        if self.with_attention:
            lm_score = res_lm_nn.getLayerFp16("lm_conv_faceflag")[0] 
        else:
            lm_score = res_lm_nn.getLayerFp16("lm_score")[0]
        if lm_score < self.flm_score_thresh: return None
        face = mpu.Face()
        face.lm_score = lm_score
        face.rect_x_center_a = res_lm_script["rect_center_x"] * self.frame_size
        face.rect_y_center_a = res_lm_script["rect_center_y"] * self.frame_size
        face.rect_w_a = face.rect_h_a = res_lm_script["rect_size"] * self.frame_size
        face.rotation = res_lm_script["rotation"]
        face.rect_points = mpu.rotated_rect_to_points(face.rect_x_center_a, face.rect_y_center_a, face.rect_w_a, face.rect_h_a, face.rotation)
        sqn_xy = res_lm_nn.getLayerFp16("pp_sqn_xy")
        sqn_z = res_lm_nn.getLayerFp16("pp_sqn_z")
        rrn_xy = res_lm_nn.getLayerFp16("pp_rrn_xy")
        rrn_z = res_lm_nn.getLayerFp16("pp_rrn_z")

        if self.with_attention:
            # rrn_xy and sqn_xy are the concatenation of 2d landmarks:
            # 468 basic landmarks
            # 80 lips landmarks
            # 71 left eye landmarks
            # 71 right eye landmarks
            # 5 left iris landmarks
            # 5 right iris landmarks
            #
            # rrn_z and sqn_z corresponds to 468 basic landmarks
            
            # face.landmarks = 3D landmarks in the original image in pixels
            lm_xy = (np.array(sqn_xy).reshape(-1,2) * self.frame_size).astype(np.int)
            lm_zone = {}
            lm_zone["lips"] = lm_xy[468:548]
            lm_zone["left eye"] = lm_xy[548:619]
            lm_zone["right eye"] = lm_xy[619:690]
            lm_zone["left iris"] = lm_xy[690:695]
            lm_zone["right iris"] = lm_xy[695:700]
            for zone in ["lips", "left eye", "right eye"]:
                idx_map = mpu.XY_REFINEMENT_IDX_MAP[zone]
                np.put_along_axis(lm_xy, idx_map, lm_zone[zone], axis=0)
            lm_xy[468:473] = lm_zone["left iris"]
            lm_xy[473:478] = lm_zone["right iris"]
            lm_xy = lm_xy[:478]
            lm_z = (np.array(sqn_z) * self.frame_size)
            left_iris_z = np.mean(lm_z[mpu.Z_REFINEMENT_IDX_MAP['left iris']])
            right_iris_z = np.mean(lm_z[mpu.Z_REFINEMENT_IDX_MAP['right iris']])
            lm_z = np.hstack((lm_z, np.repeat([left_iris_z], 5), np.repeat([right_iris_z], 5))).reshape(-1, 1)
            face.landmarks = np.hstack((lm_xy, lm_z)).astype(np.int)

            # face.norm_landmarks = 3D landmarks inside the rotated rectangle, values in [0..1]
            nlm_xy = np.array(rrn_xy).reshape(-1,2)
            nlm_zone = {}
            nlm_zone["lips"] = nlm_xy[468:548]
            nlm_zone["left eye"] = nlm_xy[548:619]
            nlm_zone["right eye"] = nlm_xy[619:690]
            nlm_zone["left iris"] = nlm_xy[690:695]
            nlm_zone["right iris"] = nlm_xy[695:700]
            for zone in ["lips", "left eye", "right eye"]:
                idx_map = mpu.XY_REFINEMENT_IDX_MAP[zone]
                np.put_along_axis(nlm_xy, idx_map, nlm_zone[zone], axis=0)
            nlm_xy[468:473] = nlm_zone["left iris"]
            nlm_xy[473:478] = nlm_zone["right iris"]
            nlm_xy = nlm_xy[:478]
            nlm_z = np.array(rrn_z)
            left_iris_z = np.mean(nlm_z[mpu.Z_REFINEMENT_IDX_MAP['left iris']])
            right_iris_z = np.mean(nlm_z[mpu.Z_REFINEMENT_IDX_MAP['right iris']])
            nlm_z = np.hstack((nlm_z, np.repeat([left_iris_z], 5), np.repeat([right_iris_z], 5))).reshape(-1, 1)
            face.norm_landmarks = np.hstack((nlm_xy, nlm_z))

        else:
            face.norm_landmarks = np.hstack((np.array(rrn_xy).reshape(-1,2), np.array(rrn_z).reshape(-1,1)))
            lm_xy = (np.array(sqn_xy) * self.frame_size).reshape(-1,2)
            lm_z = (np.array(sqn_z) * self.frame_size).reshape(-1, 1)
            face.landmarks = np.hstack((lm_xy, lm_z)).astype(np.int)

        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
        if self.pad_h > 0:
            face.landmarks[:,1] -= self.pad_h
            for i in range(len(face.rect_points)):
                face.rect_points[i][1] -= self.pad_h
        if self.pad_w > 0:
            face.landmarks[:,0] -= self.pad_w
            for i in range(len(face.rect_points)):
                face.rect_points[i][0] -= self.pad_w

        if self.use_face_pose:
            screen_landmarks = (face.landmarks / np.array([self.img_w, self.img_h, self.img_w])).T
            face.metric_landmarks, face.pose_transform_mat = get_metric_landmarks(screen_landmarks, self.pcf)
            # https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
            face.pose_transform_mat[1:3, :] = -face.pose_transform_mat[1:3, :]
            face.pose_rotation_vector, _ = cv2.Rodrigues(face.pose_transform_mat[:3, :3])
            face.pose_translation_vector = face.pose_transform_mat[:3, 3, None]
        return face

    def next_frame(self):
        if self.double_face and self.input_type != "rgb" and self.seq_num == 0:
            # Because there are 2 inferences running in parallel in double face mode, we need to send 2 frames on the first loop iteration
            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    return None, None, None
            # Cropping and/or padding of the video frame
            video_frame = frame[self.crop_h:self.crop_h+self.frame_size, self.crop_w:self.crop_w+self.frame_size]
            self.prev_video_frame = video_frame
           
            frame = dai.ImgFrame()
            frame.setType(dai.ImgFrame.Type.BGR888p)
            h,w = video_frame.shape[:2]
            frame.setWidth(w)
            frame.setHeight(h)
            frame.setData(video_frame.transpose(2,0,1).flatten())
            self.q_face_manager_in.send(frame)

        self.seq_num += 1
        self.fps.update()
        if self.input_type == "rgb":
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()  
        else:
            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    return None, None, None
            # Cropping and/or padding of the video frame
            video_frame = frame[self.crop_h:self.crop_h+self.frame_size, self.crop_w:self.crop_w+self.frame_size]
           
            frame = dai.ImgFrame()
            frame.setType(dai.ImgFrame.Type.BGR888p)
            h,w = video_frame.shape[:2]
            frame.setWidth(w)
            frame.setHeight(h)
            frame.setData(video_frame.transpose(2,0,1).flatten())
            self.q_face_manager_in.send(frame)

        # For debugging
        if self.trace & 4:
            if self.nb_hands > 0:
                pre_pd_manip = self.q_pre_pd_manip_out.tryGet()
                if pre_pd_manip:
                    pre_pd_manip = pre_pd_manip.getCvFrame()
                    cv2.imshow("pre_pd_manip", pre_pd_manip)
                pre_hlm_manip = self.q_pre_hlm_manip_out.tryGet()
                if pre_hlm_manip:
                    pre_hlm_manip = pre_hlm_manip.getCvFrame()
                    cv2.imshow("pre_hlm_manip", pre_hlm_manip)
            pre_fd_manip = self.q_pre_fd_manip_out.tryGet()
            if pre_fd_manip:
                pre_fd_manip = pre_fd_manip.getCvFrame()
                cv2.imshow("pre_fd_manip", pre_fd_manip)
            pre_flm_manip = self.q_pre_flm_manip_out.tryGet()
            if pre_flm_manip:
                pre_flm_manip = pre_flm_manip.getCvFrame()
                cv2.imshow("pre_flm_manip", pre_flm_manip)

        # Get result from device
        hands = []
        if self.nb_hands > 0:
            res = marshal.loads(self.q_hand_manager_out.get().getData())
            for i in range(len(res.get("lm_score",[]))):
                hand = self.extract_hand_data(res, i)
                hands.append(hand)

        res_lm_script = marshal.loads(self.q_face_manager_out.get().getData())
        status = res_lm_script["status"]
        faces = []
        # status = 0 means the face detector has run but detected no face
        # status = 1 means face_manager_script has initiated an face landmark inference,
        #            and the face landmark NN will send directly the result here, on the host
        if status == 1:
            res_lm_nn = self.q_flm_nn_out.get()
            face = self.extract_face_data(res_lm_script, res_lm_nn)
            if face is not None: faces.append(face)

        if self.xyz:
            t = now()
            in_depth_msgs = self.q_depth_out.getAll()
            self.depth_sync.add(in_depth_msgs)
            synced_depth_msg = self.depth_sync.get(in_video) 
            frame_depth = synced_depth_msg.getFrame()
            # !!! The 4 lines below are for disparity (not depth)
            # frame_depth = (frame_depth * 255. / self.max_disparity).astype(np.uint8)
            # frame_depth = cv2.applyColorMap(frame_depth, cv2.COLORMAP_HOT)
            # frame_depth = np.ascontiguousarray(frame_depth)
            # cv2.imshow("depth", frame_depth)
            if self.nb_hands > 0:
                for hand in hands:
                    hand.xyz, hand.xyz_zone = self.spatial_calc.get_xyz(frame_depth, hand.landmarks[0])
            for face in faces:
                face.xyz, face.xyz_zone = self.spatial_calc.get_xyz(frame_depth, face.landmarks[9,:2])

        if self.double_face and self.input_type != "rgb":
            video_frame, self.prev_video_frame = self.prev_video_frame, video_frame
        return video_frame, faces, hands


    def exit(self):
        self.device.close()
        print(f"FPS : {self.fps.get_global():.1f} f/s")
            