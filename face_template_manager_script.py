"""
This file is the template of the scripting node source code in edge mode
Substitution is made in HandFaceTracker.py

In the following:
rrn_ : normalized [0:1] coordinates in rotated rectangle coordinate systems 
sqn_ : normalized [0:1] coordinates in squared input image
"""
import marshal
from math import sin, cos, atan2, pi, degrees, floor, exp
from time import sleep, time
from collections import deque


pad_h = ${_pad_h}
img_h = ${_img_h}
img_w = ${_img_w}
frame_size = ${_frame_size}
crop_w = ${_crop_w}
with_attention = ${_with_attention}
lm_score_tresh = ${_lm_score_thresh}
double_face = ${_double_face}
track_hands = ${_track_hands}

${_TRACE1} ("Starting Face manager script node")

# BufferMgr is used to statically allocate buffers once 
# (replace dynamic allocation). 
# These buffers are used for sending result to host
class BufferMgr:
    def __init__(self):
        self._bufs = {}
    def __call__(self, size):
        try:
            buf = self._bufs[size]
        except KeyError:
            buf = self._bufs[size] = Buffer(size)
            ${_TRACE2} (f"New buffer allocated: {size}")
        return buf

buffer_mgr = BufferMgr()

def send_result(result):
    result_serial = marshal.dumps(result)
    buffer = buffer_mgr(len(result_serial))  
    buffer.getData()[:] = result_serial  
    node.io['host'].send(buffer)
    ${_TRACE2} ("Face manager sent result to host")


# status: 
# - 0: the face detection has run and detected no face 
# - 1: there is a face landmark inference running (result will be sent to the host directly from the landmark NN),  
#      rect_center_x, rect_center_y, rect_size, rotation contains valid information
def send_result_no_face():
    result = dict([("status", 0)])
    send_result(result)

def send_result_face(rect_center_x=0, rect_center_y=0, rect_size=0, rotation=0):
    result = dict([("status", 1), ("rotation", rotation),
            ("rect_center_x", rect_center_x), ("rect_center_y", rect_center_y), ("rect_size", rect_size)])
    send_result(result)

def normalize_radians(angle):
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))

# send_new_frame_to_branch defines on which branch new incoming frames are sent
# 1 = face detection branch 
# 2 = face landmark branch
send_new_frame_to_branch = 1

cfg_pre_fd = ImageManipConfig()
cfg_pre_fd.setResizeThumbnail(128, 128, 0, 0, 0)

id_wrist = 0
id_index_mcp = 5
id_middle_mcp = 9
id_ring_mcp =13
ids_for_bounding_box = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]

lm_input_size = 192
inv_lm_input_size = 1 / lm_input_size
if double_face:
    double_face_sent = False # True when we have started an inference on the 2nd face landmark instance
    wait_duration = deque(maxlen=10)
    wait_duration.append(0.01)
    wait_duration2 = deque(maxlen=10)
    wait_duration2.append(0.03)
    sleep_duration = 0

    def mean(l):
        return sum(l) / len(l)

while True:
    # Read frame from cam
    cam_frame = node.io['cam_in'].get()
    ${_TRACE2} ("Face manager received input frame")

    # Send it to host and hand_manager
    if track_hands:
        node.io['hand_manager'].send(cam_frame)
        ${_TRACE2} ("Face manager sent input frame to hand manager")

    ${_IF_SEND_RGB_TO_HOST}
    node.io['cam_out'].send(cam_frame)
    ${_TRACE2} ("Face manager  sent input frame to host")
    ${_IF_SEND_RGB_TO_HOST}

    if send_new_frame_to_branch == 1: # Routing frame to fd branch
        node.io['pre_fd_manip_frame'].send(cam_frame)
        # import time; time.sleep(5)
        # node.io['pre_fd_manip_cfg'].send(cfg_pre_fd)
        ${_TRACE2} ("Face manager sent thumbnail config to pre_fd manip")
        # Wait for fd post processing's result 
        inference = node.io['from_post_fd_nn'].get()
        detection = inference.getLayerFp16("pp_result")
        valid_outputs = inference.getLayerInt32("pp_result@shape")[0]
        ${_TRACE2} (f"Face manager received fd results (valid dets={valid_outputs}) : "+str(detection))
        # detection is list of 17 floats
        # score, cx, cy, w, h, 6x kps (values are normalized)
        # Two firsts kps are left eye and right eye (we don't care about others kps)
        fd_score, box_cx, box_cy, box_w, box_h, k0x, k0y, k1x, k1y = detection[:9]
        box_size = max(box_w, box_h)
        
        if valid_outputs == 0:
            send_result_no_face()
            send_new_frame_to_branch = 1
            ${_TRACE1} (f"Face detection - no face detected")
            continue
        ${_TRACE1} (f"Face detection - face detected")

        k01x = k1x - k0x
        k01y = k1y - k0y
        sqn_rr_size = 1.5 * box_size
        rotation = - atan2(-k01y, k01x)
        rotation = normalize_radians(rotation)
        
        sqn_rr_center_x = box_cx
        sqn_rr_center_y = box_cy


    node.io['pre_lm_manip_frame'].send(cam_frame)

    # Tell pre_lm_manip how to crop face region 
    rr = RotatedRect()
    rr.center.x    = sqn_rr_center_x
    rr.center.y    = (sqn_rr_center_y * frame_size - pad_h) / img_h
    rr.size.width  = sqn_rr_size
    rr.size.height = sqn_rr_size * frame_size / img_h
    rr.angle       = degrees(rotation)
    
    cos_rot = cos(rotation)
    sin_rot = sin(rotation)

    cfg = ImageManipConfig()
    cfg.setCropRotatedRect(rr, True)
    cfg.setResize(lm_input_size, lm_input_size)
    node.io['pre_lm_manip_cfg'].send(cfg)
    ${_TRACE2} ("Face manager sent config to pre_lm manip")

    sqn_rr_data = NNData(12)
    sqn_rr_data.setLayer('sqn_rr', [sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size])
    node.io['sqn_rr'].send(sqn_rr_data)
    rot_data = NNData(16)
    rot_data.setLayer('rot', [cos_rot, sin_rot, -sin_rot, cos_rot])
    if double_face:
        if sleep_duration < 0:
            ${_TRACE2} (f"Sleep {-sleep_duration} - mean1: {mean(wait_duration)} - mean2: {mean(wait_duration2)}")

            sleep(-sleep_duration)

    node.io['rot'].send(rot_data)
    ${_TRACE2} ("Face manager sent params to lm NN")

    # Let the host know there is a face landmark inference running (so that the host knows it will need to wait the result directly sent by the face landmark NN)
    send_result_face(sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, rotation)



    if double_face:
        # Before the result for instance 1 of face landmark comes back,
        # We have time to receive the result of previous 2nd request
        # and send a new second request on the 2nd instance of face landmark NN
        # with a new frame (but with the same parameters)
        # Waitng time for old model ~ 0.035s
        # Waitng time for model with attention ~ 0.087s 
        if double_face_sent:
            start_timer = time()

            # Receive previous 2nd request
            lm_result = node.io['from_lm_nn2'].get()
            wait = time()-start_timer
            ${_TRACE2} (f"Just waited {wait} s")
            wait_duration.append(wait)
            ${_TRACE2} ("== Face manager received result from lm NN 2")


        sleep_duration = (mean(wait_duration2) -mean(wait_duration))/2
        if sleep_duration > 0:
            ${_TRACE2} (f"== Sleep {sleep_duration} - mean1: {mean(wait_duration)} - mean2: {mean(wait_duration2)}")
            sleep(sleep_duration)
        
        # Send a new 2n request on a new frame
        # Read frame from cam
        cam_frame = node.io['cam_in'].get()
        ${_TRACE2} ("== Face manager received input frame")

        # Send it to host and hand_manager
        if track_hands:
            node.io['hand_manager'].send(cam_frame)
        ${_IF_SEND_RGB_TO_HOST}
        node.io['cam_out'].send(cam_frame)
        ${_IF_SEND_RGB_TO_HOST}

        node.io['pre_lm_manip_frame2'].send(cam_frame)
        node.io['pre_lm_manip_cfg2'].send(cfg)
        ${_TRACE2} ("== Face manager sent config to pre_lm manip 2")

        node.io['sqn_rr2'].send(sqn_rr_data)
        node.io['rot2'].send(rot_data)
        ${_TRACE2} ("== Face manager sent params to lm NN 2")
        # Let the host know there is a face landmark inference running (so that the host knows it will need to wait the result directly sent by the face landmark NN)
        send_result_face(sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, rotation)
        double_face_sent = True
        start_timer2 = time()


    # Wait for face landmark request
    lm_result = node.io['from_lm_nn'].get()
    
    ${_TRACE2} ("= Face manager received result from lm nn")
    if double_face:
        wait = time()-start_timer2
        ${_TRACE2} (f"== Just waited {wait} s")
        wait_duration2.append(wait)

    if with_attention:
        lm_score = lm_result.getLayerFp16("lm_conv_faceflag")[0] 
    else:
        lm_score = lm_result.getLayerFp16("lm_score")[0]
    lm_score = 1 / (1 + exp(-lm_score))
    ${_TRACE1} (f"= Landmark score: {lm_score}")

    if lm_score > lm_score_tresh:
        sqn_xy = lm_result.getLayerFp16("pp_sqn_xy")
        min_max = lm_result.getLayerFp16("pp_min_max")
        send_new_frame_to_branch = 2 

        # # Calculate the ROI for next frame
        x_min, y_min, x_max, y_max = min_max
        sqn_rr_size = 1.5 * max(x_max-x_min, y_max-y_min)
        sqn_rr_center_x = (x_min + x_max) / 2
        sqn_rr_center_y = (y_min + y_max) /2
        x0 = sqn_xy[66] # left eye = 33
        y0 = sqn_xy[67]
        x1 = sqn_xy[526] # right eye = 263
        y1 = sqn_xy[527]
        rotation = - atan2(-(y1-y0), x1-x0)
        rotation = normalize_radians(rotation)
        ${_TRACE1} (f"Landmarks - face confirmed")
    else:
        send_new_frame_to_branch = 1
        ${_TRACE1} (f"Landmarks - face not confirmed")
