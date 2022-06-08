#!/usr/bin/env python3

import sys

sys.path.append("../..")

from HandFaceTracker import HandFaceTracker
from HandFaceRenderer import HandFaceRenderer
from face_mesh_connections import FACEMESH_RIGHT_EYE, FACEMESH_LEFT_EYE, FACEMESH_IRISES_CROSS
import argparse
import numpy as np
import cv2
from RollingGraph import RollingGraph
from FPS import now
from collections import deque
from math import exp
from time import time


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, 
        help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser.add_argument('-e', '--eye_opening_threshold', type=float, default=0.05,
            help="Vertical distance between upper and lower eyelids below which the eye is considered blinking")
parser.add_argument('-f', '--face_movement_threshold', type=float, default=0.05,
            help="Face movement distance between 2 consecutive frames above which a potential blinking is considered invalid")
args = parser.parse_args()

tracker = HandFaceTracker(
        input_src=args.input, 
        with_attention=True,
        nb_hands=0,
        resolution="ultra"
        )

renderer = HandFaceRenderer(tracker=tracker)

def distance(a, b):
    """
    a, b: 2 points (in 2D or 3D)
    """
    return np.linalg.norm(a-b)

def warp_rect_img(rect_points, img, w, h):
    src = np.array(rect_points[1:], dtype=np.float32) # rect_points[0] is left bottom point !
    dst = np.array([(0, 0), (h, 0), (h, w)], dtype=np.float32)
    mat = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, mat, (w, h))

def draw_line_set(frame, face, line_set, color=(255,255,255), thickness=1):
    """
    Draw face landmarks on normalized face image.
    The normalized face image is the "unrotated" image that is fed the face landmark model
    """
    h,w = frame.shape[:2]
    pl = [np.array([[face.norm_landmarks[i1,:2]*w, face.norm_landmarks[i2,:2]*h]]).astype(int) for i1,i2 in line_set]
    cv2.polylines(frame, pl, False, color, thickness, lineType=cv2.LINE_AA)

def left_weight(face):
    """
    Depending on the yaw angle of the face, the eyes are more or less visible from the camera.
    Potentially, an eye may be completely invisible. We want to give more weight to the most visible eye.
    We can estimate the yaw of the face by measuring the z coordinate of the 2 "ear" landmarks 
    (127 for left ear, 356 for right ear). Here we compute the weight for the left eye.
    The weight for the right eye is simply 2 - left_weight. Note that the sum of the 2 weights is 2,
    for aestetical reasons (avoid superposition of the curves on the rolling graph)
    """
    left_ear_z = face.norm_landmarks[127][2]
    right_ear_z = face.norm_landmarks[356][2]
    left_w = 2 * right_ear_z / (left_ear_z + right_ear_z)
    return left_w

face_output_size = 1000
left_color = (0,255,0)
right_color = (60,76,231)
eye_opening_graph = RollingGraph("Eye opening distance", height=300, y_min=0, y_max=0.12, threshold=args.eye_opening_threshold, colors=[left_color, right_color, (255,255,255)], thickness=[2,2,2], waitKey=False)

face_movement_graph = RollingGraph("Face movement", height=300, y_min=0, y_max=0.1, threshold=args.face_movement_threshold, colors=[(189,105,165)], thickness=[2], waitKey=False)

prev_ref_loc = None
blinking = False
face_movement_hist = []
blink_count = 0

while True:
    frame, faces, hands = tracker.next_frame()
    if frame is None: break
    if len(faces) > 0:
        face = faces[0]
        
        # Draw the normalize/"unrotated" face zone
        img_face = warp_rect_img(face.rect_points, frame, face_output_size, face_output_size)
        draw_line_set(img_face, face, FACEMESH_LEFT_EYE, left_color, 2)
        draw_line_set(img_face, face, FACEMESH_RIGHT_EYE, right_color, 2)
        draw_line_set(img_face, face, FACEMESH_IRISES_CROSS, (0,255,255), 1)
        # Determine the eyes zone in img_face
        # between landmarks 130 and 359
        zx_min = int(face.norm_landmarks[130][0] * face_output_size)
        zx_max = int(face.norm_landmarks[359][0] * face_output_size)
        zx_center = (zx_min + zx_max) // 2
        zw_2 = int((zx_max - zx_min) * 0.75)
        zy_center = int((face.norm_landmarks[130][1] + face.norm_landmarks[359][1]) * face_output_size / 2)
        zh_2 = zw_2 // 4
        zone_eyes = img_face[zy_center - zh_2:zy_center + zh_2, zx_center - zw_2:zx_center + zw_2,:]
        # Resize to a constant width window (600)
        zone_eyes = cv2.resize(zone_eyes, (600, int(zone_eyes.shape[0]*600/zone_eyes.shape[1])))
        cv2.imshow("Eyes", zone_eyes)

        # Vertical distance between eyelids.
        # Landmarks 159 for left upper eyelid 
        # Landmarks 145 for left lower eyelid 
        left_opening = distance(face.norm_landmarks[159], face.norm_landmarks[145])
        left_w = left_weight(face)
        # Landmarks 386 for right upper eyelid 
        # Landmarks 374 for right lower eyelid 
        right_opening = distance(face.norm_landmarks[386], face.norm_landmarks[374])
        right_w = 2 - left_w
        
        combined_opening = left_w*left_opening+right_w*right_opening
        eye_opening_graph.new_iter([left_opening, right_opening, combined_opening])

        # False positive blinks can happen when the face is moving
        # Is the face moving ? We look how a reference point (landmark 8) is moving in the image
        # We measure the shift of the reference between 2 consecutive frames in pixel
        # Then we normalize it by the size of the bounding rotated rectangle in pixel (face.rect_w_a) 
        ref_loc = face.landmarks[8]
        if prev_ref_loc is not None:
            ref_shift = distance(ref_loc, prev_ref_loc) / face.rect_w_a
            face_movement_graph.new_iter([ref_shift])
        prev_ref_loc = ref_loc

        closed = combined_opening < args.eye_opening_threshold
        if closed:
            if not blinking:
                # A protential blinking is beginning
                blink_start_timestamp = time()
                blinking = True
            # Store the face movement during the blinking
            face_movement_hist.append(ref_shift)
            
        elif blinking: # End of the blinking
            blink_duration = time() - blink_start_timestamp
            # Has the face moved too much during the blink ?
            for m in face_movement_hist:
                if m > args.face_movement_threshold:
                    # The blink is invalid
                    break
            else:
                # The blink is valid
                blink_count += 1
                # print(f"Blink # {blink_count} - duration: {blink_duration:.3f}")

            blinking = False
            face_movement_hist = []


        


    frame = renderer.draw(frame, faces, hands)
    # Draw the blink count in frame
    h,w = frame.shape[:2]
    frame[h-200:h,0:200,:] = 0
    cv2.putText(frame, "# blinks:", (40, h-160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    text_size = cv2.getTextSize(str(blink_count), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, thickness=3)[0]
    cv2.putText(frame, str(blink_count), (int(100-0.5*text_size[0]), h-40), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)

    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
    elif key == ord('z'):
        blink_count = 0
renderer.exit()
tracker.exit()
