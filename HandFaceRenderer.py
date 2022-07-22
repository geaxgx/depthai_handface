import cv2
import numpy as np
from mediapipe_utils import *
from face_mesh_connections import FACEMESH_TESSELATION, FACEMESH_LIPS, FACEMESH_EYES_EYEBROWS, \
            FACEMESH_IRISES, FACEMESH_FACE_OVAL
from shapely.geometry import Polygon, Point

import random, math
from FPS import now

LINES_HAND = [[0,1],[1,2],[2,3],[3,4], 
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]

# LINES_BODY to draw the body skeleton when Body Pre Focusing is used
LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
            [10,8],[8,6],[6,5],[5,7],[7,9],
            [6,12],[12,11],[11,5],
            [12,14],[14,16],[11,13],[13,15]]

color_list = [(255,230,0), (153,255,0), (0,255,218), (100,150,255), (255, 125,240)]
class HandFaceRenderer:
    def __init__(self, 
                tracker,
                output=None):

        self.tracker = tracker

        # Rendering flags
        self.show_hand_rot_rect = False
        self.show_hand_landmarks = True
        self.show_gesture = self.tracker.use_gesture
        self.hand_style = 0
        self.show_face_rot_rect = False
        self.show_face_landmarks = True
        self.face_style = 1
        self.window_title = "Hand & Face Tracking" if self.tracker.nb_hands > 0 else "Face Tracking"

        if self.tracker.with_attention:
            # Landmark indexes that are not concerned by refinement
            self.not_refined_lm_idx = set(range(468)) - set(REFINEMENT_IDX_MAP['lips']) - set(REFINEMENT_IDX_MAP['left eye']) - set(REFINEMENT_IDX_MAP['right eye'])
        else:
            self.not_refined_lm_idx = set(range(468))

        self.show_xyz_zone = self.show_xyz = self.tracker.xyz
        self.show_fps = True
        self.laconic = False # If True, display a black frame instead of the original frame
        self.show_metric_landmarks = self.show_pose = self.tracker.use_face_pose
        if self.tracker.use_face_pose:
            # Smoothing filter for metric_landmarks
            self.metric_filter = OneEuroFilter(min_cutoff=0.5, beta=2)
        self.smooth_metric = False
        self.metric_style = 0
        self.pose_style = 1
        self.magnify_factor = 1.
        self.radius = 1
        self.i = 0
        self.random_color = random.choice(color_list)


        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,self.tracker.video_fps,(self.tracker.img_w, self.tracker.img_h)) 

    def norm2abs(self, x_y):
        x = int(x_y[0] * self.tracker.frame_size - self.tracker.pad_w)
        y = int(x_y[1] * self.tracker.frame_size - self.tracker.pad_h)
        return (x, y)

    def draw_hand(self, hand):

        # (info_ref_x, info_ref_y): coords in the image of a reference point 
        # relatively to which hands information (score, handedness, xyz,...) are drawn
        info_ref_x = hand.landmarks[0,0]
        info_ref_y = np.max(hand.landmarks[:,1])

        # thick_coef is used to adapt the size of the draw landmarks features according to the size of the hand.
        thick_coef = hand.rect_w_a / 400
        if hand.lm_score > self.tracker.hlm_score_thresh:
            if self.show_hand_rot_rect:
                cv2.polylines(self.frame, [np.array(hand.rect_points)], True, (219, 152, 52), 2, cv2.LINE_AA)
            if self.show_hand_landmarks:
                lines = [np.array([hand.landmarks[point] for point in line]).astype(np.int) for line in LINES_HAND]
                if self.hand_style == 2:
                    color = (0,255,0) if hand.handedness > 0.5 else (0,0,255)
                else:
                    color = (219, 152, 52)
                cv2.polylines(self.frame, lines, False, color, int(1+thick_coef*3), cv2.LINE_AA)
                radius = int(1+thick_coef*5)
                
                if self.hand_style == 0:
                    color = (0,255,0) if hand.handedness > 0.5 else (0,0,255)
                else: 
                    color = (0,128,255)
                for x,y in hand.landmarks[:,:2]:
                    cv2.circle(self.frame, (int(x), int(y)), radius, color, -1)
            if self.tracker.use_gesture and self.show_gesture:
                    cv2.putText(self.frame, hand.gesture, (info_ref_x-20, info_ref_y-50), 
                            cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

                
        if self.show_xyz:
            x0, y0 = info_ref_x - 40, info_ref_y + 40
            cv2.rectangle(self.frame, (x0,y0), (x0+100, y0+85), (220,220,240), -1)
            cv2.putText(self.frame, f"X:{hand.xyz[0]/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_PLAIN, 1, (20,180,0), 2)
            cv2.putText(self.frame, f"Y:{hand.xyz[1]/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
            cv2.putText(self.frame, f"Z:{hand.xyz[2]/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
        if self.show_xyz_zone:
            # Show zone on which the spatial data were calculated
            cv2.rectangle(self.frame, tuple(hand.xyz_zone[0:2]), tuple(hand.xyz_zone[2:4]), (180,0,180), 2)

    def draw_line_set(self, frame, landmarks, line_set, color=(255,255,255), thickness=1):
        
        pl = [np.array([[landmarks[i1,:2], landmarks[i2,:2]]]) for i1,i2 in line_set]
        cv2.polylines(frame, pl, False, color, thickness)

    def draw_face(self, face):

        # (info_ref_x, info_ref_y): coords in the image of a reference point 
        # relatively to which faces information (score, faceedness, xyz,...) are drawn
        info_ref_x = face.landmarks[0,0]
        info_ref_y = np.max(face.landmarks[:,1])

        if face.lm_score > self.tracker.flm_score_thresh:
            if self.show_face_rot_rect:
                cv2.polylines(self.frame, [np.array(face.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
            radius = 2
            if self.show_face_landmarks:
                if self.face_style == 0:
                    for i in self.not_refined_lm_idx:
                        cv2.circle(self.frame, tuple(face.landmarks[i,:2]), radius, (255,255,255), -1)
                    if self.tracker.with_attention:
                        for i in REFINEMENT_IDX_MAP['lips']:
                            cv2.circle(self.frame, tuple(face.landmarks[i,:2]), radius, (0,128,255), -1)
                        for i in REFINEMENT_IDX_MAP['left eye']:
                            cv2.circle(self.frame, tuple(face.landmarks[i,:2]), radius, (0,255,0), -1)
                        for i in REFINEMENT_IDX_MAP['right eye']:
                            cv2.circle(self.frame, tuple(face.landmarks[i,:2]), radius, (0,255,0), -1)
                        for i in REFINEMENT_IDX_MAP['left iris']:
                            cv2.circle(self.frame, tuple(face.landmarks[i,:2]), radius, (0,0,255), -1)
                        for i in REFINEMENT_IDX_MAP['right iris']:
                            cv2.circle(self.frame, tuple(face.landmarks[i,:2]), radius, (0,0,255), -1)

                elif self.face_style in [1, 2]:
                    if self.face_style == 2:
                        self.draw_line_set(self.frame, face.landmarks, FACEMESH_TESSELATION, (255,255,255), 1)
                    self.draw_line_set(self.frame, face.landmarks, FACEMESH_LIPS, (0,128,255), 2)
                    self.draw_line_set(self.frame, face.landmarks, FACEMESH_EYES_EYEBROWS, (0,255,0), 2)
                    self.draw_line_set(self.frame, face.landmarks, FACEMESH_FACE_OVAL, (255,255,255), 2)
                    if self.tracker.with_attention:
                        self.draw_line_set(self.frame, face.landmarks, FACEMESH_IRISES, (0,0,255), 2)

                elif self.face_style == 3:
                    p_right_eye = Polygon([ face.landmarks[i,:2] for i in RIGHT_EYE_IDX_MAP])
                    radius_right_iris = np.linalg.norm(face.landmarks[REFINEMENT_IDX_MAP['right iris'][0],:2]-face.landmarks[REFINEMENT_IDX_MAP['right iris'][1],:2])
                    p_right_iris = Point(face.landmarks[REFINEMENT_IDX_MAP['right iris'][0],:2]).buffer(radius_right_iris, resolution=3)
                    try:
                        p_right_iris = p_right_eye.intersection(p_right_iris)
                        cv2.polylines(self.frame, [np.array([(round(x), round(y)) for x,y  in list(p_right_iris.exterior.coords)])], True, (0,0,255), 2)
                    except:
                        pass
                    p_left_eye = Polygon([ face.landmarks[i,:2] for i in LEFT_EYE_IDX_MAP])
                    radius_left_iris = np.linalg.norm(face.landmarks[REFINEMENT_IDX_MAP['left iris'][0],:2]-face.landmarks[REFINEMENT_IDX_MAP['left iris'][1],:2])
                    p_left_iris = Point(face.landmarks[REFINEMENT_IDX_MAP['left iris'][0],:2]).buffer(radius_left_iris, resolution=3)
                    try:
                        p_left_iris = p_left_eye.intersection(p_left_iris)
                        cv2.polylines(self.frame, [np.array([(round(x), round(y)) for x,y  in list(p_left_iris.exterior.coords)])], True, (0,0,255), 2)
                    except:
                        pass
                    self.draw_line_set(self.frame, face.landmarks, FACEMESH_LIPS, (0,128,255), 2)
                    self.draw_line_set(self.frame, face.landmarks, FACEMESH_EYES_EYEBROWS, (0,255,0), 2)
                

            if self.show_metric_landmarks:
                frame_metric = np.zeros((650, 650, 3), dtype=np.uint8)
                cv2.putText(frame_metric, f"Press 's' {'for smoothing' if not self.smooth_metric else 'to cancel smoothing'}", (20,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1)
                points = face.metric_landmarks[:,:2].copy()
                points[:,1] = -points[:,1]
                if self.smooth_metric:
                    points = self.metric_filter.apply(points, timestamp=now())

                points = (points * 29 + 325).astype(int)
                if self.metric_style == 0:
                    for i in self.not_refined_lm_idx:
                        cv2.circle(frame_metric, tuple(points[i]), radius, (255,255,255), -1)
                    if self.tracker.with_attention:
                        for i in REFINEMENT_IDX_MAP['lips']:
                            cv2.circle(frame_metric, tuple(points[i]), radius, (0,128,255), -1)
                        for i in REFINEMENT_IDX_MAP['left eye']:
                            cv2.circle(frame_metric, tuple(points[i]), radius, (0,255,0), -1)
                        for i in REFINEMENT_IDX_MAP['right eye']:
                            cv2.circle(frame_metric, tuple(points[i]), radius, (0,255,0), -1)
                        for i in REFINEMENT_IDX_MAP['left iris']:
                            cv2.circle(frame_metric, tuple(points[i]), radius, (0,0,255), -1)
                        for i in REFINEMENT_IDX_MAP['right iris']:
                            cv2.circle(frame_metric, tuple(points[i]), radius, (0,0,255), -1)
                elif self.metric_style in [1, 2]:

                    if self.metric_style == 2:
                        self.draw_line_set(frame_metric, points, FACEMESH_TESSELATION, (255,255,255), 1)
                    self.draw_line_set(frame_metric, points, FACEMESH_LIPS, (0,128,255), 2)
                    self.draw_line_set(frame_metric, points, FACEMESH_EYES_EYEBROWS, (0,255,0), 2)
                    self.draw_line_set(frame_metric, points, FACEMESH_FACE_OVAL, (255,255,255), 2)
                    if self.tracker.with_attention:
                        self.draw_line_set(frame_metric, points, FACEMESH_IRISES, (0,0,255), 2)


                elif self.metric_style == 3:
                    p_right_eye = Polygon([ points[i] for i in RIGHT_EYE_IDX_MAP])
                    radius_right_iris = np.linalg.norm(points[REFINEMENT_IDX_MAP['right iris'][0]]-points[REFINEMENT_IDX_MAP['right iris'][1]])
                    p_right_iris = Point(points[REFINEMENT_IDX_MAP['right iris'][0]]).buffer(radius_right_iris, resolution=3)
                    try:
                        p_right_iris = p_right_eye.intersection(p_right_iris)
                        cv2.polylines(frame_metric, [np.array([(round(x), round(y)) for x,y  in list(p_right_iris.exterior.coords)])], True, (0,0,255), 2)
                    except:
                        pass
                    p_left_eye = Polygon([ points[i] for i in LEFT_EYE_IDX_MAP])
                    radius_left_iris = np.linalg.norm(points[REFINEMENT_IDX_MAP['left iris'][0]]-points[REFINEMENT_IDX_MAP['left iris'][1]])
                    p_left_iris = Point(points[REFINEMENT_IDX_MAP['left iris'][0]]).buffer(radius_left_iris, resolution=3)
                    try:
                        p_left_iris = p_left_eye.intersection(p_left_iris)
                        cv2.polylines(frame_metric, [np.array([(round(x), round(y)) for x,y  in list(p_left_iris.exterior.coords)])], True, (0,0,255), 2)
                    except:
                        pass
                    self.draw_line_set(frame_metric, points, FACEMESH_LIPS, (0,128,255), 2)
                    self.draw_line_set(frame_metric, points, FACEMESH_EYES_EYEBROWS, (0,255,0), 2)
                cv2.imshow("Metric landmarks", frame_metric)

            if self.show_pose and self.pose_style > 0:
                if self.pose_style == 1: # Draw Origin, x y and z axis
                    points_to_project = [(0,0,0), (10,0,0), (0,10,0), (0,0,10)]
                elif self.pose_style == 2: # Draw a line starting from the nose tip
                    nose_tip = face.metric_landmarks[4]
                    points_to_project = [nose_tip, 2.5 * nose_tip]
                else:
                    points_to_project = [face.metric_landmarks[i] * self.magnify_factor for i in range(468)]

                points2D,_ = cv2.projectPoints(
                    np.array(points_to_project, dtype=np.float32),
                    face.pose_rotation_vector,
                    face.pose_translation_vector,
                    self.tracker.rgb_matrix,
                    self.tracker.rgb_dist_coef
                )
                points2D = points2D.squeeze().astype(int)
                if self.pose_style == 1:
                    origin, x_axis, y_axis, z_axis = points2D
                    cv2.line(self.frame, origin, x_axis, (0, 0, 255), 3)
                    cv2.line(self.frame, origin, y_axis, (0, 255, 0), 3)
                    cv2.line(self.frame, origin, z_axis, (255, 0, 0), 3)
                elif self.pose_style == 2:
                    nose_tip , nose_tip_extended = points2D
                    cv2.line(self.frame, nose_tip, nose_tip_extended, (255, 0, 0), 3)
                else:
                    # pose_style = 3 : bubble effect
                    # pose_style = 4 : bubble effect with pulsation
                    for i in range(468):
                        try:
                            cv2.circle(self.frame, tuple(points2D[i]), int(self.radius), self.random_color, 2)
                        except:
                            pass
                    if self.pose_style == 3:
                        self.radius *= 1.18
                        self.magnify_factor *= 1.1
                        if self.magnify_factor > 10:
                            self.magnify_factor = 1
                            self.radius = 1
                            self.random_color = random.choice(color_list)
                    else:
                        self.i += 1
                        step = 10
                        self.radius = 1 + 7*(1 - abs(math.cos(self.i*1.57/step)))
                        self.magnify_factor = 1 + abs(1.5*math.sin(self.i*1.57/step))
                        if self.i % (2*step) == 0:
                            self.random_color = random.choice(color_list)

            if self.show_xyz:
                x0, y0 = info_ref_x - 40, info_ref_y + 40

                cv2.rectangle(self.frame, (x0,y0), (x0+100, y0+85), (220,220,240), -1)
                cv2.putText(self.frame, f"X:{face.xyz[0]/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_PLAIN, 1, (20,180,0), 2)
                cv2.putText(self.frame, f"Y:{face.xyz[1]/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
                cv2.putText(self.frame, f"Z:{face.xyz[2]/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            if self.show_xyz_zone:
                # Show zone on which the spatial data were calculated
                cv2.rectangle(self.frame, tuple(face.xyz_zone[0:2]), tuple(face.xyz_zone[2:4]), (180,0,180), 2)


    def draw(self, frame, faces, hands):
        if self.laconic:
            self.frame = np.zeros_like(frame)
        else:
            self.frame = frame
        for face in faces:
            self.draw_face(face)
        for hand in hands:
            self.draw_hand(hand)
        return self.frame

    def exit(self):
        if self.output:
            self.output.release()
        cv2.destroyAllWindows()

    def waitKey(self, delay=1):
        if self.show_fps:
                self.tracker.fps.draw(self.frame, orig=(50,50), size=1, color=(240,180,100))
        cv2.imshow(self.window_title, self.frame)
            
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay) 

        if key == 32:
            # Pause on space bar
            key = cv2.waitKey(0)
            if key == ord('s'):
                print("Snapshot saved in snapshot.jpg")
                cv2.imwrite("snapshot.jpg", self.frame)
        elif key == ord('1'):
            self.show_hand_rot_rect = not self.show_hand_rot_rect
        elif key == ord('2'):
            self.show_hand_landmarks = not self.show_hand_landmarks
        elif key == ord('3'):
            self.show_face_rot_rect = not self.show_face_rot_rect
        elif key == ord('4'):
            self.show_face_landmarks = not self.show_face_landmarks
        elif key == ord('5'):
            if self.tracker.xyz:
                self.show_xyz = not self.show_xyz    
        elif key == ord('6'):
            if self.tracker.xyz:
                self.show_xyz_zone = not self.show_xyz_zone 
        elif key == ord('g'):
            if self.tracker.use_gesture:
                self.show_gesture = not self.show_gesture
        elif key == ord('f'):
            nb_styles = 4 if self.tracker.with_attention else 3
            self.face_style = (self.face_style + 1) % nb_styles
        elif key == ord('m'):
            nb_styles = 5 if self.tracker.with_attention else 3
            self.metric_style = (self.metric_style + 1) % nb_styles
        elif key == ord('p'):
            nb_styles = 5
            self.pose_style = (self.pose_style + 1) % nb_styles
        elif key == ord('h'):
            nb_styles = 3
            self.hand_style = (self.hand_style + 1) % nb_styles
        elif key == ord('b'):
            self.laconic = not self.laconic
        elif key == ord('s'):
            # Smoothing filter on metric_landmarks
            self.smooth_metric = not self.smooth_metric
            if self.smooth_metric:
                self.metric_filter.reset()
        return key
