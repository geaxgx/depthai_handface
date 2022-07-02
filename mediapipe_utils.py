import cv2
import numpy as np
from collections import namedtuple
from math import ceil, sqrt, exp, pi, floor, sin, cos, atan2, gcd

# To not display: RuntimeWarning: overflow encountered in exp
# in line:  scores = 1 / (1 + np.exp(-scores))
np.seterr(over='ignore')

class HandRegion:
    """
        Attributes:
        pd_score : detection score
        pd_box : detection box [x, y, w, h], normalized [0,1] in the squared image
        pd_kps : detection keypoints coordinates [x, y], normalized [0,1] in the squared image
        rect_x_center, rect_y_center : center coordinates of the rotated bounding rectangle, normalized [0,1] in the squared image
        rect_w, rect_h : width and height of the rotated bounding rectangle, normalized in the squared image (may be > 1)
        rotation : rotation angle of rotated bounding rectangle with y-axis in radian
        rect_x_center_a, rect_y_center_a : center coordinates of the rotated bounding rectangle, in pixels in the squared image
        rect_w, rect_h : width and height of the rotated bounding rectangle, in pixels in the squared image
        rect_points : list of the 4 points coordinates of the rotated bounding rectangle, in pixels 
                expressed in the squared image during processing,
                expressed in the source rectangular image when returned to the user
        lm_score: global landmark score
        norm_landmarks : 3D landmarks coordinates in the rotated bounding rectangle, normalized [0,1]
        landmarks : 2D landmark coordinates in pixel in the source rectangular image
        world_landmarks : 3D landmark coordinates in meter
        handedness: float between 0. and 1., > 0.5 for right hand, < 0.5 for left hand,
        label: "left" or "right", handedness translated in a string,
        xyz: real 3D world coordinates of the wrist landmark, or of the palm center (if landmarks are not used),
        xyz_zone: (left, top, right, bottom), pixel coordinates in the source rectangular image 
                of the rectangular zone used to estimate the depth
        gesture: (optional, set in recognize_gesture() when use_gesture==True) string corresponding to recognized gesture ("ONE","TWO","THREE","FOUR","FIVE","FIST","OK","PEACE") 
                or None if no gesture has been recognized
        """
    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
        self.pd_score = pd_score # Palm detection score 
        self.pd_box = pd_box # Palm detection box [x, y, w, h] normalized
        self.pd_kps = pd_kps # Palm detection keypoints

    def get_rotated_world_landmarks(self):
        world_landmarks_rotated = self.world_landmarks.copy()
        sin_rot = sin(self.rotation)
        cos_rot = cos(self.rotation)
        rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
        world_landmarks_rotated[:,:2] = np.dot(world_landmarks_rotated[:,:2], rot_m)
        return world_landmarks_rotated

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

class HandednessAverage:
    """
    Used to store the average handeness
    Why ? Handedness inferred by the landmark model is not perfect. For certain poses, it is not rare that the model thinks 
    that a right hand is a left hand (or vice versa). Instead of using the last inferred handedness, we prefer to use the average 
    of the inferred handedness on the last frames. This gives more robustness.
    """
    def __init__(self):
        self._total_handedness = 0
        self._nb = 0
    def update(self, new_handedness):
        self._total_handedness += new_handedness
        self._nb += 1
        return self._total_handedness / self._nb
    def reset(self):
        self._total_handedness = self._nb = 0

class Face:
    """
        Attributes:
        - rect_x_center, rect_y_center : center coordinates of the rotated bounding rectangle, normalized [0,1] in the squared image
        - rect_w, rect_h : width and height of the rotated bounding rectangle, normalized in the squared image (may be > 1)
        - rotation : rotation angle of rotated bounding rectangle with y-axis in radian
        - rect_x_center_a, rect_y_center_a : center coordinates of the rotated bounding rectangle, in pixels in the squared image
        - rect_w_a, rect_h_a : width and height of the rotated bounding rectangle, in pixels in the squared image
        - rect_points : list of the 4 points coordinates of the rotated bounding rectangle, in pixels 
                expressed in the squared image during processing,
                expressed in the source rectangular image when returned to the user
        - lm_score: global landmark score
        - norm_landmarks : 3D landmarks coordinates in the rotated bounding rectangle, normalized [0,1]
        - landmarks : 3D landmark coordinates in pixel in the source rectangular image
        
        The following attributes are populated only if HandFaceTracker.xyz is True:
        - xyz: real 3D world coordinates of the wrist landmark, or of the palm center (if landmarks are not used),
        - xyz_zone: (left, top, right, bottom), pixel coordinates in the source rectangular image 
                of the rectangular zone used to estimate the depth
        
        The 4 following attributes are populated only if HandFaceTracker.use_face_pose is True:
        - pose_transform_mat: face pose transformation matrix that provides mapping from the static canonical face model to the runtime face. 
                Tries to distinguish a head pose change from a facial expression change and to only reflect the former.
                Is a 4x4 matrix and contains only the following components:
                    * Uniform scale
                    * Rotation
                    * Translation
                The last row is `[0 0 0 1]`
        The 2 following attributes contains the same information than 'pose_transform_mat'
        but are more practical when we want to project points using cv2.projectPoints :
        - pose_rotation_vector: cv2.Rodrigues(face.pose_transform_mat[:3, :3])
        - pose_translation_vector: pose_transform_mat[:3, 3, None]
        - metric_landmarks: 
                Corresponds to the 3D runtime face metric landmarks (unit=cm) aligned with the canonical metric face landmarks.
        """

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))
        

        
def rotated_rect_to_points(cx, cy, w, h, rotation):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    points = []
    p0x = cx - a*h - b*w
    p0y = cy + b*h - a*w
    p1x = cx + a*h - b*w
    p1y = cy - b*h - a*w
    p2x = int(2*cx - p0x)
    p2y = int(2*cy - p0y)
    p3x = int(2*cx - p1x)
    p3y = int(2*cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [[p0x,p0y], [p1x,p1y], [p2x,p2y], [p3x,p3y]]


def find_isp_scale_params(size, resolution, is_height=True):
    """
    Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
    This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
    resolution: sensor resolution (width, height)
    is_height : boolean that indicates if the value 'size' represents the height or the width of the image
    Returns: valid size, (numerator, denominator)
    """
    # We want size >= 288 (first compatible size > lm_input_size)
    if size < 288:
        size = 288

    width, height = resolution

    # We are looking for the list on integers that are divisible by 16 and
    # that can be written like n/d where n <= 16 and d <= 63
    if is_height:
        reference = height 
        other = width
    else:
        reference = width 
        other = height
    size_candidates = {}
    for s in range(288,reference,16):
        f = gcd(reference, s)
        n = s//f
        d = reference//f
        if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
            size_candidates[s] = (n, d)
            
    # What is the candidate size closer to 'size' ?
    min_dist = -1
    for s in size_candidates:
        dist = abs(size - s)
        if min_dist == -1:
            min_dist = dist
            candidate = s
        else:
            if dist > min_dist: break
            candidate = s
            min_dist = dist
    return candidate, size_candidates[candidate]

def distance(a, b):
    """
    a, b: 2 points (in 2D or 3D)
    """
    return np.linalg.norm(a-b)

def angle(a, b, c):
    # https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    # a, b and c : points as np.array([x, y, z]) 
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def recognize_gesture(hand):           
    # Finger states
    # state: -1=unknown, 0=close, 1=open
    d_3_5 = distance(hand.norm_landmarks[3], hand.norm_landmarks[5])
    d_2_3 = distance(hand.norm_landmarks[2], hand.norm_landmarks[3])
    angle0 = angle(hand.norm_landmarks[0], hand.norm_landmarks[1], hand.norm_landmarks[2])
    angle1 = angle(hand.norm_landmarks[1], hand.norm_landmarks[2], hand.norm_landmarks[3])
    angle2 = angle(hand.norm_landmarks[2], hand.norm_landmarks[3], hand.norm_landmarks[4])
    hand.thumb_angle = angle0+angle1+angle2
    if angle0+angle1+angle2 > 460 and d_3_5 / d_2_3 > 1.2: 
        hand.thumb_state = 1
    else:
        hand.thumb_state = 0

    if hand.norm_landmarks[8][1] < hand.norm_landmarks[7][1] < hand.norm_landmarks[6][1]:
        hand.index_state = 1
    elif hand.norm_landmarks[6][1] < hand.norm_landmarks[8][1]:
        hand.index_state = 0
    else:
        hand.index_state = -1

    if hand.norm_landmarks[12][1] < hand.norm_landmarks[11][1] < hand.norm_landmarks[10][1]:
        hand.middle_state = 1
    elif hand.norm_landmarks[10][1] < hand.norm_landmarks[12][1]:
        hand.middle_state = 0
    else:
        hand.middle_state = -1

    if hand.norm_landmarks[16][1] < hand.norm_landmarks[15][1] < hand.norm_landmarks[14][1]:
        hand.ring_state = 1
    elif hand.norm_landmarks[14][1] < hand.norm_landmarks[16][1]:
        hand.ring_state = 0
    else:
        hand.ring_state = -1

    if hand.norm_landmarks[20][1] < hand.norm_landmarks[19][1] < hand.norm_landmarks[18][1]:
        hand.little_state = 1
    elif hand.norm_landmarks[18][1] < hand.norm_landmarks[20][1]:
        hand.little_state = 0
    else:
        hand.little_state = -1

    # Gesture
    if hand.thumb_state == 1 and hand.index_state == 1 and hand.middle_state == 1 and hand.ring_state == 1 and hand.little_state == 1:
        hand.gesture = "FIVE"
    elif hand.thumb_state == 0 and hand.index_state == 0 and hand.middle_state == 0 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "FIST"
    elif hand.thumb_state == 1 and hand.index_state == 0 and hand.middle_state == 0 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "OK" 
    elif hand.thumb_state == 0 and hand.index_state == 1 and hand.middle_state == 1 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "PEACE"
    elif hand.thumb_state == 0 and hand.index_state == 1 and hand.middle_state == 0 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "ONE"
    elif hand.thumb_state == 1 and hand.index_state == 1 and hand.middle_state == 0 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "TWO"
    elif hand.thumb_state == 1 and hand.index_state == 1 and hand.middle_state == 1 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "THREE"
    elif hand.thumb_state == 0 and hand.index_state == 1 and hand.middle_state == 1 and hand.ring_state == 1 and hand.little_state == 1:
        hand.gesture = "FOUR"
    else:
        hand.gesture = None


# From: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt

REFINEMENT_IDX_MAP = {
    "lips":
        [
          # Lower outer.
          61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
          # Upper outer (excluding corners).
          185, 40, 39, 37, 0, 267, 269, 270, 409,
          # Lower inner.
          78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
          # Upper inner (excluding corners).
          191, 80, 81, 82, 13, 312, 311, 310, 415,
          # Lower semi-outer.
          76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306,
          # Upper semi-outer (excluding corners).
          184, 74, 73, 72, 11, 302, 303, 304, 408,
          # Lower semi-inner.
          62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292,
          # Upper semi-inner (excluding corners).
          183, 42, 41, 38, 12, 268, 271, 272, 407
        ],

    "left eye":
        [
          # Lower contour.
          33, 7, 163, 144, 145, 153, 154, 155, 133,
          # upper contour (excluding corners).
          246, 161, 160, 159, 158, 157, 173,
          # Halo x2 lower contour.
          130, 25, 110, 24, 23, 22, 26, 112, 243,
          # Halo x2 upper contour (excluding corners).
          247, 30, 29, 27, 28, 56, 190,
          # Halo x3 lower contour.
          226, 31, 228, 229, 230, 231, 232, 233, 244,
          # Halo x3 upper contour (excluding corners).
          113, 225, 224, 223, 222, 221, 189,
          # Halo x4 upper contour (no lower because of mesh structure) or
          # eyebrow inner contour.
          35, 124, 46, 53, 52, 65,
          # Halo x5 lower contour.
          143, 111, 117, 118, 119, 120, 121, 128, 245,
          # Halo x5 upper contour (excluding corners) or eyebrow outer contour.
          156, 70, 63, 105, 66, 107, 55, 193
        ],
      
    "right eye":
        [
          # Lower contour.
          263, 249, 390, 373, 374, 380, 381, 382, 362,
          # Upper contour (excluding corners).
          466, 388, 387, 386, 385, 384, 398,
          # Halo x2 lower contour.
          359, 255, 339, 254, 253, 252, 256, 341, 463,
          # Halo x2 upper contour (excluding corners).
          467, 260, 259, 257, 258, 286, 414,
          # Halo x3 lower contour.
          446, 261, 448, 449, 450, 451, 452, 453, 464,
          # Halo x3 upper contour (excluding corners).
          342, 445, 444, 443, 442, 441, 413,
          # Halo x4 upper contour (no lower because of mesh structure) or
          # eyebrow inner contour.
          265, 353, 276, 283, 282, 295,
          # Halo x5 lower contour.
          372, 340, 346, 347, 348, 349, 350, 357, 465,
          # Halo x5 upper contour (excluding corners) or eyebrow outer contour.
          383, 300, 293, 334, 296, 336, 285, 417
        ],
    "left iris":
        [
          # Center.
          468,
          # Iris right edge.
          469,
          # Iris top edge.
          470,
          # Iris left edge.
          471,
          # Iris bottom edge.
          472
        ],
    "right iris":
        [
          # Center.
          473,
          # Iris right edge.
          474,
          # Iris top edge.
          475,
          # Iris left edge.
          476,
          # Iris bottom edge.
          477
        ]
}

XY_REFINEMENT_IDX_MAP = {k:np.column_stack((v, v)) for k,v in REFINEMENT_IDX_MAP.items()}

Z_REFINEMENT_IDX_MAP = {
    "left iris": 
        [
            # Lower contour.
            33, 7, 163, 144, 145, 153, 154, 155, 133,
            # Upper contour (excluding corners).
            246, 161, 160, 159, 158, 157, 173
        ],
    "right iris":
        [
            # Lower contour.
            263, 249, 390, 373, 374, 380, 381, 382, 362,
            # Upper contour (excluding corners).
            466, 388, 387, 386, 385, 384, 398
        ]
}

RIGHT_EYE_IDX_MAP = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                    398, 384, 385, 386, 387, 388, 466 ]
RIGHT_IRIS_IDX_MAP = [474, 475, 476, 477]
LEFT_EYE_IDX_MAP = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                    173, 157, 158, 159, 160, 161, 246 ]
LEFT_IRIS_IDX_MAP = [469, 470, 471, 472]

#
# Filters
#

class OneEuroFilter: 
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/one_euro_filter.cc
    Paper: https://cristal.univ-lille.fr/~casiez/1euro/

    frequency:  
                Frequency of incoming frames defined in seconds. Used
                only if can't be calculated from provided events (e.g.
                on the very first frame). Default=30
    min_cutoff:  
                Minimum cutoff frequency. Start by tuning this parameter while
                keeping `beta=0` to reduce jittering to the desired level. 1Hz
                (the default value) is a a good starting point.
    beta:       
                Cutoff slope. After `min_cutoff` is configured, start
                increasing `beta` value to reduce the lag introduced by the
                `min_cutoff`. Find the desired balance between jittering and lag. Default=0
    derivate_cutoff: 
                Cutoff frequency for derivate. It is set to 1Hz in the
                original algorithm, but can be turned to further smooth the
                speed (i.e. derivate) on the object. Default=1
    '''
    def __init__(self,
                frequency=30,
                min_cutoff=1,
                beta=0,
                derivate_cutoff=1,
                ):
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivate_cutoff = derivate_cutoff
        self.x = LowPassFilter(self.get_alpha(min_cutoff))
        self.dx = LowPassFilter(self.get_alpha(derivate_cutoff))
        self.last_timestamp = 0

    def get_alpha(self, cutoff):
        '''
        te = 1.0 / self.frequency
        tau = 1.0 / (2 * Math.PI * cutoff)
        result = 1 / (1.0 + (tau / te))
        '''
        return 1.0 / (1.0 + (self.frequency / (2 * np.pi * cutoff)))

    def apply(self, value, timestamp):
        '''
        Applies filter to the value.
        timestamp in s associated with the value (for instance,
        timestamp of the frame where you got value from).
        '''
        value = value.copy()
        if self.last_timestamp >= timestamp:
            # Results are unpreditable in this case, so nothing to do but return same value.
            return value

        # Update the sampling frequency based on timestamps.
        if self.last_timestamp != 0 and timestamp != 0:
            self.frequency = 1 / (timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

        # Estimate the current variation per second.
        if self.x.has_last_raw_value():
            dvalue = (value - self.x.last_raw_value()) * self.frequency
        else:
            dvalue = 0
        edvalue = self.dx.apply_with_alpha(dvalue, self.get_alpha(self.derivate_cutoff))

        # Use it to update the cutoff frequency
        cutoff = self.min_cutoff + self.beta * np.abs(edvalue)

        # filter the given value.
        return self.x.apply_with_alpha(value, self.get_alpha(cutoff))

    def reset(self):
        self.x.reset()
        self.dx.reset()

        
class LowPassFilter:
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/low_pass_filter.cc
    Note that 'value' can be a numpy array
    '''
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.initialized = False

    def apply(self, value):
        if self.initialized:
            # Regular lowpass filter.
            result = self.alpha * value + (1 - self.alpha) * self.stored_value
        else:
            result = value
            self.initialized = True
        self.raw_value = value
        self.stored_value = result
        return result

    def apply_with_alpha(self, value, alpha):
        self.alpha = alpha
        return self.apply(value)

    def has_last_raw_value(self):
        return self.initialized

    def last_raw_value(self):
        return self.raw_value

    def last_value(self):
        return self.stored_value

    def reset(self):
        self.initialized = False

