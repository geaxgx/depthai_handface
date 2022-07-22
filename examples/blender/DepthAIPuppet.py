# import sys

# sys.path.append("/home/gx/g2/depthai_handface/examples/blender/")

# import demo_vincent
# import importlib
# importlib.reload(demo_vincent)
# demo_vincent.main()


# Modify the path to find HandFaceTracker and HandFaceRenderer
import sys
import cv2
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from HandFaceTracker import HandFaceTracker
from HandFaceRenderer import HandFaceRenderer

from Filters import OneEuroFilter

from time import monotonic

import bpy
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import IntProperty, BoolProperty, PointerProperty, EnumProperty, FloatProperty

from math import pi

from mathutils import Matrix    

from pathlib import Path

# Force render engine to workbench and shading
bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
for area in bpy.context.screen.areas: 
    if area.type == 'VIEW_3D':
        for space in area.spaces: 
            if space.type == 'VIEW_3D':
                space.shading.type = 'RENDERED'

script_dir = Path( __file__ ).parent.absolute()
img_dir = script_dir / 'media'

background_image_name = {
    'DEFAULT': 'background_default',
    'SNAPSHOT': 'background_snapshot'
}
background_image_file = {
    'DEFAULT': script_dir / 'media/background_default.jpg',
    'SNAPSHOT': script_dir / 'media/background_snapshot_init.png'
}
background_image_alpha = {
    'DEFAULT': 0.68,
    'SNAPSHOT': 1
}

background_images = {}

for k in background_image_name.keys():
    for img in bpy.data.images:
        if img.name == background_image_name[k]:
            break
    else:
        img = bpy.data.images.load(str(background_image_file[k]))
    img.name = background_image_name[k]
    background_images[k] = img

class Range:
    def __init__(self, src_pts, target_pts, dynamic_min_max=True, limited=False):
        assert len(src_pts) >= 2, "At least 2 points are required"
        assert len(src_pts) == len(target_pts), "Number of source points must be equal to number of target points"
        for i in range(1, len(src_pts)):
            assert src_pts[i-1] < src_pts[i], "Source points not in ascending order"
        self.src_pts = src_pts
        self.target_pts = target_pts
        self.dynamic_min_max = dynamic_min_max
        self.limited = limited
    def target(self, src):
        if self.dynamic_min_max:
            if src < self.src_pts[0]:
                self.src_pts[0] = src
            elif src > self.src_pts[-1]:
                self.src_pts[-1] = src
        for i in range(1, len(self.src_pts)):
            if src <= self.src_pts[i]: break
        ratio = (src - self.src_pts[i-1]) / (self.src_pts[i] - self.src_pts[i-1])
        target = ratio * (self.target_pts[i] -self.target_pts[i-1]) + self.target_pts[i-1]
        if self.limited:
            if (self.target_pts[0] < self.target_pts[1] and target < self.target_pts[0]) or \
                (self.target_pts[0] > self.target_pts[1] and target > self.target_pts[0]):
                target = self.target_pts[0]
            elif (self.target_pts[-2] < self.target_pts[-1] and target > self.target_pts[-1]) or \
                (self.target_pts[-2] > self.target_pts[-1] and target < self.target_pts[-1]):
                target = self.target_pts[-1]
        return target


def tag_redraw(context, space_type="VIEW_3D", region_type="UI"):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.spaces[0].type == space_type:
                for region in area.regions:
                    if region.type == region_type:
                        region.tag_redraw()



def upd_eyeglasses(self, context):
    bpy.data.objects['GEO-vincent_eyeglasses'].hide_set(not self.eyeglasses)

def upd_facemesh_with_attention(self, context):
    if not self.facemesh_with_attention:
        self.looking_direction = 'AHEAD'

def get_look_items(self, context):
    if self.facemesh_with_attention:
        items = [ ('TRACKED', 'tracked', 'Following irises movement (only with model with attention)', 0)]
    else:
        items = []
    items += [
            ('AHEAD', 'ahead', 'Always looking ahead', 1),
            ('CAMERA', 'camera', 'Always looking at the blender camera', 2)
            ]
    return items

def upd_looking_direction(self, context):
    constraints = bpy.data.objects['RIG-Vincent'].pose.bones['look_mstr'].constraints
    for c in constraints:
        c.enabled = self.looking_direction == "AHEAD"
    if self.looking_direction != "TRACKED":
        bpy.data.objects["RIG-Vincent"].pose.bones['look'].matrix_basis = Matrix()

def upd_background_image(self, context):
    cam = bpy.data.cameras['Camera']
    bg_img = cam.background_images[0]
    # for img in bpy.data.images:
    #     if img.name == background_image_name[self.background_image]:
    #         break
    
    # bg_img.image = img
    bg_img.image = background_images[self.background_image]
    bg_img.alpha = background_image_alpha[self.background_image]

class DepthAIPuppetProperties(PropertyGroup):
    facemesh_with_attention: BoolProperty(
        name = "With Attention",
        description = "Use Facemesh with attention model.\n\
The 'with attention' model can track irises but is slower.\n\
With the faster basic model, the eyes are always looking toward the camera or ahead depending on the 'Look' setting.",
        default = True,
        update=upd_facemesh_with_attention
    )
    
    depthai_xyz: BoolProperty(
        name = "XYZ location",
        description = "Use DepthAI spatial location.\n\
If checked, the puppet head can move in space following the tracked face.",
        default = True
    )
    
    depthai_autofocus: BoolProperty(
        name = "Autofocus",
        description = "Enable camera autofocus\n\
Note: when 'xyz' is checked, the focus used during camera calibration is used.",
        default = True
    ) 
    
    depthai_focus_value: IntProperty(
        name = "Manual focus value",
        description = "Manual focus value when autofocus is disabled",
        min = 1,
        max = 255,
        default= 130
    )


    depthai_max_hands: EnumProperty(
        items = [('0', '0', 'No hand tracking'), ('1', '1', '1'), ('2', '2', '2')],
        name = "Max hands",
        description="Maximum number of hands tracked (0, 1 or 2). 0 means no hand tracking.\n\
Currently, hands are not drawn, hand tracking is only used to control the puppet.",
        default = "1"
    )

    is_tracker_running: BoolProperty(
        name = "is the tracker running",
        description = "Switch to run/stop the tracker",
        default = False,
    )

    eyebrow_sensitivity: FloatProperty(
        name = "Eyebrow sensitivity",
        description = "Eyebrow sensitivity",
        min = 0.5,
        max = 1.5,
        default = 1.0
    )

    eyeglasses: BoolProperty(
        name = "Eyeglasses",
        description = "Eyeglasses",
        default = True,
        update = upd_eyeglasses
    )

    looking_direction: EnumProperty(
        items = get_look_items,
        name = "Look",
        description = "Looking direction",
        update = upd_looking_direction
    )

    horiz_look_sensitivity: FloatProperty(
        name = "Horizontal look sensitivity",
        description = "Horizontal look sensitivity",
        min = 0.5,
        max = 1.5,
        default = 1.0
    )

    vert_look_sensitivity: FloatProperty(
        name = "Vertical look sensitivity",
        description = "Vertical look sensitivity",
        min = 0.5,
        max = 1.5,
        default = 1.0
    )

    vert_mouth_opening_sensitivity: FloatProperty(
        name = "Vertical mouth opening sensitivity",
        description = "Vertical mouth opening sensitivity",
        min = 0.5,
        max = 1.5,
        default = 1.0
    )

    mouth_width_sensitivity: FloatProperty(
        name = "Mouth width sensitivity",
        description = "Mouth width sensitivity",
        min = 0.5,
        max = 1.5,
        default = 1.0
    )

    smile_sensitivity: FloatProperty(
        name = "Smile sensitivity",
        description = "Smile (v) sensitivity",
        min = 0.5,
        max = 1.5,
        default = 1.0
    )

    antismile_sensitivity: FloatProperty(
        name = "Anti-smile sensitivity",
        description = "Anti-smile (^) sensitivity",
        min = 0.5,
        max = 1.5,
        default = 1.0
    )
    
    background_image: EnumProperty(
        items = [('DEFAULT', 'Default', 'Default'), ('SNAPSHOT', 'Snapshot', 'Snapshot from the RGB camera')],
        name = "Background image",
        description = "Image used for the background",
        update = upd_background_image
    )
    take_snapshot: BoolProperty(
        name = "Take snapshot",
        description = "Take a snapshot from the RGB camera for the background",
        default = False
    )
    
class DepthAIPuppetPanel(Panel):

    bl_label = "DepthAI Puppet"
    bl_idname = "OBJECT_PT_depthai_puppet"
    bl_space_type = 'VIEW_3D' 
    bl_region_type = 'UI' 
    bl_category = "DepthAI Puppet"
    
    def draw(self, context):
        props = context.scene.dai_props
        layout = self.layout
        box = layout.box()
        box.label(text="DepthAI settings", icon='TOOL_SETTINGS')
        box.enabled = not props.is_tracker_running

        row = box.row()
        row.prop(props, "facemesh_with_attention")
        row = box.row(align=True)
        row.label(text="Hands:")
        row.prop(props, "depthai_max_hands", expand=True)
        row = box.row()
        row.prop(props, "depthai_xyz")

        if not props.depthai_xyz:
            row = box.row()
            row.prop(props, "depthai_autofocus")
            if not props.depthai_autofocus:
                row.prop(props, "depthai_focus_value")

        row = layout.row()
        row.scale_y = 2.0
        if props.is_tracker_running:
            row.prop(props, "is_tracker_running", text="STOP", icon="SNAP_FACE", toggle=1)
        else:
            row.operator("wm.handface_operator", text="Start !", icon="PLAY")

        box = layout.box()
        box.label(text="Runtime settings", icon='SETTINGS')
        row = box.row(align=True)
        row.label(text="Look:")
        row.prop(props, "looking_direction", expand=True)
        col = box.column(align=True)
        col.label(text="Sensitivity:")
        col.prop(props, "horiz_look_sensitivity", text="Horizontal look")
        col.prop(props, "vert_look_sensitivity", text="Vertical look")
        col.prop(props, "eyebrow_sensitivity", text="Eyebrows")
        col.prop(props, "vert_mouth_opening_sensitivity", text="Mouth =")
        col.prop(props, "mouth_width_sensitivity", text="Mouth ||")

        col.prop(props, "smile_sensitivity", text="Smile v")
        col.prop(props, "antismile_sensitivity", text="Anti-smile ^")

        row = box.row()
        row.prop(props, "eyeglasses")


        row = box.row(align=True)
        row.label(text="Background:")
        row.prop(props, "background_image", expand=True)
        row = box.row()
        row.enabled = props.is_tracker_running
        row.prop(props, "take_snapshot", toggle=1)

class DepthAIPuppetOperator(Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.handface_operator"
    bl_label = "HandFace Animation Operator"
    bl_description = "Run the Face tracker"
    nb_hands = 1

    stop = False
    selfie = True
    head_filter = OneEuroFilter(min_cutoff=0.1, beta=1.5)
    first_smile = None
    first_brow_LR = None

    # Head rotation: directly determined from face.pose_transform_mat
    init_head_rot_quat = None
    head_rot_filter = OneEuroFilter(min_cutoff=1, beta=1)

    # Head translation: based on face.xyz (real face location in camera C.S).
    #  We don't directly translate the head. Instead we translate the torso.
    init_xyz = None
    torso_filter = OneEuroFilter(min_cutoff=0.1, beta=5)

    # Horizontal looking direction (only for facemesh model with attention):
    # Based on the iris location on the horizontal segment left-eye corner| right-eye corner.
    # More precisely, we calculate the ratio "distance from iris to eye left corner"
    # upon "distance left corner - right corner"
    # We don't one looking direction per eye. Instead, the looking direction is shared and is given by the average ratio.
    # In practice, ratio stays between 0.33 (looking right from own person perspective) and 0.66 (looking left)
    # horiz_look_range = Range(src_pts=[-0.15, 0.15], target_pts=[0.35, -0.35])
    horiz_look_sensitivity = None
    horiz_look_filter = OneEuroFilter(min_cutoff=0.1, beta=5)
    init_horiz_look = None

    # Vertical looking direction: we use the same landmarks as for the horizontal looking
    # but we focus only on the y component.
    # vert_look_range = Range(src_pts=[-0.15, 0.15], target_pts=[-0.3, 0.3])
    vert_look_sensitivity = None
    vert_look_filter = OneEuroFilter(min_cutoff=0.1, beta=5)
    init_vert_look = None

    # Eye brows vertical movements:
    # Based on the ratio of distance between 3 points located on a roughly vertical line:
    # One point on the eye borw, one point above the first point on the forehead, the thrid point below the eye.
    # Each eye brow can move independantly from the other.
    eyebrow_sensitivity = None
    # brow_range = Range(src_pts=[0.17, 0.23], target_pts=[0.018, -0.018])
    brows_filter = OneEuroFilter(min_cutoff=0.1, beta=20)

    # Eye brow inclination: give v-shape or ^-shape to the eye brows.
    # Driven by one hand rotation (so of course, hand tracking must be active).
    # Both eye brows are driven together (but with opposite inclination)
    if nb_hands > 0:
        brow_rotation_range = Range(src_pts=[-0.6, 0.3], target_pts=[0.32, -0.18], dynamic_min_max=False, limited=True)

    # Eye opening/blinking: based on distance between upper and lower eyelids
    eye_opening_range = Range(src_pts=[0.4, 1.2], target_pts=[-0.02, 0.004], dynamic_min_max=False, limited=True)
    eye_opening_filter = OneEuroFilter(min_cutoff=5, beta=40)

    # Mouth opening
    # Vertical opening based on distance middle upper lip (lm 13) and middle lower lip (lm 14)
    # Horizontal width: comparing current distance between left mouth corner (lm 78) and right mouth corner (lm 308)
    # with initial distance
    vert_mouth_opening_sensitivity = 1
    vert_mouth_opening_range = Range(src_pts=[0, 0.1], target_pts=[0, -0.04*vert_mouth_opening_sensitivity])
    vert_mouth_opening_filter = OneEuroFilter(min_cutoff=1, beta=10)
    mouth_width_sensitivity = 1
    mouth_width_range = Range(src_pts=[-1.5, 1], target_pts=[0.025*mouth_width_sensitivity, -0.006*mouth_width_sensitivity])
    mouth_width_filter = OneEuroFilter(min_cutoff=1, beta=10)
    init_mouth_width = None

    #Smiling
    init_smile = None
    smile_sensitivity = 1
    antismile_sensitivity = 1
    smile_range = Range(src_pts=[-0.6, 0, 0.4], target_pts=[smile_sensitivity*45*pi/180, 0, -antismile_sensitivity*25*pi/180], dynamic_min_max=False, limited=True)
    smile_filter = OneEuroFilter(min_cutoff=0.5, beta=0)

    hand_visible_in_prev_frame = False

    set_neutral_position_frames_left = 30
    snp_horiz_look_avg = []
    snp_vert_look_avg = []
    bpy.data.objects['RedCross'].hide_set(False)



    # hand_visible_in_prev_frame = {"left": False, "right": False}

    # # hand_ik_ctrl = {"left": "hand_ik_ctrl_L", "right": "hand_ik_ctrl_R" }
    # hand_ik_ctrl = {"left": "LeftFlag", "right": "RightFlag" }
    # hand_ctrl_name = {"left": "LeftHandLoc", "right": "RightHandLoc"}
    # hand_filter = {"left":OneEuroFilter(min_cutoff=0.1, beta=20), "right":OneEuroFilter(min_cutoff=0.1, beta=20)}


    def modal(self, context, event):
        if (event.type in {'RIGHTMOUSE', 'ESC'}) or self.stop == True:
            return self.cancel(context)
        if event.type == "N":
            if event.value == "RELEASE":
                bpy.data.objects['RedCross'].hide_set(False)
                self.set_neutral_position()
                self.set_neutral_position_frames_left = 20
                self.snp_horiz_look_avg = []
                self.snp_vert_look_avg = []
            return {'RUNNING_MODAL'}
        if event.type == 'TIMER':
            props = context.scene.dai_props
            if not props.is_tracker_running:
                return self.cancel(context)

            frame, faces, hands = self.tracker.next_frame()
            if frame is None: 
                print("Frame is none")
                return self.cancel(context)

            if props.take_snapshot:
                print("SNAPSHOT")
                h,w = frame.shape[:2]
                old_image = bpy.data.images.get(background_image_name['SNAPSHOT'], None)
                if old_image:
                    bpy.data.images.remove(old_image)
                rgb = np.flip(frame, axis=[0, 2])

                rgba = np.ones((h, w, 4), dtype=np.float32)
                rgba[:,:,:-1] = np.float32(rgb) / 255
                new_image = bpy.data.images.new(background_image_name['SNAPSHOT'], w, h)
                new_image.pixels = rgba.flatten()
                background_images['SNAPSHOT'] = new_image
                if props.background_image == 'SNAPSHOT':
                    cam = bpy.data.cameras['Camera']
                    bg_img = cam.background_images[0]
                    bg_img.image = background_images['SNAPSHOT']

                props.take_snapshot = False
                tag_redraw(context)
            
            if len(faces) > 0:

                # Check property updates
                if self.eyebrow_sensitivity != props.eyebrow_sensitivity:
                    self.eyebrow_sensitivity = props.eyebrow_sensitivity
                    self.brow_range = Range(src_pts=[0.17, 0.23], target_pts=[0.018*self.eyebrow_sensitivity, -0.018*self.eyebrow_sensitivity])
                if self.horiz_look_sensitivity != props.horiz_look_sensitivity:
                    self.horiz_look_sensitivity = props.horiz_look_sensitivity
                    self.horiz_look_range = Range(src_pts=[-0.15, 0.15], target_pts=[0.35*self.horiz_look_sensitivity, -0.35*self.horiz_look_sensitivity])
                if self.vert_look_sensitivity != props.vert_look_sensitivity:
                    self.vert_look_sensitivity = props.vert_look_sensitivity
                    self.vert_look_range = Range(src_pts=[-0.15, 0.15], target_pts=[-0.3*self.vert_look_sensitivity, 0.3*self.vert_look_sensitivity])
                if self.vert_mouth_opening_sensitivity != props.vert_mouth_opening_sensitivity:
                    self.vert_mouth_opening_sensitivity = props.vert_mouth_opening_sensitivity
                    self.vert_mouth_opening_range = Range(src_pts=[0, 0.1], target_pts=[0, -0.04*self.vert_mouth_opening_sensitivity])
                if self.mouth_width_sensitivity != props.mouth_width_sensitivity:
                    self.mouth_width_sensitivity = props.mouth_width_sensitivity
                    self.mouth_width_range = Range(src_pts=[-1.5, 1], target_pts=[0.025*self.mouth_width_sensitivity, -0.006*self.mouth_width_sensitivity])
                smile_sensitivity_changed = False
                if self.smile_sensitivity != props.smile_sensitivity:
                    self.smile_sensitivity = props.smile_sensitivity
                    smile_sensitivity_changed = True
                if self.antismile_sensitivity != props.antismile_sensitivity:
                    self.antismile_sensitivity = props.antismile_sensitivity
                    smile_sensitivity_changed = True
                if smile_sensitivity_changed:
                    self.smile_range = Range(src_pts=[-0.6, 0, 0.4], target_pts=[self.smile_sensitivity*45*pi/180, 0, -self.antismile_sensitivity*25*pi/180], dynamic_min_max=False, limited=True)



                timestamp=monotonic()
                face = faces[0]
                bones = bpy.data.objects["RIG-Vincent"].pose.bones

                # Head rotation
                rot_mat = Matrix(face.pose_transform_mat[:3,:3])
                rot_quat = rot_mat.to_quaternion()
                if self.init_head_rot_quat is None:
                    self.init_head_rot_quat = rot_quat
                rot_euler = self.init_head_rot_quat.rotation_difference(rot_quat).to_euler("ZXY")
                rot_euler = self.head_rot_filter.apply(np.array(rot_euler), timestamp=timestamp)
                bones["head_fk"].rotation_euler = rot_euler

                # Head translation. To change the head location, we move the torso !
                if self.tracker.xyz and not np.isnan(face.xyz[0]):
                    if self.init_xyz is None:
                        self.init_xyz = np.array(face.xyz)
                    else:
                        torso_delta = self.torso_filter.apply((np.array(face.xyz) - self.init_xyz)/1000,timestamp) 
                        bones["master_torso"].location[0] = -torso_delta[0]
                        bones["master_torso"].location[1] = -torso_delta[2]
                        bones["master_torso"].location[2] = torso_delta[1]
                        # if len(hands) > 0:
                        #     # Get the world coordinates of the bpy.data.objects["RIG-Vincent"].pose.bones["brow_ctrl_L"].tail
                        #     # which is close to the face reference
                        #     face_ref_loc = np.array(bones["brow_ctrl_L"].tail)
                        # hand_visible_in_current_frame = {"left":False, "right":False}
                        # for hand in hands:
                        #     hand_visible_in_current_frame[hand.label] = True
                        #     hc_name = self.hand_ctrl_name[hand.label]
                        #     # hand_rel_loc = (np.array(hand.xyz) - np.array(face.xyz)).dot([[1,0,0],[0,0,1],[0,1,0]]) / 1000 
                        #     hand_rel_loc = np.array([[1,0,0],[0,0,1],[0,1,0]]).dot(np.array(hand.xyz) - np.array(face.xyz)) / 1000 
                        #     if not self.hand_visible_in_prev_frame[hand.label]:
                        #         bpy.data.objects[self.hand_ik_ctrl[hand.label]].constraints["Copy Location"].target = bpy.data.objects[self.hand_ctrl_name[hand.label]]
                        #         # bones[self.hand_ik_ctrl[hand.label]].constraints["Copy Location"].target = 
                        #         self.hand_visible_in_prev_frame[hand.label] = True
                        #         print(hand.label, "UP")

                        #     hand_loc = self.hand_filter[hand.label].apply(face_ref_loc + hand_rel_loc, timestamp)

                            
                        #     # print(hc_name, "face_ref_loc", face_ref_loc, "hand_rel_loc", hand_rel_loc, "hand_loc", hand_loc)
                        #     bpy.data.objects[hc_name].location[0] = hand_loc[0]
                        #     bpy.data.objects[hc_name].location[1] = hand_loc[1]
                        #     bpy.data.objects[hc_name].location[2] = hand_loc[2]

                        # for label in ["left", "right"]:
                        #     if not hand_visible_in_current_frame[label] and self.hand_visible_in_prev_frame[label]:
                        #         bpy.data.objects[self.hand_ik_ctrl[label]].constraints["Copy Location"].target = None
                        #         self.hand_visible_in_prev_frame[label] = False
                        #         print(label, "DOWN")


                if self.tracker.with_attention and props.looking_direction == "TRACKED":
                    # Horizontal looking direction
                    # left_iris_lm = 468
                    # right_eye_right_corner = 33
                    # right_eye_left_corner = 133
                    # left_iris_lm = 473
                    # left_eye_right_corner = 362
                    # left_eye_left_corner = 263
                    horiz_right_look = np.linalg.norm(face.metric_landmarks[468]-face.metric_landmarks[33]) / np.linalg.norm(face.metric_landmarks[133]-face.metric_landmarks[33])
                    horiz_left_look = np.linalg.norm(face.metric_landmarks[473]-face.metric_landmarks[362]) / np.linalg.norm(face.metric_landmarks[263]-face.metric_landmarks[362])
                    horiz_look = (horiz_right_look + horiz_left_look) / 2
                    # Vertical looking direction
                    vert_right_look = face.metric_landmarks[468][1]-(face.metric_landmarks[33][1] + face.metric_landmarks[133][1])/2
                    vert_left_look = face.metric_landmarks[473][1]-(face.metric_landmarks[362][1] + face.metric_landmarks[263][1])/2
                    vert_look = (vert_right_look + vert_left_look) / 2
                    if self.set_neutral_position_frames_left >= 0:
                        self.snp_horiz_look_avg.append(horiz_look)
                        self.snp_vert_look_avg.append(vert_look)
                        self.init_horiz_look = np.mean(self.snp_horiz_look_avg)
                        self.init_vert_look = np.mean(self.snp_vert_look_avg)

                    delta_horiz_look = horiz_look - self.init_horiz_look
                    target_horiz_look = self.horiz_look_range.target(delta_horiz_look)
                    # print("horiz_look", delta_horiz_look, target_horiz_look)
                    bones["look"].location[0] = self.horiz_look_filter.apply(target_horiz_look, timestamp)

                    delta_vert_look = vert_look - self.init_vert_look
                    target_vert_look = self.vert_look_range.target(delta_vert_look)
                    # print("vert_look", delta_vert_look, target_vert_look)
                    bones["look"].location[2] = self.vert_look_filter.apply(target_vert_look, timestamp)

                # Eye brows vertical movements
                brow_L = np.linalg.norm(face.norm_landmarks[334] - face.norm_landmarks[333]) / np.linalg.norm(face.norm_landmarks[330] - face.norm_landmarks[333])
                brow_R = np.linalg.norm(face.norm_landmarks[105] - face.norm_landmarks[104]) / np.linalg.norm(face.norm_landmarks[101] - face.norm_landmarks[104])
                # print("b_L:", brow_L, "b_R:", brow_R)
                brow_LR = self.brows_filter.apply(np.array([brow_L, brow_R]), timestamp=timestamp)
                bones["brow_ctrl_L"].location[2] = self.brow_range.target(brow_LR[0])
                bones["brow_ctrl_R"].location[2] = self.brow_range.target(brow_LR[1])

                # Hand can be used to control:
                # - Eye brow inclination driven by hand rotation,
                if self.nb_hands > 0:
                    if len(hands) > 0:
                        hand = hands[0]
                        # Control eyebrows inclination
                        hand_rot = hand.rotation
                        target_brow_rot = self.brow_rotation_range.target(hand_rot)
                        bones["brow_ctrl_L"].rotation_euler[1] = -target_brow_rot
                        bones["brow_ctrl_R"].rotation_euler[1] = target_brow_rot
                        self.hand_visible_in_prev_frame = True
                    else:
                        if self.hand_visible_in_prev_frame:
                            # Control eyebrows inclination
                            bones["brow_ctrl_L"].rotation_euler[1] = 0
                            bones["brow_ctrl_R"].rotation_euler[1] = 0
                            self.hand_visible_in_prev_frame = False


                # Eyelids - eye opening
                eye_open_R = np.linalg.norm(face.metric_landmarks[159] - face.metric_landmarks[145])
                eye_open_L = np.linalg.norm(face.metric_landmarks[386] - face.metric_landmarks[374])
                eyes_open = (eye_open_R + eye_open_L) / 2.0 # looks weird if both eyes aren't the same...
                # print("eyes_open", eyes_open)
                eyes_open = self.eye_opening_filter.apply(eyes_open, timestamp)
                target_eyes_open = self.eye_opening_range.target(eyes_open)
                bones["eyelid_up_ctrl_R"].location[2] =   target_eyes_open 
                bones["eyelid_low_ctrl_R"].location[2] =  -target_eyes_open 
                bones["eyelid_up_ctrl_L"].location[2] =   target_eyes_open
                bones["eyelid_low_ctrl_L"].location[2] =  -target_eyes_open
                

                # Mouth opening
                vert_mouth_opening = np.linalg.norm(face.norm_landmarks[13][1] - face.norm_landmarks[14][1])
                # print("mouth open", vert_mouth_opening)
                vert_mouth_opening = self.vert_mouth_opening_filter.apply(vert_mouth_opening, timestamp=timestamp)
                target_vert_mouth_opening = self.vert_mouth_opening_range.target(vert_mouth_opening)
                bones["mouth_ctrl"].location[2] = target_vert_mouth_opening

                mouth_width = np.linalg.norm(face.metric_landmarks[78]- face.metric_landmarks[308])
                if self.init_mouth_width is None:
                    self.init_mouth_width = mouth_width
                delta_mouth_width = mouth_width - self.init_mouth_width
                # print("delta mouth width", delta_mouth_width)
                delta_mouth_width = self.mouth_width_filter.apply(delta_mouth_width, timestamp=timestamp)
                target_mouth_width = self.mouth_width_range.target(delta_mouth_width)
                bones["mouth_ctrl"].location[0] = target_mouth_width

                # Smiling
                # If the mouth is open, no smiling
                smile = (face.metric_landmarks[13][1] + face.metric_landmarks[14][1] - face.metric_landmarks[61][1] - face.metric_landmarks[292][1])  # > 0 when smiling
                if self.init_smile is None:
                    self.init_smile = smile
                if face.metric_landmarks[13][1] - face.metric_landmarks[14][1] > 0.5:
                    print(face.metric_landmarks[13][1] - face.metric_landmarks[14][1])
                    smile = self.init_smile
                delta_smile = self.smile_filter.apply(smile-self.init_smile, timestamp)
                target_smile = self.smile_range.target(delta_smile)
                # print("smile", delta_smile, target_smile)
                bones["mouth_ctrl"].rotation_euler[1] = target_smile

                if self.set_neutral_position_frames_left >= 0:
                    if self.set_neutral_position_frames_left == 0:
                        self.set_neutral_position()
                        bpy.data.objects['RedCross'].hide_set(True)
                    self.set_neutral_position_frames_left -= 1

            # Draw face and hands on the OpenCV window
            frame = self.renderer.draw(frame, faces, hands)
            key = self.renderer.waitKey(delay=1)
            if key == 27:
                self.stop = True
            elif key == ord("n"):
                # Set neutral position
                bpy.data.objects['RedCross'].hide_set(False)
                self.set_neutral_position_frames_left = 20
                self.snp_horiz_look_avg = []
                self.snp_vert_look_avg = []
                self.set_neutral_position()
        return {'PASS_THROUGH'}
    
    def clear_pose(self):
        rig = bpy.data.objects["RIG-Vincent"]
        for pb in rig.pose.bones:
            pb.matrix_basis.identity()

    def toggle_look(self):

        self.fixed_look = not self.fixed_look
        print("TOGGLE LOOK fixed:", self.fixed_look)
        constraints = bpy.data.objects['RIG-Vincent'].pose.bones['look_mstr'].constraints
        for c in constraints:
            c.enabled = not self.fixed_look
        if self.fixed_look:
            bpy.data.objects["RIG-Vincent"].pose.bones['look'].matrix_basis = Matrix()



    def set_neutral_position(self):
        # bpy.data.objects["RIG-Vincent"].pose.bones['look'].matrix_basis = Matrix()
        print("SET NEUTRAL POSITION")
        self.clear_pose()
        
        self.init_head_rot_quat = None
        self.init_xyz = None
        self.init_mouth_width = None
        self.init_smile = None
        self.horiz_look_filter.reset()
        self.vert_look_filter.reset()

    def execute(self, context):
        self.fixed_look = False
        # Save and change some settings 
        # bpy.context.space_data.overlay.show_overlays = False
        self.show_bones = bpy.context.space_data.overlay.show_bones
        bpy.context.space_data.overlay.show_bones = False
        self.show_extras = bpy.context.space_data.overlay.show_extras
        bpy.context.space_data.overlay.show_extras = False
        self.show_floor = bpy.context.space_data.overlay.show_floor
        bpy.context.space_data.overlay.show_floor = False


        props = context.scene.dai_props
        if not props.is_tracker_running:
            props.is_tracker_running = True
            
            self.tracker = HandFaceTracker(
                with_attention= props.facemesh_with_attention,
                nb_hands=int(props.depthai_max_hands),
                use_gesture=int(props.depthai_max_hands) > 0,
                use_face_pose=True,
                xyz = props.depthai_xyz,
                hlm_score_thresh=0.9,
                focus=None if props.depthai_autofocus else props.depthai_focus_value, 
                double_face=False
            )
            self.renderer = HandFaceRenderer(self.tracker)
            self.renderer.show_metric_landmarks = False
            wm = context.window_manager
            self._timer = wm.event_timer_add(0.1, window=context.window)
            self.status = "RUNNING"
            wm.modal_handler_add(self)
            return {'RUNNING_MODAL'}


    def cancel(self, context):
        context.scene.dai_props.is_tracker_running = False
        bpy.context.space_data.overlay.show_bones = self.show_bones
        bpy.context.space_data.overlay.show_extras = self.show_extras
        bpy.context.space_data.overlay.show_floor = self.show_floor

        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        self.renderer.exit()
        self.tracker.exit()
        tag_redraw(context)
        return {'FINISHED'}

_classes = [DepthAIPuppetProperties, DepthAIPuppetOperator, DepthAIPuppetPanel]

# Register and add to the "object" menu (required to also use F3 search "Simple Object Operator" for quick access)
def register():
    for cls in _classes:
        print(f"Registering {cls}")
        bpy.utils.register_class(cls)
    bpy.types.Scene.dai_props = PointerProperty(type=DepthAIPuppetProperties)


def unregister():
    for cls in _classes:
        bpy.utils.unregister_class(cls)
        bpy.types.Scene.dai_props = None    



def main():
    """
    Is called by blender
    """
    register()
