#!/usr/bin/env python3

from HandFaceTracker import HandFaceTracker
from HandFaceRenderer import HandFaceRenderer
import argparse

parser = argparse.ArgumentParser()
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("-a", "--with_attention", action="store_true",
                    help="Use face landmark with attention model")
parser_tracker.add_argument('-p', "--use_face_pose", action="store_true", 
                    help="Calculate the face pose tranformation matrix and metric landmarks")
parser_tracker.add_argument('-2', "--double_face", action="store_true", 
                    help="EXPERIMENTAL. Run a 2nd occurence of the face landmark Neural Network to improve fps. Hand tracking is disabled.")
parser_tracker.add_argument('-n', '--nb_hands', type=int, choices=[0,1,2], default=2, 
                    help="Number of hands tracked (default=%(default)i)")                    
parser_tracker.add_argument('-xyz', "--xyz", action="store_true", 
                    help="Enable spatial location measure of hands and face")
parser_tracker.add_argument('-g', '--gesture', action="store_true", 
                    help="Enable gesture recognition")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels")   
parser_tracker.add_argument('-t', '--trace', type=int, nargs="?", const=1, default=0, 
                    help="Print some debug infos. The type of info depends on the optional argument.")                
parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-o', '--output', 
                    help="Path to output video file")
args = parser.parse_args()
dargs = vars(args)
tracker_args = {a:dargs[a] for a in ['internal_fps', 'internal_frame_height'] if dargs[a] is not None}

tracker = HandFaceTracker(
        input_src=args.input, 
        double_face=args.double_face,
        use_face_pose=args.use_face_pose,
        use_gesture=args.gesture,
        xyz=args.xyz,
        with_attention=args.with_attention,
        nb_hands=args.nb_hands,
        trace=args.trace,
        **tracker_args
        )

renderer = HandFaceRenderer(
        tracker=tracker,
        output=args.output)

while True:
    frame, faces, hands = tracker.next_frame()
    if frame is None: break
    # Draw face and hands
    frame = renderer.draw(frame, faces, hands)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
tracker.exit()
