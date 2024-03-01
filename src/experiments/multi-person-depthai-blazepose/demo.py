#!/usr/bin/env python3

from BlazeposeRenderer import BlazeposeRenderer
import argparse
import cv2
from fall_detection import fall_detect_3d, fall_detect, person_bbox, face_bbox
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")                 
parser_tracker.add_argument('-i', '--input', type=str, default="rgb", 
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default=%(default)s)")
parser_tracker.add_argument("--pd_m", type=str,
                    help="Path to an .blob file for pose detection model")
parser_tracker.add_argument("--lm_m", type=str,
                    help="Landmark model ('full' or 'lite' or 'heavy') or path to an .blob file")
parser_tracker.add_argument('-xyz', '--xyz', action="store_true", 
                    help="Get (x,y,z) coords of reference body keypoint in camera coord system (only for compatible devices)")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape before feeding pose detection model")
parser_tracker.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument('--internal_frame_height', type=int, default=640,                                                                                    
                    help="Internal color camera frame height in pixels (default=%(default)i)")                    
parser_tracker.add_argument('-s', '--stats', action="store_true", 
                    help="Print some statistics at exit")
parser_tracker.add_argument('-t', '--trace', action="store_true", 
                    help="Print some debug messages")
parser_tracker.add_argument('--force_detection', action="store_true", 
                    help="Force person detection on every frame (never use landmarks from previous frame to determine ROI)")
parser_tracker.add_argument('--multi_detection', action="store_true",
                        help="Force multiple person detection (at your own risk, the original Mediapipe implementation is designed for one person tracking)")

parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-3', '--show_3d', choices=[None, "image", "world", "mixed"], default=None,
                    help="Display skeleton in 3d in a separate window. See README for description.")
parser_renderer.add_argument("-o","--output",
                    help="Path to output video file")
parser_renderer.add_argument("-b","--black", action="store_true", help="only skeleton is shown")
 

args = parser.parse_args()

if args.edge:
    from BlazeposeDepthaiEdge import BlazeposeDepthai
else:
    from BlazeposeDepthai import BlazeposeDepthai
tracker = BlazeposeDepthai(input_src=args.input, 
            pd_model=args.pd_m,
            lm_model=args.lm_m,
            smoothing=not args.no_smoothing,   
            xyz=args.xyz,            
            crop=args.crop,
            internal_fps=args.internal_fps,
            internal_frame_height=args.internal_frame_height,
            force_detection=args.force_detection,
            stats=True,
            trace=args.trace,
            multi_detection=args.multi_detection)

renderer = BlazeposeRenderer(
                tracker, 
                show_3d=args.show_3d, 
                output=args.output)


def falling_alarm(frame, bbox):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255),
                  thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(frame, 'Person Fell down', (11, 100), 0, 1, [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)


while True:
    # Run blazepose on next frame
    frame, bodies = tracker.next_frame()
    

    if frame is None: break
    # blacksreen
    black = args.black
    if black:
        frame = np.zeros((700, 1000, 3), np.uint8)

    # Draw 2d skeleton
    print("bodies", len((bodies)))
    if len((bodies)) >= 1:
        for body in bodies:
            frame = renderer.draw(frame, body)
    else:
        frame = renderer.draw(frame, None)

    #fall detect
    alert = False
    

    
    #blur person
    """
    if body is not None:
        #bbox= person_bbox(tracker, body)
        bbox= face_bbox(tracker, body)
        if bbox and not black:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255),
                        thickness=5, lineType=cv2.LINE_AA)
            
            
            #blur the person
            topLeft = ( int(bbox[0]), int(bbox[1]) )
            bottomRight = ( int(bbox[2]), int(bbox[3]) )
            x, y = topLeft[0], topLeft[1]
            w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
            ROI = frame[y:y+h, x:x+w]
            if int(bbox[0]) != int(bbox[2]) or int(bbox[1]) != int(bbox[3]):
                blur = cv2.GaussianBlur(ROI, (101,101), 0) 
                frame[y:y+h, x:x+w] = blur
    """
    #fall detect 3d
    args.show_3d = True # for debug
    if not args.show_3d:
        print("not in 3d mode")
    else:
        for body in bodies:
            alert = fall_detect_3d(args.show_3d, body)
            if alert:
                cv2.putText(frame, 'Person Fell down', (11, 100), 0, 1, [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)
                alert = False

    # fall detect 2d

    #if body:
    #    alert, bbox = fall_detect(tracker, body)

    #if alert:
    #    falling_alarm(frame, bbox)
        

    alert = False

    key = renderer.waitKey(delay=1)
    #key = cv2.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
tracker.exit()