#!/usr/bin/env python3

from BlazeposeRenderer import BlazeposeRenderer
import argparse
import cv2
from fall_detection import fall_detect_3d, \
         face_bbox, read_landmarks_pt, fall_detected_by_velocity
from fall_detection import hip_window
import numpy as np
import time

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
parser_tracker.add_argument("--lm_score_thresh", type=float, default=0.63,
                            help ="confidence score to determine whether landmarks prediction is reliable (a float between 0 and 1)")
parser_tracker.add_argument("--pd_score_thresh", type=float, default=0.5,
                            help ="confidence score to determine whether a detection is reliable (a float between 0 and 1)")




parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-3', '--show_3d', choices=[None, "image", "world", "mixed"], default=None,
                             help="Display skeleton in 3d in a separate window. See README for description.")
parser_renderer.add_argument("-o", "--output",
                             help="Path to output video file")
parser_renderer.add_argument("-b", "--black", action="store_true", help="only skeleton is shown")

parser_fall_detect = parser.add_argument_group("Fall detection")
parser_fall_detect.add_argument("-v", "--velocity_threshold", default=0.1, type=float, help="a condition to determine whether one falls or not")
parser_fall_detect.add_argument("-dm", "--detection_mode", choices=["image", "world", "mixed"], default="mixed")
parser_fall_detect.add_argument("-ss", "--show_stat", action="store_true",
                                help="enable to show statistic of body angles against the floor and person's velocity")
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
                           pd_score_thresh=args.pd_score_thresh,
                           lm_score_thresh=args.lm_score_thresh
                           )

renderer = BlazeposeRenderer(
    tracker,
    show_3d=args.show_3d,
    output=args.output)

def blur_face(frame, tracker, body, black):
    bbox = face_bbox(tracker, body)
    if bbox and not black:
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255),
                      thickness=5, lineType=cv2.LINE_AA)

        # blur the person
        topLeft = (int(bbox[0]), int(bbox[1]))
        bottomRight = (int(bbox[2]), int(bbox[3]))
        x, y = topLeft[0], topLeft[1]
        w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
        ROI = frame[y:y + h, x:x + w]
        if int(bbox[0]) != int(bbox[2]) and int(bbox[1]) != int(bbox[3]):  # make sure region is non empty
            blur = cv2.GaussianBlur(ROI, (101, 101), 0)
            frame[y:y + h, x:x + w] = blur


# for saving skeleton across frames
"""nb_frame = 0
frame_interval = 30
lm_array = np.zeros((frame_interval, 33, 3))
f = open("output.npy", "wb")
save_time = 0
"""

# for calculating velocity of a person
size = 5
hip_position_window = hip_window(size)

# in edge mode
v_threshold = args.velocity_threshold

# Once fall is detected, show the alarm for [alarm_count_threshold] frames
has_alarm_on_previous = False
alarm_count = 0
alarm_count_threshold = 10

while True:
    # Run blazepose on next frame
    frame, body = tracker.next_frame()

    #  for saving pose result
    """
    if not body:
        lm_array = np.zeros((frame_interval, 33, 3))  # frame x nb_keypoint x nb_coordinate
        nb_frame = 0
    else:
        #print(lm_array[nb_frame:nb_frame + 1, :, :].shape)
        #print("body", np.expand_dims(body.norm_landmarks[:33,:3], axis=0).shape)
        lm_array[nb_frame:nb_frame + 1, :, :] = np.expand_dims(body.norm_landmarks[:33,:3], axis=0)
        nb_frame += 1
        if nb_frame == frame_interval:
            save_time += 1
            np.save(f, lm_array)

            #  reset
            lm_array = np.zeros((frame_interval, 33, 3))
            nb_frame = 0
    """

    if frame is None: break
    # blacksreen
    black = args.black
    if black:
        frame = np.zeros((648, 1152, 3), np.uint8)

    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    # blur person
    if body is not None:
        blur_face(frame, tracker, body, black)

    # fall detect
    alert = False
    # fall detect 3d
    fall_detect_mode = args.detection_mode  # choices: image or world or mixed
    points = read_landmarks_pt(fall_detect_mode, body)
    if points is not None:
        alert, body_angle_with_respect_to_floor = fall_detect_3d(points)

        # put stat on the frame
        if args.show_stat:
            cv2.putText(frame, body_angle_with_respect_to_floor[0],
                        (11, 130), 0, 1, [255, 0, 100], thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(frame, body_angle_with_respect_to_floor[1],
                        (11, 165), 0, 1, [255, 0, 100], thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(frame, body_angle_with_respect_to_floor[2],
                        (11, 200), 0, 1, [255, 0, 100], thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(frame, body_angle_with_respect_to_floor[3],
                        (11, 240), 0, 1, [255, 0, 100], thickness=3, lineType=cv2.LINE_AA)

        # calculate velocity
        current_time = time.perf_counter()
        hip_position_window.update( [points], current_time )
        velocity = fall_detected_by_velocity(hip_position_window, size)

        if velocity is not None:
            if args.show_stat:
                cv2.putText(frame, f"velocity: {velocity:.2f}",
                            (11, 275), 0, 1, [255, 0, 100], thickness=3, lineType=cv2.LINE_AA)

            if alert and ( velocity > v_threshold ):
                has_alarm_on_previous = True

    # show alarm for [alarm_count_threshold] frames once fall is detected
    if has_alarm_on_previous:
        cv2.putText(frame, 'Person Fell down', (11, 100), 0, 1, [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)
        alarm_count += 1
        if alarm_count >= alarm_count_threshold:
            has_alarm_on_previous = False
            alarm_count = 0

    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break



# for saving skeleton
# print("save_time", save_time)
# f.close()

renderer.exit()
tracker.exit()
