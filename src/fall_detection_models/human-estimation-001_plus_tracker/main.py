import threading
import depthai as dai
import cv2
import time
import argparse
from FPS import FPS
from fall_detection import fall_detect, hip_window, get_velocity
from utils import show, get_person_keypoint, matching_poses_with_bboxes, draw_bboxes
from gen2_pose import DepthaiPose
import marshal
from collections import deque

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

nn_input_size = (456, 256)
mobilenet_input_size = (300, 300)


# for storing fall detection result by pose only for last 3 frames
# except for the first three frames, whether [id] person falls is indicated in the rightmost entry of the queue
# by indexing the result_dictionary by [id]
result_dictionary = dict([(i, deque([False, False, False], maxlen=3)) for i in range(30)])  # id: [ bool, bool, boll ]

if __name__ == "__main__":
    depthai_pose = DepthaiPose()
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--black', action="store_true", help="only show skeleton in black mode")
    parser.add_argument("-v", "--velocity_threshold", type=float, default=150, help="a threshold to determine falling conditions")

    args = parser.parse_args()
    velocity_threshold = args.velocity_threshold

    #  for fall detection
    interval = 2
    hips_coordinates_window = hip_window(interval)

    # extend the alarm
    has_previous_alarm = False
    alarm_threshold = 5
    alarm_count = 0

    fps_calculate = FPS(mean_nb_frames=10)
    pipeline = depthai_pose.create_pipeline()
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        preview = device.getOutputQueue("preview", 4, False)
        manager_out = device.getOutputQueue("manager_out", 4, False)

        frame = None
        max_velocity = 0
        while True:
            fps_calculate.update()

            imgFrame = preview.get()
            frame = imgFrame.getCvFrame()

            pose_result = marshal.loads(manager_out.get().getData())
            person_state = pose_result["person_state"]
            tracker_id = []
            # decode pose result
            th_decode = threading.Thread(target=depthai_pose.decode_pose_result,
                                         args=(pose_result["heatmaps"], pose_result["pafs"]))
            th_draw_bbox = threading.Thread(target=draw_bboxes, args=(frame, person_state))
            th_decode.start()
            th_draw_bbox.start()

            th_decode.join()
            keypoints_for_detection = get_person_keypoint(frame, depthai_pose.PersonwiseKeypoints,
                                                          depthai_pose.keypoints_list, depthai_pose.keypoints)
            pose_to_bbox_map = matching_poses_with_bboxes(tracklets=person_state, poses=keypoints_for_detection)

            show(frame, depthai_pose.keypoints_list,
                 depthai_pose.keypoints, depthai_pose.PersonwiseKeypoints)

            # fall detection
            for person_id, state in person_state.items():
                tracker_id.append(person_id)

            current_time = time.perf_counter()
            hips_coordinates_window.update(keypoints_for_detection, current_time, pose_to_bbox_map)
            dictionary_of_velocity = get_velocity(hip_position_window=hips_coordinates_window,
                                                  interval=interval, tracker_ids=tracker_id)

            for pose_id, bbox_id in pose_to_bbox_map.items():
                alarm = fall_detect(keypoints_for_detection[pose_id])

                # storing the alarm result
                deque_result = result_dictionary[bbox_id]
                deque_result.popleft()
                deque_result.append(alarm)
                has_fallen = any(deque_result)

                velocity = dictionary_of_velocity.get(bbox_id)
                if velocity is None:
                    continue

                #  for debugging
                if velocity > max_velocity:
                    max_velocity = velocity
                print(velocity)
                print(f"Person {bbox_id} fall status:", alarm)

                if velocity > velocity_threshold and has_fallen:
                    has_previous_alarm = True
                # if velocity is too high, consider the person fallen
                elif velocity > 400:
                    has_previous_alarm = True

            # show alarm for [alarm_threshold] frames
            if has_previous_alarm:
                alarm_count += 1
                if alarm_count > alarm_threshold:
                    has_previous_alarm = False
                    alarm_count = 0
                else:
                    cv2.putText(frame, "Person has fallen", (11, 100), 0, 1, [12, 120, 255], thickness=3,
                                lineType=cv2.LINE_AA)
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) == ord('q'):
                print(f"max_velocity: {max_velocity}")
                break

    print('Devices closed')
    print(f"FPS : {fps_calculate.get_global():.1f} f/s (# frames = {fps_calculate.nbf})")
