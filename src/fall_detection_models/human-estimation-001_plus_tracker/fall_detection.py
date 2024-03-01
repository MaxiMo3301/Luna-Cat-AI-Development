import numpy
import numpy as np
import math
import cv2
import copy

keypoints_dict = {'Nose': 0, 'Neck': 1, 'R-Sho': 2, 'R-Elb': 3, 'R-Wr': 4, 'L-Sho': 5, 'L-Elb': 6, 'L-Wr': 7,
                  'R-Hip': 8, 'R-Knee': 9, 'R-Ank': 10,
                  'L-Hip': 11, 'L-Knee': 12, 'L-Ank': 13, 'R-Eye': 14, 'L-Eye': 15, 'R-Ear': 16, 'L-Ear': 17}


class hip_position_and_timestamp:
    """y -coordinate of mid-hip and a timestamp"""

    def __init__(self, y: list = [-1 for i in range(30)], timestamp: float = -1) -> None:
        """
        :param y: a list of y-coordinates of hip for all tracked people
        :param timestamp: the time when hip coordinates are detected
        :return None
        """
        self.hip_y = [_ for _ in y]  # id of a person maps to his/her hip y-coordinate
        self.timestamp = timestamp


class hip_window:
    """
    used to record x and y coordinate of shoulders of people across frames
    Attributes:
        list: a list of hip_position_and_timestamp
    """
    def __init__(self, size: int):
        self.list = [hip_position_and_timestamp() for _ in range(size)]

    def update(self, keypoint_for_detection: list, timestamp: float, pose2bbox_dict: dict) -> None:
        """
        pop leftmost entry of self.list and append new entry
        :param keypoint_for_detection: a list of size 18 with each entry contains (x, y)
                that indicates the positions of a keypoint on the current frame
        :param timestamp: current time in second
        :param pose2bbox_dict: a dictionary mapping poses to tracked people
        :return: None
        """
        if not keypoint_for_detection:
            return

        new_list = copy.copy(self.list[1:len(self.list)])
        tmp = [-1 for _ in range(30)]  # dummy value
        for index, person_keypoint in enumerate(keypoint_for_detection):
            l_hip = person_keypoint[keypoints_dict['L-Sho']]
            r_hip = person_keypoint[keypoints_dict['R-Sho']]
            neck = person_keypoint[keypoints_dict["Neck"]]
            # if l_hip != -1 and r_hip != -1:
            if neck != -1:
                # hip_y = (l_hip[1] + r_hip[1]) / 2
                # hip_x = (l_hip[0] + r_hip[0]) / 2
                tracker_id = pose2bbox_dict.get(index)
                if tracker_id is not None:
                    # tmp[tracker_id] = (hip_x, hip_y)
                    tmp[tracker_id] = (neck[0], neck[1])

        new_list.append(hip_position_and_timestamp(tmp, timestamp))
        self.list = new_list


def falling_alarm(frame: numpy.ndarray, bbox: tuple) -> None:
    """ Draw bounding box of a person and show alarm to indicate someone is falling"""
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(75, 0, 130),
                  thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(frame, 'Person Fell down', (11, 100), 0, 1, [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)


def fall_detect(body: list) -> bool:
    """
    :param body: a list of size 18 with each entry contains (x, y) that indicates the positions of a keypoint on the frame
    :return: True if the person falls otherwise False
    """

    def is_present(body, lm_id):
        if body[lm_id] != -1:
            return True
        else:
            return False

    # -1 means the particular keypoint is not detected
    x_coor = [pt[0] for index, pt in enumerate(body)
              if index != keypoints_dict['L-Elb'] and
              index != keypoints_dict['L-Wr'] and
              index != keypoints_dict['R-Elb'] and
              index != keypoints_dict['R-Wr'] and
              pt != -1]
    y_coor = [pt[1] for index, pt in enumerate(body)
              if index != keypoints_dict['L-Elb'] and
              index != keypoints_dict['L-Wr'] and
              index != keypoints_dict['R-Elb'] and
              index != keypoints_dict['R-Wr'] and
              pt != -1]

    scalar = .9  # the lower the value, the easier the conditions trigger

    if is_present(body, keypoints_dict["L-Sho"]) \
            and is_present(body, keypoints_dict["L-Hip"]) \
            and is_present(body, keypoints_dict["L-Ank"]):

        left_shoulder_y = body[keypoints_dict['L-Sho']][1]
        left_shoulder_x = body[keypoints_dict['L-Sho']][0]
        left_body_y = body[keypoints_dict['L-Hip']][1]
        left_body_x = body[keypoints_dict['L-Hip']][0]
        left_foot_y = body[keypoints_dict['L-Ank']][1]
        len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))

        condition_1 = left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y \
                      - (len_factor / scalar) and left_shoulder_y > left_body_y - (len_factor / scalar)

    else:
        condition_1 = False

    if is_present(body, keypoints_dict["R-Sho"]) \
            and is_present(body, keypoints_dict["R-Hip"]) \
            and is_present(body, keypoints_dict["R-Ank"]):
        right_shoulder_y = body[keypoints_dict['R-Sho']][1]
        right_shoulder_x = body[keypoints_dict['R-Sho']][0]
        right_body_y = body[keypoints_dict['R-Hip']][1]
        right_body_x = body[keypoints_dict['R-Hip']][0]
        right_foot_y = body[keypoints_dict['R-Ank']][1]
        len_factor = math.sqrt(((right_shoulder_y - right_body_y) ** 2 + (right_shoulder_x - right_body_x) ** 2))

        condition_2 = right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y \
                      - (len_factor / scalar) and right_shoulder_y > right_body_y - (len_factor / scalar)
    else:
        condition_2 = False

    if x_coor and y_coor:
        xmin, ymin = min(x_coor), min(y_coor)
        xmax, ymax = max(x_coor), max(y_coor)
    else:
        return False, None
    dx = int(xmax) - int(xmin)
    dy = int(ymax) - int(ymin)
    difference = dy - dx
    head_detected = False
    lower_body_detected = False
    # should detect head and lower body for fall detection
    if is_present(body, keypoints_dict["Nose"]) \
            or is_present(body, keypoints_dict["R-Eye"]) \
            or is_present(body, keypoints_dict["L-Eye"]) \
            or is_present(body, keypoints_dict["R-Ear"]) \
            or is_present(body, keypoints_dict["L-Ear"]):
        head_detected = True

    if is_present(body, keypoints_dict["R-Ank"]) \
            or is_present(body, keypoints_dict["L-Ank"]) \
            or is_present(body, keypoints_dict["R-Knee"]) \
            or is_present(body, keypoints_dict["L-Knee"]):
        lower_body_detected = True

    # for debugging
    lower_body_detected = True

    if not head_detected or not lower_body_detected:
        condition_3 = False
    else:
        condition_3 = difference < 0

    if condition_1 or condition_2 or condition_3:
        return True
    return False


def get_velocity(hip_position_window: "hip_window", interval: int, tracker_ids: list) -> dict:
    """
    To calculate the velocity of a person by using his/her hip position
    :param hip_position_window: a window containing timestamp and hip position across frames
    :param interval: an integer to define displacement of a person across frames
    :param tracker_ids: a list of identity numbers of all detected people
    :return: a dictionary of tracker_id: velocity
    """
    velocity_dict = {}
    time_elasped = hip_position_window.list[interval - 1].timestamp - hip_position_window.list[0].timestamp

    def find_displacement(xy1, xy2):
        """ (x1, y1) is a point and (x2, y2) is a point"""
        return math.sqrt((xy2[0] - xy1[0]) ** 2 + (xy2[1] - xy1[1]) ** 2)

    for tracker_id in tracker_ids:

        hip_position_timestamp_2 = hip_position_window.list[interval - 1].hip_y[tracker_id]
        hip_position_timestamp_1 = hip_position_window.list[0].hip_y[tracker_id]

        velocity = None
        if hip_position_timestamp_1 != -1 and \
                hip_position_timestamp_2 != -1:
            #displacement = find_displacement(hip_position_timestamp_1, hip_position_timestamp_2)
            displacement = hip_position_timestamp_2[1] - hip_position_timestamp_1[1] # only calculate the difference between y-coordinate
            velocity = displacement / time_elasped
        velocity_dict[tracker_id] = velocity

    return velocity_dict
