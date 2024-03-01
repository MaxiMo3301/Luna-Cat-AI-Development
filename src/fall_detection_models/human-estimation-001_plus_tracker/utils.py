import cv2
import math
from math import gcd

import numpy
import numpy as np
nn_input_size = (456, 256)

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
keypoints_dict = {'Nose': 0 , 'Neck': 1, 'R-Sho': 2, 'R-Elb': 3, 'R-Wr': 4, 'L-Sho': 5, 'L-Elb': 6, 'L-Wr': 7, 'R-Hip': 8, 'R-Knee': 9, 'R-Ank': 10,
                    'L-Hip': 11, 'L-Knee': 12 , 'L-Ank': 13 , 'R-Eye': 14, 'L-Eye': 15, 'R-Ear': 16, 'L-Ear':17}
def face_bbox(body):
    def is_present(lm_id):
        return body[lm_id] != -1

    head_index = [0, 14, 15, 16, 17]
    head_pts = []
    sho_index = [2, 5]
    sho_pts = []
    for i, x_y in enumerate(body):
        if i in head_index and is_present(i):
            head_pts.append(x_y)
        if i in sho_index and is_present(i):
            sho_pts.append(x_y)
    if head_pts is not None and len(sho_pts) == 2:
        x_pts = [x_y[0] for x_y in head_pts]
        y_pts = [x_y[1] for x_y in head_pts]
        x_pts.append(sho_pts[0][0])
        x_pts.append(sho_pts[1][0])
        y_pts.append(sho_pts[0][1])
        y_pts.append(sho_pts[1][1])

        dist_between_sho = math.sqrt((sho_pts[0][0] - sho_pts[1][0]) ** 2 + (sho_pts[0][1] - sho_pts[1][1])**2 )
        xmin, ymin = min(x_pts), min( y_pts)
        xmax, ymax = max(x_pts), max( y_pts)
        ymin = ymin - dist_between_sho
        xmin = xmin - dist_between_sho
        xmax = xmax + dist_between_sho
        ymax = ymax + dist_between_sho

        #larger the width of the face bbox
        xmax = xmax
        xmin = xmin
        #check if out of bound
        # default cam image size: unknown
        x_max_size = 1792
        y_max_size = 1008
        xmin = 0 if int(xmin) < 0 else xmin
        xmax = 0 if int(xmax) < 0 else xmax
        xmin = x_max_size if int(xmin) > x_max_size else xmin
        xmax = x_max_size if int(xmax) > x_max_size else xmax

        ymin = 0 if int(ymin) < 0 else ymin
        ymax = 0 if int(ymax) < 0 else ymax
        ymin = y_max_size if int(ymin) > y_max_size else ymin
        ymax = y_max_size if int(ymax) > y_max_size else ymax

        return (xmin, ymin, xmax, ymax)

    return None


def blur_face(frame, body, black) -> list:
    bbox = face_bbox(body)
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
    return frame


def show(frame: numpy.ndarray, keypoints_list, detected_keypoints: list, personwiseKeypoints: list) -> None:
    """
    Show poses on frame
    :param frame: RGB frame
    :param keypoints_list: list of key points with ids
    :param detected_keypoints: list of detected key points, shape: (3 x arbitrary)
    :param personwiseKeypoints: list of key points index , shape: (19 x number of people)
    :return: None
    """

    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
        #scale_factor = frame.shape[0] / nn_input_size[1]
        #offset_w = int(frame.shape[1] - nn_input_size[0] * scale_factor) // 2
        scale_factor = frame.shape[0] / nn_input_size[1]
        offset_w = int(frame.shape[1] - nn_input_size[0] * scale_factor) // 2

        def scale(point):
            return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)

        for i in range(18):
            for j in range(len(detected_keypoints[i])):
                try:
                    cv2.circle(frame, scale(detected_keypoints[i][j][0:2]), 5, colors[i], -1, cv2.LINE_AA)
                except IndexError:
                    print("IndexError")
                    continue

        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                try:
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frame, scale((B[0], A[0])), scale((B[1], A[1])), colors[i], 3, cv2.LINE_AA)

                except IndexError:
                    print("IndexError")
                    continue


def get_person_keypoint(frame, personwiseKeypoints, keypoints_list, detected_keypoints) -> list:
    """
    get keypoints from personwiseKeypoints and store them in keypoint_for_detection
    all keypoints of each person is stored in an entry of keypoint_for_detection

    Returns list, shape: (number of person x 18)
    """
    if keypoints_list is None or \
            detected_keypoints is None or \
            personwiseKeypoints is None:
        return None
    scale_factor = frame.shape[0] / nn_input_size[1]
    offset_w = int(frame.shape[1] - nn_input_size[0] * scale_factor) // 2

    def scale(point):
        return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)

    keypoint_for_detection = [[-1 for j in range(18)] for i in range(len(personwiseKeypoints))]
    for i in range(18):
        for n in range(len(personwiseKeypoints)):
            try:
                index = personwiseKeypoints[n][np.array([1, i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                keypoint_for_detection[n][i] = scale((B[1], A[1]))
            except IndexError:
                continue

    return keypoint_for_detection




def find_isp_scale_params(size, is_height=True):
    """
    Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
    This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
    is_height : boolean that indicates if the value is the height or the width of the image
    Returns: valid size, (numerator, denominator)
    """
    # We want size >= 288
    if size < 288:
        size = 288

    # We are looking for the list on integers that are divisible by 16 and
    # that can be written like n/d where n <= 16 and d <= 63
    if is_height:
        reference = 1080
        other = 1920
    else:
        reference = 1920
        other = 1080
    size_candidates = {}
    for s in range(288, reference, 16):
        f = gcd(reference, s)
        n = s // f
        d = reference // f
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


def draw_bboxes(frame, person_state) -> None:
    """
    Draw bounding boxes on frame
    """
    for person_id, state in person_state.items():
        x1 = state["topLeft_x"]
        y1 = state["topLeft_y"]
        x2 = state["bottomRight_x"]
        y2 = state["bottomRight_y"]

        cv2.rectangle(frame, (x1, y1),
                      (x2, y2), (200, 0, 0), lineType=cv2.LINE_AA)
        cv2.putText(frame, f"person {person_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 233, 0), 2, cv2.LINE_AA)


def matching_poses_with_bboxes(tracklets, poses) -> dict:
    """
    :param tracklets: dictionary of id : (top_left_x, top_right_y, bottom_right_x, bottom_right_y, status)
    :param poses: list, shape: (number of person x 18)
    :return: a dictionary that maps ids of pose to ids of bbox
    """
    def is_pose_in_bbox(person_pose, bbox) -> bool:

        def is_between(value, x1, x2) -> bool:
            """integer comparison"""
            return ( x1 <= value <= x2 )

        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        neck_keypoints = person_pose[keypoints_dict["Neck"]]
        neck_in_bbox = False
        if neck_keypoints != -1:
            neck_in_bbox = is_between(neck_keypoints[0], x1, x2) \
                                        and is_between(neck_keypoints[1], y1, y2)


        R_Sho_keypoints = person_pose[keypoints_dict["R-Sho"]]
        L_Sho_keypoints = person_pose[keypoints_dict["L-Sho"]]
        R_Sho_in_bbox = False
        if R_Sho_keypoints != -1:
            R_Sho_in_bbox = is_between(R_Sho_keypoints[0], x1, x2) \
                           and is_between(R_Sho_keypoints[1], y1, y2)
        L_Sho_in_bbox = False
        if L_Sho_keypoints != -1:
            L_Sho_in_bbox = is_between(L_Sho_keypoints[0], x1, x2) \
                           and is_between(L_Sho_keypoints[1], y1, y2)

        R_Hip_keypoints = person_pose[keypoints_dict["R-Hip"]]
        L_Hip_keypoints = person_pose[keypoints_dict["L-Hip"]]
        R_Hip_in_bbox = False
        if R_Hip_keypoints != -1:
            R_Hip_in_bbox = is_between(R_Hip_keypoints[0], x1, x2) \
                            and is_between(R_Hip_keypoints[1], y1, y2)
        L_Hip_in_bbox = False
        if L_Hip_keypoints != -1:
            L_Hip_in_bbox = is_between(L_Hip_keypoints[0], x1, x2) \
                            and is_between(L_Hip_keypoints[1], y1, y2)

        if neck_in_bbox and (R_Sho_in_bbox or L_Sho_in_bbox) \
            and (R_Hip_in_bbox or L_Hip_in_bbox):
            return True


        return False

    mapping_dict = {}
    used_bboxes = []
    for i in range(len(poses)):
        for (id, state) in tracklets.items():
            if id in used_bboxes:
                continue
            in_bbox = is_pose_in_bbox(poses[i], ( state["topLeft_x"],
                                                  state["topLeft_y"],
                                                  state["bottomRight_x"],
                                                  state["bottomRight_y"] ) )
            if in_bbox:
                mapping_dict[i] = id
                used_bboxes.append(id)
                break

    return mapping_dict


