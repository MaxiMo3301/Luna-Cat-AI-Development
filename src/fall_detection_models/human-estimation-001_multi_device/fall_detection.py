
import numpy as np
import math
import cv2
import copy

# Define object index
# maximum 17 joint for this humand detection module

keypoints_dict = {'Nose': 0 , 'Neck': 1, 'R-Sho': 2, 'R-Elb': 3, 'R-Wr': 4, 'L-Sho': 5, 'L-Elb': 6, 'L-Wr': 7, 'R-Hip': 8, 'R-Knee': 9, 'R-Ank': 10,
                    'L-Hip': 11, 'L-Knee': 12 , 'L-Ank': 13 , 'R-Eye': 14, 'L-Eye': 15, 'R-Ear': 16, 'L-Ear':17}

#for fall detection
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


# get the angle between two joint 
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# When Falling = True -> Display Fall Detected Message
def falling_alarm(frame, bbox):

    # Get the coordinate of the human body joint detected
    x_min, y_min, x_max, y_max = bbox

    # Draw The Frame Around the Humand Body
    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(75, 0, 130),
                  thickness=5, lineType=cv2.LINE_AA)
    
    # Print The Fall Detected Message
    cv2.putText(frame, 'Person Fell down', (11, 100), 0, 1, [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)

# Define the Falling State
def fall_detect(body):
    """ Return boolean, bbox.
        for one person"""
    #print(body)
    def is_present(body, lm_id):
        if body[lm_id] != -1:
            return True
        else:
            return False

    # -1 means the particular keypoint is not detected
    # get the point of the left and right hands
    # define x point 
    x_coor = [pt[0] for index, pt in enumerate(body)
              if index != keypoints_dict['L-Elb'] and
              index != keypoints_dict['L-Wr'] and
              index != keypoints_dict['R-Elb'] and
              index != keypoints_dict['R-Wr'] and
              pt != -1]
    
    # define y point
    y_coor = [pt[1] for index, pt in enumerate(body)
              if index != keypoints_dict['L-Elb'] and
              index != keypoints_dict['L-Wr'] and
              index != keypoints_dict['R-Elb'] and
              index != keypoints_dict['R-Wr'] and
              pt != -1]
    
    scalar = 1.1

    # get the point of the Body
    # From Shoudler to Ankle

    # Get the point of the Left Side Body
    if is_present(body, keypoints_dict["L-Sho"]) \
            and is_present(body, keypoints_dict["L-Hip"]) \
            and is_present(body, keypoints_dict["L-Ank"]):

        left_shoulder_y = body[ keypoints_dict['L-Sho'] ][1]
        left_shoulder_x = body[ keypoints_dict['L-Sho'] ][0]
        left_body_y = body[ keypoints_dict['L-Hip'] ][1]
        left_body_x = body[ keypoints_dict['L-Hip'] ][0]
        left_foot_y = body[ keypoints_dict['L-Ank'] ][1]
        len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))

        condition_1 = left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y \
                - (len_factor / scalar) and left_shoulder_y > left_body_y - (len_factor / scalar)

    else: condition_1 = False

    # Get the point of the right side body
    if is_present(body, keypoints_dict["R-Sho"]) \
            and is_present(body, keypoints_dict["R-Hip"]) \
            and is_present(body, keypoints_dict["R-Ank"]):
        right_shoulder_y = body[ keypoints_dict['R-Sho'] ][1]
        right_shoulder_x = body[ keypoints_dict['R-Sho'] ][0]
        right_body_y = body[ keypoints_dict['R-Hip'] ][1]
        right_body_x = body[ keypoints_dict['R-Hip'] ][0]
        right_foot_y = body[ keypoints_dict['R-Ank'] ][1]
        len_factor = math.sqrt(((right_shoulder_y - right_body_y) ** 2 + (right_shoulder_x - right_body_x) ** 2))

        condition_2 = right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y \
                 - (len_factor / scalar) and right_shoulder_y > right_body_y - (len_factor / scalar)
    #     print(f"right_shoulder_y > right_foot_y - len_factor :", right_shoulder_y, right_foot_y - len_factor )
    #     print(f"right_body_y > right_foot_y - (len_factor / {scalar}): ", right_body_y, right_foot_y - (len_factor / scalar))
    #     print(f"right_shoulder_y > right_body_y - (len_factor / {scalar})", right_shoulder_y, right_body_y - (len_factor / scalar))
    else: condition_2 = False


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
    #should detect head and lower body for fall detection
    # get the point of the head
    if  is_present(body, keypoints_dict["Nose"]) \
            or is_present(body, keypoints_dict["R-Eye"]) \
            or is_present(body, keypoints_dict["L-Eye"]) \
            or is_present(body, keypoints_dict["R-Ear"]) \
            or is_present(body, keypoints_dict["L-Ear"]):
        head_detected = True

    # get the point of the foot
    if  is_present(body, keypoints_dict["R-Ank"]) \
            or is_present(body, keypoints_dict["L-Ank"]) \
            or is_present(body, keypoints_dict["R-Knee"]) \
            or is_present(body, keypoints_dict["L-Knee"]):
        lower_body_detected = True
    if not head_detected or not lower_body_detected:
        condition_3 = False
    else:
        condition_3 = difference < 0

    if condition_1 or condition_2 or condition_3 :
        return True, (xmin, ymin, xmax, ymax)
    return False, (xmin, ymin, xmax, ymax)

# Define Keypoint for detection
def get_person_keypoint(personwiseKeypoints, keypoints_list, detected_keypoints, nm, frame):
    # get keypoints from personwiseKeypoints and store them in keypoint_for_detection
    # all keypoints of each person is stored in an entry of keypoint_for_detection
    if keypoints_list is None or \
            detected_keypoints is None or \
            personwiseKeypoints is None:
        return None
    scale_factor = frame.shape[0] / nm.inputSize[1]
    offset_w = int(frame.shape[1] - nm.inputSize[0] * scale_factor) // 2

    # Define Scale of the keypoint
    def scale(point):
        return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)
    keypoint_for_detection = [ [-1 for j in range(18)] for i in range(len(personwiseKeypoints))]
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

# Define box for body 

def face_bbox(body):
    def is_present(lm_id):
        return body[lm_id] != -1

# Define the joint of the head of human
    head_index = [0, 14, 15, 16, 17]
    head_pts = []

# Define the joint of the shoulder
    sho_index = [2, 5]
    sho_pts = []

# 
    for i, x_y in enumerate(body):

        # Create list for the head point
        if i in head_index and is_present(i):
            head_pts.append(x_y)

        # Create list for the shoulder
        if i in sho_index and is_present(i):
            sho_pts.append(x_y)
# 
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
        x_max_size = 1920
        y_max_size = 1080
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

