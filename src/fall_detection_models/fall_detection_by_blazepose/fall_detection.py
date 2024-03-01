import mediapipe_utils as mpu
import numpy as np
import math

keypoints_dict = {'Nose': 0 , 'Neck': 1, 'R-Sho': 2, 'R-Elb': 3, 'R-Wr': 4, 'L-Sho': 5, 'L-Elb': 6, 'L-Wr': 7, 'R-Hip': 8, 'R-Knee': 9, 'R-Ank': 10,
                    'L-Hip': 11, 'L-Knee': 12 , 'L-Ank': 13 , 'R-Eye': 14, 'L-Eye': 15, 'R-Ear': 16, 'L-Ear':17}

class hip_shoulder_timestamp:
    """y -coordinate of mid-hip and timestamp"""
    def __init__(self, hip: list = [], shoulder: list = [], timestamp=-1) -> None:
        """
        :param hip: a list of x, y-coordinates of hip for all tracked people
        :param shoulder: a list of x, y-coordinates of shoulder for all tracked people
        :param timestamp: the time when hip coordinates are detected
        :return None
        """
        self.hip_xy = [ _ for _ in hip]
        self.shoulder_xy = [ _ for _ in shoulder]
        self.timestamp = timestamp
class hip_window:
    """
    used to record x and y coordinate of shoulders and hips of people across frames
    Attributes:
        list: a list of hip_position_and_timestamp
    """
    def __init__(self, size):
        self.list = [hip_shoulder_timestamp() for i in range(size)]

    def update(self, keypoint_for_detection, timestamp):
        """
        pop leftmost entry of self.list and append new entry
        :param keypoint_for_detection: a list of size 18 with each entry contains (x, y)
                that indicates the positions of a keypoint on the current frame
        :param timestamp: current time in second
        :return: None
        """
        new_list = self.list[1:len(self.list)]
        if not keypoint_for_detection:
            new_list.append(hip_shoulder_timestamp([None], [None], timestamp))
            self.list = new_list
        else:
            tmp_hip = [-1 for _ in range(len(keypoint_for_detection))]
            tmp_shoulder = [-1 for _ in range(len(keypoint_for_detection))]
            for index, points in enumerate(keypoint_for_detection):
                l_hip = points[ mpu.KEYPOINT_DICT['left_hip'] ]
                r_hip = points[ mpu.KEYPOINT_DICT['right_hip'] ]
                hip_y = ( l_hip[1] + r_hip[1] )/2
                hip_x = ( l_hip[0] + r_hip[0] )/2
                l_shoulder = points[mpu.KEYPOINT_DICT['left_shoulder']]
                r_shoulder = points[mpu.KEYPOINT_DICT['right_shoulder']]
                shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
                shoulder_x = (l_shoulder[0] + r_shoulder[0]) / 2

                tmp_hip[index] = (hip_x, hip_y)
                tmp_shoulder[index] = (shoulder_x, shoulder_y)

            new_list.append(hip_shoulder_timestamp(tmp_hip, tmp_shoulder, timestamp))
            self.list = new_list


def fall_detected_by_velocity(hip_shoulder_timestamp, interval):
    # hip_shoulder_timestamp contains x, y-coordinate of mid-hip and mid-shoulder of a person
    # and a timestamp
    # for single person only
    # velocity is calculated by centre of the body across the frames
    x_factor = 0.4
    y_factor = 0.6
    if not hip_shoulder_timestamp.list[interval - 1].hip_xy \
            or not hip_shoulder_timestamp.list[0].hip_xy \
            or not hip_shoulder_timestamp.list[interval - 1].shoulder_xy\
            or not hip_shoulder_timestamp.list[0].shoulder_xy:
        return None


    hip_x2 = hip_shoulder_timestamp.list[interval - 1].hip_xy[0][0]
    hip_x1 = hip_shoulder_timestamp.list[0].hip_xy[0][0]
    hip_y2 = hip_shoulder_timestamp.list[interval - 1].hip_xy[0][1]
    hip_y1 = hip_shoulder_timestamp.list[0].hip_xy[0][1]
    shoulder_x2 = hip_shoulder_timestamp.list[interval - 1].shoulder_xy[0][0]
    shoulder_x1 = hip_shoulder_timestamp.list[0].shoulder_xy[0][0]
    shoulder_y2 = hip_shoulder_timestamp.list[interval - 1].shoulder_xy[0][1]
    shoulder_y1 = hip_shoulder_timestamp.list[0].shoulder_xy[0][1]
    body_centre_x1 = (shoulder_x1 + hip_x1) / 2
    body_centre_x2 = (shoulder_x2 + hip_x2) / 2
    body_centre_y1 = (shoulder_y1 + hip_y1) / 2
    body_centre_y2 = (shoulder_y2 + hip_y2) / 2

    displacement = math.sqrt(((body_centre_x2 - body_centre_x1) * x_factor)**2 +
                              ((body_centre_y2 - body_centre_y1) * y_factor)**2)
    time_elasped = hip_shoulder_timestamp.list[interval - 1].timestamp - hip_shoulder_timestamp.list[0].timestamp
    velocity = displacement / time_elasped

    return velocity


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'

    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def read_landmarks_pt(fall_detect_mode, body):
    # return points for fall detection
    if body is not None:

        # world landmarks are real-world 3D coordinates in meters with
        # the origin at the center between hips.
        # world landmarks share the same landmark topology as landmarks.
        # However, landmarks provide coordinates (in pixels) of a 3D object projected onto
        # the 2D image surface, while world landmarks provide coordinates (in meters) of
        # the 3D object itself.
        points = body.landmarks if fall_detect_mode == "image" else body.landmarks_world
        if fall_detect_mode == "mixed":
            if body.xyz_ref:
                """
                Beware, the y value of landmarks_world coordinates is negative for landmarks 
                above the mid hips (like shoulders) and negative for landmarks below (like feet).
                The y value of (x,y,z) coordinates given by depth sensor is negative in the lower part
                of the image and positive in the upper part.
                """
                translation = body.xyz / 1000
                translation[1] = -translation[1]
                if body.xyz_ref == "mid_hips":
                    points = points + translation
                elif body.xyz_ref == "mid_shoulders":
                    mid_hips_to_mid_shoulders = np.mean([
                        points[mpu.KEYPOINT_DICT['right_shoulder']],
                        points[mpu.KEYPOINT_DICT['left_shoulder']]],
                        axis=0)
                    points = points + translation - mid_hips_to_mid_shoulders

            else:
                return None
        return points
    return None


def fall_detect_3d(points) -> (bool,tuple):
    """
    :param points: a list-like object containing 33 keypoints
    :return True if fall is detected else False, body angles with respect to floor
    """

    #get vector form mid_hips_to_nose
    mid_shoulders = np.mean([
                points[mpu.KEYPOINT_DICT['right_shoulder']],
                points[mpu.KEYPOINT_DICT['left_shoulder']]],
                axis=0)
    nose = points[mpu.KEYPOINT_DICT['nose']]

    mid_ankle = np.mean([
                points[mpu.KEYPOINT_DICT['right_ankle']],
                points[mpu.KEYPOINT_DICT['left_ankle']]],
                axis=0)

    mid_hip = np.mean([
                points[mpu.KEYPOINT_DICT['right_hip']],
                points[mpu.KEYPOINT_DICT['left_hip']]],
                axis=0)

    mid_knee = np.mean([
                points[mpu.KEYPOINT_DICT['right_knee']],
                points[mpu.KEYPOINT_DICT['left_knee']]],
                axis=0)

    body_vector_shoulder_btw_ankle = mid_shoulders - mid_ankle  #should perpendicular to the floor when stand and parallel to the floor when fall
    body_vector_shoulder_btw_knee = mid_shoulders - mid_knee
    body_vector_nose_btw_hip = nose - mid_hip
    body_vector_nose_btw_ankle = nose - mid_ankle
    body_vector_shoulder_btw_hip = mid_shoulders - mid_hip
    body_vector_hip_btw_ankle = mid_hip - mid_ankle

    v1 = (0, 0, 2)
    v2 = (2, 0, 0)

    normal1 = (0, 4, 0) # normal of the floor
    angle_shoulder_btw_foot = angle_between(normal1, body_vector_shoulder_btw_ankle)
    angle_shoulder_btw_foot = np.pi/2 - angle_shoulder_btw_foot if angle_shoulder_btw_foot <= np.pi/2 else angle_shoulder_btw_foot - np.pi/2
    angle_shoulder_btw_foot = angle_shoulder_btw_foot / (np.pi)  * 180

    angle_shoulder_btw_knee = angle_between(normal1, body_vector_shoulder_btw_knee)
    angle_shoulder_btw_knee = np.pi/2 - angle_shoulder_btw_knee if angle_shoulder_btw_knee <= np.pi/2 else angle_shoulder_btw_knee - np.pi/2
    angle_shoulder_btw_knee = angle_shoulder_btw_knee / (np.pi)  * 180

    angle_nose_btw_hip = angle_between(normal1, body_vector_nose_btw_hip)
    angle_nose_btw_hip = np.pi/2 - angle_nose_btw_hip\
                            if angle_nose_btw_hip <= np.pi/2 else angle_nose_btw_hip - np.pi/2
    angle_nose_btw_hip = angle_nose_btw_hip / (np.pi)  * 180

    angle_nose_btw_ankle = angle_between(normal1, body_vector_nose_btw_ankle)
    angle_nose_btw_ankle = np.pi / 2 - angle_nose_btw_ankle \
        if angle_nose_btw_ankle <= np.pi / 2 else angle_nose_btw_ankle - np.pi / 2
    angle_nose_btw_ankle = angle_nose_btw_ankle / (np.pi) * 180

    angle_shoulder_btw_hip = angle_between(normal1, body_vector_shoulder_btw_hip)
    angle_shoulder_btw_hip = np.pi / 2 - angle_shoulder_btw_hip \
        if angle_shoulder_btw_hip <= np.pi / 2 else angle_shoulder_btw_hip - np.pi / 2
    angle_shoulder_btw_hip = angle_shoulder_btw_hip / (np.pi) * 180

    angle_hip_btw_ankle = angle_between(normal1, body_vector_hip_btw_ankle)
    angle_hip_btw_ankle = np.pi / 2 - angle_hip_btw_ankle \
        if angle_hip_btw_ankle <= np.pi / 2 else angle_hip_btw_ankle - np.pi / 2
    angle_hip_btw_ankle = angle_hip_btw_ankle / (np.pi) * 180

    diff_y_in_knee_hip = mid_knee[1] - mid_hip[1]

    # for showing stats
    body_angle_with_respect_to_floor = (f'angle_nose_btw_hip:{angle_nose_btw_hip:.2f}',
                                        f'angle_shoulder_btw_foot:{angle_shoulder_btw_foot:.2f}',
                                        f'angle_shoulder_btw_knee:{angle_shoulder_btw_knee:.2f}',
                                        f"diff_y_in_knee_hip: {diff_y_in_knee_hip:.2f}")

    if angle_nose_btw_hip > 54:
        if angle_hip_btw_ankle < 20 or diff_y_in_knee_hip < -0.22:
            return True, body_angle_with_respect_to_floor
        return False, body_angle_with_respect_to_floor
    elif angle_shoulder_btw_knee < 50 or angle_shoulder_btw_foot < 50 or angle_shoulder_btw_hip < 50:
        return True, body_angle_with_respect_to_floor

    return False, body_angle_with_respect_to_floor


def face_bbox(tracker, body):

    def is_present(body, lm_id):
        return body.presence[lm_id] > tracker.presence_threshold

    eyes_outer_and_shoulder_index = [3, 6, 11, 12]
    eyes_outer_and_shoulder_pts = []
    left_eyes_outer_exist = False
    right_eyes_outer_exist = False
    left_shoulder_exist = False
    right_shoulder_exist = False
    for i, x_y in enumerate(body.landmarks[:tracker.nb_kps,:2]):
        if i in eyes_outer_and_shoulder_index and is_present(body, i):
            eyes_outer_and_shoulder_pts.append(x_y)
            if i == 3:
                left_eyes_outer_exist = True
            elif i == 6:
                right_eyes_outer_exist = True
            elif i == 11:
                left_shoulder_exist = True
            elif i == 12:
                right_shoulder_exist = True       
                
    
    if left_eyes_outer_exist and right_eyes_outer_exist\
        and left_shoulder_exist and right_shoulder_exist:

        x_pts = [eyes_outer_and_shoulder_pts[0][0], eyes_outer_and_shoulder_pts[1][0], 
             eyes_outer_and_shoulder_pts[2][0], eyes_outer_and_shoulder_pts[3][0] ]
        y_pts = [eyes_outer_and_shoulder_pts[0][1], eyes_outer_and_shoulder_pts[1][1], 
             eyes_outer_and_shoulder_pts[2][1], eyes_outer_and_shoulder_pts[3][1] ]
        
        dist_btw_eyes = math.sqrt( (eyes_outer_and_shoulder_pts[0][0] - eyes_outer_and_shoulder_pts[1][0])**2 + 
                                  ( eyes_outer_and_shoulder_pts[0][1] - eyes_outer_and_shoulder_pts[1][1] )**2 )

        xmin, ymin = min(x_pts), min( y_pts)
        xmax, ymax = max(x_pts), max( y_pts)
        if int(xmax) - int(xmin) == 0 or int(ymax) - int(ymin) == 0:
            return None

        ymin = ymin - dist_btw_eyes
        xmin = xmin - dist_btw_eyes
        xmax = xmax + dist_btw_eyes
        ymax = ymax + dist_btw_eyes

        #larger the width of the face bbox
        xmax = xmax * 1.15
        xmin = xmin / 1.15
        #check if out of bound
        # default cam image size: 1152 x 648
        x_max_size = 1152
        y_max_size = 648

        # for video mode
        #x_max_size = 1920
        #y_max_size = 1080
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
