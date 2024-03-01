import mediapipe_utils as mpu
import numpy as np
import math

#for fall detection
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def fall_detect(tracker, body):
    """ Return boolean, bbox"""
    def is_present(body, lm_id):
        return body.presence[lm_id] > tracker.presence_threshold
    


    hands_pt = [20, 18, 16, 22, 14, 13, 15, 21, 17, 19]
    list_keypoint_present_x = []
    list_keypoint_present_y = []
    knee_pts =[25, 26]
    present_pts = []
    for i, x_y in enumerate(body.landmarks[:tracker.nb_kps,:2]):
        if i in hands_pt:
            continue
        if is_present(body, i):
            present_pts.append(i)
            list_keypoint_present_x.append(x_y[0]) 
            list_keypoint_present_y.append(x_y[1])

    #should detect knee for fall detection
    if not ( 25 in present_pts ) and not ( 26 in present_pts ):
        return False, None
    
    if list_keypoint_present_x and list_keypoint_present_y:
        xmin, ymin = min(list_keypoint_present_x), min(list_keypoint_present_y)
        xmax, ymax = max(list_keypoint_present_x), max(list_keypoint_present_y)

    left_shoulder_y = body.landmarks[:,:2][11][1]
    left_shoulder_x = body.landmarks[:,:2][11][0]
    right_shoulder_y = body.landmarks[:,:2][12][1]
    left_body_y = body.landmarks[:,:2][23][1]
    left_body_x = body.landmarks[:,:2][23][0]
    right_body_y = body.landmarks[:,:2][24][1]
    len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
    left_foot_y = body.landmarks[:,:2][31][1]
    right_foot_y = body.landmarks[:,:2][32][1]
    dx = int(xmax) - int(xmin)
    dy = int(ymax) - int(ymin)
    difference = dy - dx

    if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
            len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (
            right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (
            len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) \
            or difference < 0:
        return True, (xmin, ymin, xmax, ymax)
    return False, None


def fall_detect_3d(show_3d, body) -> bool:
    if body is not None:
        points = body.landmarks if show_3d == "image" else body.landmarks_world

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
        #self.vis3d.add_segment(body_vector, mid_shoulders, color= [1, 0, 0])
        #self.vis3d.add_segment(body_vector, nose, color= [1, 0, 0])
        v1 = (0, 0, 2)
        v2 = (2, 0, 0)
        #v3 = (0, 0, -2)
        #v4 = (-2, 0, 0)
        #angle1 = angle_between(  v1 , body_vector)
        #angle1 = angle1 / (2 * np.pi)  * 360
        #angle1 = angle1 if angle1 <= 90.0 else 180.0 - angle1
        #print("angle1: ", angle1)

        normal1 = (0, 4, 0) # normal of the floor
        angle_shoulder_btw_foot = angle_between(normal1, body_vector_shoulder_btw_ankle)
        angle_shoulder_btw_foot = np.pi/2 - angle_shoulder_btw_foot if angle_shoulder_btw_foot <= np.pi/2 else angle_shoulder_btw_foot - np.pi/2
        angle_shoulder_btw_foot = angle_shoulder_btw_foot / (np.pi)  * 180
        #print("angle_shoulder_btw_foot: ",  angle_shoulder_btw_foot)

        
        angle_shoulder_btw_knee = angle_between(normal1, body_vector_shoulder_btw_knee)
        angle_shoulder_btw_knee = np.pi/2 - angle_shoulder_btw_knee if angle_shoulder_btw_knee <= np.pi/2 else angle_shoulder_btw_knee - np.pi/2
        angle_shoulder_btw_knee = angle_shoulder_btw_knee / (np.pi)  * 180
        #print("angle_shoulder_btw_knee: ",  angle_shoulder_btw_knee)
        #with open("log.txt", "a") as f:
        #    f.write(f"angle_shoulder_btw_foot: ,{angle_shoulder_btw_foot}\n")

        angle_nose_btw_hip = angle_between(normal1, body_vector_nose_btw_hip)
        angle_nose_btw_hip = np.pi/2 - angle_nose_btw_hip\
                                if angle_nose_btw_hip <= np.pi/2 else angle_nose_btw_hip - np.pi/2
        angle_nose_btw_hip = angle_nose_btw_hip / (np.pi)  * 180



        threshold = 50
        if angle_shoulder_btw_foot <= threshold :
            return True
        #if angle_nose_btw_hip <= threshold:
        #    return True
        
    return False

def person_bbox(tracker, body):
    def is_present(body, lm_id):
        return body.presence[lm_id] > tracker.presence_threshold
    
    hands_pt = [20, 18, 16, 22, 14, 13, 15, 21, 17, 19]
    list_keypoint_present_x = []
    list_keypoint_present_y = []
    knee_pts =[25, 26]
    present_pts = []
    for i, x_y in enumerate(body.landmarks[:tracker.nb_kps,:2]):
        if i in hands_pt:
            continue
        if is_present(body, i):
            present_pts.append(i)
            list_keypoint_present_x.append(x_y[0]) 
            list_keypoint_present_y.append(x_y[1])
    
    if list_keypoint_present_x and list_keypoint_present_y:
        xmin, ymin = min(list_keypoint_present_x), min(list_keypoint_present_y)
        xmax, ymax = max(list_keypoint_present_x), max(list_keypoint_present_y)

        #check if out of bound
        # default cam image size: 1152 x 648
        x_max_size = 1152
        y_max_size = 648
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

def face_bbox(tracker, body):
    def is_present(body, lm_id):
        return body.presence[lm_id] > tracker.presence_threshold
    
    list_keypoint_present_x = []
    list_keypoint_present_y = []
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
        #dist_btw_eyes *= 1.2 # scale up the dist
        xmin, ymin = min(x_pts), min( y_pts)
        xmax, ymax = max(x_pts), max( y_pts)
        ymin = ymin - dist_btw_eyes
        xmin = xmin - dist_btw_eyes
        xmax = xmax + dist_btw_eyes
        ymax = ymax + dist_btw_eyes

        #larger the width of the face bbox
        xmax = xmax *1.07
        xmin = xmin / 1.1
        #check if out of bound
        # default cam image size: 1152 x 648
        x_max_size = 1152
        y_max_size = 648
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