import cv2
import numpy as np

# LINE_BODY and COLORS_BODY are used when drawing the skeleton in 3D. 
rgb = {"right":(0,1,0), "left":(1,0,0), "middle":(1,1,0)}
LINES_BODY = [[9,10],[4,6],[1,3],
            [12,14],[14,16],[16,20],[20,18],[18,16],
            [12,11],[11,23],[23,24],[24,12],
            [11,13],[13,15],[15,19],[19,17],[17,15],
            [24,26],[26,28],[32,30],
            [23,25],[25,27],[29,31]]

COLORS_BODY = ["middle","right","left",
                "right","right","right","right","right",
                "middle","middle","middle","middle",
                "left","left","left","left","left",
                "right","right","right","left","left","left"]
COLORS_BODY = [rgb[x] for x in COLORS_BODY]

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

class BlazeposeRenderer:
    def __init__(self,
                tracker=None,
                output=None):
        self.tracker = tracker
        self.frame = None
        self.pause = False

        # Rendering flags
        self.show_rot_rect = False
        self.show_landmarks = True
        self.show_score = False
        self.show_fps = True


        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,tracker.video_fps,(tracker.img_w, tracker.img_h)) 

    @staticmethod
    def is_present(body, lm_id):
        return body.presence[lm_id] > 0.1

    def draw_landmarks(self, body):
        # show bbox
        #cv2.polylines(self.frame, [np.array(body.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)

        list_connections = LINES_BODY
        lines = [np.array([body.landmarks[point,:2] for point in line]) for line in list_connections if self.is_present(body, line[0]) and self.is_present(body, line[1])]
        cv2.polylines(self.frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)

        # for i,x_y in enumerate(body.landmarks_padded[:,:2]):
        for i,x_y in enumerate(body.landmarks[:33,:2]):  # tracker.nb_kps = 33
            if self.is_present(body, i):
                if i > 10:
                    color = (0, 255 ,0) if i % 2 == 0 else (0, 0, 255)
                elif i == 0:
                    color = (0,255,255)
                elif i in [4,5,6,8,10]:
                    color = (0,255,0)
                else:
                    color = (0,0,255)
                cv2.circle(self.frame, (x_y[0], x_y[1]), 4, color, -11)

                
        
    def draw(self, frame, body):
        self.frame = frame
        if body:
            self.draw_landmarks(body)
        self.body = body
        return self.frame
    
    def exit(self):
        if self.output:
            self.output.release()

    def waitKey(self, delay=1):
        if self.show_fps:
            self.tracker.fps.draw(self.frame, orig=(50,50), size=1, color=(240,180,100))
        cv2.imshow("Blazepose", self.frame)
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay) 
        if key == 32:
            # Pause on space bar
            self.pause = not self.pause
        elif key == ord('r'):
            self.show_rot_rect = not self.show_rot_rect
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('s'):
            self.show_score = not self.show_score
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        elif key == ord('x'):
            if self.tracker.xyz:
                self.show_xyz = not self.show_xyz    
        elif key == ord('z'):
            if self.tracker.xyz:
                self.show_xyz_zone = not self.show_xyz_zone 
        return key
        
    
    
    
            