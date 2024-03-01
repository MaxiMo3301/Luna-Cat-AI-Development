import time
import marshal
from math import sin, cos, atan2, pi, hypot, degrees, floor
mobilenet_input_size = (300, 300)
lm_input_size = (256, 256)
frame_size = (1920, 1080)
lm_score_thresh = 0.4
pd_score_thresh = 0.7
scale_x = 1920 / mobilenet_input_size[0]
scale_y = 1080 / mobilenet_input_size[1]
cropped_frame_size = (960, 960)
rect_transf_scale = .8

# BufferMgr is used to statically allocate buffers once
# (replace dynamic allocation).
# These buffers are used for sending result to host
class BufferMgr:
    def __init__(self):
        self._bufs = {}
    def __call__(self, size):
        try:
            buf = self._bufs[size]
        except KeyError:
            buf = self._bufs[size] = Buffer(size)
            node.warn(f"New buffer allocated: {size}")
        return buf

buffer_mgr = BufferMgr()


def send_result(type=0, people_state=0):
    # type : 0
    #   0 : pose detection only (detection score < threshold) or no person is detected
    #   1 : pose detection + landmark regression
    # people_state is a dictionary containing lm_score, lms, lm_world, sqn_rr_center_x
    # sqn_rr_center_y, sqn_rr_size, rotation, topLeft_x, topLeft_y,
    # bottomRight_x, bottomRight_y, status.name

    result = dict([("type", type), ("people_state", people_state)])

    result_serial = marshal.dumps(result, 2)
    buffer = buffer_mgr(len(result_serial))
    buffer.getData()[:] = result_serial
    #node.warn("len result:" + str(len(result_serial)))

    node.io['host'].send(buffer)
    #node.warn("Manager sent result to host")

def correct_bbox(x_min, y_min, x_max, y_max):
    # range: (0, 1)
    x_min_cor = max(x_min, 0.0)
    y_min_cor = max(y_min, 0.0)
    x_max_cor = min(x_max, 1.0)
    y_max_cor = min(y_max, 1.0)
    #node.warn(f"{x_min_cor}, {y_min_cor}, {x_max_cor}, {y_max_cor}")
    return x_min_cor, y_min_cor, x_max_cor, y_max_cor

def denormalize_bbox(x1, y1, x2, y2):
    # cropped frame size: 960 x 960
    scale_factor = 960 / mobilenet_input_size[1]
    offset_w = int(960 - mobilenet_input_size[0] * scale_factor) // 2
    def scale(point):
        return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)

    new_x1, new_y1 = scale([x1, y1])
    new_x2, new_y2 = scale([x2, y2])

    return new_x1, new_y1, new_x2, new_y2


def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier


# Main loop
while True:
    time.sleep(0.001)
    # Get image frame
    img = node.io["rgb_frame"].get()
    # Get detection output
    track = node.io["tracker_out"].get()
    trackletsData = track.tracklets

    people_state = {}
    if len(trackletsData) >= 1 and img is not None:
        for t in trackletsData:
            roi = t.roi.denormalize(mobilenet_input_size[0], mobilenet_input_size[1])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)
            #node.warn(dir(t))


            people_state[t.id] = dict([("topLeft_x", x1), ("topLeft_y", y1),
                                       ("bottomRight_x", x2), ("bottomRight_y", y2), ("status", t.status.name)])

            node.warn(t.status.name)
            if t.status.name != "TRACKED":
                continue

            x1, y1, x2, y2 = denormalize_bbox(x1, y1, x2, y2)
            x1 = x1 / cropped_frame_size[0] - 0.16 # offset is added
            y1 = y1 / cropped_frame_size[1] - 0.16
            x2 = x2 / cropped_frame_size[0] + 0.16
            y2 = y2 / cropped_frame_size[1] + 0.16

            x1, y1, x2, y2 = correct_bbox(x1, y1, x2, y2)
            y1 = float(f"{y1:.3f}")
            y2 = float(f"{y2:.3f}")
            x1 = float(f"{x1:.3f}")
            x2 = float(f"{x2:.3f}")
            #y1 = 0.0
            #y2 = 1.0


            cfg_pre_pd_cropped = ImageManipConfig()
            cfg_pre_pd_cropped.setKeepAspectRatio(False)
            cfg_pre_pd_cropped.setCropRect(x1, y1, x2, y2)
            #cfg_pre_pd_cropped.setCropRotatedRect(rect, True)
            cfg_pre_pd_resize = ImageManipConfig()
            cfg_pre_pd_resize.setKeepAspectRatio(False)
            cfg_pre_pd_resize.setResizeThumbnail(224, 224, 0, 0, 0)

            node.io["to_pd_input"].send(img)
            node.io['pre_pd_manip_cropped_cfg'].send(cfg_pre_pd_cropped)
            node.io["pre_pd_manip_resize_cfg"].send(cfg_pre_pd_resize)
            node.warn("Manager sent thumbnail config to pre_pd manip")
            # Wait for pd post processing's result
            detection = node.io['from_post_pd_nn'].get().getLayerFp16("result")
            node.warn("Manager received pd result: " + str(detection))
            pd_score, sqn_rr_center_x, sqn_rr_center_y, sqn_scale_x, sqn_scale_y = detection
            if pd_score < pd_score_thresh:
                send_result(0)
                continue
            scale_center_x = sqn_scale_x - sqn_rr_center_x
            scale_center_y = sqn_scale_y - sqn_rr_center_y
            sqn_rr_size = 2 * rect_transf_scale * hypot(scale_center_x, scale_center_y)
            rotation = 0.5 * pi - atan2(-scale_center_y, scale_center_x)
            rotation = rotation - 2 * pi * floor((rotation + pi) / (2 * pi))


            # Routing frame to lm
            sin_rot = sin(rotation)  # We will often need these values later
            cos_rot = cos(rotation)
            #Tell pre_lm_manip how to crop body region



            rr = RotatedRect()
            rr.center.x = truncate_float(sqn_rr_center_x, 2)
            rr.center.y = truncate_float(sqn_rr_center_y, 2)
            sqn_rr_size =  1.0
            rr.size.width = truncate_float(sqn_rr_size, 3)
            rr.size.height = truncate_float(sqn_rr_size, 3)
            rr.angle = degrees(rotation)



            # for debugging
            node.warn(f"sqn_rr_size:{sqn_rr_size}")
            node.warn(f"rr.center.x:{rr.center.x}")
            node.warn(f" rr.center.y:{rr.center.y}")
            node.warn(f"rr.size.width:{rr.size.width}")
            node.warn(f"rr.size.height:{rr.size.height}")

            #rr = RotatedRect()
            #rr.center.x = (x2 + x1) / 2
            #rr.center.y = (y2 + y1) / 2
            # for debugging
            #node.warn(f"sqn_rr_size:{sqn_rr_size}")
            #sqn_rr_size = sqn_rr_size if sqn_rr_size < 1.095 else 1.095
            #rr.size.width = x2 - x1
            #rr.size.height = y2 - y1



            cfg_rotated_crop = ImageManipConfig()
            cfg_rotated_crop.setCropRotatedRect(rr, True)
            cfg_rotated_crop.setKeepAspectRatio(False)
            #cfg_rotated_crop.setRotationDegrees(degrees(rotation))

            cfg_resize = ImageManipConfig()
            cfg_resize.setKeepAspectRatio(False)
            cfg_resize.setResize(256, 256)

            node.io["lm_image"].send(img)
            node.io['pre_lm_manip_rotated_crop_cfg'].send(cfg_rotated_crop)
            node.warn("Manager sent config to pre_lm manip CROP")

            node.io["pre_lm_manip_resize_cfg"].send(cfg_resize)
            node.warn("Manager sent config to pre_lm manip Resize")







            # Wait for lm's result
            lm_result = node.io['from_lm_nn'].get()
            node.warn("Manager received result from lm nn")
            lm_score = lm_result.getLayerFp16("Identity_1")[0]
            people_state[t.id]["lm_score"] = lm_score
            if lm_score > lm_score_thresh:
                lms = lm_result.getLayerFp16("Identity")
                lms_world = lm_result.getLayerFp16("Identity_4")[:99]
                people_state[t.id]["lms"] = lms
                people_state[t.id]["lms_world"] = lms_world
                people_state[t.id]["sqn_rr_center_x"] = sqn_rr_center_x
                people_state[t.id]["sqn_rr_center_y"] = sqn_rr_center_y
                people_state[t.id]["sqn_rr_size"] = sqn_rr_size
                people_state[t.id]["rotation"] = rotation




        send_result(type=1, people_state=people_state)
        del people_state

    else:
        send_result(type=0, people_state=people_state)
        del people_state

