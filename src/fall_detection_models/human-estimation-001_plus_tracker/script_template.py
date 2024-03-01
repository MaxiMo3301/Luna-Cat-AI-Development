

import marshal
import time

nn_input_size = (456, 256)
mobilenet_input_size = (300, 300)


def denormalize_bbox(x1, y1, x2, y2, input_size_w, input_size_h):
    """Rescale tracker bbox on the original frame"""
    scale_factor = ${_img_w} / input_size_h
    offset_w = int(${_img_h} - input_size_w * scale_factor) // 2

    def scale(point):
        return point[0] * scale_factor + offset_w, point[1] * scale_factor

    new_x1, new_y1 = scale([x1, y1])
    new_x2, new_y2 = scale([x2, y2])
    new_y1 = new_y1 - ${_pad_h}
    new_y2 = new_y2 - ${_pad_h}
    new_x2 = new_x2 + ${_pad_h}
    new_x1 = new_x1 + ${_pad_h}

    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)
# send result
class BufferMgr:
    def __init__(self):
        self._bufs = {}
    def __call__(self, size):
        try:
            buf = self._bufs[size]
        except KeyError:
            buf = self._bufs[size] = Buffer(size)
            ${_TRACE} (f"New buffer allocated: {size}")
        return buf


def correct_bbox(x_min, y_min, x_max, y_max):
    """Asure four points should be in [0, 1]"""
    x_min_cor = max(x_min, 0.0)
    y_min_cor = max(y_min, 0.0)
    x_max_cor = min(x_max, 1.0)
    y_max_cor = min(y_max, 1.0)
    # for debugging
    #node.warn(f"{x_min_cor}, {y_min_cor}, {x_max_cor}, {y_max_cor}")
    return x_min_cor, y_min_cor, x_max_cor, y_max_cor

def send_result(person_state=0, heatmaps=0, pafs=0):
    """
    person dictionary with an entry: (top_left_x, top_right_y, bottom_right_x,
    bottom_right_y, status, heatmaps, pafs)
    """
    if person_state == 0:
        person_state = dict()
    result = dict( [("person_state", person_state), ("heatmaps", heatmaps), ("pafs", pafs)] )
    result_serial = marshal.dumps(result, 2)
    buffer = buffer_mgr(len(result_serial))
    buffer.getData()[:] = result_serial
    ${_TRACE}("len result:" + str(len(result_serial)))

    node.io['host'].send(buffer)
    ${_TRACE}("Manager sent result to host")

buffer_mgr = BufferMgr()

pre_mobile_manip_cfg = ImageManipConfig()
pre_mobile_manip_cfg.setKeepAspectRatio(True)
pre_mobile_manip_cfg.setResizeThumbnail(mobilenet_input_size[0], mobilenet_input_size[1], 0, 0, 0)

pre_pose_manip_cfg = ImageManipConfig()
pre_pose_manip_cfg.setKeepAspectRatio(True)
pre_pose_manip_cfg.setResizeThumbnail(nn_input_size[0], nn_input_size[1], 0, 0, 0)

while True:
    time.sleep(0.01)
    # send cfg to both tracker and gen2_pose
    node.io["to_pre_pose_manip_resize"].send(pre_pose_manip_cfg)
    node.io["pre_mobile_manip_resize"].send(pre_mobile_manip_cfg)

    # first get the tracker output
    track = node.io["from_tracker"].get()
    trackletsData = track.tracklets
    people_state = {}

    for t in trackletsData:
        roi = t.roi.denormalize(mobilenet_input_size[0], mobilenet_input_size[1])
        x1 = roi.topLeft().x
        y1 = roi.topLeft().y
        x2 = roi.bottomRight().x
        y2 = roi.bottomRight().y

        x1, y1, x2, y2 = denormalize_bbox(x1, y1, x2, y2, mobilenet_input_size[0], mobilenet_input_size[1])


        people_state[t.id] =  dict([("topLeft_x", x1), ("topLeft_y", y1),
                                   ("bottomRight_x", x2), ("bottomRight_y", y2), ("status", t.status.name)])
    # get pose result
    pose_out = node.io["from_pose_detection"].get()
    heatmaps = pose_out.getLayerFp16('Mconv7_stage2_L2')
    pafs = pose_out.getLayerFp16('Mconv7_stage2_L1')
    send_result(people_state, heatmaps, pafs)
