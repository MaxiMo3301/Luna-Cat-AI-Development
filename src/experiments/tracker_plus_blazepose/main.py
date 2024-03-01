#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import marshal
import utils
from utils import find_isp_scale_params
from math import sin, cos
import BlazeposeRenderer


labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

SCRIPT_DIR = Path(__file__).resolve().parent
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/pose_landmark_full_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/pose_landmark_lite_sh4.blob")
DIVIDE_BY_255_MODEL = str(SCRIPT_DIR / "custom_models/DivideBy255_sh1.blob")
POSE_DETECTION_MODEL = str(SCRIPT_DIR / "models/pose_detection_sh4.blob")
DETECTION_POSTPROCESSING_MODEL = str(SCRIPT_DIR / "custom_models/DetectionBestCandidate_sh1.blob")
nnPathDefault = str((SCRIPT_DIR / Path('models/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)

args = parser.parse_args()

mobilenet_input_size = (300, 300)
pd_input_length = 224
lm_input_size = (256, 256)
internal_frame_height = 1080
#frame_size, scale_nd = find_isp_scale_params(internal_frame_height)
#print(f"frame size", frame_size)

# mobilenet image crop coordinates
cropped_frame_size = (960, 960)
mobilenet_crop_coordintes = (640/1920, 1. - 960/1080, 1600/1920, 1.)
#cropped_frame_size = (1920, 1080)
#mobilenet_crop_coordintes = (0., 1., 0., 1.)


width, scale_nd = find_isp_scale_params(cropped_frame_size[0] * 1920 / 1080, is_height=False)
img_h = int(round(1080 * scale_nd[0] / scale_nd[1]))
img_w = int(round(1920 * scale_nd[0] / scale_nd[1]))
pad_h = (img_w - img_h) // 2
pad_w = 0
frame_size = img_w
crop_w = 0

def create_pipeline(lm_model, crop=True):

    # Create pipeline
    pipeline = dai.Pipeline()

    # script node
    script_node = pipeline.create(dai.node.Script)
    script_node.setScriptPath("script.py")


    # cam Properties
    camRgb = pipeline.create(dai.node.ColorCamera)
    #camRgb.setIspScale(scale_nd[0], scale_nd[1])
    #camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.setFps(4)

    if crop:
        camRgb.setPreviewSize(frame_size, frame_size)
    else:
        camRgb.setPreviewSize(1920, 1080)


    # Define mobilenet pre processing crop the image first then resize into mobilenet input size
    pre_mobilenet_manip_cropped_first = pipeline.create(dai.node.ImageManip)
    pre_mobilenet_manip_cropped_first.setMaxOutputFrameSize(cropped_frame_size[0] * cropped_frame_size[1] * 3)
    pre_mobilenet_manip_cropped_first.inputConfig.setWaitForMessage(False)
    pre_mobilenet_manip_cropped_first.initialConfig.setCropRect(mobilenet_crop_coordintes)
    pre_mobilenet_manip_cropped_first.initialConfig.setKeepAspectRatio(False)
    pre_mobilenet_manip_cropped_first.inputImage.setQueueSize(1)
    pre_mobilenet_manip_cropped_first.inputImage.setBlocking(False)
    camRgb.preview.link(pre_mobilenet_manip_cropped_first.inputImage)

    pre_mobilenet_manip_resize_second = pipeline.create(dai.node.ImageManip)
    pre_mobilenet_manip_resize_second.setMaxOutputFrameSize(mobilenet_input_size[0] * mobilenet_input_size[1] * 3)
    pre_mobilenet_manip_resize_second.inputConfig.setWaitForMessage(False)
    pre_mobilenet_manip_resize_second.initialConfig.setKeepAspectRatio(False)
    pre_mobilenet_manip_resize_second.initialConfig.setResizeThumbnail(mobilenet_input_size[0], mobilenet_input_size[1])
    pre_mobilenet_manip_resize_second.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    pre_mobilenet_manip_resize_second.inputImage.setQueueSize(1)
    pre_mobilenet_manip_resize_second.inputImage.setBlocking(False)
    pre_mobilenet_manip_cropped_first.out.link(pre_mobilenet_manip_resize_second.inputImage)





    # send rgb frame into script_node
    pre_mobilenet_manip_cropped_first.out.link(script_node.inputs["rgb_frame"])




    # testing MobileNet DetectionNetwork
    detectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
    detectionNetwork.setBlobPath(args.nnPath)
    detectionNetwork.setConfidenceThreshold(0.6)
    detectionNetwork.input.setBlocking(False)

    # define object tracker
    objectTracker = pipeline.create(dai.node.ObjectTracker)
    objectTracker.setDetectionLabelsToTrack([15])  # track only person
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)



    # set object tracker output
    pre_mobilenet_manip_resize_second.out.link(detectionNetwork.input)
    detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
    detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    detectionNetwork.out.link(objectTracker.inputDetections)



    # output tracker frame
    xlinkOut = pipeline.create(dai.node.XLinkOut)
    xlinkOut.setStreamName("preview")
    #  originally the frame is directly pass to host
    #objectTracker.passthroughTrackerFrame.link(xlinkOut.input)
    #camRgb.preview.link(xlinkOut.input)

    # pass cropped rgb frame to preview xlinkout
    pre_mobilenet_manip_cropped_first.out.link(xlinkOut.input)




    # Define pose detection pre processing (resize preview to (self.pd_input_length, self.pd_input_length))
    print("Creating Pose Detection pre processing image manip CROP...")
    pre_pd_manip_cropped = pipeline.create(dai.node.ImageManip)
    pre_pd_manip_cropped.setMaxOutputFrameSize(cropped_frame_size[0] * cropped_frame_size[1] * 3)
    pre_pd_manip_cropped.inputConfig.setWaitForMessage(True)
    pre_pd_manip_cropped.inputImage.setQueueSize(1)
    pre_pd_manip_cropped.inputImage.setBlocking(False)
    script_node.outputs["to_pd_input"].link(pre_pd_manip_cropped.inputImage)
    #pre_mobilenet_manip_cropped_first.out.link(pre_pd_manip_cropped.inputImage)
    script_node.outputs['pre_pd_manip_cropped_cfg'].link(pre_pd_manip_cropped.inputConfig)

    print("Creating Pose Detection pre processing image manip RESIZE...")
    pre_pd_manip_resize = pipeline.create(dai.node.ImageManip)
    pre_pd_manip_resize.setMaxOutputFrameSize(pd_input_length * pd_input_length * 3)
    pre_pd_manip_resize.inputConfig.setWaitForMessage(True)
    pre_pd_manip_resize.inputImage.setQueueSize(1)
    pre_pd_manip_resize.inputImage.setBlocking(False)
    pre_pd_manip_cropped.out.link(pre_pd_manip_resize.inputImage)
    script_node.outputs['pre_pd_manip_resize_cfg'].link(pre_pd_manip_resize.inputConfig)

    # for debugging
    output_pre_pd = pipeline.create(dai.node.XLinkOut)
    output_pre_pd.setStreamName("pre_pd_manip")
    pre_pd_manip_resize.out.link(output_pre_pd.input)

    # Define pose detection model
    print("Creating Pose Detection Neural Network...")
    pd_nn = pipeline.create(dai.node.NeuralNetwork)
    pd_nn.setBlobPath(POSE_DETECTION_MODEL)
    # Increase threads for detection
    # pd_nn.setNumInferenceThreads(2)
    pre_pd_manip_resize.out.link(pd_nn.input)


    # Define pose detection post processing "model"
    print("Creating Pose Detection post processing Neural Network...")
    post_pd_nn = pipeline.create(dai.node.NeuralNetwork)
    post_pd_nn.setBlobPath(DETECTION_POSTPROCESSING_MODEL)
    pd_nn.out.link(post_pd_nn.input)
    post_pd_nn.out.link(script_node.inputs['from_post_pd_nn'])

    # tracker output
    objectTracker.out.link(script_node.inputs["tracker_out"])




    # Define link to send result to host
    manager_out = pipeline.create(dai.node.XLinkOut)
    manager_out.setStreamName("manager_out")
    script_node.outputs['host'].link(manager_out.input)

    # pre landmark manip rotated crop
    pre_lm_manip_rotated_crop = pipeline.create(dai.node.ImageManip)
    pre_lm_manip_rotated_crop.setMaxOutputFrameSize(cropped_frame_size[0] * cropped_frame_size[1] * 3)
    pre_lm_manip_rotated_crop.inputConfig.setWaitForMessage(True)
    pre_lm_manip_rotated_crop.inputImage.setQueueSize(1)
    pre_lm_manip_rotated_crop.inputImage.setBlocking(False)
    script_node.outputs["pre_lm_manip_rotated_crop_cfg"].link(pre_lm_manip_rotated_crop.inputConfig)
    script_node.outputs["lm_image"].link(pre_lm_manip_rotated_crop.inputImage)
    #pre_pd_manip_cropped.out.link(pre_lm_manip_rotated_crop.inputImage)


    # pre landmark manip resize
    pre_lm_manip_resize = pipeline.create(dai.node.ImageManip)
    pre_lm_manip_resize.setMaxOutputFrameSize(lm_input_size[0] * lm_input_size[1] * 3)
    pre_lm_manip_resize.inputConfig.setWaitForMessage(True)
    pre_lm_manip_resize.inputImage.setQueueSize(1)
    pre_lm_manip_resize.inputImage.setBlocking(False)
    script_node.outputs["pre_lm_manip_resize_cfg"].link(pre_lm_manip_resize.inputConfig)
    pre_lm_manip_rotated_crop.out.link(pre_lm_manip_resize.inputImage)




    print("Creating DiveideBy255 Neural Network...")
    divide_nn = pipeline.create(dai.node.NeuralNetwork)
    divide_nn.setBlobPath(DIVIDE_BY_255_MODEL)
    pre_lm_manip_resize.out.link(divide_nn.input)

    # landmark detection node
    lm_nn = pipeline.create(dai.node.NeuralNetwork)
    lm_nn.setBlobPath(lm_model)
    divide_nn.out.link(lm_nn.input)
    lm_nn.out.link(script_node.inputs['from_lm_nn'])

    # for debugging
    debug = pipeline.create(dai.node.XLinkOut)
    debug.setStreamName("xlinkout_lm_image")
    pre_lm_manip_resize.out.link(debug.input)



    return pipeline

def denormalize_bbox(x1, y1, x2, y2):
    # cropped frame size: 960 x 960
    scale_factor = 960 / mobilenet_input_size[1]
    offset_w = int(960 - mobilenet_input_size[0] * scale_factor) // 2
    def scale(point):
        return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)

    new_x1, new_y1 = scale([x1, y1])
    new_x2, new_y2 = scale([x2, y2])

    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

def lm_postprocess(body, lms, lms_world):
    lm_input_length = lm_input_size[0]
    nb_kps = 33
    # lms : landmarks sent by Manager script node to host (list of 39*5 elements for full body or 31*5 for upper body)
    lm_raw = np.array(lms).reshape(-1, 5)
    # Each keypoint have 5 information:
    # - X,Y coordinates are local to the body of
    # interest and range from [0.0, 255.0].
    # - Z coordinate is measured in "image pixels" like
    # the X and Y coordinates and represents the
    # distance relative to the plane of the subject's
    # hips, which is the origin of the Z axis. Negative
    # values are between the hips and the camera;
    # positive values are behind the hips. Z coordinate
    # scale is similar with X, Y scales but has different
    # nature as obtained not via human annotation, by
    # fitting synthetic data (GHUM model) to the 2D
    # annotation.
    # - Visibility, after user-applied sigmoid denotes the
    # probability that a keypoint is located within the
    # frame and not occluded by another bigger body
    # part or another object.
    # - Presence, after user-applied sigmoid denotes the
    # probability that a keypoint is located within the
    # frame.

    # Normalize x,y,z. Scaling in z = scaling in x = 1/self.lm_input_length
    lm_raw[:, :3] /= lm_input_length
    # Apply sigmoid on visibility and presence (if used later)
    body.visibility = 1 / (1 + np.exp(-lm_raw[:, 3]))
    body.presence = 1 / (1 + np.exp(-lm_raw[:, 4]))

    # body.norm_landmarks contains the normalized ([0:1]) 3D coordinates of landmarks in the square rotated body bounding box
    body.norm_landmarks = lm_raw[:, :3]
    # Now calculate body.landmarks = the landmarks in the image coordinate system (in pixel) (body.landmarks)
    src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
    dst = np.array([(x, y) for x, y in body.rect_points[1:]],
                   dtype=np.float32)  # body.rect_points[0] is left bottom point and points going clockwise!
    mat = cv2.getAffineTransform(src, dst)
    lm_xy = np.expand_dims(body.norm_landmarks[:nb_kps, :2], axis=0)
    lm_xy = np.squeeze(cv2.transform(lm_xy, mat))

    # A segment of length 1 in the coordinates system of body bounding box takes body.rect_w_a pixels in the
    # original image. Then we arbitrarily divide by 4 for a more realistic appearance.
    lm_z = body.norm_landmarks[:nb_kps, 2:3] * body.rect_w_a / 4
    lm_xyz = np.hstack((lm_xy, lm_z))

    # World landmarks are predicted in meters rather than in pixels of the image
    # and have origin in the middle of the hips rather than in the corner of the
    # pose image (cropped with given rectangle). Thus only rotation (but not scale
    # and translation) is applied to the landmarks to transform them back to
    # original  coordinates.
    body.landmarks_world = np.array(lms_world).reshape(-1, 3)
    sin_rot = sin(body.rotation)
    cos_rot = cos(body.rotation)
    rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
    body.landmarks_world[:, :2] = np.dot(body.landmarks_world[:, :2], rot_m)

    # if self.smoothing:
    #     timestamp = now()
    #     object_scale = body.rect_w_a
    #     lm_xyz[:self.nb_kps] = self.filter_landmarks.apply(lm_xyz[:self.nb_kps], timestamp, object_scale)
    #     lm_xyz[self.nb_kps:] = self.filter_landmarks_aux.apply(lm_xyz[self.nb_kps:], timestamp, object_scale)
    #     body.landmarks_world = self.filter_landmarks_world.apply(body.landmarks_world, timestamp)

    body.landmarks = lm_xyz.astype(np.int32)

# preprocessing
pipeline = create_pipeline(lm_model=LANDMARK_MODEL_LITE, crop=False)

render = BlazeposeRenderer.BlazeposeRenderer()
# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    preview = device.getOutputQueue("preview", 4, False)
    manager_out = device.getOutputQueue(name="manager_out", maxSize=1, blocking=False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    frame = None

    # for debugging
    debug_lm_image = device.getOutputQueue(name="xlinkout_lm_image", maxSize=1, blocking=False)
    pre_pd_image = device.getOutputQueue(name="pre_pd_manip", maxSize=1, blocking=False)

    while(True):
        imgFrame = preview.get()
        frame = imgFrame.getCvFrame()
        tracker_out = marshal.loads(manager_out.get().getData())



        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        color = (255, 0, 0)

        print(tracker_out["type"])
        if tracker_out["type"] == 1:
            for person_id, state in tracker_out["people_state"].items():
                x1 = state["topLeft_x"]
                y1 = state["topLeft_y"]
                x2 = state["bottomRight_x"]
                y2 = state["bottomRight_y"]

                # rescale bbox
                x1, y1, x2, y2 = denormalize_bbox(x1, y1, x2, y2)

                #cv2.putText(frame, "Person", (x1+ 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                #cv2.putText(frame, f"ID: {[person_id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                #cv2.putText(frame, state["status"], (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
                # for debugging
                pd_image = pre_pd_image.get().getCvFrame()
                lm_image = debug_lm_image.get()
                lm_image = lm_image.getCvFrame()

                print(state.get("lms"))
                if state.get("lms") is not None:


                    scale_factor = 244 / 256
                    offset_w = int(244 - 256 * scale_factor) // 2

                    body = utils.Body()
                    body.rect_x_center_a = state["sqn_rr_center_x"] * 224 + offset_w
                    body.rect_y_center_a = state["sqn_rr_center_y"] * 224
                    body.rect_w_a = state["sqn_rr_size"] * 224 + offset_w
                    body.rect_h_a = state["sqn_rr_size"] * 224
                    body.rotation = state["rotation"]
                    body.rect_points = utils.rotated_rect_to_points(body.rect_x_center_a, body.rect_y_center_a, body.rect_w_a,
                                                                    body.rect_h_a, body.rotation)
                    body.lm_score = state["lm_score"]
                    lm_postprocess(body, state['lms'], state['lms_world'])
                    pd_image = render.draw(pd_image, body)


                #  for debugging
                cv2.imshow("lm_image" + str(person_id), lm_image)
                cv2.imshow("pd_image" + str(person_id), pd_image)

            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (4, frame.shape[0] - 50), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

            frame = cv2.resize(frame, (700, 600))
            cv2.imshow("tracker", frame)

        if cv2.waitKey(1) == ord('q'):
            break
