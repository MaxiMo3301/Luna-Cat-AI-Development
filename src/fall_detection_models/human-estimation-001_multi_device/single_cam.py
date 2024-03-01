# import module

import depthai as dai
import blobconverter
from pathlib import Path
import numpy as np
import contextlib
import cv2
import time
import argparse
from FPS import FPS
import threading
from fall_detection import fall_detect, falling_alarm, face_bbox
from pose import getKeypoints, getValidPairs, getPersonwiseKeypoints

# define the color of the body part pair
colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

# define the body part pair
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

# define the size of neural network
nn_input_size = (456, 256)

# create list 
decode_thread_list = []

running = True # run decode while cam is on

def decode_thread(in_queue, id, dict_out):
    #counter = 0
    while running:
        #counter += 1
        try:
            raw_in = in_queue.get()
        except RuntimeError:
            return
        heatmaps = np.array(raw_in.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
        pafs = np.array(raw_in.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
        heatmaps = heatmaps.astype('float32')
        pafs = pafs.astype('float32')
        outputs = np.concatenate((heatmaps, pafs), axis=1)

        new_keypoints = []
        new_keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        for row in range(18):
            probMap = outputs[0, row, :, :]
            probMap = cv2.resize(probMap, (456, 256))  # (456, 256)
            keypoints = getKeypoints(probMap, 0.3) # original 0.3
            new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
            keypoints_with_id = []

            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoint_id += 1

            new_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = getValidPairs(outputs, w=456, h=256, detected_keypoints=new_keypoints)
        newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

        dict_out["out" + id][1] = (new_keypoints, new_keypoints_list, newPersonwiseKeypoints)


# Get and detected the human body joint and store the data
def get_person_keypoint(frame, personwiseKeypoints, keypoints_list, detected_keypoints) -> list:
    # get keypoints from personwiseKeypoints and store them in keypoint_for_detection
    # all keypoints of each person is stored in an entry of keypoint_for_detection
    if keypoints_list is None or \
            detected_keypoints is None or \
            personwiseKeypoints is None:
        return None
    scale_factor = frame.shape[0] / nn_input_size[1]
    offset_w = int(frame.shape[1] - nn_input_size[0] * scale_factor) // 2

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

# Display the keypoint of the detected coordinate
def show(frame, keypoints_list, detected_keypoints, personwiseKeypoints) -> None:
    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
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

# Create the pipeline and neural network for the camera
def create_pipeline(img_w, img_h, fps):
    # using 2 devices: 6 fps , 1 device: 8 fps

    print("Creating pipeline...")
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
    print("Creating Color Camera...")
    cam = pipeline.createColorCamera()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    #cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam.setInterleaved(False)
    cam.setFps(fps)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setVideoSize(img_w, img_h)
    cam.setPreviewSize(img_w, img_h)

    cam_out = pipeline.createXLinkOut()
    cam_out.setStreamName("cam_out")
    cam_out.input.setQueueSize(1)
    cam_out.input.setBlocking(False)
    cam.video.link(cam_out.input)

    # Define pose detection pre processing (resize preview to (self.nn_input_length, self.nn_input_length))
    print("Creating Pose Detection pre processing image manip...")
    pre_nn_manip = pipeline.create(dai.node.ImageManip)
    pre_nn_manip.setMaxOutputFrameSize(nn_input_size[0] * nn_input_size[1] * 3)
    pre_nn_manip.inputConfig.setWaitForMessage(True)
    pre_nn_manip.inputImage.setQueueSize(1)
    pre_nn_manip.inputImage.setBlocking(False)
    cam.preview.link(pre_nn_manip.inputImage)

    pre_nn_manip_cfg_in = pipeline.create(dai.node.XLinkIn)
    pre_nn_manip_cfg_in.setStreamName("pre_nn_manip_cfg")
    pre_nn_manip_cfg_in.out.link(pre_nn_manip.inputConfig)

    # Define pose detection model
    print("Creating Pose Detection Neural Network...")
    nn_nn = pipeline.createNeuralNetwork()
    nn_nn.setBlobPath(Path(blob_path))
    pre_nn_manip.out.link(nn_nn.input)

    # Pose detection output
    nn_out = pipeline.createXLinkOut()
    nn_out.setStreamName("nn_out")
    nn_nn.out.link(nn_out.input)


    return pipeline

# Display The Detected Device
def worker(dev_info, stack, dic_out) -> None:
    #openvino_version = dai.OpenVINO.Version.VERSION_2021_4
    #device: dai.Device = stack.enter_context(dai.Device(openvino_version, dev_info, False))
    device: dai.Device = stack.enter_context(dai.Device(dev_info))

    # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
    print("=== Connected to " + dev_info.getMxId())
    mxid = device.getMxId()
    cameras = device.getConnectedCameras()
    usb_speed = device.getUsbSpeed()
    print("   >>> MXID:", mxid)
    print("   >>> Cameras:", *[c.name for c in cameras])
    print("   >>> USB speed:", usb_speed.name)

    device.startPipeline(create_pipeline(args.width, args.height, args.fps))
    pre_nn_manip_cfg = device.getInputQueue(name="pre_nn_manip_cfg")
    pre_nn_manip_cfg_list.append(pre_nn_manip_cfg)

    dic_out["out" + mxid] = [device.getOutputQueue(name="cam_out"), (None, None, None)]  # None for storing decode result

    t = threading.Thread(target=decode_thread, args=(device.getOutputQueue(name="nn_out"), mxid, dic_out))
    t.start()
    decode_thread_list.append(t)



# Funcation to blur the face of the detected body
def blur_face(frame, body, black) -> None:
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


# Main Function
if __name__ == "__main__":

    # Set the path of the detecttion module
    blob_path = blobconverter.from_zoo(name="human-pose-estimation-0001", shaves=6)

    # Set the bassic parametor of the display frame settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=900, help="height of internal camera")
    parser.add_argument("--width", type=int, default=1600, help="width of internal camera")
    parser.add_argument("--fps", type=int, default=6, help="fps of internal camera")
    parser.add_argument('-b', '--black', action="store_true", help="only show skeleton in black mode") # skeleton only mode

    args = parser.parse_args()
    cfg_pre_nn = dai.ImageManipConfig()
    cfg_pre_nn.setResizeThumbnail(nn_input_size[0], nn_input_size[1])
    pre_nn_manip_cfg_list = []

    # get and print the detected camera
    device_infos = dai.Device.getAllAvailableDevices()
    print(f'Found {len(device_infos)} devices')

    fps_calculate = FPS(mean_nb_frames=10)

    with contextlib.ExitStack() as stack:
        queues_out = {}
        threads = []
        for dev in device_infos:
            time.sleep(1) # Currently required due to XLink race issues
            thread = threading.Thread(target=worker, args=(dev, stack, queues_out))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join() # Wait for all threads to finish

        while True:
            fps_calculate.update()

            for cfg in pre_nn_manip_cfg_list:
                cfg.send(cfg_pre_nn)
            for index, (name, (rgb, nn_out_decode)) in enumerate(queues_out.items()):
                timestamp = time.perf_counter()

                if rgb.has():
                    frame = rgb.get().getCvFrame()
                    if args.black:
                        shape = (args.height, args.width, 3)
                        frame = np.zeros(shape)
                    if nn_out_decode[0] is not None and nn_out_decode[1] is not None and nn_out_decode[2] is not None:
                        newPersonwiseKeypoints = nn_out_decode[2]
                        new_keypoints_list = nn_out_decode[1]
                        new_keypoints = nn_out_decode[0]

                        keypoint_for_detection = \
                            get_person_keypoint(frame, newPersonwiseKeypoints, new_keypoints_list, new_keypoints)

                        #  fall detection by keypoints
                        if keypoint_for_detection:
                            any_alarm = False
                            for h in range(len(keypoint_for_detection)):
                                alarm, bbox = fall_detect(keypoint_for_detection[h])
                                blur_face(frame=frame, body=keypoint_for_detection[h], black=args.black)
                                if alarm:
                                    falling_alarm(frame, bbox)
                                    any_alarm = True
                            print(any_alarm)

                        show(frame, new_keypoints_list, new_keypoints, newPersonwiseKeypoints)

                    #  show frame
                    cv2.imshow(name, frame)

            if cv2.waitKey(1) == ord('q'):
                running = False
                break

    for thread in decode_thread_list:
        thread.join()
    print('Devices closed')
    print(f"FPS : {fps_calculate.get_global():.1f} f/s (# frames = {fps_calculate.nbf})")