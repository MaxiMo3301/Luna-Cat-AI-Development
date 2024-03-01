import depthai as dai
import blobconverter
from pathlib import Path
import numpy as np
from pose import getKeypoints, getValidPairs, getPersonwiseKeypoints
import cv2
import utils
from string import Template

SCRIPT_DIR = Path(__file__).resolve().parent
blob_path = blobconverter.from_zoo(name="human-pose-estimation-0001", shaves=6)
mobile_net_PathDefault = str((SCRIPT_DIR / Path('mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
TEMPLATE_MANAGER_SCRIPT = str(SCRIPT_DIR / "script_template.py")
nn_input_size = (456, 256)
mobilenet_input_size = (300, 300)

resolution = (1920, 1080)
class DepthaiPose:
    def __init__(self, internal_height=1080, fps=3):
        """
        - fps: set internal camera fps
        - internal_height: height of rgb preview
        """
        width, self.scale_nd = utils.find_isp_scale_params(internal_height * 1920 / 1080, is_height=False)
        self.img_h = int(round(resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
        self.img_w = int(round(resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
        self.pad_h = (self.img_w - self.img_h) // 2
        self.pad_w = 0
        self.frame_size = self.img_w
        self.crop_w = 0
        self.trace = False
        self.fps = fps
        self.keypoints = None
        self.keypoints_list = None
        self.PersonwiseKeypoints = None

    def create_pipeline(self):
        """
        Create two neural network pose detection NN and person tracker NN,
        and pass the inference results of this two NN to the host
        """

        # Create pipeline
        pipeline = dai.Pipeline()

        # script node
        script_node = pipeline.create(dai.node.Script)
        script_node.setScript(self.create_script())


        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        detectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
        objectTracker = pipeline.create(dai.node.ObjectTracker)

        xlinkOut_preview = pipeline.create(dai.node.XLinkOut)
        trackerOut = pipeline.create(dai.node.XLinkOut)

        xlinkOut_preview.setStreamName("preview")
        trackerOut.setStreamName("tracklets")

        # Properties
        camRgb.setPreviewSize(self.img_w, self.img_h)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setIspScale(self.scale_nd[0], self.scale_nd[1])
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        camRgb.setFps(self.fps)

        # pre mobile net manip
        pre_mobile_manip_set_frame_type = pipeline.create(dai.node.ImageManip)
        pre_mobile_manip_set_frame_type.setMaxOutputFrameSize(self.img_w * self.img_h * 3)
        pre_mobile_manip_set_frame_type.inputConfig.setWaitForMessage(False)
        pre_mobile_manip_set_frame_type.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        pre_mobile_manip_set_frame_type.inputImage.setQueueSize(1)
        pre_mobile_manip_set_frame_type.inputImage.setBlocking(False)
        camRgb.preview.link(pre_mobile_manip_set_frame_type.inputImage)

        pre_mobile_manip_resize = pipeline.create(dai.node.ImageManip)
        pre_mobile_manip_resize.setMaxOutputFrameSize(mobilenet_input_size[0] * mobilenet_input_size[1] * 3)
        pre_mobile_manip_resize.inputConfig.setWaitForMessage(True)
        pre_mobile_manip_resize.inputImage.setQueueSize(1)
        pre_mobile_manip_resize.inputImage.setBlocking(False)
        pre_mobile_manip_set_frame_type.out.link(pre_mobile_manip_resize.inputImage)
        script_node.outputs["pre_mobile_manip_resize"].link(pre_mobile_manip_resize.inputConfig)


        # testing MobileNet DetectionNetwork
        detectionNetwork.setBlobPath(mobile_net_PathDefault)
        detectionNetwork.setConfidenceThreshold(0.45)
        detectionNetwork.input.setBlocking(False)

        objectTracker.setDetectionLabelsToTrack([15])  # track only person
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        # Linking
        pre_mobile_manip_resize.out.link(detectionNetwork.input)
        camRgb.preview.link(xlinkOut_preview.input)

        detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
        detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        detectionNetwork.out.link(objectTracker.inputDetections)
        objectTracker.out.link(script_node.inputs["from_tracker"])

        # Define link to send result to host
        manager_out = pipeline.create(dai.node.XLinkOut)
        manager_out.setStreamName("manager_out")
        script_node.outputs['host'].link(manager_out.input)

        # gen2_pose_image_manipulation
        pre_pose_manip_resize = pipeline.create(dai.node.ImageManip)
        pre_pose_manip_resize.setMaxOutputFrameSize(nn_input_size[0] * nn_input_size[1] * 3)
        pre_pose_manip_resize.inputConfig.setWaitForMessage(True)
        pre_pose_manip_resize.inputImage.setQueueSize(1)
        pre_pose_manip_resize.inputImage.setBlocking(False)
        camRgb.preview.link(pre_pose_manip_resize.inputImage)
        script_node.outputs["to_pre_pose_manip_resize"].link(pre_pose_manip_resize.inputConfig)

        # Define pose detection model
        print("Creating Pose Detection Neural Network...")
        nn_nn = pipeline.createNeuralNetwork()
        nn_nn.setBlobPath(Path(blob_path))
        pre_pose_manip_resize.out.link(nn_nn.input)

        nn_nn.out.link(script_node.inputs["from_pose_detection"])

        return pipeline

    def decode_pose_result(self, heatmaps_raw, pafs_raw):
        """
        Update self.keypoints, self.keypoints_list, self.PersonwiseKeypoints
        by decoding heatmaps and pafs
        """

        heatmaps = np.array(heatmaps_raw).reshape((1, 19, 32, 57))
        pafs = np.array(pafs_raw).reshape((1, 38, 32, 57))
        heatmaps = heatmaps.astype('float32')
        pafs = pafs.astype('float32')
        outputs = np.concatenate((heatmaps, pafs), axis=1)

        new_keypoints = []
        new_keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        for row in range(18):
            probMap = outputs[0, row, :, :]
            probMap = cv2.resize(probMap, (nn_input_size[0], nn_input_size[1]))  # (456, 256)
            keypoints = getKeypoints(probMap, 0.3)  # original 0.3
            new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
            keypoints_with_id = []

            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoint_id += 1

            new_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = getValidPairs(outputs, w=nn_input_size[0],
                                                   h=nn_input_size[1], detected_keypoints=new_keypoints)
        newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

        self.keypoints = new_keypoints
        self.keypoints_list = new_keypoints_list
        self.PersonwiseKeypoints = newPersonwiseKeypoints

    def denormalize_bbox(self, x1, y1, x2, y2, input_size_w, input_size_h):
        """
        :param x1: x-coordinate of topLeft
        :param y1: y-coordinate of topLeft
        :param x2: x-coordinate of bottomRight
        :param y2: y-coordinate of bottomRight
        :param input_size_w: input size of NN output size
        :param input_size_h: input size of NN output size
        :return: Coordinates of topLeft, bottomRight of the bounding box on rgb preview
        """
        scale_factor = self.img_w / input_size_h
        offset_w = int(self.img_h - input_size_w * scale_factor) // 2

        def scale(point):
            return point[0] * scale_factor + offset_w, point[1] * scale_factor

        new_x1, new_y1 = scale([x1, y1])
        new_x2, new_y2 = scale([x2, y2])
        new_y1 = new_y1 - self.pad_h
        new_y2 = new_y2 - self.pad_h
        new_x2 = new_x2 + self.pad_h
        new_x1 = new_x1 + self.pad_h

        return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

    def create_script(self):
        # Read the template
        with open(TEMPLATE_MANAGER_SCRIPT, 'r') as file:
            template = Template(file.read())

        # Perform the substitution
        code = template.substitute(
            _TRACE="node.warn" if self.trace else "#",
            _pad_h=self.pad_h,
            _img_h=self.img_h,
            _img_w=self.img_w
        )
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub('\n\s*\n', '\n', code)

        return code




