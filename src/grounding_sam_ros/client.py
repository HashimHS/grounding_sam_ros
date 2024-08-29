import rospy
from cv_bridge import CvBridge
from grounding_sam_ros.srv import VitDetection, VitDetectionResponse
import cv2
import numpy as np

class SamDetector():
    """ 
    Client for the vit_detection service
    Args:
        - rgb: RGB image in cv2 format
        - text: Description of the object to be detected

    Returns:
        - annotated_frame: Annotated image from Grounding DINO
        - boxes: Bounding boxes in y1 x1 y2 x2 format
        - masks: Segmentation masks
        - labels: List of detected objects
        - scores: Confidence scores of the detected objects
    """
    def __init__(self):
        self.cv_bridge = CvBridge()
        rospy.loginfo("Checking the vit_detection service")
        rospy.wait_for_service('vit_detection')
        self.vit_detection = rospy.ServiceProxy('vit_detection', VitDetection)
        rospy.loginfo("vit_detection service is ready")

    def detect(self, rgb, text):
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(np.array(rgb))
        results = self.vit_detection(rgb_msg, text)
        rospy.loginfo('Detected Objects: {}' .format(results.labels))
        labels = results.labels
        scores = results.scores
        boxes = results.boxes
        masks = self.maskmsg_to_array(results.segmasks, rgb.shape[:2])
        boxes = np.array(results.boxes.data).reshape((results.boxes.layout.dim[0].size, results.boxes.layout.dim[0].stride))
        annotated_frame = self.cv_bridge.imgmsg_to_cv2(results.annotated_frame)
        return annotated_frame, boxes, masks, labels, scores
    
    def maskmsg_to_array(self, mask_msg, mask_shape):
        masks = np.array(mask_msg.data).reshape((mask_msg.layout.dim[0].size, mask_msg.layout.dim[0].stride))
        masks = masks.reshape((masks.shape[0], mask_shape[0], mask_shape[1])).astype(np.uint8)
        return masks