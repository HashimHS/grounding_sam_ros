import rospy
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.inference import Model
import cv2
import torch

from grounding_sam_ros.srv import VitDetection, VitDetectionResponse
from cv_bridge import CvBridge
from std_msgs.msg import MultiArrayDimension

# SAM
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import supervision as sv

import os

LINKS = {
    "SAM-H": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "SAM-L": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "SAM-B": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "DINO": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
}
MODELS = {
    "SAM-H": "vit_h",
    "SAM-L": "vit_l",
    "SAM-B": "vit_b",
}


class VitDetectionServer(object):
    def __init__(self, model_path, config, sam_checkpoint, sam_model, box_threshold=0.35, text_threshold=0.25):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():    
            self.device = torch.device("cuda")
        else:
            print("No GPU available")
            exit()

        rospy.loginfo("Loading models...")

        # Building GroundingDINO inference model
        if not os.path.exists(model_path):
            rospy.loginfo("Downloading DINO model...")
            if not os.path.exists("weights"):
                os.makedirs("weights")
            os.system("wget {} -O {}".format(LINKS["DINO"], model_path))
        self.grounding_dino_model = Model(model_config_path=config, model_checkpoint_path=model_path)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Building SAM inference model
        if not os.path.exists(sam_checkpoint):
            rospy.loginfo("Downloading SAM model...")
            if not os.path.exists("weights"):
                os.makedirs("weights")
            os.system("wget {} -O {}".format(LINKS[sam_model], sam_checkpoint))
        self.sam_predictor = SamPredictor(sam_model_registry[MODELS[sam_model]](checkpoint=sam_checkpoint).to(self.device))

        rospy.loginfo("Models are loaded")

        # ros service
        self.cv_bridge = CvBridge()
        rospy.Service("vit_detection", VitDetection, self.callback)
        rospy.loginfo("vit_detection service has started")

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def detect(self, image, text):

        # labels = text.split(',')

        # GroundingDINO Model
        # detect objects
        detections, labels = self.grounding_dino_model.predict_with_caption(
                    image=image,
                    caption=text,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold
                )

        # Segment Anything Model
        # convert detections to masks
        detections.mask = self.segment(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        # mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        result = [
            f"{labels[class_id]}:{confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        
        # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        return detections, labels, annotated_image

    def callback(self, request):
        img, prompt = request.color_image, request.prompt
        img = self.cv_bridge.imgmsg_to_cv2(img)
        detections, labels, annotated_frame = self.detect(img, prompt)
        boxes = detections.xyxy
        scores = detections.confidence
        masks = detections.mask

        rospy.loginfo("Detected objects: {}".format(labels))
        rospy.loginfo("Detection scores: {}".format(scores))
        stride = masks.shape[1] * masks.shape[2]
        response = VitDetectionResponse()
        response.labels = labels
        response.class_id = detections.class_id
        response.scores = scores.tolist()
        response.boxes.layout.dim = [MultiArrayDimension(label="boxes", size=boxes.shape[0], stride=4)]
        response.boxes.data = boxes.flatten().tolist()
        response.annotated_frame = self.cv_bridge.cv2_to_imgmsg(annotated_frame)
        response.segmasks.layout.dim = [MultiArrayDimension(label="masks", size=masks.shape[0], stride=stride)]
        response.segmasks.data = masks.flatten().tolist()

        

        # We release the gpu memory
        torch.cuda.empty_cache()
        
        return response
    
if __name__ == '__main__':
    rospy.init_node('grounding_sam_ros')

    # get arguments from the ros parameter server
    model_path = rospy.get_param('~model_path')
    config = rospy.get_param('~config')
    sam_checkpoint = rospy.get_param('~sam_checkpoint')
    sam_model = rospy.get_param('~sam_model')
    box_threshold = rospy.get_param('~box_threshold')
    text_threshold = rospy.get_param('~text_threshold')

    # start the server
    VitDetectionServer(model_path, config, sam_checkpoint, sam_model, box_threshold, text_threshold)
    rospy.spin()