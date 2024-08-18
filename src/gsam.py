import rospy
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.inference import Model
import cv2
import torch

from grounding_sam_ros.srv import VitDetection, VitDetectionResponse
from cv_bridge import CvBridge
from std_msgs.msg import MultiArrayDimension

# SAM
from segment_anything import build_sam_vit_l, SamPredictor, sam_model_registry
import numpy as np
import supervision as sv

import os

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

class VitDetectionServer(object):
    def __init__(self, model_path, sam_checkpoint, config, box_threshold=0.35, text_threshold=0.25, save=False):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():    
            self.device = torch.device("cuda")
        else:
            print("No GPU available")
            exit()
        rospy.loginfo("Loading model...")
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=config, model_checkpoint_path=model_path)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.cv_bridge = CvBridge()
        rospy.loginfo("Model loaded")

        # ros service
        rospy.Service("vit_detection", VitDetection, self.callback)
        rospy.loginfo("Start vit_detection service")

        # 'sam_vit_l_0b3195'
        self.sam_predictor = SamPredictor(sam_model_registry['vit_l'](checkpoint=sam_checkpoint).to(self.device))

        self.i = 1

        try:
            # Check if the directory exists
            if not os.path.exists('./annotated'):
                raise FileNotFoundError("The directory 'annotated' does not exist.")

            # List files in the directory if it exists
            for files in os.listdir('./annotated'):
                # Your code to handle each file
                if files.endswith('.jpg'):
                    self.i += 1
                    # print(files)

        except FileNotFoundError as e:
            print(e)

    def detect(self, image, text):

        # labels = text.split(',')

        # GroundingDINO Model
        # detect objects
        # image_source, image = load_image(image_path)
        # detections = self.grounding_dino_model.predict_with_classes(
        #     image=image,
        #     classes=labels,
        #     box_threshold=self.box_threshold,
        #     text_threshold=self.text_threshold
        # )

        detections, labels = self.grounding_dino_model.predict_with_caption(
                    image=image,
                    caption=text,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold
                )

        # Segment Anything Model
        # convert detections to masks
        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{labels[class_id]}:{confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        if save:
            path = "./annotated/annotated_frame_{}.jpg".format(self.i)
            cv2.imwrite(path, annotated_image)
            self.i += 1


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
    sam_checkpoint = rospy.get_param('~sam_checkpoint')
    config = rospy.get_param('~config')
    box_threshold = rospy.get_param('~box_threshold')
    text_threshold = rospy.get_param('~text_threshold')
    save = rospy.get_param('~save')

    # start the server
    VitDetectionServer(model_path, sam_checkpoint, config, box_threshold, text_threshold)
    rospy.spin()