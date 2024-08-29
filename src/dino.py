import rospy
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.inference import Model
import cv2
import torch
import numpy as np

from grounding_sam_ros.srv import VitDetection, VitDetectionResponse
from cv_bridge import CvBridge
from std_msgs.msg import MultiArrayDimension
from groundingdino.util.inference import box_convert
import os

LINKS = {"DINO": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"}

class VitDetectionServer(object):
    def __init__(self, model_path, config, box_threshold=0.35, text_threshold=0.25):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():    
            self.device = torch.device("cuda")
        else:
            print("No GPU available - using CPU")
            self.device = torch.device("cpu") 

        rospy.loginfo("Loading model...")
        if not os.path.exists(model_path):
            rospy.loginfo("Downloading DINO model...")
            if not os.path.exists("weights"):
                os.makedirs("weights")
            os.system("wget {} -O {}".format(LINKS["DINO"], model_path))
        self.model = load_model(config, model_path)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        rospy.loginfo("Model loaded")

        # ros service
        self.cv_bridge = CvBridge()
        rospy.Service("vit_detection", VitDetection, self.callback)
        rospy.loginfo("vit_detection service has started")

    def detect(self, image, text):
        cv2.imwrite("image.jpg", image)
        image_path = "image.jpg"
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=self.model, 
            image=image, 
            caption=text, 
            box_threshold=0.35, 
            text_threshold=0.25
        )
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        masks = np.zeros((len(boxes), image_source.shape[0], image_source.shape[1]), dtype=np.uint8)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            masks[i, int(y1):int(y2), int(x1):int(x2)] = 255

        return annotated_frame, xyxy, logits, phrases, masks
        

    def callback(self, request):
        img, prompt = request.color_image, request.prompt
        img = self.cv_bridge.imgmsg_to_cv2(img)
        annotated_frame, boxes, logits, labels, masks = self.detect(img, prompt)
        scores = logits.cpu().detach().numpy()

        rospy.loginfo("Detected objects: {}".format(labels))

        response = VitDetectionResponse()
        response.labels = labels
        response.scores = scores.tolist()
        response.boxes.layout.dim = [MultiArrayDimension(label="boxes", size=boxes.shape[0], stride=4)]
        response.boxes.data = boxes.flatten().tolist()
        response.annotated_frame = self.cv_bridge.cv2_to_imgmsg(annotated_frame)
        response.segmasks.layout.dim = [MultiArrayDimension(label="masks", size=masks.shape[0], stride=masks.shape[1] * masks.shape[2])]
        response.segmasks.data = masks.flatten().tolist()

        # We release the gpu memory
        torch.cuda.empty_cache()
        
        return response
    
if __name__ == '__main__':
    rospy.init_node('grounding_sam_ros')

    # get arguments from the ros parameter server
    model_path = rospy.get_param('~model_path')
    config = rospy.get_param('~config')
    box_threshold = rospy.get_param('~box_threshold')
    text_threshold = rospy.get_param('~text_threshold')

    # start the server
    VitDetectionServer(model_path, config, box_threshold, text_threshold)
    rospy.spin()