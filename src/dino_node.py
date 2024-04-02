#! /usr/bin/env python
import rospy
from groundingdino.util.inference import load_model, load_image, predict, annotate
# from GroundingDINO.groundingdino.util.inference import Model
import cv2
import torch

from grounding_sam_ros.srv import VitDetection, VitDetectionResponse
from cv_bridge import CvBridge
from std_msgs.msg import MultiArrayDimension
from groundingdino.util.inference import box_convert
import os

class VitDetectionServer(object):
    def __init__(self, model_path, config, box_threshold=0.35, text_threshold=0.25, save=False, save_path="./annotated"):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():    
            self.device = torch.device("cuda")
        else:
            print("No GPU available")
            exit()
        rospy.loginfo("Loading model...")
        self.model = load_model(config, model_path)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.cv_bridge = CvBridge()
        self.save_path = save_path
        rospy.loginfo("Model loaded")

        # ros service
        rospy.Service("vit_detection", VitDetection, self.callback)
        rospy.loginfo("Start vit_detection service")

        self.i = 1
        for files in os.listdir(save_path):
            if files.endswith('.jpg'):
                self.i += 1

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
        if save:
            path = os.path.join(self.save_path, "annotated_frame_{}.jpg".format(self.i))
            cv2.imwrite(path, annotated_frame)
            path = os.path.join(self.save_path, "rgb_{}.jpg".format(self.i))
            rgb = cv2.imread(image_path)
            cv2.imwrite(path, rgb)
            self.i += 1


        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        return annotated_frame, xyxy, logits, phrases
        

    def callback(self, request):
        img, prompt = request.color_image, request.prompt
        img = self.cv_bridge.imgmsg_to_cv2(img)
        annotated_frame, boxes, logits, labels = self.detect(img, prompt)
        
        # Get prediction scores
        # scores = torch.sigmoid(logits.values).cpu().detach().numpy()
        scores = logits.cpu().detach().numpy()
        print(scores)


        rospy.loginfo("Detected objects: {}".format(labels))

        response = VitDetectionResponse()
        response.labels = labels
        response.scores = scores.tolist()
        response.boxes.layout.dim = [MultiArrayDimension(label="boxes", size=boxes.shape[0], stride=4)]
        response.boxes.data = boxes.flatten().tolist()
        response.annotated_frame = self.cv_bridge.cv2_to_imgmsg(annotated_frame)
        response.segmask = self.cv_bridge.cv2_to_imgmsg(annotated_frame)

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
    save = rospy.get_param('~save')
    save_path = rospy.get_param('~save_path')

    # start the server
    VitDetectionServer(model_path, config, box_threshold, text_threshold, save=False, save_path=save_path)
    rospy.spin()