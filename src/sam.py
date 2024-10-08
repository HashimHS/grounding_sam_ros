import rospy
from groundingdino.util.inference import load_model, load_image, predict, annotate, box_convert
from groundingdino.util import box_ops
import cv2
import torch

from grounding_sam_ros.srv import VitDetection, VitDetectionResponse
from cv_bridge import CvBridge
from std_msgs.msg import MultiArrayDimension

# SAM
from segment_anything import build_sam_vit_l, SamPredictor, sam_model_registry, SamAutomaticMaskGenerator, build_sam_vit_h, build_sam_vit_b
import numpy as np

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
            # self.device = torch.device("cpu") 
            print("No GPU available")
            exit()

        rospy.loginfo("Loading model...")
        if not os.path.exists(model_path):
            rospy.loginfo("Downloading DINO model...")
            os.system("wget {} -O {}".format(LINKS["DINO"], model_path))
        self.model = load_model(config, model_path)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        if not os.path.exists(sam_checkpoint):
            rospy.loginfo("Downloading SAM model...")
            if not os.path.exists("weights"):
                os.makedirs("weights")
            os.system("wget {} -O {}".format(LINKS[sam_model], sam_checkpoint))       
        # self.sam_predictor = SamPredictor(sam_model_registry[MODELS[sam_model]](checkpoint=sam_checkpoint).to(self.device))
        self.mask_generator = SamAutomaticMaskGenerator(sam_model_registry[MODELS[sam_model]](checkpoint=sam_checkpoint).to(self.device))

        rospy.loginfo("Model loaded")

        # ros service
        self.cv_bridge = CvBridge()
        rospy.Service("vit_detection", VitDetection, self.callback)
        rospy.loginfo("Start vit_detection service")

    def crop(self, image, box):
        ''' 
        Crop the image to the box
        Args:
            image: the image to crop
            box: the box to crop the image in the format [x1, y1, x2, y2]
        '''
        x1, y1, x2, y2 = [round(i) for i in box]
        return image[y1:y2,x1:x2]

    def segmentation_map(self, image):
        '''
        Args:
            image: the image to segment

        Returns:
            segmap: a numpy array with the same shape as the image where each pixel is the number of the mask it belongs to        
        '''
        masks = self.mask_generator.generate(image)
        segmap = np.zeros_like(masks[0]['segmentation'], dtype=np.uint8)
        for i in range(len(masks)):
            segmap[masks[i]['segmentation'] == 1] = i + 1
        return segmap


    def detect(self, image, text):
        '''
        Args:
            image: the image to detect objects in
            text: the text prompt

        Returns:
            annotated_frame: the image with the detected objects annotated
            boxes: the bounding boxes of the detected objects
            logits: the logits of the detected objects
            phrases: the generated labels of the detected objects
            segmaps: list of segmentation maps for each box
        '''

        self.i = 1
        cv2.imwrite("image.jpg", image)
        image_path = "image.jpg"

        # GroundingDINO Model
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=self.model, 
            image=image, 
            caption=text, 
            box_threshold=self.box_threshold, 
            text_threshold=self.text_threshold
        )
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        
        # Segment Anything Model

        # Box prompted segment
        h, w, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(self.device)
        # self.sam_predictor.set_image(image_source)
        # masks, _, _ = self.sam_predictor.predict_torch(
        #     point_coords = None,
        #     point_labels = None,
        #     boxes = transformed_boxes,
        #     multimask_output = False,
        # )

        # Cropped Image Segmentations
        segmaps = np.zeros((len(boxes), image_source.shape[0], image_source.shape[1]), dtype=np.uint8)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [round(i) for i in xyxy[i]]
            cropped_image = self.crop(image_source, xyxy[i])
            segmap = self.segmentation_map(cropped_image)
            segmaps[i][y1:y2, x1:x2] = segmap
        
        return annotated_frame, xyxy, logits, phrases, segmaps

    def callback(self, request):
        img, prompt = request.color_image, request.prompt
        img = self.cv_bridge.imgmsg_to_cv2(img)
        annotated_frame, boxes, logits, labels, segmaps = self.detect(img, prompt)

        # Get prediction scores
        scores = logits.cpu().detach().numpy()
        print(scores)

        rospy.loginfo("Detected objects: {}".format(labels))

        response = VitDetectionResponse()
        response.labels = labels
        response.scores = scores.tolist()
        response.boxes.layout.dim = [MultiArrayDimension(label="boxes", size=boxes.shape[0], stride=4)]
        response.boxes.data = boxes.flatten().tolist()
        response.annotated_frame = self.cv_bridge.cv2_to_imgmsg(annotated_frame)
        response.segmasks.layout.dim = [MultiArrayDimension(label="masks", size=segmaps.shape[0], stride=segmaps.shape[1]*segmaps.shape[2])]
        response.segmasks.data = segmaps.flatten().tolist()

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