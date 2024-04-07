# Grounded SAM ROS Service Server

This repository contains an implementation of Grounded SAM in ROS as a service server.

## About
Grounded Segment-Anything-Model (SAM) is a state-of-the-art model for detecting and segmenting objects in images. This implementation provides a ROS service server for utilizing the Grounded SAM model within ROS-based applications.

## Original Repository
Check out the original repository of the model at [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything).

## Compatibility
- Tested on ROS Noetic, might work with other ROS distributions.

## Hardware Requirements
- A GPU with a minimum of 8 GB VRAM for Grounded SAM or 4 GB for the Grounding Dino model alone.

## Installation
To install and use this ROS service server, follow these steps:

1. Clone this repository:
    ```bash
    cd ~/catkin_ws/src
    git clone https://github.com/HashimHS/grounding_sam_ros.git
    ```

2. Install Grounding SAM environment:
    ```bash
    # Navigate to the cloned repository directory
    cd grounding_sam_ros
    # Install the conda environment
    conda env create -f gsam.yaml
    # Install the models
    python -m pip install git+https://github.com/IDEA-Research/GroundingDINO.git
    python -m pip install git+https://github.com/facebookresearch/segment-anything.git
    ```

3. Download the weights:
    ```bash
    mkdir weights
    cd weights
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    ```

4. Build the ROS workspace:
    ```bash
    # Navigate to the root of your ROS workspace
    cd ~/catkin_ws
    # Build the workspace
    catkin build
    ```

## Usage
To use the Grounded SAM ROS service server, follow these steps:

1. Launch the ROS node:
    ```bash
    conda activate gsam
    roslaunch grounded_sam_ros gsam.launch
    ```
    Alternatively you can launch Dino only for detection without segmentation
    ```bash
    conda activate gsam
    roslaunch grounded_sam_ros dino.launch
    ```
    You should now find a new service server with the name "vit_detection".

2. Use the service for segmenting objects in images. An example of client code:
    ```bash
    from cv_bridge import CvBridge
    import cv2

    text_prompt ='OBJECT YOU WANT TO DETECT'
    vit_detection = rospy.ServiceProxy('vit_detection', VitDetection)
    cv_bridge = CvBridge()
    rgb_msg = cv_bridge.cv2_to_imgmsg(np.array(rgb_image))
    results = vit_detection(rgb_msg, text_prompt)

    # Annotated image from Grounding Dino
    annotated_frame = results.annotated_frame
    
    # List of detected objects
    labels = results.labels

    # Bounding boxes in y1 x1 y2 x2 format
    boxes = results.boxes

    # Detection score
    scores = results.scores

    # Image Segmentation Mask
    mask = results.segmask
    ```
## Troubleshooting:
In case you get an error:
    ```bash
    ERROR: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
    ```
    
You have to set `LD_PRELOAD` environment variable before launching the node:
    ```bash
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
    ```
## Contributors
This ROS package is made possible by:
- Hashim Ismail ([HashimHS](https://github.com/HashimHS)).
- JLL ([Taokt](https://github.com/Taokt)).
