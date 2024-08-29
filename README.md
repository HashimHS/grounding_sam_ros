# Grounded SAM ROS Service Server

This repository contains an implementation of Grounded SAM in ROS as a service server.

## About
Grounded Segment-Anything-Model (SAM) is a state-of-the-art model for detecting and segmenting objects in images. This implementation provides a ROS service server for utilizing the Grounded SAM model within ROS-based applications.

## Original Repository
Check out the original repository of the model at [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything).

## Compatibility
- Tested on ROS Noetic, might work with other ROS distributions.

## Hardware Requirements
- A GPU with a minimum of 8 GB VRAM for Grounded SAM or 4 GB for the Grounding DINO model alone.

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
    conda activate gsam
    
    # Install the Grounding DINO
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO/
    pip install -e .
    cd ..
    rm -rf GroundingDINO/
    
    # Install SAM
    python -m pip install git+https://github.com/facebookresearch/segment-anything.git
    ```

3. Build the ROS workspace:
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
    roslaunch grounded_sam_ros gsam.launch sam_model:="SAM-L" venv:="path/to/python/env"
    ```
    Alternatively you can launch Grounding DINO only for detection without segmentation
    ```bash
    roslaunch grounded_sam_ros dino.launch venv:="path/to/python/env"
    ```
    You should now find a new service server with the name "vit_detection".

2. Use the service for segmenting objects in images. An example of client code:
    ```bash
    from grounding_sam_ros.client import SamDetector

    text_prompt ='DESCRIBE THE OBJECT YOU WANT TO DETECT'
    detector = SamDetector()
    annotated_frame, boxes, masks, labels, scores = detector.detect(rgb_image, text_prompt)
    ```

The model will automatically downloaded the needed model weights.

## Troubleshooting:
- Make sure to provide the path to your conda / virtual python environment in the launch files by changing the argument "venv"

- In case you get an error:
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
